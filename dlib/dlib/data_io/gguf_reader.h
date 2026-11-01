// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// GGUF file reader (header-only).
//
// Parses the GGUF container used by the open-weight model ecosystem:
//   header (magic, version, tensor_count, metadata_kv_count)
//   metadata key/value pairs (typed scalars, strings and arrays)
//   tensor descriptors (name, ggml type, dimensions, data offset)
//   aligned tensor data blob (accessed lazily, tensor by tensor)
//
// The reader exposes metadata generically and tensor descriptors with a computed
// byte size, plus raw (possibly quantized) tensor bytes. Dequantization and weight
// repacking are handled by the converter, not here.
//
// Assumes a little-endian host (x86/MSVC), which matches GGUF's on-disk byte order.

#ifndef DLIB_GGUF_READER_H_
#define DLIB_GGUF_READER_H_

#include "gguf_reader_abstract.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <istream>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <utility>

namespace dlib
{
    // ggml tensor data types. Values match the ggml enum so they can be read directly.
    enum class ggml_type : uint32_t
    {
        F32 = 0, F16 = 1,
        Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
        Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
        IQ4_NL = 20, BF16 = 30
    };

    // GGUF metadata value types (values match the GGUF specification).
    enum class gguf_type : uint32_t
    {
        UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
        UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
        STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12
    };

    // Block geometry of a ggml type: number of elements per block and bytes per block.
    // nbytes(tensor) = n_elements / block_size * type_size (first dim is block-aligned).
    struct ggml_type_traits { uint32_t block_size; uint32_t type_size; };

    inline ggml_type_traits get_ggml_type_traits(ggml_type t)
    {
        switch (t)
        {
        case ggml_type::F32:  return { 1, 4 };
        case ggml_type::F16:  return { 1, 2 };
        case ggml_type::Q4_0: return { 32, 18 };
        case ggml_type::Q4_1: return { 32, 20 };
        case ggml_type::Q5_0: return { 32, 22 };
        case ggml_type::Q5_1: return { 32, 24 };
        case ggml_type::Q8_0: return { 32, 34 };
        case ggml_type::Q8_1: return { 32, 36 };
        case ggml_type::Q2_K: return { 256, 84 };
        case ggml_type::Q3_K: return { 256, 110 };
        case ggml_type::Q4_K: return { 256, 144 };
        case ggml_type::Q5_K: return { 256, 176 };
        case ggml_type::Q6_K: return { 256, 210 };
        case ggml_type::Q8_K: return { 256, 292 };
        case ggml_type::IQ4_NL: return { 32, 18 };
        case ggml_type::BF16: return { 1, 2 };
        default:
            throw std::runtime_error("gguf_reader: unsupported ggml type "
                + std::to_string(static_cast<uint32_t>(t)));
        }
    }

    // A parsed metadata value. Only the field matching `type` (or `array_type` for
    // arrays) is populated. Integers are widened to int64, floats to double.
    struct gguf_value
    {
        gguf_type type = gguf_type::UINT32;
        gguf_type array_type = gguf_type::UINT32;

        int64_t     i = 0;     // any integer/bool scalar
        double      d = 0.0;   // float32/float64 scalar
        std::string s;         // string scalar

        std::vector<std::string> arr_str;
        std::vector<int64_t>     arr_int;
        std::vector<double>      arr_float;

        bool is_array() const { return type == gguf_type::ARRAY; }

        int64_t as_int() const
        {
            if (type == gguf_type::FLOAT32 || type == gguf_type::FLOAT64)
            {
                // Saturated, since the conversion of a value beyond the range of int64 is undefined.
                if (!(d == d)) return 0;
                if (d >= 9.2e18) return INT64_MAX;
                if (d <= -9.2e18) return INT64_MIN;
                return static_cast<int64_t>(d);
            }
            return i;
        }
        double as_double() const
        {
            if (type == gguf_type::FLOAT32 || type == gguf_type::FLOAT64) return d;
            return static_cast<double>(i);
        }
        const std::string& as_string() const { return s; }
    };

    struct gguf_tensor_info
    {
        std::string name;
        ggml_type type = ggml_type::F32;
        std::vector<uint64_t> dims;   // ggml order: dims[0] is the contiguous dimension
        uint64_t offset = 0;          // relative to the start of the tensor-data section
        uint64_t nbytes = 0;          // size in bytes on disk

        uint64_t n_elements() const
        {
            uint64_t n = 1;
            for (uint64_t e : dims) n *= e;
            return n;
        }
        bool is_quantized() const
        {
            return type != ggml_type::F32 && type != ggml_type::F16 && type != ggml_type::BF16;
        }
    };

    class gguf_reader
    {
    public:
        explicit gguf_reader(const std::string& path)
            : in_(nullptr)
        {
            std::unique_ptr<std::filebuf> file(new std::filebuf);
            if (!file->open(path, std::ios::in | std::ios::binary))
                throw std::runtime_error("gguf_reader: cannot open " + path);
            buf_ = std::move(file);
            in_.rdbuf(buf_.get());
            parse();
        }

        /* Reads a GGUF container held in memory, for instance a file a program mapped
           into its address space.  The memory must outlive this object, and
           read_tensor_raw() copies tensors out of it. */
        gguf_reader(const void* data, uint64_t size)
            : in_(nullptr)
        {
            if (data == nullptr && size != 0)
                throw std::runtime_error("gguf_reader: null memory block");
            buf_.reset(new memory_buffer(static_cast<const char*>(data), size));
            in_.rdbuf(buf_.get());
            parse();
        }

        gguf_reader(const gguf_reader&) = delete;
        gguf_reader& operator=(const gguf_reader&) = delete;

    private:

        void parse()
        {
            in_.seekg(0, std::ios::end);
            const std::streamoff end = in_.tellg();
            in_.seekg(0, std::ios::beg);
            if (!in_ || end < 0) throw std::runtime_error("gguf_reader: cannot determine the size of the container");
            size_ = static_cast<uint64_t>(end);
            pos_ = 0;

            const uint32_t magic = read_raw<uint32_t>();
            if (magic != 0x46554747u /* "GGUF" */)
                throw std::runtime_error("gguf_reader: not a GGUF file (bad magic)");

            version_ = read_raw<uint32_t>();
            if (version_ < 2 || version_ > 3)
                throw std::runtime_error("gguf_reader: unsupported GGUF version "
                    + std::to_string(version_));

            const uint64_t tensor_count = read_raw<uint64_t>();
            const uint64_t kv_count = read_raw<uint64_t>();
            /* A metadata entry takes 13 bytes at least (the length of its key, its type and a
               value of one byte) and a tensor descriptor 32 (the length of its name, one
               dimension, its type and its offset); larger counts come from a damaged file, as
               do counts beyond 2^20, far above those of the published models, which bound the
               memory of the parsed header whatever the size of the file. */
            if (tensor_count > remaining() / 32 || kv_count > remaining() / 13
                || tensor_count > max_entries || kv_count > max_entries)
                throw std::runtime_error("gguf_reader: corrupt header (entry counts exceed the file size)");

            // A key or a tensor name given twice comes from a damaged or forged file: its readers
            // would take different entries for the same name.
            for (uint64_t k = 0; k < kv_count; ++k)
            {
                std::string key = read_string();
                gguf_type vt = static_cast<gguf_type>(read_raw<uint32_t>());
                gguf_value v = read_value(vt);
                if (!meta_.emplace(key, std::move(v)).second)
                    throw std::runtime_error("gguf_reader: duplicate metadata key '" + key + "'");
            }

            if (has("general.alignment"))
            {
                const int64_t a = at("general.alignment").as_int();
                if (a <= 0 || a > 65536 || (a & (a - 1)) != 0)
                    throw std::runtime_error("gguf_reader: invalid alignment " + std::to_string(a));
                alignment_ = static_cast<uint64_t>(a);
            }

            tensors_.reserve(static_cast<size_t>(std::min<uint64_t>(tensor_count, 1 << 16)));
            for (uint64_t t = 0; t < tensor_count; ++t)
            {
                gguf_tensor_info info;
                info.name = read_string();
                const uint32_t ndim = read_raw<uint32_t>();
                if (ndim == 0 || ndim > 8)
                    throw std::runtime_error("gguf_reader: invalid number of dimensions for '" + info.name + "'");
                info.dims.resize(ndim);
                uint64_t elements = 1;
                for (uint32_t d = 0; d < ndim; ++d)
                {
                    info.dims[d] = read_raw<uint64_t>();
                    if (info.dims[d] != 0 && elements > (uint64_t(1) << 60) / info.dims[d])
                        throw std::runtime_error("gguf_reader: dimensions of '" + info.name + "' overflow");
                    elements *= info.dims[d];
                }
                info.type = static_cast<ggml_type>(read_raw<uint32_t>());
                info.offset = read_raw<uint64_t>();

                // The blocks of a quantized type never straddle two rows, and the data of each
                // tensor starts on the alignment of the file.
                const ggml_type_traits tr = get_ggml_type_traits(info.type);
                const uint64_t n = info.n_elements();
                if (tr.block_size > 1 && ((n % tr.block_size) != 0 || (info.dims[0] % tr.block_size) != 0))
                    throw std::runtime_error("gguf_reader: element count of '" + info.name
                        + "' is not block-aligned");
                if (info.offset % alignment_ != 0)
                    throw std::runtime_error("gguf_reader: tensor '" + info.name
                        + "' does not start on the alignment of the file");
                info.nbytes = (n / tr.block_size) * tr.type_size;
                if (!tensor_index_.emplace(info.name, tensors_.size()).second)
                    throw std::runtime_error("gguf_reader: duplicate tensor name '" + info.name + "'");
                tensors_.push_back(std::move(info));
            }

            data_offset_ = align_up(pos_, alignment_);
            for (const auto& t : tensors_)
                if (t.offset > size_ || t.nbytes > size_ || data_offset_ + t.offset + t.nbytes > size_)
                    throw std::runtime_error("gguf_reader: tensor '" + t.name + "' lies beyond the end of the file");
        }

    public:

        uint32_t version() const { return version_; }
        uint64_t size() const { return size_; }
        uint64_t alignment() const { return alignment_; }
        uint64_t data_offset() const { return data_offset_; }
        const std::map<std::string, gguf_value>& metadata() const { return meta_; }
        const std::vector<gguf_tensor_info>& tensors() const { return tensors_; }

        bool has(const std::string& key) const { return meta_.find(key) != meta_.end(); }

        const gguf_value& at(const std::string& key) const
        {
            auto it = meta_.find(key);
            if (it == meta_.end())
                throw std::runtime_error("gguf_reader: missing metadata key '" + key + "'");
            return it->second;
        }

        std::string get_str(const std::string& key, const std::string& def = "") const
        {
            return has(key) ? at(key).as_string() : def;
        }
        int64_t get_int(const std::string& key, int64_t def = 0) const
        {
            return has(key) ? at(key).as_int() : def;
        }
        double get_double(const std::string& key, double def = 0.0) const
        {
            return has(key) ? at(key).as_double() : def;
        }

        const gguf_tensor_info* find_tensor(const std::string& name) const
        {
            const auto it = tensor_index_.find(name);
            return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
        }

        // Read the raw on-disk bytes of a tensor (possibly quantized) into dst.
        void read_tensor_raw(const gguf_tensor_info& t, std::vector<uint8_t>& dst)
        {
            dst.resize(static_cast<size_t>(t.nbytes));
            in_.clear();
            in_.seekg(static_cast<std::streamoff>(data_offset_ + t.offset), std::ios::beg);
            if (t.nbytes) in_.read(reinterpret_cast<char*>(dst.data()),
                static_cast<std::streamsize>(t.nbytes));
            if (!in_) throw std::runtime_error("gguf_reader: failed reading tensor '" + t.name + "'");
        }

    private:
        static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

        static const uint64_t max_entries = uint64_t(1) << 20;
        static const uint64_t max_array_elements = uint64_t(1) << 24;

        template <typename T> T read_raw()
        {
            T x{};
            in_.read(reinterpret_cast<char*>(&x), sizeof(T));
            if (!in_) throw std::runtime_error("gguf_reader: unexpected end of file");
            pos_ += sizeof(T);
            return x;
        }

        /* Bytes left after the parsing position, which parse() tracks itself rather than
           asking the stream for it at every string of the vocabulary. */
        uint64_t remaining() const { return pos_ > size_ ? 0 : size_ - pos_; }

        std::string read_string()
        {
            const uint64_t len = read_raw<uint64_t>();
            if (len > remaining())
                throw std::runtime_error("gguf_reader: string length exceeds the file size");
            std::string str;
            str.resize(static_cast<size_t>(len));
            if (len) in_.read(&str[0], static_cast<std::streamsize>(len));
            if (!in_) throw std::runtime_error("gguf_reader: unexpected end of file in string");
            pos_ += len;
            return str;
        }

        // Bytes of a scalar of type t in the file, 0 for the strings, arrays and unknown types.
        static uint64_t scalar_size(gguf_type t)
        {
            switch (t)
            {
            case gguf_type::UINT8: case gguf_type::INT8: case gguf_type::BOOL: return 1;
            case gguf_type::UINT16: case gguf_type::INT16: return 2;
            case gguf_type::UINT32: case gguf_type::INT32: case gguf_type::FLOAT32: return 4;
            case gguf_type::UINT64: case gguf_type::INT64: case gguf_type::FLOAT64: return 8;
            default: return 0;
            }
        }

        void read_scalar_into(gguf_value& v, gguf_type t)
        {
            switch (t)
            {
            case gguf_type::UINT8:   v.i = read_raw<uint8_t>();  break;
            case gguf_type::INT8:    v.i = read_raw<int8_t>();   break;
            case gguf_type::UINT16:  v.i = read_raw<uint16_t>(); break;
            case gguf_type::INT16:   v.i = read_raw<int16_t>();  break;
            case gguf_type::UINT32:  v.i = read_raw<uint32_t>(); break;
            case gguf_type::INT32:   v.i = read_raw<int32_t>();  break;
            case gguf_type::UINT64:  v.i = static_cast<int64_t>(read_raw<uint64_t>()); break;
            case gguf_type::INT64:   v.i = read_raw<int64_t>();  break;
            case gguf_type::BOOL:    v.i = read_raw<uint8_t>() ? 1 : 0; break;
            case gguf_type::FLOAT32: v.d = read_raw<float>();    break;
            case gguf_type::FLOAT64: v.d = read_raw<double>();   break;
            case gguf_type::STRING:  v.s = read_string();        break;
            default:
                throw std::runtime_error("gguf_reader: invalid scalar metadata type");
            }
        }

        gguf_value read_value(gguf_type t)
        {
            gguf_value v;
            v.type = t;
            if (t != gguf_type::ARRAY) { read_scalar_into(v, t); return v; }

            const gguf_type et = static_cast<gguf_type>(read_raw<uint32_t>());
            const uint64_t n = read_raw<uint64_t>();
            v.array_type = et;
            // The smallest size of an element in the file; a string holds at least its length.
            const uint64_t least = et == gguf_type::STRING ? 8 : scalar_size(et);
            if (least == 0) throw std::runtime_error("gguf_reader: invalid array element type");
            /* An array of more than 2^24 elements, sixty-four times the largest vocabularies,
               comes from a damaged file, whose elements, empty strings or zeros, would take
               several times the size of the file in memory. */
            if (n > remaining() / least || n > max_array_elements)
                throw std::runtime_error("gguf_reader: array length exceeds the file size");
            /* The memory follows the elements actually read, and a declared length within the
               file reserves a bounded amount at first. */
            const size_t reserved = static_cast<size_t>(std::min<uint64_t>(n, 1 << 20));

            if (et == gguf_type::STRING)
            {
                v.arr_str.reserve(reserved);
                for (uint64_t k = 0; k < n; ++k) v.arr_str.push_back(read_string());
            }
            else if (et == gguf_type::FLOAT32 || et == gguf_type::FLOAT64)
            {
                v.arr_float.reserve(reserved);
                gguf_value tmp;
                for (uint64_t k = 0; k < n; ++k) { read_scalar_into(tmp, et); v.arr_float.push_back(tmp.d); }
            }
            else if (et == gguf_type::ARRAY)
            {
                throw std::runtime_error("gguf_reader: nested arrays are not supported");
            }
            else
            {
                v.arr_int.reserve(reserved);
                gguf_value tmp;
                for (uint64_t k = 0; k < n; ++k) { read_scalar_into(tmp, et); v.arr_int.push_back(tmp.i); }
            }
            return v;
        }

        /* A read-only stream buffer over a block of memory of any size, seekable in both
           directions.  Its get area is a window of at most 1 GiB of the block, moved by the
           reads and the seeks: the standard library of Visual Studio counts the characters
           of a get area in an int, which a get area over a block of 2 GiB or more wraps,
           so that the end of a large mapped model file could not be reached. */
        class memory_buffer : public std::streambuf
        {
        public:
            memory_buffer(const char* data, uint64_t size) : data_(data), size_(size)
            {
                place(0);
            }
        protected:
            int_type underflow() override
            {
                const uint64_t at = position();
                if (at >= size_) return traits_type::eof();
                place(at);
                return traits_type::to_int_type(*gptr());
            }
            std::streamsize xsgetn(char* s, std::streamsize n) override
            {
                const uint64_t at = position();
                const uint64_t left = at < size_ ? size_ - at : 0;
                const uint64_t count = n > 0 ? std::min<uint64_t>(left, static_cast<uint64_t>(n)) : 0;
                if (count != 0) std::memcpy(s, data_ + at, static_cast<size_t>(count));
                place(at + count);
                return static_cast<std::streamsize>(count);
            }
            std::streamsize showmanyc() override
            {
                const uint64_t at = position();
                return at < size_ ? static_cast<std::streamsize>(size_ - at) : std::streamsize(-1);
            }
            pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override
            {
                if (!(which & std::ios_base::in)) return pos_type(off_type(-1));
                const uint64_t base = dir == std::ios_base::beg ? 0 : dir == std::ios_base::cur ? position() : size_;
                uint64_t target = 0;
                if (off < 0)
                {
                    // -(off + 1) + 1 is the magnitude of off, without the overflow of -off.
                    const uint64_t back = static_cast<uint64_t>(-(off + 1)) + 1;
                    if (back > base) return pos_type(off_type(-1));
                    target = base - back;
                }
                else
                {
                    if (static_cast<uint64_t>(off) > size_ - base) return pos_type(off_type(-1));
                    target = base + static_cast<uint64_t>(off);
                }
                place(target);
                return pos_type(static_cast<off_type>(target));
            }
            pos_type seekpos(pos_type pos, std::ios_base::openmode which) override
            {
                return seekoff(off_type(pos), std::ios_base::beg, which);
            }
        private:
            uint64_t position() const
            {
                return start_ + static_cast<uint64_t>(gptr() - eback());
            }
            // Makes the get area the window of the block that starts at offset at.
            void place(uint64_t at)
            {
                const uint64_t window = uint64_t(1) << 30;
                start_ = at;
                const uint64_t length = std::min<uint64_t>(size_ - at, window);
                char* p = const_cast<char*>(data_) + at;
                setg(p, p, p + length);
            }
            const char* data_;
            uint64_t size_;
            uint64_t start_ = 0;
        };

        std::unique_ptr<std::streambuf> buf_;
        std::istream in_;
        uint64_t size_ = 0;
        uint64_t pos_ = 0;
        uint32_t version_ = 0;
        uint64_t alignment_ = 32;
        uint64_t data_offset_ = 0;
        std::map<std::string, gguf_value> meta_;
        std::vector<gguf_tensor_info> tensors_;
        std::map<std::string, size_t> tensor_index_;
    };

    /* The metadata a model declares, read within the bounds a sane file respects: a value
       outside them comes from a damaged or forged file, and a reader that narrowed it into
       a long or used it as a count would compute nonsense or allocate without limit.  An
       array, which some architectures give per layer, reads as the default. */
    inline int64_t gguf_bounded_int(const gguf_reader& g, const std::string& key, int64_t def,
        int64_t lo, int64_t hi)
    {
        if (!g.has(key) || g.at(key).is_array()) return def;
        const gguf_value& v = g.at(key);
        if (v.type == gguf_type::STRING)
            throw std::runtime_error("gguf: metadata '" + key + "' is not a number");
        if (v.type == gguf_type::FLOAT32 || v.type == gguf_type::FLOAT64)
        {
            if (!(v.d >= static_cast<double>(lo) && v.d <= static_cast<double>(hi)) || v.d != std::floor(v.d))
                throw std::runtime_error("gguf: metadata '" + key + "' is not an integer from "
                    + std::to_string(lo) + " to " + std::to_string(hi));
            return static_cast<int64_t>(v.d);
        }
        /* The unsigned types of 64 bits arrive as int64: a value of 2^63 or more reads
           negative and fails the lower bound of every count. */
        if (v.i < lo || v.i > hi)
            throw std::runtime_error("gguf: metadata '" + key + "' = " + std::to_string(v.i) + " lies outside "
                + std::to_string(lo) + " .. " + std::to_string(hi));
        return v.i;
    }

    inline double gguf_bounded_double(const gguf_reader& g, const std::string& key, double def,
        double lo, double hi)
    {
        if (!g.has(key) || g.at(key).is_array()) return def;
        const gguf_value& v = g.at(key);
        if (v.type == gguf_type::STRING)
            throw std::runtime_error("gguf: metadata '" + key + "' is not a number");
        const double x = v.as_double();
        if (!(x >= lo && x <= hi))
            throw std::runtime_error("gguf: metadata '" + key + "' is not a finite number from "
                + std::to_string(lo) + " to " + std::to_string(hi));
        return x;
    }
}

#endif // DLIB_GGUF_READER_H_
