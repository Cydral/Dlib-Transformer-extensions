// GGUF file reader (header-only).
//
// Parses the GGUF container used by llama.cpp / HuggingFace GGUF downloads:
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

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <stdexcept>

namespace dlib
{
    // ggml tensor data types. Values match the ggml enum so they can be read directly.
    enum class ggml_type : uint32_t
    {
        F32 = 0, F16 = 1,
        Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
        Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15
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
                return static_cast<int64_t>(d);
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
        bool is_quantized() const { return type != ggml_type::F32 && type != ggml_type::F16; }
    };

    class gguf_reader
    {
    public:
        explicit gguf_reader(const std::string& path)
            : in_(path, std::ios::binary)
        {
            if (!in_) throw std::runtime_error("gguf_reader: cannot open " + path);

            const uint32_t magic = read_raw<uint32_t>();
            if (magic != 0x46554747u /* "GGUF" */)
                throw std::runtime_error("gguf_reader: not a GGUF file (bad magic)");

            version_ = read_raw<uint32_t>();
            if (version_ < 2 || version_ > 3)
                throw std::runtime_error("gguf_reader: unsupported GGUF version "
                    + std::to_string(version_));

            const uint64_t tensor_count = read_raw<uint64_t>();
            const uint64_t kv_count = read_raw<uint64_t>();

            for (uint64_t k = 0; k < kv_count; ++k)
            {
                std::string key = read_string();
                gguf_type vt = static_cast<gguf_type>(read_raw<uint32_t>());
                meta_[key] = read_value(vt);
            }

            if (has("general.alignment"))
                alignment_ = static_cast<uint64_t>(at("general.alignment").as_int());
            if (alignment_ == 0) alignment_ = 32;

            tensors_.reserve(static_cast<size_t>(tensor_count));
            for (uint64_t t = 0; t < tensor_count; ++t)
            {
                gguf_tensor_info info;
                info.name = read_string();
                const uint32_t ndim = read_raw<uint32_t>();
                info.dims.resize(ndim);
                for (uint32_t d = 0; d < ndim; ++d) info.dims[d] = read_raw<uint64_t>();
                info.type = static_cast<ggml_type>(read_raw<uint32_t>());
                info.offset = read_raw<uint64_t>();

                const ggml_type_traits tr = get_ggml_type_traits(info.type);
                const uint64_t n = info.n_elements();
                if (tr.block_size > 1 && (n % tr.block_size) != 0)
                    throw std::runtime_error("gguf_reader: element count of '" + info.name
                        + "' is not block-aligned");
                info.nbytes = (n / tr.block_size) * tr.type_size;
                tensors_.push_back(std::move(info));
            }

            const std::streampos pos = in_.tellg();
            data_offset_ = align_up(static_cast<uint64_t>(pos), alignment_);
        }

        uint32_t version() const { return version_; }
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
            for (const auto& t : tensors_) if (t.name == name) return &t;
            return nullptr;
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

        template <typename T> T read_raw()
        {
            T x{};
            in_.read(reinterpret_cast<char*>(&x), sizeof(T));
            if (!in_) throw std::runtime_error("gguf_reader: unexpected end of file");
            return x;
        }

        std::string read_string()
        {
            const uint64_t len = read_raw<uint64_t>();
            std::string str;
            str.resize(static_cast<size_t>(len));
            if (len) in_.read(&str[0], static_cast<std::streamsize>(len));
            if (!in_) throw std::runtime_error("gguf_reader: unexpected end of file in string");
            return str;
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

            if (et == gguf_type::STRING)
            {
                v.arr_str.reserve(static_cast<size_t>(n));
                for (uint64_t k = 0; k < n; ++k) v.arr_str.push_back(read_string());
            }
            else if (et == gguf_type::FLOAT32 || et == gguf_type::FLOAT64)
            {
                v.arr_float.reserve(static_cast<size_t>(n));
                gguf_value tmp;
                for (uint64_t k = 0; k < n; ++k) { read_scalar_into(tmp, et); v.arr_float.push_back(tmp.d); }
            }
            else if (et == gguf_type::ARRAY)
            {
                throw std::runtime_error("gguf_reader: nested arrays are not supported");
            }
            else
            {
                v.arr_int.reserve(static_cast<size_t>(n));
                gguf_value tmp;
                for (uint64_t k = 0; k < n; ++k) { read_scalar_into(tmp, et); v.arr_int.push_back(tmp.i); }
            }
            return v;
        }

        std::ifstream in_;
        uint32_t version_ = 0;
        uint64_t alignment_ = 32;
        uint64_t data_offset_ = 0;
        std::map<std::string, gguf_value> meta_;
        std::vector<gguf_tensor_info> tensors_;
    };
}

#endif // DLIB_GGUF_READER_H_
