// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_UTILITIES_H_
#define DLIB_DNn_UTILITIES_H_

#include "../cuda/tensor.h"
#include "utilities_abstract.h"
#include "../geometry.h"
#include "../rand.h"
#include "../general_hash/murmur_hash3.h"
#include "../statistics/statistics.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline void randomize_parameters (
        tensor& params,
        unsigned long num_inputs_and_outputs,
        dlib::rand& rnd
    )
    {
        for (auto& val : params)
        {
            // Draw a random number to initialize the layer according to formula (16)
            // from Understanding the difficulty of training deep feedforward neural
            // networks by Xavier Glorot and Yoshua Bengio.
            val = 2*rnd.get_random_float()-1;
            val *= std::sqrt(6.0/(num_inputs_and_outputs));
        }
    }

// ----------------------------------------------------------------------------------------

    inline void randomize_parameters_truncated_normal (
        tensor& params,
        double deviation,
        dlib::rand& rnd
    )
    {
        // A unit normal law truncated at two deviations keeps a deviation of 0.8796; the
        // draw is scaled by its inverse, so the values have the deviation asked for.
        const double scale = 1.1368472343385565 * deviation;
        for (auto& val : params)
        {
            double v = rnd.get_random_gaussian();
            while (std::abs(v) > 2)
                v = rnd.get_random_gaussian();
            val = static_cast<float>(scale * v);
        }
    }

    // How a projection draws its weights when it is set up.
    enum class weight_init
    {
        glorot_uniform,
        truncated_lecun_normal
    };

// ----------------------------------------------------------------------------------------

    template <typename label_type>
    struct weighted_label
    {
        weighted_label()
        {}

        weighted_label(label_type label, float weight = 1.f)
            : label(label), weight(weight)
        {}

        label_type label{};
        float weight = 1.f;
    };

// ----------------------------------------------------------------------------------------

    inline double log1pexp(double x)
    {
        using std::exp;
        if (x <= -37)
            return exp(x);
        else if (-37 < x && x <= 18)
            return log1p(exp(x));
        else if (18 < x && x <= 33.3)
            return x + exp(-x);
        else
            return x;
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    T safe_log(T input, T epsilon = 1e-10)
    {
        // Prevent trying to calculate the logarithm of a very small number (let alone zero)
        return std::log(std::max(input, epsilon));
    }

// ----------------------------------------------------------------------------------------

    inline size_t tensor_index(
        const tensor& t,
        const long sample,
        const long k,
        const long r,
        const long c
    )
    {
        return ((sample * t.k() + k) * t.nr() + r) * t.nc() + c;
    }

// ----------------------------------------------------------------------------------------

    template <typename LayerType>
    class layer_helper
    {
    public:
        layer_helper() = default;

        LayerType& get() { return layer_; }
        const LayerType& get() const { return layer_; }

        // Run the wrapped layer's setup with `input_proxy` standing in for
        // sub.get_output(). Useful when the host layer wants to pre-allocate
        // caches that depend on input shape
        void setup(const tensor& input_proxy)
        {
            forward_subnet adapter{ input_proxy };
            layer_.setup(adapter);
        }

        // Forward: applies the wrapped layer to `input` and writes to `output`
        void forward(const tensor& input, resizable_tensor& output)
        {
            forward_subnet adapter{ input };
            layer_.forward(adapter, output);
        }

        // Backward: given the gradient w.r.t. the layer's output and the
        // original input that was passed to forward, computes the gradient
        // w.r.t. that input and writes it into `input_grad`
        //
        // The destination `input_grad` is initialised to zero internally so
        // that wrapped layers using add-to semantics in their backward
        // (most Dlib layers) produce the expected stand-alone gradient
        void backward(
            const tensor& gradient_input,
            const tensor& original_input,
            resizable_tensor& input_grad)
        {
            input_grad.copy_size(original_input);
            input_grad = 0;
            backward_subnet adapter{ original_input, input_grad };
            resizable_tensor unused_params_grad;
            layer_.backward(gradient_input, adapter, unused_params_grad);
        }

        friend void serialize(const layer_helper& item, std::ostream& out)
        {
            serialize("layer_helper_v1", out);
            serialize(item.layer_, out);
        }

        friend void deserialize(layer_helper& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "layer_helper_v1")
                throw serialization_error(
                    "Unexpected version while deserializing layer_helper: " + version);
            deserialize(item.layer_, in);
        }

    private:
        LayerType layer_;

        // Adapter exposing only get_output() (used by setup and forward)
        struct forward_subnet
        {
            const tensor& output_ref;
            const tensor& get_output() const { return output_ref; }
        };

        // Adapter exposing get_output() and get_gradient_input() for backward
        struct backward_subnet
        {
            const tensor& output_ref;
            tensor& gradient_input_ref;
            const tensor& get_output() const { return output_ref; }
            tensor& get_gradient_input() { return gradient_input_ref; }
        };
    };

// ----------------------------------------------------------------------------------------
// Parameter vectors and their consolidation across replicas
// ----------------------------------------------------------------------------------------

    namespace impl
    {
        /* The bytes left in a stream, or the largest count when the stream cannot tell. */
        inline uint64 bytes_left (std::istream& in)
        {
            const std::streampos here = in.tellg();
            if (here < 0)
                return std::numeric_limits<uint64>::max();
            in.seekg(0, std::ios::end);
            const std::streampos end = in.tellg();
            in.seekg(here);
            if (end < here)
                return 0;
            return static_cast<uint64>(end - here);
        }

        /* Reads the tag a serialize() of the classes below starts with, bounded likewise. */
        inline void check_version (std::istream& in, const std::string& expected)
        {
            unsigned long size = 0;
            deserialize(size, in);
            if (size > 64)
                throw serialization_error("Unexpected version found while deserializing dlib::" + expected + ".");
            std::string version(size, '\0');
            if (size != 0)
                in.read(&version[0], static_cast<std::streamsize>(size));
            if (!in || version != expected)
                throw serialization_error("Unexpected version '" + version +
                                          "' found while deserializing dlib::" + expected + ".");
        }
    }

    /* Reads what serialize() wrote for a vector, refusing a declared length larger than
       the bytes left in the stream before anything is allocated, since every element
       takes one byte at least: the objects below travel between hosts, and a length
       forged in a payload would otherwise reserve memory without limit. */
    inline void deserialize_bounded (std::vector<unsigned char>& item, std::istream& in)
    {
        unsigned long size = 0;
        deserialize(size, in);
        if (static_cast<uint64>(size) > impl::bytes_left(in))
            throw serialization_error("a vector declares more elements than the stream holds");
        item.clear();
        char buffer[4096];
        for (unsigned long done = 0; done < size; )
        {
            const std::streamsize n = static_cast<std::streamsize>(std::min<unsigned long>(size - done, sizeof(buffer)));
            in.read(buffer, n);
            if (!in)
                throw serialization_error("a vector ends before its last element");
            item.insert(item.end(), buffer, buffer + n);
            done += static_cast<unsigned long>(n);
        }
    }

    template <typename T>
    void deserialize_bounded (std::vector<T>& item, std::istream& in)
    {
        unsigned long size = 0;
        deserialize(size, in);
        if (static_cast<uint64>(size) > impl::bytes_left(in))
            throw serialization_error("a vector declares more elements than the stream holds");
        /* The memory follows the elements actually read, so a stream that cannot tell its
           length reserves no more than it delivers. */
        item.clear();
        item.reserve(static_cast<size_t>(std::min<unsigned long>(size, 1u << 16)));
        for (unsigned long i = 0; i < size; ++i)
        {
            T x;
            deserialize(x, in);
            item.push_back(std::move(x));
        }
    }

    template <typename T>
    void deserialize_bounded (std::vector<std::vector<T>>& item, std::istream& in)
    {
        unsigned long size = 0;
        deserialize(size, in);
        if (static_cast<uint64>(size) > impl::bytes_left(in))
            throw serialization_error("a vector declares more elements than the stream holds");
        item.clear();
        item.reserve(static_cast<size_t>(std::min<unsigned long>(size, 1u << 12)));
        for (unsigned long i = 0; i < size; ++i)
        {
            std::vector<T> x;
            deserialize_bounded(x, in);
            item.push_back(std::move(x));
        }
    }


    struct parameter_block
    {
        size_t offset = 0;
        size_t size = 0;
        long rows = 0;
        long cols = 0;

        bool is_matrix () const { return rows > 1 && cols > 1; }
    };

    inline void serialize (const parameter_block& item, std::ostream& out)
    {
        serialize(static_cast<uint64>(item.offset), out);
        serialize(static_cast<uint64>(item.size), out);
        serialize(item.rows, out);
        serialize(item.cols, out);
    }

    inline void deserialize (parameter_block& item, std::istream& in)
    {
        uint64 offset = 0, size = 0;
        deserialize(offset, in);
        deserialize(size, in);
        deserialize(item.rows, in);
        deserialize(item.cols, in);
        item.offset = static_cast<size_t>(offset);
        item.size = static_cast<size_t>(size);
    }

// ----------------------------------------------------------------------------------------

    class parameter_layout
    {
    public:
        void clear ()
        {
            blocks_.clear();
            total_ = 0;
        }

        void add_block (long rows, long cols)
        {
            DLIB_CASSERT(rows > 0 && cols > 0, "parameter_layout: a block needs a positive shape");
            parameter_block b;
            b.offset = total_;
            b.rows = rows;
            b.cols = cols;
            b.size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
            blocks_.push_back(b);
            total_ += b.size;
        }

        void split_block (
            size_t index,
            const std::vector<std::pair<long,long>>& shapes
        )
        {
            DLIB_CASSERT(index < blocks_.size(), "parameter_layout: no block " << index);
            const parameter_block whole = blocks_[index];
            std::vector<parameter_block> parts;
            size_t used = 0;
            for (const auto& s : shapes)
            {
                DLIB_CASSERT(s.first > 0 && s.second > 0, "parameter_layout: a block needs a positive shape");
                parameter_block b;
                b.rows = s.first;
                b.cols = s.second;
                b.size = static_cast<size_t>(s.first) * static_cast<size_t>(s.second);
                b.offset = whole.offset + used;
                used += b.size;
                DLIB_CASSERT(used <= whole.size, "parameter_layout: the parts exceed the block they split");
                parts.push_back(b);
            }
            if (used < whole.size)
            {
                parameter_block rest;
                rest.offset = whole.offset + used;
                rest.size = whole.size - used;
                rest.rows = 1;
                rest.cols = static_cast<long>(rest.size);
                parts.push_back(rest);
            }
            blocks_.erase(blocks_.begin() + static_cast<std::ptrdiff_t>(index));
            blocks_.insert(blocks_.begin() + static_cast<std::ptrdiff_t>(index), parts.begin(), parts.end());
        }

        size_t num_blocks () const { return blocks_.size(); }
        const parameter_block& block (size_t i) const { return blocks_.at(i); }
        const std::vector<parameter_block>& blocks () const { return blocks_; }
        size_t size () const { return total_; }

        uint64 fingerprint () const
        {
            std::vector<int64> shape;
            shape.reserve(2 * blocks_.size() + 1);
            shape.push_back(static_cast<int64>(total_));
            for (const auto& b : blocks_)
            {
                shape.push_back(b.rows);
                shape.push_back(b.cols);
            }
            return murmur_hash3_128bit(shape.data(), static_cast<int>(shape.size() * sizeof(int64)), 0x5eed).first;
        }

        bool operator== (const parameter_layout& other) const
        {
            if (total_ != other.total_ || blocks_.size() != other.blocks_.size())
                return false;
            for (size_t i = 0; i < blocks_.size(); ++i)
            {
                const parameter_block& a = blocks_[i];
                const parameter_block& b = other.blocks_[i];
                if (a.offset != b.offset || a.size != b.size || a.rows != b.rows || a.cols != b.cols)
                    return false;
            }
            return true;
        }
        bool operator!= (const parameter_layout& other) const { return !(*this == other); }

        friend void serialize (const parameter_layout& item, std::ostream& out)
        {
            serialize("parameter_layout", out);
            serialize(item.blocks_, out);
        }

        friend void deserialize (parameter_layout& item, std::istream& in)
        {
            impl::check_version(in, "parameter_layout");
            std::vector<parameter_block> blocks;
            deserialize_bounded(blocks, in);
            /* The blocks must tile the vector, one after the other, with sizes matching their
               shapes: anything else comes from a damaged or forged stream. */
            size_t total = 0;
            for (const auto& b : blocks)
            {
                if (b.rows <= 0 || b.cols <= 0 || b.offset != total ||
                    static_cast<uint64>(b.rows) > (uint64(1) << 40) / static_cast<uint64>(b.cols) ||
                    b.size != static_cast<size_t>(b.rows) * static_cast<size_t>(b.cols) ||
                    static_cast<uint64>(b.size) > (uint64(1) << 40) - static_cast<uint64>(total))
                    throw serialization_error("parameter_layout: inconsistent block in the stream");
                total += b.size;
            }
            item.blocks_ = std::move(blocks);
            item.total_ = total;
        }

    private:
        std::vector<parameter_block> blocks_;
        size_t total_ = 0;
    };

    void serialize (const parameter_layout& item, std::ostream& out);
    void deserialize (parameter_layout& item, std::istream& in);

// ----------------------------------------------------------------------------------------

    inline uint64 parameter_digest (
        const std::vector<float>& values
    )
    {
        /* The bytes are hashed in slices that an int length can address, each slice seeded
           with the hash of the previous one. */
        /* The values are hashed in a canonical form, a negative zero as zero and every NaN
           as one NaN, since the serialization of a float keeps neither distinction: a
           replica that received a state reads the digest of the replica that sent it. */
        uint64 h = 0x9e3779b97f4a7c15ULL ^ static_cast<uint64>(values.size());
        std::vector<uint32> buffer;
        for (size_t at = 0; at < values.size(); at += 65536)
        {
            const size_t n = std::min<size_t>(values.size() - at, 65536);
            buffer.resize(n);
            for (size_t i = 0; i < n; ++i)
            {
                const float v = values[at + i];
                uint32 bits = 0x7fc00000u;
                if (v == v)
                {
                    const float c = v == 0 ? 0.0f : v;
                    std::memcpy(&bits, &c, sizeof(bits));
                }
                buffer[i] = bits;
            }
            h = murmur_hash3_128bit(buffer.data(), static_cast<int>(n * sizeof(uint32)), h).first;
        }
        return h;
    }

// ----------------------------------------------------------------------------------------

    inline double scale_adam_beta2 (
        double beta2,
        double reference_replicas,
        double replicas
    )
    {
        DLIB_CASSERT(0 <= beta2 && beta2 < 1 && reference_replicas > 0 && replicas > 0);
        return std::pow(beta2, reference_replicas / replicas);
    }

// ----------------------------------------------------------------------------------------

    inline double privatize_delta (
        std::vector<float>& delta,
        double clip_norm,
        double noise_multiplier,
        dlib::rand& rnd
    )
    {
        DLIB_CASSERT(clip_norm > 0 && noise_multiplier >= 0);
        double sq = 0;
        for (float v : delta)
            sq += static_cast<double>(v) * v;
        const double norm = std::sqrt(sq);
        const double factor = norm > clip_norm ? clip_norm / norm : 1.0;
        const double sigma = noise_multiplier * clip_norm;
        for (float& v : delta)
            v = static_cast<float>(v * factor + sigma * rnd.get_random_gaussian());
        return norm;
    }

// ----------------------------------------------------------------------------------------

    struct delta_codec_options
    {
        double keep_fraction = 1.0;
        int bits = 8;
    };

    inline void serialize (const delta_codec_options& item, std::ostream& out)
    {
        serialize(item.keep_fraction, out);
        serialize(item.bits, out);
    }

    inline void deserialize (delta_codec_options& item, std::istream& in)
    {
        deserialize(item.keep_fraction, in);
        deserialize(item.bits, in);
        if (!(item.keep_fraction > 0 && item.keep_fraction <= 1) ||
            (item.bits != 4 && item.bits != 8 && item.bits != 32))
            throw serialization_error("delta_codec_options: invalid values in the stream");
    }

    class compressed_delta
    {
    public:
        size_t dimension () const { return static_cast<size_t>(dimension_); }
        bool empty () const { return dimension_ == 0; }
        int bits () const { return bits_; }

        size_t kept () const
        {
            uint64 k = 0;
            for (uint64 v : kept_) k += v;
            return static_cast<size_t>(k);
        }

        size_t byte_size () const
        {
            return 16 + 16 * sizes_.size() + indices_.size() + 4 * scales_.size() + codes_.size();
        }

        friend void serialize (const compressed_delta& item, std::ostream& out)
        {
            serialize("compressed_delta", out);
            serialize(item.dimension_, out);
            serialize(item.bits_, out);
            serialize(item.sizes_, out);
            serialize(item.kept_, out);
            serialize(item.indices_, out);
            serialize(item.scales_, out);
            serialize(item.codes_, out);
        }

        friend void deserialize (compressed_delta& item, std::istream& in)
        {
            impl::check_version(in, "compressed_delta");
            deserialize(item.dimension_, in);
            deserialize(item.bits_, in);
            deserialize_bounded(item.sizes_, in);
            deserialize_bounded(item.kept_, in);
            deserialize_bounded(item.indices_, in);
            deserialize_bounded(item.scales_, in);
            deserialize_bounded(item.codes_, in);
            item.validate();
        }

    private:
        friend class delta_encoder;
        friend void decode_delta (const compressed_delta& in, std::vector<float>& out);

        static size_t chunk () { return 32; }

        /* Checks every count against every other before a single value is decoded: a
           payload comes from another host and is read as untrusted input. */
        void validate () const
        {
            if (bits_ != 4 && bits_ != 8 && bits_ != 32)
                throw serialization_error("compressed_delta: unsupported number of bits");
            if (sizes_.size() != kept_.size())
                throw serialization_error("compressed_delta: block tables of different lengths");
            if (dimension_ > (uint64(1) << 40))
                throw serialization_error("compressed_delta: dimension out of range");
            uint64 total = 0, kept_total = 0, sparse_kept = 0;
            for (size_t b = 0; b < sizes_.size(); ++b)
            {
                // Each block fits in what the previous ones leave of the dimension.
                if (kept_[b] > sizes_[b] || sizes_[b] > dimension_ - total)
                    throw serialization_error("compressed_delta: the blocks exceed the dimension");
                total += sizes_[b];
                kept_total += kept_[b];
                if (kept_[b] < sizes_[b])
                    sparse_kept += kept_[b];
            }
            if (total != dimension_)
                throw serialization_error("compressed_delta: the blocks do not cover the dimension");
            if (indices_.size() < sparse_kept || indices_.size() > 10 * sparse_kept)
                throw serialization_error("compressed_delta: index stream of a wrong length");
            if (bits_ == 32)
            {
                if (!scales_.empty() || codes_.size() != 4 * kept_total)
                    throw serialization_error("compressed_delta: value stream of a wrong length");
                for (size_t i = 0; i < codes_.size(); i += 4)
                {
                    float v;
                    std::memcpy(&v, &codes_[i], sizeof(float));
                    if (!(std::abs(v) <= std::numeric_limits<float>::max()))
                        throw serialization_error("compressed_delta: invalid value");
                }
                return;
            }
            uint64 chunks = 0, code_bytes = 0;
            for (size_t b = 0; b < sizes_.size(); ++b)
            {
                chunks += (kept_[b] + chunk() - 1) / chunk();
                code_bytes += bits_ == 8 ? kept_[b] : (kept_[b] + 1) / 2;
            }
            if (scales_.size() != chunks || codes_.size() != code_bytes)
                throw serialization_error("compressed_delta: value stream of a wrong length");
            /* A scale that a code of the largest magnitude, -128 or -8, would carry beyond the
               range of a float comes from a forged payload. Every value a validated payload
               decodes to is thus finite. */
            const float largest = std::numeric_limits<float>::max() / (bits_ == 8 ? 128.0f : 8.0f);
            for (float s : scales_)
                if (!(s >= 0 && s <= largest))
                    throw serialization_error("compressed_delta: invalid scale");
        }

        uint64 dimension_ = 0;
        int32 bits_ = 8;
        std::vector<uint64> sizes_;
        std::vector<uint64> kept_;
        std::vector<unsigned char> indices_;
        std::vector<float> scales_;
        std::vector<unsigned char> codes_;
    };

    void serialize (const compressed_delta& item, std::ostream& out);
    void deserialize (compressed_delta& item, std::istream& in);

    inline void decode_delta (
        const compressed_delta& in,
        std::vector<float>& out
    )
    {
        out.assign(in.dimension(), 0.0f);
        size_t ipos = 0, spos = 0, cpos = 0, base = 0;
        bool low_nibble = true;
        for (size_t b = 0; b < in.sizes_.size(); ++b)
        {
            const size_t n = static_cast<size_t>(in.sizes_[b]);
            const size_t k = static_cast<size_t>(in.kept_[b]);
            const bool dense = k == n;
            size_t index = 0;
            for (size_t j = 0; j < k; ++j)
            {
                if (dense)
                {
                    index = j;
                }
                else
                {
                    /* LEB128 gap to the previous kept index; the first gap counts from 0. */
                    uint64 gap = 0;
                    int shift = 0;
                    while (true)
                    {
                        if (ipos >= in.indices_.size() || shift > 56)
                            throw serialization_error("compressed_delta: truncated index stream");
                        const unsigned char c = in.indices_[ipos++];
                        gap |= static_cast<uint64>(c & 0x7f) << shift;
                        shift += 7;
                        if ((c & 0x80) == 0) break;
                    }
                    if (gap >= n)
                        throw serialization_error("compressed_delta: index beyond its block");
                    const uint64 next = (j == 0 ? 0 : static_cast<uint64>(index) + 1) + gap;
                    if (next >= n)
                        throw serialization_error("compressed_delta: index beyond its block");
                    index = static_cast<size_t>(next);
                }
                float value = 0;
                if (in.bits_ == 32)
                {
                    std::memcpy(&value, &in.codes_[cpos], sizeof(float));
                    cpos += 4;
                }
                else
                {
                    const float scale = in.scales_[spos + j / compressed_delta::chunk()];
                    int code = 0;
                    if (in.bits_ == 8)
                    {
                        code = static_cast<int>(static_cast<signed char>(in.codes_[cpos++]));
                    }
                    else
                    {
                        const unsigned char c = in.codes_[cpos];
                        code = (low_nibble ? (c & 0x0f) : (c >> 4)) - 8;
                        if (!low_nibble) ++cpos;
                        low_nibble = !low_nibble;
                    }
                    value = scale * code;
                }
                if (base + index >= out.size())
                    throw serialization_error("compressed_delta: index beyond the dimension");
                out[base + index] = value;
            }
            if (in.bits_ != 32)
                spos += (k + compressed_delta::chunk() - 1) / compressed_delta::chunk();
            if (in.bits_ == 4 && !low_nibble)
            {
                // A block with an odd count leaves the high nibble of its last byte unused.
                ++cpos;
                low_nibble = true;
            }
            base += n;
        }
    }

    class delta_encoder
    {
    public:
        delta_encoder () = default;

        explicit delta_encoder (
            const delta_codec_options& options
        )
        {
            set_options(options);
        }

        const delta_codec_options& options () const { return options_; }

        void set_options (const delta_codec_options& options)
        {
            DLIB_CASSERT(options.keep_fraction > 0 && options.keep_fraction <= 1,
                "delta_encoder: keep_fraction must lie in (0, 1]");
            DLIB_CASSERT(options.bits == 4 || options.bits == 8 || options.bits == 32,
                "delta_encoder: bits must be 4, 8 or 32");
            options_ = options;
        }

        void reset () { residual_.clear(); }
        const std::vector<float>& residual () const { return residual_; }

        void encode (
            const parameter_layout& layout,
            const std::vector<float>& delta,
            compressed_delta& out
        )
        {
            DLIB_CASSERT(delta.size() == layout.size(), "delta_encoder: the delta does not match the layout");
            /* What is sent is the delta plus the residual. Quantized, it must leave room for
               the scale of its group within the range validate() accepts. */
            const float limit = options_.bits == 32 ? std::numeric_limits<float>::max()
                                                    : std::numeric_limits<float>::max() / 2;
            const bool carried = residual_.size() == delta.size();
            for (size_t i = 0; i < delta.size(); ++i)
            {
                const float v = carried ? residual_[i] + delta[i] : delta[i];
                if (!(std::abs(v) <= limit))
                    throw error("delta_encoder: the delta holds a value that is not finite or too large to encode");
            }
            if (!carried)
                residual_.assign(delta.size(), 0.0f);

            out = compressed_delta();
            out.dimension_ = delta.size();
            out.bits_ = options_.bits;
            const float qmax = options_.bits == 8 ? 127.0f : 7.0f;
            std::vector<size_t> order;
            std::vector<float> kept;
            bool low_nibble = true;
            for (const parameter_block& b : layout.blocks())
            {
                /* The delta and what the previous encodings left out are sent together; the
                   part this encoding leaves out is carried to the next one. */
                float* v = residual_.data() + b.offset;
                const float* d = delta.data() + b.offset;
                for (size_t i = 0; i < b.size; ++i)
                    v[i] += d[i];

                size_t k = b.size;
                if (options_.keep_fraction < 1)
                    k = std::max<size_t>(1, static_cast<size_t>(std::ceil(options_.keep_fraction * b.size)));
                order.resize(b.size);
                for (size_t i = 0; i < b.size; ++i) order[i] = i;
                if (k < b.size)
                {
                    std::nth_element(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                        [v](size_t x, size_t y) { return std::abs(v[x]) > std::abs(v[y]); });
                    order.resize(k);
                    std::sort(order.begin(), order.end());
                    size_t previous = 0;
                    for (size_t j = 0; j < k; ++j)
                    {
                        uint64 gap = j == 0 ? order[0] : order[j] - previous - 1;
                        previous = order[j];
                        do
                        {
                            unsigned char c = static_cast<unsigned char>(gap & 0x7f);
                            gap >>= 7;
                            if (gap) c |= 0x80;
                            out.indices_.push_back(c);
                        } while (gap);
                    }
                }
                out.sizes_.push_back(b.size);
                out.kept_.push_back(k);

                kept.resize(k);
                for (size_t j = 0; j < k; ++j)
                    kept[j] = v[order[j]];
                if (options_.bits == 32)
                {
                    const size_t at = out.codes_.size();
                    out.codes_.resize(at + 4 * k);
                    if (k > 0)
                        std::memcpy(&out.codes_[at], kept.data(), 4 * k);
                }
                else
                {
                    for (size_t c0 = 0; c0 < k; c0 += compressed_delta::chunk())
                    {
                        const size_t c1 = std::min(k, c0 + compressed_delta::chunk());
                        float amax = 0;
                        for (size_t j = c0; j < c1; ++j)
                            amax = std::max(amax, std::abs(kept[j]));
                        const float scale = amax / qmax;
                        out.scales_.push_back(scale);
                        const float inv = scale > 0 ? 1.0f / scale : 0.0f;
                        for (size_t j = c0; j < c1; ++j)
                        {
                            const int code = static_cast<int>(std::max(-qmax, std::min(qmax, std::round(kept[j] * inv))));
                            kept[j] = scale * code;
                            if (options_.bits == 8)
                            {
                                out.codes_.push_back(static_cast<unsigned char>(static_cast<signed char>(code)));
                            }
                            else
                            {
                                const unsigned char nibble = static_cast<unsigned char>(code + 8);
                                if (low_nibble) out.codes_.push_back(nibble);
                                else out.codes_.back() |= static_cast<unsigned char>(nibble << 4);
                                low_nibble = !low_nibble;
                            }
                        }
                    }
                    low_nibble = true;
                }
                // What was sent leaves the residual; the rest stays in it.
                for (size_t j = 0; j < k; ++j)
                    v[order[j]] -= kept[j];
            }
        }

        friend void serialize (const delta_encoder& item, std::ostream& out)
        {
            serialize("delta_encoder", out);
            serialize(item.options_, out);
            serialize(item.residual_, out);
        }

        friend void deserialize (delta_encoder& item, std::istream& in)
        {
            impl::check_version(in, "delta_encoder");
            deserialize(item.options_, in);
            deserialize_bounded(item.residual_, in);
        }

    private:
        delta_codec_options options_;
        std::vector<float> residual_;
    };

    void serialize (const delta_encoder& item, std::ostream& out);
    void deserialize (delta_encoder& item, std::istream& in);

// ----------------------------------------------------------------------------------------

    class curvature_summary
    {
    public:
        bool empty () const { return rows_.empty(); }
        size_t num_blocks () const { return rows_.size(); }

        void build (
            const parameter_layout& layout,
            const std::vector<float>& diagonal
        )
        {
            DLIB_CASSERT(diagonal.size() == layout.size(), "curvature_summary: the diagonal does not match the layout");
            clear();
            for (const parameter_block& b : layout.blocks())
            {
                rows_.push_back(b.rows);
                cols_.push_back(b.cols);
                const float* d = diagonal.data() + b.offset;
                if (b.is_matrix())
                {
                    /* Factored as the second moments of Adafactor: the row sums and the column
                       sums of the block, whose outer product divided by the total restores a
                       rank-one estimate of every entry. */
                    std::vector<float> r(static_cast<size_t>(b.rows), 0.0f), c(static_cast<size_t>(b.cols), 0.0f);
                    for (long i = 0; i < b.rows; ++i)
                        for (long j = 0; j < b.cols; ++j)
                        {
                            const float x = std::max(0.0f, d[static_cast<size_t>(i) * b.cols + j]);
                            r[static_cast<size_t>(i)] += x;
                            c[static_cast<size_t>(j)] += x;
                        }
                    row_.push_back(std::move(r));
                    col_.push_back(std::move(c));
                }
                else
                {
                    std::vector<float> r(d, d + b.size);
                    for (float& x : r) x = std::max(0.0f, x);
                    row_.push_back(std::move(r));
                    col_.push_back(std::vector<float>());
                }
            }
        }

        void clear ()
        {
            rows_.clear();
            cols_.clear();
            row_.clear();
            col_.clear();
        }

        bool matches (const parameter_layout& layout) const
        {
            if (rows_.size() != layout.num_blocks())
                return false;
            for (size_t i = 0; i < rows_.size(); ++i)
            {
                const parameter_block& b = layout.block(i);
                if (rows_[i] != b.rows || cols_[i] != b.cols)
                    return false;
                if (b.is_matrix())
                {
                    if (row_[i].size() != static_cast<size_t>(b.rows) || col_[i].size() != static_cast<size_t>(b.cols))
                        return false;
                }
                else if (row_[i].size() != b.size || !col_[i].empty())
                {
                    return false;
                }
            }
            return true;
        }

        void expand_normalized (
            size_t index,
            float* dst
        ) const
        {
            DLIB_CASSERT(index < rows_.size());
            /* Computed in double precision and bounded, so that statistics too small or too
               large for a float, and the products of their factors, stay finite. */
            const double bound = 1e6;
            const std::vector<float>& r = row_[index];
            const std::vector<float>& c = col_[index];
            if (c.empty())
            {
                double mean = 0;
                for (float x : r) mean += x;
                mean /= std::max<size_t>(1, r.size());
                for (size_t i = 0; i < r.size(); ++i)
                    dst[i] = mean > 0 ? static_cast<float>(std::min(bound, r[i] / mean)) : 1.0f;
                return;
            }
            double sr = 0, sc = 0;
            for (float x : r) sr += x;
            for (float x : c) sc += x;
            const long nr = rows_[index], nc = cols_[index];
            if (!(sr > 0 && sc > 0))
            {
                std::fill(dst, dst + static_cast<size_t>(nr) * nc, 1.0f);
                return;
            }
            // r_i c_j / S divided by its mean over the block, S r_i c_j m n / (sr sc S).
            const double ki = static_cast<double>(nr) / sr, kj = static_cast<double>(nc) / sc;
            for (long i = 0; i < nr; ++i)
            {
                const double ri = r[static_cast<size_t>(i)] * ki;
                float* row = dst + static_cast<size_t>(i) * nc;
                for (long j = 0; j < nc; ++j)
                    row[j] = static_cast<float>(std::min(bound, ri * (c[static_cast<size_t>(j)] * kj)));
            }
        }

        friend void serialize (const curvature_summary& item, std::ostream& out)
        {
            serialize("curvature_summary", out);
            serialize(item.rows_, out);
            serialize(item.cols_, out);
            serialize(item.row_, out);
            serialize(item.col_, out);
        }

        friend void deserialize (curvature_summary& item, std::istream& in)
        {
            impl::check_version(in, "curvature_summary");
            deserialize_bounded(item.rows_, in);
            deserialize_bounded(item.cols_, in);
            const size_t n = item.rows_.size();
            if (item.cols_.size() != n)
                throw serialization_error("curvature_summary: tables of different lengths");
            for (size_t i = 0; i < n; ++i)
                if (item.rows_[i] <= 0 || item.cols_[i] <= 0 ||
                    static_cast<uint64>(item.rows_[i]) > (uint64(1) << 40) / static_cast<uint64>(item.cols_[i]))
                    throw serialization_error("curvature_summary: invalid block shape");
            deserialize_bounded(item.row_, in);
            deserialize_bounded(item.col_, in);
            if (item.row_.size() != n || item.col_.size() != n)
                throw serialization_error("curvature_summary: tables of different lengths");
            for (size_t i = 0; i < n; ++i)
            {
                const bool matrix = item.rows_[i] > 1 && item.cols_[i] > 1;
                const size_t r = static_cast<size_t>(item.rows_[i]), c = static_cast<size_t>(item.cols_[i]);
                if (matrix ? (item.row_[i].size() != r || item.col_[i].size() != c)
                           : (item.row_[i].size() != r * c || !item.col_[i].empty()))
                    throw serialization_error("curvature_summary: a table does not match its block");
                for (float x : item.row_[i])
                    if (!(x >= 0 && x < std::numeric_limits<float>::infinity()))
                        throw serialization_error("curvature_summary: invalid value");
                for (float x : item.col_[i])
                    if (!(x >= 0 && x < std::numeric_limits<float>::infinity()))
                        throw serialization_error("curvature_summary: invalid value");
            }
        }

    private:
        std::vector<long> rows_, cols_;
        std::vector<std::vector<float>> row_, col_;
    };

    void serialize (const curvature_summary& item, std::ostream& out);
    void deserialize (curvature_summary& item, std::istream& in);

// ----------------------------------------------------------------------------------------

    enum class consolidation_method
    {
        average = 0,
        outer_momentum = 1,
        curvature_consensus = 2
    };

    inline std::string consolidation_method_name (consolidation_method m)
    {
        switch (m)
        {
            case consolidation_method::average: return "average";
            case consolidation_method::outer_momentum: return "outer_momentum";
            case consolidation_method::curvature_consensus: return "curvature_consensus";
        }
        return "average";
    }

    inline consolidation_method consolidation_method_from_name (const std::string& name)
    {
        if (name == "outer_momentum") return consolidation_method::outer_momentum;
        if (name == "curvature_consensus") return consolidation_method::curvature_consensus;
        if (name == "average") return consolidation_method::average;
        throw error("consolidation_method_from_name: unknown method '" + name + "'");
    }

    struct consolidation_options
    {
        consolidation_method method = consolidation_method::curvature_consensus;
        double outer_learning_rate = 1.0;
        double outer_momentum = 0.0;
        double curvature_mix = 0.5;
        double curvature_ceiling = 10;
        double consensus_power = 0.5;
        double spectral_clip = 0.0;
        long spectral_rank = 8;
        double norm_bound = 2.5;
        double trust_ratio = 0;
        double staleness_decay = 0.5;
        long max_staleness = 1;
        std::vector<double> step_multipliers = {0.5, 1.0, 2.0};
    };

    inline void serialize (const consolidation_options& item, std::ostream& out)
    {
        serialize("consolidation_options", out);
        serialize(static_cast<int>(item.method), out);
        serialize(item.outer_learning_rate, out);
        serialize(item.outer_momentum, out);
        serialize(item.curvature_mix, out);
        serialize(item.curvature_ceiling, out);
        serialize(item.consensus_power, out);
        serialize(item.spectral_clip, out);
        serialize(item.spectral_rank, out);
        serialize(item.norm_bound, out);
        serialize(item.trust_ratio, out);
        serialize(item.staleness_decay, out);
        serialize(item.max_staleness, out);
        serialize(item.step_multipliers, out);
    }

    inline void deserialize (consolidation_options& item, std::istream& in)
    {
        impl::check_version(in, "consolidation_options");
        int method = 0;
        deserialize(method, in);
        if (method < 0 || method > 2)
            throw serialization_error("consolidation_options: unknown method");
        item.method = static_cast<consolidation_method>(method);
        deserialize(item.outer_learning_rate, in);
        deserialize(item.outer_momentum, in);
        deserialize(item.curvature_mix, in);
        deserialize(item.curvature_ceiling, in);
        deserialize(item.consensus_power, in);
        deserialize(item.spectral_clip, in);
        deserialize(item.spectral_rank, in);
        deserialize(item.norm_bound, in);
        deserialize(item.trust_ratio, in);
        deserialize(item.staleness_decay, in);
        deserialize(item.max_staleness, in);
        deserialize_bounded(item.step_multipliers, in);
        const consolidation_options& o = item;
        bool valid = o.outer_learning_rate > 0 && o.outer_learning_rate <= 1e6 && o.outer_momentum >= 0 &&
            o.outer_momentum < 1 && o.curvature_mix >= 0 && o.curvature_mix <= 1 && o.curvature_ceiling >= 1 &&
            o.curvature_ceiling <= 1e6 && o.consensus_power >= 0 && o.consensus_power <= 100 &&
            o.spectral_clip >= 0 && o.spectral_clip <= 1e6 && o.spectral_rank > 0 && o.spectral_rank <= 4096 &&
            o.norm_bound >= 0 && o.norm_bound <= 1e6 && o.trust_ratio >= 0 && o.trust_ratio <= 1e6 &&
            o.staleness_decay >= 0 && o.staleness_decay <= 1 && o.max_staleness >= 0 && o.max_staleness <= 1000000 &&
            o.step_multipliers.size() <= 64;
        for (double m : o.step_multipliers)
            valid = valid && m > 0 && m <= 1e6;
        if (!valid)
            throw serialization_error("consolidation_options: values out of range");
    }

// ----------------------------------------------------------------------------------------

    inline void apply_consolidated_step (
        const parameter_layout& layout,
        std::vector<float>& params,
        const std::vector<float>& direction,
        double step,
        double trust_ratio = 0
    )
    {
        DLIB_CASSERT(params.size() == layout.size() && direction.size() == layout.size(),
            "apply_consolidated_step: the vectors do not match the layout");
        for (const parameter_block& b : layout.blocks())
        {
            float* p = params.data() + b.offset;
            const float* d = direction.data() + b.offset;
            double factor = step;
            if (trust_ratio > 0)
            {
                /* The step of a block stays within a fixed fraction of the norm of the block,
                   with a floor for blocks still near zero. Every replica computes the same
                   factor from the same values, so the states remain identical. */
                double pn = 0, dn = 0;
                for (size_t i = 0; i < b.size; ++i)
                {
                    pn += static_cast<double>(p[i]) * p[i];
                    dn += static_cast<double>(d[i]) * d[i];
                }
                const double limit = trust_ratio * std::max(std::sqrt(pn), 1e-3 * std::sqrt(static_cast<double>(b.size)));
                const double length = std::abs(step) * std::sqrt(dn);
                if (length > limit && length > 0)
                    factor = step * limit / length;
            }
            const float f = static_cast<float>(factor);
            for (size_t i = 0; i < b.size; ++i)
                p[i] -= f * d[i];
        }
    }

// ----------------------------------------------------------------------------------------

    class delta_consolidator
    {
    public:
        delta_consolidator () = default;

        delta_consolidator (
            const parameter_layout& layout,
            const consolidation_options& options
        )
        {
            setup(layout, options);
        }

        void setup (
            const parameter_layout& layout,
            const consolidation_options& options
        )
        {
            DLIB_CASSERT(layout.size() > 0, "delta_consolidator: empty layout");
            DLIB_CASSERT(options.outer_learning_rate > 0 && options.outer_momentum >= 0 && options.outer_momentum < 1);
            DLIB_CASSERT(options.curvature_mix >= 0 && options.curvature_mix <= 1 && options.curvature_ceiling >= 1);
            DLIB_CASSERT(options.consensus_power >= 0 && options.spectral_clip >= 0 && options.norm_bound >= 0);
            DLIB_CASSERT(options.spectral_rank > 0 && options.trust_ratio >= 0 && options.max_staleness >= 0);
            DLIB_CASSERT(options.staleness_decay >= 0 && options.staleness_decay <= 1);
            for (double m : options.step_multipliers)
                DLIB_CASSERT(m > 0, "delta_consolidator: step multipliers must be positive");
            layout_ = layout;
            options_ = options;
            momentum_.assign(layout.size(), 0.0f);
            pending_momentum_.clear();
            proposal_.clear();
            contributions_.clear();
            round_ = 0;
        }

        const parameter_layout& layout () const { return layout_; }
        const consolidation_options& options () const { return options_; }
        long round () const { return round_; }
        size_t num_contributions () const { return contributions_.size(); }
        void clear_contributions () { contributions_.clear(); }
        bool has_proposal () const { return !proposal_.empty(); }
        const std::vector<float>& proposal () const { return proposal_; }
        const std::vector<float>& momentum () const { return momentum_; }

        void add (
            const compressed_delta& delta,
            double weight,
            const curvature_summary& curvature = curvature_summary(),
            long base_round = -1
        )
        {
            check_contribution(delta.dimension(), weight, curvature);
            contribution c;
            c.packed = delta;
            c.is_packed = true;
            c.weight = weight;
            c.curvature = curvature;
            c.base_round = base_round < 0 ? round_ : base_round;
            contributions_.push_back(std::move(c));
        }

        void add (
            const std::vector<float>& delta,
            double weight,
            const curvature_summary& curvature = curvature_summary(),
            long base_round = -1
        )
        {
            check_contribution(delta.size(), weight, curvature);
            for (float v : delta)
                if (!(std::abs(v) < std::numeric_limits<float>::infinity()))
                    throw error("delta_consolidator: a contribution holds a value that is not finite");
            contribution c;
            c.dense = delta;
            c.weight = weight;
            c.curvature = curvature;
            c.base_round = base_round < 0 ? round_ : base_round;
            contributions_.push_back(std::move(c));
        }

        double step_size (double multiplier) const
        {
            return options_.method == consolidation_method::average ? multiplier
                                                                    : options_.outer_learning_rate * multiplier;
        }

        bool consolidate ()
        {
            proposal_.clear();
            pending_momentum_.clear();
            const size_t n = layout_.size();

            // Contributions too stale are dropped; the others lose weight with their age.
            std::vector<size_t> used;
            std::vector<double> w;
            double heaviest = 0;
            for (size_t k = 0; k < contributions_.size(); ++k)
            {
                const long age = round_ - contributions_[k].base_round;
                if (age < 0 || age > options_.max_staleness)
                    continue;
                const double weight = contributions_[k].weight * std::pow(options_.staleness_decay, static_cast<double>(age));
                if (!(weight > 0))
                    continue;
                used.push_back(k);
                w.push_back(weight);
                heaviest = std::max(heaviest, weight);
            }
            if (used.empty())
                return false;
            // Relative weights, which a float holds whatever the scale of the ones given.
            for (double& x : w)
                x /= heaviest;

            const bool full = options_.method == consolidation_method::curvature_consensus;
            const size_t nb = layout_.num_blocks();

            /* Norm bounding: each block of each contribution is brought back within a
               multiple of the weighted median of the norms of that block, which bounds what
               a single replica, faulty or hostile, moves. */
            std::vector<std::vector<float>> clip(used.size(), std::vector<float>(nb, 1.0f));
            std::vector<float> scratch;
            if (full && options_.norm_bound > 0 && used.size() >= 3)
            {
                std::vector<std::vector<double>> norms(nb, std::vector<double>(used.size(), 0.0));
                for (size_t u = 0; u < used.size(); ++u)
                {
                    const std::vector<float>& d = values(contributions_[used[u]], scratch);
                    for (size_t b = 0; b < nb; ++b)
                    {
                        const parameter_block& blk = layout_.block(b);
                        double s = 0;
                        for (size_t i = 0; i < blk.size; ++i)
                            s += static_cast<double>(d[blk.offset + i]) * d[blk.offset + i];
                        norms[b][u] = std::sqrt(s);
                    }
                }
                /* The reference is the median of the replicas, one vote each: a replica that
                   declares a large weight moves the mean, never the bound. The median is the
                   lower one, so a block that half the replicas or more leave unchanged is left
                   unchanged by all of them. */
                const std::vector<double> votes(used.size(), 1.0);
                for (size_t b = 0; b < nb; ++b)
                {
                    const double ref = weighted_median(norms[b], votes);
                    for (size_t u = 0; u < used.size(); ++u)
                        if (norms[b][u] > options_.norm_bound * ref)
                            clip[u][b] = static_cast<float>(options_.norm_bound * ref / norms[b][u]);
                }
            }

            /* Weighted sums per coordinate: of the values, of the weights, and of the absolute
               values. The weight of a coordinate mixes the weight of its replica with the
               curvature that replica measured there. */
            std::vector<float> num(n, 0.0f), den(n, 0.0f), mag;
            if (full && options_.consensus_power > 0)
                mag.assign(n, 0.0f);
            std::vector<float> fbuf;
            const float alpha = static_cast<float>(full ? options_.curvature_mix : 0.0);
            /* The normalized curvature is capped: a replica concentrating its whole measure on
               a few coordinates would otherwise own them. */
            const float ceiling = static_cast<float>(options_.curvature_ceiling);
            for (size_t u = 0; u < used.size(); ++u)
            {
                const contribution& c = contributions_[used[u]];
                const std::vector<float>& d = values(c, scratch);
                const bool curved = alpha > 0 && !c.curvature.empty();
                for (size_t b = 0; b < nb; ++b)
                {
                    const parameter_block& blk = layout_.block(b);
                    const float scale = clip[u][b];
                    const float wk = static_cast<float>(w[u]);
                    if (curved)
                    {
                        fbuf.resize(blk.size);
                        c.curvature.expand_normalized(b, fbuf.data());
                    }
                    for (size_t i = 0; i < blk.size; ++i)
                    {
                        const size_t at = blk.offset + i;
                        const float f = curved ? (1 - alpha) + alpha * (fbuf[i] <= ceiling ? fbuf[i] : ceiling) : 1.0f;
                        const float x = scale * d[at];
                        num[at] += wk * f * x;
                        den[at] += wk * f;
                        if (!mag.empty())
                            mag[at] += wk * f * std::abs(x);
                    }
                }
            }

            std::vector<float> direction(n, 0.0f);
            const float gamma = static_cast<float>(options_.consensus_power);
            for (size_t i = 0; i < n; ++i)
            {
                if (!(den[i] > 0))
                    continue;
                float v = num[i] / den[i];
                /* Agreement of the replicas at the coordinate, one when every replica moves
                   it the same way and zero when they cancel out, used as a signal-to-noise
                   ratio to shrink the directions on which they disagree. */
                if (!mag.empty())
                    v = mag[i] > 0 ? v * std::pow(std::abs(num[i]) / mag[i], gamma) : 0.0f;
                direction[i] = v;
            }

            if (full && options_.spectral_clip > 0)
                for (const parameter_block& blk : layout_.blocks())
                    if (blk.is_matrix())
                        clip_spectrum(direction.data() + blk.offset, blk.rows, blk.cols);

            // Values at the edge of the range of a float can still overflow the sums: no direction then.
            for (float v : direction)
                if (!(std::abs(v) <= std::numeric_limits<float>::max()))
                    return false;

            if (options_.method == consolidation_method::average)
            {
                proposal_ = std::move(direction);
                return true;
            }

            // Nesterov look-ahead on the consolidated direction, kept pending until commit().
            const float mu = static_cast<float>(options_.outer_momentum);
            pending_momentum_.resize(n);
            proposal_.resize(n);
            for (size_t i = 0; i < n; ++i)
            {
                pending_momentum_[i] = mu * momentum_[i] + direction[i];
                proposal_[i] = direction[i] + mu * pending_momentum_[i];
            }
            return true;
        }

        void apply (
            std::vector<float>& params,
            double multiplier
        ) const
        {
            DLIB_CASSERT(has_proposal(), "delta_consolidator: no proposal to apply");
            apply_consolidated_step(layout_, params, proposal_, step_size(multiplier), options_.trust_ratio);
        }

        static size_t choose_multiplier (
            const std::vector<std::vector<double>>& losses,
            const std::vector<double>& weights
        )
        {
            /* losses[k][j] is the validation loss replica k measured for candidate j, the
               current state being candidate 0. Returns the candidate whose change of loss
               over the current state is the lowest and negative, 0 when none is. */
            DLIB_CASSERT(losses.size() == weights.size() && !losses.empty());
            const size_t candidates = losses[0].size();
            DLIB_CASSERT(candidates > 0);
            /* A replica of zero weight, or whose loss of the current state is not finite, is
               left out of every candidate. A candidate loss that is not finite counts as an
               infinite increase, so that a step which breaks one replica is never chosen on
               the losses of the others alone. */
            const double largest = std::numeric_limits<double>::max();
            std::vector<size_t> rows;
            for (size_t k = 0; k < losses.size(); ++k)
            {
                DLIB_CASSERT(losses[k].size() == candidates && weights[k] >= 0);
                if (weights[k] > 0 && std::abs(losses[k][0]) <= largest)
                    rows.push_back(k);
            }
            size_t best = 0;
            double best_change = 0;
            if (rows.empty())
                return best;
            std::vector<double> change(rows.size()), negated(rows.size()), w(rows.size());
            for (size_t i = 0; i < rows.size(); ++i)
                w[i] = weights[rows[i]];
            for (size_t j = 1; j < candidates; ++j)
            {
                for (size_t i = 0; i < rows.size(); ++i)
                {
                    const double d = losses[rows[i]][j] - losses[rows[i]][0];
                    change[i] = std::abs(d) <= largest ? d : std::numeric_limits<double>::infinity();
                    negated[i] = -change[i];
                }
                /* With three replicas or more, a candidate is judged by the weighted median of
                   its changes, the midpoint of the lower and the upper weighted medians: replicas
                   holding less than half of the weight keep it within the changes of the others.
                   With fewer, no median outvotes anyone, and the weighted mean judges, so that a
                   step that helps one replica and harms the other as much is not taken. */
                double m = 0;
                if (rows.size() >= 3)
                {
                    m = (weighted_median(change, w) - weighted_median(negated, w)) / 2;
                }
                else
                {
                    double sum = 0, total = 0;
                    for (size_t i = 0; i < rows.size(); ++i)
                    {
                        sum += w[i] * change[i];
                        total += w[i];
                    }
                    m = sum / total;
                }
                if (m < best_change)
                {
                    best_change = m;
                    best = j;
                }
            }
            return best;
        }

        void commit ()
        {
            DLIB_CASSERT(has_proposal(), "delta_consolidator: no proposal to commit");
            if (!pending_momentum_.empty())
                momentum_.swap(pending_momentum_);
            finish_round();
        }

        void reject ()
        {
            // The direction is discarded and the momentum loses part of what it carried.
            const float mu = static_cast<float>(options_.outer_momentum);
            for (float& m : momentum_)
                m *= mu;
            finish_round();
        }

        friend void serialize (const delta_consolidator& item, std::ostream& out)
        {
            serialize("delta_consolidator", out);
            serialize(item.layout_, out);
            serialize(item.options_, out);
            serialize(item.momentum_, out);
            serialize(item.round_, out);
        }

        friend void deserialize (delta_consolidator& item, std::istream& in)
        {
            impl::check_version(in, "delta_consolidator");
            deserialize(item.layout_, in);
            deserialize(item.options_, in);
            deserialize_bounded(item.momentum_, in);
            deserialize(item.round_, in);
            if (item.momentum_.size() != item.layout_.size() || item.round_ < 0)
                throw serialization_error("delta_consolidator: momentum of a wrong size");
            for (float v : item.momentum_)
                if (!(std::abs(v) <= std::numeric_limits<float>::max()))
                    throw serialization_error("delta_consolidator: momentum that is not finite");
            item.pending_momentum_.clear();
            item.proposal_.clear();
            item.contributions_.clear();
        }

    private:
        struct contribution
        {
            compressed_delta packed;
            std::vector<float> dense;
            bool is_packed = false;
            double weight = 0;
            curvature_summary curvature;
            long base_round = 0;
        };

        void check_contribution (
            size_t dimension,
            double weight,
            const curvature_summary& curvature
        ) const
        {
            if (dimension != layout_.size())
                throw error("delta_consolidator: a contribution of dimension " + std::to_string(dimension) +
                            " does not match the layout of dimension " + std::to_string(layout_.size()));
            if (!(weight > 0 && weight < std::numeric_limits<double>::infinity()))
                throw error("delta_consolidator: a contribution needs a positive finite weight");
            if (!curvature.empty() && !curvature.matches(layout_))
                throw error("delta_consolidator: the curvature of a contribution does not match the layout");
        }

        static const std::vector<float>& values (
            const contribution& c,
            std::vector<float>& scratch
        )
        {
            if (!c.is_packed)
                return c.dense;
            decode_delta(c.packed, scratch);
            return scratch;
        }

        void finish_round ()
        {
            pending_momentum_.clear();
            proposal_.clear();
            contributions_.clear();
            ++round_;
        }

        /* Brings the singular values of a block above spectral_clip times their root mean
           square down to that level, the part of the isotropic correction that carries
           most of its benefit. The leading singular triplets come from a randomized
           decomposition; the root mean square of the whole spectrum comes from the
           Frobenius norm. */
        void clip_spectrum (
            float* data,
            long rows,
            long cols
        ) const
        {
            const long smallest = std::min(rows, cols);
            if (smallest < 2)
                return;
            double fro = 0;
            const size_t count = static_cast<size_t>(rows) * cols;
            for (size_t i = 0; i < count; ++i)
                fro += static_cast<double>(data[i]) * data[i];
            if (!(fro > 0))
                return;
            const double threshold = options_.spectral_clip * std::sqrt(fro / smallest);
            matrix<double> a(rows, cols), u, s, v;
            for (long i = 0; i < rows; ++i)
                for (long j = 0; j < cols; ++j)
                    a(i, j) = data[static_cast<size_t>(i) * cols + j];
            const long rank = std::min<long>(options_.spectral_rank, smallest);
            if (smallest <= 2 * rank)
                svd3(a, u, s, v);
            else
                svd_fast(a, u, s, v, static_cast<unsigned long>(rank), 2);
            for (long k = 0; k < s.nr(); ++k)
            {
                const double excess = s(k) - threshold;
                if (!(excess > 0))
                    continue;
                for (long i = 0; i < rows; ++i)
                {
                    const double ui = excess * u(i, k);
                    float* row = data + static_cast<size_t>(i) * cols;
                    for (long j = 0; j < cols; ++j)
                        row[j] -= static_cast<float>(ui * v(j, k));
                }
            }
        }

        parameter_layout layout_;
        consolidation_options options_;
        std::vector<float> momentum_, pending_momentum_, proposal_;
        std::vector<contribution> contributions_;
        long round_ = 0;
    };

    void serialize (const delta_consolidator& item, std::ostream& out);
    void deserialize (delta_consolidator& item, std::istream& in);

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_H_ 



