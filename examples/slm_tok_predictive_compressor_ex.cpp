// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Lossless text compression with a Transformer trained in dlib, its tokenizer and weights
    compiled into the program.

    A language model is a distribution over what comes next, and an arithmetic coder turns a
    distribution into a code whose length is the model's cross-entropy on the data. This
    program trains a small decoder on a corpus, compiles the tokenizer and the quantized
    weights into its own source through a generated header, and codes text with the
    network's predictions mixed with those of adaptive context models.

    HOW A TOKEN IS CODED

    The network predicts a distribution over tokens. The context models, which learn the
    data as it is coded, predict bytes. The two meet on the bytes: the tokens are kept in
    lexicographic order of their byte patterns, so that the tokens sharing a prefix form a
    contiguous range and each further bit of the next byte splits that range in two. The
    probability the network gives to a bit is then the mass of one half of the range over
    the mass of the whole, which is exact. A token is coded as the bits of its bytes, plus,
    wherever the vocabulary allows both, a decision saying whether the token ends at the
    current byte. Bits that the vocabulary leaves no choice for are not coded.

    Each decision is predicted by the network, by maps that recalibrate it, and by 107
    context models over the bytes and tokens already coded (orders one to thirty-two, sparse
    contexts, words, the line above, the markup of the wiki text, the current token and page,
    long-range matches), which keep a bit history per context and decision and learn what
    each history adds to the network. Eight mixers, whose weights are selected by small
    contexts, combine these predictions, a final mixer combines theirs, four adaptive maps
    refine the result, and a binary arithmetic coder writes it.

    The network also learns while it codes: each stream has output biases that move after
    every token by a step of gradient descent on the coding cost, so that the distribution
    follows the tokens the stream's own text uses. The decoder takes the same steps.

    WHY THE CODING PATH HAS ITS OWN ARITHMETIC

    A decoder must reproduce every probability the encoder used, to the last bit, or it
    decodes something else. The coding path therefore runs on the processor, on one thread,
    with sums taken in a fixed order, and computes exponentials, logarithms and rotations
    with its own routines built from the basic IEEE operations, since the implementations
    of the C library differ between versions and dispatch on the processor. Training
    runs on whatever dlib uses, a graphics card included; only the weights cross over.
    Multiplications and additions are fused only where the source asks for it. The matrix
    products run on integers: the inputs of a product are rounded to signed bytes with one
    scale per row, multiplied by the quantized weights, and summed exactly in 32 bits before
    the two scales bring the result back to floating point. An integer sum does not depend
    on its order, so the kernels for AVX-512 VNNI, AVX-VNNI and AVX2 give the values of the
    portable code. The attention products have kernels for AVX2 and FMA that apply the same
    fused operations in the same order as the scalar code, and the normalisations, the
    softmax and the gates have AVX2 kernels that apply the scalar code's operations, element
    by element, with nothing fused. Every kernel is compiled whatever the options of the
    build and chosen at run time from the extensions dlib::get_cpu_features() finds, so that
    every processor computes the same values; DLIB_DISABLE_CPU_FEATURES runs a build through
    the paths of a processor with fewer extensions.

    HOW THE TEXT IS CUT FOR THE NETWORK

    A long input is cut at line boundaries into streams coded side by side: the network
    advances every stream by one token per step, so that each step multiplies the weights by
    several rows at once. The context models are shared by all streams and the match models
    search all of them. Before that, the pages of a wiki dump are put in another order, grouped
    by category, which the decoder undoes by sorting them by number.

    WHAT IS COMPILED IN

    slm_tok_predictive_compressor_ex.h holds the merge table of the tokenizer and the
    weights, quantized per output channel and compressed by the program's own context mixing
    coder. It is written by this program: the tokenizer by --build-tokenizer, the network by
    --train at the end of training or by --embed from a saved model, in the working
    directory unless --header names another path. Placed next to this file, it is compiled
    in by the next build. Without a network the program still compresses, with the context
    models alone.

    --build-tokenizer cuts the corpus as --segmentation says before learning the merges:
    spaces, the default, lets a word merge with the space before it; classes also cuts where
    letters, digits and other bytes meet, so that a word keeps the same tokens whatever
    punctuation or markup touches it. With --fold-capitals, the tokenizer works on a form of
    the text where every ASCII capital is written in lower case after a marker byte, so that
    "The" and "the" share their tokens; the program folds the text before tokenizing it and
    unfolds what it decodes, and the archive still checks the original. A network belongs to
    the tokenizer it was trained with. --build-tokenizer therefore writes a header without a
    network, the saved network and the checkpoint of a run record their tokenizer, and
    --train and --embed refuse a saved network or a checkpoint whose tokenizer differs from
    the compiled-in one.

    Two macros reduce the program to what coding needs, without the network definition and
    the training: SLM_CODEC_ONLY set to 1 keeps compression and decompression, and
    SLM_DECOMPRESSOR_ONLY set to 1 keeps decompression alone, the executable to hand over
    with an archive. Both keep the same options and use the same parts of dlib for the
    tokenizer, the checksum and the command line, which are header-only, so that one
    compiler command builds them, for instance from a build directory under the one holding
    this file:
        g++ -std=c++14 -O3 -s -DSLM_DECOMPRESSOR_ONLY=1 -I../.. ../slm_tok_predictive_compressor_ex.cpp

    Typical sequence, from a build directory under the one holding this file:
        ./slm_tok_predictive_compressor_ex --build-tokenizer --input corpus.txt \
            --header ../slm_tok_predictive_compressor_ex.h
        (rebuild)
        ./slm_tok_predictive_compressor_ex --train --input corpus.txt \
            --header ../slm_tok_predictive_compressor_ex.h
        (rebuild)
        ./slm_tok_predictive_compressor_ex --compress --input data.txt
        ./slm_tok_predictive_compressor_ex --decompress --input data.txt.dlc
*/

#ifndef SLM_DECOMPRESSOR_ONLY
#define SLM_DECOMPRESSOR_ONLY 0
#endif
#ifndef SLM_CODEC_ONLY
#define SLM_CODEC_ONLY 0
#endif
#if SLM_DECOMPRESSOR_ONLY && SLM_CODEC_ONLY
#error "SLM_DECOMPRESSOR_ONLY and SLM_CODEC_ONLY exclude each other"
#endif
#if SLM_DECOMPRESSOR_ONLY || SLM_CODEC_ONLY
#define SLM_TRAINING 0
#else
#define SLM_TRAINING 1
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// The vector kernels are compiled for their instruction sets whatever the options of the build.
#if defined(__x86_64__) || defined(_M_X64)
#define SLM_CODEC_X86 1
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#define SLM_AVX2_FMA __attribute__((target("avx2,fma")))
#define SLM_AVX2 __attribute__((target("avx2")))
#define SLM_AVX512_VNNI __attribute__((target("avx2,avx512f,avx512vl,avx512vnni")))
#define SLM_AVX_VNNI __attribute__((target("avx2,avxvnni")))
#else
#define SLM_AVX2_FMA
#define SLM_AVX2
#define SLM_AVX512_VNNI
#define SLM_AVX_VNNI
#endif
// The compilers that know the VNNI instructions.
#if (defined(__clang__) && __clang_major__ >= 12) || (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 11) \
    || (defined(_MSC_VER) && _MSC_VER >= 1930)
#define SLM_VNNI_KERNELS 1
#else
#define SLM_VNNI_KERNELS 0
#endif
#else
#define SLM_CODEC_X86 0
#define SLM_VNNI_KERNELS 0
#endif

#include <dlib/cmd_line_parser.h>
#include <dlib/crc32.h>
#include <dlib/simd/cpu_features.h>
#include <dlib/tokenizer.h>

#if SLM_TRAINING
#include <dlib/dnn.h>
#include <dlib/misc_api.h>

#include <random>
#endif

/*
    The compiled-in tokenizer and network. The header is optional: without it the program
    can only build a tokenizer. A header created after a previous build of the target needs
    an explicit rebuild. Defining WITH_EMBEDDED_CODEC_DATA, or naming another file through
    EMBEDDED_CODEC_HEADER, takes priority over the detection.
*/
#ifndef EMBEDDED_CODEC_HEADER
#  define EMBEDDED_CODEC_HEADER "slm_tok_predictive_compressor_ex.h"
#endif
#if !defined(WITH_EMBEDDED_CODEC_DATA) && defined(__has_include)
#  if __has_include(EMBEDDED_CODEC_HEADER)
#    define WITH_EMBEDDED_CODEC_DATA 1
#  endif
#endif
#ifdef WITH_EMBEDDED_CODEC_DATA
#  include EMBEDDED_CODEC_HEADER
#endif

/*
    The coding path fuses a multiplication and an addition only where it calls std::fma or an
    FMA instruction. The pragmas below tell Clang and Visual C++ so. GCC finds nothing to fuse
    in the vector kernels, which multiply only inside fused multiply-adds, and fuses elsewhere
    only when FMA belongs to the instructions of the whole build (-mfma, -march=native, 64-bit
    ARM); such a build also needs -ffp-contract=off for its archives to decode with other
    builds.
*/
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(_MSC_VER)
#pragma fp_contract (off)
#endif

namespace tok_codec
{
    // ----------------------------------------------------------------------------------------
    // Deterministic arithmetic

    namespace det
    {
        inline double round_even (double x)
        {
            const double magic = 6755399441055744.0;    // 1.5 * 2^52, exact for |x| < 2^51
            return (x + magic) - magic;
        }

        inline double pow2i (int k)
        {
            const uint64_t bits = static_cast<uint64_t>(k + 1023) << 52;
            double r;
            std::memcpy(&r, &bits, sizeof(r));
            return r;
        }

        inline double exp (double x)
        {
            if (x < -700.0) x = -700.0;
            if (x > 700.0) x = 700.0;
            const double k = round_even(x * 1.44269504088896340736);
            const double r = (x - k * 6.93147180369123816490e-01) - k * 1.90821492927058770002e-10;
            double p = 1.0 / 6227020800.0;
            p = p * r + 1.0 / 479001600.0;
            p = p * r + 1.0 / 39916800.0;
            p = p * r + 1.0 / 3628800.0;
            p = p * r + 1.0 / 362880.0;
            p = p * r + 1.0 / 40320.0;
            p = p * r + 1.0 / 5040.0;
            p = p * r + 1.0 / 720.0;
            p = p * r + 1.0 / 120.0;
            p = p * r + 1.0 / 24.0;
            p = p * r + 1.0 / 6.0;
            p = p * r + 0.5;
            p = p * r + 1.0;
            p = p * r + 1.0;
            return p * pow2i(static_cast<int>(k));
        }

        inline double log (double x)
        {
            uint64_t bits;
            std::memcpy(&bits, &x, sizeof(bits));
            int e = static_cast<int>((bits >> 52) & 0x7ff) - 1023;
            double m = x * pow2i(-e);       // exact: the mantissa, in [1, 2)
            if (m > 1.41421356237309504880) { m *= 0.5; e += 1; }
            const double f = (m - 1.0) / (m + 1.0);
            const double f2 = f * f;
            double s = 1.0 / 25.0;
            for (int k = 23; k >= 3; k -= 2) s = s * f2 + 1.0 / k;
            s = s * f2 + 1.0;
            return e * 0.69314718055994530942 + 2.0 * f * s;
        }

        // Sine and cosine of an angle of moderate size (|x| below a few thousand radians).
        inline void sincos (double x, double& s, double& c)
        {
            const double k = round_even(x * 0.63661977236758134308);
            const double r = (x - k * 1.57079632673412561417e+00) - k * 6.07710050650619224932e-11;
            const double r2 = r * r;
            double sn = -1.0 / 1307674368000.0;
            sn = sn * r2 + 1.0 / 6227020800.0;
            sn = sn * r2 - 1.0 / 39916800.0;
            sn = sn * r2 + 1.0 / 362880.0;
            sn = sn * r2 - 1.0 / 5040.0;
            sn = sn * r2 + 1.0 / 120.0;
            sn = sn * r2 - 1.0 / 6.0;
            sn = (sn * r2) * r + r;
            double cs = 1.0 / 20922789888000.0;
            cs = cs * r2 - 1.0 / 87178291200.0;
            cs = cs * r2 + 1.0 / 479001600.0;
            cs = cs * r2 - 1.0 / 3628800.0;
            cs = cs * r2 + 1.0 / 40320.0;
            cs = cs * r2 - 1.0 / 720.0;
            cs = cs * r2 + 1.0 / 24.0;
            cs = cs * r2 - 0.5;
            cs = cs * r2 + 1.0;
            switch (static_cast<int>(static_cast<int64_t>(k) & 3)) {
                case 0: s = sn; c = cs; break;
                case 1: s = cs; c = -sn; break;
                case 2: s = -sn; c = -cs; break;
                default: s = -cs; c = sn; break;
            }
        }

        /*
            exp in single precision, written so that loops over arrays vectorize. The input is
            clamped to [-87, 88] through its bit pattern, which orders the floats below -87 as
            unsigned integers and those above 88 as signed ones, and which a compiler compares
            in vectors under any floating-point options.
        */
        inline float expf (float x)
        {
            uint32_t bits_in;
            std::memcpy(&bits_in, &x, sizeof(bits_in));
            bits_in = bits_in > 0xC2AE0000u ? 0xC2AE0000u : bits_in;           // -87.0f
            int32_t signed_in = static_cast<int32_t>(bits_in);
            signed_in = signed_in > 0x42B00000 ? 0x42B00000 : signed_in;        // 88.0f
            std::memcpy(&x, &signed_in, sizeof(x));
            const float magic = 12582912.0f;                // 1.5 * 2^23
            const float k = (x * 1.44269504088896341f + magic) - magic;
            float r = x - k * 0.693359375f;
            r = r - k * -2.12194440e-4f;
            float p = 1.9875691500e-4f;
            p = p * r + 1.3981999507e-3f;
            p = p * r + 8.3334519073e-3f;
            p = p * r + 4.1665795894e-2f;
            p = p * r + 1.6666665459e-1f;
            p = p * r + 5.0000001201e-1f;
            p = (p * (r * r) + r) + 1.0f;
            const int32_t bits = (static_cast<int32_t>(k) + 127) << 23;
            float scale;
            std::memcpy(&scale, &bits, sizeof(scale));
            return p * scale;
        }
    }

    // ----------------------------------------------------------------------------------------
    // Probabilities in the logistic domain

    /*
        Probabilities are 16-bit integers, P(bit = 1) * 65536. Models are combined in the
        logistic domain, where a probability p is held as st = 256 * ln(p / (1 - p)), clamped
        to +/- ST_LIMIT. Both conversions are tables built at start-up with the routines above.
    */
    const int ST_LIMIT = 3071;

    class logistic
    {
    public:
        logistic ()
        {
            squash_table.resize(2 * ST_LIMIT + 1);
            for (int st = -ST_LIMIT; st <= ST_LIMIT; ++st) {
                const double p = 65536.0 / (1.0 + det::exp(-st / 256.0));
                const int v = static_cast<int>(det::round_even(p));
                squash_table[st + ST_LIMIT] = static_cast<uint16_t>(std::min(65535, std::max(1, v)));
            }
            stretch_table.resize(65536);
            for (int p = 0; p < 65536; ++p) {
                const double q = (p + 0.5) / 65536.0;
                const int v = static_cast<int>(det::round_even(256.0 * det::log(q / (1.0 - q))));
                stretch_table[p] = static_cast<int16_t>(std::min(ST_LIMIT, std::max(-ST_LIMIT, v)));
            }
        }

        int squash (int st) const
        {
            st = st > ST_LIMIT ? ST_LIMIT : (st < -ST_LIMIT ? -ST_LIMIT : st);
            return squash_table[st + ST_LIMIT];
        }

        int stretch (int p16) const { return stretch_table[p16 & 0xffff]; }

        // The table of stretch, for loops that read it many times.
        const int16_t* stretch_data () const { return stretch_table.data(); }

    private:
        std::vector<uint16_t> squash_table;
        std::vector<int16_t> stretch_table;
    };

    inline const logistic& logistic_tables ()
    {
        static const logistic tables;
        return tables;
    }

    // ----------------------------------------------------------------------------------------
    // Binary arithmetic coder

    class bit_encoder
    {
    public:
        explicit bit_encoder (std::vector<uint8_t>& out_) : out(out_) {}

        // p16 is P(bit == 1) * 65536, in [1, 65535].
        void encode (int bit, uint32_t p16)
        {
            const uint32_t xmid = x1 + static_cast<uint32_t>((static_cast<uint64_t>(x2 - x1) * p16) >> 16);
            if (bit) x2 = xmid; else x1 = xmid + 1;
            while (((x1 ^ x2) & 0xff000000u) == 0) {
                out.push_back(static_cast<uint8_t>(x2 >> 24));
                x1 <<= 8;
                x2 = (x2 << 8) | 255;
            }
        }

        void flush ()
        {
            for (int i = 0; i < 4; ++i) { out.push_back(static_cast<uint8_t>(x1 >> 24)); x1 <<= 8; }
        }

    private:
        std::vector<uint8_t>& out;
        uint32_t x1 = 0, x2 = 0xffffffffu;
    };

    class bit_decoder
    {
    public:
        bit_decoder (const uint8_t* begin_, const uint8_t* end_) : cur(begin_), end(end_)
        {
            for (int i = 0; i < 4; ++i) x = (x << 8) | next();
        }

        int decode (uint32_t p16)
        {
            const uint32_t xmid = x1 + static_cast<uint32_t>((static_cast<uint64_t>(x2 - x1) * p16) >> 16);
            const int bit = x <= xmid;
            if (bit) x2 = xmid; else x1 = xmid + 1;
            while (((x1 ^ x2) & 0xff000000u) == 0) {
                x1 <<= 8;
                x2 = (x2 << 8) | 255;
                x = (x << 8) | next();
            }
            return bit;
        }

    private:
        uint32_t next () { return cur < end ? *cur++ : 0u; }
        const uint8_t* cur;
        const uint8_t* end;
        uint32_t x1 = 0, x2 = 0xffffffffu, x = 0;
    };

    // ----------------------------------------------------------------------------------------
    // Adaptive components

    /*
        An adaptive probability in 32 bits: 22 bits of probability and a 10-bit count. The
        rate is 1 / (n + 1.5) while the count n is below its limit, then stays there, so a
        fresh context learns from its first observations and an old one tracks slow drift.
    */
    class counter_rates
    {
    public:
        counter_rates ()
        {
            for (int n = 0; n < 1024; ++n) rate[n] = static_cast<uint32_t>(131072 / (2 * n + 3));
        }
        uint32_t operator[] (uint32_t n) const { return rate[n]; }
        const uint32_t* data () const { return rate; }
    private:
        uint32_t rate[1024];
    };

    inline const counter_rates& rates ()
    {
        static const counter_rates r;
        return r;
    }

    const uint32_t COUNTER_INIT = 0x80000000u;

    inline int counter_p16 (uint32_t slot)
    {
        const int p = static_cast<int>(slot >> 16);
        return p < 1 ? 1 : (p > 65535 ? 65535 : p);
    }

    inline void counter_update (uint32_t& slot, int bit, uint32_t limit)
    {
        const uint32_t n = slot & 1023u;
        const int64_t p = slot >> 10;
        const int64_t target = bit ? (1 << 22) - 1 : 0;
        const int64_t np = p + (((target - p) * rates()[n]) >> 16);
        slot = (static_cast<uint32_t>(np) << 10) | (n < limit ? n + 1 : n);
    }

    /*
        Gated linear mixing in the logistic domain: a set of weights selected by a small
        context, an output squash(sum w_i * st_i), and weights moved along the gradient of the
        coding cost, err * st_i.
    */
    class mixer
    {
    public:
        void init (int inputs_, int sets, int rate_, int initial_weight)
        {
            inputs = inputs_;
            rate = rate_;
            weights.assign(static_cast<size_t>(inputs) * sets, initial_weight);
            x.assign(inputs, 0);
        }

        int* input () { return x.data(); }

        int mix (int set)
        {
            selected = &weights[static_cast<size_t>(set) * inputs];
            int64_t dot = 0;
            for (int i = 0; i < inputs; ++i) dot += static_cast<int64_t>(selected[i]) * x[i];
            int st = static_cast<int>(dot >> 16);
            st = st > ST_LIMIT ? ST_LIMIT : (st < -ST_LIMIT ? -ST_LIMIT : st);
            out_p16 = logistic_tables().squash(st);
            return st;
        }

        void update (int bit)
        {
            const int err = (bit ? 65535 : 0) - out_p16;
            for (int i = 0; i < inputs; ++i)
                selected[i] += static_cast<int32_t>((static_cast<int64_t>(x[i]) * err * rate) >> 20);
        }

        int p16 () const { return out_p16; }

    private:
        int inputs = 0, rate = 0;
        std::vector<int32_t> weights;
        std::vector<int> x;
        int32_t* selected = nullptr;
        int out_p16 = 32768;
    };

    /*
        Adaptive probability map: refines a probability given a context by interpolating
        between 33 cells spanning the logistic domain, and moves the two cells nearest the
        input toward each coded bit.
    */
    class apm
    {
    public:
        void init (int contexts, int rate_shift_)
        {
            rate_shift = rate_shift_;
            cells.resize(static_cast<size_t>(contexts) * 33);
            for (int c = 0; c < contexts; ++c)
                for (int j = 0; j < 33; ++j)
                    cells[static_cast<size_t>(c) * 33 + j] =
                        static_cast<uint16_t>(logistic_tables().squash((j - 16) * (ST_LIMIT + 1) / 16));
        }

        int refine (int st, int context)
        {
            const int scale = ST_LIMIT + 1;
            st = st > ST_LIMIT ? ST_LIMIT : (st < -ST_LIMIT ? -ST_LIMIT : st);
            const int pos = (st + scale) * 16;
            const int lo = pos / scale;
            weight_hi = pos % scale;
            index = static_cast<size_t>(context) * 33 + lo;
            const size_t hi = lo < 32 ? index + 1 : index;
            const int p = (cells[index] * (scale - weight_hi) + cells[hi] * weight_hi) / scale;
            return p < 1 ? 1 : (p > 65535 ? 65535 : p);
        }

        void update (int bit)
        {
            const int target = bit ? 65535 : 0;
            const int scale = ST_LIMIT + 1;
            int lo = cells[index];
            lo += ((target - lo) * (scale - weight_hi) / scale) >> rate_shift;
            cells[index] = static_cast<uint16_t>(lo);
            if (index % 33 < 32) {
                int hi = cells[index + 1];
                hi += ((target - hi) * weight_hi / scale) >> rate_shift;
                cells[index + 1] = static_cast<uint16_t>(hi);
            }
        }

    private:
        std::vector<uint16_t> cells;
        size_t index = 0;
        int weight_hi = 0;
        int rate_shift = 7;
    };

    // ----------------------------------------------------------------------------------------
    // Hashing and checksums

    inline uint64_t hash64 (uint64_t a, uint64_t b)
    {
        uint64_t h = a * 0x9E3779B97F4A7C15ULL + b;
        h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 32; h *= 0x94D049BB133111EBULL;
        h ^= h >> 29;
        return h;
    }

    // The CRC-32 of a block of bytes, by dlib::crc32.
    inline uint32_t checksum (const uint8_t* data, size_t size)
    {
        dlib::crc32 crc;
        for (size_t i = 0; i < size; ++i) crc.add(data[i]);
        return static_cast<uint32_t>(crc.get_checksum());
    }

    // ----------------------------------------------------------------------------------------
    // Folded capitals

    /*
        A reversible rewriting of the text in which ASCII capitals become lower case after a
        marker byte, so that a word and its capitalized form share their tokens. Marker 1 makes
        the next ASCII letter upper case, marker 2 the whole run of ASCII letters that follows,
        used for runs of two capitals or more. The marker of the first letter of a run that
        follows a space is written before the space, so that the run keeps the space-prefixed
        tokens of its lower-case form. Bytes 1, 2 and 3 of the text are written after the
        escape byte 3.
    */
    const uint8_t FOLD_ONE = 1, FOLD_RUN = 2, FOLD_ESCAPE = 3;

    inline bool ascii_upper (uint8_t b) { return b >= 'A' && b <= 'Z'; }
    inline bool ascii_lower (uint8_t b) { return b >= 'a' && b <= 'z'; }

    template <typename container>
    container fold_capitals (const uint8_t* text, size_t size)
    {
        typedef typename container::value_type value;
        container out;
        out.reserve(size + size / 16);
        size_t i = 0;
        while (i < size) {
            const uint8_t b = text[i];
            if (b >= FOLD_ONE && b <= FOLD_ESCAPE) {
                out.push_back(static_cast<value>(FOLD_ESCAPE));
                out.push_back(static_cast<value>(b));
                ++i;
                continue;
            }
            if (!ascii_upper(b) && !ascii_lower(b)) {
                out.push_back(static_cast<value>(b));
                ++i;
                continue;
            }
            size_t j = i;
            bool all_upper = true;
            while (j < size && (ascii_upper(text[j]) || ascii_lower(text[j])))
                all_upper = ascii_upper(text[j++]) && all_upper;
            // A space before the run is already written; the marker of the first letter goes before it.
            const bool after_space = i > 0 && text[i - 1] == ' ';
            auto mark = [&](uint8_t marker, bool first) {
                if (first && after_space) out.insert(out.end() - 1, static_cast<value>(marker));
                else out.push_back(static_cast<value>(marker));
            };
            if (all_upper && j - i >= 2) {
                mark(FOLD_RUN, true);
                for (size_t k = i; k < j; ++k) out.push_back(static_cast<value>(text[k] + 32));
            }
            else {
                for (size_t k = i; k < j; ++k) {
                    if (ascii_upper(text[k])) {
                        mark(FOLD_ONE, k == i);
                        out.push_back(static_cast<value>(text[k] + 32));
                    }
                    else {
                        out.push_back(static_cast<value>(text[k]));
                    }
                }
            }
            i = j;
        }
        return out;
    }

    inline std::vector<uint8_t> unfold_capitals (const uint8_t* text, size_t size)
    {
        std::vector<uint8_t> out;
        out.reserve(size);
        bool one = false, run_pending = false, run_active = false;
        for (size_t i = 0; i < size; ++i) {
            uint8_t b = text[i];
            if (run_active) {
                if (ascii_lower(b)) {
                    out.push_back(static_cast<uint8_t>(b - 32));
                    continue;
                }
                run_active = false;
            }
            if (b == FOLD_ESCAPE) {
                if (++i < size) out.push_back(text[i]);
                continue;
            }
            if (b == FOLD_ONE) { one = true; continue; }
            if (b == FOLD_RUN) { run_pending = true; continue; }
            if (ascii_lower(b) && one) {
                b = static_cast<uint8_t>(b - 32);
                one = false;
            }
            else if (ascii_lower(b) && run_pending) {
                b = static_cast<uint8_t>(b - 32);
                run_pending = false;
                run_active = true;
            }
            out.push_back(b);
        }
        return out;
    }

    // ----------------------------------------------------------------------------------------
    // Page order

    /*
        A dump of wiki pages in XML, each page from a line "  <page>" to a line "  </page>" with
        its number in its first <id> element, is coded with its pages in another order when
        those numbers increase strictly through the input: the decoder sorts the decoded pages
        by number, which restores the input with no record of the order. The pages are grouped
        by their first category, then by title, and a redirect follows the page it points to,
        so that the pages of a subject are coded side by side, where the adaptive models and the
        match model find what they share. The text before the first page, and what follows the
        last complete page, stay in place.
    */
    struct page_layout
    {
        size_t head = 0;                // where the first page starts
        size_t tail = 0;                // where what follows the last complete page starts
        std::vector<size_t> starts;     // where each complete page starts, then tail
    };

    inline bool ascii_space (uint8_t b) { return b == ' ' || (b >= 9 && b <= 13); }

    // The first position at or after from where line starts a line of the text, or size.
    inline size_t find_line (const uint8_t* text, size_t size, size_t from, const char* line)
    {
        const size_t n = std::strlen(line);
        for (size_t p = from; p + n <= size; ++p) {
            const void* hit = std::memchr(text + p, line[0], size - p);
            if (!hit) break;
            p = static_cast<size_t>(static_cast<const uint8_t*>(hit) - text);
            if (p + n > size) break;
            if ((p == 0 || text[p - 1] == '\n') && std::memcmp(text + p, line, n) == 0) return p;
        }
        return size;
    }

    // The first position of what in [from, to), or to.
    inline size_t find_in (const uint8_t* text, size_t from, size_t to, const char* what)
    {
        const size_t n = std::strlen(what);
        for (size_t p = from; p + n <= to; ++p) {
            const void* hit = std::memchr(text + p, what[0], to - p);
            if (!hit) break;
            p = static_cast<size_t>(static_cast<const uint8_t*>(hit) - text);
            if (p + n > to) break;
            if (std::memcmp(text + p, what, n) == 0) return p;
        }
        return to;
    }

    // Splits a text into pages. Returns false unless it holds two complete pages or more.
    inline bool find_pages (const uint8_t* text, size_t size, page_layout& layout)
    {
        static const char open[] = "  <page>\n", close[] = "  </page>\n";
        const size_t close_size = sizeof(close) - 1;
        auto closed = [&](size_t begin, size_t end) {
            return end - begin >= close_size && std::memcmp(text + end - close_size, close, close_size) == 0;
        };
        std::vector<size_t> starts;
        for (size_t p = find_line(text, size, 0, open); p < size; p = find_line(text, size, p + 1, open))
            starts.push_back(p);
        if (starts.empty()) return false;
        // Only the last page may be incomplete; it then stays where it is, with what follows it.
        if (closed(starts.back(), size)) starts.push_back(size);
        layout.head = starts.front();
        layout.tail = starts.back();
        layout.starts.swap(starts);
        for (size_t k = 0; k + 1 < layout.starts.size(); ++k)
            if (!closed(layout.starts[k], layout.starts[k + 1])) return false;
        return layout.starts.size() >= 3;
    }

    // The number in the first <id> element of the page [begin, end); false if it has none.
    inline bool page_number (const uint8_t* text, size_t begin, size_t end, uint64_t& number)
    {
        const size_t p = find_in(text, begin, end, "<id>");
        if (p == end) return false;
        size_t q = p + 4;
        number = 0;
        while (q < end && q < p + 4 + 18 && text[q] >= '0' && text[q] <= '9') number = number * 10 + (text[q++] - '0');
        return q > p + 4 && q + 5 <= end && std::memcmp(text + q, "</id>", 5) == 0;
    }

    /*
        The order in which the pages of a text are coded, as indices of its pages; empty when the
        numbers of the pages do not increase strictly. The pages that are not redirects come by
        their first category, those without one last, then by title, both compared in lower case;
        each is followed by the redirects to it, in their order in the text, and the redirects to
        pages that are missing or are redirects themselves close the order, by target.
    */
    inline std::vector<size_t> page_order (const uint8_t* text, const page_layout& layout)
    {
        const size_t n = layout.starts.size() - 1;
        std::vector<std::string> titles(n), categories(n), targets(n);
        std::vector<char> redirect(n, 0), has_category(n, 0);
        uint64_t previous = 0;
        for (size_t k = 0; k < n; ++k) {
            const size_t b = layout.starts[k], e = layout.starts[k + 1];
            uint64_t number = 0;
            if (!page_number(text, b, e, number) || (k > 0 && number <= previous)) return std::vector<size_t>();
            previous = number;
            const size_t t = find_in(text, b, e, "<title>");
            const size_t t_end = t < e ? find_in(text, t, e, "</title>") : e;
            if (t_end < e) titles[k].assign(text + t + 7, text + t_end);
            const size_t c = find_in(text, b, e, "[[Category:");
            if (c < e) {
                size_t q = c + 11;
                while (q < e && text[q] != ']' && text[q] != '|') ++q;
                categories[k].assign(text + c + 11, text + q);
                has_category[k] = 1;
            }
            // A redirect: a text opening with #REDIRECT, in any case, then a link.
            size_t q = find_in(text, b, e, "<text");
            while (q < e && text[q] != '>') ++q;
            static const char word[] = "#redirect";
            size_t w = 0;
            while (w < 9 && q + 1 + w < e) {
                const uint8_t ch = text[q + 1 + w];
                if ((ascii_upper(ch) ? ch + 32 : ch) != word[w]) break;
                ++w;
            }
            if (w < 9) continue;
            q += 10;
            while (q < e && ascii_space(text[q])) ++q;
            if (q + 2 > e || text[q] != '[' || text[q + 1] != '[') continue;
            size_t l = q + 2, r = l;
            while (r < e && text[r] != ']' && text[r] != '|' && text[r] != '#') ++r;
            while (l < r && ascii_space(text[l])) ++l;
            while (r > l && ascii_space(text[r - 1])) --r;
            targets[k].assign(text + l, text + r);
            redirect[k] = 1;
        }
        auto lower = [](std::string s) {
            for (char& ch : s) if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch + 32);
            return s;
        };
        std::vector<std::string> category_key(n), title_key(n), target_key(n);
        std::vector<size_t> articles, orphans;
        for (size_t k = 0; k < n; ++k) {
            category_key[k] = lower(categories[k]);
            title_key[k] = lower(titles[k]);
            target_key[k] = lower(targets[k]);
            if (!redirect[k]) articles.push_back(k);
        }
        std::stable_sort(articles.begin(), articles.end(), [&](size_t a, size_t b) {
            if (has_category[a] != has_category[b]) return has_category[a] > has_category[b];
            if (category_key[a] != category_key[b]) return category_key[a] < category_key[b];
            return title_key[a] < title_key[b];
        });
        std::map<std::string, size_t> by_title;
        for (size_t k = 0; k < n; ++k) by_title.insert(std::make_pair(titles[k], k));
        std::vector<std::vector<size_t>> followers(n);
        for (size_t k = 0; k < n; ++k) {
            if (!redirect[k]) continue;
            // The target as written, or with its first letter in upper case as titles have it.
            auto it = by_title.find(targets[k]);
            if (it == by_title.end() && !targets[k].empty() && ascii_lower(static_cast<uint8_t>(targets[k][0]))) {
                std::string t = targets[k];
                t[0] = static_cast<char>(t[0] - 32);
                it = by_title.find(t);
            }
            if (it != by_title.end() && !redirect[it->second]) followers[it->second].push_back(k);
            else orphans.push_back(k);
        }
        std::stable_sort(orphans.begin(), orphans.end(), [&](size_t a, size_t b) {
            return target_key[a] < target_key[b];
        });
        std::vector<size_t> order;
        order.reserve(n);
        for (size_t k : articles) {
            order.push_back(k);
            order.insert(order.end(), followers[k].begin(), followers[k].end());
        }
        order.insert(order.end(), orphans.begin(), orphans.end());
        return order;
    }

    // The text with its pages in the given order.
    inline std::vector<uint8_t> apply_page_order (const uint8_t* text, size_t size, const page_layout& layout,
        const std::vector<size_t>& order)
    {
        std::vector<uint8_t> out;
        out.reserve(size);
        out.insert(out.end(), text, text + layout.head);
        for (size_t k : order) out.insert(out.end(), text + layout.starts[k], text + layout.starts[k + 1]);
        out.insert(out.end(), text + layout.tail, text + size);
        return out;
    }

    // The inverse, for the decoder: the pages sorted by number.
    inline std::vector<uint8_t> restore_page_order (const std::vector<uint8_t>& text)
    {
        page_layout layout;
        if (!find_pages(text.data(), text.size(), layout)) throw std::runtime_error("the decoded text has no pages");
        const size_t n = layout.starts.size() - 1;
        std::vector<std::pair<uint64_t, size_t>> numbers(n);
        for (size_t k = 0; k < n; ++k) {
            if (!page_number(text.data(), layout.starts[k], layout.starts[k + 1], numbers[k].first))
                throw std::runtime_error("a decoded page has no number");
            numbers[k].second = k;
        }
        std::sort(numbers.begin(), numbers.end());
        std::vector<size_t> order(n);
        for (size_t k = 0; k < n; ++k) order[k] = numbers[k].second;
        return apply_page_order(text.data(), text.size(), layout, order);
    }

    /*
        Puts the pages of a wiki dump in the order they are coded in, when the text is one whose
        order the decoder restores. Returns the number of pages, or 0 when the text is left as it
        is. The inverse is checked on the result, so that no archive depends on it untested.
    */
    inline size_t order_pages (std::vector<uint8_t>& text)
    {
        page_layout layout;
        if (!find_pages(text.data(), text.size(), layout)) return 0;
        const std::vector<size_t> order = page_order(text.data(), layout);
        if (order.empty()) return 0;
        std::vector<uint8_t> moved = apply_page_order(text.data(), text.size(), layout, order);
        if (restore_page_order(moved) != text) return 0;
        text.swap(moved);
        return order.size();
    }

    // ----------------------------------------------------------------------------------------
    // Vocabulary

    /*
        The byte pattern of every token of a dlib::bpe_tokenizer, special tokens aside, and the
        tokens in lexicographic order of their patterns. In that order the tokens sharing a
        prefix form a contiguous range, the token equal to the prefix comes first, and each
        further bit of the next byte splits the range in two.
    */
    class vocabulary
    {
    public:
        void build (const dlib::bpe_tokenizer& tokenizer)
        {
            const int n = static_cast<int>(tokenizer.get_vocab_without_specials_size());
            offset.assign(n + 1, 0);
            bytes.clear();
            for (int t = 0; t < n; ++t) {
                offset[t] = static_cast<uint32_t>(bytes.size());
                const std::string pattern = tokenizer.decode(t, false);
                bytes.insert(bytes.end(), pattern.begin(), pattern.end());
            }
            offset[n] = static_cast<uint32_t>(bytes.size());
            order.resize(n);
            for (int t = 0; t < n; ++t) order[t] = t;
            std::sort(order.begin(), order.end(), [this](int a, int b) {
                return std::lexicographical_compare(pattern(a), pattern(a) + length(a),
                    pattern(b), pattern(b) + length(b));
            });
        }

        int size () const { return static_cast<int>(order.size()); }
        const uint8_t* pattern (int token) const { return bytes.data() + offset[token]; }
        int length (int token) const { return static_cast<int>(offset[token + 1] - offset[token]); }
        int token_at (int r) const { return order[r]; }

        // First rank in [lo, hi) whose byte at depth is >= value. Every token of the range
        // shares the prefix up to depth and is longer than depth.
        int lower_bound (int lo, int hi, int depth, int value) const
        {
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (pattern(order[mid])[depth] < value) lo = mid + 1; else hi = mid;
            }
            return lo;
        }

    private:
        std::vector<uint32_t> offset;
        std::vector<uint8_t> bytes;
        std::vector<int> order;
    };

    // ----------------------------------------------------------------------------------------
    // Bit histories

    /*
        A bit history is a state, at most 255 of them, standing for a count of zeros n0 and a
        count of ones n1, and for small totals the last bit seen. A new bit increments its
        count and discounts the other one beyond 2, so that a history follows a context whose
        statistics change. The states are enumerated at start-up from this rule, from the
        empty state 0.
    */
    class bit_states
    {
    public:
        bit_states ()
        {
            typedef std::tuple<int, int, int> history;    // n0, n1, last bit or 2
            std::map<history, int> index;
            std::vector<history> list;
            auto cap = [](int other) {
                static const int c[] = { 40, 24, 13, 9, 7, 6, 5 };
                return other < 7 ? c[other] : 5;
            };
            auto discount = [](int n) { return n > 2 ? n / 2 + 1 : n; };
            list.push_back(history(0, 0, 2));
            index[list[0]] = 0;
            for (size_t i = 0; i < list.size(); ++i) {
                for (int y = 0; y < 2; ++y) {
                    int n0 = std::get<0>(list[i]), n1 = std::get<1>(list[i]);
                    if (y) { n1 += 1; n0 = discount(n0); } else { n0 += 1; n1 = discount(n1); }
                    n0 = std::min(n0, cap(n1));
                    n1 = std::min(n1, cap(n0));
                    const history h(n0, n1, n0 > 0 && n1 > 0 && n0 + n1 <= 12 ? y : 2);
                    auto it = index.find(h);
                    if (it == index.end()) {
                        it = index.insert(std::make_pair(h, static_cast<int>(list.size()))).first;
                        list.push_back(h);
                    }
                    next_[i][y] = static_cast<uint8_t>(it->second);
                }
            }
            for (size_t i = 0; i < list.size(); ++i) {
                n0_[i] = static_cast<uint8_t>(std::get<0>(list[i]));
                n1_[i] = static_cast<uint8_t>(std::get<1>(list[i]));
            }
        }

        uint8_t next (uint8_t s, int bit) const { return next_[s][bit]; }
        const uint8_t* next_data () const { return &next_[0][0]; }
        int n0 (uint8_t s) const { return n0_[s]; }
        int n1 (uint8_t s) const { return n1_[s]; }
        // A history that has seen bits of one value only.
        bool deterministic (uint8_t s) const { return s != 0 && (n0_[s] == 0 || n1_[s] == 0); }

    private:
        uint8_t next_[256][2] = {};
        uint8_t n0_[256] = {}, n1_[256] = {};
    };

    inline const bit_states& histories ()
    {
        static const bit_states s;
        return s;
    }

    /*
        What a bit history predicts, learned for each context model and each kind of decision
        by an adaptive counter, which starts from the counts of the history.
    */
    class state_map
    {
    public:
        void init (int sets)
        {
            const bit_states& bs = histories();
            t.resize(static_cast<size_t>(sets) * 256);
            for (int c = 0; c < sets; ++c)
                for (int s = 0; s < 256; ++s) {
                    const uint32_t n0 = static_cast<uint32_t>(bs.n0(static_cast<uint8_t>(s)));
                    const uint32_t n1 = static_cast<uint32_t>(bs.n1(static_cast<uint8_t>(s)));
                    const uint32_t p = ((2 * n1 + 1) << 22) / (2 * (n0 + n1) + 2);
                    t[static_cast<size_t>(c) * 256 + s] = std::min(p, (1u << 22) - 1) << 10;
                }
        }

        int p (int set, uint8_t s)
        {
            index = static_cast<size_t>(set) * 256 + s;
            return counter_p16(t[index]);
        }

        void update (int bit) { counter_update(t[index], bit, 1023); }

    private:
        std::vector<uint32_t> t;
        size_t index = 0;
    };

    /*
        The bit histories of a context are kept per nibble: a slot holds the 15 histories of
        the decisions of one nibble behind an 8-bit check of the context's hash. Four slots
        share a 64-byte bucket, so that a nibble costs one cache line per context, and a
        context missing from its bucket takes the slot whose first history has seen the
        fewest bits.
    */
    struct history_slot
    {
        uint8_t check;
        uint8_t st[15];
    };

    class slot_table
    {
    public:
        void init (int bits)
        {
            bucket_bits = bits - 2;
            t.assign(size_t(4) << bucket_bits, history_slot());
            std::memset(t.data(), 0, t.size() * sizeof(history_slot));
        }

        // Brings the bucket of h toward the processor ahead of find.
        void prefetch (uint64_t h) const
        {
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(&t[(h >> (64 - bucket_bits)) << 2]);
#elif SLM_CODEC_X86
            _mm_prefetch(reinterpret_cast<const char*>(&t[(h >> (64 - bucket_bits)) << 2]), _MM_HINT_T0);
#else
            (void)h;
#endif
        }

        size_t bytes () const { return t.size() * sizeof(history_slot); }

        history_slot* find (uint64_t h)
        {
            history_slot* b = &t[(h >> (64 - bucket_bits)) << 2];
            const uint8_t check = static_cast<uint8_t>(h);
            for (int j = 0; j < 4; ++j) if (b[j].check == check) return &b[j];
            const bit_states& bs = histories();
            int victim = 0, fewest = 1 << 30;
            for (int j = 0; j < 4; ++j) {
                const int n = bs.n0(b[j].st[0]) + bs.n1(b[j].st[0]);
                if (n < fewest) { fewest = n; victim = j; }
            }
            std::memset(&b[victim], 0, sizeof(history_slot));
            b[victim].check = check;
            return &b[victim];
        }

    private:
        std::vector<history_slot> t;
        int bucket_bits = 0;
    };

    // ----------------------------------------------------------------------------------------
    // Processor features

    /*
        The kernels the processor runs, chosen once from the extensions Dlib detects: the
        integer products in AVX-512 VNNI, in AVX-VNNI or in AVX2, the floating-point kernels
        in AVX2 with FMA, and the portable code otherwise. Every path writes the same
        archives, so that DLIB_DISABLE_CPU_FEATURES can send a run through any of them.
    */
    struct processor_features
    {
        bool avx2_fma = false, avx512_vnni = false, avx_vnni = false;
    };

    inline processor_features detect_processor ()
    {
        processor_features f;
#if SLM_CODEC_X86
        const dlib::cpu_features& c = dlib::get_cpu_features();
        f.avx2_fma = c.avx2 && c.fma;
        // The dot products of AVX-512 on 256-bit vectors also need AVX-512VL.
        f.avx512_vnni = f.avx2_fma && c.avx512_vnni && c.avx512vl;
        f.avx_vnni = f.avx2_fma && c.avx_vnni;
#endif
        return f;
    }

    inline const processor_features& processor ()
    {
        static const processor_features f = detect_processor();
        return f;
    }

    // The instructions of the integer products, as the reports name them.
    inline const char* product_kernels ()
    {
        if (SLM_VNNI_KERNELS && processor().avx512_vnni) return "AVX-512 VNNI";
        if (SLM_VNNI_KERNELS && processor().avx_vnni) return "AVX-VNNI";
        if (processor().avx2_fma) return "AVX2";
        return "portable code";
    }

    // ----------------------------------------------------------------------------------------
    // Context mixing

    /*
        Predicts one binary decision at a time over byte streams. A decision is a bit of the
        next byte, identified by the partial byte c0 (a one followed by the bits already known),
        or, when the stream is coded token by token, whether the current token ends at the
        current byte, identified by 256 plus its depth.

        107 context models each hash a context of the bytes and tokens already coded and keep,
        per context, a bit history for each decision: the histories of the bits of a nibble in
        a slot of 16 bytes behind a check byte, those of the token ends in a table of their own,
        each behind a check byte. The contexts are orders one to thirty-two; sparse contexts;
        the last letters and the last bytes other than spaces; the classes of the last bytes;
        the current word with the previous ones in 26 combinations; the word that followed the
        previous words the last time; the line above, the opening of the line and the column;
        the state of the links, templates, tables, quotes and entities of the wiki text;
        numbers; the bytes that followed the last bytes the previous times; the previous tokens
        with the bytes of the current one and the network's most likely token; the bytes the
        matches predict; the current page.

        A map learned per model, per kind of decision and per bucket of the network's
        prediction turns a history into what it adds to the centre of the bucket. Each model
        gives two inputs to the mixers: that correction, and the same again when its history
        has only seen one value; none when its history has seen nothing. Three match models, the
        last 6, 12 and 24 bytes found again in any stream, give the bit their predicted byte has,
        and the network enters raw and through four maps that recalibrate it per kind of
        decision and per small context. Eight mixers select their weights by the kind of
        decision with the network's confidence, by the previous byte, by the match, by the
        number of contexts already seen, by what the network's most likely token costs, by the
        class of the last byte and the kind of line, by the length of the current word inside or
        outside links, and by what the network has paid for the current token so far; a final
        mixer combines theirs, and four adaptive probability maps refine the result.
    */

    /*
        Mixing over many inputs with 16-bit weights and 16-bit inputs, in vectors. A weight w
        stands for w / 2^14; the output in the logistic domain is the sum of the products, an
        integer that wraps modulo 2^32, shifted by 14 bits. An update adds to each weight its
        input times the error, rounded to the nearest unit as the instruction that multiplies in
        vectors rounds it, and saturates the result to 16 bits. Every operation is exact, so the
        portable loops and the AVX2 kernels give the same weights and the same outputs.
    */
    const int VMIX_SHIFT = 14;

#if SLM_CODEC_X86
    namespace mixing_avx2
    {
        SLM_AVX2 inline int32_t dot (const int16_t* x, const int16_t* w, int n)
        {
            __m256i acc = _mm256_setzero_si256();
            for (int i = 0; i < n; i += 16)
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)),
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i))));
            __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
            return _mm_cvtsi128_si32(s);
        }

        SLM_AVX2 inline void train (const int16_t* x, int16_t* w, int n, int e)
        {
            const __m256i ev = _mm256_set1_epi16(static_cast<int16_t>(e));
            for (int i = 0; i < n; i += 16) {
                __m256i* wp = reinterpret_cast<__m256i*>(w + i);
                const __m256i d = _mm256_mulhrs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)), ev);
                _mm256_storeu_si256(wp, _mm256_adds_epi16(_mm256_loadu_si256(wp), d));
            }
        }
    }
#endif

    namespace mixing
    {
        // The sum of x[i] * w[i] over n, a multiple of 16, modulo 2^32; vector selects the AVX2 kernel.
        inline int32_t dot (const int16_t* x, const int16_t* w, int n, bool vector)
        {
#if SLM_CODEC_X86
            if (vector) return mixing_avx2::dot(x, w, n);
#else
            (void)vector;
#endif
            uint32_t acc = 0;
            for (int i = 0; i < n; ++i) acc += static_cast<uint32_t>(static_cast<int32_t>(x[i]) * w[i]);
            return static_cast<int32_t>(acc);
        }

        // w[i] += round(x[i] * e / 2^15), saturated to 16 bits, for n a multiple of 16; |e| < 2^15.
        inline void train (const int16_t* x, int16_t* w, int n, int e, bool vector)
        {
#if SLM_CODEC_X86
            if (vector) { mixing_avx2::train(x, w, n, e); return; }
#else
            (void)vector;
#endif
            for (int i = 0; i < n; ++i) {
                const int v = w[i] + ((static_cast<int32_t>(x[i]) * e + 0x4000) >> 15);
                w[i] = static_cast<int16_t>(v < -32768 ? -32768 : (v > 32767 ? 32767 : v));
            }
        }
    }

    class vector_mixer
    {
    public:
        // inputs is rounded up to a multiple of 16; the inputs beyond those given must be zero.
        void init (int inputs_, int sets, int rate_, int initial_weight)
        {
            inputs = (inputs_ + 15) & ~15;
            rate = rate_;
            vector = processor().avx2_fma;
            weights.assign(static_cast<size_t>(inputs) * sets, 0);
            for (int s = 0; s < sets; ++s)
                for (int i = 0; i < inputs_; ++i)
                    weights[static_cast<size_t>(s) * inputs + i] = static_cast<int16_t>(initial_weight);
        }

        // Gives input i the weight w in every set.
        void set_weight (int i, int w)
        {
            for (size_t s = 0; s < weights.size(); s += inputs) weights[s + i] = static_cast<int16_t>(w);
        }

        int size () const { return inputs; }

        int mix (const int16_t* x_, int set)
        {
            x = x_;
            selected = &weights[static_cast<size_t>(set) * inputs];
            int st = mixing::dot(x, selected, inputs, vector) >> VMIX_SHIFT;
            st = st > ST_LIMIT ? ST_LIMIT : (st < -ST_LIMIT ? -ST_LIMIT : st);
            out_p16 = logistic_tables().squash(st);
            return st;
        }

        void update (int bit)
        {
            const int err = (bit ? 65535 : 0) - out_p16;
            int e = (err * rate) >> (21 - VMIX_SHIFT);
            e = e > 32767 ? 32767 : (e < -32767 ? -32767 : e);
            mixing::train(x, selected, inputs, e, vector);
        }

        int p16 () const { return out_p16; }

    private:
        int inputs = 0, rate = 0;
        bool vector = false;
        std::vector<int16_t> weights;
        const int16_t* x = nullptr;
        int16_t* selected = nullptr;
        int out_p16 = 32768;
    };

    // The final mixer: a few inputs, 32-bit weights scaled by 2^16, updates rounded to the nearest unit.
    class final_mixer
    {
    public:
        void init (int inputs_, int sets, int rate_, int initial_weight)
        {
            inputs = inputs_;
            rate = rate_;
            weights.assign(static_cast<size_t>(inputs) * sets, initial_weight);
        }

        int mix (const int* x_, int set)
        {
            x = x_;
            selected = &weights[static_cast<size_t>(set) * inputs];
            int64_t dot = 0;
            for (int i = 0; i < inputs; ++i) dot += static_cast<int64_t>(selected[i]) * x[i];
            int st = static_cast<int>(dot >> 16);
            st = st > ST_LIMIT ? ST_LIMIT : (st < -ST_LIMIT ? -ST_LIMIT : st);
            out_p16 = logistic_tables().squash(st);
            return st;
        }

        void update (int bit)
        {
            const int64_t err = static_cast<int64_t>((bit ? 65535 : 0) - out_p16) * rate;
            for (int i = 0; i < inputs; ++i) selected[i] += static_cast<int32_t>((x[i] * err + (1 << 19)) >> 20);
        }

        int p16 () const { return out_p16; }

    private:
        int inputs = 0, rate = 0;
        std::vector<int32_t> weights;
        const int* x = nullptr;
        int32_t* selected = nullptr;
        int out_p16 = 32768;
    };

    // Adaptive probabilities indexed directly, for the match models.
    class counter_table
    {
    public:
        void init (size_t n) { t.assign(n, COUNTER_INIT); }
        int p (size_t i) { index = i; return counter_p16(t[i]); }
        void update (int bit) { counter_update(t[index], bit, 1023); }
    private:
        std::vector<uint32_t> t;
        size_t index = 0;
    };

    /*
        The class of a byte: 0 letter, 1 digit, 2 space, 3 line break, 4 end of sentence, 5 other
        punctuation, 6 opening bracket, 7 closing bracket, 8 markup, 9 ampersand and semicolon,
        10 capital marker, 11 byte above 127, 12 double quote, 13 slash, 14 other.
    */
    inline int byte_class (uint8_t b)
    {
        static const struct table_t {
            uint8_t c[256];
            table_t ()
            {
                for (int i = 0; i < 256; ++i) c[i] = i >= 128 ? 11 : 14;
                for (int i = 'a'; i <= 'z'; ++i) c[i] = 0;
                for (int i = 'A'; i <= 'Z'; ++i) c[i] = 0;
                for (int i = '0'; i <= '9'; ++i) c[i] = 1;
                c[' '] = 2; c['\t'] = 2; c['\n'] = 3;
                c['.'] = 4; c['!'] = 4; c['?'] = 4;
                c[','] = 5; c[':'] = 5; c['-'] = 5;
                c['['] = 6; c['('] = 6; c['{'] = 6; c['<'] = 6;
                c[']'] = 7; c[')'] = 7; c['}'] = 7; c['>'] = 7;
                c['='] = 8; c['|'] = 8; c['*'] = 8; c['#'] = 8; c['\''] = 8; c['~'] = 8; c['_'] = 8;
                c['&'] = 9; c[';'] = 9;
                c[1] = 10; c[2] = 10; c[3] = 10;
                c['"'] = 12; c['/'] = 13;
            }
        } table;
        return table.c[b];
    }

    inline bool word_byte (uint8_t b) { return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b >= 128; }

    const int CM_MAX_CONTEXTS = 112;
    const int CM_MATCHES = 3;

    // What the predictor keeps of one stream.
    struct stream_state
    {
        std::vector<uint8_t> hist;      // bytes of the stream so far
        uint64_t l8 = 0, l16 = 0;       // the last eight bytes, the newest in the low byte, and the eight before
        uint64_t classes = 0;           // classes of the last sixteen bytes, four bits each
        uint64_t letters = 0;           // the last eight bytes of words
        uint64_t nonspace = 0;          // the last eight bytes other than spaces
        uint64_t ctx[CM_MAX_CONTEXTS] = {};
        // Words
        uint64_t w[5] = {};             // w[0] the word being written, w[1] to w[4] the previous ones
        int word_len = 0, words_in_sentence = 0;
        int caps = 0;                   // the capital marker just before the current word
        uint64_t sep = 0;               // the bytes since the previous word
        // Lines
        size_t line_start = 0, prev_line_start = 0, prev2_line_start = 0;
        uint64_t line_tag = 0, line_head = 0;   // the opening of the line, up to a space or '>'; its first bytes
        int tag_len = 0;
        bool tag_open = true;
        int line_kind = 0;              // 0 text, 1 heading, 2 list, 3 table, 4 element, 5 template, 6 link
        // Markup
        int link_depth = 0, link_part = 0;
        uint64_t link_name = 0;
        int tmpl_depth = 0, tmpl_param = 0;
        uint64_t tmpl_name = 0;
        bool tmpl_naming = false;
        int cell = 0, quote_run = 0, quote_state = 0;
        bool in_entity = false, in_ref = false, in_xml_tag = false;
        uint64_t entity = 0, xml_tag = 0;
        // Numbers
        int num_len = 0;
        uint64_t num_hash = 0, prev_num = 0;
        // Page
        uint64_t article = 0;
        size_t article_start = 0;
        // Tokens and the network
        int tk[3] = { -1, -1, -1 };     // the last tokens, tk[0] the last one
        uint64_t token_hash = 0;        // the last token and the bytes of the current one
        int depth = 0;                  // bytes of the current token already coded
        int top_token = -1, top_cost = 31;  // the network's most likely token and what it costs, in quarter bits
        int token_cost = 0;             // what the network paid for the current token so far, in quarter bits
        // Match models
        uint64_t match_ptr[CM_MATCHES] = {};    // stream << 40 | position of the predicted byte
        int match_len[CM_MATCHES] = {};
        int match_byte[CM_MATCHES] = { -1, -1, -1 };
        int match_miss[CM_MATCHES] = {};
    };

    class predictor
    {
    public:
        /*
            table_bits sizes the tables: the slots of a context model of the largest class take
            2^table_bits units of four bytes, those of the others down to 4096 times fewer, and the
            histories of token ends a quarter of the memory of their model's slots.
        */
        void init (int table_bits, std::vector<stream_state>* streams_)
        {
            streams = streams_;
            follow1.assign(256, 0);
            follow2.assign(65536, 0);
            next_word.assign(size_t(1) << 22, 0);
            next_word2.assign(size_t(1) << 22, 0);
            // A dry run of the list of contexts gives their number and their sizes.
            stream_state dummy;
            num_ctx = fill_contexts(dummy, 0, true);
            const int bits = table_bits - 2;
            const bit_states& bs = histories();
            for (int s = 0; s < 256; ++s) deterministic[s] = bs.deterministic(static_cast<uint8_t>(s)) ? 1 : 0;
            // What each history predicts starts, in every bucket of the network, at the bucket's centre.
            maps.resize(static_cast<size_t>(num_ctx) * MAP_SETS * 256);
            for (size_t k = 0; k < maps.size(); ++k)
                maps[k] = static_cast<uint32_t>(
                    logistic_tables().squash(bucket_centre(static_cast<int>((k >> 8) % NN_BUCKETS)))) << 16;
            for (int i = 0; i < num_ctx; ++i) {
                tables[i].init(std::max(14, bits + size_class[i]));
                end_tables[i].assign(size_t(2) << std::max(9, bits + 1 + std::min(0, size_class[i])), 0);
            }
            static const int match_bits[CM_MATCHES] = { 0, -2, -4 };
            for (int m = 0; m < CM_MATCHES; ++m) {
                const int b = std::max(16, bits + match_bits[m]);
                match_table[m].assign(size_t(1) << b, 0);
                match_shift[m] = 64 - b;
                match_map[m].init(64 * 2);
            }
            nn_map.init(24, 7);
            nn_map2.init(1 << 16, 6);
            nn_map3.init(1 << 16, 6);
            nn_map4.init(1 << 16, 6);
            inputs = 2 * num_ctx + 2 * CM_MATCHES + 7;
            x.assign((inputs + 15) & ~15, 0);
            const int w0 = static_cast<int>(4.0 * (1 << VMIX_SHIFT) / inputs);
            mixers[0].init(inputs, 12 * 8, 8, w0);
            mixers[1].init(inputs, 512, 8, w0);
            mixers[2].init(inputs, 12 * 4 * 3, 8, w0);
            mixers[3].init(inputs, 12 * 16, 8, w0);
            mixers[4].init(inputs, 12 * 32, 8, w0);
            mixers[5].init(inputs, 12 * 15 * 8, 8, w0);
            mixers[6].init(inputs, 12 * 8 * 2, 8, w0);
            mixers[7].init(inputs, 12 * 16 * 2, 8, w0);
            final.init(NUM_MIXERS + 1, 12 * 4, 8, 65536 / NUM_MIXERS);
            a1.init(1 << 17, 6);
            a2.init(1 << 16, 6);
            a3.init(12 * 64 * 3, 6);
            a4.init(1 << 16, 6);
            cached_stream = -1;
            static const struct cost_table {
                uint8_t c[2 * ST_LIMIT + 1];
                cost_table ()
                {
                    // 4 log2(1 + e^(-st / 256)), the cost in quarter bits of a bit of probability squash(st).
                    for (int st = -ST_LIMIT; st <= ST_LIMIT; ++st) {
                        const double q = 4.0 * det::log(1.0 + det::exp(-st / 256.0)) / 0.69314718055994530942;
                        c[st + ST_LIMIT] = static_cast<uint8_t>(std::min(255.0, det::round_even(q)));
                    }
                }
            } costs;
            quarter_bits = costs.c;
        }

        // Bytes held by the tables of the context models and of the match models.
        size_t memory () const
        {
            size_t m = 0;
            for (int i = 0; i < num_ctx; ++i) m += tables[i].bytes() + end_tables[i].size();
            for (int k = 0; k < CM_MATCHES; ++k) m += match_table[k].size() * sizeof(uint64_t);
            return m;
        }

        int contexts () const { return num_ctx; }

        // The network's most likely token for stream s and what it costs, in quarter bits up to 31.
        void network_top (int s, int token, int quarter_bits_)
        {
            stream_state& st = (*streams)[s];
            st.top_token = token;
            st.top_cost = quarter_bits_;
            // The contexts holding the network's choice take the choice for the token about to be coded.
            for (int i : top_ctx)
                st.ctx[i] = hash64(st.ctx[i] ^ 0x99,
                    static_cast<uint64_t>(token + 2) | (static_cast<uint64_t>(quarter_bits_) << 16));
            cached_stream = -1;
        }

        // Returns P(bit == 1) as a 16-bit probability. nn_valid says whether nn_st, the
        // network's prediction in the logistic domain, exists.
        int predict (int s, int code, int nn_st, bool nn_valid)
        {
            stream_state& st = (*streams)[s];
            const logistic& lg = logistic_tables();
            const bool end_decision = code >= 256;
            const int kind = end_decision ? 8 + std::min(3, code - 256) : bit_position(code);
            // The inputs of the mixers: two per model, then the matches, then the network.
            int16_t* in = x.data();
            const int16_t* stretch = lg.stretch_data();
            const int bucket = nn_valid ? nn_bucket(nn_st) : NN_BUCKETS - 1;
            const int centre = bucket_centre(bucket);
            uint32_t* map = maps.data() + (kind * NN_BUCKETS + bucket) * 256;
            /*
                Two inputs per model: what its history predicts beyond the centre of the network's
                bucket, and the same again when the history has seen one value only; none for a
                history that has seen nothing.
            */
            auto put_inputs = [&](int i, uint8_t* h) {
                const uint8_t state = *h;
                history[i] = h;
                if (state == 0) {
                    // What an empty history would predict is never read: its update goes to a spare cell.
                    cells[i] = &spare_cell;
                    in[2 * i] = 0;
                    in[2 * i + 1] = 0;
                    return;
                }
                uint32_t* c = map + static_cast<size_t>(i) * MAP_SETS * 256 + state;
                cells[i] = c;
                int d = stretch[counter_p16(*c)] - centre;
                d = d > 4095 ? 4095 : (d < -4095 ? -4095 : d);
                in[2 * i] = static_cast<int16_t>(d);
                in[2 * i + 1] = static_cast<int16_t>(deterministic[state] ? d : 0);
            };
            if (end_decision) {
                // Token ends have histories of their own, per context and depth, behind a check byte.
                const int depth = code - 256;
                for (int i = 0; i < num_ctx; ++i) {
                    std::vector<uint8_t>& et = end_tables[i];
                    const uint64_t h = hash64(st.ctx[i], static_cast<uint64_t>(depth));
                    const size_t e = (h & ((et.size() >> 1) - 1)) << 1;
                    const uint8_t check = static_cast<uint8_t>((h >> 56) | 1);
                    if (et[e] != check) { et[e] = check; et[e + 1] = 0; }
                    put_inputs(i, &et[e + 1]);
                }
            }
            else {
                // The slots of the current nibble, looked up at its first decision.
                const int nibble = kind >= 4 ? 1 : 0;
                const uint64_t key = (static_cast<uint64_t>(st.hist.size()) << 1) | static_cast<uint64_t>(nibble);
                if (cached_stream != s || cached_key != key) {
                    cached_stream = s;
                    cached_key = key;
                    // In the second nibble, the context includes the first.
                    const uint64_t high = nibble ? static_cast<uint64_t>(code >> (kind - 4)) : 0;
                    for (int i = 0; i < num_ctx; ++i)
                        tables[i].prefetch(nibble ? hash64(st.ctx[i], high) : st.ctx[i]);
                    for (int i = 0; i < num_ctx; ++i)
                        slots[i] = tables[i].find(nibble ? hash64(st.ctx[i], high) : st.ctx[i]);
                }
                const int k = nibble ? kind - 4 : kind;
                const int node = (1 << k) - 1 + (code & ((1 << k) - 1));
                for (int i = 0; i < num_ctx; ++i) put_inputs(i, &slots[i]->st[node]);
            }

            // The match models: the bit the predicted byte has here, by length and earlier misses.
            int16_t* xm = in + 2 * num_ctx;
            int expected0 = -1, length0 = 0, match_state = 0;
            for (int m = 0; m < CM_MATCHES; ++m) {
                match_used[m] = -1;
                int expected = -1;
                if (st.match_byte[m] >= 0 && !end_decision && ((st.match_byte[m] + 256) >> (8 - kind)) == code)
                    expected = (st.match_byte[m] >> (7 - kind)) & 1;
                const int len = st.match_len[m];
                const int lb = len < 16 ? len : (len < 32 ? 16 + (len - 16) / 4
                    : (len < 64 ? 20 + (len - 32) / 8 : 24 + std::min(7, (len - 64) / 32)));
                if (m == 0) {
                    expected0 = expected;
                    length0 = lb;
                    match_state = len == 0 ? 0 : (len < 16 ? 1 : (len < 32 ? 2 : 3));
                }
                if (expected >= 0) {
                    const int idx = (std::min(lb, 31) * 2 + std::min(st.match_miss[m], 1)) * 2 + expected;
                    match_used[m] = idx;
                    xm[2 * m] = static_cast<int16_t>(lg.stretch(match_map[m].p(static_cast<size_t>(idx))));
                    xm[2 * m + 1] = static_cast<int16_t>((expected ? 1 : -1) * std::min(lb, 31) * 64);
                }
                else {
                    xm[2 * m] = 0;
                    xm[2 * m + 1] = 0;
                }
            }

            // The network, raw and through maps keyed by small contexts.
            nn_active = nn_valid;
            nn_stream = s;
            nn_value = nn_st;
            int16_t* xn = xm + 2 * CM_MATCHES;
            xn[0] = static_cast<int16_t>(nn_valid ? nn_st : 0);
            xn[1] = static_cast<int16_t>(nn_valid ? lg.stretch(nn_map.refine(nn_st, kind + 12 * (st.depth > 0))) : 0);
            const int c2 = static_cast<int>(hash64(st.l8 & 0xffff, static_cast<uint64_t>(kind)) >> 48);
            xn[2] = static_cast<int16_t>(nn_valid ? lg.stretch(nn_map2.refine(nn_st, c2)) : 0);
            const int c3 = static_cast<int>(hash64(st.w[0] ^ (st.w[1] << 1), static_cast<uint64_t>(kind)) >> 48);
            xn[3] = static_cast<int16_t>(nn_valid ? lg.stretch(nn_map3.refine(nn_st, c3)) : 0);
            const int c4 = static_cast<int>(hash64(static_cast<uint64_t>(st.top_cost) * 16 + kind,
                static_cast<uint64_t>(st.line_kind * 16 + byte_class(static_cast<uint8_t>(st.l8)))) >> 48);
            xn[4] = static_cast<int16_t>(nn_valid ? lg.stretch(nn_map4.refine(nn_st, c4)) : 0);
            xn[5] = static_cast<int16_t>(nn_valid ? (st.top_cost < 4 ? nn_st : nn_st / 2) : 0);
            xn[6] = 256;

            const int confidence = nn_valid ? std::min(7, 1 + std::abs(nn_st) / 384) : 0;
            const int16_t* xv = x.data();
            const int last = static_cast<int>(st.l8 & 0xff);
            int seen = 0;
            for (int i = 1; i <= 15 && i < num_ctx; ++i) seen += *history[i] != 0;
            int* xf = final_inputs;
            xf[0] = mixers[0].mix(xv, kind * 8 + confidence);
            xf[1] = mixers[1].mix(xv, end_decision ? 256 + std::min(255, code - 256) : last);
            xf[2] = mixers[2].mix(xv, (kind * 4 + match_state) * 3 + (expected0 + 1));
            xf[3] = mixers[3].mix(xv, kind * 16 + std::min(15, seen));
            xf[4] = mixers[4].mix(xv, kind * 32 + std::min(15, st.top_cost / 2) * 2 + (*history[word_ctx] != 0));
            xf[5] = mixers[5].mix(xv, (kind * 15 + byte_class(static_cast<uint8_t>(last))) * 8
                + std::min(7, st.line_kind));
            xf[6] = mixers[6].mix(xv, (kind * 8 + std::min(7, st.word_len)) * 2
                + (st.link_depth > 0 || st.tmpl_depth > 0));
            xf[7] = mixers[7].mix(xv, (kind * 16 + std::min(15, st.token_cost / 4)) * 2 + (confidence > 3));
            xf[NUM_MIXERS] = 256;
            const int smix = final.mix(xf, kind * 4 + match_state);

            const int p1 = a1.refine(smix, (last << 9) | std::min(code, 511));
            const int p2 = a2.refine(smix, static_cast<int>(hash64(st.l8 & 0xffff, static_cast<uint64_t>(code)) >> 48));
            const int p3 = a3.refine(smix, (kind * 64 + std::min(length0, 63)) * 3 + (expected0 + 1));
            const int p4 = a4.refine(smix,
                static_cast<int>(hash64(st.w[0] + st.w[1] * 3, static_cast<uint64_t>(code)) >> 48));
            const int p = (2 * final.p16() + p1 + 2 * p2 + 2 * p3 + p4) >> 3;
            return p < 1 ? 1 : (p > 65535 ? 65535 : p);
        }

        void update (int bit)
        {
            // The maps move toward the bit as counter_update does, the histories take it.
            const uint32_t* rate = rates().data();
            const uint8_t* next = histories().next_data() + bit;
            const int64_t target = bit ? (1 << 22) - 1 : 0;
            for (int i = 0; i < num_ctx; ++i) {
                uint32_t& c = *cells[i];
                const uint32_t n = c & 1023u;
                const int64_t p = c >> 10;
                c = (static_cast<uint32_t>(p + (((target - p) * rate[n]) >> 16)) << 10) | (n < 1023 ? n + 1 : n);
                *history[i] = next[2 * *history[i]];
            }
            for (int m = 0; m < CM_MATCHES; ++m)
                if (match_used[m] >= 0) {
                    match_map[m].p(static_cast<size_t>(match_used[m]));
                    match_map[m].update(bit);
                }
            if (nn_active) {
                nn_map.update(bit);
                nn_map2.update(bit);
                nn_map3.update(bit);
                nn_map4.update(bit);
                int& cost = (*streams)[nn_stream].token_cost;
                cost = std::min(255, cost + quarter_bits[(bit ? nn_value : -nn_value) + ST_LIMIT]);
            }
            for (int i = 0; i < NUM_MIXERS; ++i) mixers[i].update(bit);
            final.update(bit);
            a1.update(bit);
            a2.update(bit);
            a3.update(bit);
            a4.update(bit);
        }

        // A byte of stream s is complete: append it and recompute the contexts.
        void byte_done (int s, uint8_t byte, bool)
        {
            stream_state& st = (*streams)[s];
            st.hist.push_back(byte);
            const uint8_t prev = static_cast<uint8_t>(st.l8 & 0xff);
            const uint8_t prev2 = static_cast<uint8_t>((st.l8 >> 8) & 0xff);
            st.l16 = (st.l16 << 8) | (st.l8 >> 56);
            st.l8 = (st.l8 << 8) | byte;
            st.classes = (st.classes << 4) | static_cast<uint64_t>(byte_class(byte));
            if (byte != ' ') st.nonspace = (st.nonspace << 8) | byte;
            ++st.depth;
            const size_t n = st.hist.size();
            // Indirect tables, shared by the streams: the two bytes that followed each byte and each pair.
            follow1[prev] = static_cast<uint16_t>((follow1[prev] << 8) | byte);
            uint16_t& f2 = follow2[static_cast<size_t>(prev2) << 8 | prev];
            f2 = static_cast<uint16_t>((f2 << 8) | byte);
            track_words(st, byte, prev);
            track_lines(st, byte, n);
            track_markup(st, byte, prev);
            track_numbers(st, byte);
            for (int m = 0; m < CM_MATCHES; ++m) update_match(s, st, byte, m);
            fill_contexts(st, byte, false);
            cached_stream = -1;
        }

        // A token of stream s is complete.
        void token_done (int s, int token)
        {
            stream_state& st = (*streams)[s];
            st.depth = 0;
            st.tk[2] = st.tk[1];
            st.tk[1] = st.tk[0];
            st.tk[0] = token;
            st.token_hash = hash64(0x5bd1e995ULL, static_cast<uint64_t>(token));
            st.token_cost = 0;
            // The contexts holding the current token start the next one.
            for (int i : token_ctx) st.ctx[i] = hash64(st.ctx[i] ^ 0x77, static_cast<uint64_t>(token) + 1);
            cached_stream = -1;
        }

    private:
        static const int NUM_MIXERS = 8;
        // The network's prediction in twelve buckets, and a thirteenth when there is none.
        static const int NN_BUCKETS = 13;
        static const int MAP_SETS = 12 * NN_BUCKETS;

        static int nn_bucket (int st)
        {
            static const int edges[11] = { -1024, -640, -384, -192, -64, 0, 64, 192, 384, 640, 1024 };
            int b = 0;
            while (b < 11 && st >= edges[b]) ++b;
            return b;
        }

        static int bucket_centre (int b)
        {
            static const int centres[NN_BUCKETS] =
                { -1400, -832, -512, -288, -128, -32, 32, 128, 288, 512, 832, 1400, 0 };
            return centres[b];
        }

        static int bit_position (int c0)
        {
            int p = 0;
            while (c0 > 1) { c0 >>= 1; ++p; }
            return p;
        }

        // The current word, the capital marker before it, and the previous words.
        static void track_words (stream_state& st, uint8_t byte, uint8_t prev)
        {
            if (word_byte(byte)) {
                if (st.word_len == 0) st.caps = prev == FOLD_ONE ? 1 : (prev == FOLD_RUN ? 2 : 0);
                st.w[0] = (st.w[0] + byte + 1) * 0x2F0F3A1BDULL;
                ++st.word_len;
                st.letters = (st.letters << 8) | byte;
                return;
            }
            if (st.word_len > 0) {
                for (int k = 4; k >= 1; --k) st.w[k] = st.w[k - 1];
                st.w[0] = 0;
                st.word_len = 0;
                st.sep = 0;
                ++st.words_in_sentence;
            }
            if (byte != FOLD_ONE && byte != FOLD_RUN && byte != FOLD_ESCAPE) st.sep = (st.sep << 8) | byte;
            if (byte == '.' || byte == '!' || byte == '?' || byte == '\n') st.words_in_sentence = 0;
        }

        // Line starts, the opening of the line and its kind, and the title line of the current page.
        static void track_lines (stream_state& st, uint8_t byte, size_t n)
        {
            if (byte == '\n') {
                size_t i = st.line_start;
                while (i < n && st.hist[i] == ' ') ++i;
                if (i + 7 < n && std::memcmp(&st.hist[i], "<title>", 7) == 0) {
                    uint64_t h = 0x7171;
                    for (size_t j = i; j < n; ++j) h = hash64(h, st.hist[j]);
                    st.article = h;
                    st.article_start = n;
                }
                st.prev2_line_start = st.prev_line_start;
                st.prev_line_start = st.line_start;
                st.line_start = n;
                st.line_tag = 0;
                st.line_head = 0;
                st.tag_len = 0;
                st.tag_open = true;
                st.line_kind = 0;
                st.cell = 0;
                st.link_depth = 0;
                st.tmpl_naming = false;
                st.quote_state = 0;
                return;
            }
            const size_t col = n - st.line_start;
            if (col <= 8) st.line_head = (st.line_head << 8) | byte;
            if (col == 1) {
                st.line_kind = byte == '=' ? 1 : (byte == '*' || byte == '#' || byte == ':' ? 2
                    : (byte == '|' || byte == '!' ? 3 : (byte == '<' || byte == ' ' ? 4
                    : (byte == '{' ? 5 : (byte == '[' ? 6 : 0)))));
            }
            if (st.tag_open && !(byte == ' ' && st.tag_len == 0)) {
                st.line_tag = hash64(st.line_tag + 1, byte);
                ++st.tag_len;
                if (byte == '>' || byte == ' ' || st.tag_len >= 12) st.tag_open = false;
            }
        }

        // Links, templates, table cells, bold and italics, character entities and the tags they write.
        static void track_markup (stream_state& st, uint8_t byte, uint8_t prev)
        {
            if (byte == '[' && prev == '[') { ++st.link_depth; st.link_part = 0; st.link_name = 0; }
            else if (byte == ']' && prev == ']') { if (st.link_depth > 0) --st.link_depth; }
            else if (st.link_depth > 0) {
                if (byte == '|') st.link_part = 1;
                else if (st.link_part == 0) st.link_name = hash64(st.link_name + 7, byte);
            }
            if (byte == '{' && prev == '{') {
                ++st.tmpl_depth;
                st.tmpl_param = 0;
                st.tmpl_name = 0;
                st.tmpl_naming = true;
            }
            else if (byte == '}' && prev == '}') { if (st.tmpl_depth > 0) --st.tmpl_depth; st.tmpl_naming = false; }
            else if (st.tmpl_depth > 0) {
                if (byte == '|') { ++st.tmpl_param; st.tmpl_naming = false; }
                else if (st.tmpl_naming && byte != '\n') st.tmpl_name = hash64(st.tmpl_name + 5, byte);
            }
            if (byte == '|' && prev == '|') ++st.cell;
            if (byte == '\'') ++st.quote_run;
            else {
                if (st.quote_run >= 2) st.quote_state ^= st.quote_run == 2 ? 1 : 2;
                st.quote_run = 0;
            }
            static const uint64_t lt = hash64(hash64(0x51, 'l'), 't'), gt = hash64(hash64(0x51, 'g'), 't');
            if (byte == '&') { st.in_entity = true; st.entity = 0; }
            else if (st.in_entity) {
                if (byte == ';' || !(word_byte(byte) || (byte >= '0' && byte <= '9') || byte == '#')) {
                    st.in_entity = false;
                    // &lt; opens a tag of the wiki text; &gt; closes it.
                    if (st.entity == lt) { st.in_xml_tag = true; st.xml_tag = 0; }
                    else if (st.entity == gt) st.in_xml_tag = false;
                }
                else st.entity = hash64(st.entity == 0 ? 0x51 : st.entity, byte);
            }
            else if (st.in_xml_tag && byte != ' ' && st.xml_tag != 1) {
                if (word_byte(byte) || byte == '/') st.xml_tag = hash64(st.xml_tag + 3, byte);
                else st.xml_tag = 1;
            }
            static const uint64_t ref_open = hash64(hash64(hash64(3, 'r') + 3, 'e') + 3, 'f');
            static const uint64_t ref_close = hash64(hash64(hash64(hash64(3, '/') + 3, 'r') + 3, 'e') + 3, 'f');
            if (st.in_xml_tag && st.xml_tag == ref_open) st.in_ref = true;
            if (st.in_xml_tag && st.xml_tag == ref_close) st.in_ref = false;
        }

        static void track_numbers (stream_state& st, uint8_t byte)
        {
            if (byte >= '0' && byte <= '9') {
                st.num_hash = hash64(st.num_hash + 11, byte);
                ++st.num_len;
            }
            else if (st.num_len > 0) {
                st.prev_num = st.num_hash;
                st.num_hash = 0;
                st.num_len = 0;
            }
        }

        /*
            Match model m: the last occurrence, in any stream, of the last 6, 12 or 24 bytes, found
            through a hash table and checked backwards, so that a collision yields a short length.
        */
        void update_match (int s, stream_state& st, uint8_t byte, int m)
        {
            static const int min_len[CM_MATCHES] = { 6, 12, 24 };
            const int ml = min_len[m];
            const size_t n = st.hist.size();
            if (st.match_byte[m] >= 0) {
                if (st.match_byte[m] == byte) { ++st.match_len[m]; ++st.match_ptr[m]; }
                else { st.match_len[m] = 0; st.match_miss[m] = 1; }
            }
            if (n >= static_cast<size_t>(ml)) {
                uint64_t key = 0x5A5A + m;
                if (ml <= 8) key = hash64(key, st.l8 & ((1ULL << (8 * ml)) - 1));
                else if (ml <= 16) key = hash64(hash64(key, st.l8), st.l16 & ((1ULL << (8 * (ml - 8))) - 1));
                else {
                    key = hash64(hash64(key, st.l8), st.l16);
                    for (int j = 16; j < ml; ++j) key = hash64(key, st.hist[n - 1 - j]);
                }
                uint64_t& entry = match_table[m][key >> match_shift[m]];
                if (st.match_len[m] == 0 && entry != 0) {
                    const stream_state& src = (*streams)[static_cast<size_t>(entry >> 40)];
                    const uint64_t pos = entry & ((uint64_t(1) << 40) - 1);
                    int len = 0;
                    while (len < 400 && static_cast<uint64_t>(len) < pos && static_cast<size_t>(len) < n
                        && src.hist[pos - 1 - len] == st.hist[n - 1 - len]) ++len;
                    if (len >= ml) { st.match_ptr[m] = entry; st.match_len[m] = len; st.match_miss[m] = 0; }
                }
                entry = (static_cast<uint64_t>(s) << 40) | n;
            }
            st.match_byte[m] = -1;
            if (st.match_len[m] > 0) {
                const stream_state& src = (*streams)[static_cast<size_t>(st.match_ptr[m] >> 40)];
                const uint64_t pos = st.match_ptr[m] & ((uint64_t(1) << 40) - 1);
                if (pos < src.hist.size()) st.match_byte[m] = src.hist[pos];
                else st.match_len[m] = 0;
            }
        }

        /*
            Computes the contexts of stream st after byte. With dry, records instead the size class
            of each context. Returns the number of contexts.
        */
        int fill_contexts (stream_state& st, uint8_t byte, bool dry)
        {
            int i = 0;
            token_ctx.clear();
            top_ctx.clear();
            const uint64_t l8 = st.l8, l16 = st.l16;
            const size_t n = st.hist.size();
            const size_t col = n - st.line_start;
            const size_t above_pos = st.prev_line_start + col;
            const uint64_t above = (st.line_start > 0 && above_pos < st.line_start) ? st.hist[above_pos] : 0;
            const size_t above2_pos = st.prev2_line_start + col;
            const uint64_t above2 = (st.prev_line_start > 0 && above2_pos < st.prev_line_start)
                ? st.hist[above2_pos] : 0;
            const uint64_t c1 = static_cast<uint64_t>(byte_class(byte));
            const bool in_word = st.word_len > 0;
            const uint64_t W0 = in_word ? st.w[0] : hash64(0x99, st.sep & 0xffff);
            const uint64_t markup = static_cast<uint64_t>(std::min(st.link_depth, 3) | (st.link_part << 2)
                | (std::min(st.tmpl_depth, 3) << 3) | (st.in_xml_tag << 5) | (st.in_entity << 6) | (st.quote_state << 7)
                | (st.line_kind << 9));
            // A context: its size class, relative to the largest tables, and its value.
            auto put = [&](int size, uint64_t v) {
                if (dry) size_class[i] = size;
                else st.ctx[i] = hash64(static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL + 0x1234567ULL, v);
                ++i;
            };
            // A context holding the current token, which changes when the token is complete.
            auto put_token = [&](int size, uint64_t v) { token_ctx.push_back(i); put(size, v); };
            // A context holding the network's choice, which changes when the network makes it.
            auto put_top = [&](int size, uint64_t v) { top_ctx.push_back(i); put(size, v); };

            // Orders 1 to 8, 10, 12, 14, 16, 20, 24 and 32.
            put(-12, l8 & 0xff);
            put(-6, l8 & 0xffff);
            put(-2, l8 & 0xffffff);
            put(0, l8 & 0xffffffffULL);
            put(0, l8 & 0xffffffffffULL);
            put(0, l8 & 0xffffffffffffULL);
            put(0, l8 & 0xffffffffffffffULL);
            put(0, l8);
            put(0, hash64(l8, l16 & 0xffff));
            put(0, hash64(l8, l16 & 0xffffffffULL));
            put(-1, hash64(l8, l16 & 0xffffffffffffULL));
            put(-1, hash64(l8, l16));
            uint64_t h20 = hash64(l8, l16), h24, h32;
            for (int j = 16; j < 20 && static_cast<size_t>(j) < n; ++j) h20 = hash64(h20, st.hist[n - 1 - j]);
            h24 = h20;
            for (int j = 20; j < 24 && static_cast<size_t>(j) < n; ++j) h24 = hash64(h24, st.hist[n - 1 - j]);
            h32 = h24;
            for (int j = 24; j < 32 && static_cast<size_t>(j) < n; ++j) h32 = hash64(h32, st.hist[n - 1 - j]);
            put(-1, h20);
            put(-1, h24);
            put(-1, h32);
            // Sparse contexts.
            put(-8, (l8 >> 8) & 0xff);
            put(-8, (l8 >> 16) & 0xff);
            put(-8, (l8 >> 24) & 0xff);
            put(-4, (l8 >> 8) & 0xffff);
            put(-4, l8 & 0xff00ff);
            put(-4, l8 & 0xff0000ff);
            put(-4, l8 & 0xffff0000ULL);
            put(-4, l8 & 0xff00ff00ULL);
            put(-2, l8 & 0xffff00ffULL);
            put(-2, l8 & 0xff00ffffULL);
            put(-2, l8 & 0xffffff00ULL);
            put(-2, (l8 >> 32) & 0xffffffffULL);
            put(-1, l8 & 0xffff0000ffffULL);
            put(-1, l8 & 0xff00ff00ff00ffULL);
            put(-1, l8 & 0xffffff000000ULL);
            put(-1, l8 & 0xff000000ffULL);
            // The last bytes of words, and the last bytes other than spaces.
            put(-3, st.letters & 0xffff);
            put(-1, st.letters & 0xffffffULL);
            put(0, st.letters & 0xffffffffULL);
            put(0, st.letters & 0xffffffffffffULL);
            put(-1, st.nonspace & 0xffffULL);
            put(0, st.nonspace & 0xffffffffULL);
            put(0, st.nonspace);
            // Classes of the last bytes.
            put(-8, st.classes & 0xffff);
            put(-6, st.classes & 0xffffffffULL);
            put(-3, st.classes);
            put(-4, (st.classes & 0xffffff) | (static_cast<uint64_t>(byte) << 24));
            put(-2, (st.classes & 0xffffffffULL) | ((l8 & 0xffff) << 32));
            // Words: the current one, or between words the bytes since the last one, with the previous ones.
            word_ctx = i;
            put(0, W0);
            put(0, hash64(W0, st.w[1]));
            put(0, hash64(hash64(W0, st.w[1]), st.w[2]));
            put(0, hash64(hash64(hash64(W0, st.w[1]), st.w[2]), st.w[3]));
            put(0, hash64(W0, st.w[2]));
            put(0, hash64(W0, st.w[3]));
            put(0, hash64(hash64(W0, st.w[1]), st.w[3]));
            put(0, hash64(hash64(W0, st.w[2]), st.w[3]));
            put(0, hash64(W0, st.w[4]));
            put(-1, hash64(hash64(W0, st.w[1]), st.w[4]));
            put(-1, hash64(hash64(hash64(hash64(W0, st.w[1]), st.w[2]), st.w[3]), st.w[4]));
            put(0, hash64(st.w[1], (st.sep & 0xffff) | (static_cast<uint64_t>(byte) << 16)));
            put(0, hash64(hash64(st.w[1], st.w[2]), (st.sep & 0xff) | (static_cast<uint64_t>(byte) << 8)));
            put(-1, hash64(W0, st.sep & 0xffffff));
            put(-2, hash64(W0, static_cast<uint64_t>(std::min(st.words_in_sentence, 15)) | (st.caps << 4)));
            put(-1, hash64(W0, markup));
            put(-1, hash64(W0, st.article));
            put(-1, hash64(st.article, l8 & 0xffffff));
            put(-1, hash64(W0, l8 & 0xff));
            put(-1, hash64(hash64(W0, st.w[1]), l8 & 0xffff));
            put(-1, hash64(W0, static_cast<uint64_t>(std::min<size_t>(col, 40))));
            put(-2, hash64(W0, above));
            put(-2, hash64(W0 ^ st.w[1] ^ st.w[2] ^ st.w[3], 0x11));
            put(-2, hash64(W0, (st.w[1] ^ st.w[2]) + st.w[3] * 3));
            put(-1, hash64(W0, st.w[1] & 0xffff0000ULL));
            put(-2, hash64(W0, static_cast<uint64_t>(st.word_len) | (c1 << 8)));
            // The word that followed the previous word, and the previous two, the last time.
            const uint64_t nw = next_word[hash64(st.w[1], 0x61) >> 42];
            const uint64_t nw2 = next_word2[hash64(hash64(st.w[1], st.w[2]), 0x62) >> 42];
            put(-1, hash64(nw, W0));
            put(-1, hash64(nw2, W0));
            if (!dry && !in_word && st.w[1] != 0) {
                next_word[hash64(st.w[2], 0x61) >> 42] = st.w[1];
                next_word2[hash64(hash64(st.w[2], st.w[3]), 0x62) >> 42] = st.w[1];
            }
            // Lines, columns and tables.
            put(-2, above | (static_cast<uint64_t>(byte) << 8));
            put(-2, above | (above2 << 8) | (static_cast<uint64_t>(byte) << 16));
            put(-2, hash64(st.line_tag, std::min<uint64_t>(col, 63)));
            put(-2, hash64(st.line_head, std::min<uint64_t>(col, 63) | (static_cast<uint64_t>(byte) << 8)));
            put(-3, std::min<uint64_t>(col, 255) | (static_cast<uint64_t>(byte) << 8));
            put(-2, hash64(above, l8 & 0xffff));
            put(-3, hash64(static_cast<uint64_t>(st.cell), (static_cast<uint64_t>(st.line_kind) << 8) | byte));
            put(-2, hash64(hash64(static_cast<uint64_t>(st.line_kind), std::min<uint64_t>(col, 15)), l8 & 0xffff));
            // The state of the markup.
            put(-4, hash64(markup, l8 & 0xff));
            put(-2, hash64(markup, l8 & 0xffff));
            put(-1, hash64(markup, l8 & 0xffffff));
            put(-1, hash64(st.link_depth > 0 ? st.link_name : 0x3,
                (l8 & 0xffff) | (static_cast<uint64_t>(st.link_part) << 16)));
            put(-1, hash64(st.tmpl_depth > 0 ? st.tmpl_name : 0x4,
                (static_cast<uint64_t>(std::min(st.tmpl_param, 31)) << 16) | (l8 & 0xffff)));
            put(-1, hash64(st.tmpl_depth > 0 ? st.tmpl_name : 0x5, W0));
            put(-2, hash64(st.in_xml_tag ? st.xml_tag : 0x6, l8 & 0xffff));
            put(-3, hash64(st.in_entity ? st.entity : 0x7, l8 & 0xff));
            put(-2, hash64(static_cast<uint64_t>(st.in_ref), l8 & 0xffffff));
            put(-2, hash64(static_cast<uint64_t>(st.quote_state) | (static_cast<uint64_t>(st.line_kind) << 2), W0));
            // Numbers.
            put(-3, hash64(static_cast<uint64_t>(std::min(st.num_len, 15)), (l8 & 0xffff)));
            put(-2, hash64(st.num_hash, static_cast<uint64_t>(std::min(st.num_len, 15))));
            put(-2, hash64(st.prev_num, (l8 & 0xff) | (static_cast<uint64_t>(std::min(st.num_len, 15)) << 8)));
            // The bytes that followed the last byte and the last two, the previous times.
            put(-4, hash64(follow1[byte], byte));
            put(-2, hash64(follow2[l8 & 0xffff], l8 & 0xffff));
            put(-2, hash64(follow1[byte] & 0xff, l8 & 0xffffff));
            // The previous tokens with the bytes of the current one, and the network's choice.
            st.token_hash = hash64(st.token_hash, byte);
            put_token(-1, hash64(st.token_hash, 0x21));
            put_token(-1, hash64(hash64(st.token_hash, static_cast<uint64_t>(st.tk[1] + 2)), 0x22));
            put_token(-1, hash64(hash64(hash64(st.token_hash, static_cast<uint64_t>(st.tk[1] + 2)),
                static_cast<uint64_t>(st.tk[2] + 2)), 0x23));
            put_top(-1, hash64(static_cast<uint64_t>(st.top_token + 2),
                static_cast<uint64_t>(st.depth) | ((l8 & 0xff) << 8)));
            put_top(-1, hash64(static_cast<uint64_t>(st.top_token + 2) * 31 + static_cast<uint64_t>(st.tk[0] + 2),
                l8 & 0xffff));
            put_top(-2, hash64(static_cast<uint64_t>(std::min(st.top_cost, 31)), l8 & 0xffff));
            // The bytes the matches predict, with their lengths.
            for (int m = 0; m < CM_MATCHES; ++m)
                put(-1, st.match_len[m] > 0 ? hash64(static_cast<uint64_t>(st.match_byte[m])
                    | (static_cast<uint64_t>(std::min(st.match_len[m], 15)) << 8), 0x40 + m)
                    : hash64(l8 & 0xff, 0x50 + m));
            put(-1, st.match_len[0] > 0 ? hash64(static_cast<uint64_t>(st.match_byte[0]) | ((l8 & 0xffff) << 8), 0x48)
                : hash64(l8 & 0xffff, 0x49));
            // The current page.
            put(-2, hash64(st.article, static_cast<uint64_t>(std::min<size_t>((n - st.article_start) >> 10, 63))
                | ((l8 & 0xff) << 8)));
            put(-1, hash64(st.article, W0 ^ (st.w[1] << 1)));
            return i;
        }

        std::vector<stream_state>* streams = nullptr;
        int num_ctx = 0, inputs = 0;
        int size_class[CM_MAX_CONTEXTS] = {};
        std::vector<int> token_ctx, top_ctx;
        int word_ctx = 0;
        slot_table tables[CM_MAX_CONTEXTS];
        std::vector<uint8_t> end_tables[CM_MAX_CONTEXTS];
        std::vector<uint32_t> maps;     // per model, kind of decision and history, an adaptive probability
        uint8_t deterministic[256] = {};
        history_slot* slots[CM_MAX_CONTEXTS] = {};
        uint8_t* history[CM_MAX_CONTEXTS] = {};
        uint32_t* cells[CM_MAX_CONTEXTS] = {};
        uint32_t spare_cell = 0;
        std::vector<int16_t> x;
        int final_inputs[NUM_MIXERS + 1] = {};
        int cached_stream = -1;
        uint64_t cached_key = 0;
        std::vector<uint64_t> match_table[CM_MATCHES];
        int match_shift[CM_MATCHES] = {};
        counter_table match_map[CM_MATCHES];
        int match_used[CM_MATCHES] = {};
        std::vector<uint16_t> follow1, follow2;
        std::vector<uint64_t> next_word, next_word2;
        apm nn_map, nn_map2, nn_map3, nn_map4;
        bool nn_active = false;
        int nn_stream = 0, nn_value = 0;
        const uint8_t* quarter_bits = nullptr;
        vector_mixer mixers[NUM_MIXERS];
        final_mixer final;
        apm a1, a2, a3, a4;
    };

    // ----------------------------------------------------------------------------------------
    // Transformer inference

    struct engine_shape
    {
        int vocab = 0, dim = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, ffn = 0, window = 0;
        int q_dim () const { return heads * head_dim; }
        int kv_dim () const { return kv_heads * head_dim; }
    };

    class byte_reader
    {
    public:
        byte_reader (const uint8_t* data_, size_t size_) : data(data_), size(size_) {}
        uint32_t u32 ()
        {
            need(4);
            const uint32_t v = data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16)
                | (static_cast<uint32_t>(data[pos + 3]) << 24);
            pos += 4;
            return v;
        }
        int32_t i32 () { return static_cast<int32_t>(u32()); }
        float f32 () { const uint32_t v = u32(); float f; std::memcpy(&f, &v, sizeof(f)); return f; }
        int8_t i8 () { need(1); return static_cast<int8_t>(data[pos++]); }
        uint16_t u16 ()
        {
            need(2);
            const uint16_t v = static_cast<uint16_t>(data[pos] | (data[pos + 1] << 8));
            pos += 2;
            return v;
        }
        bool done () const { return pos == size; }
    private:
        void need (size_t n) const { if (pos + n > size) throw std::runtime_error("truncated embedded data"); }
        const uint8_t* data;
        size_t size, pos = 0;
    };

    /*
        The products of the engine. On x86-64 the floating-point ones have a version with AVX2
        and FMA, chosen at run time when the processor has both and the system saves the vector
        registers. The two versions accumulate every element through the same fused
        multiply-adds in the same order, so they give the same values. The vector versions
        multiply only inside fused multiply-adds, which leaves a compiler nothing to fuse on
        its own. The products of the weight matrices are on integers, exact in any order: they
        have versions for the VNNI instructions, with 512-bit or 256-bit encodings, and for
        AVX2, besides the portable one.
    */

#if SLM_CODEC_X86
    namespace kernels_avx2
    {
        SLM_AVX2_FMA inline void scores (const float* q, const float* keys, size_t stride, int hd, int cnt, float* sc)
        {
            int j = 0;
            for (; j + 32 <= cnt; j += 32) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                const float* col = keys + j;
                for (int i = 0; i < hd; ++i, col += stride) {
                    const __m256 qi = _mm256_broadcast_ss(q + i);
                    a0 = _mm256_fmadd_ps(qi, _mm256_loadu_ps(col), a0);
                    a1 = _mm256_fmadd_ps(qi, _mm256_loadu_ps(col + 8), a1);
                    a2 = _mm256_fmadd_ps(qi, _mm256_loadu_ps(col + 16), a2);
                    a3 = _mm256_fmadd_ps(qi, _mm256_loadu_ps(col + 24), a3);
                }
                _mm256_storeu_ps(sc + j, a0);
                _mm256_storeu_ps(sc + j + 8, a1);
                _mm256_storeu_ps(sc + j + 16, a2);
                _mm256_storeu_ps(sc + j + 24, a3);
            }
            for (; j + 8 <= cnt; j += 8) {
                __m256 a = _mm256_setzero_ps();
                const float* col = keys + j;
                for (int i = 0; i < hd; ++i, col += stride)
                    a = _mm256_fmadd_ps(_mm256_broadcast_ss(q + i), _mm256_loadu_ps(col), a);
                _mm256_storeu_ps(sc + j, a);
            }
            for (; j < cnt; ++j) {
                float a = 0.0f;
                for (int i = 0; i < hd; ++i) a = std::fma(q[i], keys[i * stride + j], a);
                sc[j] = a;
            }
        }

        SLM_AVX2_FMA inline void weighted_sum (const float* w, const float* vals, size_t stride, int cnt, int hd,
            float* out)
        {
            int i = 0;
            for (; i + 32 <= hd; i += 32) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                const float* v = vals + i;
                for (int j = 0; j < cnt; ++j, v += stride) {
                    const __m256 wj = _mm256_broadcast_ss(w + j);
                    a0 = _mm256_fmadd_ps(wj, _mm256_loadu_ps(v), a0);
                    a1 = _mm256_fmadd_ps(wj, _mm256_loadu_ps(v + 8), a1);
                    a2 = _mm256_fmadd_ps(wj, _mm256_loadu_ps(v + 16), a2);
                    a3 = _mm256_fmadd_ps(wj, _mm256_loadu_ps(v + 24), a3);
                }
                _mm256_storeu_ps(out + i, a0);
                _mm256_storeu_ps(out + i + 8, a1);
                _mm256_storeu_ps(out + i + 16, a2);
                _mm256_storeu_ps(out + i + 24, a3);
            }
            for (; i + 8 <= hd; i += 8) {
                __m256 a = _mm256_setzero_ps();
                const float* v = vals + i;
                for (int j = 0; j < cnt; ++j, v += stride)
                    a = _mm256_fmadd_ps(_mm256_broadcast_ss(w + j), _mm256_loadu_ps(v), a);
                _mm256_storeu_ps(out + i, a);
            }
            for (; i < hd; ++i) {
                float a = 0.0f;
                for (int j = 0; j < cnt; ++j) a = std::fma(w[j], vals[j * stride + i], a);
                out[i] = a;
            }
        }

        SLM_AVX2_FMA inline float dot (const float* a, const float* b, int n)
        {
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= n; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
            float lane[8];
            _mm256_storeu_ps(lane, acc);
            for (int j = 0; i < n; ++i, ++j) lane[j] = std::fma(a[i], b[i], lane[j]);
            return ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
        }

        /*
            The scores of two query heads sharing their keys, in one pass over the keys: each
            score accumulates the same fused products in the same order as scores() gives it.
        */
        SLM_AVX2_FMA inline void scores2 (const float* q0, const float* q1, const float* keys, size_t stride, int hd,
            int cnt, float* sc0, float* sc1)
        {
            int j = 0;
            for (; j + 32 <= cnt; j += 32) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                const float* col = keys + j;
                for (int i = 0; i < hd; ++i, col += stride) {
                    const __m256 k0 = _mm256_loadu_ps(col), k1 = _mm256_loadu_ps(col + 8);
                    const __m256 k2 = _mm256_loadu_ps(col + 16), k3 = _mm256_loadu_ps(col + 24);
                    const __m256 x = _mm256_broadcast_ss(q0 + i), y = _mm256_broadcast_ss(q1 + i);
                    a0 = _mm256_fmadd_ps(x, k0, a0); a1 = _mm256_fmadd_ps(x, k1, a1);
                    a2 = _mm256_fmadd_ps(x, k2, a2); a3 = _mm256_fmadd_ps(x, k3, a3);
                    b0 = _mm256_fmadd_ps(y, k0, b0); b1 = _mm256_fmadd_ps(y, k1, b1);
                    b2 = _mm256_fmadd_ps(y, k2, b2); b3 = _mm256_fmadd_ps(y, k3, b3);
                }
                _mm256_storeu_ps(sc0 + j, a0); _mm256_storeu_ps(sc0 + j + 8, a1);
                _mm256_storeu_ps(sc0 + j + 16, a2); _mm256_storeu_ps(sc0 + j + 24, a3);
                _mm256_storeu_ps(sc1 + j, b0); _mm256_storeu_ps(sc1 + j + 8, b1);
                _mm256_storeu_ps(sc1 + j + 16, b2); _mm256_storeu_ps(sc1 + j + 24, b3);
            }
            if (j < cnt) {
                scores(q0, keys + j, stride, hd, cnt - j, sc0 + j);
                scores(q1, keys + j, stride, hd, cnt - j, sc1 + j);
            }
        }

        // The weighted sums of two heads sharing their values, in one pass over the values.
        SLM_AVX2_FMA inline void weighted_sum2 (const float* w0, const float* w1, const float* vals, size_t stride,
            int cnt, int hd, float* out0, float* out1)
        {
            int i = 0;
            for (; i + 32 <= hd; i += 32) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                const float* v = vals + i;
                for (int j = 0; j < cnt; ++j, v += stride) {
                    const __m256 v0 = _mm256_loadu_ps(v), v1 = _mm256_loadu_ps(v + 8);
                    const __m256 v2 = _mm256_loadu_ps(v + 16), v3 = _mm256_loadu_ps(v + 24);
                    const __m256 x = _mm256_broadcast_ss(w0 + j), y = _mm256_broadcast_ss(w1 + j);
                    a0 = _mm256_fmadd_ps(x, v0, a0); a1 = _mm256_fmadd_ps(x, v1, a1);
                    a2 = _mm256_fmadd_ps(x, v2, a2); a3 = _mm256_fmadd_ps(x, v3, a3);
                    b0 = _mm256_fmadd_ps(y, v0, b0); b1 = _mm256_fmadd_ps(y, v1, b1);
                    b2 = _mm256_fmadd_ps(y, v2, b2); b3 = _mm256_fmadd_ps(y, v3, b3);
                }
                _mm256_storeu_ps(out0 + i, a0); _mm256_storeu_ps(out0 + i + 8, a1);
                _mm256_storeu_ps(out0 + i + 16, a2); _mm256_storeu_ps(out0 + i + 24, a3);
                _mm256_storeu_ps(out1 + i, b0); _mm256_storeu_ps(out1 + i + 8, b1);
                _mm256_storeu_ps(out1 + i + 16, b2); _mm256_storeu_ps(out1 + i + 24, b3);
            }
            if (i < hd) {
                weighted_sum(w0, vals + i, stride, cnt, hd - i, out0 + i);
                weighted_sum(w1, vals + i, stride, cnt, hd - i, out1 + i);
            }
        }

        /*
            The elementwise kernels below compute what the scalar code computes, element by
            element and in the same order; they are compiled for AVX2 without FMA, so that no
            multiplication can fuse with an addition.
        */

        // det::expf on eight floats.
        SLM_AVX2 inline __m256 expf8 (__m256 x)
        {
            __m256i bits = _mm256_min_epu32(_mm256_castps_si256(x), _mm256_set1_epi32(static_cast<int>(0xC2AE0000u)));
            bits = _mm256_min_epi32(bits, _mm256_set1_epi32(0x42B00000));
            x = _mm256_castsi256_ps(bits);
            const __m256 magic = _mm256_set1_ps(12582912.0f);
            const __m256 k = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), magic),
                magic);
            __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(k, _mm256_set1_ps(0.693359375f)));
            r = _mm256_sub_ps(r, _mm256_mul_ps(k, _mm256_set1_ps(-2.12194440e-4f)));
            __m256 p = _mm256_set1_ps(1.9875691500e-4f);
            p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.3981999507e-3f));
            p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(8.3334519073e-3f));
            p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(4.1665795894e-2f));
            p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.6666665459e-1f));
            p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(5.0000001201e-1f));
            p = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(p, _mm256_mul_ps(r, r)), r), _mm256_set1_ps(1.0f));
            const __m256i exponent = _mm256_add_epi32(_mm256_cvttps_epi32(k), _mm256_set1_epi32(127));
            return _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(exponent, 23)));
        }

        SLM_AVX2 inline float max_of (__m256 m)
        {
            float lane[8];
            _mm256_storeu_ps(lane, m);
            float mx = lane[0];
            for (int j = 1; j < 8; ++j) mx = lane[j] > mx ? lane[j] : mx;
            return mx;
        }

        // The sum of n floats in eight interleaved partial sums, combined in a fixed order.
        SLM_AVX2 inline float sum (const float* a, int n)
        {
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= n; i += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
            float lane[8];
            _mm256_storeu_ps(lane, acc);
            for (int j = 0; i < n; ++i, ++j) lane[j] += a[i];
            return ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
        }

        // e[i] = exp(a[i] - mx), then e divided by its sum.
        SLM_AVX2 inline void exp_normalize (const float* a, float mx, int n, float* e)
        {
            const __m256 m = _mm256_set1_ps(mx);
            int i = 0;
            for (; i + 8 <= n; i += 8) _mm256_storeu_ps(e + i, expf8(_mm256_sub_ps(_mm256_loadu_ps(a + i), m)));
            for (; i < n; ++i) e[i] = det::expf(a[i] - mx);
            const float inv = 1.0f / sum(e, n);
            const __m256 s = _mm256_set1_ps(inv);
            for (i = 0; i + 8 <= n; i += 8) _mm256_storeu_ps(e + i, _mm256_mul_ps(_mm256_loadu_ps(e + i), s));
            for (; i < n; ++i) e[i] *= inv;
        }

        // The attention weights of n scores: each multiplied by scale, then their softmax, in place.
        SLM_AVX2 inline void attention_weights (float* sc, int n, float scale)
        {
            const __m256 s = _mm256_set1_ps(scale);
            __m256 m = _mm256_set1_ps(-3.0e38f);
            int i = 0;
            for (; i + 8 <= n; i += 8) {
                const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(sc + i), s);
                _mm256_storeu_ps(sc + i, v);
                m = _mm256_max_ps(m, v);
            }
            float mx = max_of(m);
            for (; i < n; ++i) { sc[i] *= scale; mx = sc[i] > mx ? sc[i] : mx; }
            exp_normalize(sc, mx, n, sc);
        }

        // The distribution over the vocabulary: logits plus the two biases, then their softmax into p.
        SLM_AVX2 inline void distribution (float* lg, const float* b1, const float* b2, int n, float* p)
        {
            __m256 m = _mm256_set1_ps(-3.0e38f);
            int i = 0;
            for (; i + 8 <= n; i += 8) {
                const __m256 v = _mm256_add_ps(_mm256_loadu_ps(lg + i),
                    _mm256_add_ps(_mm256_loadu_ps(b1 + i), _mm256_loadu_ps(b2 + i)));
                _mm256_storeu_ps(lg + i, v);
                m = _mm256_max_ps(m, v);
            }
            float mx = max_of(m);
            for (; i < n; ++i) { lg[i] += b1[i] + b2[i]; mx = lg[i] > mx ? lg[i] : mx; }
            exp_normalize(lg, mx, n, p);
        }

        // a[i] = (b[i] + bb[i]) * silu(a[i] + ba[i]), the gate of SwiGLU.
        SLM_AVX2 inline void swiglu (float* a, const float* b, const float* ba, const float* bb, int n)
        {
            const __m256 one = _mm256_set1_ps(1.0f), sign = _mm256_set1_ps(-0.0f);
            int i = 0;
            for (; i + 8 <= n; i += 8) {
                const __m256 za = _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(ba + i));
                const __m256 g = _mm256_div_ps(za, _mm256_add_ps(one, expf8(_mm256_xor_ps(za, sign))));
                const __m256 h = _mm256_add_ps(_mm256_loadu_ps(b + i), _mm256_loadu_ps(bb + i));
                _mm256_storeu_ps(a + i, _mm256_mul_ps(h, g));
            }
            for (; i < n; ++i) {
                const float za = a[i] + ba[i];
                a[i] = (b[i] + bb[i]) * (za / (1.0f + det::expf(-za)));
            }
        }

        // dst[i] = row[i] * inv * gain[i].
        SLM_AVX2 inline void scale_by_gain (const float* row, float inv, const float* gain, int n, float* dst)
        {
            const __m256 s = _mm256_set1_ps(inv);
            int i = 0;
            for (; i + 8 <= n; i += 8)
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(row + i), s),
                    _mm256_loadu_ps(gain + i)));
            for (; i < n; ++i) dst[i] = row[i] * inv * gain[i];
        }

        // x[i] += o[i], or x[i] += o[i] + b[i] when b is given.
        SLM_AVX2 inline void accumulate (float* x, const float* o, const float* b, int n)
        {
            int i = 0;
            if (b) {
                for (; i + 8 <= n; i += 8)
                    _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i),
                        _mm256_add_ps(_mm256_loadu_ps(o + i), _mm256_loadu_ps(b + i))));
                for (; i < n; ++i) x[i] += o[i] + b[i];
            }
            else {
                for (; i + 8 <= n; i += 8)
                    _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(o + i)));
                for (; i < n; ++i) x[i] += o[i];
            }
        }

        // o[c] = a[c] * (sx * scale[c]): the integer sums of a row brought back to floating point.
        SLM_AVX2 inline void rescale (const int32_t* a, float sx, const float* scale, int n, float* o)
        {
            const __m256 s = _mm256_set1_ps(sx);
            int c = 0;
            for (; c + 8 <= n; c += 8)
                _mm256_storeu_ps(o + c, _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + c))),
                    _mm256_mul_ps(s, _mm256_loadu_ps(scale + c))));
            for (; c < n; ++c) o[c] = static_cast<float>(a[c]) * (sx * scale[c]);
        }

        // b[i] -= rate * p[i].
        SLM_AVX2 inline void descend (float* b, const float* p, float rate, int n)
        {
            const __m256 r = _mm256_set1_ps(rate);
            int i = 0;
            for (; i + 8 <= n; i += 8)
                _mm256_storeu_ps(b + i,
                    _mm256_sub_ps(_mm256_loadu_ps(b + i), _mm256_mul_ps(r, _mm256_loadu_ps(p + i))));
            for (; i < n; ++i) b[i] -= rate * p[i];
        }

        // The index of the first largest of n floats.
        SLM_AVX2 inline int first_max (const float* p, int n)
        {
            __m256 m = _mm256_loadu_ps(p);
            int i = 8;
            for (; i + 8 <= n; i += 8) m = _mm256_max_ps(m, _mm256_loadu_ps(p + i));
            float mx = max_of(m);
            for (; i < n; ++i) mx = p[i] > mx ? p[i] : mx;
            int r = 0;
            while (p[r] != mx) ++r;
            return r;
        }

        // The largest magnitude of n floats.
        SLM_AVX2 inline float max_abs (const float* a, int n)
        {
            const __m256 sign = _mm256_set1_ps(-0.0f);
            __m256 m = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= n; i += 8) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(a + i)));
            float mx = max_of(m);
            for (; i < n; ++i) mx = std::max(mx, std::fabs(a[i]));
            return mx;
        }

        // q[i] = round(a[i] * inv), ties to even, as signed bytes.
        SLM_AVX2 inline void round_to_bytes (const float* a, float inv, int n, int8_t* q)
        {
            const __m256 s = _mm256_set1_ps(inv), magic = _mm256_set1_ps(12582912.0f);
            const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            int i = 0;
            for (; i + 32 <= n; i += 32) {
                __m256i v[4];
                for (int j = 0; j < 4; ++j) {
                    const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(a + i + 8 * j), s);
                    v[j] = _mm256_cvttps_epi32(_mm256_sub_ps(_mm256_add_ps(scaled, magic), magic));
                }
                const __m256i words0 = _mm256_packs_epi32(v[0], v[1]), words1 = _mm256_packs_epi32(v[2], v[3]);
                const __m256i bytes = _mm256_packs_epi16(words0, words1);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + i), _mm256_permutevar8x32_epi32(bytes, order));
            }
            for (; i < n; ++i) {
                const float magic_s = 12582912.0f;
                q[i] = static_cast<int8_t>((a[i] * inv + magic_s) - magic_s);
            }
        }

        // u[i] = q[i] + 128, as unsigned bytes.
        SLM_AVX2 inline void to_unsigned (const int8_t* q, size_t n, uint8_t* u)
        {
            const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
            size_t i = 0;
            for (; i + 32 <= n; i += 32)
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(u + i), _mm256_xor_si256(flip,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i))));
            for (; i < n; ++i) u[i] = static_cast<uint8_t>(q[i] + 128);
        }
    }
#endif

    namespace kernels
    {
        inline bool vector_paths () { return processor().avx2_fma; }

        // sc[j] = sum over i of q[i] * keys[i * stride + j], for j below cnt.
        inline void scores (const float* q, const float* keys, size_t stride, int hd, int cnt, float* sc)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::scores(q, keys, stride, hd, cnt, sc); return; }
#endif
            for (int j = 0; j < cnt; ++j) sc[j] = 0.0f;
            for (int i = 0; i < hd; ++i) {
                const float qi = q[i];
                const float* col = keys + i * stride;
                for (int j = 0; j < cnt; ++j) sc[j] = std::fma(qi, col[j], sc[j]);
            }
        }

        // out[i] = sum over j of w[j] * vals[j * stride + i], for i below hd.
        inline void weighted_sum (const float* w, const float* vals, size_t stride, int cnt, int hd, float* out)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::weighted_sum(w, vals, stride, cnt, hd, out); return; }
#endif
            for (int i = 0; i < hd; ++i) out[i] = 0.0f;
            for (int j = 0; j < cnt; ++j) {
                const float a = w[j];
                const float* v = vals + j * stride;
                for (int i = 0; i < hd; ++i) out[i] = std::fma(a, v[i], out[i]);
            }
        }

        // Dot product with eight interleaved partial sums, combined in a fixed order.
        inline float dot (const float* a, const float* b, int n)
        {
#if SLM_CODEC_X86
            if (vector_paths()) return kernels_avx2::dot(a, b, n);
#endif
            float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            int i = 0;
            for (; i + 8 <= n; i += 8)
                for (int j = 0; j < 8; ++j) acc[j] = std::fma(a[i + j], b[i + j], acc[j]);
            for (int j = 0; i < n; ++i, ++j) acc[j] = std::fma(a[i], b[i], acc[j]);
            return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        }

        // scores() for two query heads sharing their keys.
        inline void scores2 (const float* q0, const float* q1, const float* keys, size_t stride, int hd, int cnt,
            float* sc0, float* sc1)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::scores2(q0, q1, keys, stride, hd, cnt, sc0, sc1); return; }
#endif
            scores(q0, keys, stride, hd, cnt, sc0);
            scores(q1, keys, stride, hd, cnt, sc1);
        }

        // weighted_sum() for two heads sharing their values.
        inline void weighted_sum2 (const float* w0, const float* w1, const float* vals, size_t stride, int cnt,
            int hd, float* out0, float* out1)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::weighted_sum2(w0, w1, vals, stride, cnt, hd, out0, out1); return; }
#endif
            weighted_sum(w0, vals, stride, cnt, hd, out0);
            weighted_sum(w1, vals, stride, cnt, hd, out1);
        }

        // The sum of n floats in eight interleaved partial sums, combined in a fixed order.
        inline float sum (const float* a, int n)
        {
#if SLM_CODEC_X86
            if (vector_paths()) return kernels_avx2::sum(a, n);
#endif
            float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            int i = 0;
            for (; i + 8 <= n; i += 8)
                for (int j = 0; j < 8; ++j) acc[j] += a[i + j];
            for (int j = 0; i < n; ++i, ++j) acc[j] += a[i];
            return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        }

        // The attention weights of n scores: each multiplied by scale, then their softmax, in place.
        inline void attention_weights (float* sc, int n, float scale)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::attention_weights(sc, n, scale); return; }
#endif
            float mx = -3.0e38f;
            for (int j = 0; j < n; ++j) { sc[j] *= scale; mx = sc[j] > mx ? sc[j] : mx; }
            for (int j = 0; j < n; ++j) sc[j] = det::expf(sc[j] - mx);
            const float inv = 1.0f / sum(sc, n);
            for (int j = 0; j < n; ++j) sc[j] *= inv;
        }

        // The distribution over the vocabulary: logits plus the two biases, then their softmax into p.
        inline void distribution (float* lg, const float* b1, const float* b2, int n, float* p)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::distribution(lg, b1, b2, n, p); return; }
#endif
            float mx = -3.0e38f;
            for (int i = 0; i < n; ++i) { lg[i] += b1[i] + b2[i]; mx = lg[i] > mx ? lg[i] : mx; }
            for (int i = 0; i < n; ++i) p[i] = det::expf(lg[i] - mx);
            const float inv = 1.0f / sum(p, n);
            for (int i = 0; i < n; ++i) p[i] *= inv;
        }

        // a[i] = (b[i] + bb[i]) * silu(a[i] + ba[i]), the gate of SwiGLU.
        inline void swiglu (float* a, const float* b, const float* ba, const float* bb, int n)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::swiglu(a, b, ba, bb, n); return; }
#endif
            for (int i = 0; i < n; ++i) {
                const float za = a[i] + ba[i];
                a[i] = (b[i] + bb[i]) * (za / (1.0f + det::expf(-za)));
            }
        }

        // dst[i] = row[i] * inv * gain[i].
        inline void scale_by_gain (const float* row, float inv, const float* gain, int n, float* dst)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::scale_by_gain(row, inv, gain, n, dst); return; }
#endif
            for (int i = 0; i < n; ++i) dst[i] = row[i] * inv * gain[i];
        }

        // x[i] += o[i], or x[i] += o[i] + b[i] when b is given.
        inline void accumulate (float* x, const float* o, const float* b, int n)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::accumulate(x, o, b, n); return; }
#endif
            if (b) for (int i = 0; i < n; ++i) x[i] += o[i] + b[i];
            else for (int i = 0; i < n; ++i) x[i] += o[i];
        }

        // o[c] = a[c] * (sx * scale[c]).
        inline void rescale (const int32_t* a, float sx, const float* scale, int n, float* o)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::rescale(a, sx, scale, n, o); return; }
#endif
            for (int c = 0; c < n; ++c) o[c] = static_cast<float>(a[c]) * (sx * scale[c]);
        }

        // b[i] -= rate * p[i].
        inline void descend (float* b, const float* p, float rate, int n)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::descend(b, p, rate, n); return; }
#endif
            for (int i = 0; i < n; ++i) b[i] -= rate * p[i];
        }

        // The index of the first largest of n floats.
        inline int first_max (const float* p, int n)
        {
#if SLM_CODEC_X86
            if (vector_paths() && n >= 8) return kernels_avx2::first_max(p, n);
#endif
            float top = p[0];
            int top_id = 0;
            for (int r = 1; r < n; ++r) if (p[r] > top) { top = p[r]; top_id = r; }
            return top_id;
        }

        // The largest magnitude of n floats.
        inline float max_abs (const float* a, int n)
        {
#if SLM_CODEC_X86
            if (vector_paths()) return kernels_avx2::max_abs(a, n);
#endif
            float mx = 0.0f;
            for (int i = 0; i < n; ++i) mx = std::max(mx, std::fabs(a[i]));
            return mx;
        }

        // q[i] = round(a[i] * inv), ties to even, as signed bytes.
        inline void round_to_bytes (const float* a, float inv, int n, int8_t* q)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::round_to_bytes(a, inv, n, q); return; }
#endif
            const float magic = 12582912.0f;    // 1.5 * 2^23: rounds to the nearest integer, ties to even
            for (int i = 0; i < n; ++i) q[i] = static_cast<int8_t>((a[i] * inv + magic) - magic);
        }

        // u[i] = q[i] + 128, as unsigned bytes.
        inline void to_unsigned (const int8_t* q, size_t n, uint8_t* u)
        {
#if SLM_CODEC_X86
            if (vector_paths()) { kernels_avx2::to_unsigned(q, n, u); return; }
#endif
            for (size_t i = 0; i < n; ++i) u[i] = static_cast<uint8_t>(q[i] + 128);
        }
    }

    /*
        A weight matrix for integer products: signed bytes with one scale per column, in
        strips of sixteen columns. Within a strip, the rows go by groups of four, each group
        holding the four bytes of every column side by side, the order the VNNI instructions
        read and the AVX2 byte products too when the weights fit in 7 bits; processors with
        AVX2 alone read pairs of 16-bit integers for weights of 8 bits. The rows are padded to
        a multiple of four and the columns to a multiple of sixteen with zeros.
    */
    struct int_matrix
    {
        int rows = 0, cols = 0, kpad = 0, strips = 0;
        bool small = false;            // no weight exceeds 63 in magnitude, as the AVX2 byte products need
        std::vector<float> scale;      // per column
        std::vector<int8_t> q4;        // per strip, per group of four rows: sixteen columns of four bytes
        std::vector<int16_t> q2;       // per strip, per pair of rows: sixteen columns of two, for AVX2
        std::vector<int32_t> colsum;   // per padded column: the sum of its weights
    };

    inline bool vnni_paths () { return SLM_VNNI_KERNELS && (processor().avx512_vnni || processor().avx_vnni); }

    // m holds rows x cols signed bytes, row by row.
    inline int_matrix pack_int (const std::vector<int8_t>& m, std::vector<float> scale, int rows, int cols)
    {
        int_matrix w;
        w.rows = rows;
        w.cols = cols;
        w.kpad = (rows + 3) & ~3;
        w.strips = (cols + 15) / 16;
        w.scale = std::move(scale);
        w.q4.assign(static_cast<size_t>(w.strips) * w.kpad * 16, 0);
        w.colsum.assign(static_cast<size_t>(w.strips) * 16, 0);
        int largest = 0;
        for (const int8_t v : m) largest = std::max(largest, std::abs(static_cast<int>(v)));
        w.small = largest <= 63;
        const bool pairs = !vnni_paths() && processor().avx2_fma && !w.small;
        if (pairs) w.q2.assign(w.q4.size(), 0);
        for (int st = 0; st < w.strips; ++st)
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < 16 && st * 16 + c < cols; ++c) {
                    const int8_t v = m[static_cast<size_t>(r) * cols + st * 16 + c];
                    const size_t base = static_cast<size_t>(st) * w.kpad * 16;
                    w.q4[base + static_cast<size_t>(r & ~3) * 16 + c * 4 + (r & 3)] = v;
                    if (pairs) w.q2[base + static_cast<size_t>(r & ~1) * 16 + c * 2 + (r & 1)] = v;
                    w.colsum[static_cast<size_t>(st) * 16 + c] += v;
                }
        return w;
    }

    /*
        out[r][c] = sum over k of x[r][k] * w[k][c], exactly, for n rows of signed bytes padded
        to w.kpad. The byte paths take the inputs as unsigned bytes, x + 128, and subtract 128
        times the column sums: the VNNI instructions, and in AVX2 the products of bytes summed
        by pairs into 16 bits, which stay exact for weights of at most 63 in magnitude, since a
        pair then stays below 2^15. For larger weights the AVX2 path takes the inputs as 16-bit
        integers.
    */
    namespace kernels_int
    {
        inline void store_tile (const int32_t (*tile)[16], int rows, int c0, int cols, int m, int r0, int32_t* out)
        {
            for (int r = 0; r < rows; ++r)
                std::memcpy(out + static_cast<size_t>(r0 + r) * m + c0, tile[r], sizeof(int32_t) * cols);
        }

        inline void portable (const int8_t* x, int n, const int_matrix& w, int32_t* out)
        {
            for (int st = 0; st < w.strips; ++st) {
                const int8_t* strip = &w.q4[static_cast<size_t>(st) * w.kpad * 16];
                const int c0 = st * 16, cols = std::min(16, w.cols - c0);
                for (int r0 = 0; r0 < n; r0 += 4) {
                    const int rows = std::min(4, n - r0);
                    int32_t tile[4][16] = {};
                    for (int r = 0; r < rows; ++r) {
                        const int8_t* xr = x + static_cast<size_t>(r0 + r) * w.kpad;
                        for (int k = 0; k < w.kpad; k += 4) {
                            const int8_t* g = strip + static_cast<size_t>(k) * 16;
                            for (int c = 0; c < 16; ++c)
                                for (int j = 0; j < 4; ++j)
                                    tile[r][c] += static_cast<int32_t>(xr[k + j]) * g[c * 4 + j];
                        }
                    }
                    store_tile(tile, rows, c0, cols, w.cols, r0, out);
                }
            }
        }

#if SLM_CODEC_X86
        // The accumulators of four rows by sixteen columns are named one by one, which keeps them in registers.
#define SLM_INT_TILE(STEP, LOADX, KSTEP, PTRSTEP)                                                        \
            __m256i a00 = _mm256_setzero_si256(), a01 = a00, a10 = a00, a11 = a00;                          \
            __m256i a20 = a00, a21 = a00, a30 = a00, a31 = a00;                                             \
            for (int k = 0; k < w.kpad; k += KSTEP, p += PTRSTEP) {                                         \
                const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));                 \
                const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + PTRSTEP / 2));   \
                __m256i b = LOADX(xr[0] + k); a00 = STEP(a00, b, w0); a01 = STEP(a01, b, w1);                \
                b = LOADX(xr[1] + k); a10 = STEP(a10, b, w0); a11 = STEP(a11, b, w1);                       \
                b = LOADX(xr[2] + k); a20 = STEP(a20, b, w0); a21 = STEP(a21, b, w1);                       \
                b = LOADX(xr[3] + k); a30 = STEP(a30, b, w0); a31 = STEP(a31, b, w1);                       \
            }

        // Broadcasts the four bytes (or two 16-bit integers) at p to every 32-bit lane.
        SLM_AVX2 inline __m256i broadcast32 (const void* p)
        {
            int32_t v;
            std::memcpy(&v, p, 4);
            return _mm256_set1_epi32(v);
        }

        SLM_AVX2 inline __m256i madd16 (__m256i acc, __m256i x, __m256i w)
        {
            return _mm256_add_epi32(acc, _mm256_madd_epi16(x, w));
        }

        SLM_AVX2 inline void avx2 (const int16_t* x, int n, const int_matrix& w, int32_t* out)
        {
            for (int st = 0; st < w.strips; ++st) {
                const int c0 = st * 16, cols = std::min(16, w.cols - c0);
                for (int r0 = 0; r0 < n; r0 += 4) {
                    const int rows = std::min(4, n - r0);
                    const int16_t* xr[4];
                    for (int r = 0; r < 4; ++r) xr[r] = x + static_cast<size_t>(r0 + (r < rows ? r : 0)) * w.kpad;
                    const int16_t* p = &w.q2[static_cast<size_t>(st) * w.kpad * 16];
                    SLM_INT_TILE(madd16, broadcast32, 2, 32)
                    int32_t tile[4][16];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile[0]), a00);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile[0] + 8), a01);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile[1]), a10);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile[1] + 8), a11);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile[2]), a20);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile[2] + 8), a21);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile[3]), a30);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tile[3] + 8), a31);
                    store_tile(tile, rows, c0, cols, w.cols, r0, out);
                }
            }
        }

        // 128 times the column sums from c on: what the inputs, taken as unsigned bytes, add.
        SLM_AVX2 inline __m256i offset_sums (const int32_t* c)
        {
            return _mm256_slli_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(c)), 7);
        }

        // Stores the sixteen sums of a row of a tile, their offsets removed.
        SLM_AVX2 inline void store_row (int32_t* row, __m256i lo, __m256i hi, __m256i s0, __m256i s1)
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(row), _mm256_sub_epi32(lo, s0));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(row + 8), _mm256_sub_epi32(hi, s1));
        }

        // The four products of each 32-bit lane in AVX2: pairs summed into 16 bits, then into 32.
        SLM_AVX2 inline __m256i bytes_avx2 (__m256i acc, __m256i x, __m256i w)
        {
            return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), _mm256_set1_epi16(1)));
        }

#define SLM_BYTE_GEMM(NAME, TARGET, STEP)                                                                   \
        TARGET inline __m256i NAME##_step (__m256i acc, __m256i x, __m256i w) { return STEP(acc, x, w); }   \
        TARGET inline __m256i NAME##_load (const uint8_t* p)                                                \
        {                                                                                                   \
            int32_t v;                                                                                      \
            std::memcpy(&v, p, 4);                                                                          \
            return _mm256_set1_epi32(v);                                                                    \
        }                                                                                                   \
        TARGET inline void NAME (const uint8_t* x, int n, const int_matrix& w, int32_t* out)                \
        {                                                                                                   \
            for (int st = 0; st < w.strips; ++st) {                                                         \
                const int c0 = st * 16, cols = std::min(16, w.cols - c0);                                   \
                const __m256i s0 = offset_sums(&w.colsum[c0]), s1 = offset_sums(&w.colsum[c0 + 8]);         \
                for (int r0 = 0; r0 < n; r0 += 4) {                                                         \
                    const int rows = std::min(4, n - r0);                                                   \
                    const uint8_t* xr[4];                                                                   \
                    for (int r = 0; r < 4; ++r)                                                             \
                        xr[r] = x + static_cast<size_t>(r0 + (r < rows ? r : 0)) * w.kpad;                  \
                    const int8_t* p = &w.q4[static_cast<size_t>(st) * w.kpad * 16];                         \
                    SLM_INT_TILE(NAME##_step, NAME##_load, 4, 64)                                           \
                    int32_t tile[4][16];                                                                    \
                    store_row(tile[0], a00, a01, s0, s1);                                                   \
                    store_row(tile[1], a10, a11, s0, s1);                                                   \
                    store_row(tile[2], a20, a21, s0, s1);                                                   \
                    store_row(tile[3], a30, a31, s0, s1);                                                   \
                    store_tile(tile, rows, c0, cols, w.cols, r0, out);                                      \
                }                                                                                           \
            }                                                                                               \
        }

        SLM_BYTE_GEMM(avx2_bytes, SLM_AVX2, bytes_avx2)
#if SLM_VNNI_KERNELS
        SLM_BYTE_GEMM(vnni512, SLM_AVX512_VNNI, _mm256_dpbusd_epi32)
        SLM_BYTE_GEMM(vnni256, SLM_AVX_VNNI, _mm256_dpbusd_avx_epi32)
#endif
#undef SLM_BYTE_GEMM
#undef SLM_INT_TILE
#endif
    }

    /*
        Runs the decoder the training network defines, one position at a time for several
        independent streams at once:

            x = E[token]
            per block:  x += W_o attention(rope(W_q n1(x)), rope(W_k n1(x)), W_v n1(x))
                        x += W_d (silu(W_a n2(x) + b_a) * (W_b n2(x) + b_b)) + b_d
            logits = W_out nf(x) + b_out

        n being an RMS normalisation scaled by a learned gain. E[token] is the column of the
        head for the token times a learned scale when the input and the head share one table,
        as the training of this program makes them, and a row of a table of its own in the
        earlier form of the data.

        Each stream keeps the keys and values of its last window positions in rings. Rotary
        attention depends on positions only through their differences, so the keys are kept
        rotated to their position relative to a base, which moves forward by a window every
        window steps; the keys still in the ring are then rotated again from their unrotated
        copy. Relative positions stay below twice the window and every key is rotated about
        twice in its life, rather than at every step.

        Every accumulation runs in an order fixed by the shapes alone, never by how many
        streams share a step, so a position gets the same probabilities alone or in a batch.
    */
    class transformer_engine
    {
    public:
        bool loaded () const { return shape_.vocab > 0; }
        const engine_shape& shape () const { return shape_; }

        // shared_table: whether the input reads its table from the head, or has one of its own.
        void load (byte_reader& in, bool shared_table)
        {
            engine_shape s;
            s.vocab = in.i32(); s.dim = in.i32(); s.layers = in.i32(); s.heads = in.i32();
            s.kv_heads = in.i32(); s.head_dim = in.i32(); s.ffn = in.i32(); s.window = in.i32();
            if (s.vocab <= 0 || s.dim <= 0 || s.layers <= 0 || s.heads <= 0 || s.kv_heads <= 0
                || s.heads % s.kv_heads != 0 || s.head_dim % 2 != 0 || s.head_dim > 256 || s.ffn <= 0 || s.window <= 0)
                throw std::runtime_error("invalid network shape in the embedded data");
            eps = in.f32();
            inv_freq.resize(s.head_dim / 2);
            for (auto& f : inv_freq) f = in.f32();
            float input_scale = 0;
            if (shared_table) input_scale = in.f32();
            else emb = read_rows(in, s.vocab, s.dim);

            blocks.resize(s.layers);
            for (auto& b : blocks) {
                b.g1 = read_vector(in, s.dim);
                b.wq = read_int_matrix(in, s.dim, s.q_dim());
                b.wk = read_int_matrix(in, s.dim, s.kv_dim());
                b.wv = read_int_matrix(in, s.dim, s.kv_dim());
                b.wo = read_int_matrix(in, s.q_dim(), s.dim);
                b.g2 = read_vector(in, s.dim);
                b.wa = read_int_matrix(in, s.dim, s.ffn);
                b.ba = read_vector(in, s.ffn);
                b.wb = read_int_matrix(in, s.dim, s.ffn);
                b.bb = read_vector(in, s.ffn);
                b.wd = read_int_matrix(in, s.ffn, s.dim);
                b.bd = read_vector(in, s.dim);
            }
            gf = read_vector(in, s.dim);
            if (shared_table) {
                // The head, whose column t times the input scale is also the vector of token t.
                std::vector<float> head_scales(s.vocab);
                for (auto& f : head_scales) f = in.f32();
                std::vector<int8_t> head_values(static_cast<size_t>(s.dim) * s.vocab);
                for (auto& v : head_values) v = in.i8();
                emb.resize(static_cast<size_t>(s.vocab) * s.dim);
                for (int t = 0; t < s.vocab; ++t)
                    for (int i = 0; i < s.dim; ++i)
                        emb[static_cast<size_t>(t) * s.dim + i] = input_scale
                            * (static_cast<float>(head_values[static_cast<size_t>(i) * s.vocab + t]) * head_scales[t]);
                wout = pack_int(head_values, std::move(head_scales), s.dim, s.vocab);
            }
            else wout = read_int_matrix(in, s.dim, s.vocab);
            bout = read_vector(in, s.vocab);
            shape_ = s;
            wide = !wout.small;
            for (const auto& b : blocks)
                for (const int_matrix* m : { &b.wq, &b.wk, &b.wv, &b.wo, &b.wa, &b.wb, &b.wd })
                    wide = wide || !m->small;

            // Rotations of the relative positions 0 .. 2 * window - 1. The training layer forms
            // each angle in single precision, and so does this table.
            const int half = s.head_dim / 2;
            rope_cos.resize(static_cast<size_t>(2 * s.window) * half);
            rope_sin.resize(rope_cos.size());
            for (int p = 0; p < 2 * s.window; ++p)
                for (int i = 0; i < half; ++i) {
                    const float angle = static_cast<float>(p) * inv_freq[i];
                    double sn, cs;
                    det::sincos(static_cast<double>(angle), sn, cs);
                    rope_cos[static_cast<size_t>(p) * half + i] = static_cast<float>(cs);
                    rope_sin[static_cast<size_t>(p) * half + i] = static_cast<float>(sn);
                }
        }

        void start (int streams)
        {
            const engine_shape& s = shape_;
            const size_t per_stream = static_cast<size_t>(s.layers) * s.window * s.kv_dim();
            kraw.assign(per_stream * streams, 0.0f);
            krot.assign(per_stream * streams, 0.0f);
            vals.assign(per_stream * streams, 0.0f);
            position.assign(streams, 0);
            base.assign(streams, 0);
            adapted.assign(static_cast<size_t>(streams) * s.vocab, 0.0f);
        }

        /*
            Stream sid has coded token t with the distribution p the last step gave it. Each
            output bias of the stream moves against its probability, and that of t up by the
            same rate: a step of gradient descent on the coding cost, which follows the tokens
            the stream's own text uses. The decoder takes the same steps after the same tokens.
        */
        void learn (int sid, int t, const float* p)
        {
            const int V = shape_.vocab;
            float* b = &adapted[static_cast<size_t>(sid) * V];
            kernels::descend(b, p, ADAPT_RATE, V);
            b[t] += ADAPT_RATE;
        }

        // Feeds the newest token of each listed stream and writes its next-token distribution.
        void step (const int* ids, const int* tokens, int n, float* probs)
        {
            const engine_shape& s = shape_;
            const int D = s.dim, QD = s.q_dim(), KVD = s.kv_dim(), F = s.ffn, V = s.vocab, W = s.window;
            const int HD = s.head_dim, half = HD / 2, R = s.heads / s.kv_heads;
            reserve(n);

            for (int r = 0; r < n; ++r) {
                const int t = tokens[r];
                if (t < 0 || t >= V) throw std::runtime_error("token outside the network vocabulary");
                std::memcpy(&x[static_cast<size_t>(r) * D], &emb[static_cast<size_t>(t) * D], sizeof(float) * D);
                const int sid = ids[r];
                if (position[sid] - base[sid] == 2 * static_cast<int64_t>(W)) {
                    base[sid] += W;
                    rebase(sid);
                }
            }

            const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
            for (int l = 0; l < s.layers; ++l) {
                const block& b = blocks[l];
                rms_norm(x.data(), b.g1.data(), n, D, h.data());
                quantize(h.data(), n, D);
                gemm(n, b.wq, q.data());
                gemm(n, b.wk, k.data());
                gemm(n, b.wv, v.data());

                for (int r = 0; r < n; ++r) {
                    const int sid = ids[r];
                    const int64_t p = position[sid];
                    const int rel = static_cast<int>(p - base[sid]);
                    const int slot = static_cast<int>(p % W);
                    const int cnt = p + 1 < W ? static_cast<int>(p + 1) : W;
                    const size_t off = (static_cast<size_t>(sid) * s.layers + l) * W * KVD;
                    float* kr = &kraw[off];
                    float* kt = &krot[off];
                    float* vr = &vals[off];

                    // Store the new key, unrotated and rotated, and its value.
                    const float* kn = &k[static_cast<size_t>(r) * KVD];
                    std::memcpy(kr + static_cast<size_t>(slot) * KVD, kn, sizeof(float) * KVD);
                    std::memcpy(vr + static_cast<size_t>(slot) * KVD, &v[static_cast<size_t>(r) * KVD],
                        sizeof(float) * KVD);
                    float rotated[512];
                    for (int g = 0; g < s.kv_heads; ++g) rotate(kn + g * HD, rel, half, rotated + g * HD);
                    for (int i = 0; i < KVD; ++i) kt[static_cast<size_t>(i) * W + slot] = rotated[i];

                    // Scores of all the cached keys, accumulated over the head dimension, then the
                    // weighted values; the query heads of a key-value head read its keys and
                    // values together when there are two of them.
                    float* ctx_row = &ctx[static_cast<size_t>(r) * QD];
                    for (int g = 0; g < s.kv_heads; ++g) {
                        const float* keys = kt + static_cast<size_t>(g * HD) * W;
                        const float* values = vr + g * HD;
                        int hh = g * R;
                        if (R == 2) {
                            float q0[256], q1[256];
                            rotate(&q[static_cast<size_t>(r) * QD + hh * HD], rel, half, q0);
                            rotate(&q[static_cast<size_t>(r) * QD + (hh + 1) * HD], rel, half, q1);
                            float* sc0 = att.data();
                            float* sc1 = att.data() + W;
                            kernels::scores2(q0, q1, keys, W, HD, cnt, sc0, sc1);
                            kernels::attention_weights(sc0, cnt, scale);
                            kernels::attention_weights(sc1, cnt, scale);
                            kernels::weighted_sum2(sc0, sc1, values, KVD, cnt, HD, ctx_row + hh * HD,
                                ctx_row + (hh + 1) * HD);
                            continue;
                        }
                        for (; hh < (g + 1) * R; ++hh) {
                            float qr[256];
                            rotate(&q[static_cast<size_t>(r) * QD + hh * HD], rel, half, qr);
                            float* sc = att.data();
                            kernels::scores(qr, keys, W, HD, cnt, sc);
                            kernels::attention_weights(sc, cnt, scale);
                            kernels::weighted_sum(sc, values, KVD, cnt, HD, ctx_row + hh * HD);
                        }
                    }
                }

                quantize(ctx.data(), n, QD);
                gemm(n, b.wo, o.data());
                kernels::accumulate(x.data(), o.data(), nullptr, n * D);

                rms_norm(x.data(), b.g2.data(), n, D, h.data());
                quantize(h.data(), n, D);
                gemm(n, b.wa, fa.data());
                gemm(n, b.wb, fb.data());
                for (int r = 0; r < n; ++r)
                    kernels::swiglu(&fa[static_cast<size_t>(r) * F], &fb[static_cast<size_t>(r) * F], b.ba.data(),
                        b.bb.data(), F);
                quantize(fa.data(), n, F);
                gemm(n, b.wd, o.data());
                for (int r = 0; r < n; ++r)
                    kernels::accumulate(&x[static_cast<size_t>(r) * D], &o[static_cast<size_t>(r) * D], b.bd.data(), D);
            }

            rms_norm(x.data(), gf.data(), n, D, h.data());
            quantize(h.data(), n, D);
            gemm(n, wout, logits.data());
            for (int r = 0; r < n; ++r)
                kernels::distribution(&logits[static_cast<size_t>(r) * V], bout.data(),
                    &adapted[static_cast<size_t>(ids[r]) * V], V, probs + static_cast<size_t>(r) * V);
            for (int r = 0; r < n; ++r) ++position[ids[r]];
        }

    private:
        struct block
        {
            std::vector<float> g1, g2, ba, bb, bd;
            int_matrix wq, wk, wv, wo, wa, wb, wd;
        };

        static std::vector<float> read_vector (byte_reader& in, int n)
        {
            std::vector<float> v(n);
            for (auto& f : v) f = in.f32();
            return v;
        }

        // A weight matrix of the blocks or of the head: one scale per column, then the integers row by row.
        static int_matrix read_int_matrix (byte_reader& in, int rows, int cols)
        {
            std::vector<float> scales(cols);
            for (auto& f : scales) f = in.f32();
            std::vector<int8_t> m(static_cast<size_t>(rows) * cols);
            for (auto& v : m) v = in.i8();
            return pack_int(m, std::move(scales), rows, cols);
        }

        // An input table of its own: one scale per row, then the integers row by row.
        static std::vector<float> read_rows (byte_reader& in, int rows, int cols)
        {
            std::vector<float> scales(rows);
            for (auto& f : scales) f = in.f32();
            std::vector<float> m(static_cast<size_t>(rows) * cols);
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    m[static_cast<size_t>(r) * cols + c] = static_cast<float>(in.i8()) * scales[r];
            return m;
        }

        void reserve (int n)
        {
            const engine_shape& s = shape_;
            const size_t rows = static_cast<size_t>(n);
            if (x.size() >= rows * s.dim) return;
            x.resize(rows * s.dim); h.resize(rows * s.dim); o.resize(rows * s.dim);
            q.resize(rows * s.q_dim()); ctx.resize(rows * s.q_dim());
            k.resize(rows * s.kv_dim()); v.resize(rows * s.kv_dim());
            fa.resize(rows * s.ffn); fb.resize(rows * s.ffn);
            logits.resize(rows * s.vocab);
            att.resize(2 * static_cast<size_t>(s.window));
        }

        // Rotates every cached key of a stream to its position relative to the new base.
        void rebase (int sid)
        {
            const engine_shape& s = shape_;
            const int W = s.window, KVD = s.kv_dim(), HD = s.head_dim;
            const int64_t p = position[sid];
            for (int l = 0; l < s.layers; ++l) {
                const size_t off = (static_cast<size_t>(sid) * s.layers + l) * W * KVD;
                for (int64_t pj = std::max<int64_t>(0, p - W + 1); pj < p; ++pj) {
                    const int slot = static_cast<int>(pj % W);
                    float rotated[512];
                    for (int g = 0; g < s.kv_heads; ++g)
                        rotate(&kraw[off + static_cast<size_t>(slot) * KVD + g * HD], static_cast<int>(pj - base[sid]),
                            HD / 2, rotated + g * HD);
                    for (int i = 0; i < KVD; ++i) krot[off + static_cast<size_t>(i) * W + slot] = rotated[i];
                }
            }
        }

        void rms_norm (const float* in, const float* gain, int n, int d, float* out) const
        {
            for (int r = 0; r < n; ++r) {
                const float* row = in + static_cast<size_t>(r) * d;
                const float inv = 1.0f / std::sqrt(kernels::dot(row, row, d) / static_cast<float>(d) + eps);
                kernels::scale_by_gain(row, inv, gain, d, out + static_cast<size_t>(r) * d);
            }
        }

        // Interleaved pairs (2i, 2i+1) rotated by the angles of relative position rel.
        void rotate (const float* src, int rel, int half, float* dst) const
        {
            const float* cs = &rope_cos[static_cast<size_t>(rel) * half];
            const float* sn = &rope_sin[static_cast<size_t>(rel) * half];
            for (int i = 0; i < half; ++i) {
                const float x0 = src[2 * i], x1 = src[2 * i + 1];
                dst[2 * i] = x0 * cs[i] - x1 * sn[i];
                dst[2 * i + 1] = x0 * sn[i] + x1 * cs[i];
            }
        }

        /*
            The inputs of the next products, each row quantized to integers in [-127, 127] with
            the scale of its largest magnitude, kept in the form the processor's path reads.
        */
        void quantize (const float* in, int n, int kdim)
        {
            const int kpad = (kdim + 3) & ~3;
            xq.assign(static_cast<size_t>(n) * kpad, 0);
            xs.resize(n);
            for (int r = 0; r < n; ++r) {
                const float* row = in + static_cast<size_t>(r) * kdim;
                const float mx = kernels::max_abs(row, kdim);
                xs[r] = mx / 127.0f;
                if (!(mx > 0.0f)) continue;
                kernels::round_to_bytes(row, 127.0f / mx, kdim, &xq[static_cast<size_t>(r) * kpad]);
            }
            // The byte products read the inputs as unsigned bytes, the AVX2 ones of 8-bit weights as 16-bit integers.
            if (vnni_paths() || kernels::vector_paths()) {
                xu.resize(xq.size());
                kernels::to_unsigned(xq.data(), xq.size(), xu.data());
            }
            if (!vnni_paths() && kernels::vector_paths() && wide) {
                x16.resize(xq.size());
                for (size_t i = 0; i < xq.size(); ++i) x16[i] = xq[i];
            }
        }

        /*
            out[n x cols] = in[n x rows] * w for the inputs last quantized, which must have
            w.rows columns. The integer sums are exact in any order, so every path gives the
            same; each is then scaled by its row's and its column's scales, in that order.
        */
        void gemm (int n, const int_matrix& w, float* out)
        {
            acc.resize(static_cast<size_t>(n) * w.cols);
#if SLM_CODEC_X86
#if SLM_VNNI_KERNELS
            if (processor().avx512_vnni) kernels_int::vnni512(xu.data(), n, w, acc.data());
            else if (processor().avx_vnni) kernels_int::vnni256(xu.data(), n, w, acc.data());
            else
#endif
            if (kernels::vector_paths()) {
                if (w.small) kernels_int::avx2_bytes(xu.data(), n, w, acc.data());
                else kernels_int::avx2(x16.data(), n, w, acc.data());
            }
            else kernels_int::portable(xq.data(), n, w, acc.data());
#else
            kernels_int::portable(xq.data(), n, w, acc.data());
#endif
            for (int r = 0; r < n; ++r)
                kernels::rescale(&acc[static_cast<size_t>(r) * w.cols], xs[r], w.scale.data(), w.cols,
                    out + static_cast<size_t>(r) * w.cols);
        }

        engine_shape shape_;
        float eps = 1e-5f;
        std::vector<float> inv_freq, rope_cos, rope_sin;
        std::vector<float> emb, gf, bout;
        int_matrix wout;
        std::vector<block> blocks;

        std::vector<float> kraw, krot, vals;
        std::vector<int64_t> position, base;
        std::vector<float> adapted;    // output biases learned while coding, per stream
        static constexpr float ADAPT_RATE = 0.1f;

        std::vector<float> x, h, o, q, k, v, ctx, fa, fb, logits, att;
        std::vector<int8_t> xq;        // the quantized inputs of the products
        std::vector<uint8_t> xu;
        std::vector<int16_t> x16;
        std::vector<float> xs;
        std::vector<int32_t> acc;
        bool wide = false;             // some weights exceed 63 in magnitude: the AVX2 path also needs x16
    };

    // ----------------------------------------------------------------------------------------
    // The compiled-in data

    /*
        The embedded data is one container: the tokenizer, then the network when there is one,
        compressed by the coder of this program in byte mode. Its checksum identifies the
        build: an archive records the checksum of the data it was coded with, and a build
        holding other data refuses to decode it.

            "TQC2"  u32 segmentation  u32 folded capitals  u32 merges  merges x (u16 left, u16 right)
            u32 network (0: none, 1: a network with an input table of its own, 2: a network
                whose input table is its head)
            [network: shape, epsilon, rotation frequencies, the input table in form 1 or its
                scale in form 2, the tensors of the blocks and of the head]

        A container of the earlier format "TQC1" has no field for the folded capitals. The
        training of this program writes form 2; form 1 is read for the networks trained by
        its earlier versions.
    */
    struct codec_data
    {
        int segmentation = 1;    // the value of dlib::bpe_segmentation
        bool folded = false;     // whether the tokens are those of the text with folded capitals
        dlib::bpe_tokenizer tokenizer;
        vocabulary vocab;
        transformer_engine engine;
        uint32_t fingerprint = 0;
        size_t container_size = 0;
        bool has_tokenizer () const { return vocab.size() > 256; }
        bool has_network () const { return engine.loaded(); }
        // Text is coded in tokens with a network, in bytes otherwise; capitals fold in token mode.
        bool token_mode () const { return has_network() && has_tokenizer(); }
        bool folds () const { return folded && token_mode(); }
    };

    // The names of the segmentations on the command line and in the reports.
    inline const char* segmentation_name (int segmentation)
    {
        switch (segmentation) {
            case 0: return "whitespace";
            case 1: return "spaces";
            case 2: return "classes";
            default: return "unknown";
        }
    }

    // How the tokenizer is described in the reports and the generated header.
    inline std::string tokenizer_form (int segmentation, bool folded)
    {
        return std::string("segmentation ") + segmentation_name(segmentation) + (folded ? ", folded capitals" : "");
    }

    inline void load_codec_data (const std::vector<uint8_t>& raw, codec_data& data)
    {
        byte_reader in(raw.data(), raw.size());
        const uint32_t format = in.u32();
        if (format != 0x31435154u && format != 0x32435154u)    // "TQC1", "TQC2"
            throw std::runtime_error("unknown embedded data format");
        data.segmentation = in.i32();
        if (data.segmentation < 0 || data.segmentation > 2)
            throw std::runtime_error("unknown segmentation in the embedded data");
        const uint32_t folded = format == 0x32435154u ? in.u32() : 0;
        if (folded > 1) throw std::runtime_error("unknown text form in the embedded data");
        data.folded = folded == 1;
        std::vector<std::pair<int, int>> merges(in.u32());
        for (auto& m : merges) {
            m.first = in.u16();
            m.second = in.u16();
        }
        data.tokenizer.set_segmentation(static_cast<dlib::bpe_segmentation>(data.segmentation));
        data.tokenizer.set_merges(merges);
        data.vocab.build(data.tokenizer);
        const uint32_t network = in.u32();
        if (network > 2) throw std::runtime_error("unknown network form in the embedded data");
        if (network != 0) data.engine.load(in, network == 2);
        if (!in.done()) throw std::runtime_error("trailing bytes in the embedded data");
        data.fingerprint = checksum(raw.data(), raw.size());
    }

    // ----------------------------------------------------------------------------------------
    // Coding

    /*
        Archive layout, little-endian:

            "DLC4"  u64 original size  u32 CRC-32 of the original  u32 data fingerprint
            u8 mode (0: bytes, 1: tokens)  u8 log2 of the table units per model  u8 streams
            u8 options (bit 0: the pages of a wiki dump coded in another order)
            per stream: varint byte length, and in token mode varint token count
            arithmetic code

        The archives of the earlier format "DLC3" were coded with other context models.
    */
    const uint32_t ARCHIVE_MAGIC = 0x34434C44u;    // "DLC4"
    const uint32_t ARCHIVE_MAGIC_V3 = 0x33434C44u;
    const uint8_t PAGES_REORDERED = 1;

    inline void put_u32 (std::vector<uint8_t>& out, uint32_t v)
    {
        for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }

    inline void put_varint (std::vector<uint8_t>& out, uint64_t v)
    {
        while (v >= 128) { out.push_back(static_cast<uint8_t>(v | 128)); v >>= 7; }
        out.push_back(static_cast<uint8_t>(v));
    }

    class archive_reader
    {
    public:
        explicit archive_reader (const std::vector<uint8_t>& data_) : data(data_) {}
        uint8_t u8 () { need(1); return data[pos++]; }
        uint32_t u32 ()
        {
            uint32_t v = 0;
            for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(u8()) << (8 * i);
            return v;
        }
        uint64_t u64 () { const uint64_t lo = u32(); return lo | (static_cast<uint64_t>(u32()) << 32); }
        uint64_t varint ()
        {
            uint64_t v = 0;
            for (int shift = 0; shift < 64; shift += 7) {
                const uint8_t b = u8();
                v |= static_cast<uint64_t>(b & 127) << shift;
                if (!(b & 128)) return v;
            }
            throw std::runtime_error("corrupt archive header");
        }
        size_t position () const { return pos; }
    private:
        void need (size_t n) const { if (pos + n > data.size()) throw std::runtime_error("truncated archive"); }
        const std::vector<uint8_t>& data;
        size_t pos = 0;
    };

    // Codes one decision, in either direction.
    struct encoding_side
    {
        bit_encoder& coder;
        int operator() (predictor& model, int s, int code, int nn_st, bool nn_valid, int bit)
        {
            coder.encode(bit, model.predict(s, code, nn_st, nn_valid));
            model.update(bit);
            return bit;
        }
    };

    struct decoding_side
    {
        bit_decoder& coder;
        int operator() (predictor& model, int s, int code, int nn_st, bool nn_valid, int)
        {
            const int bit = coder.decode(model.predict(s, code, nn_st, nn_valid));
            model.update(bit);
            return bit;
        }
    };

    // The logistic form of m1 / (m0 + m1), the network's probability of the bit-1 half.
    inline int split_to_st (double m0, double m1)
    {
        if (!(m1 > 0.0)) return -ST_LIMIT;
        if (!(m0 > 0.0)) return ST_LIMIT;
        const double st = 256.0 * (det::log(m1) - det::log(m0));
        if (st >= ST_LIMIT) return ST_LIMIT;
        if (st <= -ST_LIMIT) return -ST_LIMIT;
        return static_cast<int>(det::round_even(st));
    }

    /*
        Codes one token of stream s: the bits of its bytes and, where the vocabulary allows
        both, whether it ends. cum holds the network's probabilities summed in lexicographic
        order of the tokens. token is the token to encode, or -1 when decoding.
    */
    template <typename SIDE>
    int code_token (SIDE& side, predictor& model, const vocabulary& voc, int s, const double* cum, int token)
    {
        const uint8_t* pat = token >= 0 ? voc.pattern(token) : nullptr;
        const int len = token >= 0 ? voc.length(token) : 0;
        int lo = 0, hi = voc.size(), depth = 0;
        for (;;) {
            if (depth > 0) {
                const bool can_end = voc.length(voc.token_at(lo)) == depth;
                const bool can_go = hi - lo > (can_end ? 1 : 0);
                if (can_end && can_go) {
                    const int st = split_to_st(cum[hi] - cum[lo + 1], cum[lo + 1] - cum[lo]);
                    if (side(model, s, 256 + depth, st, true, len == depth ? 1 : 0)) break;
                    ++lo;
                }
                else if (can_end) {
                    break;
                }
            }
            int c0 = 1;
            for (int b = 7; b >= 0; --b) {
                const int known = 7 - b;
                const int value = ((c0 - (1 << known)) << (b + 1)) | (1 << b);
                const int mid = voc.lower_bound(lo, hi, depth, value);
                int bit;
                if (mid == lo) bit = 1;
                else if (mid == hi) bit = 0;
                else bit = side(model, s, c0, split_to_st(cum[mid] - cum[lo], cum[hi] - cum[mid]), true,
                    pat ? (pat[depth] >> b) & 1 : 0);
                if (bit) lo = mid; else hi = mid;
                c0 = (c0 << 1) | bit;
            }
            model.byte_done(s, static_cast<uint8_t>(c0 & 0xff), true);
            ++depth;
        }
        const int t = voc.token_at(lo);
        model.token_done(s, t);
        return t;
    }

    template <typename SIDE>
    uint8_t code_byte (SIDE& side, predictor& model, int s, int byte)
    {
        int c0 = 1;
        for (int b = 7; b >= 0; --b) c0 = (c0 << 1) | side(model, s, c0, 0, false, (byte >> b) & 1);
        model.byte_done(s, static_cast<uint8_t>(c0 & 0xff), false);
        return static_cast<uint8_t>(c0 & 0xff);
    }

    // Statistics of a coding run, for the report.
    struct coding_stats
    {
        double network_bits = 0;    // what the network alone would have cost
        uint64_t tokens = 0;
        int contexts = 0;           // context models
        size_t model_bytes = 0;     // what their tables hold
    };

    /*
        Runs the token streams side by side. tokens[s] is read when encoding; when decoding the
        tokens are not kept, the bytes being in the stream histories. counts[s] is the number
        of tokens of each stream.
    */
    template <typename SIDE>
    void run_token_streams (SIDE& side, codec_data& data, int table_bits,
        const std::vector<std::vector<uint16_t>>& tokens, const std::vector<uint64_t>& counts,
        std::vector<stream_state>& streams, coding_stats& stats, bool encoding, void (*progress)(uint64_t, uint64_t))
    {
        const int num = static_cast<int>(counts.size());
        const int V = data.engine.shape().vocab;
        const int n_voc = data.vocab.size();
        // Streams start at the beginning of a line, so the network is first given a line break,
        // a token it has seen in that role throughout training.
        const int line_break = '\n';
        predictor model;
        model.init(table_bits, &streams);
        stats.contexts = model.contexts();
        stats.model_bytes = model.memory();
        data.engine.start(num);

        std::vector<float> probs(static_cast<size_t>(num) * V);
        std::vector<double> cum(n_voc + 1);
        std::vector<int> ids(num), feed(num), prev(num, line_break);
        std::vector<uint64_t> done(num, 0);
        uint64_t total = 0, coded = 0;
        for (uint64_t c : counts) total += c;

        while (coded < total) {
            int n = 0;
            for (int s = 0; s < num; ++s)
                if (done[s] < counts[s]) { ids[n] = s; feed[n] = prev[s]; ++n; }
            data.engine.step(ids.data(), feed.data(), n, probs.data());
            for (int i = 0; i < n; ++i) {
                const int s = ids[i];
                const float* p = &probs[static_cast<size_t>(i) * V];
                cum[0] = 0.0;
                for (int r = 0; r < n_voc; ++r) cum[r + 1] = cum[r] + p[data.vocab.token_at(r)];
                // The network's most likely token and what it would cost, which the context models use.
                const int top_id = kernels::first_max(p, V);
                const double top_bits =
                    -det::log(std::max(1e-30, static_cast<double>(p[top_id]) / cum[n_voc])) / 0.69314718055994530942;
                model.network_top(s, top_id, static_cast<int>(std::min(31.0, det::round_even(4.0 * top_bits))));
                const int want = encoding ? tokens[s][done[s]] : -1;
                const int t = code_token(side, model, data.vocab, s, cum.data(), want);
                const double pt = std::max(1e-30, static_cast<double>(p[t]) / cum[n_voc]);
                stats.network_bits -= det::log(pt) / 0.69314718055994530942;
                data.engine.learn(s, t, p);
                prev[s] = t;
                ++done[s];
                ++coded;
            }
            if (progress && (coded & 0x3ff) < static_cast<uint64_t>(n)) progress(coded, total);
        }
        stats.tokens = total;
    }

    // Decodes an archive. Nothing here depends on dlib.
    inline std::vector<uint8_t> decompress (const std::vector<uint8_t>& archive, codec_data& data,
        void (*progress)(uint64_t, uint64_t))
    {
        archive_reader in(archive);
        const uint32_t magic = in.u32();
        if (magic == ARCHIVE_MAGIC_V3) throw std::runtime_error("an archive of an earlier version of this program");
        if (magic != ARCHIVE_MAGIC) throw std::runtime_error("not an archive of this program");
        const uint64_t original_size = in.u64();
        const uint32_t crc = in.u32();
        const uint32_t fingerprint = in.u32();
        const int mode = in.u8();
        const int table_bits = in.u8();
        const int num = in.u8();
        const int options = in.u8();
        if (fingerprint != data.fingerprint)
            throw std::runtime_error("this archive was made with another tokenizer or network than this build's");
        if (mode > 1 || num < 1 || table_bits < 16 || table_bits > 24
            || (options & ~PAGES_REORDERED) != 0)
            throw std::runtime_error("corrupt archive header");
        if (mode == 1 && !data.has_network()) throw std::runtime_error("the archive needs the compiled-in network");

        std::vector<uint64_t> lengths(num), counts(num, 0);
        for (int s = 0; s < num; ++s) {
            lengths[s] = in.varint();
            if (mode == 1) counts[s] = in.varint();
        }
        bit_decoder coder(archive.data() + in.position(), archive.data() + archive.size());
        decoding_side side{ coder };
        std::vector<stream_state> streams(num);
        std::vector<uint8_t> out;
        out.reserve(static_cast<size_t>(original_size));

        if (mode == 0) {
            predictor model;
            model.init(table_bits, &streams);
            for (uint64_t i = 0; i < lengths[0]; ++i) {
                out.push_back(code_byte(side, model, 0, 0));
                if (progress && (i & 0xffff) == 0) progress(i, lengths[0]);
            }
        }
        else {
            const std::vector<std::vector<uint16_t>> no_tokens(num);
            coding_stats stats;
            run_token_streams(side, data, table_bits, no_tokens, counts, streams, stats, false, progress);
            for (int s = 0; s < num; ++s) {
                if (streams[s].hist.size() != lengths[s]) throw std::runtime_error("stream length mismatch");
                out.insert(out.end(), streams[s].hist.begin(), streams[s].hist.end());
                std::vector<uint8_t>().swap(streams[s].hist);
            }
            if (data.folded) out = unfold_capitals(out.data(), out.size());
        }
        if (options & PAGES_REORDERED) out = restore_page_order(out);
        if (out.size() != original_size) throw std::runtime_error("decoded size differs from the recorded size");
        if (checksum(out.data(), out.size()) != crc)
            throw std::runtime_error("CRC-32 mismatch: the output is not the original");
        return out;
    }

#if !SLM_DECOMPRESSOR_ONLY

    // ----------------------------------------------------------------------------------------
    // Encoding

    /*
        How a text is coded: in tokens when a network is compiled in, in bytes otherwise; with
        tables that grow with the input, the largest of them one four-byte unit for 32 to 64
        bytes of text, up to 2^24 units; and, in token mode, in streams cut at line boundaries,
        each of at least a megabyte. The text is the input itself, or in token mode its form
        with folded capitals when the tokenizer was built on that form.
    */
    struct coding_plan
    {
        bool token_mode = false;
        int table_bits = 0;
        uint8_t options = 0;            // PAGES_REORDERED when the pages are coded in another order
        uint32_t crc = 0;               // CRC-32 of the input as given, before any transform
        std::vector<size_t> bounds;     // stream s covers [bounds[s], bounds[s + 1]) of the coded text
        int streams () const { return static_cast<int>(bounds.size()) - 1; }
    };

    inline coding_plan plan_coding (const std::vector<uint8_t>& text, const codec_data& data, int streams_wanted,
        int table_bits)
    {
        coding_plan plan;
        plan.token_mode = data.token_mode();
        plan.table_bits = table_bits;
        if (plan.table_bits <= 0) {
            plan.table_bits = 16;
            while (plan.table_bits < 24 && (size_t(1) << (plan.table_bits + 6)) < text.size()) ++plan.table_bits;
        }
        const int megabytes = static_cast<int>(std::min<size_t>(text.size() >> 20, 255));
        const int num = plan.token_mode ? std::max(1, std::min(streams_wanted, megabytes)) : 1;
        plan.bounds.assign(1, 0);
        for (int s = 1; s < num; ++s) {
            size_t cut = text.size() * s / num;
            while (cut < text.size() && text[cut - 1] != '\n') ++cut;
            if (cut > plan.bounds.back() && cut < text.size()) plan.bounds.push_back(cut);
        }
        plan.bounds.push_back(text.size());
        return plan;
    }

    /*
        Codes a text into an archive. original is the input, its pages in the order of the
        plan's options, and folded its form with folded capitals when the tokenizer folds them,
        empty otherwise; the form that is coded is the folded one when there is one. The archive
        records the size of the input and the checksum the plan holds, that of the input as
        given. In token mode both are released once tokenized, so that coding holds the tokens
        alone. Nothing here depends on dlib.
    */
    inline std::vector<uint8_t> compress (std::vector<uint8_t>& original, std::vector<uint8_t>& folded,
        codec_data& data, const coding_plan& plan, coding_stats& stats, size_t& header_size,
        void (*progress)(uint64_t, uint64_t))
    {
        const std::vector<uint8_t>& text = folded.empty() ? original : folded;
        const int num = plan.streams();
        std::vector<uint8_t> out;
        put_u32(out, ARCHIVE_MAGIC);
        put_u32(out, static_cast<uint32_t>(original.size()));
        put_u32(out, static_cast<uint32_t>(static_cast<uint64_t>(original.size()) >> 32));
        put_u32(out, plan.crc);
        put_u32(out, data.fingerprint);
        out.push_back(plan.token_mode ? 1 : 0);
        out.push_back(static_cast<uint8_t>(plan.table_bits));
        out.push_back(static_cast<uint8_t>(num));
        out.push_back(plan.options);

        std::vector<std::vector<uint16_t>> tokens(num);
        std::vector<uint64_t> counts(num, 0);
        if (plan.token_mode) {
            if (data.tokenizer.get_vocab_size() > 65536)
                throw std::runtime_error("coding tokens needs a tokenizer of at most 65536 entries");
            for (int s = 0; s < num; ++s) {
                const std::string part(text.begin() + plan.bounds[s], text.begin() + plan.bounds[s + 1]);
                const std::vector<int> ids = data.tokenizer.encode(part);
                tokens[s].assign(ids.begin(), ids.end());
                counts[s] = tokens[s].size();
            }
            std::vector<uint8_t>().swap(original);
            std::vector<uint8_t>().swap(folded);
        }
        for (int s = 0; s < num; ++s) {
            put_varint(out, plan.bounds[s + 1] - plan.bounds[s]);
            if (plan.token_mode) put_varint(out, counts[s]);
        }
        header_size = out.size();

        bit_encoder coder(out);
        encoding_side side{ coder };
        std::vector<stream_state> streams(num);
        if (plan.token_mode) {
            run_token_streams(side, data, plan.table_bits, tokens, counts, streams, stats, true, progress);
        }
        else {
            predictor model;
            model.init(plan.table_bits, &streams);
            stats.contexts = model.contexts();
            stats.model_bytes = model.memory();
            for (size_t i = 0; i < text.size(); ++i) {
                code_byte(side, model, 0, text[i]);
                if (progress && (i & 0xffff) == 0) progress(i, text.size());
            }
        }
        coder.flush();
        return out;
    }

#endif

    // ----------------------------------------------------------------------------------------
    // Container coding

    /*
        The embedded container is mostly quantized weights, whose bytes depend on little but
        their own distribution, and 32-bit fields. It is coded with a small model of its own:
        the byte so far, the previous one and the previous two, each also keyed by the position
        modulo four, mixed and refined. The decompressor reads its data with no other code than
        the coder it already holds.
    */
    class container_model
    {
    public:
        container_model ()
        {
            order0.assign(4 * 256, COUNTER_INIT);
            order1.assign(4 * 65536, COUNTER_INIT);
            order2.assign(size_t(1) << 22, COUNTER_INIT);
            mix.init(4, 4 * 8, 20, 21845);
            map.init(4 * 256, 7);
        }

        int predict (int c0)
        {
            const logistic& lg = logistic_tables();
            const int lane = static_cast<int>(pos & 3);
            slots[0] = &order0[lane * 256 + c0];
            slots[1] = &order1[(static_cast<size_t>(lane) << 16) | (c1 << 8) | c0];
            const uint64_t h2 = hash64(static_cast<uint64_t>(c1 | (c2 << 8) | (lane << 16)), static_cast<uint64_t>(c0));
            slots[2] = &order2[h2 >> 42];
            int* x = mix.input();
            for (int i = 0; i < 3; ++i) x[i] = lg.stretch(counter_p16(*slots[i]));
            x[3] = 256;
            int known = 0;
            for (int c = c0; c > 1; c >>= 1) ++known;
            const int st = mix.mix(lane * 8 + known);
            const int p = (mix.p16() + map.refine(st, lane * 256 + c0)) >> 1;
            return p < 1 ? 1 : (p > 65535 ? 65535 : p);
        }

        void update (int bit)
        {
            counter_update(*slots[0], bit, 1023);
            counter_update(*slots[1], bit, 1023);
            counter_update(*slots[2], bit, 255);
            mix.update(bit);
            map.update(bit);
        }

        void byte_done (int byte) { c2 = c1; c1 = byte; ++pos; }

    private:
        std::vector<uint32_t> order0, order1, order2;
        uint32_t* slots[3];
        mixer mix;
        apm map;
        uint64_t pos = 0;
        int c1 = 0, c2 = 0;
    };

    inline std::vector<uint8_t> pack_container (const std::vector<uint8_t>& raw)
    {
        std::vector<uint8_t> out;
        put_u32(out, static_cast<uint32_t>(raw.size()));
        bit_encoder coder(out);
        container_model model;
        for (uint8_t b : raw) {
            int c0 = 1;
            for (int k = 7; k >= 0; --k) {
                const int bit = (b >> k) & 1;
                coder.encode(bit, model.predict(c0));
                model.update(bit);
                c0 = (c0 << 1) | bit;
            }
            model.byte_done(b);
        }
        coder.flush();
        return out;
    }

    inline std::vector<uint8_t> unpack_container (const uint8_t* packed, size_t packed_size)
    {
        if (packed_size < 4) return std::vector<uint8_t>();
        const uint32_t size = packed[0] | (packed[1] << 8) | (packed[2] << 16)
            | (static_cast<uint32_t>(packed[3]) << 24);
        bit_decoder coder(packed + 4, packed + packed_size);
        container_model model;
        std::vector<uint8_t> raw(size);
        for (uint32_t i = 0; i < size; ++i) {
            int c0 = 1;
            for (int k = 0; k < 8; ++k) {
                const int bit = coder.decode(model.predict(c0));
                model.update(bit);
                c0 = (c0 << 1) | bit;
            }
            raw[i] = static_cast<uint8_t>(c0 & 0xff);
            model.byte_done(raw[i]);
        }
        return raw;
    }

    // The container compiled into this build, decoded on first use.
    inline codec_data& embedded_codec_data ()
    {
        static codec_data data;
        static bool ready = false;
        if (!ready) {
            ready = true;
#ifdef WITH_EMBEDDED_CODEC_DATA
            data.container_size = embedded_codec::packed_size;
            if (data.container_size > 0)
                load_codec_data(unpack_container(embedded_codec::packed, embedded_codec::packed_size), data);
#endif
        }
        return data;
    }
}

// ----------------------------------------------------------------------------------------
// File helpers

static std::vector<uint8_t> read_file (const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void write_file (const std::string& path, const std::vector<uint8_t>& bytes)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot create " + path);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!f) throw std::runtime_error("cannot write " + path);
}

static std::chrono::steady_clock::time_point progress_start;

static void print_progress (uint64_t done, uint64_t total)
{
    static double last_shown = -1.0;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - progress_start).count();
    if (secs < last_shown + 1.0 && secs >= last_shown) return;
    last_shown = secs;
    const double frac = total ? static_cast<double>(done) / total : 1.0;
    const double eta = frac > 0 ? secs * (1.0 - frac) / frac : 0.0;
    std::cout << "\r  " << std::fixed << std::setprecision(1) << 100.0 * frac << "%  "
        << static_cast<long>(secs) << " s elapsed, " << static_cast<long>(eta) << " s left      " << std::flush;
}

static int decompress_command (const std::string& input, const std::string& output)
{
    tok_codec::codec_data& data = tok_codec::embedded_codec_data();
    const std::vector<uint8_t> archive = read_file(input);
    progress_start = std::chrono::steady_clock::now();
    const std::vector<uint8_t> restored = tok_codec::decompress(archive, data, print_progress);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - progress_start).count();
    write_file(output, restored);
    std::cout << "\r" << input << " (" << archive.size() << " bytes) -> " << output << " (" << restored.size()
        << " bytes), CRC-32 verified, " << std::fixed << std::setprecision(1) << secs << " s            " << std::endl;
    return 0;
}

#if !SLM_DECOMPRESSOR_ONLY

static int compress_command (const std::string& input, const std::string& output, int streams_wanted, int table_bits,
    bool keep_order)
{
    tok_codec::codec_data& data = tok_codec::embedded_codec_data();
    std::vector<uint8_t> original = read_file(input);
    const uint32_t crc = tok_codec::checksum(original.data(), original.size());
    const size_t pages = keep_order ? 0 : tok_codec::order_pages(original);
    std::vector<uint8_t> folded;
    if (data.folds()) folded = tok_codec::fold_capitals<std::vector<uint8_t>>(original.data(), original.size());
    tok_codec::coding_plan plan = tok_codec::plan_coding(data.folds() ? folded : original, data, streams_wanted,
        table_bits);
    plan.crc = crc;
    plan.options = pages > 0 ? tok_codec::PAGES_REORDERED : 0;
    const size_t n = std::max<size_t>(1, original.size());
    std::cout << "=== Compression ===\n  input   : " << input << " (" << original.size() << " bytes)\n"
        << "  mode    : " << (plan.token_mode ? "tokens, network and context models" : "bytes, context models only")
        << (data.folds() ? ", folded capitals" : "")
        << "\n  streams : " << plan.streams() << "\n  tables  : 2^" << plan.table_bits << " four-byte units per model";
    if (pages > 0) std::cout << "\n  pages   : " << pages << ", coded grouped by category";
    const std::string& brand = dlib::get_cpu_features().brand;
    std::cout << "\n  kernels : " << (brand.empty() ? std::string("this processor") : brand) << ", products in "
        << tok_codec::product_kernels() << std::endl;

    progress_start = std::chrono::steady_clock::now();
    tok_codec::coding_stats stats;
    size_t header_size = 0;
    const std::vector<uint8_t> out = tok_codec::compress(original, folded, data, plan, stats, header_size,
        print_progress);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - progress_start).count();
    write_file(output, out);

    std::cout << "\r  time    : " << std::fixed << std::setprecision(1) << secs << " s, " << std::setprecision(2)
        << 1e6 * secs / n << " us/byte                         \n"
        << "  archive : " << out.size() << " bytes, " << std::setprecision(4) << 8.0 * (out.size() - header_size) / n
        << " bits per byte" << std::endl;
    if (plan.token_mode)
        std::cout << "  network alone : " << std::setprecision(0) << stats.network_bits / 8 << " bytes, "
            << std::setprecision(4) << stats.network_bits / n << " bits per byte, " << stats.tokens << " tokens"
            << std::endl;
    std::cout << "  context models : " << stats.contexts << ", tables of " << std::setprecision(0)
        << stats.model_bytes / 1048576.0 << " MB" << std::endl;
    std::cout << "  compiled-in data : " << data.container_size << " bytes" << std::endl;
    return 0;
}

#endif

#if SLM_TRAINING

using namespace dlib;
using std::cout;
using std::endl;

// ----------------------------------------------------------------------------------------
// The network

/*
    The constants of the network, which can be set at compile time. The engine reads the shape
    from the compiled-in data, so that a build with other constants still decodes what an
    earlier one coded.
*/
#ifndef SLM_VOCAB_SIZE
#define SLM_VOCAB_SIZE 8192
#endif
#ifndef SLM_NUM_LAYERS
#define SLM_NUM_LAYERS 8
#endif
#ifndef SLM_NUM_HEADS
#define SLM_NUM_HEADS 4
#endif
#ifndef SLM_NUM_KV_HEADS
#define SLM_NUM_KV_HEADS 2
#endif
#ifndef SLM_EMBEDDING_DIM
#define SLM_EMBEDDING_DIM 256
#endif
#ifndef SLM_WINDOW
#define SLM_WINDOW 256
#endif

namespace dlib
{
    /*
        WHAT THIS OBJECT REPRESENTS
            An input layer that reads its table from the weights of the network's output layer:
            the vector of token t is column t of the head's weight matrix, times a learned scale,
            the only parameter of the layer. A network built with it holds one table of tokens
            instead of two.

            The head runs its backward pass before this layer, which adds the gradient of the
            inputs to the head's gradient; the solvers then move the shared weights with both.
            tie() gives the layer the head's parameters and their gradient. A copy of the layer
            is untied, while a layer deserialized in place keeps its references, which is what
            a trainer reloading its synchronization file needs.
    */
    template <unsigned long num_tokens, unsigned long dim>
    class tied_embeddings_
    {
    public:
        tied_embeddings_ () = default;
        tied_embeddings_ (const tied_embeddings_& other) : params(other.params) {}
        tied_embeddings_& operator= (const tied_embeddings_& other)
        {
            params = other.params;
            head = nullptr;
            head_grad = nullptr;
            return *this;
        }

        // weights: the parameters of the head, [dim + 1 x num_tokens], its biases in the last row.
        void tie (tensor& weights, tensor& weights_grad)
        {
            head = &weights;
            head_grad = &weights_grad;
        }

        bool is_tied () const { return head != nullptr; }

        float scale () const { return params.size() > 0 ? static_cast<const tensor&>(params).host()[0] : 1.0f; }

        template <typename SUBNET>
        void setup (const SUBNET&)
        {
            /* The head's weights start with a deviation of sqrt(2 / (dim + num_tokens)), which
               this scale brings to one, the deviation of an ordinary table at its start. */
            params.set_size(1);
            params = std::sqrt((dim + num_tokens) / 2.0f);
        }

        template <typename SUBNET>
        void forward (const SUBNET& sub, resizable_tensor& output)
        {
            DLIB_CASSERT(head != nullptr, "the input table is not tied to the head: call tie() first");
            const tensor& prev = sub.get_output();
            /* The head is set up by the first forward pass, after this layer: until then the
               inputs are zero. */
            read_head = head->size() >= num_tokens * dim;
            if (!read_head) {
                output.set_size(prev.num_samples(), prev.k(), prev.nr(), dim);
                output = 0;
                return;
            }
            gather(prev);
            output.copy_size(gathered);
            tt::affine_transform(output, gathered, scale());
        }

        template <typename SUBNET>
        void backward (const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            params_grad = 0;
            if (!read_head || head_grad == nullptr || head_grad->size() != head->size()) return;
            const tensor& prev = sub.get_output();
            // The scale: the gradients against the vectors read, gathered again.
            gather(prev);
            tt::dot(gradient_input, gathered, params_grad, 0);
            // The weights: each column receives the gradients of the positions holding its token.
            if (freqs.size() != num_tokens) {
                freqs.set_size(num_tokens);
                freqs = 0;
            }
            table_grad.set_size(num_tokens, dim);
            table_grad = 0;
            tt::embeddings_gradient(prev, gradient_input, table_grad, freqs, -scale(), false);
            alias_tensor columns(1, 1, num_tokens, dim), rows(1, 1, dim, num_tokens);
            auto grad_rows = rows(*head_grad, 0);
            tt::transpose(true, grad_rows, columns(table_grad, 0));
        }

        const tensor& get_layer_params () const { return params; }
        tensor& get_layer_params () { return params; }

        void clean ()
        {
            table.clear();
            gathered.clear();
            table_grad.clear();
        }

        friend void serialize (const tied_embeddings_& item, std::ostream& out)
        {
            serialize("tied_embeddings_", out);
            serialize(item.params, out);
        }

        friend void deserialize (tied_embeddings_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "tied_embeddings_")
                throw serialization_error("Unexpected version '" + version
                    + "' found while deserializing tied_embeddings_.");
            deserialize(item.params, in);
        }

        friend std::ostream& operator<< (std::ostream& out, const tied_embeddings_&)
        {
            out << "tied_embeddings\t (num_tokens=" << num_tokens << ", dim=" << dim << ")";
            return out;
        }

        friend void to_xml (const tied_embeddings_&, std::ostream& out)
        {
            out << "<tied_embeddings num_tokens='" << num_tokens << "' dim='" << dim << "'/>\n";
        }

    private:
        // The columns of the head as a table of tokens, and the rows of that table each position reads.
        void gather (const tensor& prev)
        {
            alias_tensor rows(1, 1, dim, num_tokens), columns(num_tokens, dim);
            table.set_size(1, 1, num_tokens, dim);
            tt::transpose(false, table, rows(*head, 0));
            gathered.set_size(prev.num_samples(), prev.k(), prev.nr(), dim);
            tt::embeddings(gathered, prev, columns(table, 0));
        }

        resizable_tensor params;
        tensor* head = nullptr;
        tensor* head_grad = nullptr;
        bool read_head = false;
        resizable_tensor table, gathered, table_grad, freqs;
    };

    template <unsigned long num_tokens, unsigned long dim, typename SUBNET>
    using tied_embeddings = add_layer<tied_embeddings_<num_tokens, dim>, SUBNET>;

    /*
        WHAT THIS OBJECT REPRESENTS
            The configuration of the network of the compressor, in the manner of the library's
            transformer configurations: the dimensions fixed at compile time and checked, the
            blocks spelled out, and network_type<is_training> giving the whole network.

            A token is read from the table of the head, through tied_embeddings, so that the
            vocabulary costs one matrix. Each block is pre-norm: an RMS normalisation, the
            fused grouped-query attention with rotary positions and a causal mask, a residual
            connection, then an RMS normalisation, a SwiGLU of 8/3 of the width and a residual
            connection. A final RMS normalisation and the head score every position.

            The decoder runs the network on one core for every token of the text and carries
            its weights, so each part of the structure is weighed by the bits it saves against
            the time it takes per token and the bytes its weights add to the decoder. The
            shared table saves the weights of one matrix of the vocabulary, at no cost per
            token. Adaptive computation time and the hierarchical recurrence of the library
            repeat blocks a varying number of times per token, and a mixture of experts
            multiplies the weights of each block; the time per token and the size of the
            decoder leave them out.

            network_type<true> and network_type<false> are the same type, as for the library's
            grouped-query configuration. tie() connects the input table to the head; it is
            called after a network is built, loaded or copied.
    */
    template <long vocab_size, long num_layers, long num_heads, long num_kv_heads, long embedding_dim>
    struct tok_compressor_config
    {
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long NUM_KV_HEADS = num_kv_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long HEAD_DIM = embedding_dim / num_heads;

        struct validation
        {
            static_assert(VOCAB_SIZE > 0 && NUM_LAYERS > 0, "the vocabulary and the depth must be positive");
            static_assert(NUM_HEADS > 0 && NUM_KV_HEADS > 0 && NUM_HEADS % NUM_KV_HEADS == 0,
                "the query heads must be a multiple of the key-value heads");
            static_assert(EMBEDDING_DIM % NUM_HEADS == 0 && HEAD_DIM % 2 == 0,
                "the width must split into heads of an even dimension, for the rotary positions");
        };

        template <typename SUBNET>
        using attention = add_prev1<gqa_attention<EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
            rms_norm<tag1<SUBNET>>>>;

        template <typename SUBNET>
        using feed_forward = add_prev5<swiglu<EMBEDDING_DIM, 8, 3, rms_norm<tag5<SUBNET>>>>;

        template <typename SUBNET>
        using block = feed_forward<attention<SUBNET>>;

        template <long remaining, typename SUBNET>
        struct stack_of
        {
            using type = block<typename stack_of<remaining - 1, SUBNET>::type>;
        };
        template <typename SUBNET>
        struct stack_of<0, SUBNET>
        {
            using type = SUBNET;
        };

        template <bool is_training>
        using network_type = loss_cross_entropy_per_token<linear<VOCAB_SIZE, rms_norm<typename stack_of<NUM_LAYERS,
            tied_embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>::type>>>;

        template <typename NET>
        static void tie (NET& net)
        {
            auto& head = layer<1>(net);
            layer<NET::num_layers - 2>(net).layer_details().tie(head.layer_details().get_layer_params(),
                head.get_parameter_gradient());
        }

        struct model_info
        {
            static std::string describe ()
            {
                std::ostringstream out;
                out << NUM_LAYERS << " blocks, width " << EMBEDDING_DIM << ", " << NUM_HEADS << " heads over "
                    << NUM_KV_HEADS << (NUM_KV_HEADS == 1 ? " key-value head" : " key-value heads")
                    << ", one table of " << VOCAB_SIZE << " tokens shared by the input and the head";
                return out.str();
            }
        };
    };
}

const long VOCAB_SIZE    = SLM_VOCAB_SIZE;      // tokenizer target, special tokens included
const long NUM_LAYERS    = SLM_NUM_LAYERS;
const long NUM_HEADS     = SLM_NUM_HEADS;
const long NUM_KV_HEADS  = SLM_NUM_KV_HEADS;
const long EMBEDDING_DIM = SLM_EMBEDDING_DIM;
const long WINDOW        = SLM_WINDOW;          // training sequence, and attention span when coding

using model_config = tok_compressor_config<VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, EMBEDDING_DIM>;
using net_type = model_config::network_type<true>;

const std::string MODEL_FILE = "slm_tok_predictive_compressor.dat";
const std::string HEADER_FILE = "slm_tok_predictive_compressor_ex.h";     // default path of the generated header

// ----------------------------------------------------------------------------------------
// Tokenizer

static std::string read_corpus (const std::string& path, size_t max_bytes)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (max_bytes > 0 && text.size() > max_bytes) text.resize(max_bytes);
    return text;
}

// The size of a file, at most max_bytes when that is not zero.
static size_t corpus_size (const std::string& path, size_t max_bytes)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const size_t size = static_cast<size_t>(f.tellg());
    return max_bytes > 0 ? std::min(size, max_bytes) : size;
}

// size bytes of a file from offset.
static std::string read_corpus_part (const std::string& path, size_t offset, size_t size)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(static_cast<std::streamoff>(offset));
    std::string text(size, '\0');
    f.read(&text[0], static_cast<std::streamsize>(size));
    text.resize(static_cast<size_t>(f.gcount()));
    return text;
}

// ----------------------------------------------------------------------------------------
// The generated header

static void put_f32 (std::vector<uint8_t>& out, float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    tok_codec::put_u32(out, u);
}

/*
    Writes the header: the coded container as an array of bytes, so that the executable holds
    exactly the coded bytes, with nothing between them.
*/
static void write_header (const std::vector<uint8_t>& raw, const std::string& description, const std::string& path)
{
    const std::vector<uint8_t> packed = tok_codec::pack_container(raw);
    std::ostringstream out;
    out << "// Generated by slm_tok_predictive_compressor_ex --build-tokenizer, --train or --embed. Do not edit.\n"
        << "// " << description << "\n"
        << "// The merge table and the quantized network, coded by the program's own context models.\n"
        << "#ifndef SLM_TOK_PREDICTIVE_COMPRESSOR_EX_H_\n#define SLM_TOK_PREDICTIVE_COMPRESSOR_EX_H_\n\n"
        << "#include <cstddef>\n\nnamespace embedded_codec\n{\n"
        << "    const std::size_t packed_size = " << packed.size() << ";\n"
        << "    const unsigned char packed[] = {\n";
    for (size_t i = 0; i < packed.size(); i += 28) {
        out << "        ";
        for (size_t j = i; j < std::min(packed.size(), i + 28); ++j) out << static_cast<int>(packed[j]) << ',';
        out << "\n";
    }
    if (packed.empty()) out << "        0\n";
    out << "    };\n}\n\n#endif // SLM_TOK_PREDICTIVE_COMPRESSOR_EX_H_\n";

    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f << out.str();
        if (!f) throw std::runtime_error("cannot write " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        // Some systems refuse to rename over an existing file.
        std::remove(path.c_str());
        if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot replace " + path);
    }
    cout << "Wrote " << path << ": " << raw.size() << " bytes of data coded into " << packed.size()
        << " bytes. It takes effect next to slm_tok_predictive_compressor_ex.cpp, after a rebuild." << endl;
}

static void append_tokenizer (std::vector<uint8_t>& raw, int segmentation, bool folded,
    const std::vector<std::pair<int, int>>& merges)
{
    tok_codec::put_u32(raw, 0x32435154u);    // "TQC2"
    tok_codec::put_u32(raw, static_cast<uint32_t>(segmentation));
    tok_codec::put_u32(raw, folded ? 1 : 0);
    tok_codec::put_u32(raw, static_cast<uint32_t>(merges.size()));
    for (const auto& m : merges) {
        raw.push_back(static_cast<uint8_t>(m.first));
        raw.push_back(static_cast<uint8_t>(m.first >> 8));
        raw.push_back(static_cast<uint8_t>(m.second));
        raw.push_back(static_cast<uint8_t>(m.second >> 8));
    }
}

// The text the tokenizer works on: the corpus itself, or its form with folded capitals.
static std::string tokenizer_text (const std::string& corpus, bool folded)
{
    if (!folded) return corpus;
    return tok_codec::fold_capitals<std::string>(reinterpret_cast<const uint8_t*>(corpus.data()), corpus.size());
}

static int build_tokenizer_command (const std::string& input, size_t max_bytes, long vocab_size, int segmentation,
    bool folded, const std::string& header)
{
    const std::string corpus = read_corpus(input, max_bytes);
    const std::string form = tok_codec::tokenizer_form(segmentation, folded);
    cout << "=== Building the tokenizer ===\n  corpus : " << input << " (" << corpus.size() << " bytes)\n"
        << "  target : " << vocab_size << " entries, special tokens included\n"
        << "  form   : " << form << "\n" << endl;
    const std::string text = tokenizer_text(corpus, folded);
    bpe_tokenizer tok;
    tok.set_segmentation(static_cast<bpe_segmentation>(segmentation));
    tok.train(text, vocab_size, 0, true);

    /*
        The corpus is tokenized and restored by blocks of whole lines, the lines of the folded
        text matching those of the corpus one for one, so that the check holds one block at a
        time beside the two texts.
    */
    const size_t block = size_t(16) << 20;
    size_t tokens = 0, a = 0, b = 0;
    while (a < text.size()) {
        size_t a_end = std::min(text.size(), a + block);
        while (a_end < text.size() && text[a_end - 1] != '\n') ++a_end;
        const std::string piece = text.substr(a, a_end - a);
        const std::vector<int> ids = tok.encode(piece);
        tokens += ids.size();
        const std::string decoded = tok.decode(ids, false);
        const size_t lines = static_cast<size_t>(std::count(piece.begin(), piece.end(), '\n'));
        size_t b_end = b;
        for (size_t seen = 0; b_end < corpus.size() && (seen < lines || a_end == text.size()); ++b_end)
            if (corpus[b_end] == '\n') ++seen;
        const bool restored = folded
            ? tok_codec::unfold_capitals(reinterpret_cast<const uint8_t*>(decoded.data()), decoded.size())
                == std::vector<uint8_t>(corpus.begin() + b, corpus.begin() + b_end)
            : decoded == piece;
        if (!restored) throw std::runtime_error("the tokenizer does not restore its corpus");
        a = a_end;
        b = b_end;
    }
    cout << "\n" << tokens << " tokens, " << std::fixed << std::setprecision(3)
        << static_cast<double>(corpus.size()) / tokens << " bytes of the corpus per token" << endl;

    std::vector<uint8_t> raw;
    append_tokenizer(raw, segmentation, folded, tok.get_merges());
    tok_codec::put_u32(raw, 0);    // no network: a network belongs to the tokenizer it was trained with
    write_header(raw, "Tokenizer of " + std::to_string(tok.get_vocab_size()) + " entries, " + form
        + ", no network.", header);
    return 0;
}

// ----------------------------------------------------------------------------------------
// The tokenizer of a saved network

/*
    A network belongs to the tokenizer it was trained with, since the columns of its table
    are token ids. The saved network is followed by its tokenizer and by
    whether the capitals were folded, as other language model examples follow theirs with the
    tokenizer, and the checkpoint of a run, which the trainer writes itself, has them in a file
    of its own. A saved network or a checkpoint whose tokenizer differs from the compiled-in
    one is refused. One that records no tokenizer, written by an earlier version of this
    program, is taken with a warning.
*/
struct tokenizer_record
{
    bpe_tokenizer tokenizer;
    bool folded = false;
};

static void write_record (std::ostream& out, const tok_codec::codec_data& data)
{
    serialize(data.tokenizer, out);
    serialize(data.folded, out);
}

// Reads the record that follows a network, or fills a checkpoint's own file. False when there is none.
static bool read_record (std::istream& in, tokenizer_record& record)
{
    if (in.peek() == std::char_traits<char>::eof()) return false;
    deserialize(record.tokenizer, in);
    deserialize(record.folded, in);
    return true;
}

template <typename net_t>
static bool load_network (const std::string& path, net_t& net, tokenizer_record& record)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("cannot open " + path);
    deserialize(net, fin);
    return read_record(fin, record);
}

static void check_tokenizer (const std::string& path, bool recorded, const tokenizer_record& saved,
    const tok_codec::codec_data& data, const std::string& remedy)
{
    if (!recorded) {
        cout << "Warning: " << path << " records no tokenizer; it is taken as trained with the compiled-in one."
            << endl;
        return;
    }
    const bool same = saved.folded == data.folded
        && saved.tokenizer.get_segmentation() == data.tokenizer.get_segmentation()
        && saved.tokenizer.get_merges() == data.tokenizer.get_merges();
    if (!same)
        throw std::runtime_error(path + " was trained with another tokenizer than the compiled-in one: " + remedy);
}

// ----------------------------------------------------------------------------------------
// Exporting the trained network

/*
    Gathers the parameters of the trained network in the order the engine reads them. The
    computational layers are visited from the output down; the list is reversed and checked
    against the expected sequence: the scale of the input table, then per block a
    normalisation, the attention, a normalisation and the three projections of SwiGLU, then
    the final normalisation and the head, whose weights are also the input table.
*/
struct exported_layer
{
    enum kind_t { input_scale, norm, attention, projection } kind;
    std::vector<float> values;      // matrices as [inputs x outputs]; the scale of the input table
    std::vector<float> bias;
    long rows = 0, cols = 0;
};

struct weight_collector
{
    std::vector<exported_layer>* out;

    template <unsigned long N, unsigned long D>
    void operator() (tied_embeddings_<N, D>& l)
    {
        exported_layer x;
        x.kind = exported_layer::input_scale;
        x.values.assign(1, l.scale());
        out->push_back(std::move(x));
    }

    void operator() (rms_norm_& l)
    {
        const tensor& p = l.get_layer_params();
        exported_layer x;
        x.kind = exported_layer::norm;
        x.values.assign(p.host(), p.host() + p.size());
        out->push_back(std::move(x));
    }

    template <long E, long H, long KV, long HD, bool QK, bool CAUSAL>
    void operator() (gqa_attention_<E, H, KV, HD, QK, CAUSAL>& l)
    {
        static_assert(!QK, "the engine has no QK normalisation");
        DLIB_CASSERT(!l.qkv_bias_enabled() && !l.adapters_active(), "unsupported attention options");
        const tensor& p = l.get_layer_params();
        const long q = H * HD, kv = KV * HD;
        DLIB_CASSERT(static_cast<long>(p.size()) >= E * q + 2 * E * kv + q * E);
        exported_layer x;
        x.kind = exported_layer::attention;
        x.values.assign(p.host(), p.host() + E * q + 2 * E * kv + q * E);
        x.rows = E;
        x.cols = q;
        out->push_back(std::move(x));
    }

    template <unsigned long N, linear_bias_mode M>
    void operator() (linear_<N, M>& l)
    {
        const tensor& p = l.get_layer_params();
        const long in = static_cast<long>(l.get_num_inputs()), outs = static_cast<long>(N);
        exported_layer x;
        x.kind = exported_layer::projection;
        x.rows = in;
        x.cols = outs;
        x.values.assign(p.host(), p.host() + in * outs);
        if (M == LINEAR_HAS_BIAS) x.bias.assign(p.host() + in * outs, p.host() + in * outs + outs);
        else x.bias.assign(outs, 0.0f);
        out->push_back(std::move(x));
    }

    template <typename T>
    void operator() (T&) {}
};

/*
    Symmetric quantization of an [inputs x outputs] matrix with one scale per output channel,
    per token for the head. The clipping point of each channel is chosen among a few fractions
    of its largest magnitude as the one with the least squared error, so that a single outlier
    no longer decides the step of the whole channel.
*/
static void put_quantized (std::vector<uint8_t>& raw, const std::vector<float>& m, long rows, long cols, int bits,
    double& sq_err, double& sq_ref)
{
    const int qmax = (1 << (bits - 1)) - 1;
    const long groups = cols, per = rows;
    auto at = [&](long g, long i) { return m[i * cols + g]; };
    std::vector<float> scales(groups);
    for (long g = 0; g < groups; ++g) {
        float amax = 0;
        for (long i = 0; i < per; ++i) amax = std::max(amax, std::abs(at(g, i)));
        float best_scale = amax > 0 ? amax / qmax : 1.0f;
        double best_err = -1;
        for (int c = 0; c < 8 && amax > 0; ++c) {
            const float scale = amax * (1.0f - 0.06f * c) / qmax;
            double err = 0;
            for (long i = 0; i < per; ++i) {
                const float v = at(g, i);
                const long q = std::max<long>(-qmax, std::min<long>(qmax, std::lround(v / scale)));
                err += (v - q * scale) * (v - q * scale);
            }
            if (best_err < 0 || err < best_err) { best_err = err; best_scale = scale; }
        }
        scales[g] = best_scale;
        put_f32(raw, best_scale);
    }
    for (long r = 0; r < rows; ++r)
        for (long c = 0; c < cols; ++c) {
            const float v = m[r * cols + c];
            const float scale = scales[c];
            const long q = std::max<long>(-qmax, std::min<long>(qmax, std::lround(v / scale)));
            raw.push_back(static_cast<uint8_t>(static_cast<int8_t>(q)));
            sq_err += (v - q * scale) * (v - q * scale);
            sq_ref += v * v;
        }
}

static void put_floats (std::vector<uint8_t>& raw, const std::vector<float>& v)
{
    for (float f : v) put_f32(raw, f);
}

static std::vector<uint8_t> export_network (net_type& net, const tok_codec::codec_data& data, int bits)
{
    std::vector<exported_layer> layers;
    weight_collector collector{ &layers };
    visit_computational_layers(net, collector);
    std::reverse(layers.begin(), layers.end());

    const size_t expected = 1 + 6 * NUM_LAYERS + 2;
    if (layers.size() != expected) throw std::runtime_error("unexpected network layout");
    const long head_dim = EMBEDDING_DIM / NUM_HEADS, q = NUM_HEADS * head_dim, kv = NUM_KV_HEADS * head_dim;

    std::vector<uint8_t> raw;
    append_tokenizer(raw, data.segmentation, data.folded, data.tokenizer.get_merges());
    tok_codec::put_u32(raw, 2);                     // a network whose input table is its head
    const long ffn = layers[5].cols;
    const long vocab = layers.back().cols;
    for (long v : { vocab, EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, head_dim, ffn, WINDOW })
        tok_codec::put_u32(raw, static_cast<uint32_t>(v));

    // The training layer's normalisation epsilon and rotary frequencies.
    put_f32(raw, static_cast<float>(DEFAULT_RMS_NORM_EPS));
    for (long i = 0; i < head_dim / 2; ++i)
        put_f32(raw, std::pow(10000.0f, -2.0f * i / static_cast<float>(head_dim)));

    double err = 0, ref = 0;
    size_t idx = 0;
    // The input table is the head's: only its scale is written here.
    const exported_layer& e = layers[idx++];
    if (e.kind != exported_layer::input_scale) throw std::runtime_error("unexpected network layout");
    put_floats(raw, e.values);
    for (long l = 0; l < NUM_LAYERS; ++l) {
        const exported_layer& n1 = layers[idx++];
        const exported_layer& at = layers[idx++];
        const exported_layer& n2 = layers[idx++];
        const exported_layer& pa = layers[idx++];
        const exported_layer& pb = layers[idx++];
        const exported_layer& pd = layers[idx++];
        if (n1.kind != exported_layer::norm || at.kind != exported_layer::attention || n2.kind != exported_layer::norm
            || pa.kind != exported_layer::projection || pb.kind != exported_layer::projection
            || pd.kind != exported_layer::projection)
            throw std::runtime_error("unexpected network layout");
        put_floats(raw, n1.values);
        const float* p = at.values.data();
        const long E = EMBEDDING_DIM;
        put_quantized(raw, std::vector<float>(p, p + E * q), E, q, bits, err, ref);
        p += E * q;
        put_quantized(raw, std::vector<float>(p, p + E * kv), E, kv, bits, err, ref);
        p += E * kv;
        put_quantized(raw, std::vector<float>(p, p + E * kv), E, kv, bits, err, ref);
        p += E * kv;
        put_quantized(raw, std::vector<float>(p, p + q * E), q, E, bits, err, ref);
        put_floats(raw, n2.values);
        put_quantized(raw, pa.values, pa.rows, pa.cols, bits, err, ref);
        put_floats(raw, pa.bias);
        put_quantized(raw, pb.values, pb.rows, pb.cols, bits, err, ref);
        put_floats(raw, pb.bias);
        put_quantized(raw, pd.values, pd.rows, pd.cols, bits, err, ref);
        put_floats(raw, pd.bias);
    }
    const exported_layer& nf = layers[idx++];
    const exported_layer& head = layers[idx++];
    put_floats(raw, nf.values);
    put_quantized(raw, head.values, head.rows, head.cols, bits, err, ref);
    put_floats(raw, head.bias);
    cout << "Quantized to " << bits << " bits: relative weight error " << std::scientific << std::setprecision(3)
        << std::sqrt(err / std::max(ref, 1e-30)) << std::defaultfloat << endl;
    return raw;
}

// ----------------------------------------------------------------------------------------
// Training

struct train_options
{
    size_t max_bytes = 0;
    size_t slice_bytes = 0;         // the corpus read in consecutive slices of this size, 0 for all at once
    long steps = 20000;
    long batch = 32;
    double learning_rate = 1e-3;
    double min_rate_ratio = 0.1;
    long warmup = 500;
    double weight_decay = 0.01;
    int bits = 8;
    std::string header = HEADER_FILE;
};

static matrix<double, 0, 1> rate_schedule (const train_options& o)
{
    matrix<double, 0, 1> schedule(o.steps);
    for (long s = 0; s < o.steps; ++s) {
        double r;
        if (s < o.warmup) r = o.learning_rate * (s + 1) / o.warmup;
        else {
            const double t = static_cast<double>(s - o.warmup) / std::max<long>(1, o.steps - o.warmup);
            const double cosine = 0.5 * (1 + std::cos(3.14159265358979 * t));
            r = o.learning_rate * (o.min_rate_ratio + (1 - o.min_rate_ratio) * cosine);
        }
        schedule(s) = r;
    }
    return schedule;
}

/*
    Runs the last window of the corpus through the trained network and through the engine
    fed with the exported, quantized weights, and compares their cross-entropies. A wrong
    layout shows up as a large gap; the gap of a correct export is what quantization costs.
*/
static void check_network (net_type& net, const std::vector<uint8_t>& raw, const std::vector<uint16_t>& tokens,
    double bytes_per_token)
{
    if (tokens.size() < static_cast<size_t>(WINDOW + 1)) return;
    tok_codec::codec_data exported;
    tok_codec::load_codec_data(raw, exported);
    const size_t start = tokens.size() - WINDOW - 1;

    matrix<int, 0, 1> win(WINDOW);
    for (long i = 0; i < WINDOW; ++i) win(i) = tokens[start + i];
    resizable_tensor x;
    net.subnet().to_tensor(&win, &win + 1, x);
    net.subnet().forward(x);
    const tensor& logits = net.subnet().get_output();
    const long V = logits.nc();
    const float* lg = logits.host();

    exported.engine.start(1);
    std::vector<float> probs(exported.engine.shape().vocab);
    double ce_net = 0, ce_engine = 0;
    for (long i = 0; i < WINDOW; ++i) {
        const float* row = lg + i * V;
        const float mx = *std::max_element(row, row + V);
        double z = 0;
        for (long j = 0; j < V; ++j) z += std::exp(static_cast<double>(row[j] - mx));
        const int target = tokens[start + i + 1];
        ce_net += -(row[target] - mx - std::log(z));
        const int id = 0, t = tokens[start + i];
        exported.engine.step(&id, &t, 1, probs.data());
        ce_engine += -std::log(std::max(1e-30, static_cast<double>(probs[target])));
    }
    const double k = 1.0 / (WINDOW * std::log(2.0) * bytes_per_token);
    cout << "Check on " << WINDOW << " tokens: network " << std::fixed << std::setprecision(4) << ce_net * k
        << " bits/byte, compiled form " << ce_engine * k << " bits/byte" << endl;
}

// The shape the constants give the network, as the reports and the header state it.
static std::string network_shape ()
{
    return model_config::model_info::describe() + ", window " + std::to_string(WINDOW);
}

static std::string network_description (const tok_codec::codec_data& data)
{
    return "Tokenizer of " + std::to_string(data.tokenizer.get_vocab_size()) + " entries, "
        + tok_codec::tokenizer_form(data.segmentation, data.folded) + "; network of " + network_shape() + ".";
}

// The tokens of the corpus from offset, size bytes of it, and the rate of bytes per token.
static std::vector<uint16_t> corpus_tokens (const std::string& path, size_t offset, size_t size,
    const tok_codec::codec_data& data, double& bytes_per_token)
{
    std::vector<uint16_t> tokens;
    const std::string corpus = read_corpus_part(path, offset, size);
    const std::vector<int> ids = data.tokenizer.encode(tokenizer_text(corpus, data.folded));
    tokens.assign(ids.begin(), ids.end());
    bytes_per_token = static_cast<double>(corpus.size()) / std::max<size_t>(1, tokens.size());
    return tokens;
}

static int train_command (const std::string& input, const train_options& o)
{
    tok_codec::codec_data& data = tok_codec::embedded_codec_data();
    if (!data.has_tokenizer())
        throw std::runtime_error("no compiled-in tokenizer: run --build-tokenizer and rebuild first");
    const bpe_tokenizer& tok = data.tokenizer;
    if (static_cast<long>(tok.get_vocab_size()) > VOCAB_SIZE)
        throw std::runtime_error("the compiled-in tokenizer is larger than VOCAB_SIZE");

    /*
        The corpus is read whole, or in consecutive slices that each take an equal share of the
        steps, the slice of a step following from its number, so that a resumed run reads the
        slice it stopped in. Only the tokens of the current slice are held, as 16-bit ids. The
        rates per byte are those of the text as read, whether or not its capitals are folded.
    */
    const size_t total_bytes = corpus_size(input, o.max_bytes);
    const size_t slice_bytes = o.slice_bytes > 0 ? std::min(o.slice_bytes, total_bytes) : total_bytes;
    const long slices = static_cast<long>((total_bytes + slice_bytes - 1) / std::max<size_t>(1, slice_bytes));
    const long steps_per_slice = (o.steps + slices - 1) / std::max(1L, slices);
    auto slice_of = [&](long step) { return std::min(slices - 1, step / steps_per_slice); };

    cout << "=== Training ===\n"
        << "  corpus    : " << input << " (" << total_bytes << " bytes";
    if (slices > 1)
        cout << ", " << slices << " slices of " << slice_bytes << " bytes, " << steps_per_slice << " steps each";
    cout << ")\n"
        << "  tokenizer : " << tok.get_vocab_size() << " entries, " << tok_codec::tokenizer_form(data.segmentation,
            data.folded) << "\n"
        << "  network   : " << network_shape() << "\n"
        << "  schedule  : " << o.steps << " steps of " << o.batch << " windows, peak rate " << std::setprecision(5)
        << o.learning_rate << ", warmup " << o.warmup << ", ending at " << o.min_rate_ratio << " of the peak\n" << endl;

    // A checkpoint or a saved network trained with another tokenizer stops the run here.
    net_type net;
    const std::string sync = "chkpt-" + MODEL_FILE, sync_tokenizer = sync + ".tokenizer";
    tokenizer_record saved;
    if (file_exists(sync) || file_exists(sync + "_")) {
        bool recorded = false;
        if (file_exists(sync_tokenizer)) {
            std::ifstream fin(sync_tokenizer, std::ios::binary);
            recorded = read_record(fin, saved);
        }
        check_tokenizer(sync, recorded, saved, data, "delete " + sync + "* to train from scratch");
    }
    else if (file_exists(MODEL_FILE)) {
        const bool recorded = load_network(MODEL_FILE, net, saved);
        check_tokenizer(MODEL_FILE, recorded, saved, data, "delete it to train from scratch");
        cout << "Continuing from " << MODEL_FILE << endl;
    }
    {
        std::ofstream fout(sync_tokenizer, std::ios::binary);
        write_record(fout, data);
        if (!fout) throw std::runtime_error("cannot write " + sync_tokenizer);
    }
    dnn_trainer<net_type, adamw> trainer(net, adamw(o.weight_decay, 0.9, 0.98));
    trainer.set_learning_rate_schedule(rate_schedule(o));
    /*
        The state of the run goes to the trainer's own pair of synchronization files, written and
        read as the trainer writes and reads them, but by this loop: a trainer that finds its loss
        rising since its last synchronization reloads the state it saved, which a slice harder than
        the one before would trigger at each change of slice. A loss that is not a number reloads
        the last state here instead.
    */
    auto load_state = [&]() {
        std::ifstream fin(select_newest_file(sync, sync + "_"), std::ios::binary);
        deserialize(trainer, fin);
    };
    auto save_state = [&]() {
        trainer.get_net();
        serialize(select_oldest_file(sync, sync + "_")) << trainer;
    };
    if (file_exists(sync) || file_exists(sync + "_")) load_state();
    model_config::tie(net);
    /*
        The synchronization file holds the schedule the run started with. A resumed run follows
        from the step it reached the schedule its options give, so that lengthening a run
        lengthens its cosine and another --min-rate changes where the cosine ends.
    */
    const long reached = static_cast<long>(trainer.get_train_one_step_calls());
    if (reached > 0 && reached < o.steps) {
        const matrix<double, 0, 1> full = rate_schedule(o);
        trainer.set_learning_rate_schedule(rowm(full, range(reached, o.steps - 1)));
        cout << "Resuming at step " << reached << " on the schedule of " << o.steps << " steps" << endl;
    }
    trainer.be_quiet();
    network_context::set_optimizer_params(o.weight_decay, 0.9, 0.98);
    cout << "Parameters: " << count_network_parameters(net, WINDOW).total << endl;

    std::mt19937_64 rng(42 + trainer.get_train_one_step_calls());
    std::vector<uint16_t> tokens;
    double bytes_per_token = 1;
    long current = -1;
    std::uniform_int_distribution<size_t> pick;
    std::vector<matrix<int, 0, 1>> xs(o.batch, matrix<int, 0, 1>(WINDOW));
    std::vector<matrix<unsigned long, 0, 1>> ys(o.batch, matrix<unsigned long, 0, 1>(WINDOW));
    const auto t0 = std::chrono::steady_clock::now();
    auto last_report = t0, last_save = t0;
    const unsigned long first_step = trainer.get_train_one_step_calls();

    while (trainer.get_train_one_step_calls() < static_cast<unsigned long>(o.steps)
        && !signal_handler::is_triggered()) {
        const long k = slice_of(static_cast<long>(trainer.get_train_one_step_calls()));
        if (k != current) {
            const size_t offset = static_cast<size_t>(k) * slice_bytes;
            std::vector<uint16_t>().swap(tokens);
            tokens = corpus_tokens(input, offset, std::min(slice_bytes, total_bytes - offset), data, bytes_per_token);
            if (tokens.size() < static_cast<size_t>(4 * WINDOW))
                throw std::runtime_error("corpus or slice too small for the window");
            pick = std::uniform_int_distribution<size_t>(0, tokens.size() - WINDOW - 1);
            current = k;
            cout << (slices > 1 ? "slice " + std::to_string(k + 1) + "/" + std::to_string(slices) + ": " : "")
                << tokens.size() << " tokens, " << std::fixed << std::setprecision(3) << bytes_per_token
                << " bytes per token" << endl;
        }
        for (long b = 0; b < o.batch; ++b) {
            const size_t start = pick(rng);
            for (long i = 0; i < WINDOW; ++i) {
                xs[b](i) = tokens[start + i];
                ys[b](i) = tokens[start + i + 1];
            }
        }
        trainer.train_one_step(xs, ys);
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_report).count() >= 30.0) {
            last_report = now;
            const unsigned long step = trainer.get_train_one_step_calls();
            const double secs = std::chrono::duration<double>(now - t0).count();
            const double rate = (step - first_step) / std::max(secs, 1e-9);
            // The mean over the steps since the previous line, the trainer's mean being cleared after each.
            const double loss = trainer.get_average_loss();
            trainer.clear_average_loss();
            if (!std::isfinite(loss)) {
                cout << "The loss is not a number: reloading the last saved state" << endl;
                load_state();
                model_config::tie(net);
                continue;
            }
            if (std::chrono::duration<double>(now - last_save).count() >= 600.0) {
                save_state();
                last_save = now;
            }
            cout << "step " << step << "/" << o.steps << "  loss " << std::fixed << std::setprecision(4) << loss
                << " nats/token, " << std::setprecision(3) << loss / std::log(2.0) / bytes_per_token << " bits/byte"
                << "  rate " << std::scientific << std::setprecision(2) << trainer.get_learning_rate() << std::fixed
                << "  " << std::setprecision(0) << rate * o.batch * WINDOW << " tokens/s, "
                << (o.steps - step) / std::max(rate, 1e-9) / 60.0 << " min left" << endl;
        }
    }
    if (signal_handler::is_triggered()) {
        cout << "\nInterrupted: saving what was learned." << endl;
        signal_handler::reset();
    }
    save_state();
    net.clean();
    {
        std::ofstream fout(MODEL_FILE, std::ios::binary);
        serialize(net, fout);
        write_record(fout, data);
        if (!fout) throw std::runtime_error("cannot write " + MODEL_FILE);
    }
    cout << "Saved " << MODEL_FILE << endl;
    const std::vector<uint8_t> raw = export_network(net, data, o.bits);
    if (tokens.empty()) tokens = corpus_tokens(input, total_bytes - slice_bytes, slice_bytes, data, bytes_per_token);
    check_network(net, raw, tokens, bytes_per_token);
    write_header(raw, network_description(data), o.header);
    return 0;
}

static int embed_command (const std::string& model_file, int bits, const std::string& header)
{
    tok_codec::codec_data& data = tok_codec::embedded_codec_data();
    if (!data.has_tokenizer()) throw std::runtime_error("no compiled-in tokenizer");
    net_type net;
    tokenizer_record saved;
    const bool recorded = load_network(model_file, net, saved);
    model_config::tie(net);
    check_tokenizer(model_file, recorded, saved, data, "rebuild with the header of its tokenizer");
    write_header(export_network(net, data, bits), network_description(data), header);
    return 0;
}

#endif

// ----------------------------------------------------------------------------------------

int main (int argc, char** argv)
{
    try {
        dlib::command_line_parser parser;
        parser.set_group_name("Modes");
#if !SLM_DECOMPRESSOR_ONLY
        parser.add_option("compress", "Compress --input into --output (default: input + .dlc).");
#endif
        parser.add_option("decompress", "Decompress --input into --output (default: input + .out).");
#if SLM_TRAINING
        parser.add_option("build-tokenizer", "Train the tokenizer on --input and write the header.");
        parser.add_option("train", "Train the network on --input with the compiled-in tokenizer and write the header.");
        parser.add_option("embed", "Write the header from a saved network file.", 1);
#endif
        parser.set_group_name("Options");
        parser.add_option("input", "Input file.", 1);
        parser.add_option("output", "Output file.", 1);
#if !SLM_DECOMPRESSOR_ONLY
        parser.add_option("streams", "Streams coded side by side in token mode (default: 16).", 1);
        parser.add_option("tables", "Size of the largest tables of the context models, as log2 of their four-byte "
            "units, 16 to 24 (default: from the input size, up to 24).", 1);
        parser.add_option("keep-order", "Code the pages of a wiki dump in their own order rather than grouped by "
            "category.");
#endif
#if SLM_TRAINING
        parser.add_option("max-bytes", "Use at most this many megabytes of the corpus (default: all).", 1);
        parser.add_option("slice", "With --train, read the corpus in consecutive slices of this many megabytes, "
            "one at a time, each for an equal share of the steps (default: the whole corpus at once).", 1);
        parser.add_option("vocab", "Tokenizer size, special tokens included (default: "
            + std::to_string(VOCAB_SIZE) + ").", 1);
        parser.add_option("segmentation", "How --build-tokenizer cuts the text: spaces, a word with the space "
            "before it, or classes, which also cuts between letters, digits and other bytes (default: spaces).", 1);
        parser.add_option("fold-capitals", "With --build-tokenizer, write capitals as lower case after a marker "
            "byte, so that a word and its capitalized form share their tokens.");
        parser.add_option("steps", "Training steps (default: 20000).", 1);
        parser.add_option("batch", "Windows per training step (default: 32).", 1);
        parser.add_option("rate", "Peak learning rate (default: 0.001).", 1);
        parser.add_option("warmup", "Warmup steps (default: 500).", 1);
        parser.add_option("min-rate", "Rate at the end of the cosine decay, as a fraction of the peak rate "
            "(default: 0.1).", 1);
        parser.add_option("bits", "Bits per quantized weight, 4 to 8 (default: 8).", 1);
        parser.add_option("header", "Path of the header to write (default: " + HEADER_FILE + ").", 1);
#endif
        parser.add_option("h", "Display this help.");
        parser.parse(argc, argv);

        // The options of this build given more than once, or out of range.
        const char* single[] = { "input", "output", "streams", "tables", "max-bytes", "slice", "vocab",
            "segmentation", "steps", "batch", "rate", "warmup", "min-rate", "bits", "header" };
        for (const char* name : single)
            if (parser.option_is_defined(name) && parser.option(name).count() > 1)
                throw std::runtime_error(std::string("--") + name + " is given more than once");
        if (parser.option_is_defined("tables")) parser.check_option_arg_range("tables", 16, 24);
        if (parser.option_is_defined("streams")) parser.check_option_arg_range("streams", 1, 255);
        if (parser.option_is_defined("bits")) parser.check_option_arg_range("bits", 4, 8);
        if (parser.option_is_defined("min-rate")) parser.check_option_arg_range("min-rate", 0.0, 1.0);
        if (parser.option_is_defined("slice")) parser.check_option_arg_range("slice", 1.0, 1e9);

        // One mode; the decompressor alone needs none.
        int modes = 0;
        for (const char* mode : { "compress", "decompress", "build-tokenizer", "train", "embed" })
            if (parser.option_is_defined(mode) && parser.option(mode)) ++modes;
        if (parser.option("h") || modes > 1 || (modes == 0 && !SLM_DECOMPRESSOR_ONLY)) {
#if SLM_TRAINING
            std::cout << "Usage: " << argv[0]
                << " --compress | --decompress | --build-tokenizer | --train | --embed <file>\n";
#elif SLM_CODEC_ONLY
            std::cout << "Usage: " << argv[0] << " --compress | --decompress --input <file> [--output <file>]\n";
#else
            std::cout << "Usage: " << argv[0] << " --input <archive> [--output <file>]\n";
#endif
            parser.print_options();
            const tok_codec::codec_data& data = tok_codec::embedded_codec_data();
            const std::string tokenizer = data.has_tokenizer()
                ? std::to_string(data.vocab.size()) + "-token tokenizer ("
                    + tok_codec::tokenizer_form(data.segmentation, data.folded) + ")"
                : std::string("no tokenizer");
            std::cout << "\nCompiled in: " << tokenizer << ", " << (data.has_network() ? "a network" : "no network")
                << "." << std::endl;
            return parser.option("h") ? 0 : 1;
        }

#if SLM_TRAINING
        dlib::signal_handler::setup();
        const std::string header = get_option(parser, "header", HEADER_FILE);
        if (parser.option("embed"))
            return embed_command(parser.option("embed").argument(), get_option(parser, "bits", 8), header);
#endif
        const std::string input = get_option(parser, "input", "");
        if (input.empty()) throw std::runtime_error("--input is required");
#if SLM_TRAINING
        const size_t max_bytes = static_cast<size_t>(get_option(parser, "max-bytes", 0.0) * 1024 * 1024);
        if (parser.option("build-tokenizer")) {
            const std::string name = get_option(parser, "segmentation", std::string("spaces"));
            if (name != "spaces" && name != "classes")
                throw std::runtime_error("--segmentation is spaces or classes, not " + name);
            const int segmentation = static_cast<int>(name == "classes" ? bpe_segmentation::character_classes
                                                                        : bpe_segmentation::space_prefixed);
            return build_tokenizer_command(input, max_bytes, get_option(parser, "vocab", VOCAB_SIZE), segmentation,
                parser.option("fold-capitals").count() > 0, header);
        }
        if (parser.option("train")) {
            train_options o;
            o.max_bytes = max_bytes;
            o.slice_bytes = static_cast<size_t>(get_option(parser, "slice", 0.0) * 1024 * 1024);
            o.steps = get_option(parser, "steps", o.steps);
            o.batch = get_option(parser, "batch", o.batch);
            o.learning_rate = get_option(parser, "rate", o.learning_rate);
            o.warmup = get_option(parser, "warmup", o.warmup);
            o.min_rate_ratio = get_option(parser, "min-rate", o.min_rate_ratio);
            o.bits = get_option(parser, "bits", o.bits);
            o.header = header;
            return train_command(input, o);
        }
#endif
#if !SLM_DECOMPRESSOR_ONLY
        if (parser.option("compress"))
            return compress_command(input, get_option(parser, "output", input + ".dlc"),
                get_option(parser, "streams", 16), get_option(parser, "tables", 0),
                parser.option("keep-order").count() > 0);
#endif
        return decompress_command(input, get_option(parser, "output", input + ".out"));
    }
    catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
}
