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

    Each decision is predicted by context models over the bytes already coded (orders one
    to eight, sparse contexts, words, the line above, the current token and article, a
    long-range match), which keep a bit history per context and decision and learn what
    each history predicts, by the network, and by maps that recalibrate the network for each
    kind of decision and each pair of previous bytes. Five mixers, whose weights are
    selected by the kind of decision with the network's confidence, by the previous byte,
    by the match, by the number of contexts already seen, and by the network's confidence
    in its own first choice, combine these predictions, a sixth mixer combines theirs,
    three adaptive maps refine the result, and a binary arithmetic coder writes it.

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
    fused operations in the same order as the scalar code. Every kernel is compiled whatever
    the options of the build and chosen at run time, so that every processor computes the
    same values.

    HOW THE TEXT IS CUT FOR THE NETWORK

    A long input is cut at line boundaries into streams coded side by side: the network
    advances every stream by one token per step, so that each step multiplies the weights by
    several rows at once. The context models are shared by all streams and the match model
    searches all of them.

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
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
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
    // Context mixing

    /*
        Predicts one binary decision at a time over byte streams. A decision is a bit of the
        next byte, identified by the partial byte c0 (a one followed by the bits already known),
        or, when the stream is coded token by token, whether the current token ends at the
        current byte, identified by 256 plus its depth.

        Twenty context models each give the bit history their context has for the decision,
        turned into a probability by a map learned per model, and a second input when the
        history is deterministic. Their contexts are the last one to eight bytes, two sparse
        contexts, the current word alone and with one or two previous words, the byte above in
        the previous line, the current token, the opening of the current line with the column,
        the byte a long-range match predicts, and the current word and the last three bytes
        within the current article. A match model gives the bit the predicted byte has, and
        the network enters raw and through two maps that recalibrate it. Five mixers, whose
        weights are selected by the kind of decision with the network's confidence, by the
        previous byte, by the match, by the number of contexts already seen, and by the
        network's confidence in its own first choice, combine these inputs; a sixth mixer
        combines theirs, and three adaptive maps refine the result.
    */
    const int CM_CONTEXTS = 20;

    struct stream_state
    {
        std::vector<uint8_t> hist;     // bytes of the stream so far
        uint64_t last8 = 0;            // last eight bytes, the newest in the low byte
        uint64_t ctx[CM_CONTEXTS] = {};
        uint32_t w0 = 0, w1 = 0, w2 = 0;   // the current word and the two before, case folded
        uint64_t token_hash = 0;       // previous token and the bytes of the current one
        int depth = 0;                 // bytes of the current token already coded
        uint64_t match_ptr = 0;        // stream << 40 | position of the predicted byte
        int match_len = 0;
        int match_byte = -1;           // byte the match predicts, or -1
        size_t line_start = 0, prev_line_start = 0;
        uint64_t line_tag = 0;         // the opening of the current line, up to a space or '>'
        int tag_len = 0;
        bool tag_open = true;
        uint64_t article = 0;          // the title line of the current page
        int top_cost = 31;             // what the network's most likely token costs, in quarter bits
    };

    class predictor
    {
    public:
        static const int NUM_CTX = CM_CONTEXTS;
        static const int MIN_MATCH = 6;
        // Two inputs per context model, the match model's two, the network's three and a bias.
        static const int INPUTS = 2 * NUM_CTX + 2 + 3 + 1;

        void init (int table_bits, std::vector<stream_state>* streams_)
        {
            streams = streams_;
            for (int i = 0; i < NUM_CTX; ++i) {
                // A context with few distinct values fills a table 32 times smaller.
                const bool few = i <= 1 || i == 8 || i == 9 || i == 14 || i == 17;
                tables[i].init(few ? std::min(table_bits - 2, std::max(16, table_bits - 7)) : table_bits - 2);
                end_tables[i].assign(size_t(1) << std::max(16, table_bits - 4), 0);
                maps[i].init(8);
                end_maps[i].init(4);
            }
            end_shift = 64 - std::max(16, table_bits - 4);
            const int match_bits = std::max(16, table_bits - 2);
            match_table.assign(size_t(1) << match_bits, 0);
            match_shift = 64 - match_bits;
            match_map.init(19 * 2);
            nn_map.init(24, 7);
            nn_map2.init(1 << 16, 6);
            m1.init(INPUTS, 12 * 8, 12, 16384);
            m2.init(INPUTS, 512, 12, 16384);
            m3.init(INPUTS, 12 * 4 * 3, 12, 16384);
            m4.init(INPUTS, 12 * 8, 12, 16384);
            m5.init(INPUTS, 12 * 32, 12, 16384);
            mf.init(6, 12 * 4, 8, 16384);
            a1.init(1 << 17, 6);
            a2.init(1 << 16, 6);
            a3.init(12 * 19 * 3, 6);
            cached_stream = -1;
        }

        // The cost of the network's most likely next token for stream s, in quarter bits up to 31.
        void network_confidence (int s, int quarter_bits) { (*streams)[s].top_cost = quarter_bits; }

        // Returns P(bit == 1) as a 16-bit probability. nn_valid says whether nn_st, the
        // network's prediction in the logistic domain, exists.
        int predict (int s, int code, int nn_st, bool nn_valid)
        {
            stream_state& st = (*streams)[s];
            const logistic& lg = logistic_tables();
            const bit_states& bs = histories();
            const bool end_decision = code >= 256;
            const int kind = end_decision ? 8 + std::min(3, code - 256) : bit_position(code);
            int* x = m1.input();

            if (end_decision) {
                // Token ends have histories of their own, per context and depth.
                const int depth = code - 256;
                for (int i = 0; i < NUM_CTX; ++i) {
                    uint8_t* h = &end_tables[i][hash64(st.ctx[i], static_cast<uint64_t>(depth)) >> end_shift];
                    history[i] = h;
                    const int stp = lg.stretch(end_maps[i].p(std::min(3, depth - 1), *h));
                    x[2 * i] = stp;
                    x[2 * i + 1] = bs.deterministic(*h) ? stp : 0;
                }
            }
            else {
                // The slots of the current nibble, looked up at its first decision.
                const int nibble = kind >= 4 ? 1 : 0;
                const uint64_t key = (static_cast<uint64_t>(st.hist.size()) << 1) | static_cast<uint64_t>(nibble);
                if (cached_stream != s || cached_key != key) {
                    cached_stream = s;
                    cached_key = key;
                    for (int i = 0; i < NUM_CTX; ++i)
                        slots[i] = tables[i].find(nibble
                            ? hash64(st.ctx[i], static_cast<uint64_t>(code >> (kind - 4))) : st.ctx[i]);
                }
                const int k = nibble ? kind - 4 : kind;
                const int node = (1 << k) - 1 + (code & ((1 << k) - 1));
                for (int i = 0; i < NUM_CTX; ++i) {
                    uint8_t* h = &slots[i]->st[node];
                    history[i] = h;
                    const int stp = lg.stretch(maps[i].p(kind, *h));
                    x[2 * i] = stp;
                    x[2 * i + 1] = bs.deterministic(*h) ? stp : 0;
                }
            }
            end_active = end_decision;

            // Match model: the bit the predicted byte has here, by length and expected bit.
            match_active = false;
            int expected = -1;
            if (st.match_byte >= 0 && !end_decision && ((st.match_byte + 256) >> (8 - kind)) == code)
                expected = (st.match_byte >> (7 - kind)) & 1;
            const int length_class = st.match_len < 16 ? st.match_len
                : (st.match_len < 32 ? 16 : (st.match_len < 64 ? 17 : 18));
            if (expected >= 0) {
                match_active = true;
                x[2 * NUM_CTX] = lg.stretch(match_map.p(length_class * 2 + expected, 0));
                x[2 * NUM_CTX + 1] = expected ? length_class * 64 : -length_class * 64;
            }
            else {
                x[2 * NUM_CTX] = 0;
                x[2 * NUM_CTX + 1] = 0;
            }

            // The network, raw and through two maps, per kind of decision and per two previous bytes.
            nn_active = nn_valid;
            x[2 * NUM_CTX + 2] = nn_valid ? nn_st : 0;
            x[2 * NUM_CTX + 3] = nn_valid ? lg.stretch(nn_map.refine(nn_st, kind + 12 * (st.depth > 0))) : 0;
            const int nn_ctx2 = static_cast<int>(hash64(st.last8 & 0xffff, static_cast<uint64_t>(kind)) >> 48);
            x[2 * NUM_CTX + 4] = nn_valid ? lg.stretch(nn_map2.refine(nn_st, nn_ctx2)) : 0;
            x[2 * NUM_CTX + 5] = 256;

            std::copy(x, x + INPUTS, m2.input());
            std::copy(x, x + INPUTS, m3.input());
            std::copy(x, x + INPUTS, m4.input());
            std::copy(x, x + INPUTS, m5.input());
            const int confidence = nn_valid ? std::min(7, 1 + std::abs(nn_st) / 384) : 0;
            const int last = static_cast<int>(st.last8 & 0xff);
            const int match_state = st.match_len == 0 ? 0 : (st.match_len < 16 ? 1 : (st.match_len < 32 ? 2 : 3));
            int seen = 0;
            for (int i = 1; i <= 7; ++i) seen += *history[i] != 0;
            const int article_seen = *history[18] != 0 ? 1 : 0;
            int* xf = mf.input();
            xf[0] = m1.mix(kind * 8 + confidence);
            xf[1] = m2.mix(end_decision ? 256 + std::min(255, code - 256) : last);
            xf[2] = m3.mix((kind * 4 + match_state) * 3 + (expected + 1));
            xf[3] = m4.mix(kind * 8 + std::min(7, seen));
            xf[4] = m5.mix(kind * 32 + std::min(15, st.top_cost / 2) * 2 + article_seen);
            xf[5] = 256;
            const int smix = mf.mix(kind * 4 + match_state);

            const int p1 = a1.refine(smix, (last << 9) | std::min(code, 511));
            const int p2 = a2.refine(smix,
                static_cast<int>(hash64(st.last8 & 0xffff, static_cast<uint64_t>(code)) >> 48));
            const int p3 = a3.refine(smix, (kind * 19 + length_class) * 3 + (expected + 1));
            const int p = (2 * mf.p16() + p1 + 3 * p2 + 2 * p3) >> 3;
            return p < 1 ? 1 : (p > 65535 ? 65535 : p);
        }

        void update (int bit)
        {
            const bit_states& bs = histories();
            for (int i = 0; i < NUM_CTX; ++i) {
                if (end_active) end_maps[i].update(bit); else maps[i].update(bit);
                *history[i] = bs.next(*history[i], bit);
            }
            if (match_active) match_map.update(bit);
            if (nn_active) { nn_map.update(bit); nn_map2.update(bit); }
            m1.update(bit);
            m2.update(bit);
            m3.update(bit);
            m4.update(bit);
            m5.update(bit);
            mf.update(bit);
            a1.update(bit);
            a2.update(bit);
            a3.update(bit);
        }

        // A byte of stream s is complete: append it and recompute the contexts.
        void byte_done (int s, uint8_t byte, bool token_mode)
        {
            stream_state& st = (*streams)[s];
            st.hist.push_back(byte);
            st.last8 = (st.last8 << 8) | byte;
            ++st.depth;
            const size_t n = st.hist.size();

            const bool letter = (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || byte >= 128;
            if (letter) {
                const uint32_t lower = (byte >= 'A' && byte <= 'Z') ? byte + 32u : byte;
                st.w0 = (st.w0 + lower + 1) * 0x2F0F3A1Bu;
            }
            else if (st.w0 != 0) {
                st.w2 = st.w1;
                st.w1 = st.w0;
                st.w0 = 0;
            }
            track_lines(st, byte, n);

            const size_t column = n - st.line_start;
            const size_t above_pos = st.prev_line_start + column;
            const uint64_t above = (st.line_start > 0 && above_pos < st.line_start) ? st.hist[above_pos] : 0;
            const uint64_t l8 = st.last8;
            // Orders one to eight, then the two bytes before the last one, and the last byte with the third.
            st.ctx[0] = hash64(1, l8 & 0xff);
            st.ctx[1] = hash64(2, l8 & 0xffff);
            st.ctx[2] = hash64(3, l8 & 0xffffff);
            st.ctx[3] = hash64(4, l8 & 0xffffffffULL);
            st.ctx[4] = hash64(5, l8 & 0xffffffffffULL);
            st.ctx[5] = hash64(6, l8 & 0xffffffffffffULL);
            st.ctx[6] = hash64(7, l8 & 0xffffffffffffffULL);
            st.ctx[7] = hash64(8, l8);
            st.ctx[8] = hash64(10, (l8 >> 8) & 0xffff);
            st.ctx[9] = hash64(13, l8 & 0xff00ff);
            // The word being written, alone and after the previous words; between words, the
            // previous word and the byte that ended it.
            const uint64_t after_word = st.w1 ^ (static_cast<uint64_t>(byte) << 32);
            st.ctx[10] = letter ? hash64(11, st.w0) : hash64(12, after_word);
            st.ctx[11] = hash64(hash64(14, st.w0), st.w1);
            st.ctx[12] = hash64(hash64(hash64(15, st.w0), st.w1), st.w2);
            st.ctx[13] = hash64(hash64(16, st.w0), st.w2);
            // The byte above in the previous line, for tables and aligned text.
            st.ctx[14] = hash64(17, above | (static_cast<uint64_t>(byte) << 8));
            // The previous token and the bytes of the current one; in byte mode, the column.
            st.token_hash = hash64(st.token_hash, byte);
            st.ctx[15] = token_mode ? hash64(18, st.token_hash)
                : hash64(9, byte | (std::min<uint64_t>(column, 255) << 8));
            // The opening of the line with the column, for the fields of structured lines.
            st.ctx[16] = hash64(hash64(19, st.line_tag), std::min<uint64_t>(column, 63));
            // The word and the last three bytes within the current article.
            st.ctx[18] = hash64(hash64(22, st.article), letter ? st.w0 : after_word);
            st.ctx[19] = hash64(hash64(23, st.article), l8 & 0xffffff);

            update_match(s, st, byte);
            // The byte the match predicts with its length, or the last byte.
            st.ctx[17] = st.match_len > 0
                ? hash64(20, static_cast<uint64_t>(st.match_byte)
                    | (static_cast<uint64_t>(std::min(st.match_len, 15)) << 8))
                : hash64(21, l8 & 0xff);
            cached_stream = -1;
        }

        // A token of stream s is complete.
        void token_done (int s, int token)
        {
            stream_state& st = (*streams)[s];
            st.depth = 0;
            st.token_hash = hash64(0x5bd1e995ULL, static_cast<uint64_t>(token));
            st.ctx[15] = hash64(18, st.token_hash);
            cached_stream = -1;
        }

    private:
        static int bit_position (int c0)
        {
            int p = 0;
            while (c0 > 1) { c0 >>= 1; ++p; }
            return p;
        }

        // Line starts, the opening of the current line, and the title line of the current page.
        static void track_lines (stream_state& st, uint8_t byte, size_t n)
        {
            if (byte == '\n') {
                size_t i = st.line_start;
                while (i < n && st.hist[i] == ' ') ++i;
                if (i + 7 < n && std::memcmp(&st.hist[i], "<title>", 7) == 0) {
                    uint64_t h = 0x7171;
                    for (size_t j = i; j < n; ++j) h = hash64(h, st.hist[j]);
                    st.article = h;
                }
                st.prev_line_start = st.line_start;
                st.line_start = n;
                st.line_tag = 0;
                st.tag_len = 0;
                st.tag_open = true;
            }
            else if (st.tag_open && !(byte == ' ' && st.tag_len == 0)) {
                st.line_tag = hash64(st.line_tag + 1, byte);
                ++st.tag_len;
                if (byte == '>' || byte == ' ' || st.tag_len >= 12) st.tag_open = false;
            }
        }

        void update_match (int s, stream_state& st, uint8_t byte)
        {
            const size_t n = st.hist.size();
            if (st.match_byte >= 0 && st.match_byte == byte) { ++st.match_len; ++st.match_ptr; }
            else st.match_len = 0;

            if (n >= static_cast<size_t>(MIN_MATCH)) {
                uint64_t& entry = match_table[hash64(0xA5A5u, st.last8 & 0xffffffffffffULL) >> match_shift];
                if (st.match_len == 0 && entry != 0) {
                    // Measured backwards, so that a hash collision yields a short length.
                    const stream_state& src = (*streams)[static_cast<size_t>(entry >> 40)];
                    const uint64_t pos = entry & ((uint64_t(1) << 40) - 1);
                    int len = 0;
                    while (len < 64 && static_cast<uint64_t>(len) < pos && static_cast<size_t>(len) < n
                        && src.hist[pos - 1 - len] == st.hist[n - 1 - len]) ++len;
                    if (len >= MIN_MATCH) { st.match_ptr = entry; st.match_len = len; }
                }
                entry = (static_cast<uint64_t>(s) << 40) | n;
            }
            st.match_byte = -1;
            if (st.match_len > 0) {
                const stream_state& src = (*streams)[static_cast<size_t>(st.match_ptr >> 40)];
                const uint64_t pos = st.match_ptr & ((uint64_t(1) << 40) - 1);
                if (pos < src.hist.size()) st.match_byte = src.hist[pos];
                else st.match_len = 0;
            }
        }

        std::vector<stream_state>* streams = nullptr;
        slot_table tables[NUM_CTX];
        std::vector<uint8_t> end_tables[NUM_CTX];
        int end_shift = 48;
        state_map maps[NUM_CTX], end_maps[NUM_CTX];
        history_slot* slots[NUM_CTX] = {};
        uint8_t* history[NUM_CTX] = {};
        int cached_stream = -1;
        uint64_t cached_key = 0;
        bool end_active = false;
        std::vector<uint64_t> match_table;
        int match_shift = 44;
        state_map match_map;
        bool match_active = false;
        apm nn_map, nn_map2;
        bool nn_active = false;
        mixer m1, m2, m3, m4, m5, mf;
        apm a1, a2, a3;
    };

    // ----------------------------------------------------------------------------------------
    // Token hints

    /*
        A match model over the tokens of several streams, which gives the network a hint at
        every position: the token that followed the most recent earlier occurrence of the last
        six tokens, or failing that of the last two, in any stream, verified backwards; a hint
        that comes true extends the match. The level of a hint grows with the length of its
        match, from 1 for one or two tokens to HINT_LEVELS from 32 on, and a hint is coded
        level * vocab + token, 0 meaning none, as dlib::hinted_embeddings_ reads it. Training
        runs it over the corpus as one stream, the coder over its streams side by side; it
        depends only on the tokens already coded, so that the decoder finds the same hints.
    */
    const int HINT_LEVELS = 7;

    class token_matcher
    {
    public:
        // Tables sized for a text of about total_tokens tokens.
        void init (int streams, uint64_t total_tokens)
        {
            int bits = 16;
            while (bits < 24 && (uint64_t(1) << (bits + 2)) < total_tokens) ++bits;
            shift = 64 - bits;
            long_table.assign(size_t(1) << bits, 0);
            short_table.assign(size_t(1) << bits, 0);
            hist.assign(streams, std::vector<uint16_t>());
            state.assign(streams, match_state());
        }

        void reserve (int s, size_t tokens) { hist[s].reserve(tokens); }

        // Appends token t to stream s and returns the hint for the token that follows it.
        uint32_t push (int s, uint16_t t, uint32_t vocab)
        {
            match_state& m = state[s];
            if (m.len > 0 && predicted(m) == t) { ++m.len; ++m.ptr; }
            else m.len = 0;
            std::vector<uint16_t>& h = hist[s];
            h.push_back(t);
            const uint64_t n = h.size();
            if (n >= 2) {
                const size_t short_slot = static_cast<size_t>(hash64(h[n - 2] + 1, h[n - 1]) >> shift);
                size_t long_slot = 0;
                if (n >= 6) {
                    uint64_t k = 0x6c6f6e67;
                    for (uint64_t i = n - 6; i < n; ++i) k = hash64(k, h[i]);
                    long_slot = static_cast<size_t>(k >> shift);
                }
                if (m.len == 0 && n >= 6 && long_table[long_slot] != 0) try_match(m, s, long_table[long_slot], 6);
                if (m.len == 0 && short_table[short_slot] != 0) try_match(m, s, short_table[short_slot], 2);
                const uint64_t here = (static_cast<uint64_t>(s) << 40) | n;
                if (n >= 6) long_table[long_slot] = here;
                short_table[short_slot] = here;
            }
            const int p = m.len > 0 ? predicted(m) : -1;
            if (p < 0) return 0;
            const uint32_t len = m.len;
            const uint32_t level = len <= 2 ? 1 : len == 3 ? 2 : len <= 5 ? 3 : len <= 8 ? 4 : len <= 15 ? 5
                : len <= 31 ? 6 : HINT_LEVELS;
            return level * vocab + static_cast<uint32_t>(p);
        }

    private:
        struct match_state
        {
            uint64_t ptr = 0;    // stream << 40 | position of the predicted token
            uint32_t len = 0;
        };

        // The predicted token, or -1 when its stream has not reached it yet.
        int predicted (const match_state& m) const
        {
            const std::vector<uint16_t>& src = hist[static_cast<size_t>(m.ptr >> 40)];
            const uint64_t pos = m.ptr & ((uint64_t(1) << 40) - 1);
            return pos < src.size() ? src[pos] : -1;
        }

        // Takes the candidate when the tokens before it match the last ones of stream s over at least min_len.
        void try_match (match_state& m, int s, uint64_t candidate, uint32_t min_len)
        {
            const std::vector<uint16_t>& src = hist[static_cast<size_t>(candidate >> 40)];
            const std::vector<uint16_t>& cur = hist[s];
            const uint64_t pos = candidate & ((uint64_t(1) << 40) - 1);
            const uint64_t n = cur.size();
            uint32_t len = 0;
            while (len < 64 && len < pos && len < n && src[pos - 1 - len] == cur[n - 1 - len]) ++len;
            if (len >= min_len) { m.ptr = candidate; m.len = len; }
        }

        std::vector<std::vector<uint16_t>> hist;
        std::vector<match_state> state;
        std::vector<uint64_t> long_table, short_table;
        int shift = 48;
    };

    // ----------------------------------------------------------------------------------------
    // Transformer inference

    struct engine_shape
    {
        int vocab = 0, dim = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, ffn = 0, window = 0;
        int hint_levels = 0;    // 0 when the network takes no hints
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
    struct processor_features
    {
        bool avx2_fma = false, avx512_vnni = false, avx_vnni = false;
    };

#if SLM_CODEC_X86
    inline void cpuid (int leaf, int sub, unsigned r[4])
    {
#if defined(_MSC_VER)
        int v[4];
        __cpuidex(v, leaf, sub);
        for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(v[i]);
#else
        __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
    }

#if defined(_MSC_VER) && defined(__clang__)
    __attribute__((target("xsave")))
#endif
    inline uint64_t xcr0 ()
    {
#if defined(_MSC_VER)
        return _xgetbv(0);
#else
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
    }

    inline processor_features detect_processor ()
    {
        processor_features f;
        unsigned r[4];
        cpuid(0, 0, r);
        if (r[0] < 7) return f;
        cpuid(1, 0, r);
        const unsigned fma_osxsave_avx = (1u << 12) | (1u << 27) | (1u << 28);
        if ((r[2] & fma_osxsave_avx) != fma_osxsave_avx) return f;
        const uint64_t saved = xcr0();
        if ((saved & 6) != 6) return f;
        cpuid(7, 0, r);
        f.avx2_fma = (r[1] & (1u << 5)) != 0;
        // AVX-512 with 256-bit vectors still needs the system to save the AVX-512 state.
        f.avx512_vnni = f.avx2_fma && (r[1] & (1u << 16)) && (r[1] & (1u << 31)) && (r[2] & (1u << 11))
            && (saved & 0xe0) == 0xe0;
        cpuid(7, 1, r);
        f.avx_vnni = f.avx2_fma && (r[0] & (1u << 4));
        return f;
    }
#else
    inline processor_features detect_processor () { return processor_features(); }
#endif

    inline const processor_features& processor ()
    {
        static const processor_features f = detect_processor();
        return f;
    }

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
    }

    /*
        A weight matrix for integer products: signed bytes with one scale per column, in
        strips of sixteen columns. Within a strip, the rows go by groups of four, each group
        holding the four bytes of every column side by side, the order the VNNI instructions
        read; processors with AVX2 alone read pairs of 16-bit integers instead. The rows are
        padded to a multiple of four and the columns to a multiple of sixteen with zeros.
    */
    struct int_matrix
    {
        int rows = 0, cols = 0, kpad = 0, strips = 0;
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
        const bool pairs = !vnni_paths() && processor().avx2_fma;
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
        to w.kpad. The VNNI paths take the inputs as unsigned bytes, x + 128, and subtract 128
        times the column sums; the AVX2 path takes them as 16-bit integers.
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

#if SLM_VNNI_KERNELS
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

#define SLM_VNNI_GEMM(NAME, TARGET, STEP)                                                                   \
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

        SLM_VNNI_GEMM(vnni512, SLM_AVX512_VNNI, _mm256_dpbusd_epi32)
        SLM_VNNI_GEMM(vnni256, SLM_AVX_VNNI, _mm256_dpbusd_avx_epi32)
#undef SLM_VNNI_GEMM
#endif
#undef SLM_INT_TILE
#endif
    }

    /*
        Runs the decoder the training network defines, one position at a time for several
        independent streams at once:

            x = embedding(token)
            per block:  x += W_o attention(rope(W_q n1(x)), rope(W_k n1(x)), W_v n1(x))
                        x += W_d (silu(W_a n2(x) + b_a) * (W_b n2(x) + b_b)) + b_d
            logits = W_out nf(x) + b_out

        n being an RMS normalisation scaled by a learned gain.

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

        // Reads the network; with_hints says whether the data gives it hint levels.
        void load (byte_reader& in, bool with_hints)
        {
            engine_shape s;
            s.vocab = in.i32(); s.dim = in.i32(); s.layers = in.i32(); s.heads = in.i32();
            s.kv_heads = in.i32(); s.head_dim = in.i32(); s.ffn = in.i32(); s.window = in.i32();
            s.hint_levels = with_hints ? in.i32() : 0;
            if (s.vocab <= 0 || s.dim <= 0 || s.layers <= 0 || s.heads <= 0 || s.kv_heads <= 0
                || s.heads % s.kv_heads != 0 || s.head_dim % 2 != 0 || s.head_dim > 256 || s.ffn <= 0 || s.window <= 0
                || s.hint_levels < 0 || s.hint_levels > 255 || (with_hints && s.hint_levels == 0))
                throw std::runtime_error("invalid network shape in the embedded data");
            eps = in.f32();
            inv_freq.resize(s.head_dim / 2);
            for (auto& f : inv_freq) f = in.f32();

            emb = read_matrix(in, s.vocab, s.dim, true);
            if (s.hint_levels > 0) {
                hint_projection = read_int_matrix(in, s.dim, s.dim);
                hint_gates = read_vector(in, s.hint_levels * s.dim);
            }
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
            wout = read_int_matrix(in, s.dim, s.vocab);
            bout = read_vector(in, s.vocab);
            shape_ = s;

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
            for (int i = 0; i < V; ++i) b[i] -= ADAPT_RATE * p[i];
            b[t] += ADAPT_RATE;
        }

        /*
            Feeds the newest token of each listed stream, with its hint when the network takes
            hints (hints may be null otherwise), and writes its next-token distribution.
        */
        void step (const int* ids, const int* tokens, const uint32_t* hints, int n, float* probs)
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
            if (s.hint_levels > 0 && hints) add_hints(hints, n);

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

                    float* ctx_row = &ctx[static_cast<size_t>(r) * QD];
                    for (int hh = 0; hh < s.heads; ++hh) {
                        const int g = hh / R;
                        float qr[256];
                        rotate(&q[static_cast<size_t>(r) * QD + hh * HD], rel, half, qr);
                        // Scores of all the cached keys, accumulated over the head dimension.
                        float* sc = att.data();
                        kernels::scores(qr, kt + static_cast<size_t>(g * HD) * W, W, HD, cnt, sc);
                        float mx = -3.0e38f;
                        for (int j = 0; j < cnt; ++j) { sc[j] *= scale; mx = sc[j] > mx ? sc[j] : mx; }
                        for (int j = 0; j < cnt; ++j) sc[j] = det::expf(sc[j] - mx);
                        const float inv = 1.0f / sum(sc, cnt);
                        for (int j = 0; j < cnt; ++j) sc[j] *= inv;
                        kernels::weighted_sum(sc, vr + g * HD, KVD, cnt, HD, ctx_row + hh * HD);
                    }
                }

                quantize(ctx.data(), n, QD);
                gemm(n, b.wo, o.data());
                for (size_t i = 0; i < static_cast<size_t>(n) * D; ++i) x[i] += o[i];

                rms_norm(x.data(), b.g2.data(), n, D, h.data());
                quantize(h.data(), n, D);
                gemm(n, b.wa, fa.data());
                gemm(n, b.wb, fb.data());
                for (int r = 0; r < n; ++r) {
                    float* pa = &fa[static_cast<size_t>(r) * F];
                    const float* pb = &fb[static_cast<size_t>(r) * F];
                    for (int i = 0; i < F; ++i) {
                        const float za = pa[i] + b.ba[i];
                        pa[i] = (pb[i] + b.bb[i]) * (za / (1.0f + det::expf(-za)));
                    }
                }
                quantize(fa.data(), n, F);
                gemm(n, b.wd, o.data());
                for (int r = 0; r < n; ++r)
                    for (int i = 0; i < D; ++i)
                        x[static_cast<size_t>(r) * D + i] += o[static_cast<size_t>(r) * D + i] + b.bd[i];
            }

            rms_norm(x.data(), gf.data(), n, D, h.data());
            quantize(h.data(), n, D);
            gemm(n, wout, logits.data());
            for (int r = 0; r < n; ++r) {
                float* lg = &logits[static_cast<size_t>(r) * V];
                float* pr = probs + static_cast<size_t>(r) * V;
                const float* ab = &adapted[static_cast<size_t>(ids[r]) * V];
                float mx = -3.0e38f;
                for (int i = 0; i < V; ++i) { lg[i] += bout[i] + ab[i]; mx = lg[i] > mx ? lg[i] : mx; }
                for (int i = 0; i < V; ++i) pr[i] = det::expf(lg[i] - mx);
                const float inv = 1.0f / sum(pr, V);
                for (int i = 0; i < V; ++i) pr[i] *= inv;
            }
            for (int r = 0; r < n; ++r) ++position[ids[r]];
        }

    private:
        struct block
        {
            std::vector<float> g1, g2, ba, bb, bd;
            int_matrix wq, wk, wv, wo, wa, wb, wd;
        };

        // x += G[level - 1] * (E[hinted token] P) for the rows that have a hint.
        void add_hints (const uint32_t* hints, int n)
        {
            const engine_shape& s = shape_;
            const int D = s.dim;
            const uint32_t V = static_cast<uint32_t>(s.vocab);
            hint_rows.clear();
            for (int r = 0; r < n; ++r) {
                const uint32_t code = hints[r];
                if (code < V) continue;
                const uint32_t level = code / V;
                if (level > static_cast<uint32_t>(s.hint_levels)) throw std::runtime_error("hint level out of range");
                std::memcpy(&h[hint_rows.size() * D], &emb[static_cast<size_t>(code % V) * D], sizeof(float) * D);
                hint_rows.push_back(r);
            }
            const int m = static_cast<int>(hint_rows.size());
            if (m == 0) return;
            quantize(h.data(), m, D);
            gemm(m, hint_projection, o.data());
            for (int j = 0; j < m; ++j) {
                const int r = hint_rows[j];
                const float* g = &hint_gates[static_cast<size_t>(hints[r] / V - 1) * D];
                const float* pr = &o[static_cast<size_t>(j) * D];
                float* xr = &x[static_cast<size_t>(r) * D];
                for (int i = 0; i < D; ++i) xr[i] += g[i] * pr[i];
            }
        }

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

        // A quantized matrix: one scale per row when by_row, per column otherwise, then the
        // integers row by row.
        static std::vector<float> read_matrix (byte_reader& in, int rows, int cols, bool by_row)
        {
            std::vector<float> scales(by_row ? rows : cols);
            for (auto& f : scales) f = in.f32();
            std::vector<float> m(static_cast<size_t>(rows) * cols);
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    m[static_cast<size_t>(r) * cols + c] =
                        static_cast<float>(in.i8()) * (by_row ? scales[r] : scales[c]);
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
            att.resize(s.window);
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

        // Sums with eight interleaved partial sums, combined in a fixed order.
        static float sum (const float* a, int n)
        {
            float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            int i = 0;
            for (; i + 8 <= n; i += 8)
                for (int j = 0; j < 8; ++j) acc[j] += a[i + j];
            for (int j = 0; i < n; ++i, ++j) acc[j] += a[i];
            return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        }

        void rms_norm (const float* in, const float* gain, int n, int d, float* out) const
        {
            for (int r = 0; r < n; ++r) {
                const float* row = in + static_cast<size_t>(r) * d;
                float* dst = out + static_cast<size_t>(r) * d;
                const float inv = 1.0f / std::sqrt(kernels::dot(row, row, d) / static_cast<float>(d) + eps);
                for (int i = 0; i < d; ++i) dst[i] = row[i] * inv * gain[i];
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
                float mx = 0.0f;
                for (int i = 0; i < kdim; ++i) mx = std::max(mx, std::fabs(row[i]));
                xs[r] = mx / 127.0f;
                if (!(mx > 0.0f)) continue;
                const float inv = 127.0f / mx;
                int8_t* q8 = &xq[static_cast<size_t>(r) * kpad];
                for (int i = 0; i < kdim; ++i) {
                    const float magic = 12582912.0f;    // 1.5 * 2^23: rounds to the nearest integer, ties to even
                    q8[i] = static_cast<int8_t>((row[i] * inv + magic) - magic);
                }
            }
            if (vnni_paths()) {
                xu.resize(xq.size());
                for (size_t i = 0; i < xq.size(); ++i) xu[i] = static_cast<uint8_t>(xq[i] + 128);
            }
            else if (kernels::vector_paths()) {
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
            if (kernels::vector_paths()) kernels_int::avx2(x16.data(), n, w, acc.data());
            else kernels_int::portable(xq.data(), n, w, acc.data());
#else
            kernels_int::portable(xq.data(), n, w, acc.data());
#endif
            for (int r = 0; r < n; ++r) {
                const float sx = xs[r];
                const int32_t* a = &acc[static_cast<size_t>(r) * w.cols];
                float* o = out + static_cast<size_t>(r) * w.cols;
                for (int c = 0; c < w.cols; ++c) o[c] = static_cast<float>(a[c]) * (sx * w.scale[c]);
            }
        }

        engine_shape shape_;
        float eps = 1e-5f;
        std::vector<float> inv_freq, rope_cos, rope_sin;
        std::vector<float> emb, gf, bout;
        int_matrix hint_projection;
        std::vector<float> hint_gates;
        std::vector<int> hint_rows;
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
    };

    // ----------------------------------------------------------------------------------------
    // The compiled-in data

    /*
        The embedded data is one container: the tokenizer, then the network when there is one,
        compressed by the coder of this program in byte mode. Its checksum identifies the
        build: an archive records the checksum of the data it was coded with, and a build
        holding other data refuses to decode it.

            "TQC2"  u32 segmentation  u32 folded capitals  u32 merges  merges x (u16 left, u16 right)
            u32 network (0: none, 1: a network, 2: a network that takes hints)
            [network: shape, hint levels when it takes hints, epsilon, rotation frequencies, tensors]

        A container of the earlier format "TQC1" has no field for the folded capitals.
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

            "DLC3"  u64 original size  u32 CRC-32 of the original  u32 data fingerprint
            u8 mode (0: bytes, 1: tokens)  u8 log2 of the table units per model  u8 streams
            per stream: varint byte length, and in token mode varint token count
            arithmetic code
    */
    const uint32_t ARCHIVE_MAGIC = 0x33434C44u;    // "DLC3"

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
        data.engine.start(num);

        std::vector<float> probs(static_cast<size_t>(num) * V);
        std::vector<double> cum(n_voc + 1);
        std::vector<int> ids(num), feed(num), prev(num, line_break);
        std::vector<uint64_t> done(num, 0);
        uint64_t total = 0, coded = 0;
        for (uint64_t c : counts) total += c;

        // A network that takes hints gets them from the tokens of all the streams coded so far.
        const bool hinted = data.engine.shape().hint_levels > 0;
        token_matcher matcher;
        std::vector<uint32_t> hint(num, 0), feed_hint(num, 0);
        if (hinted) {
            matcher.init(num, total);
            for (int s = 0; s < num; ++s) {
                matcher.reserve(s, static_cast<size_t>(counts[s]) + 1);
                hint[s] = matcher.push(s, static_cast<uint16_t>(line_break), static_cast<uint32_t>(V));
            }
        }

        while (coded < total) {
            int n = 0;
            for (int s = 0; s < num; ++s)
                if (done[s] < counts[s]) { ids[n] = s; feed[n] = prev[s]; feed_hint[n] = hint[s]; ++n; }
            data.engine.step(ids.data(), feed.data(), hinted ? feed_hint.data() : nullptr, n, probs.data());
            for (int i = 0; i < n; ++i) {
                const int s = ids[i];
                const float* p = &probs[static_cast<size_t>(i) * V];
                cum[0] = 0.0;
                for (int r = 0; r < n_voc; ++r) cum[r + 1] = cum[r] + p[data.vocab.token_at(r)];
                // What the network's most likely token would cost, a confidence the mixers use.
                float top = p[0];
                for (int r = 1; r < V; ++r) top = p[r] > top ? p[r] : top;
                const double top_bits =
                    -det::log(std::max(1e-30, static_cast<double>(top) / cum[n_voc])) / 0.69314718055994530942;
                model.network_confidence(s, static_cast<int>(std::min(31.0, det::round_even(4.0 * top_bits))));
                const int want = encoding ? tokens[s][done[s]] : -1;
                const int t = code_token(side, model, data.vocab, s, cum.data(), want);
                const double pt = std::max(1e-30, static_cast<double>(p[t]) / cum[n_voc]);
                stats.network_bits -= det::log(pt) / 0.69314718055994530942;
                data.engine.learn(s, t, p);
                if (hinted) hint[s] = matcher.push(s, static_cast<uint16_t>(t), static_cast<uint32_t>(V));
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
        if (in.u32() != ARCHIVE_MAGIC) throw std::runtime_error("not an archive of this program");
        const uint64_t original_size = in.u64();
        const uint32_t crc = in.u32();
        const uint32_t fingerprint = in.u32();
        const int mode = in.u8();
        const int table_bits = in.u8();
        const int num = in.u8();
        if (fingerprint != data.fingerprint)
            throw std::runtime_error("this archive was made with another tokenizer or network than this build's");
        if (mode > 1 || num < 1 || table_bits < 16 || table_bits > 30)
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
        tables that grow with the input, four counters per byte up to 2^25 per model, since
        collisions between contexts cost more than the memory they would save; and, in token
        mode, in streams cut at line boundaries, each of at least a megabyte. The text is the
        input itself, or in token mode its form with folded capitals when the tokenizer was
        built on that form.
    */
    struct coding_plan
    {
        bool token_mode = false;
        int table_bits = 0;
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
            plan.table_bits = 18;
            while (plan.table_bits < 26 && (size_t(1) << (plan.table_bits - 2)) < text.size()) ++plan.table_bits;
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
        Codes a text into an archive. original is the input, whose size and checksum the archive
        records, and folded its form with folded capitals when the tokenizer folds them, empty
        otherwise; the form that is coded is the folded one when there is one. In token mode both
        are released once tokenized, so that coding holds the tokens alone. Nothing here depends
        on dlib.
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
        put_u32(out, checksum(original.data(), original.size()));
        put_u32(out, data.fingerprint);
        out.push_back(plan.token_mode ? 1 : 0);
        out.push_back(static_cast<uint8_t>(plan.table_bits));
        out.push_back(static_cast<uint8_t>(num));

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

static int compress_command (const std::string& input, const std::string& output, int streams_wanted, int table_bits)
{
    tok_codec::codec_data& data = tok_codec::embedded_codec_data();
    std::vector<uint8_t> original = read_file(input);
    std::vector<uint8_t> folded;
    if (data.folds()) folded = tok_codec::fold_capitals<std::vector<uint8_t>>(original.data(), original.size());
    const tok_codec::coding_plan plan = tok_codec::plan_coding(data.folds() ? folded : original, data, streams_wanted,
        table_bits);
    const size_t n = std::max<size_t>(1, original.size());
    std::cout << "=== Compression ===\n  input   : " << input << " (" << original.size() << " bytes)\n"
        << "  mode    : " << (plan.token_mode ? "tokens, network and context models" : "bytes, context models only")
        << (data.folds() ? ", folded capitals" : "")
        << "\n  streams : " << plan.streams() << "\n  tables  : 2^" << plan.table_bits << " four-byte units per model"
        << std::endl;

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
    A decoder of the library's unified GQA stack, without adaptive computation: pre-norm
    blocks of fused attention (rotary positions, grouped key-value heads) and SwiGLU, a final
    RMS normalisation and a linear head scored at every position. With SLM_HINTS=1 its input is
    hinted_embeddings: each token comes with the hint of tok_codec::token_matcher, the token a
    match over everything coded before predicts next, which brings the network the repeats
    beyond its window. The network alone gains from it, but the context models already code
    these repeats, so the default keeps plain embeddings. The constants can be set at compile
    time; the engine reads the shape from the compiled-in data, so a build with other
    constants still decodes what an earlier one coded.
*/
#ifndef SLM_HINTS
#define SLM_HINTS 0
#endif
#ifndef SLM_VOCAB_SIZE
#define SLM_VOCAB_SIZE 8192
#endif
#ifndef SLM_NUM_LAYERS
#define SLM_NUM_LAYERS 4
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

const long VOCAB_SIZE    = SLM_VOCAB_SIZE;      // tokenizer target, special tokens included
const long NUM_LAYERS    = SLM_NUM_LAYERS;
const long NUM_HEADS     = SLM_NUM_HEADS;
const long NUM_KV_HEADS  = SLM_NUM_KV_HEADS;
const long EMBEDDING_DIM = SLM_EMBEDDING_DIM;
const long WINDOW        = SLM_WINDOW;          // training sequence, and attention span when coding

const bool HINTS         = SLM_HINTS != 0;      // whether the network takes the matcher's hints

#if SLM_HINTS
using token_input = hinted_embeddings<VOCAB_SIZE, EMBEDDING_DIM, tok_codec::HINT_LEVELS, input<matrix<int, 0, 1>>>;
#else
using token_input = embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>;
#endif
using net_type = classification_head<VOCAB_SIZE, impl::gqa_stack_selector<attention_impl::unified, NUM_LAYERS,
    EMBEDDING_DIM, NUM_HEADS, NUM_KV_HEADS, token_input, false>::type>;

/*
    The windows the network reads: the tokens alone, or with hints the tokens followed by
    their hints, as hinted_embeddings takes them.
*/
static void fill_window (matrix<int, 0, 1>& x, const std::vector<uint16_t>& tokens, const std::vector<uint32_t>& hints,
    size_t start)
{
    for (long i = 0; i < WINDOW; ++i) {
        x(i) = tokens[start + i];
        if (HINTS) x(WINDOW + i) = static_cast<int>(hints[start + i]);
    }
}

// The hint of every position of a corpus, for the token after it, the corpus being one stream.
static std::vector<uint32_t> corpus_hints (const std::vector<uint16_t>& tokens)
{
    std::vector<uint32_t> hints;
    if (!HINTS) return hints;
    hints.resize(tokens.size());
    tok_codec::token_matcher matcher;
    matcher.init(1, tokens.size());
    matcher.reserve(0, tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i)
        hints[i] = matcher.push(0, tokens[i], static_cast<uint32_t>(VOCAB_SIZE));
    return hints;
}

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
    A network belongs to the tokenizer it was trained with, since its embedding rows and
    output columns are token ids. The saved network is followed by its tokenizer and by
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
    against the expected sequence: the embedding table, then per block a normalisation, the
    attention, a normalisation and the three projections of SwiGLU, then the final
    normalisation and the head.
*/
struct exported_layer
{
    enum kind_t { embedding, hint, norm, attention, projection } kind;
    std::vector<float> values;      // matrices as [inputs x outputs], the embedding as [tokens x dim]
    std::vector<float> bias;        // for the hints, the gates of the levels
    long rows = 0, cols = 0;
};

struct weight_collector
{
    std::vector<exported_layer>* out;

    template <unsigned long N, unsigned long D>
    void operator() (embeddings_<N, D>& l)
    {
        const tensor& e = l.get_embeddings();
        exported_layer x;
        x.kind = exported_layer::embedding;
        x.rows = e.num_samples();
        x.cols = e.k();
        x.values.assign(e.host(), e.host() + e.size());
        out->push_back(std::move(x));
    }

    // The table, then the projection of the hinted tokens and the gates, in the reversed order of the visit.
    template <unsigned long N, unsigned long D, unsigned long L>
    void operator() (hinted_embeddings_<N, D, L>& l)
    {
        const float* p = l.get_layer_params().host();
        exported_layer h;
        h.kind = exported_layer::hint;
        h.rows = D;
        h.cols = D;
        h.values.assign(p + N * D, p + (N + D) * D);
        h.bias.assign(p + (N + D) * D, p + (N + D + L) * D);
        out->push_back(std::move(h));
        exported_layer e;
        e.kind = exported_layer::embedding;
        e.rows = N;
        e.cols = D;
        e.values.assign(p, p + N * D);
        out->push_back(std::move(e));
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
    Symmetric quantization with one scale per output channel (per token for the embedding
    table). The clipping point of each channel is chosen among a few fractions of its largest
    magnitude as the one with the least squared error, so that a single outlier no longer
    decides the step of the whole channel.
*/
static void put_quantized (std::vector<uint8_t>& raw, const std::vector<float>& m, long rows, long cols, bool by_row,
    int bits, double& sq_err, double& sq_ref)
{
    const int qmax = (1 << (bits - 1)) - 1;
    const long groups = by_row ? rows : cols, per = by_row ? cols : rows;
    auto at = [&](long g, long i) { return by_row ? m[g * cols + i] : m[i * cols + g]; };
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
            const float scale = scales[by_row ? r : c];
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

    const size_t first_block = HINTS ? 2 : 1;
    const size_t expected = first_block + 6 * NUM_LAYERS + 2;
    if (layers.size() != expected) throw std::runtime_error("unexpected network layout");
    const long head_dim = EMBEDDING_DIM / NUM_HEADS, q = NUM_HEADS * head_dim, kv = NUM_KV_HEADS * head_dim;

    std::vector<uint8_t> raw;
    append_tokenizer(raw, data.segmentation, data.folded, data.tokenizer.get_merges());
    tok_codec::put_u32(raw, HINTS ? 2 : 1);
    const long ffn = layers[first_block + 4].cols;
    const long vocab = layers.back().cols;
    for (long v : { vocab, EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, head_dim, ffn, WINDOW })
        tok_codec::put_u32(raw, static_cast<uint32_t>(v));
    if (HINTS) tok_codec::put_u32(raw, static_cast<uint32_t>(tok_codec::HINT_LEVELS));

    // The training layer's normalisation epsilon and rotary frequencies.
    put_f32(raw, static_cast<float>(DEFAULT_RMS_NORM_EPS));
    for (long i = 0; i < head_dim / 2; ++i)
        put_f32(raw, std::pow(10000.0f, -2.0f * i / static_cast<float>(head_dim)));

    double err = 0, ref = 0;
    size_t idx = 0;
    const exported_layer& e = layers[idx++];
    if (e.kind != exported_layer::embedding) throw std::runtime_error("unexpected network layout");
    put_quantized(raw, e.values, e.rows, e.cols, true, bits, err, ref);
    if (HINTS) {
        const exported_layer& h = layers[idx++];
        if (h.kind != exported_layer::hint) throw std::runtime_error("unexpected network layout");
        put_quantized(raw, h.values, h.rows, h.cols, false, bits, err, ref);
        put_floats(raw, h.bias);
    }
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
        put_quantized(raw, std::vector<float>(p, p + E * q), E, q, false, bits, err, ref);
        p += E * q;
        put_quantized(raw, std::vector<float>(p, p + E * kv), E, kv, false, bits, err, ref);
        p += E * kv;
        put_quantized(raw, std::vector<float>(p, p + E * kv), E, kv, false, bits, err, ref);
        p += E * kv;
        put_quantized(raw, std::vector<float>(p, p + q * E), q, E, false, bits, err, ref);
        put_floats(raw, n2.values);
        put_quantized(raw, pa.values, pa.rows, pa.cols, false, bits, err, ref);
        put_floats(raw, pa.bias);
        put_quantized(raw, pb.values, pb.rows, pb.cols, false, bits, err, ref);
        put_floats(raw, pb.bias);
        put_quantized(raw, pd.values, pd.rows, pd.cols, false, bits, err, ref);
        put_floats(raw, pd.bias);
    }
    const exported_layer& nf = layers[idx++];
    const exported_layer& head = layers[idx++];
    put_floats(raw, nf.values);
    put_quantized(raw, head.values, head.rows, head.cols, false, bits, err, ref);
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
    const std::vector<uint32_t>& hints, double bytes_per_token)
{
    if (tokens.size() < static_cast<size_t>(WINDOW + 1)) return;
    tok_codec::codec_data exported;
    tok_codec::load_codec_data(raw, exported);
    const size_t start = tokens.size() - WINDOW - 1;

    matrix<int, 0, 1> win(HINTS ? 2 * WINDOW : WINDOW);
    fill_window(win, tokens, hints, start);
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
        exported.engine.step(&id, &t, HINTS ? &hints[start + i] : nullptr, 1, probs.data());
        ce_engine += -std::log(std::max(1e-30, static_cast<double>(probs[target])));
    }
    const double k = 1.0 / (WINDOW * std::log(2.0) * bytes_per_token);
    cout << "Check on " << WINDOW << " tokens: network " << std::fixed << std::setprecision(4) << ce_net * k
        << " bits/byte, compiled form " << ce_engine * k << " bits/byte" << endl;
}

// The shape the constants give the network, as the reports and the header state it.
static std::string network_shape ()
{
    return std::to_string(NUM_LAYERS) + " blocks, width " + std::to_string(EMBEDDING_DIM) + ", "
        + std::to_string(NUM_HEADS) + " heads over " + std::to_string(NUM_KV_HEADS)
        + (NUM_KV_HEADS == 1 ? " key-value head" : " key-value heads") + ", window " + std::to_string(WINDOW);
}

static std::string network_description (const tok_codec::codec_data& data)
{
    return "Tokenizer of " + std::to_string(data.tokenizer.get_vocab_size()) + " entries, "
        + tok_codec::tokenizer_form(data.segmentation, data.folded) + "; network of " + network_shape() + ".";
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
        The corpus is held as 16-bit tokens only, the text and the wider ids being released. The
        rates per byte are those of the corpus as read, whether or not its capitals are folded.
    */
    std::vector<uint16_t> tokens;
    size_t corpus_bytes = 0;
    {
        const std::string corpus = read_corpus(input, o.max_bytes);
        corpus_bytes = corpus.size();
        const std::vector<int> ids = tok.encode(tokenizer_text(corpus, data.folded));
        tokens.assign(ids.begin(), ids.end());
    }
    const double bytes_per_token = static_cast<double>(corpus_bytes) / std::max<size_t>(1, tokens.size());
    if (tokens.size() < static_cast<size_t>(4 * WINDOW)) throw std::runtime_error("corpus too small for the window");
    const std::vector<uint32_t> hints = corpus_hints(tokens);

    cout << "=== Training ===\n"
        << "  corpus    : " << input << " (" << corpus_bytes << " bytes, " << tokens.size() << " tokens, "
        << std::fixed << std::setprecision(3) << bytes_per_token << " bytes per token)\n"
        << "  tokenizer : " << tok.get_vocab_size() << " entries, " << tok_codec::tokenizer_form(data.segmentation,
            data.folded) << "\n"
        << "  network   : " << network_shape() << ", vocabulary " << VOCAB_SIZE << (HINTS ? ", with hints" : "")
        << "\n"
        << "  schedule  : " << o.steps << " steps of " << o.batch << " windows, peak rate " << std::setprecision(5)
        << o.learning_rate << ", warmup " << o.warmup << "\n" << endl;

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
    trainer.set_synchronization_file(sync, std::chrono::minutes(10));
    /*
        The synchronization file holds the schedule the run started with. A run resumed with
        another --steps follows the schedule of that length from the step it reached, so that
        lengthening a run lengthens its cosine.
    */
    const long reached = static_cast<long>(trainer.get_train_one_step_calls());
    if (trainer.get_learning_rate_schedule().size() != o.steps && reached < o.steps) {
        const matrix<double, 0, 1> full = rate_schedule(o);
        trainer.set_learning_rate_schedule(rowm(full, range(reached, o.steps - 1)));
        cout << "Resuming at step " << reached << " on the schedule of " << o.steps << " steps" << endl;
    }
    trainer.be_quiet();
    network_context::set_optimizer_params(o.weight_decay, 0.9, 0.98);
    cout << "Parameters: " << count_network_parameters(net, WINDOW).total << endl;

    std::mt19937_64 rng(42 + trainer.get_train_one_step_calls());
    std::uniform_int_distribution<size_t> pick(0, tokens.size() - WINDOW - 1);
    std::vector<matrix<int, 0, 1>> xs(o.batch, matrix<int, 0, 1>(HINTS ? 2 * WINDOW : WINDOW));
    std::vector<matrix<unsigned long, 0, 1>> ys(o.batch, matrix<unsigned long, 0, 1>(WINDOW));
    const auto t0 = std::chrono::steady_clock::now();
    auto last_report = t0;
    const unsigned long first_step = trainer.get_train_one_step_calls();

    while (trainer.get_train_one_step_calls() < static_cast<unsigned long>(o.steps)
        && !signal_handler::is_triggered()) {
        for (long b = 0; b < o.batch; ++b) {
            const size_t start = pick(rng);
            fill_window(xs[b], tokens, hints, start);
            for (long i = 0; i < WINDOW; ++i) ys[b](i) = tokens[start + i + 1];
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
    trainer.get_net();
    net.clean();
    {
        std::ofstream fout(MODEL_FILE, std::ios::binary);
        serialize(net, fout);
        write_record(fout, data);
        if (!fout) throw std::runtime_error("cannot write " + MODEL_FILE);
    }
    cout << "Saved " << MODEL_FILE << endl;
    const std::vector<uint8_t> raw = export_network(net, data, o.bits);
    check_network(net, raw, tokens, hints, bytes_per_token);
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
        parser.add_option("tables", "Size of each context model's table, as log2 of its four-byte units, 16 to 30 "
            "(default: from the input size, up to 26).", 1);
#endif
#if SLM_TRAINING
        parser.add_option("max-bytes", "Use at most this many megabytes of the corpus (default: all).", 1);
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
        parser.add_option("bits", "Bits per quantized weight, 4 to 8 (default: 8).", 1);
        parser.add_option("header", "Path of the header to write (default: " + HEADER_FILE + ").", 1);
#endif
        parser.add_option("h", "Display this help.");
        parser.parse(argc, argv);

        // The options of this build given more than once, or out of range.
        const char* single[] = { "input", "output", "streams", "tables", "max-bytes", "vocab", "segmentation",
            "steps", "batch", "rate", "warmup", "bits", "header" };
        for (const char* name : single)
            if (parser.option_is_defined(name) && parser.option(name).count() > 1)
                throw std::runtime_error(std::string("--") + name + " is given more than once");
        if (parser.option_is_defined("tables")) parser.check_option_arg_range("tables", 16, 30);
        if (parser.option_is_defined("streams")) parser.check_option_arg_range("streams", 1, 255);
        if (parser.option_is_defined("bits")) parser.check_option_arg_range("bits", 4, 8);

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
            o.steps = get_option(parser, "steps", o.steps);
            o.batch = get_option(parser, "batch", o.batch);
            o.learning_rate = get_option(parser, "rate", o.learning_rate);
            o.warmup = get_option(parser, "warmup", o.warmup);
            o.bits = get_option(parser, "bits", o.bits);
            o.header = header;
            return train_command(input, o);
        }
#endif
#if !SLM_DECOMPRESSOR_ONLY
        if (parser.option("compress"))
            return compress_command(input, get_option(parser, "output", input + ".dlc"),
                get_option(parser, "streams", 16), get_option(parser, "tables", 0));
#endif
        return decompress_command(input, get_option(parser, "output", input + ".out"));
    }
    catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
}
