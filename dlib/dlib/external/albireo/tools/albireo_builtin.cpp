// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    albireo-builtin: prepares the models that come with Albireo, for the build.

        albireo-builtin --manifest <builtin_models.json> --cache <folder> [--models-out <folder>]
            [--setup <albireo-setup-online.exe> --setup-out <albireo-setup.exe>] [--threads <n>]

    The manifest (resources/builtin_models.json) names each model, the file it comes from,
    with its address, its size and its SHA-256, and the quantization Albireo ships.  The
    build downloads the files into the cache, named by their hash (cmake/builtin_models.cmake);
    this program then works on them as the import example of dlib does on a GGUF file
    (examples/slm_gguf_import_ex.cpp), with the code the engine runs:

      - the file is read (gguf_reader) and its geometry resolved (model_config), which
        rejects an architecture or a configuration the engine does not run;
      - its tokenizer must give back a text it encoded, and a model that converses must
        come with a chat template the engine recognizes;
      - the model is converted: its weights are quantized again when the manifest asks for
        another quantization than the published one, and its metadata receive its identity
        in Albireo (albireo.id, albireo.name, albireo.roles, albireo.inputs,
        albireo.builtin, albireo.source_sha256), with which every copy of the file
        describes itself;
      - the converted file is read and checked again.

    The converted files stay in the cache, named after the conversion, so that a new build
    reuses them.  --models-out gives them to a folder with their manifest builtin.json (the
    "models" folder of the programs of a build tree), by hard links when the folder is on
    the volume of the cache; --setup appends them to a copy of the installer without models,
    as the overlay of common/builtin_models.h, the complete installer.

    QUANTIZATION
        The Q4_K_M mix of the GGUF format: the matrices in Q4_K (blocks of 256 weights in 8
        groups of 32, each group with a 6-bit scale and a 6-bit minimum, fitted by a
        weighted least squares search), with Q6_K for the output matrix (the embeddings when
        the model shares them with its output), for the values of the attention and the
        down projections of the first and last eighths of the layers and of one layer in
        three between them.  The search and the mix are those of the reference quantizer of
        the format, so that the result matches the files published in Q4_K_M; a matrix whose
        rows are not a multiple of 256 takes Q5_0 (Q8_0 instead of Q6_K).  The vectors (norms,
        biases) keep their type.
*/

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dlib/data_io/gguf_dequantize.h>
#include <dlib/data_io/gguf_reader.h>
#include <dlib/tokenizer/hf_tokenizer.h>

#include "../common/builtin_models.h"
#include "../common/json.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "../engine/chat_format.h"
#include "../engine/model_config.h"
#include "../engine/model_file.h"
#include "../host/crypto.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using dlib::ggml_type;
using dlib::gguf_type;
using dlib::gguf_tensor_info;
using dlib::gguf_value;

namespace albireo
{
namespace
{

    // The version of the conversion, part of the name of a converted file in the cache.
    const char* const conversion_version = "albireo-builtin 1";

    std::FILE* open_file(const std::string& path, const char* mode)
    {
#if defined(_WIN32)
        return _wfopen(widen(path).c_str(), widen(mode).c_str());
#else
        return std::fopen(path.c_str(), mode);
#endif
    }

    bool ends_with(const std::string& s, const std::string& tail)
    {
        return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
    }

    // ---- Half precision

    // The nearest half precision value, ties to even, as the scales of the blocks are stored.
    uint16_t to_half(float value)
    {
        uint32_t x;
        std::memcpy(&x, &value, 4);
        const uint32_t sign = (x >> 16) & 0x8000u;
        const uint32_t a = x & 0x7fffffffu;
        if (a >= 0x7f800000u) return static_cast<uint16_t>(sign | (a > 0x7f800000u ? 0x7e00u : 0x7c00u));
        if (a >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);
        const int e = static_cast<int>(a >> 23);
        if (e < 113)
        {
            // Below the smallest normal half: a multiple of 2^-24.
            const int shift = 126 - e;
            if (shift > 24) return static_cast<uint16_t>(sign);
            const uint32_t m = (a & 0x7fffffu) | 0x800000u;
            uint32_t h = m >> shift;
            const uint32_t rest = m & ((1u << shift) - 1);
            const uint32_t half = 1u << (shift - 1);
            if (rest > half || (rest == half && (h & 1))) ++h;
            return static_cast<uint16_t>(sign | h);
        }
        uint32_t h = (static_cast<uint32_t>(e - 112) << 10) | ((a & 0x7fffffu) >> 13);
        const uint32_t rest = a & 0x1fffu;
        if (rest > 0x1000u || (rest == 0x1000u && (h & 1))) ++h;
        return static_cast<uint16_t>(sign | h);
    }

    float from_half(uint16_t h)
    {
        return dlib::gguf_half_to_float(h);
    }

    void put_half(uint8_t* p, uint16_t h)
    {
        p[0] = static_cast<uint8_t>(h & 0xff);
        p[1] = static_cast<uint8_t>(h >> 8);
    }

    // Rounds to the nearest integer, ties to even, through the addition of 1.5 * 2^23.
    int nearest_int(float v)
    {
        const float t = v + 12582912.f;
        int32_t i;
        std::memcpy(&i, &t, 4);
        return (i & 0x007fffff) - 0x00400000;
    }

    // ---- Quantization

    /*
        The scale and the minimum of a group of n values on nmax + 1 levels that minimize
        the error weighted by weights: the levels of the extremes first, then nstep + 1
        scales around them, each one refined by a least squares fit of the scale and the
        minimum.  L receives the levels; returns the scale and sets the_min to the
        opposite of the minimum.
    */
    float fit_scale_min(int n, int nmax, const float* x, const float* weights, uint8_t* L, float& the_min,
        uint8_t* Laux, float rmin, float rdelta, int nstep)
    {
        float min = x[0];
        float max = x[0];
        float sum_w = weights[0];
        float sum_x = sum_w * x[0];
        for (int i = 1; i < n; ++i)
        {
            if (x[i] < min) min = x[i];
            if (x[i] > max) max = x[i];
            const float w = weights[i];
            sum_w += w;
            sum_x += w * x[i];
        }
        if (min > 0) min = 0;
        if (max == min)
        {
            for (int i = 0; i < n; ++i) L[i] = 0;
            the_min = -min;
            return 0.f;
        }
        float iscale = nmax / (max - min);
        float scale = 1 / iscale;
        float best_error = 0;
        for (int i = 0; i < n; ++i)
        {
            const int l = nearest_int(iscale * (x[i] - min));
            L[i] = static_cast<uint8_t>(std::max(0, std::min(nmax, l)));
            float diff = scale * L[i] + min - x[i];
            diff = diff * diff;
            const float w = weights[i];
            best_error += w * diff;
        }
        for (int is = 0; is <= nstep; ++is)
        {
            iscale = (rmin + rdelta * is + nmax) / (max - min);
            float sum_l = 0, sum_l2 = 0, sum_xl = 0;
            for (int i = 0; i < n; ++i)
            {
                int l = nearest_int(iscale * (x[i] - min));
                l = std::max(0, std::min(nmax, l));
                Laux[i] = static_cast<uint8_t>(l);
                const float w = weights[i];
                sum_l += w * l;
                sum_l2 += w * l * l;
                sum_xl += w * l * x[i];
            }
            const float D = sum_w * sum_l2 - sum_l * sum_l;
            if (D > 0)
            {
                float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
                float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
                if (this_min > 0)
                {
                    this_min = 0;
                    this_scale = sum_xl / sum_l2;
                }
                float cur_error = 0;
                for (int i = 0; i < n; ++i)
                {
                    float diff = this_scale * Laux[i] + this_min - x[i];
                    diff = diff * diff;
                    const float w = weights[i];
                    cur_error += w * diff;
                }
                if (cur_error < best_error)
                {
                    for (int i = 0; i < n; ++i) L[i] = Laux[i];
                    best_error = cur_error;
                    scale = this_scale;
                    min = this_min;
                }
            }
        }
        the_min = -min;
        return scale;
    }

    /*
        The scale of a group of n values on the symmetric levels -nmax .. nmax - 1, the
        error weighted by the square of each value: the scale of the largest magnitude,
        then 18 scales around it.  L receives the levels shifted by nmax.
    */
    float fit_symmetric(int n, int nmax, const float* x, int8_t* L)
    {
        float max = 0;
        float amax = 0;
        for (int i = 0; i < n; ++i)
        {
            const float ax = std::fabs(x[i]);
            if (ax > amax)
            {
                amax = ax;
                max = x[i];
            }
        }
        if (amax < 1e-15f)
        {
            for (int i = 0; i < n; ++i) L[i] = 0;
            return 0.f;
        }
        float iscale = -nmax / max;
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i)
        {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            L[i] = static_cast<int8_t>(l + nmax);
            const float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        float scale = suml2 ? sumlx / suml2 : 0.0f;
        float best = scale * sumlx;
        for (int is = -9; is <= 9; ++is)
        {
            if (is == 0) continue;
            iscale = -(nmax + 0.1f * is) / max;
            sumlx = suml2 = 0;
            for (int i = 0; i < n; ++i)
            {
                int l = nearest_int(iscale * x[i]);
                l = std::max(-nmax, std::min(nmax - 1, l));
                const float w = x[i] * x[i];
                sumlx += w * x[i] * l;
                suml2 += w * l * l;
            }
            if (suml2 > 0 && sumlx * sumlx > best * suml2)
            {
                for (int i = 0; i < n; ++i)
                {
                    const int l = nearest_int(iscale * x[i]);
                    L[i] = static_cast<int8_t>(nmax + std::max(-nmax, std::min(nmax - 1, l)));
                }
                scale = sumlx / suml2;
                best = scale * sumlx;
            }
        }
        return scale;
    }

    // Q4_K: d and dmin (half), 12 bytes of packed 6-bit scales and minimums, 128 bytes of levels.
    void quantize_q4_k(const float* x, uint64_t n, uint8_t* y)
    {
        for (uint64_t b = 0; b < n / 256; ++b, x += 256, y += 144)
        {
            uint8_t L[256];
            uint8_t Laux[32];
            float weights[32];
            float mins[8];
            float scales[8];
            float max_scale = 0;
            float max_min = 0;
            for (int j = 0; j < 8; ++j)
            {
                float sum_x2 = 0;
                for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
                const float av_x = std::sqrt(sum_x2 / 32);
                for (int l = 0; l < 32; ++l) weights[l] = av_x + std::fabs(x[32 * j + l]);
                scales[j] = fit_scale_min(32, 15, x + 32 * j, weights, L + 32 * j, mins[j], Laux, -1.f, 0.1f, 20);
                if (scales[j] > max_scale) max_scale = scales[j];
                if (mins[j] > max_min) max_min = mins[j];
            }
            const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
            const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
            uint8_t* sc = y + 4;
            for (int j = 0; j < 8; ++j)
            {
                uint8_t ls = static_cast<uint8_t>(nearest_int(inv_scale * scales[j]));
                uint8_t lm = static_cast<uint8_t>(nearest_int(inv_min * mins[j]));
                ls = std::min<uint8_t>(63, ls);
                lm = std::min<uint8_t>(63, lm);
                if (j < 4)
                {
                    sc[j] = ls;
                    sc[j + 4] = lm;
                }
                else
                {
                    sc[j + 4] = static_cast<uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
                    sc[j - 4] |= static_cast<uint8_t>((ls >> 4) << 6);
                    sc[j] |= static_cast<uint8_t>((lm >> 4) << 6);
                }
            }
            const uint16_t d = to_half(max_scale / 63.f);
            const uint16_t dmin = to_half(max_min / 63.f);
            put_half(y, d);
            put_half(y + 2, dmin);
            for (int j = 0; j < 8; ++j)
            {
                uint8_t s, m;
                dlib::gguf_get_scale_min_k4(j, sc, s, m);
                const float dd = from_half(d) * s;
                if (!dd) continue;
                const float dm = from_half(dmin) * m;
                for (int i = 0; i < 32; ++i)
                {
                    const int l = nearest_int((x[32 * j + i] + dm) / dd);
                    L[32 * j + i] = static_cast<uint8_t>(std::max(0, std::min(15, l)));
                }
            }
            uint8_t* q = y + 16;
            for (int j = 0; j < 256; j += 64, q += 32)
                for (int l = 0; l < 32; ++l) q[l] = static_cast<uint8_t>(L[j + l] | (L[j + l + 32] << 4));
        }
    }

    // Q6_K: 128 bytes of low bits, 64 of high bits, 16 signed 8-bit scales, d (half).
    void quantize_q6_k(const float* x, uint64_t n, uint8_t* y)
    {
        for (uint64_t b = 0; b < n / 256; ++b, x += 256, y += 210)
        {
            int8_t L[256];
            float scales[16];
            float max_scale = 0;
            float max_abs_scale = 0;
            for (int ib = 0; ib < 16; ++ib)
            {
                const float scale = fit_symmetric(16, 32, x + 16 * ib, L + 16 * ib);
                scales[ib] = scale;
                const float abs_scale = std::fabs(scale);
                if (abs_scale > max_abs_scale)
                {
                    max_abs_scale = abs_scale;
                    max_scale = scale;
                }
            }
            if (max_abs_scale < 1e-15f)
            {
                std::memset(y, 0, 210);
                continue;
            }
            const float iscale = -128.f / max_scale;
            const uint16_t d = to_half(1 / iscale);
            put_half(y + 208, d);
            int8_t* sc = reinterpret_cast<int8_t*>(y + 192);
            for (int ib = 0; ib < 16; ++ib)
                sc[ib] = static_cast<int8_t>(std::min(127, nearest_int(iscale * scales[ib])));
            for (int j = 0; j < 16; ++j)
            {
                const float dd = from_half(d) * sc[j];
                if (!dd) continue;
                for (int ii = 0; ii < 16; ++ii)
                {
                    int l = nearest_int(x[16 * j + ii] / dd);
                    l = std::max(-32, std::min(31, l));
                    L[16 * j + ii] = static_cast<int8_t>(l + 32);
                }
            }
            uint8_t* ql = y;
            uint8_t* qh = y + 128;
            for (int j = 0; j < 256; j += 128, ql += 64, qh += 32)
            {
                for (int l = 0; l < 32; ++l)
                {
                    const uint8_t q1 = L[j + l] & 0xF;
                    const uint8_t q2 = L[j + l + 32] & 0xF;
                    const uint8_t q3 = L[j + l + 64] & 0xF;
                    const uint8_t q4 = L[j + l + 96] & 0xF;
                    ql[l] = static_cast<uint8_t>(q1 | (q3 << 4));
                    ql[l + 32] = static_cast<uint8_t>(q2 | (q4 << 4));
                    qh[l] = static_cast<uint8_t>((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2)
                        | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
                }
            }
        }
    }

    // Q8_0: d (half) and 32 signed bytes.
    void quantize_q8_0(const float* x, uint64_t n, uint8_t* y)
    {
        for (uint64_t b = 0; b < n / 32; ++b, x += 32, y += 34)
        {
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
            const float d = amax / 127;
            const float id = d ? 1.0f / d : 0.0f;
            put_half(y, to_half(d));
            for (int j = 0; j < 32; ++j) y[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(std::round(x[j] * id)));
        }
    }

    // Q5_0: d (half), the fifth bits of the 32 levels, their low 4 bits.
    void quantize_q5_0(const float* x, uint64_t n, uint8_t* y)
    {
        for (uint64_t b = 0; b < n / 32; ++b, x += 32, y += 22)
        {
            float amax = 0.0f;
            float max = 0.0f;
            for (int j = 0; j < 32; ++j)
            {
                if (amax < std::fabs(x[j]))
                {
                    amax = std::fabs(x[j]);
                    max = x[j];
                }
            }
            const float d = max / -16;
            const float id = d ? 1.0f / d : 0.0f;
            put_half(y, to_half(d));
            uint32_t qh = 0;
            for (int j = 0; j < 16; ++j)
            {
                const float x0 = x[j] * id;
                const float x1 = x[16 + j] * id;
                const int l0 = static_cast<int8_t>(x0 + 16.5f);
                const int l1 = static_cast<int8_t>(x1 + 16.5f);
                const uint8_t xi0 = static_cast<uint8_t>(std::min(31, l0));
                const uint8_t xi1 = static_cast<uint8_t>(std::min(31, l1));
                y[6 + j] = static_cast<uint8_t>((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
                qh |= ((xi0 & 0x10u) >> 4) << j;
                qh |= ((xi1 & 0x10u) >> 4) << (j + 16);
            }
            std::memcpy(y + 2, &qh, 4);
        }
    }

    uint64_t row_bytes(ggml_type t, uint64_t n)
    {
        const dlib::ggml_type_traits tr = dlib::get_ggml_type_traits(t);
        return n / tr.block_size * tr.type_size;
    }

    void quantize_row(ggml_type t, const float* x, uint64_t n, uint8_t* y)
    {
        switch (t)
        {
        case ggml_type::Q4_K: quantize_q4_k(x, n, y); break;
        case ggml_type::Q6_K: quantize_q6_k(x, n, y); break;
        case ggml_type::Q8_0: quantize_q8_0(x, n, y); break;
        case ggml_type::Q5_0: quantize_q5_0(x, n, y); break;
        case ggml_type::F16:
            for (uint64_t i = 0; i < n; ++i) put_half(y + 2 * i, to_half(x[i]));
            break;
        case ggml_type::F32: std::memcpy(y, x, n * sizeof(float)); break;
        default: throw std::runtime_error("no quantization to this type");
        }
    }

    const char* type_name(ggml_type t)
    {
        switch (t)
        {
        case ggml_type::F32: return "F32";
        case ggml_type::F16: return "F16";
        case ggml_type::BF16: return "BF16";
        case ggml_type::Q4_K: return "Q4_K";
        case ggml_type::Q5_0: return "Q5_0";
        case ggml_type::Q6_K: return "Q6_K";
        case ggml_type::Q8_0: return "Q8_0";
        default: return "other";
        }
    }

    // ---- The Q4_K_M mix

    bool quantizable(const gguf_tensor_info& t)
    {
        return t.dims.size() >= 2 && ends_with(t.name, "weight") && t.name.find("_norm.weight") == std::string::npos
            && t.name.find("ffn_gate_inp.weight") == std::string::npos;
    }

    // The first and last eighths of the layers, and one layer in three between them.
    bool more_bits(int i, int n)
    {
        return i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2;
    }

    // The order of the tensors: those outside the layers, then the layers in their order,
    // by name in each.
    int layer_of(const std::string& name)
    {
        int layer = -1;
        if (name.compare(0, 4, "blk.") == 0) layer = std::atoi(name.c_str() + 4);
        return layer;
    }

    bool tensor_before(const gguf_tensor_info* a, const gguf_tensor_info* b)
    {
        const int la = layer_of(a->name), lb = layer_of(b->name);
        if (la != lb) return la < lb;
        return a->name < b->name;
    }

    struct tensor_plan
    {
        const gguf_tensor_info* src = nullptr;
        ggml_type type = ggml_type::F32;
        uint64_t nbytes = 0;
        uint64_t offset = 0;
    };

    std::vector<tensor_plan> plan_tensors(const dlib::gguf_reader& r, const std::string& quantization, int n_layers)
    {
        std::vector<const gguf_tensor_info*> order;
        for (const auto& t : r.tensors()) order.push_back(&t);
        std::sort(order.begin(), order.end(), tensor_before);
        const bool requantize = quantization == "Q4_K_M";
        bool tied = true;
        int n_attn_v = 0;
        for (const auto* t : order)
        {
            if (t->name == "output.weight") tied = false;
            if (t->name.find("attn_v.weight") != std::string::npos) ++n_attn_v;
        }
        int i_attn_v = 0;
        int i_ffn_down = 0;
        std::vector<tensor_plan> plan;
        uint64_t offset = 0;
        for (const auto* t : order)
        {
            tensor_plan p;
            p.src = t;
            p.type = t->type;
            if (requantize && quantizable(*t))
            {
                const uint64_t ne0 = t->dims[0];
                ggml_type type = ggml_type::Q4_K;
                if (t->name == "output.weight" || (tied && t->name == "token_embd.weight"))
                {
                    type = ne0 % 256 == 0 ? ggml_type::Q6_K : ggml_type::Q8_0;
                }
                else if (t->name.find("attn_v.weight") != std::string::npos)
                {
                    if (more_bits(i_attn_v, n_attn_v)) type = ggml_type::Q6_K;
                    ++i_attn_v;
                }
                else if (t->name.find("ffn_down") != std::string::npos)
                {
                    if (more_bits(i_ffn_down, n_layers)) type = ggml_type::Q6_K;
                    ++i_ffn_down;
                }
                if (ne0 % 256 != 0)
                {
                    if (type == ggml_type::Q4_K) type = ggml_type::Q5_0;
                    else if (type == ggml_type::Q6_K) type = ggml_type::Q8_0;
                    if (ne0 % 32 != 0) type = ggml_type::F16;
                }
                p.type = type;
            }
            p.nbytes = row_bytes(p.type, t->n_elements());
            p.offset = offset;
            offset += (p.nbytes + r.alignment() - 1) / r.alignment() * r.alignment();
            plan.push_back(p);
        }
        return plan;
    }

    // ---- Writing a GGUF file

    class gguf_out
    {
    public:
        gguf_out(std::FILE* f, sha256_stream& sha) : f_(f), sha_(sha) {}

        void bytes(const void* p, size_t n)
        {
            if (n && std::fwrite(p, 1, n, f_) != n) throw std::runtime_error("the converted file cannot be written");
            sha_.update(p, n);
            written_ += n;
        }
        template <typename T> void raw(T v) { bytes(&v, sizeof(v)); }
        void str(const std::string& s)
        {
            raw<uint64_t>(s.size());
            bytes(s.data(), s.size());
        }
        void zeros(uint64_t n)
        {
            static const char z[64] = { 0 };
            while (n > 0)
            {
                const size_t k = static_cast<size_t>(std::min<uint64_t>(n, sizeof(z)));
                bytes(z, k);
                n -= k;
            }
        }
        uint64_t written() const { return written_; }

    private:
        std::FILE* f_;
        sha256_stream& sha_;
        uint64_t written_ = 0;
    };

    void put_scalar(gguf_out& o, gguf_type t, int64_t i, double d, const std::string& s)
    {
        switch (t)
        {
        case gguf_type::UINT8: o.raw<uint8_t>(static_cast<uint8_t>(i)); break;
        case gguf_type::INT8: o.raw<int8_t>(static_cast<int8_t>(i)); break;
        case gguf_type::UINT16: o.raw<uint16_t>(static_cast<uint16_t>(i)); break;
        case gguf_type::INT16: o.raw<int16_t>(static_cast<int16_t>(i)); break;
        case gguf_type::UINT32: o.raw<uint32_t>(static_cast<uint32_t>(i)); break;
        case gguf_type::INT32: o.raw<int32_t>(static_cast<int32_t>(i)); break;
        case gguf_type::UINT64: o.raw<uint64_t>(static_cast<uint64_t>(i)); break;
        case gguf_type::INT64: o.raw<int64_t>(i); break;
        case gguf_type::BOOL: o.raw<uint8_t>(i ? 1 : 0); break;
        case gguf_type::FLOAT32: o.raw<float>(static_cast<float>(d)); break;
        case gguf_type::FLOAT64: o.raw<double>(d); break;
        case gguf_type::STRING: o.str(s); break;
        default: throw std::runtime_error("unknown type of metadata value");
        }
    }

    void put_value(gguf_out& o, const gguf_value& v)
    {
        o.raw<uint32_t>(static_cast<uint32_t>(v.type));
        if (v.type != gguf_type::ARRAY)
        {
            put_scalar(o, v.type, v.i, v.d, v.s);
            return;
        }
        o.raw<uint32_t>(static_cast<uint32_t>(v.array_type));
        if (v.array_type == gguf_type::STRING)
        {
            o.raw<uint64_t>(v.arr_str.size());
            for (const auto& s : v.arr_str) o.str(s);
        }
        else if (v.array_type == gguf_type::FLOAT32 || v.array_type == gguf_type::FLOAT64)
        {
            o.raw<uint64_t>(v.arr_float.size());
            for (double d : v.arr_float) put_scalar(o, v.array_type, 0, d, std::string());
        }
        else
        {
            o.raw<uint64_t>(v.arr_int.size());
            for (int64_t i : v.arr_int) put_scalar(o, v.array_type, i, 0, std::string());
        }
    }

    gguf_value string_value(const std::string& s)
    {
        gguf_value v;
        v.type = gguf_type::STRING;
        v.array_type = gguf_type::UINT8;
        v.i = 0;
        v.d = 0;
        v.s = s;
        return v;
    }

    gguf_value strings_value(const json& list)
    {
        gguf_value v = string_value(std::string());
        v.type = gguf_type::ARRAY;
        v.array_type = gguf_type::STRING;
        for (const auto& e : list.elements()) v.arr_str.push_back(e.as_string());
        return v;
    }

    gguf_value integer_value(gguf_type t, int64_t i)
    {
        gguf_value v = string_value(std::string());
        v.type = t;
        v.i = i;
        return v;
    }

    // ---- Conversion

    struct model_entry
    {
        json desc;                 // the entry of the manifest
        std::string id;
        std::string source;        // the downloaded file
        std::string source_sha;
        std::string quantization;  // the one shipped
        std::string key;           // names the conversion
        std::string converted;     // the converted file in the cache
    };

    // The rows of a tensor as floats, from its bytes.
    void rows_as_floats(const gguf_tensor_info& t, const uint8_t* raw, uint64_t row0, uint64_t nrows,
        std::vector<float>& out)
    {
        const uint64_t ne0 = t.dims[0];
        const uint64_t rb = row_bytes(t.type, ne0);
        gguf_tensor_info part = t;
        part.dims = { ne0, nrows };
        part.offset = 0;
        part.nbytes = rb * nrows;
        const std::vector<uint8_t> bytes(raw + row0 * rb, raw + (row0 + nrows) * rb);
        dlib::gguf_dequantize(part, bytes, out);
    }

    // The relative error of a quantized row against its source.
    double row_error(const gguf_tensor_info& src, const std::vector<uint8_t>& raw, ggml_type type,
        const std::vector<uint8_t>& out, uint64_t row)
    {
        const uint64_t ne0 = src.dims[0];
        std::vector<float> a, b;
        rows_as_floats(src, raw.data(), row, 1, a);
        gguf_tensor_info q = src;
        q.type = type;
        rows_as_floats(q, out.data(), row, 1, b);
        double num = 0, den = 0;
        for (uint64_t i = 0; i < ne0; ++i)
        {
            num += (a[i] - b[i]) * static_cast<double>(a[i] - b[i]);
            den += a[i] * static_cast<double>(a[i]);
        }
        return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    }

    double error_bound(ggml_type t)
    {
        switch (t)
        {
        case ggml_type::Q4_K: return 0.2;
        case ggml_type::Q5_0: return 0.1;
        case ggml_type::Q6_K: return 0.06;
        case ggml_type::Q8_0: return 0.02;
        default: return 0.002;
        }
    }

    void convert(model_entry& m, unsigned threads, std::string& sha, uint64_t& size)
    {
        dlib::gguf_reader r(m.source);
        const std::string arch = r.get_str("general.architecture");
        const int n_layers = static_cast<int>(r.get_int(arch + ".block_count"));
        const std::vector<tensor_plan> plan = plan_tensors(r, m.quantization, n_layers);
        bool changed = false;
        for (const auto& p : plan) changed = changed || p.type != p.src->type;

        std::map<std::string, gguf_value> meta = r.metadata();
        meta.erase("general.alignment");
        if (changed)
        {
            // LLAMA_FTYPE_MOSTLY_Q4_K_M
            meta["general.file_type"] = integer_value(gguf_type::UINT32, 15);
            meta["general.quantization_version"] = integer_value(gguf_type::UINT32, 2);
        }
        meta["albireo.id"] = string_value(m.id);
        meta["albireo.name"] = string_value(m.desc["name"].as_string());
        meta["albireo.roles"] = strings_value(m.desc["roles"]);
        meta["albireo.inputs"] = strings_value(m.desc["inputs"]);
        gguf_value builtin = integer_value(gguf_type::BOOL, 1);
        meta["albireo.builtin"] = builtin;
        meta["albireo.source_sha256"] = string_value(m.source_sha);

        const std::string tmp = m.converted + ".partial";
        std::FILE* f = open_file(tmp, "wb");
        if (!f) throw std::runtime_error("cannot create " + tmp);
        std::unique_ptr<std::FILE, int (*)(std::FILE*)> closer(f, std::fclose);
        std::vector<char> buffer(4u << 20);
        std::setvbuf(f, buffer.data(), _IOFBF, buffer.size());
        sha256_stream hash;
        gguf_out o(f, hash);
        o.raw<uint32_t>(0x46554747u);   // "GGUF"
        o.raw<uint32_t>(3);
        o.raw<uint64_t>(plan.size());
        o.raw<uint64_t>(meta.size() + 1);
        // The architecture first, as the converters of the format write it.
        const uint64_t alignment = r.alignment();
        o.str("general.architecture");
        put_value(o, meta.at("general.architecture"));
        for (const auto& kv : meta)
        {
            if (kv.first == "general.architecture") continue;
            o.str(kv.first);
            put_value(o, kv.second);
        }
        o.str("general.alignment");
        put_value(o, integer_value(gguf_type::UINT32, static_cast<int64_t>(alignment)));
        for (const auto& p : plan)
        {
            o.str(p.src->name);
            o.raw<uint32_t>(static_cast<uint32_t>(p.src->dims.size()));
            for (uint64_t d : p.src->dims) o.raw<uint64_t>(d);
            o.raw<uint32_t>(static_cast<uint32_t>(p.type));
            o.raw<uint64_t>(p.offset);
        }
        o.zeros((alignment - o.written() % alignment) % alignment);
        const uint64_t data_start = o.written();

        std::vector<uint8_t> raw;
        std::vector<uint8_t> out;
        for (size_t k = 0; k < plan.size(); ++k)
        {
            const tensor_plan& p = plan[k];
            o.zeros(data_start + p.offset - o.written());
            r.read_tensor_raw(*p.src, raw);
            if (p.type == p.src->type)
            {
                o.bytes(raw.data(), raw.size());
                continue;
            }
            const uint64_t ne0 = p.src->dims[0];
            const uint64_t nrows = p.src->n_elements() / ne0;
            const uint64_t out_row = row_bytes(p.type, ne0);
            out.assign(static_cast<size_t>(p.nbytes), 0);
            // Rows in chunks, over the threads.
            std::atomic<uint64_t> next(0);
            std::atomic<bool> failed(false);
            std::string failure;
            auto work = [&]() {
                std::vector<float> rows;
                const uint64_t chunk = 16;
                try
                {
                    for (;;)
                    {
                        const uint64_t r0 = next.fetch_add(chunk);
                        if (r0 >= nrows || failed) break;
                        const uint64_t n = std::min(chunk, nrows - r0);
                        rows_as_floats(*p.src, raw.data(), r0, n, rows);
                        for (uint64_t i = 0; i < n; ++i)
                            quantize_row(p.type, rows.data() + i * ne0, ne0, out.data() + (r0 + i) * out_row);
                    }
                }
                catch (const std::exception& e)
                {
                    if (!failed.exchange(true)) failure = e.what();
                }
            };
            std::vector<std::thread> pool;
            for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
            work();
            for (auto& t : pool) t.join();
            if (failed) throw std::runtime_error(p.src->name + ": " + failure);
            // The first and the last rows against their source.
            for (uint64_t row : { uint64_t(0), nrows - 1 })
            {
                const double e = row_error(*p.src, raw, p.type, out, row);
                if (!(e <= error_bound(p.type)))
                    throw std::runtime_error(p.src->name + ": the quantized row " + std::to_string(row)
                        + " differs from its source by " + std::to_string(e));
            }
            o.bytes(out.data(), out.size());
            std::cout << "  " << p.src->name << ": " << type_name(p.src->type) << " -> " << type_name(p.type) << "\n";
        }
        if (std::fflush(f) != 0) throw std::runtime_error("the converted file cannot be written");
        closer.reset();
        sha = hash.hex();
        size = o.written();
        remove_file(m.converted);
        if (!rename_file(tmp, m.converted)) throw std::runtime_error("cannot rename " + tmp);
    }

    // ---- Checks of a file, as the engine reads it

    // Checks a file as the engine reads it; a converted file also carries its identity.
    json check_model(const std::string& path, const json& desc, bool converted)
    {
        model_file file(path);
        const model_config cfg = read_model_config(file);
        dlib::hf_tokenizer tok;
        tok.load_from_gguf(file.reader());
        if (static_cast<long>(tok.size()) != cfg.vocab)
            throw std::runtime_error("the vocabulary and the output layer of the model disagree");
        const std::string sample = "Albireo, 2026: Bonjour ! Quelle est la capitale de l'Italie ? \xC3\x89t\xC3\xA9, "
            "\xE6\xBC\xA2\xE5\xAD\x97, int main() { return 0; }";
        const std::vector<int> ids = tok.encode(sample, false, false, false);
        if (ids.empty() || tok.decode(ids, false) != sample)
            throw std::runtime_error("the tokenizer does not give back the text it encoded");
        const bool embedding = desc["embedding"].as_bool(false);
        if (cfg.is_embedding() != embedding)
            throw std::runtime_error(embedding ? "the file is not an embedding model"
                : "the file is an embedding model, the manifest names a language model");
        std::string format = "embedding";
        if (!embedding)
        {
            format = chat_format::detect(tok, cfg.chat_template)->name();
            if (format == "plain") throw std::runtime_error("the engine does not recognize the chat template");
        }
        const dlib::gguf_reader& r = file.reader();
        uint64_t parameters = 0;
        for (const auto& t : r.tensors()) parameters += t.n_elements();
        json info = json::object();
        info["arch"] = r.get_str("general.architecture");
        info["parameters"] = static_cast<unsigned long long>(parameters);
        info["context"] = static_cast<long long>(cfg.ctx_train);
        info["format"] = format;
        if (converted && r.get_str("albireo.id") != desc["id"].as_string())
            throw std::runtime_error("the identity of the converted file is wrong");
        return info;
    }

    // ---- The manifest and the files

    std::string conversion_key(const json& desc, const std::string& source_sha, const std::string& quantization)
    {
        std::string s = std::string(conversion_version) + "|" + source_sha + "|" + quantization + "|"
            + desc["id"].as_string() + "|" + desc["name"].as_string() + "|" + desc["roles"].dump() + "|"
            + desc["inputs"].dump();
        return sha256_hex(s).substr(0, 12);
    }

    bool link_or_copy(const std::string& from, const std::string& to)
    {
        remove_file(to);
#if defined(_WIN32)
        if (CreateHardLinkW(widen(to).c_str(), widen(from).c_str(), nullptr)) return true;
#else
        if (::link(from.c_str(), to.c_str()) == 0) return true;
#endif
        return copy_file(from, to);
    }

    bool copy_range(std::FILE* in, uint64_t size, std::FILE* out)
    {
        std::vector<char> b(1u << 20);
        while (size > 0)
        {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(size, b.size()));
            if (std::fread(b.data(), 1, n, in) != n || std::fwrite(b.data(), 1, n, out) != n) return false;
            size -= n;
        }
        return true;
    }

    // Appends the models to a copy of the installer, as its overlay.
    void pack_setup(const std::string& setup, const std::string& output, const std::vector<model_entry>& models,
        json manifest)
    {
        const std::string tmp = output + ".partial";
        std::FILE* out = open_file(tmp, "wb");
        if (!out) throw std::runtime_error("cannot create " + tmp);
        std::unique_ptr<std::FILE, int (*)(std::FILE*)> closer(out, std::fclose);
        std::FILE* in = open_file(setup, "rb");
        if (!in) throw std::runtime_error("cannot read " + setup);
        const uint64_t program = file_size(setup);
        const bool copied = copy_range(in, program, out);
        std::fclose(in);
        if (!copied) throw std::runtime_error("cannot copy " + setup);
        uint64_t at = 0;
        for (size_t i = 0; i < models.size(); ++i)
        {
            std::FILE* m = open_file(models[i].converted, "rb");
            if (!m) throw std::runtime_error("cannot read " + models[i].converted);
            const uint64_t size = file_size(models[i].converted);
            const bool ok = copy_range(m, size, out);
            std::fclose(m);
            if (!ok) throw std::runtime_error("cannot append " + models[i].converted);
            manifest["models"].elements()[i]["offset"] = static_cast<unsigned long long>(at);
            at += size;
        }
        const std::string text = manifest.dump();
        builtin_trailer trailer;
        trailer.manifest_offset = at;
        trailer.manifest_size = text.size();
        at += text.size();
        const uint64_t end = program + at + sizeof(trailer);
        const uint64_t pad = (8 - end % 8) % 8;
        trailer.overlay_size = at + pad + sizeof(trailer);
        std::vector<char> zeros(static_cast<size_t>(pad), 0);
        if (std::fwrite(text.data(), 1, text.size(), out) != text.size()
            || (pad && std::fwrite(zeros.data(), 1, zeros.size(), out) != zeros.size())
            || std::fwrite(&trailer, 1, sizeof(trailer), out) != sizeof(trailer) || std::fflush(out) != 0)
            throw std::runtime_error("cannot write " + tmp);
        closer.reset();
#if !defined(_WIN32)
        // The rights of the program it extends.
        struct stat st;
        if (::stat(setup.c_str(), &st) == 0) ::chmod(tmp.c_str(), st.st_mode & 07777);
#endif
        remove_file(output);
        if (!rename_file(tmp, output)) throw std::runtime_error("cannot rename " + tmp);
    }

    int usage()
    {
        std::cerr << "usage: albireo-builtin --manifest <builtin_models.json> --cache <folder>\n"
                     "           [--models-out <folder>] [--setup <setup.exe> --setup-out <setup.exe>]\n"
                     "           [--threads <n>]\n";
        return 2;
    }

    int run(int argc, char** argv)
    {
        std::map<std::string, std::string> args;
        for (int i = 1; i + 1 < argc; i += 2)
        {
            const std::string k = argv[i];
            if (k.compare(0, 2, "--") != 0) return usage();
            args[k.substr(2)] = argv[i + 1];
        }
        if (argc % 2 == 0 || !args.count("manifest") || !args.count("cache")) return usage();
        if (args.count("setup") != args.count("setup-out")) return usage();
        unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        if (args.count("threads")) threads = std::max(1, std::atoi(args["threads"].c_str()));

        std::string text;
        if (!read_file(args["manifest"], text, 16u << 20)) throw std::runtime_error("cannot read " + args["manifest"]);
        const json manifest = json::parse_or_null(text);
        if (!manifest["models"].is_array()) throw std::runtime_error("the manifest lists no models");
        const std::string cache = args["cache"];
        make_dirs(cache);

        std::vector<model_entry> models;
        json out = json::object();
        out["format"] = 1;
        out["version"] = ALBIREO_VERSION;
        out["models"] = json::array();
        for (const auto& d : manifest["models"].elements())
        {
            model_entry m;
            m.desc = d;
            m.id = d["id"].as_string();
            m.source_sha = d["from"]["sha256"].as_string();
            m.quantization = d["quantization"].str(d["from"]["quantization"].as_string());
            m.source = join_path(cache, m.source_sha + ".gguf");
            m.key = conversion_key(d, m.source_sha, m.quantization);
            m.converted = join_path(cache, m.id + "-" + m.key + ".gguf");
            std::cout << m.id << "\n";

            // A converted file of the same conversion is reused: its description, written
            // once it was checked, gives its hash.
            const std::string side = m.converted + ".json";
            std::string side_text;
            json info = file_exists(m.converted) && read_file(side, side_text, 1u << 20)
                ? json::parse_or_null(side_text) : json();
            if (!info.is_object() || info["size"].as_int(-1) != static_cast<long long>(file_size(m.converted)))
            {
                if (!file_exists(m.source))
                    throw std::runtime_error(m.id + ": " + m.source + " is missing (cmake/builtin_models.cmake "
                        "downloads it)");
                std::cout << "  checking " << m.source << "\n";
                if (sha256_file(m.source) != m.source_sha)
                    throw std::runtime_error(m.id + ": the SHA-256 of " + m.source + " differs from the manifest");
                check_model(m.source, d, false);
                std::string sha;
                uint64_t size = 0;
                std::cout << "  converting to " << m.quantization << "\n";
                convert(m, threads, sha, size);
                info = check_model(m.converted, d, true);
                info["sha256"] = sha;
                info["size"] = static_cast<unsigned long long>(size);
                if (!write_file_atomic(side, info.dump_formatted(2) + "\n"))
                    throw std::runtime_error("cannot write " + side);
            }
            std::cout << "  " << file_name(m.converted) << ", " << info["size"].as_int(0) << " bytes, "
                      << info["format"].as_string() << "\n";

            json e = d;
            e.erase("from");
            e["quantization"] = m.quantization;
            e["arch"] = info["arch"];
            e["parameters"] = info["parameters"];
            e["context"] = info["context"];
            e["file"] = m.id + ".gguf";
            e["size"] = info["size"];
            e["sha256"] = info["sha256"];
            e["source_sha256"] = m.source_sha;
            e["source_url"] = d["from"]["url"];
            out["models"].push_back(e);
            models.push_back(m);
        }

        if (args.count("models-out"))
        {
            const std::string dir = args["models-out"];
            make_dirs(dir);
            for (const auto& m : models)
                if (!link_or_copy(m.converted, join_path(dir, m.id + ".gguf")))
                    throw std::runtime_error("cannot place " + m.id + " in " + dir);
            if (!write_file_atomic(join_path(dir, builtin_manifest_name), out.dump_formatted(2) + "\n"))
                throw std::runtime_error("cannot write the manifest in " + dir);
            std::cout << "models in " << dir << "\n";
        }
        if (args.count("setup"))
        {
            pack_setup(args["setup"], args["setup-out"], models, out);
            std::cout << args["setup-out"] << ", " << file_size(args["setup-out"]) << " bytes\n";
        }
        return 0;
    }

}
}

int main(int argc, char** argv)
{
    try
    {
        return albireo::run(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "albireo-builtin: " << e.what() << "\n";
        return 1;
    }
}
