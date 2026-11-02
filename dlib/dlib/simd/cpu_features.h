// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CPU_FEATURES_Hh_
#define DLIB_CPU_FEATURES_Hh_

#include "cpu_features_abstract.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define DLIB_CPU_FEATURES_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif
#else
#define DLIB_CPU_FEATURES_X86 0
#endif

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct cpu_features
    {
        std::string vendor;
        std::string brand;

        bool sse2 = false;
        bool sse3 = false;
        bool ssse3 = false;
        bool sse41 = false;
        bool sse42 = false;
        bool popcnt = false;
        bool avx = false;
        bool f16c = false;
        bool fma = false;
        bool bmi1 = false;
        bool bmi2 = false;
        bool avx2 = false;
        bool avx512f = false;
        bool avx512dq = false;
        bool avx512cd = false;
        bool avx512bw = false;
        bool avx512vl = false;
        bool avx512_vnni = false;
        bool avx512_bf16 = false;
        bool avx512_fp16 = false;
        bool avx_vnni = false;
        bool neon = false;
    };

// ----------------------------------------------------------------------------------------

    namespace impl_cpu_features
    {
#if DLIB_CPU_FEATURES_X86
        inline void cpuid (unsigned leaf, unsigned sub, unsigned r[4])
        {
#if defined(_MSC_VER)
            int v[4];
            __cpuidex(v, static_cast<int>(leaf), static_cast<int>(sub));
            for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(v[i]);
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        }

        // XCR0, the register state the operating system saves; read only once OSXSAVE is set.
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

        inline bool bit (unsigned word, int b) { return ((word >> b) & 1u) != 0; }
#endif

        inline std::string environment (const char* name)
        {
#if defined(_MSC_VER)
            char* value = nullptr;
            size_t length = 0;
            if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) return std::string();
            const std::string text(value);
            std::free(value);
            return text;
#else
            const char* value = std::getenv(name);
            return value ? std::string(value) : std::string();
#endif
        }

        inline void clear_avx512 (cpu_features& f)
        {
            f.avx512f = f.avx512dq = f.avx512cd = f.avx512bw = f.avx512vl = false;
            f.avx512_vnni = f.avx512_bf16 = f.avx512_fp16 = false;
        }

        // Clears each feature the list names, and the features built on it.
        inline void disable (cpu_features& f, const std::string& names)
        {
            std::string name;
            for (size_t i = 0; i <= names.size(); ++i) {
                const char c = i < names.size() ? names[i] : ',';
                if (c != ',' && c != ' ' && c != ';' && c != '\t') {
                    name += static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
                    continue;
                }
                if (name == "sse2") f.sse2 = false;
                else if (name == "sse3") f.sse3 = false;
                else if (name == "ssse3") f.ssse3 = false;
                else if (name == "sse41") f.sse41 = false;
                else if (name == "sse42") f.sse42 = false;
                else if (name == "popcnt") f.popcnt = false;
                else if (name == "f16c") f.f16c = false;
                else if (name == "fma") f.fma = false;
                else if (name == "bmi1") f.bmi1 = false;
                else if (name == "bmi2") f.bmi2 = false;
                else if (name == "avx") {
                    f.avx = f.f16c = f.fma = f.avx2 = f.avx_vnni = false;
                    clear_avx512(f);
                }
                else if (name == "avx2") {
                    f.avx2 = f.avx_vnni = false;
                    clear_avx512(f);
                }
                else if (name == "avx512f" || name == "avx512") clear_avx512(f);
                else if (name == "avx512dq") f.avx512dq = false;
                else if (name == "avx512cd") f.avx512cd = false;
                else if (name == "avx512bw") f.avx512bw = false;
                else if (name == "avx512vl") f.avx512vl = false;
                else if (name == "avx512_vnni") f.avx512_vnni = false;
                else if (name == "avx512_bf16") f.avx512_bf16 = false;
                else if (name == "avx512_fp16") f.avx512_fp16 = false;
                else if (name == "avx_vnni") f.avx_vnni = false;
                else if (name == "neon") f.neon = false;
                name.clear();
            }
        }

        inline cpu_features detect ()
        {
            cpu_features f;
#if DLIB_CPU_FEATURES_X86
            unsigned r[4];
            cpuid(0, 0, r);
            const unsigned max_leaf = r[0];
            char vendor[13];
            std::memcpy(vendor, &r[1], 4);
            std::memcpy(vendor + 4, &r[3], 4);
            std::memcpy(vendor + 8, &r[2], 4);
            vendor[12] = '\0';
            f.vendor = vendor;

            cpuid(0x80000000u, 0, r);
            if (r[0] >= 0x80000004u) {
                char brand[49];
                for (unsigned i = 0; i < 3; ++i) {
                    cpuid(0x80000002u + i, 0, r);
                    std::memcpy(brand + 16 * i, r, 16);
                }
                brand[48] = '\0';
                std::string text(brand);
                const size_t first = text.find_first_not_of(' ');
                const size_t last = text.find_last_not_of(' ');
                f.brand = first == std::string::npos ? std::string() : text.substr(first, last - first + 1);
            }
            if (max_leaf < 1) return f;

            cpuid(1, 0, r);
            const unsigned ecx1 = r[2], edx1 = r[3];
            f.sse2 = bit(edx1, 26);
            f.sse3 = bit(ecx1, 0);
            f.ssse3 = bit(ecx1, 9);
            f.sse41 = bit(ecx1, 19);
            f.sse42 = bit(ecx1, 20);
            f.popcnt = bit(ecx1, 23);
            // The vector extensions count only when the system saves their registers.
            const uint64_t saved = bit(ecx1, 27) ? xcr0() : 0;
            const bool ymm = (saved & 0x6) == 0x6;
            const bool zmm = ymm && (saved & 0xe0) == 0xe0;
            f.avx = ymm && bit(ecx1, 28);
            f.fma = f.avx && bit(ecx1, 12);
            f.f16c = f.avx && bit(ecx1, 29);

            if (max_leaf >= 7) {
                cpuid(7, 0, r);
                const unsigned max_sub = r[0], ebx7 = r[1], ecx7 = r[2], edx7 = r[3];
                f.bmi1 = bit(ebx7, 3);
                f.bmi2 = bit(ebx7, 8);
                f.avx2 = f.avx && bit(ebx7, 5);
                f.avx512f = f.avx2 && zmm && bit(ebx7, 16);
                f.avx512dq = f.avx512f && bit(ebx7, 17);
                f.avx512cd = f.avx512f && bit(ebx7, 28);
                f.avx512bw = f.avx512f && bit(ebx7, 30);
                f.avx512vl = f.avx512f && bit(ebx7, 31);
                f.avx512_vnni = f.avx512f && bit(ecx7, 11);
                f.avx512_fp16 = f.avx512f && bit(edx7, 23);
                if (max_sub >= 1) {
                    cpuid(7, 1, r);
                    f.avx_vnni = f.avx2 && bit(r[0], 4);
                    f.avx512_bf16 = f.avx512f && bit(r[0], 5);
                }
            }
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
            f.neon = true;
#endif
            const std::string names = environment("DLIB_DISABLE_CPU_FEATURES");
            if (!names.empty()) disable(f, names);
            return f;
        }
    }

// ----------------------------------------------------------------------------------------

    inline const cpu_features& get_cpu_features (
    )
    {
        static const cpu_features features = impl_cpu_features::detect();
        return features;
    }

// ----------------------------------------------------------------------------------------

    inline std::string describe (
        const cpu_features& f
    )
    {
        std::string text = f.brand.empty() ? (f.vendor.empty() ? std::string("processor") : f.vendor) : f.brand;
        if (!f.brand.empty() && !f.vendor.empty()) text += " (" + f.vendor + ")";
        const struct { bool on; const char* name; } list[] = {
            { f.sse2, "SSE2" }, { f.sse3, "SSE3" }, { f.ssse3, "SSSE3" }, { f.sse41, "SSE4.1" },
            { f.sse42, "SSE4.2" }, { f.popcnt, "POPCNT" }, { f.avx, "AVX" }, { f.f16c, "F16C" },
            { f.fma, "FMA" }, { f.bmi1, "BMI1" }, { f.bmi2, "BMI2" }, { f.avx2, "AVX2" },
            { f.avx512f, "AVX-512F" }, { f.avx512dq, "AVX-512DQ" }, { f.avx512cd, "AVX-512CD" },
            { f.avx512bw, "AVX-512BW" }, { f.avx512vl, "AVX-512VL" }, { f.avx512_vnni, "AVX-512 VNNI" },
            { f.avx512_bf16, "AVX-512 BF16" }, { f.avx512_fp16, "AVX-512 FP16" }, { f.avx_vnni, "AVX-VNNI" },
            { f.neon, "NEON" }
        };
        std::string found;
        for (const auto& e : list)
            if (e.on) found += std::string(found.empty() ? "" : " ") + e.name;
        return text + ": " + (found.empty() ? std::string("no vector extension") : found);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CPU_FEATURES_Hh_
