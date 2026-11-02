// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <string>
#include <dlib/simd/cpu_features.h>

#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.cpu_features");

    class cpu_features_tester : public tester
    {
    public:
        cpu_features_tester (
        ) :
            tester ("test_cpu_features",
                    "Runs tests on the detection of the processor's instruction set extensions.")
        {}

        void perform_test (
        )
        {
            const cpu_features& f = get_cpu_features();
            dlog << LINFO << describe(f);

            // Detected once, the same object afterwards.
            DLIB_TEST(&f == &get_cpu_features());

            // Each flag implies those it is built on.
            DLIB_TEST(!f.avx2 || f.avx);
            DLIB_TEST(!f.fma || f.avx);
            DLIB_TEST(!f.f16c || f.avx);
            DLIB_TEST(!f.avx_vnni || f.avx2);
            DLIB_TEST(!f.avx512f || f.avx2);
            const bool avx512_member = f.avx512dq || f.avx512cd || f.avx512bw || f.avx512vl || f.avx512_vnni
                || f.avx512_bf16 || f.avx512_fp16;
            DLIB_TEST(!avx512_member || f.avx512f);

            // The extensions this test was compiled for are present on the machine running it.
#if defined(__SSE2__) || defined(_M_X64)
            DLIB_TEST(f.sse2);
#endif
#if defined(__AVX__)
            DLIB_TEST(f.avx);
#endif
#if defined(__AVX2__)
            DLIB_TEST(f.avx2);
#endif
#if defined(__FMA__)
            DLIB_TEST(f.fma);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
            DLIB_TEST(f.neon);
#endif

            // The masks of DLIB_DISABLE_CPU_FEATURES clear what they name and what is built on it.
            cpu_features g = f;
            impl_cpu_features::disable(g, "AVX2");
            DLIB_TEST(!g.avx2 && !g.avx_vnni && !g.avx512f && !g.avx512_vnni);
            DLIB_TEST(g.avx == f.avx && g.fma == f.fma && g.sse42 == f.sse42);
            g = f;
            impl_cpu_features::disable(g, "avx512_vnni, avx_vnni");
            DLIB_TEST(!g.avx512_vnni && !g.avx_vnni);
            DLIB_TEST(g.avx2 == f.avx2 && g.avx512f == f.avx512f && g.avx512vl == f.avx512vl);
            g = f;
            impl_cpu_features::disable(g, "avx;unknown");
            DLIB_TEST(!g.avx && !g.fma && !g.f16c && !g.avx2 && !g.avx512f);
            DLIB_TEST(g.sse2 == f.sse2 && g.bmi2 == f.bmi2);

            const std::string text = describe(f);
            DLIB_TEST(text.find(": ") != std::string::npos);
            DLIB_TEST((text.find("AVX2") != std::string::npos) == f.avx2);
            DLIB_TEST(describe(cpu_features()).find("no vector extension") != std::string::npos);
        }
    } a;
}
