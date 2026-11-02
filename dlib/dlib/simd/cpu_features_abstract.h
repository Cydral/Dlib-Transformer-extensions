// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CPU_FEATURES_ABSTRACT_Hh_
#ifdef DLIB_CPU_FEATURES_ABSTRACT_Hh_

#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct cpu_features
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The instruction set extensions of the processor a program runs on, as
                get_cpu_features() detects them when the program first asks. A program
                compiled for a baseline processor uses it to route its hot code to versions
                written for the extensions found, at run time, whatever the options its
                build was compiled with.

                On x86, a flag is true when the processor offers the extension and the
                operating system saves the registers it uses: the AVX family needs the
                system to save the 256-bit registers, and the AVX-512 family the 512-bit
                registers and the mask registers as well. A flag that the processor offers
                but the system leaves disabled is false, since its instructions would fault.

                Each flag also implies those it is built on: avx2, fma, f16c and avx_vnni
                imply avx, avx_vnni implies avx2, and every AVX-512 flag implies avx512f
                and avx2.

                neon is true on 64-bit ARM, where the extension is part of the
                architecture, and on 32-bit ARM builds compiled for it.

            THREAD SAFETY
                get_cpu_features() detects once and returns the same object afterwards; it
                can be called from any thread.
        !*/

        std::string vendor;     // the CPUID vendor string, "GenuineIntel" or "AuthenticAMD" for instance
        std::string brand;      // the CPUID brand string, the model name, without leading spaces

        bool sse2 = false;
        bool sse3 = false;
        bool ssse3 = false;
        bool sse41 = false;
        bool sse42 = false;
        bool popcnt = false;
        bool avx = false;
        bool f16c = false;
        bool fma = false;       // FMA3, the fused multiply-add of 256-bit vectors
        bool bmi1 = false;
        bool bmi2 = false;
        bool avx2 = false;
        bool avx512f = false;
        bool avx512dq = false;
        bool avx512cd = false;
        bool avx512bw = false;
        bool avx512vl = false;
        bool avx512_vnni = false;   // the byte and word dot products of AVX-512, VPDPBUSD among them
        bool avx512_bf16 = false;
        bool avx512_fp16 = false;
        bool avx_vnni = false;      // the same dot products coded in VEX, on 256-bit vectors, without AVX-512
        bool neon = false;
    };

// ----------------------------------------------------------------------------------------

    const cpu_features& get_cpu_features (
    );
    /*!
        ensures
            - returns the features of the processor the program runs on, detected on the
              first call and kept for the life of the program.
            - On a processor other than x86 or ARM, every flag is false and vendor and
              brand are empty.
            - If the environment variable DLIB_DISABLE_CPU_FEATURES is set when the first
              call is made, the features it names are reported as absent, along with those
              built on them: "avx2" also clears avx_vnni and the AVX-512 flags, "avx512f"
              every AVX-512 flag, "avx" the whole AVX family. Names are those of the
              members above, separated by commas or spaces, and "avx512" stands for
              avx512f. This lets a program be run, on one machine, through each of the
              code paths it chooses between, to time them or to check that they agree.
    !*/

// ----------------------------------------------------------------------------------------

    std::string describe (
        const cpu_features& features
    );
    /*!
        ensures
            - returns a line naming the processor and the extensions found, such as
              "Intel(R) Core(TM) i7-1165G7 @ 2.80GHz (GenuineIntel): SSE2 SSE3 SSSE3
              SSE4.1 SSE4.2 POPCNT AVX F16C FMA BMI1 BMI2 AVX2 AVX-512F ... AVX-512 VNNI".
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CPU_FEATURES_ABSTRACT_Hh_
