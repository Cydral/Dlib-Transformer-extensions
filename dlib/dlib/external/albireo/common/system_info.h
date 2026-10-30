// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_COMMON_SYSTEM_INFO_H_
#define ALBIREO_COMMON_SYSTEM_INFO_H_

#include <cstdint>
#include <string>

namespace albireo
{

    struct cpu_info
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The processor of the machine as the engines need to know it: its name, its
                cores, and the instruction sets both the processor and the operating system
                support (an extension of the vector registers is only usable when the
                system saves them on a context switch, which XGETBV tells).
        !*/
        std::string vendor;
        std::string brand;
        int physical_cores = 1;
        int logical_cores = 1;
        bool sse42 = false;
        bool avx = false;
        bool avx2 = false;
        bool fma = false;
        bool f16c = false;
        bool avx_vnni = false;
        bool avx512f = false;
        bool avx512bw = false;
        bool avx512dq = false;
        bool avx512vl = false;
        bool avx512vnni = false;
        bool amx = false;
    };

    cpu_info read_cpu_info();

    // The instruction sets beyond SSE2 that the kernels of the engines use.
    enum qk_isa_flags : unsigned
    {
        qk_isa_avx2 = 1,       // AVX2, FMA and F16C
        qk_isa_avx_vnni = 2,   // AVX-VNNI
        qk_isa_avx512 = 4      // AVX-512 F, BW, DQ and VL, with VNNI
    };

    /*!
        ensures
            - Returns the first of the qk_isa_flags of isa that cpu lacks, or 0 when cpu has
              them all.  Defined here, it is compiled with the instructions of the program
              that calls it, as are the two functions that follow.
    !*/
    inline unsigned qk_lacking_isa(const cpu_info& cpu, unsigned isa)
    {
        if ((isa & qk_isa_avx2) && !(cpu.avx2 && cpu.fma && cpu.f16c)) return qk_isa_avx2;
        if ((isa & qk_isa_avx_vnni) && !cpu.avx_vnni) return qk_isa_avx_vnni;
        if ((isa & qk_isa_avx512) && !(cpu.avx512f && cpu.avx512bw && cpu.avx512dq && cpu.avx512vl && cpu.avx512vnni))
            return qk_isa_avx512;
        return 0;
    }

    // The instructions of one of the qk_isa_flags, with the processors that have them.
    inline const char* qk_isa_text(unsigned flag)
    {
        switch (flag)
        {
        case qk_isa_avx2: return "AVX2, FMA and F16C";
        case qk_isa_avx_vnni: return "AVX-VNNI (Intel Core from Alder Lake on, AMD Zen 5)";
        case qk_isa_avx512:
            return "AVX-512 with VNNI (Intel Xeon from Cascade Lake on, Intel Core Ice Lake, Tiger Lake and Rocket "
                "Lake, AMD Zen 4 and later)";
        default: return "";
        }
    }

    // The code of one of the qk_isa_flags in the descriptions of the engines.
    inline const char* qk_isa_code(unsigned flag)
    {
        switch (flag)
        {
        case qk_isa_avx2: return "isa_avx2";
        case qk_isa_avx_vnni: return "isa_avx_vnni";
        case qk_isa_avx512: return "isa_avx512";
        default: return "";
        }
    }

    struct memory_info
    {
        uint64_t total = 0;       // physical memory, bytes
        uint64_t available = 0;   // available to a new allocation, bytes
    };

    memory_info read_memory_info();

    // "Windows 11 Pro 24H2 (build 26100)", "Linux 6.8.0 x86_64", ...
    std::string operating_system_name();

    // Name of this computer.
    std::string host_name();

}

#endif // ALBIREO_COMMON_SYSTEM_INFO_H_
