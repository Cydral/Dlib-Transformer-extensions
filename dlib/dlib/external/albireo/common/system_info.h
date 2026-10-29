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
        bool avx512vnni = false;
        bool amx = false;
    };

    cpu_info read_cpu_info();

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
