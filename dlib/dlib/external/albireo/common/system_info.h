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
        bool avx512vnni = false;
        bool amx = false;
    };

    cpu_info read_cpu_info();

    // The instruction sets beyond SSE2 that the kernels of the engines and of their tests use.
    enum qk_isa_flags : unsigned
    {
        qk_isa_avx2 = 1,       // AVX2, FMA and F16C
        qk_isa_avx_vnni = 2    // AVX-VNNI
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
        return 0;
    }

    // The instructions of one of the qk_isa_flags, with the processors that have them.
    inline const char* qk_isa_text(unsigned flag)
    {
        switch (flag)
        {
        case qk_isa_avx2: return "AVX2, FMA and F16C";
        case qk_isa_avx_vnni: return "AVX-VNNI (Intel Core from Alder Lake on, AMD Zen 5)";
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
        default: return "";
        }
    }

    struct memory_info
    {
        uint64_t total = 0;       // physical memory, bytes
        uint64_t available = 0;   // physical memory available to a new allocation, bytes
        /*
            The memory the system can commit, physical memory and paging file (Windows) or
            swap (Linux) together: the limit, and what new allocations can still take of it,
            never below the physical memory available.
        */
        uint64_t commit_total = 0;
        uint64_t commit_available = 0;
        // What new allocations can still take of the limit, as the system reports it.
        uint64_t commit_left = 0;
    };

    memory_info read_memory_info();

    // The memory this process commits (its private bytes on Windows), 0 elsewhere or when unknown.
    uint64_t process_commit_bytes();

    // "Windows 11 Pro 24H2 (build 26100)", "Linux 6.8.0 x86_64", ...
    std::string operating_system_name();

    // Name of this computer.
    std::string host_name();

    /*!
        ensures
            - Under Windows, takes the calling thread out of the power throttling of the
              system (the high quality of service of SetThreadInformation()).  The system
              infers the quality of service of a thread that asks for none: it may lower the
              processor frequency of the threads of a service or of a program whose window
              is out of focus, and keep them on the efficient cores of a hybrid processor.
              A thread that computes an answer someone waits for calls this function once.
            - Returns true when the system took the request; false, with nothing changed, on
              another system, on a Windows without this setting, or when the environment
              variable ALBIREO_THREAD_QOS is "system", which leaves the choice to Windows.
    !*/
    bool leave_thread_throttling();

    /*!
        ensures
            - As leave_thread_throttling() for every thread of the process
              (SetProcessInformation()), for a program whose whole work is to compute: the
              service of the product and its measuring tools.
    !*/
    bool leave_process_throttling();

}

#endif // ALBIREO_COMMON_SYSTEM_INFO_H_
