// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "system_info.h"

#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#include <sys/utsname.h>
#include <unistd.h>
#endif
#if !defined(_MSC_VER) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace albireo
{

    namespace
    {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
        void cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4])
        {
#if defined(_MSC_VER)
            int regs[4];
            __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(sub));
            for (int i = 0; i < 4; ++i) r[i] = static_cast<uint32_t>(regs[i]);
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        }

        // Read without the intrinsic, which GCC only allows in code compiled for XSAVE.
        uint64_t xgetbv0()
        {
#if defined(_MSC_VER)
            return _xgetbv(0);
#else
            uint32_t eax, edx;
            __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
            return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
        }
#endif

        std::string trim(const std::string& s)
        {
            size_t a = 0, b = s.size();
            while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\0')) ++a;
            while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\0')) --b;
            return s.substr(a, b - a);
        }
    }

    cpu_info read_cpu_info()
    {
        cpu_info c;
        c.logical_cores = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        c.physical_cores = c.logical_cores;

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
        uint32_t r[4];
        cpuid(0, 0, r);
        const uint32_t max_leaf = r[0];
        char vendor[13];
        std::memcpy(vendor, &r[1], 4);
        std::memcpy(vendor + 4, &r[3], 4);
        std::memcpy(vendor + 8, &r[2], 4);
        vendor[12] = 0;
        c.vendor = vendor;

        cpuid(0x80000000u, 0, r);
        if (r[0] >= 0x80000004u)
        {
            char brand[49];
            for (uint32_t i = 0; i < 3; ++i)
            {
                cpuid(0x80000002u + i, 0, r);
                std::memcpy(brand + 16 * i, r, 16);
            }
            brand[48] = 0;
            c.brand = trim(brand);
        }

        cpuid(1, 0, r);
        const bool osxsave = (r[2] >> 27) & 1;
        c.sse42 = (r[2] >> 20) & 1;
        const bool cpu_avx = (r[2] >> 28) & 1;
        const bool cpu_fma = (r[2] >> 12) & 1;
        const bool cpu_f16c = (r[2] >> 29) & 1;
        uint64_t xcr0 = 0;
        if (osxsave) xcr0 = xgetbv0();
        const bool os_ymm = (xcr0 & 0x6) == 0x6;
        const bool os_zmm = (xcr0 & 0xE6) == 0xE6;
        const bool os_amx = (xcr0 & 0x60000) == 0x60000;
        c.avx = cpu_avx && os_ymm;
        c.fma = cpu_fma && c.avx;
        c.f16c = cpu_f16c && c.avx;
        if (max_leaf >= 7)
        {
            cpuid(7, 0, r);
            c.avx2 = ((r[1] >> 5) & 1) && c.avx;
            c.avx512f = ((r[1] >> 16) & 1) && os_zmm;
            c.avx512bw = ((r[1] >> 30) & 1) && os_zmm;
            c.avx512vnni = ((r[2] >> 11) & 1) && os_zmm;
            c.amx = ((r[3] >> 24) & 1) && os_amx;
            cpuid(7, 1, r);
            c.avx_vnni = ((r[0] >> 4) & 1) && c.avx;
        }
#endif

#if defined(_WIN32)
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len > 0)
        {
            std::vector<char> buf(len);
            auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
            if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len))
            {
                int cores = 0;
                for (DWORD off = 0; off < len;)
                {
                    auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
                    if (e->Relationship == RelationProcessorCore) ++cores;
                    off += e->Size;
                }
                if (cores > 0) c.physical_cores = cores;
            }
        }
#else
        // Distinct (package, core) pairs of the online processors.
        std::set<std::pair<int, int>> cores;
        for (int cpu = 0; cpu < c.logical_cores; ++cpu)
        {
            const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
            std::ifstream pkg(base + "physical_package_id"), core(base + "core_id");
            int p = -1, k = -1;
            if (pkg >> p && core >> k) cores.insert(std::make_pair(p, k));
        }
        if (!cores.empty()) c.physical_cores = static_cast<int>(cores.size());
#endif
        if (c.physical_cores < 1) c.physical_cores = 1;
        return c;
    }

    memory_info read_memory_info()
    {
        memory_info m;
#if defined(_WIN32)
        MEMORYSTATUSEX s;
        s.dwLength = sizeof(s);
        if (GlobalMemoryStatusEx(&s))
        {
            m.total = s.ullTotalPhys;
            m.available = s.ullAvailPhys;
        }
#else
        std::ifstream in("/proc/meminfo");
        std::string key;
        uint64_t value;
        std::string unit;
        while (in >> key >> value >> unit)
        {
            if (key == "MemTotal:") m.total = value * 1024;
            else if (key == "MemAvailable:") m.available = value * 1024;
        }
#endif
        return m;
    }

    std::string operating_system_name()
    {
#if defined(_WIN32)
        std::string product = "Windows";
        HKEY key;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", 0,
                KEY_READ | KEY_WOW64_64KEY, &key) == ERROR_SUCCESS)
        {
            auto read = [&](const wchar_t* name) -> std::string {
                wchar_t buf[256];
                DWORD size = sizeof(buf);
                DWORD type = 0;
                if (RegQueryValueExW(key, name, nullptr, &type, reinterpret_cast<LPBYTE>(buf), &size) != ERROR_SUCCESS
                    || type != REG_SZ)
                    return std::string();
                buf[255] = 0;
                char out[512];
                const int n = WideCharToMultiByte(CP_UTF8, 0, buf, -1, out, sizeof(out), nullptr, nullptr);
                return n > 0 ? std::string(out) : std::string();
            };
            const std::string name = read(L"ProductName");
            const std::string display = read(L"DisplayVersion");
            const std::string build = read(L"CurrentBuild");
            RegCloseKey(key);
            if (!name.empty()) product = name;
            // Windows 11 still reports "Windows 10" in ProductName; the build tells.
            if (!build.empty() && std::atoi(build.c_str()) >= 22000)
            {
                const size_t p = product.find("Windows 10");
                if (p != std::string::npos) product.replace(p, 10, "Windows 11");
            }
            if (!display.empty()) product += " " + display;
            if (!build.empty()) product += " (build " + build + ")";
        }
        return product;
#else
        struct utsname u;
        if (uname(&u) == 0) return std::string(u.sysname) + " " + u.release + " " + u.machine;
        return "Unix";
#endif
    }

    std::string host_name()
    {
#if defined(_WIN32)
        wchar_t buf[256];
        DWORD size = 256;
        if (GetComputerNameExW(ComputerNameDnsHostname, buf, &size))
        {
            char out[512];
            const int n = WideCharToMultiByte(CP_UTF8, 0, buf, -1, out, sizeof(out), nullptr, nullptr);
            if (n > 0) return out;
        }
        return "localhost";
#else
        char buf[256];
        if (gethostname(buf, sizeof(buf)) == 0)
        {
            buf[255] = 0;
            return buf;
        }
        return "localhost";
#endif
    }

}
