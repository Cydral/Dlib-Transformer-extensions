// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    Checks the libraries that the programs and the engines of a build import, their delayed
    imports included, before the installer carries them: each one must be a library that Windows
    installs itself, among those this tool lists, found in System32, or an API set of Windows.  A
    runtime of a compiler (vcruntime, msvcp, msvcr, ucrtbase and the api-ms-win-crt sets, mfc,
    vcomp, libstdc++, libgcc, libwinpthread), which the programs link statically, a library that
    another product places in System32 (a redistributable of Visual C++, OpenCL or Vulkan of a
    graphics driver), and an engine of Pseudalis, which the service loads by its full path alone,
    are refused.  The installer then carries no library of a third party: the programs run on
    Windows as it is installed, and load no library that another installation could have placed.

        pseudalis-check-imports <program or library>...

    Prints the libraries each file imports, and returns 1 when a file imports another library or
    cannot be read as a program of Windows.  The delayed imports are those the headers list, as
    Visual C++ writes them; the linker of MinGW lists none there.  A library of Windows that a
    program comes to need joins the list below once it is known to come with Windows 10 and 11.
*/

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    // A file of the Portable Executable format, read whole, with the bounds of each read checked.
    class pe_file
    {
    public:
        bool read(const std::string& path, std::string& error)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in)
            {
                error = "cannot be read";
                return false;
            }
            bytes_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            uint32_t pe = 0, signature = 0;
            uint16_t magic = 0;
            if (!u16(0, magic) || magic != 0x5A4D || !u32(0x3C, pe) || !u32(pe, signature) || signature != 0x00004550)
            {
                error = "is not a program of Windows";
                return false;
            }
            uint16_t optional_size = 0, optional_magic = 0;
            if (!u16(pe + 6ull, sections_) || !u16(pe + 20ull, optional_size) || !u16(pe + 24ull, optional_magic))
                return truncated(error);
            optional_ = pe + 24ull;
            section_table_ = optional_ + optional_size;
            if (section_table_ + 40ull * sections_ > bytes_.size()) return truncated(error);
            pe32plus_ = optional_magic == 0x20B;
            if (!pe32plus_ && optional_magic != 0x10B)
            {
                error = "has an unknown optional header";
                return false;
            }
            if (pe32plus_)
            {
                if (!u64(optional_ + 24, image_base_)) return truncated(error);
            }
            else
            {
                uint32_t base = 0;
                if (!u32(optional_ + 28, base)) return truncated(error);
                image_base_ = base;
            }
            if (!u32(optional_ + (pe32plus_ ? 108 : 92), directories_)) return truncated(error);
            // The table of the directories, whole in the optional header and in the file.
            const uint64_t table = pe32plus_ ? 112 : 96;
            if (directories_ > 16 || table + 8ull * directories_ > optional_size ||
                optional_ + table + 8ull * directories_ > bytes_.size())
                return truncated(error);
            return true;
        }

        // The libraries of the import table, then those of the table of the delayed imports.
        bool imports(std::vector<std::string>& names, std::string& error) const
        {
            uint32_t rva = 0, size = 0;
            if (directory(1, rva, size) && rva != 0)
            {
                for (uint64_t at = rva, n = 0;; at += 20, ++n)
                {
                    uint32_t lookup = 0, name = 0, thunks = 0;
                    if (n >= 4096 || !rva_u32(at, lookup) || !rva_u32(at + 12, name) || !rva_u32(at + 16, thunks))
                    {
                        error = "has an unreadable import table";
                        return false;
                    }
                    if (lookup == 0 && name == 0 && thunks == 0) break;
                    std::string s;
                    if (!rva_string(name, s))
                    {
                        error = "has an unreadable import";
                        return false;
                    }
                    names.push_back(s);
                }
            }
            if (directory(13, rva, size) && rva != 0)
            {
                for (uint64_t at = rva, n = 0;; at += 32, ++n)
                {
                    uint32_t attributes = 0, name = 0;
                    if (n >= 4096 || !rva_u32(at, attributes) || !rva_u32(at + 4, name))
                    {
                        error = "has an unreadable table of delayed imports";
                        return false;
                    }
                    if (name == 0) break;
                    // The first format of the table held addresses rather than relative ones.
                    const uint64_t relative = (attributes & 1) ? name : name - image_base_;
                    std::string s;
                    if (relative > 0xFFFFFFFFull || !rva_string(relative, s))
                    {
                        error = "has an unreadable delayed import";
                        return false;
                    }
                    names.push_back(s);
                }
            }
            return true;
        }

    private:
        static bool truncated(std::string& error)
        {
            error = "has a truncated header";
            return false;
        }

        bool u16(uint64_t at, uint16_t& v) const
        {
            if (at + 2 > bytes_.size()) return false;
            v = static_cast<uint16_t>(byte(at) | (byte(at + 1) << 8));
            return true;
        }

        bool u32(uint64_t at, uint32_t& v) const
        {
            if (at + 4 > bytes_.size()) return false;
            v = byte(at) | (byte(at + 1) << 8) | (byte(at + 2) << 16) | (byte(at + 3) << 24);
            return true;
        }

        bool u64(uint64_t at, uint64_t& v) const
        {
            uint32_t low = 0, high = 0;
            if (!u32(at, low) || !u32(at + 4, high)) return false;
            v = (static_cast<uint64_t>(high) << 32) | low;
            return true;
        }

        uint32_t byte(uint64_t at) const { return static_cast<unsigned char>(bytes_[static_cast<size_t>(at)]); }

        // A directory whose entry the table holds (read() checked the table lies in the file).
        bool directory(uint32_t index, uint32_t& rva, uint32_t& size) const
        {
            if (index >= directories_) return false;
            const uint64_t at = optional_ + (pe32plus_ ? 112 : 96) + 8ull * index;
            return u32(at, rva) && u32(at + 4, size);
        }

        /*
            Where a relative address lies: in the data of its section in the file, at offset, or
            past that data, in the part of the section that Windows fills with zeros (zero).
        */
        bool offset_of(uint64_t rva, uint64_t& offset, bool& zero) const
        {
            for (uint32_t i = 0; i < sections_; ++i)
            {
                const uint64_t s = section_table_ + 40ull * i;
                uint32_t virtual_size = 0, address = 0, raw_size = 0, raw = 0;
                if (!u32(s + 8, virtual_size) || !u32(s + 12, address) || !u32(s + 16, raw_size) ||
                    !u32(s + 20, raw))
                    return false;
                const uint64_t extent = virtual_size > raw_size ? virtual_size : raw_size;
                if (rva < address || rva - address >= extent) continue;
                zero = rva - address >= raw_size;
                offset = static_cast<uint64_t>(raw) + (rva - address);
                return zero || offset < bytes_.size();
            }
            return false;
        }

        bool rva_u32(uint64_t rva, uint32_t& v) const
        {
            uint64_t at = 0;
            bool zero = false;
            if (!offset_of(rva, at, zero)) return false;
            if (zero)
            {
                v = 0;
                return true;
            }
            return u32(at, v);
        }

        bool rva_string(uint64_t rva, std::string& s) const
        {
            uint64_t at = 0;
            bool zero = false;
            if (!offset_of(rva, at, zero) || zero) return false;
            for (; at < bytes_.size() && s.size() < 256; ++at)
            {
                const char c = bytes_[static_cast<size_t>(at)];
                if (c == '\0') return !s.empty();
                s.push_back(c);
            }
            return false;
        }

        std::string bytes_;
        uint16_t sections_ = 0;
        uint64_t optional_ = 0;
        uint64_t section_table_ = 0;
        uint64_t image_base_ = 0;
        uint32_t directories_ = 0;
        bool pe32plus_ = false;
    };

    std::string lower(std::string s)
    {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    bool starts_with(const std::string& s, const char* prefix)
    {
        return s.compare(0, std::string(prefix).size(), prefix) == 0;
    }

    // The folder System32 of this Windows, which holds its libraries.
    std::string system_folder()
    {
#if defined(_WIN32)
        wchar_t buffer[MAX_PATH] = {};
        const UINT n = GetSystemDirectoryW(buffer, MAX_PATH);
        if (n == 0 || n >= MAX_PATH) return std::string();
        const int size = WideCharToMultiByte(CP_UTF8, 0, buffer, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
        std::string out(static_cast<size_t>(size), '\0');
        WideCharToMultiByte(CP_UTF8, 0, buffer, static_cast<int>(n), &out[0], size, nullptr, nullptr);
        return out;
#else
        return std::string();
#endif
    }

    bool system_library(const std::string& folder, const std::string& name)
    {
#if defined(_WIN32)
        const std::string path = folder + "\\" + name;
        const int size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring wide(static_cast<size_t>(size), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wide[0], size);
        const DWORD a = GetFileAttributesW(wide.c_str());
        return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
        (void)folder;
        (void)name;
        return false;
#endif
    }

    // The libraries that Windows 10 and 11 install themselves, which the programs may import.
    const char* const windows_libraries[] = {
        "advapi32.dll", "bcrypt.dll", "cfgmgr32.dll", "comctl32.dll", "comdlg32.dll", "crypt32.dll", "d2d1.dll",
        "d3d11.dll", "dbghelp.dll", "dnsapi.dll", "dwmapi.dll", "dwrite.dll", "dxgi.dll", "gdi32.dll", "gdiplus.dll",
        "imm32.dll", "iphlpapi.dll", "kernel32.dll", "kernelbase.dll", "mpr.dll", "msimg32.dll", "msvcrt.dll",
        "mswsock.dll", "ncrypt.dll", "netapi32.dll", "normaliz.dll", "ntdll.dll", "ole32.dll", "oleacc.dll",
        "oleaut32.dll", "powrprof.dll", "propsys.dll", "psapi.dll", "rpcrt4.dll", "secur32.dll", "setupapi.dll",
        "shcore.dll", "shell32.dll", "shlwapi.dll", "sspicli.dll", "user32.dll", "userenv.dll", "uxtheme.dll",
        "version.dll", "windowscodecs.dll", "winhttp.dll", "wininet.dll", "winmm.dll", "wintrust.dll",
        "wldap32.dll", "ws2_32.dll", "wsock32.dll", "wtsapi32.dll" };

    // Why a library may not be imported, or an empty string.
    std::string refusal(const std::string& folder, const std::string& name)
    {
        const std::string l = lower(name);
        static const char* const runtimes[] = { "vcruntime", "msvcp", "ucrtbase", "concrt", "vcomp", "vccorlib",
                                                "mfc", "vcamp", "libstdc++", "libgcc", "libwinpthread", "libgomp",
                                                "libssp", "api-ms-win-crt-" };
        for (const char* r : runtimes)
            if (starts_with(l, r)) return "a runtime of the compiler, which the programs must link statically";
        // msvcr100.dll, msvcr120.dll...: msvcrt.dll alone is a library of Windows.
        if (starts_with(l, "msvcr") && l.size() > 5 && std::isdigit(static_cast<unsigned char>(l[5])))
            return "a runtime of the compiler, which the programs must link statically";
        if (starts_with(l, "pseudalis-")) return "a library of Pseudalis, which the service loads by its full path";
        // The API sets of Windows, which its loader resolves itself.
        if (starts_with(l, "api-ms-win-") || starts_with(l, "ext-ms-")) return std::string();
        bool listed = false;
        for (const char* w : windows_libraries) listed = listed || l == w;
        if (!listed) return "no library that comes with Windows: the installer would have to carry it";
        if (!system_library(folder, name)) return "missing from System32";
        return std::string();
    }
}

int main(int argc, char** argv)
{
    const std::string folder = system_folder();
    if (folder.empty())
    {
        std::fprintf(stderr, "pseudalis-check-imports: the folder System32 is unknown\n");
        return 1;
    }
    int failures = 0;
    for (int i = 1; i < argc; ++i)
    {
        const std::string path = argv[i];
        const size_t slash = path.find_last_of("/\\");
        const std::string file = slash == std::string::npos ? path : path.substr(slash + 1);
        pe_file pe;
        std::vector<std::string> names;
        std::string error;
        if (!pe.read(path, error) || !pe.imports(names, error))
        {
            std::fprintf(stderr, "pseudalis-check-imports: %s %s\n", path.c_str(), error.c_str());
            ++failures;
            continue;
        }
        std::ostringstream list;
        for (const std::string& n : names)
        {
            list << ' ' << n;
            const std::string why = refusal(folder, n);
            if (!why.empty())
            {
                std::fprintf(stderr, "pseudalis-check-imports: %s imports %s, %s\n", file.c_str(), n.c_str(),
                             why.c_str());
                ++failures;
            }
        }
        std::printf("%s:%s\n", file.c_str(), list.str().c_str());
    }
    if (failures == 0) std::printf("pseudalis-check-imports: the programs import the libraries of Windows alone\n");
    return failures == 0 ? 0 : 1;
}
