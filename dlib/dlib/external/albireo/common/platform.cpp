// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "platform.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
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
#include <shlobj.h>
#include <aclapi.h>
#include <bcrypt.h>
#include <sddl.h>
#else
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace albireo
{

#if defined(_WIN32)
    std::wstring widen(const std::string& s)
    {
        if (s.empty()) return std::wstring();
        const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
        std::wstring w(static_cast<size_t>(n), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &w[0], n);
        return w;
    }

    std::string narrow(const std::wstring& w)
    {
        if (w.empty()) return std::string();
        const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
            nullptr, 0, nullptr, nullptr);
        std::string s(static_cast<size_t>(n), '\0');
        WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), &s[0], n, nullptr, nullptr);
        return s;
    }
#endif

    std::string executable_path()
    {
#if defined(_WIN32)
        std::wstring buf(32768, L'\0');
        const DWORD n = GetModuleFileNameW(nullptr, &buf[0], static_cast<DWORD>(buf.size()));
        buf.resize(n);
        return narrow(buf);
#else
        char buf[4096];
        const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n <= 0) return std::string();
        buf[n] = 0;
        return buf;
#endif
    }

    std::string executable_dir()
    {
        return parent_path(executable_path());
    }

    std::string data_dir()
    {
#if defined(_WIN32)
        // The variable of the machine, which the services receive; that of the user is left out.
        std::wstring raw(32767, L'\0');
        DWORD type = 0, size = static_cast<DWORD>(raw.size() * sizeof(wchar_t));
        if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment",
                L"ALBIREO_HOME", RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND, &type, &raw[0], &size)
            == ERROR_SUCCESS)
        {
            raw.resize(std::min(raw.find(L'\0'), raw.size()));
            std::wstring home = raw;
            if (type == REG_EXPAND_SZ)
            {
                home.assign(32767, L'\0');
                const DWORD n = ExpandEnvironmentStringsW(raw.c_str(), &home[0], static_cast<DWORD>(home.size()));
                home.resize(n > 0 && n <= home.size() ? n - 1 : 0);
            }
            if (!home.empty()) return narrow(home);
        }
        PWSTR p = nullptr;
        std::string base;
        if (SHGetKnownFolderPath(FOLDERID_ProgramData, 0, nullptr, &p) == S_OK) base = narrow(p);
        if (p) CoTaskMemFree(p);
        if (base.empty()) base = "C:\\ProgramData";
        return join_path(base, "Albireo");
#else
        const char* home = std::getenv("ALBIREO_HOME");
        if (home && *home) return home;
        const char* h = std::getenv("HOME");
        return join_path(join_path(join_path(h ? h : "/tmp", ".local"), "share"), "albireo");
#endif
    }

    std::string join_path(const std::string& a, const std::string& b)
    {
        if (a.empty()) return b;
        if (b.empty()) return a;
#if defined(_WIN32)
        const char sep = '\\';
        if (a.back() == '\\' || a.back() == '/') return a + b;
#else
        const char sep = '/';
        if (a.back() == '/') return a + b;
#endif
        return a + sep + b;
    }

    std::string parent_path(const std::string& p)
    {
        const size_t pos = p.find_last_of("/\\");
        if (pos == std::string::npos) return std::string();
        if (pos == 0) return p.substr(0, 1);
        return p.substr(0, pos);
    }

    std::string file_name(const std::string& p)
    {
        const size_t pos = p.find_last_of("/\\");
        return pos == std::string::npos ? p : p.substr(pos + 1);
    }

    bool file_exists(const std::string& p)
    {
#if defined(_WIN32)
        const DWORD a = GetFileAttributesW(widen(p).c_str());
        return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
        struct stat st;
        return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
    }

    bool dir_exists(const std::string& p)
    {
#if defined(_WIN32)
        const DWORD a = GetFileAttributesW(widen(p).c_str());
        return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
        struct stat st;
        return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
    }

    bool make_dirs(const std::string& p)
    {
        if (p.empty() || dir_exists(p)) return true;
        const std::string parent = parent_path(p);
        if (!parent.empty() && parent != p && !make_dirs(parent)) return false;
#if defined(_WIN32)
        return CreateDirectoryW(widen(p).c_str(), nullptr) || GetLastError() == ERROR_ALREADY_EXISTS;
#else
        return mkdir(p.c_str(), 0755) == 0 || dir_exists(p);
#endif
    }

    uint64_t file_size(const std::string& p)
    {
#if defined(_WIN32)
        WIN32_FILE_ATTRIBUTE_DATA d;
        if (!GetFileAttributesExW(widen(p).c_str(), GetFileExInfoStandard, &d)) return 0;
        return (static_cast<uint64_t>(d.nFileSizeHigh) << 32) | d.nFileSizeLow;
#else
        struct stat st;
        return stat(p.c_str(), &st) == 0 ? static_cast<uint64_t>(st.st_size) : 0;
#endif
    }

    bool file_being_written(const std::string& p)
    {
#if defined(_WIN32)
        // Opening for reading while refusing to share the writing fails when another
        // program holds the file open for writing, as a copy in progress does.
        const HANDLE h = CreateFileW(widen(p).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return GetLastError() == ERROR_SHARING_VIOLATION;
        CloseHandle(h);
        return false;
#else
        struct stat st;
        return stat(p.c_str(), &st) == 0 && std::time(nullptr) - st.st_mtime < 30;
#endif
    }

    bool remove_file(const std::string& p)
    {
#if defined(_WIN32)
        return DeleteFileW(widen(p).c_str()) != 0;
#else
        return std::remove(p.c_str()) == 0;
#endif
    }

    bool rename_file(const std::string& from, const std::string& to)
    {
#if defined(_WIN32)
        return MoveFileExW(widen(from).c_str(), widen(to).c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        return std::rename(from.c_str(), to.c_str()) == 0;
#endif
    }

#if defined(_WIN32)
    namespace
    {
        // Called by CopyFileExW between the blocks it copies; the copy stops when cancel is set.
        DWORD CALLBACK copy_progress(LARGE_INTEGER, LARGE_INTEGER, LARGE_INTEGER, LARGE_INTEGER, DWORD, DWORD,
            HANDLE, HANDLE, LPVOID data)
        {
            const std::atomic<bool>* cancel = static_cast<const std::atomic<bool>*>(data);
            return *cancel ? PROGRESS_CANCEL : PROGRESS_CONTINUE;
        }
    }
#endif

    bool copy_file(const std::string& from, const std::string& to, const std::atomic<bool>* cancel)
    {
#if defined(_WIN32)
        return CopyFileExW(widen(from).c_str(), widen(to).c_str(), cancel ? copy_progress : nullptr,
            const_cast<std::atomic<bool>*>(cancel), nullptr, 0) != 0;
#else
        bool ok = false;
        {
            std::ifstream in(from, std::ios::binary);
            std::ofstream out(to, std::ios::binary | std::ios::trunc);
            if (!in || !out) return false;
            std::vector<char> buf(1 << 20);
            ok = true;
            while (ok && in)
            {
                in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
                const std::streamsize n = in.gcount();
                ok = !(cancel && *cancel) && (n == 0 || static_cast<bool>(out.write(buf.data(), n)));
            }
            out.close();
            ok = ok && !in.bad() && static_cast<bool>(out);
        }
        if (!ok) std::remove(to.c_str());
        return ok;
#endif
    }

    std::vector<std::string> list_dir(const std::string& p)
    {
        std::vector<std::string> out;
#if defined(_WIN32)
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW(widen(join_path(p, "*")).c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) return out;
        do
        {
            const std::string n = narrow(fd.cFileName);
            if (n != "." && n != "..") out.push_back(n);
        } while (FindNextFileW(h, &fd));
        FindClose(h);
#else
        DIR* d = opendir(p.c_str());
        if (!d) return out;
        while (dirent* e = readdir(d))
        {
            const std::string n = e->d_name;
            if (n != "." && n != "..") out.push_back(n);
        }
        closedir(d);
#endif
        return out;
    }

    bool remove_tree(const std::string& p)
    {
        if (is_link(p))
        {
#if defined(_WIN32)
            // The attributes of a link are its own: a link to a folder is removed as a folder.
            const std::wstring w = widen(p);
            const bool folder = (GetFileAttributesW(w.c_str()) & FILE_ATTRIBUTE_DIRECTORY) != 0;
            return (folder ? RemoveDirectoryW(w.c_str()) : DeleteFileW(w.c_str())) != 0;
#else
            return unlink(p.c_str()) == 0;
#endif
        }
        if (file_exists(p)) return remove_file(p);
        if (!dir_exists(p)) return true;
        bool ok = true;
        for (const auto& n : list_dir(p)) ok = remove_tree(join_path(p, n)) && ok;
#if defined(_WIN32)
        return RemoveDirectoryW(widen(p).c_str()) != 0 && ok;
#else
        return rmdir(p.c_str()) == 0 && ok;
#endif
    }

#if defined(_WIN32)
    namespace
    {
        // Opens p itself, a link included, with the access given.
        HANDLE open_entry(const std::string& p, DWORD access)
        {
            return CreateFileW(widen(p).c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        }

        // Whether sid is the account of this program.
        bool own_sid(PSID sid)
        {
            HANDLE token = nullptr;
            if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
            std::vector<BYTE> buf(512);
            DWORD size = 0;
            const bool same = GetTokenInformation(token, TokenUser, buf.data(), static_cast<DWORD>(buf.size()), &size)
                && EqualSid(sid, reinterpret_cast<TOKEN_USER*>(buf.data())->User.Sid);
            CloseHandle(token);
            return same;
        }

        bool system_sid(PSID sid)
        {
            if (!sid || !IsValidSid(sid)) return false;
            if (IsWellKnownSid(sid, WinLocalSystemSid) || IsWellKnownSid(sid, WinBuiltinAdministratorsSid)) return true;
            /*
                The account of this program, an administrator in the installer: with the
                policy "default owner: object creator", what it made belongs to that account.
            */
            if (own_sid(sid)) return true;
            // NT SERVICE\TrustedInstaller, the owner of the folders of Windows.
            PSID installer = nullptr;
            if (!ConvertStringSidToSidW(L"S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464",
                    &installer))
                return false;
            const bool same = EqualSid(sid, installer) != 0;
            LocalFree(installer);
            return same;
        }

        bool system_owner(HANDLE h)
        {
            PSID owner = nullptr;
            PSECURITY_DESCRIPTOR sd = nullptr;
            const bool ok = GetSecurityInfo(h, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION, &owner, nullptr, nullptr,
                nullptr, &sd) == ERROR_SUCCESS && system_sid(owner);
            if (sd) LocalFree(sd);
            return ok;
        }

        // Compares two paths as the file systems of Windows do, letters in either case.
        bool same_text(const std::wstring& a, const std::wstring& b)
        {
            return CompareStringOrdinal(a.c_str(), static_cast<int>(a.size()), b.c_str(), static_cast<int>(b.size()),
                TRUE) == CSTR_EQUAL;
        }

        // A full path: "\" separators, "." and ".." resolved, long names, no "\" at its end
        // but that of a drive root.
        std::wstring full_path(const std::string& p)
        {
            const std::wstring w = widen(p);
            DWORD n = GetFullPathNameW(w.c_str(), 0, nullptr, nullptr);
            if (n == 0) return std::wstring();
            std::wstring full(n, L'\0');
            n = GetFullPathNameW(w.c_str(), n, &full[0], nullptr);
            full.resize(n < full.size() ? n : 0);
            std::wstring longer(32768, L'\0');
            const DWORD m = GetLongPathNameW(full.c_str(), &longer[0], static_cast<DWORD>(longer.size()));
            if (m > 0 && m < longer.size()) full = longer.substr(0, m);
            while (full.size() > 3 && full.back() == L'\\') full.pop_back();
            return full;
        }

        // SYSTEM and the administrators full control, the users read, for a folder and its content.
        const wchar_t* const secured_folder_sddl = L"D:PAI(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;0x1200a9;;;BU)";

        /*
            Renames what other accounts placed under dir, their links included, with the
            suffix ".untrusted-<random>", and goes down into the folders of SYSTEM and of the
            administrators, without following their links.
        */
        void set_aside_foreign(const std::string& dir, int depth)
        {
            for (const auto& name : list_dir(dir))
            {
                if (name.find(".untrusted-") != std::string::npos) continue;
                const std::string p = join_path(dir, name);
                if (!owned_by_system(p))
                    MoveFileExW(widen(p).c_str(), widen(p + ".untrusted-" + random_hex(8)).c_str(), 0);
                else if (depth < 16 && !is_link(p) && dir_exists(p))
                    set_aside_foreign(p, depth + 1);
            }
        }

        // Whether a folder has the protected list of permissions wanted already.
        bool has_dacl(HANDLE h, PACL wanted)
        {
            PACL dacl = nullptr;
            PSECURITY_DESCRIPTOR sd = nullptr;
            if (GetSecurityInfo(h, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &dacl, nullptr, &sd)
                != ERROR_SUCCESS)
                return false;
            SECURITY_DESCRIPTOR_CONTROL control = 0;
            DWORD revision = 0;
            bool same = GetSecurityDescriptorControl(sd, &control, &revision) && (control & SE_DACL_PROTECTED) && dacl
                && wanted && dacl->AceCount == wanted->AceCount;
            for (WORD i = 0; same && i < wanted->AceCount; ++i)
            {
                void* a = nullptr;
                void* b = nullptr;
                same = GetAce(dacl, i, &a) && GetAce(wanted, i, &b)
                    && static_cast<ACE_HEADER*>(a)->AceSize == static_cast<ACE_HEADER*>(b)->AceSize
                    && std::memcmp(a, b, static_cast<ACE_HEADER*>(b)->AceSize) == 0;
            }
            LocalFree(sd);
            return same;
        }

        // Gives an existing folder, reached through trusted links, the permissions of secure_folder().
        bool secure_existing_folder(const std::string& path, std::string& error, bool check_content)
        {
            const std::string target = final_path(path);
            const UINT drive = target.size() > 3 && target[1] == ':' ? GetDriveTypeW(widen(target.substr(0, 3)).c_str())
                : DRIVE_UNKNOWN;
            if (drive != DRIVE_FIXED && drive != DRIVE_REMOVABLE && drive != DRIVE_RAMDISK)
            {
                error = path + " is not a folder of a local drive";
                return false;
            }
            // The folder itself is opened, so that its new permissions go to the folder checked.
            HANDLE h = open_entry(target, READ_CONTROL | WRITE_DAC | FILE_READ_ATTRIBUTES);
            FILE_ATTRIBUTE_TAG_INFO info = {};
            // A folder, and no reparse point.
            const DWORD kind = FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT;
            if (h == INVALID_HANDLE_VALUE || !GetFileInformationByHandleEx(h, FileAttributeTagInfo, &info, sizeof(info))
                || (info.FileAttributes & kind) != FILE_ATTRIBUTE_DIRECTORY)
            {
                if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
                error = "cannot open the folder " + target;
                return false;
            }
            bool ok = true;
            if (!system_owner(h))
            {
                // The folder of another account is taken over while it is empty, its owner
                // replaced by the administrators.
                CloseHandle(h);
                h = open_entry(target, READ_CONTROL | WRITE_DAC | WRITE_OWNER | FILE_READ_ATTRIBUTES);
                BYTE sid[SECURITY_MAX_SID_SIZE];
                DWORD sid_size = sizeof(sid);
                ok = h != INVALID_HANDLE_VALUE && list_dir(target).empty()
                    && CreateWellKnownSid(WinBuiltinAdministratorsSid, nullptr, sid, &sid_size)
                    && SetSecurityInfo(h, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION, sid, nullptr, nullptr, nullptr)
                        == ERROR_SUCCESS;
                if (!ok) error = target + " belongs to another account and cannot be taken over";
            }
            PSECURITY_DESCRIPTOR sd = nullptr;
            PACL dacl = nullptr;
            BOOL present = FALSE, defaulted = FALSE;
            if (ok && (!ConvertStringSecurityDescriptorToSecurityDescriptorW(secured_folder_sddl, SDDL_REVISION_1, &sd,
                    nullptr) || !GetSecurityDescriptorDacl(sd, &present, &dacl, &defaulted)))
                ok = false;
            // A folder that had these permissions already holds nothing that other accounts placed.
            const bool secured = ok && has_dacl(h, dacl);
            // The inheritable entries of the new list reach what the folder holds.
            if (ok && !secured)
                ok = SetSecurityInfo(h, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                    nullptr, nullptr, dacl, nullptr) == ERROR_SUCCESS;
            if (!ok && error.empty()) error = "cannot set the permissions of " + target;
            if (sd) LocalFree(sd);
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
            if (ok && (!secured || check_content)) set_aside_foreign(target, 0);
            return ok;
        }
    }
#endif

    bool is_link(const std::string& p)
    {
#if defined(_WIN32)
        // The entry itself, as its folder lists it: a wildcard would list others.
        if (p.empty() || p.find_first_of("*?") != std::string::npos) return false;
        std::string entry = p;
        while (entry.size() > 3 && (entry.back() == '\\' || entry.back() == '/')) entry.pop_back();
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW(widen(entry).c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) return false;
        FindClose(h);
        // Symbolic links, junctions and mount points name another entry; the other reparse
        // points, files of a cloud or deduplicated, hold their own data.
        return (fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) && IsReparseTagNameSurrogate(fd.dwReserved0);
#else
        struct stat st;
        return lstat(p.c_str(), &st) == 0 && S_ISLNK(st.st_mode);
#endif
    }

    std::string final_path(const std::string& p)
    {
#if defined(_WIN32)
        HANDLE h = CreateFileW(widen(p).c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE
            | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
        if (h == INVALID_HANDLE_VALUE) return std::string();
        std::wstring path(32768, L'\0');
        const DWORD n = GetFinalPathNameByHandleW(h, &path[0], static_cast<DWORD>(path.size()),
            FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        CloseHandle(h);
        if (n == 0 || n >= path.size()) return std::string();
        path.resize(n);
        // The system gives \\?\C:\... and \\?\UNC\server\share\...
        if (path.compare(0, 8, L"\\\\?\\UNC\\") == 0) path = L"\\\\" + path.substr(8);
        else if (path.compare(0, 4, L"\\\\?\\") == 0) path = path.substr(4);
        return narrow(path);
#else
        char* r = realpath(p.c_str(), nullptr);
        if (!r) return std::string();
        const std::string out = r;
        std::free(r);
        return out;
#endif
    }

    bool direct_path(const std::string& p)
    {
#if defined(_WIN32)
        const std::wstring f = widen(final_path(p));
        return !f.empty() && same_text(f, full_path(p));
#else
        std::string q = p;
        while (q.size() > 1 && q.back() == '/') q.pop_back();
        return !q.empty() && final_path(q) == q;
#endif
    }

    bool path_inside(const std::string& path, const std::string& folder)
    {
        std::string p = final_path(path);
        if (p.empty())
        {
            // A file to come: its folder, links resolved, and its name.
            const std::string parent = final_path(parent_path(path));
            if (parent.empty()) return false;
            p = join_path(parent, file_name(path));
        }
        const std::string f = final_path(folder);
        if (f.empty() || p.size() < f.size()) return false;
#if defined(_WIN32)
        const std::wstring wp = widen(p), wf = widen(f);
        if (wp.size() < wf.size() || !same_text(wp.substr(0, wf.size()), wf)) return false;
        return wp.size() == wf.size() || wf.back() == L'\\' || wp[wf.size()] == L'\\';
#else
        if (p.compare(0, f.size(), f) != 0) return false;
        return p.size() == f.size() || f.back() == '/' || p[f.size()] == '/';
#endif
    }

    bool owned_by_system(const std::string& p)
    {
#if defined(_WIN32)
        HANDLE h = open_entry(p, READ_CONTROL);
        if (h == INVALID_HANDLE_VALUE) return false;
        const bool ok = system_owner(h);
        CloseHandle(h);
        return ok;
#else
        (void)p;
        return true;
#endif
    }

    bool secure_folder(const std::string& p, std::string& error, bool check_content)
    {
#if defined(_WIN32)
        const std::wstring full = full_path(p);
        if (full.size() < 4 || full[1] != L':' || full[2] != L'\\')
        {
            error = p + " is not a folder of a local drive";
            return false;
        }
        // The links on the way, the folder included, must be those of SYSTEM or the administrators.
        for (size_t at = 3; at <= full.size(); ++at)
        {
            if (at < full.size() && full[at] != L'\\') continue;
            const std::string prefix = narrow(full.substr(0, at));
            if (is_link(prefix) && !owned_by_system(prefix))
            {
                error = prefix + " is a link that another account made";
                return false;
            }
        }
        const std::string path = narrow(full);
        if (!dir_exists(path))
        {
            if (is_link(path) || file_exists(path))
            {
                error = path + " is not a folder";
                return false;
            }
            if (!make_dirs(parent_path(path)))
            {
                error = "cannot create " + parent_path(path);
                return false;
            }
            PSECURITY_DESCRIPTOR sd = nullptr;
            if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(secured_folder_sddl, SDDL_REVISION_1, &sd,
                    nullptr))
            {
                error = "cannot create " + path;
                return false;
            }
            SECURITY_ATTRIBUTES sa = { sizeof(sa), sd, FALSE };
            const bool made = CreateDirectoryW(full.c_str(), &sa) != 0;
            const DWORD e = GetLastError();
            LocalFree(sd);
            if (made) return true;
            // Made meanwhile by another program: checked as any existing folder.
            if (e != ERROR_ALREADY_EXISTS || (is_link(path) && !owned_by_system(path)))
            {
                error = "cannot create " + path + " (error " + std::to_string(e) + ")";
                return false;
            }
        }
        return secure_existing_folder(path, error, check_content);
#else
        (void)check_content;
        if (!make_dirs(p) || !dir_exists(p))
        {
            error = "cannot create " + p;
            return false;
        }
        return true;
#endif
    }

    bool read_file(const std::string& p, std::string& data, uint64_t max_bytes)
    {
#if defined(_WIN32)
        std::ifstream in(widen(p).c_str(), std::ios::binary);
#else
        std::ifstream in(p, std::ios::binary);
#endif
        if (!in) return false;
        in.seekg(0, std::ios::end);
        const std::streamoff size = in.tellg();
        if (size < 0 || static_cast<uint64_t>(size) > max_bytes) return false;
        in.seekg(0, std::ios::beg);
        data.resize(static_cast<size_t>(size));
        if (size > 0) in.read(&data[0], size);
        return static_cast<bool>(in);
    }

    bool write_file_atomic(const std::string& p, const std::string& data)
    {
        const std::string tmp = p + ".tmp-" + random_hex(8);
        {
#if defined(_WIN32)
            std::ofstream out(widen(tmp).c_str(), std::ios::binary | std::ios::trunc);
#else
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
#endif
            if (!out) return false;
            out.write(data.data(), static_cast<std::streamsize>(data.size()));
            out.flush();
            if (!out)
            {
                out.close();
                remove_file(tmp);
                return false;
            }
        }
        if (!rename_file(tmp, p))
        {
            remove_file(tmp);
            return false;
        }
        return true;
    }

    bool write_file_restricted(const std::string& p, const std::string& data, bool interactive_readers)
    {
        const std::string tmp = p + ".tmp-" + random_hex(8);
#if defined(_WIN32)
        // SYSTEM and the administrators, and the users of the console when asked: the
        // descriptor replaces the permissions inherited from the folder.
        const wchar_t* sddl = interactive_readers ? L"D:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FR;;;IU)"
                                                  : L"D:P(A;;FA;;;SY)(A;;FA;;;BA)";
        PSECURITY_DESCRIPTOR sd = nullptr;
        if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, SDDL_REVISION_1, &sd, nullptr)) return false;
        SECURITY_ATTRIBUTES sa = { sizeof(sa), sd, FALSE };
        HANDLE h = CreateFileW(widen(tmp).c_str(), GENERIC_WRITE, 0, &sa, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        LocalFree(sd);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        const bool ok = WriteFile(h, data.data(), static_cast<DWORD>(data.size()), &written, nullptr)
            && written == data.size() && FlushFileBuffers(h);
        CloseHandle(h);
#else
        (void)interactive_readers;
        const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) return false;
        size_t done = 0;
        while (done < data.size())
        {
            const ssize_t w = ::write(fd, data.data() + done, data.size() - done);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) break;
            done += static_cast<size_t>(w);
        }
        const bool ok = done == data.size() && fsync(fd) == 0;
        ::close(fd);
#endif
        if (!ok || !rename_file(tmp, p))
        {
            remove_file(tmp);
            return false;
        }
        return true;
    }

    uint64_t free_disk_space(const std::string& p)
    {
#if defined(_WIN32)
        ULARGE_INTEGER avail;
        std::string dir = p;
        while (!dir.empty() && !dir_exists(dir)) dir = parent_path(dir);
        if (GetDiskFreeSpaceExW(widen(dir.empty() ? std::string("C:\\") : dir).c_str(), &avail, nullptr, nullptr))
            return avail.QuadPart;
        return 0;
#else
        std::string dir = p;
        while (!dir.empty() && !dir_exists(dir)) dir = parent_path(dir);
        struct statvfs s;
        if (statvfs(dir.empty() ? "/" : dir.c_str(), &s) != 0) return 0;
        return static_cast<uint64_t>(s.f_bavail) * s.f_frsize;
#endif
    }

    int64_t now_unix()
    {
        return static_cast<int64_t>(std::time(nullptr));
    }

    std::string iso_time(int64_t t)
    {
        const std::time_t tt = static_cast<std::time_t>(t);
        std::tm tm;
#if defined(_WIN32)
        gmtime_s(&tm, &tt);
#else
        gmtime_r(&tt, &tm);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
        return buf;
    }

    int compare_versions(const std::string& a, const std::string& b)
    {
        std::istringstream sa(a), sb(b);
        for (int i = 0; i < 4; ++i)
        {
            long x = 0, y = 0;
            std::string pa, pb;
            if (std::getline(sa, pa, '.')) x = std::atol(pa.c_str());
            if (std::getline(sb, pb, '.')) y = std::atol(pb.c_str());
            if (x != y) return x < y ? -1 : 1;
        }
        return 0;
    }

    std::string random_hex(size_t n)
    {
        static const char* hex = "0123456789abcdef";
        std::string out;
        std::vector<unsigned char> bytes((n + 1) / 2);
#if defined(_WIN32)
        if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()),
            BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
#else
        std::ifstream urandom("/dev/urandom", std::ios::binary);
        if (!urandom.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
#endif
        {
            std::random_device rd;
            for (auto& b : bytes) b = static_cast<unsigned char>(rd());
        }
        for (unsigned char b : bytes)
        {
            out += hex[b >> 4];
            out += hex[b & 15];
        }
        out.resize(n);
        return out;
    }

    int run_process(const std::string& program, const std::vector<std::string>& args, std::string* output,
        unsigned timeout_ms, const std::atomic<bool>* stop)
    {
        // The waits last a tenth of a second at most, so that stop ends the program at once.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        auto remaining_ms = [&]() {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count();
            return left > 0 && !(stop && stop->load()) ? static_cast<int>(std::min<long long>(left, 100)) : 0;
        };
#if defined(_WIN32)
        auto quote = [](const std::string& a) {
            if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
            std::string q = "\"";
            size_t backslashes = 0;
            for (char c : a)
            {
                if (c == '\\')
                {
                    ++backslashes;
                    continue;
                }
                if (c == '"') q.append(backslashes * 2 + 1, '\\');
                else q.append(backslashes, '\\');
                backslashes = 0;
                q += c;
            }
            q.append(backslashes * 2, '\\');
            return q + "\"";
        };
        std::string cmd = quote(program);
        for (const auto& a : args) cmd += " " + quote(a);
        SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
        HANDLE read_end = nullptr, write_end = nullptr;
        if (output && !CreatePipe(&read_end, &write_end, &sa, 0)) return -1;
        if (read_end) SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);

        // The child inherits the end of its pipe and nothing else of this process.
        STARTUPINFOEXW si = {};
        si.StartupInfo.cb = sizeof(si);
        std::vector<char> attributes;
        DWORD flags = CREATE_NO_WINDOW;
        if (output)
        {
            SIZE_T size = 0;
            InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
            attributes.resize(size);
            si.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
            if (!InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &size) ||
                !UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &write_end,
                    sizeof(write_end), nullptr, nullptr))
            {
                CloseHandle(read_end);
                CloseHandle(write_end);
                return -1;
            }
            si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            si.StartupInfo.hStdOutput = write_end;
            si.StartupInfo.hStdError = write_end;
            si.StartupInfo.hStdInput = nullptr;
            flags |= EXTENDED_STARTUPINFO_PRESENT;
        }
        PROCESS_INFORMATION pi = {};
        std::wstring wcmd = widen(cmd);
        const BOOL ok = CreateProcessW(widen(program).c_str(), &wcmd[0], nullptr, nullptr, output ? TRUE : FALSE,
            flags, nullptr, nullptr, &si.StartupInfo, &pi);
        if (si.lpAttributeList) DeleteProcThreadAttributeList(si.lpAttributeList);
        if (write_end) CloseHandle(write_end);
        if (!ok)
        {
            if (read_end) CloseHandle(read_end);
            return -1;
        }
        // The output is read while the process runs, so that the time limit holds.
        std::thread reader;
        if (output)
        {
            reader = std::thread([read_end, output]() {
                char buf[4096];
                DWORD n = 0;
                while (ReadFile(read_end, buf, sizeof(buf), &n, nullptr) && n > 0) output->append(buf, n);
            });
        }
        int code = -1;
        DWORD waited = WAIT_TIMEOUT;
        for (int wait = remaining_ms(); wait > 0 && waited == WAIT_TIMEOUT; wait = remaining_ms())
            waited = WaitForSingleObject(pi.hProcess, static_cast<DWORD>(wait));
        if (waited == WAIT_OBJECT_0)
        {
            DWORD c = 0;
            GetExitCodeProcess(pi.hProcess, &c);
            code = static_cast<int>(c);
        }
        else
        {
            TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, 5000);
        }
        if (reader.joinable()) reader.join();
        if (read_end) CloseHandle(read_end);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return code;
#else
        int fds[2] = { -1, -1 };
        if (output && pipe(fds) != 0) return -1;
        const pid_t pid = fork();
        if (pid < 0)
        {
            if (output)
            {
                close(fds[0]);
                close(fds[1]);
            }
            return -1;
        }
        if (pid == 0)
        {
            if (output)
            {
                dup2(fds[1], 1);
                dup2(fds[1], 2);
            }
            // The child keeps its standard streams only.
            long max_fd = sysconf(_SC_OPEN_MAX);
            if (max_fd < 0 || max_fd > 65536) max_fd = 65536;
            for (long fd = 3; fd < max_fd; ++fd) close(static_cast<int>(fd));
            std::vector<char*> argv;
            argv.push_back(const_cast<char*>(program.c_str()));
            for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            execv(program.c_str(), argv.data());
            _exit(127);
        }
        bool ended = false;
        if (output)
        {
            close(fds[1]);
            char buf[4096];
            for (;;)
            {
                pollfd p = { fds[0], POLLIN, 0 };
                const int wait = remaining_ms();
                if (wait == 0)
                {
                    ended = true;
                    break;
                }
                const int r = poll(&p, 1, wait);
                if (r < 0 && errno == EINTR) continue;
                if (r < 0) break;
                if (r == 0) continue;
                const ssize_t n = read(fds[0], buf, sizeof(buf));
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) break;
                output->append(buf, static_cast<size_t>(n));
            }
            close(fds[0]);
        }
        int status = 0;
        for (;;)
        {
            const pid_t r = waitpid(pid, &status, WNOHANG);
            if (r == pid) break;
            if (ended || remaining_ms() == 0)
            {
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                return -1;
            }
            usleep(10000);
        }
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    }

}
