// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <winsock2.h>
#include <ws2tcpip.h>

#include "setup_actions.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// The COM declarations come first: the first declarations of exdisp.h in the Windows SDK use
// the interface macro of objbase.h, which windows.h leaves out with WIN32_LEAN_AND_MEAN.
// shellapi.h precedes shldisp.h, whose ShellExecute method then receives the same name as
// the calls.
#include <ole2.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <exdisp.h>
#include <shldisp.h>
#include <shlguid.h>
#include <aclapi.h>
#include <sddl.h>
#include <wincrypt.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "../common/builtin_models.h"
#include "../common/json.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "../host/certificates.h"
#include "../host/crypto.h"
#include "resource.h"

namespace albireo
{
namespace setup
{

    namespace
    {
        const wchar_t uninstall_key[] = L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Albireo";
        const wchar_t run_key[] = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";

        std::string known_folder(REFKNOWNFOLDERID id, const char* fallback)
        {
            PWSTR p = nullptr;
            std::string out;
            if (SHGetKnownFolderPath(id, 0, nullptr, &p) == S_OK) out = narrow(p);
            if (p) CoTaskMemFree(p);
            return out.empty() ? fallback : out;
        }

        // The shortcuts of the Start menu and of the desktop, shared by all the users.
        std::string start_menu_link()
        {
            const char* fallback = "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs";
            return join_path(known_folder(FOLDERID_CommonPrograms, fallback), "Albireo.lnk");
        }

        std::string desktop_link()
        {
            return join_path(known_folder(FOLDERID_PublicDesktop, "C:\\Users\\Public\\Desktop"), "Albireo.lnk");
        }

        std::wstring registry_string(HKEY root, const wchar_t* key, const wchar_t* name)
        {
            wchar_t buf[2048];
            DWORD size = sizeof(buf);
            const DWORD flags = RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY;
            if (RegGetValueW(root, key, name, flags, nullptr, buf, &size) != ERROR_SUCCESS) return std::wstring();
            return buf;
        }

        struct service_handles
        {
            SC_HANDLE scm = nullptr;
            SC_HANDLE svc = nullptr;
            ~service_handles()
            {
                if (svc) CloseServiceHandle(svc);
                if (scm) CloseServiceHandle(scm);
            }
        };

        bool open_service(service_handles& h, DWORD access)
        {
            h.scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT);
            if (!h.scm) return false;
            h.svc = OpenServiceW(h.scm, widen(service_name).c_str(), access);
            return h.svc != nullptr;
        }

        std::string last_error(const char* what)
        {
            return std::string(what) + " (error " + std::to_string(GetLastError()) + ")";
        }

        bool wait_for_process(DWORD pid, unsigned timeout_ms)
        {
            if (!pid) return true;
            HANDLE p = OpenProcess(SYNCHRONIZE, FALSE, pid);
            if (!p) return true;
            const bool done = WaitForSingleObject(p, timeout_ms) == WAIT_OBJECT_0;
            CloseHandle(p);
            return done;
        }

        bool create_link(const std::wstring& link, const std::wstring& target, const std::wstring& args,
            const std::wstring& description)
        {
            IShellLinkW* sl = nullptr;
            if (FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_IShellLinkW,
                    reinterpret_cast<void**>(&sl))))
                return false;
            sl->SetPath(target.c_str());
            sl->SetArguments(args.c_str());
            sl->SetDescription(description.c_str());
            sl->SetIconLocation(target.c_str(), 0);
            const std::wstring dir = target.substr(0, target.find_last_of(L'\\'));
            sl->SetWorkingDirectory(dir.c_str());
            IPersistFile* pf = nullptr;
            bool ok = false;
            if (SUCCEEDED(sl->QueryInterface(IID_IPersistFile, reinterpret_cast<void**>(&pf))))
            {
                ok = SUCCEEDED(pf->Save(link.c_str(), TRUE));
                pf->Release();
            }
            sl->Release();
            return ok;
        }

        void remove_authority_certificates(HCERTSTORE store)
        {
            for (;;)
            {
                PCCERT_CONTEXT c = CertFindCertificateInStore(store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, 0,
                    CERT_FIND_SUBJECT_STR_W, L"Albireo Local Authority", nullptr);
                if (!c) break;
                CertDeleteCertificateFromStore(c);
            }
        }

        // Compares two paths as the file systems of Windows do, letters in either case.
        bool same_text(const std::wstring& a, const std::wstring& b)
        {
            return CompareStringOrdinal(a.c_str(), static_cast<int>(a.size()), b.c_str(), static_cast<int>(b.size()),
                TRUE) == CSTR_EQUAL;
        }

        // Whether a path is folder or lies under it, both full paths without "\" at their end.
        bool same_or_inside(const std::wstring& path, const std::wstring& folder)
        {
            return path.size() >= folder.size() && same_text(path.substr(0, folder.size()), folder)
                && (path.size() == folder.size() || path[folder.size()] == L'\\');
        }

        // The restarts after a failure: twice, then the service stays stopped; the count resets daily.
        void set_failure_actions(SC_HANDLE svc, bool restart)
        {
            SC_ACTION actions[3] = { { SC_ACTION_RESTART, 10000 }, { SC_ACTION_RESTART, 60000 },
                { SC_ACTION_NONE, 0 } };
            SERVICE_FAILURE_ACTIONSW fa = {};
            fa.dwResetPeriod = 86400;
            // No action, with the array given, deletes the actions.
            fa.cActions = restart ? 3 : 0;
            fa.lpsaActions = actions;
            ChangeServiceConfig2W(svc, SERVICE_CONFIG_FAILURE_ACTIONS, &fa);
            SERVICE_FAILURE_ACTIONS_FLAG flag = { restart ? TRUE : FALSE };
            ChangeServiceConfig2W(svc, SERVICE_CONFIG_FAILURE_ACTIONS_FLAG, &flag);
        }
    }

    std::string default_program_dir()
    {
        return join_path(known_folder(FOLDERID_ProgramFiles, "C:\\Program Files"), "Albireo");
    }

    std::string default_models_dir()
    {
        return join_path(data_dir(), "models");
    }

    std::string folder_problem(const std::string& dir, bool program)
    {
        if (dir.size() < 3 || !std::isalpha(static_cast<unsigned char>(dir[0])) || dir[1] != ':'
            || (dir[2] != '\\' && dir[2] != '/'))
            return "folder_not_local";
        const UINT drive = GetDriveTypeW(widen(dir.substr(0, 2) + "\\").c_str());
        if (drive != DRIVE_FIXED && drive != DRIVE_REMOVABLE && drive != DRIVE_RAMDISK) return "folder_not_local";
        std::wstring full(32768, L'\0');
        const DWORD n = GetFullPathNameW(widen(dir).c_str(), static_cast<DWORD>(full.size()), &full[0], nullptr);
        if (n == 0 || n >= full.size()) return "folder_not_local";
        full.resize(n);
        while (full.size() > 3 && full.back() == L'\\') full.pop_back();
        if (full.size() <= 3) return "folder_system";
        // The folders of Windows and of the users, and those that hold them.
        static const KNOWNFOLDERID* const system_folders[] = { &FOLDERID_Windows, &FOLDERID_ProgramFiles,
            &FOLDERID_ProgramFilesX86, &FOLDERID_ProgramFilesCommon, &FOLDERID_ProgramData, &FOLDERID_UserProfiles,
            &FOLDERID_Public, &FOLDERID_Profile, &FOLDERID_Desktop, &FOLDERID_Documents, &FOLDERID_Downloads,
            &FOLDERID_LocalAppData, &FOLDERID_RoamingAppData };
        std::vector<std::string> folders = { data_dir() };
        for (const KNOWNFOLDERID* id : system_folders) folders.push_back(known_folder(*id, ""));
        for (const auto& f : folders)
            if (!f.empty() && same_or_inside(widen(f), full)) return "folder_system";
        // The links on its path, as secure_folder() accepts them.
        for (size_t at = 3; at <= full.size(); ++at)
        {
            if (at < full.size() && full[at] != L'\\') continue;
            const std::string prefix = narrow(full.substr(0, at));
            if (is_link(prefix) && !owned_by_system(prefix)) return "folder_foreign";
        }
        const std::string path = narrow(full);
        if (!dir_exists(path)) return file_exists(path) ? "folder_not_empty" : std::string();
        const bool empty = list_dir(path).empty();
        // An installation of Albireo, or what one left: its list of files, or the models that came with it.
        const bool albireo = file_exists(join_path(path, "albireo-service.exe"))
            || file_exists(join_path(path, "albireo-files.json"))
            || file_exists(join_path(join_path(path, builtin_models_folder), builtin_manifest_name));
        if (program && !empty && !albireo) return "folder_not_empty";
        if (!empty && !owned_by_system(final_path(path))) return "folder_foreign";
        return std::string();
    }

    std::string installed_dir()
    {
        return narrow(registry_string(HKEY_LOCAL_MACHINE, uninstall_key, L"InstallLocation"));
    }

    std::string installed_version()
    {
        return narrow(registry_string(HKEY_LOCAL_MACHINE, uninstall_key, L"DisplayVersion"));
    }

    install_options installed_options()
    {
        install_options o;
        const std::string dir = installed_dir();
        o.program_dir = dir.empty() ? default_program_dir() : dir;
        o.models_dir = default_models_dir();
        std::string text;
        if (read_file(join_path(data_dir(), "settings.json"), text, 1u << 20))
        {
            const json s = json::parse_or_null(text);
            const long long port = s["port"].as_int(0);
            if (port >= 1024 && port <= 65535) o.port = static_cast<int>(port);
            if (!s["models_dir"].as_string().empty()) o.models_dir = s["models_dir"].as_string();
            o.default_model = s["default_model"].as_bool(true);
            o.offline = s["offline"].as_bool(false);
        }
        if (!dir.empty())
        {
            o.autostart = !registry_string(HKEY_LOCAL_MACHINE, run_key, L"Albireo").empty();
            o.desktop_shortcut = file_exists(desktop_link());
        }
        return o;
    }

    bool service_exists()
    {
        service_handles h;
        return open_service(h, SERVICE_QUERY_STATUS);
    }

    bool service_running()
    {
        service_handles h;
        SERVICE_STATUS s = {};
        return open_service(h, SERVICE_QUERY_STATUS) && QueryServiceStatus(h.svc, &s)
            && s.dwCurrentState != SERVICE_STOPPED;
    }

    void suspend_recovery()
    {
        service_handles h;
        if (open_service(h, SERVICE_CHANGE_CONFIG | SERVICE_START)) set_failure_actions(h.svc, false);
    }

    void restore_recovery()
    {
        service_handles h;
        if (open_service(h, SERVICE_CHANGE_CONFIG | SERVICE_START)) set_failure_actions(h.svc, true);
    }

    bool stop_service(unsigned timeout_ms, std::string& error)
    {
        service_handles h;
        if (!open_service(h, SERVICE_STOP | SERVICE_QUERY_STATUS)) return true;
        SERVICE_STATUS_PROCESS sp = {};
        DWORD needed = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        DWORD pid = 0;
        for (;;)
        {
            LPBYTE status = reinterpret_cast<LPBYTE>(&sp);
            if (!QueryServiceStatusEx(h.svc, SC_STATUS_PROCESS_INFO, status, sizeof(sp), &needed))
            {
                error = last_error("cannot query the service");
                return false;
            }
            if (sp.dwProcessId) pid = sp.dwProcessId;
            if (sp.dwCurrentState == SERVICE_STOPPED) break;
            if (sp.dwCurrentState == SERVICE_RUNNING || sp.dwCurrentState == SERVICE_PAUSED)
            {
                SERVICE_STATUS ss;
                ControlService(h.svc, SERVICE_CONTROL_STOP, &ss);
            }
            if (std::chrono::steady_clock::now() > deadline)
            {
                // A service that does not stop in time is ended, so that its files can be replaced.
                if (pid)
                {
                    HANDLE p = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid);
                    if (p)
                    {
                        TerminateProcess(p, 1);
                        WaitForSingleObject(p, 5000);
                        CloseHandle(p);
                    }
                }
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        wait_for_process(pid, 10000);
        return true;
    }

    bool start_service(std::string& error)
    {
        service_handles h;
        if (!open_service(h, SERVICE_START | SERVICE_QUERY_STATUS))
        {
            error = last_error("cannot open the service");
            return false;
        }
        if (!StartServiceW(h.svc, 0, nullptr) && GetLastError() != ERROR_SERVICE_ALREADY_RUNNING)
        {
            error = last_error("cannot start the service");
            return false;
        }
        return true;
    }

    bool register_service(const std::string& program_dir, std::string& error)
    {
        SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
        if (!scm)
        {
            error = last_error("cannot open the service manager");
            return false;
        }
        const std::wstring name = widen(service_name);
        const std::wstring path = L"\"" + widen(join_path(program_dir, "albireo-service.exe")) + L"\"";
        SC_HANDLE svc = OpenServiceW(scm, name.c_str(), SERVICE_ALL_ACCESS);
        if (svc)
        {
            ChangeServiceConfigW(svc, SERVICE_WIN32_OWN_PROCESS, SERVICE_AUTO_START, SERVICE_ERROR_NORMAL, path.c_str(),
                nullptr, nullptr, nullptr, nullptr, nullptr, L"Albireo");
        }
        else
        {
            svc = CreateServiceW(scm, name.c_str(), L"Albireo", SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS,
                SERVICE_AUTO_START, SERVICE_ERROR_NORMAL, path.c_str(), nullptr, nullptr, nullptr, nullptr, nullptr);
        }
        if (!svc)
        {
            error = last_error("cannot register the service");
            CloseServiceHandle(scm);
            return false;
        }
        std::wstring text = L"Runs the language models of Albireo and serves its local interface.";
        SERVICE_DESCRIPTIONW desc = { &text[0] };
        ChangeServiceConfig2W(svc, SERVICE_CONFIG_DESCRIPTION, &desc);
        SERVICE_DELAYED_AUTO_START_INFO delayed = { TRUE };
        ChangeServiceConfig2W(svc, SERVICE_CONFIG_DELAYED_AUTO_START_INFO, &delayed);
        set_failure_actions(svc, true);
        SERVICE_PRESHUTDOWN_INFO pre = { 20000 };
        ChangeServiceConfig2W(svc, SERVICE_CONFIG_PRESHUTDOWN_INFO, &pre);
        CloseServiceHandle(svc);
        CloseServiceHandle(scm);
        return true;
    }

    bool remove_service(std::string& error)
    {
        suspend_recovery();
        stop_service(30000, error);
        service_handles h;
        if (!open_service(h, DELETE)) return true;
        if (!DeleteService(h.svc) && GetLastError() != ERROR_SERVICE_MARKED_FOR_DELETE)
        {
            error = last_error("cannot delete the service");
            return false;
        }
        return true;
    }

    bool close_trays(unsigned timeout_ms)
    {
        std::vector<DWORD> pids;
        HWND w = nullptr;
        while ((w = FindWindowExW(nullptr, w, L"AlbireoTrayWindow", nullptr)) != nullptr)
        {
            DWORD pid = 0;
            GetWindowThreadProcessId(w, &pid);
            pids.push_back(pid);
            PostMessageW(w, WM_CLOSE, 0, 0);
        }
        for (DWORD pid : pids)
        {
            if (wait_for_process(pid, timeout_ms)) continue;
            HANDLE p = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
            if (p)
            {
                TerminateProcess(p, 0);
                CloseHandle(p);
            }
        }
        return !pids.empty();
    }

    namespace
    {
        /*
            Renames a file, trying again for five seconds: a file that a scanner or an
            indexer reads is released soon.  A running program may be renamed.
        */
        bool move_file(const std::string& from, const std::string& to)
        {
            for (int attempt = 0; attempt < 20; ++attempt)
            {
                if (MoveFileExW(widen(from).c_str(), widen(to).c_str(), MOVEFILE_WRITE_THROUGH)) return true;
                const DWORD e = GetLastError();
                if (e != ERROR_SHARING_VIOLATION && e != ERROR_ACCESS_DENIED && e != ERROR_LOCK_VIOLATION) return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            return false;
        }

        // Deletes a file now, or under a name of its own at the next restart of Windows when it
        // is in use, so that a file installed meanwhile under its name stays.
        void delete_file(const std::string& p)
        {
            if (DeleteFileW(widen(p).c_str()) || GetLastError() == ERROR_FILE_NOT_FOUND) return;
            std::string doomed = p;
            if (p.find(".old-") == std::string::npos)
            {
                const std::string aside = p + ".old-" + random_hex(8);
                if (MoveFileExW(widen(p).c_str(), widen(aside).c_str(), 0)) doomed = aside;
            }
            MoveFileExW(widen(doomed).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
        }
    }

    staged_files::~staged_files()
    {
        if (committed_) return;
        for (const auto& e : entries_)
            if (!e.fresh.empty()) DeleteFileW(widen(e.fresh).c_str());
    }

    std::string staged_files::add(const std::string& target)
    {
        make_dirs(parent_path(target));
        entry e;
        e.target = target;
        e.fresh = target + ".new";
        entries_.push_back(e);
        return e.fresh;
    }

    bool staged_files::write(const std::string& target, const std::function<bool(const chunk_writer&)>& body,
        std::string& error)
    {
        const std::string fresh = add(target);
        HANDLE h = CreateFileW(widen(fresh).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (h == INVALID_HANDLE_VALUE)
        {
            error = last_error(("cannot create " + fresh).c_str());
            return false;
        }
        const bool ok = body([h](const void* data, size_t n) {
            const char* p = static_cast<const char*>(data);
            while (n > 0)
            {
                DWORD w = 0;
                const DWORD chunk = static_cast<DWORD>(std::min<size_t>(n, 1u << 30));
                if (!WriteFile(h, p, chunk, &w, nullptr) || w == 0) return false;
                p += w;
                n -= w;
            }
            return true;
        });
        CloseHandle(h);
        if (!ok && error.empty()) error = "cannot write " + fresh + " (disk full?)";
        return ok;
    }

    void staged_files::remove(const std::string& target)
    {
        entry e;
        e.target = target;
        e.removal = true;
        entries_.push_back(e);
    }

    bool staged_files::commit(std::string& error)
    {
        for (size_t i = 0; i < entries_.size(); ++i)
        {
            entry& e = entries_[i];
            if (GetFileAttributesW(widen(e.target).c_str()) != INVALID_FILE_ATTRIBUTES)
            {
                const std::string aside = e.target + ".old-" + random_hex(8);
                if (!move_file(e.target, aside))
                {
                    error = last_error(("cannot replace " + e.target).c_str());
                    restore(i);
                    return false;
                }
                e.aside = aside;
            }
            if (!e.removal)
            {
                if (!move_file(e.fresh, e.target))
                {
                    error = last_error(("cannot install " + e.target).c_str());
                    restore(i + 1);
                    return false;
                }
                e.placed = true;
            }
        }
        committed_ = true;
        return true;
    }

    void staged_files::restore(size_t count)
    {
        for (size_t i = count; i-- > 0;)
        {
            entry& e = entries_[i];
            // The new file goes back to its own name, which the destructor deletes.
            const DWORD replace = MOVEFILE_REPLACE_EXISTING;
            if (e.placed) MoveFileExW(widen(e.target).c_str(), widen(e.fresh).c_str(), replace);
            if (!e.aside.empty()) MoveFileExW(widen(e.aside).c_str(), widen(e.target).c_str(), replace);
            e.placed = false;
            e.aside.clear();
        }
    }

    void staged_files::rollback()
    {
        if (!committed_) return;
        restore(entries_.size());
        committed_ = false;
    }

    void staged_files::finish()
    {
        // The change stays: rollback() has nothing left to put back.
        for (auto& e : entries_)
        {
            if (!e.aside.empty()) delete_file(e.aside);
            e.aside.clear();
            e.placed = false;
        }
    }

    std::vector<std::string> staged_files::targets() const
    {
        std::vector<std::string> out;
        for (const auto& e : entries_)
            if (!e.removal) out.push_back(e.target);
        return out;
    }

    void remove_old_files(const std::string& program_dir)
    {
        for (const auto& name : list_dir(program_dir))
        {
            const std::string p = join_path(program_dir, name);
            const bool aside = name.find(".old-") != std::string::npos;
            const bool unfinished = name.size() > 4 && name.substr(name.size() - 4) == ".new";
            if (is_link(p)) continue;
            if (dir_exists(p)) remove_old_files(p);
            else if (aside || unfinished) remove_file(p);
        }
    }

    namespace
    {
        struct file_closer
        {
            void operator()(std::FILE* f) const { if (f) std::fclose(f); }
        };
        typedef std::unique_ptr<std::FILE, file_closer> file_ptr;

        file_ptr open_file(const std::string& path, const wchar_t* mode)
        {
            return file_ptr(_wfopen(widen(path).c_str(), mode));
        }

        bool read_at(std::FILE* f, uint64_t pos, void* data, size_t n)
        {
            return _fseeki64(f, static_cast<__int64>(pos), SEEK_SET) == 0 && std::fread(data, 1, n, f) == n;
        }

        template <typename T> T little(const unsigned char* p)
        {
            T v = 0;
            for (size_t i = 0; i < sizeof(T); ++i) v |= static_cast<T>(p[i]) << (8 * i);
            return v;
        }

        /*
            The layout of a program: the end of its sections, the position of the entry of
            its certificate table in its headers, and the position of the table, 0 when the
            program is not signed.
        */
        bool pe_layout(std::FILE* f, uint64_t size, uint64_t& sections_end, uint64_t& security_entry,
            uint64_t& certificates)
        {
            unsigned char dos[64];
            if (size < sizeof(dos) || !read_at(f, 0, dos, sizeof(dos)) || dos[0] != 'M' || dos[1] != 'Z') return false;
            const uint64_t pe = little<uint32_t>(dos + 0x3c);
            unsigned char head[24 + 240];
            if (pe + sizeof(head) > size || !read_at(f, pe, head, sizeof(head))) return false;
            if (std::memcmp(head, "PE\0\0", 4) != 0) return false;
            const unsigned sections = little<uint16_t>(head + 6);
            const unsigned optional = little<uint16_t>(head + 20);
            const unsigned magic = little<uint16_t>(head + 24);
            // The data directories: 96 bytes into a PE32 optional header, 112 into PE32+.
            const unsigned directories = magic == 0x20b ? 112 : 96;
            const unsigned count = little<uint32_t>(head + 24 + directories - 4);
            security_entry = pe + 24 + directories + 4 * 8;
            certificates = 0;
            if (count > 4 && directories + 5 * 8 <= optional)
            {
                const uint64_t at = little<uint32_t>(head + 24 + directories + 4 * 8);
                const uint64_t length = little<uint32_t>(head + 24 + directories + 4 * 8 + 4);
                if (at && length && at + length <= size) certificates = at;
            }
            sections_end = 0;
            const uint64_t table = pe + 24 + optional;
            for (unsigned i = 0; i < sections; ++i)
            {
                unsigned char s[40];
                if (!read_at(f, table + 40ull * i, s, sizeof(s))) return false;
                const uint64_t raw = little<uint32_t>(s + 16);
                const uint64_t at = little<uint32_t>(s + 20);
                if (raw && at + raw > sections_end) sections_end = at + raw;
            }
            return sections_end > 0 && sections_end <= size;
        }

        // Copies the first length bytes of a program, without the entry of its certificate
        // table, which the copy no longer holds.
        bool copy_program(const std::string& from, const std::wstring& to, uint64_t length, std::string& error)
        {
            file_ptr in = open_file(from, L"rb");
            file_ptr out(_wfopen(to.c_str(), L"wb"));
            if (!in || !out)
            {
                error = "cannot copy the uninstaller";
                return false;
            }
            const uint64_t size = file_size(from);
            uint64_t sections_end = 0, security_entry = 0, certificates = 0;
            const bool pe = pe_layout(in.get(), size, sections_end, security_entry, certificates);
            if (_fseeki64(in.get(), 0, SEEK_SET) != 0) return false;
            std::vector<char> b(1u << 20);
            for (uint64_t done = 0; done < length;)
            {
                const size_t n = static_cast<size_t>(std::min<uint64_t>(length - done, b.size()));
                if (std::fread(b.data(), 1, n, in.get()) != n)
                {
                    error = "cannot read " + from;
                    return false;
                }
                // The entry of the certificate table, emptied.
                if (pe && certificates && security_entry + 8 > done && security_entry < done + n)
                    for (uint64_t i = std::max(done, security_entry); i < std::min(done + n, security_entry + 8); ++i)
                        b[static_cast<size_t>(i - done)] = 0;
                if (std::fwrite(b.data(), 1, n, out.get()) != n)
                {
                    error = "cannot write the uninstaller";
                    return false;
                }
                done += n;
            }
            return std::fflush(out.get()) == 0;
        }
    }

    builtin_overlay read_builtin_overlay(const std::string& installer)
    {
        builtin_overlay o;
        file_ptr f = open_file(installer, L"rb");
        if (!f) return o;
        const uint64_t size = file_size(installer);
        uint64_t sections_end = 0, security_entry = 0, certificates = 0;
        if (!pe_layout(f.get(), size, sections_end, security_entry, certificates)) return o;
        // The overlay ends before the certificate table, which a signing tool may precede
        // with up to 7 bytes of zeros.
        const uint64_t end = certificates ? certificates : size;
        for (uint64_t pad = 0; pad < 8; ++pad)
        {
            if (end < sections_end + sizeof(builtin_trailer) + pad) break;
            const uint64_t tail = end - pad;
            builtin_trailer t;
            if (!read_at(f.get(), tail - sizeof(t), &t, sizeof(t)) || !t.valid()) continue;
            if (t.overlay_size > tail - sections_end || t.manifest_size > (16u << 20)
                || t.manifest_offset + t.manifest_size > t.overlay_size)
                break;
            const uint64_t start = tail - t.overlay_size;
            std::string text(static_cast<size_t>(t.manifest_size), '\0');
            if (!read_at(f.get(), start + t.manifest_offset, &text[0], text.size())) break;
            json manifest = json::parse_or_null(text);
            uint64_t total = 0;
            bool valid = manifest["models"].is_array();
            for (const auto& m : manifest["models"].elements())
            {
                const uint64_t at = static_cast<uint64_t>(m["offset"].as_int(-1));
                const uint64_t length = static_cast<uint64_t>(m["size"].as_int(-1));
                valid = valid && m["offset"].is_number() && m["size"].is_number() && at + length <= t.manifest_offset;
                total += length;
            }
            if (!valid) break;
            o.start = start;
            o.total = total;
            o.manifest = manifest;
            break;
        }
        return o;
    }

    namespace
    {
        // The model files that came with the installation in place: name -> SHA-256.
        std::map<std::string, std::string> installed_builtin_models(const std::string& dir)
        {
            std::string text;
            const json before = read_file(join_path(dir, builtin_manifest_name), text, 16u << 20)
                ? json::parse_or_null(text) : json();
            std::map<std::string, std::string> had;
            for (const auto& m : before["models"].elements()) had[m["file"].as_string()] = m["sha256"].as_string();
            return had;
        }

        // Whether a model file of the installer is in place already, the same, and kept.
        bool builtin_in_place(const std::map<std::string, std::string>& had, const std::string& dir, const json& m)
        {
            const std::string name = m["file"].as_string();
            const std::string target = join_path(dir, name);
            const auto it = had.find(name);
            return it != had.end() && it->second == m["sha256"].as_string() && file_exists(target)
                && file_size(target) == static_cast<uint64_t>(m["size"].as_int(0));
        }
    }

    uint64_t builtin_bytes_to_copy(const std::string& program_dir)
    {
        const builtin_overlay o = read_builtin_overlay(executable_path());
        const std::string dir = join_path(program_dir, builtin_models_folder);
        const std::map<std::string, std::string> had = installed_builtin_models(dir);
        uint64_t bytes = 0;
        for (const auto& m : o.manifest["models"].elements())
            if (!builtin_in_place(had, dir, m)) bytes += static_cast<uint64_t>(m["size"].as_int(0));
        return bytes;
    }

    bool stage_builtin_models(const std::string& program_dir, staged_files& files,
        const std::function<void(uint64_t done, uint64_t total, const std::string& file)>& progress,
        std::string& error)
    {
        const std::string installer = executable_path();
        const builtin_overlay o = read_builtin_overlay(installer);
        if (!o.present()) return true;
        const std::string dir = join_path(program_dir, builtin_models_folder);
        const std::map<std::string, std::string> had = installed_builtin_models(dir);
        file_ptr in = open_file(installer, L"rb");
        if (!in)
        {
            error = "cannot read " + installer;
            return false;
        }
        json manifest = o.manifest;
        std::set<std::string> names;
        uint64_t done = 0;
        std::vector<char> b(1u << 20);
        for (auto& m : manifest["models"].elements())
        {
            const std::string name = m["file"].as_string();
            const uint64_t size = static_cast<uint64_t>(m["size"].as_int(0));
            const std::string sha = m["sha256"].as_string();
            const uint64_t at = o.start + static_cast<uint64_t>(m["offset"].as_int(0));
            m.erase("offset");
            if (name.empty() || name.find_first_of("/\\:") != std::string::npos || name.find("..") != std::string::npos)
            {
                error = "invalid file name in the models of the installer: " + name;
                return false;
            }
            names.insert(name);
            if (builtin_in_place(had, dir, m))
            {
                log("model " + name + ": already installed");
                done += size;
                if (progress) progress(done, o.total, name);
                continue;
            }
            sha256_stream hash;
            auto copy = [&](const chunk_writer& write) {
                if (_fseeki64(in.get(), static_cast<__int64>(at), SEEK_SET) != 0) return false;
                for (uint64_t left = size; left > 0;)
                {
                    const size_t n = static_cast<size_t>(std::min<uint64_t>(left, b.size()));
                    if (std::fread(b.data(), 1, n, in.get()) != n)
                    {
                        error = "cannot read " + installer;
                        return false;
                    }
                    if (!write(b.data(), n)) return false;
                    hash.update(b.data(), n);
                    left -= n;
                    done += n;
                    if (progress) progress(done, o.total, name);
                }
                return true;
            };
            if (!files.write(join_path(dir, name), copy, error)) return false;
            if (hash.hex() != sha)
            {
                error = "the model " + name + " of the installer is damaged";
                return false;
            }
            log("model " + name + ": copied");
        }
        // The models that no longer come with Albireo.
        for (const auto& kv : had)
            if (!names.count(kv.first) && kv.first.find_first_of("/\\:") == std::string::npos
                && kv.first.find("..") == std::string::npos)
                files.remove(join_path(dir, kv.first));
        // The manifest comes last, since the service registers the models it lists.
        const std::string text = manifest.dump_formatted(2) + "\n";
        return files.write(join_path(dir, builtin_manifest_name),
            [&](const chunk_writer& write) { return write(text.data(), text.size()); }, error);
    }

    bool stage_uninstaller(const std::string& program_dir, staged_files& files, std::string& error)
    {
        const std::string self = executable_path();
        const std::string target = join_path(program_dir, "albireo-setup.exe");
        if (_wcsicmp(widen(self).c_str(), widen(target).c_str()) == 0) return true;
        const std::wstring tmp = widen(files.add(target));
        // The program alone: the models that follow it stay in the installer.
        const builtin_overlay overlay = read_builtin_overlay(self);
        if (overlay.present() ? !copy_program(self, tmp, overlay.start, error)
                : !CopyFileW(widen(self).c_str(), tmp.c_str(), FALSE))
        {
            if (error.empty()) error = last_error("cannot copy the uninstaller");
            return false;
        }
        // The copy keeps everything but the payload; when the payload cannot be removed,
        // the uninstaller is simply larger.
        HANDLE u = BeginUpdateResourceW(tmp.c_str(), FALSE);
        if (!u) log(last_error("the uninstaller keeps its payload: BeginUpdateResource"));
        else if (!UpdateResourceW(u, RT_RCDATA, MAKEINTRESOURCEW(IDR_PAYLOAD),
                     MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL), nullptr, 0))
        {
            log(last_error("the uninstaller keeps its payload: UpdateResource"));
            EndUpdateResourceW(u, TRUE);
        }
        else if (!EndUpdateResourceW(u, FALSE))
        {
            log(last_error("the uninstaller keeps its payload: EndUpdateResource"));
        }
        return true;
    }

    namespace
    {
        // The lines of the log written before the data folder is secured wait here.
        std::mutex log_m;
        bool log_ready = false;
        std::string log_waiting;

        // Appends text after the lines that wait, once the data folder is secured; log_m is held.
        void append_log(const std::string& text)
        {
            log_waiting += text;
            if (!log_ready || log_waiting.empty()) return;
            std::ofstream out(widen(join_path(join_path(data_dir(), "logs"), "setup.log")).c_str(),
                std::ios::app | std::ios::binary);
            out << log_waiting;
            log_waiting.clear();
        }
    }

    bool prepare_data_dir(std::string& error)
    {
        /*
            The installers before this one gave the data folder these permissions at their
            end only: what it holds is checked until an installation of this version records
            the check in the entry of the installed programs (register_uninstall()).
        */
        DWORD checked = 0, size = sizeof(checked);
        const bool content_checked = RegGetValueW(HKEY_LOCAL_MACHINE, uninstall_key, L"DataFolderChecked",
            RRF_RT_REG_DWORD | RRF_SUBKEY_WOW6464KEY, nullptr, &checked, &size) == ERROR_SUCCESS && checked == 1;
        const std::string dir = data_dir();
        // A data folder that another account made, or its link, is no folder of Albireo.
        if ((dir_exists(dir) || is_link(dir)) && !owned_by_system(dir))
        {
            const std::string aside = dir + ".untrusted-" + random_hex(8);
            if (!MoveFileExW(widen(dir).c_str(), widen(aside).c_str(), 0))
            {
                error = last_error(("cannot rename aside " + dir).c_str());
                return false;
            }
            log("the data folder, made by another account, is renamed " + aside);
        }
        if (!secure_folder(dir, error, !content_checked)) return false;
        for (const char* sub : { "logs", "import", "certificates" }) make_dirs(join_path(dir, sub));
        std::lock_guard<std::mutex> lock(log_m);
        log_ready = true;
        append_log(std::string());
        return true;
    }

    bool write_settings(const install_options& o, std::string& error)
    {
        const std::string path = join_path(data_dir(), "settings.json");
        std::string text;
        json s = json::object();
        if (read_file(path, text, 1u << 20))
        {
            s = json::parse_or_null(text);
            if (!s.is_object()) s = json::object();
        }
        s["port"] = o.port;
        s["models_dir"] = o.models_dir.empty() || o.models_dir == default_models_dir() ? std::string() : o.models_dir;
        s["default_model"] = o.default_model;
        s["offline"] = o.offline;
        if (!write_file_atomic(path, s.dump_formatted(2) + "\n"))
        {
            error = "cannot write " + path;
            return false;
        }
        return true;
    }

    bool stage_bundle(const std::function<void(uint64_t done, uint64_t total, const std::string& file)>& progress,
        std::string& error)
    {
        const std::string bundle = executable_dir();
        std::string body, sig;
        if (!read_file(join_path(bundle, "catalog.json"), body, 16u << 20)
            || !read_file(join_path(bundle, "catalog.json.sig"), sig, 4096))
            return true;
        // The catalog signed with the key of Albireo decides which models of the bundle go.
        const json catalog = json::parse_or_null(body);
        if (!verify_signature(body, sig, feed_public_key_pem) || !catalog["models"].is_array())
        {
            error = "the catalog next to the installer does not bear the signature of Albireo; "
                "its models are not installed";
            return false;
        }
        std::set<std::string> hashes;
        for (const auto& m : catalog["models"].elements()) hashes.insert(m["sha256"].as_string());
        const std::string inbox = join_path(data_dir(), "import");
        if (!write_file_atomic(join_path(inbox, "catalog.json.sig"), sig)
            || !write_file_atomic(join_path(inbox, "catalog.json"), body))
        {
            error = "cannot copy the catalog of the bundle into " + inbox;
            return false;
        }
        // The models folder of programs (a build tree, an installation) holds the models
        // that come with Albireo, which the installer carries already: they are no bundle.
        const std::string source = join_path(bundle, "models");
        std::set<std::string> builtin;
        std::string text;
        // A variable holds the manifest while the loop walks its array.
        const json manifest = read_file(join_path(source, builtin_manifest_name), text, 16u << 20)
            ? json::parse_or_null(text) : json();
        for (const auto& m : manifest["models"].elements()) builtin.insert(m["file"].as_string());
        std::vector<std::string> names;
        uint64_t total = 0;
        for (const auto& name : list_dir(source))
        {
            const std::string p = join_path(source, name);
            if (name.size() <= 5 || name.substr(name.size() - 5) != ".gguf" || builtin.count(name) || is_link(p)
                || !file_exists(p))
                continue;
            names.push_back(name);
            total += file_size(p);
        }
        if (!names.empty() && free_disk_space(inbox) < total + (256ull << 20))
        {
            error = "not enough free space in " + inbox + " for the models of the bundle ("
                + std::to_string(total >> 20) + " MB)";
            return false;
        }
        std::vector<char> b(1u << 20);
        uint64_t done = 0;
        std::string rejected;
        for (const auto& name : names)
        {
            // Copied under a name the service leaves alone, then named once its hash is one of the catalog.
            const std::string target = join_path(inbox, name);
            const std::string tmp = target + ".tmp-" + random_hex(8);
            sha256_stream hash;
            bool ok = false;
            {
                file_ptr in = open_file(join_path(source, name), L"rb");
                file_ptr out = open_file(tmp, L"wb");
                ok = in && out;
                for (size_t n = 0; ok && (n = std::fread(b.data(), 1, b.size(), in.get())) > 0;)
                {
                    ok = std::fwrite(b.data(), 1, n, out.get()) == n;
                    hash.update(b.data(), n);
                    done += n;
                    if (progress) progress(done, total, name);
                }
                ok = ok && !std::ferror(in.get()) && std::fflush(out.get()) == 0;
            }
            if (ok && hashes.count(hash.hex()) && rename_file(tmp, target)) continue;
            remove_file(tmp);
            rejected += (rejected.empty() ? "" : ", ") + name;
        }
        if (rejected.empty()) return true;
        error = "the models " + rejected + " of the bundle cannot be read, or are not those of its catalog";
        return false;
    }

    namespace
    {
        // Whether name constraints permit localhost and the loopback addresses, and nothing else.
        bool loopback_only(const CERT_EXTENSION* extension)
        {
            CERT_NAME_CONSTRAINTS_INFO* info = nullptr;
            DWORD size = 0;
            if (!CryptDecodeObjectEx(X509_ASN_ENCODING, X509_NAME_CONSTRAINTS, extension->Value.pbData,
                    extension->Value.cbData, CRYPT_DECODE_ALLOC_FLAG, nullptr, &info, &size))
                return false;
            // 127.0.0.0/8 and ::1/128, each address followed by its mask.
            const BYTE v4[8] = { 127, 0, 0, 0, 255, 0, 0, 0 };
            BYTE v6[32] = {};
            v6[15] = 1;
            std::memset(v6 + 16, 0xFF, 16);
            bool ok = info->cPermittedSubtree > 0;
            for (DWORD i = 0; ok && i < info->cPermittedSubtree; ++i)
            {
                const CERT_ALT_NAME_ENTRY& name = info->rgPermittedSubtree[i].Base;
                const CRYPT_DATA_BLOB& ip = name.IPAddress;
                if (name.dwAltNameChoice == CERT_ALT_NAME_DNS_NAME)
                    ok = name.pwszDNSName && _wcsicmp(name.pwszDNSName, L"localhost") == 0;
                else if (name.dwAltNameChoice == CERT_ALT_NAME_IP_ADDRESS)
                    ok = (ip.cbData == sizeof(v4) && std::memcmp(ip.pbData, v4, sizeof(v4)) == 0)
                        || (ip.cbData == sizeof(v6) && std::memcmp(ip.pbData, v6, sizeof(v6)) == 0);
                else
                    ok = false;
            }
            LocalFree(info);
            return ok;
        }

        /*
            Whether a certificate is the authority that Albireo makes (host/certificates.cpp):
            signed by its own key, named "Albireo Local Authority", and limited by its name
            constraints to localhost and the loopback addresses, so that the trust of the
            machine never reaches another site.
        */
        bool local_authority(const std::string& der)
        {
            PCCERT_CONTEXT c = CertCreateCertificateContext(X509_ASN_ENCODING,
                reinterpret_cast<const BYTE*>(der.data()), static_cast<DWORD>(der.size()));
            if (!c) return false;
            void* self = const_cast<CERT_CONTEXT*>(c);
            wchar_t name[128] = {};
            bool ok = CryptVerifyCertificateSignatureEx(0, X509_ASN_ENCODING, CRYPT_VERIFY_CERT_SIGN_SUBJECT_CERT,
                    self, CRYPT_VERIFY_CERT_SIGN_ISSUER_CERT, self, 0, nullptr)
                && CertGetNameStringW(c, CERT_NAME_ATTR_TYPE, 0, const_cast<char*>(szOID_COMMON_NAME), name, 128) > 1
                && std::wcscmp(name, L"Albireo Local Authority") == 0;
            const CERT_EXTENSION* constraints = ok ? CertFindExtension(szOID_NAME_CONSTRAINTS,
                c->pCertInfo->cExtension, c->pCertInfo->rgExtension) : nullptr;
            ok = ok && constraints && loopback_only(constraints);
            CertFreeCertificateContext(c);
            return ok;
        }
    }

    bool create_certificates(bool new_authority, bool& renewed, std::string& error)
    {
        const std::string dir = join_path(data_dir(), "certificates");
        const certificate_files f = certificate_paths(dir);
        const std::string before = certificate_thumbprint(f.ca_cert);
        if (new_authority)
        {
            // A new installation makes its own authority: none found in the folder is kept.
            for (const std::string& p : { f.ca_cert, f.ca_key, f.server_cert, f.server_key })
                if (file_exists(p) && !remove_file(p))
                {
                    error = last_error(("cannot replace " + p).c_str());
                    return false;
                }
        }
        if (!ensure_certificates(dir, error)) return false;
        renewed = certificate_thumbprint(f.ca_cert) != before;
        return true;
    }

    bool authority_trusted()
    {
        HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_SYSTEM_W, 0, 0,
            CERT_SYSTEM_STORE_LOCAL_MACHINE | CERT_STORE_OPEN_EXISTING_FLAG | CERT_STORE_READONLY_FLAG, L"ROOT");
        if (!store) return false;
        PCCERT_CONTEXT c = CertFindCertificateInStore(store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, 0,
            CERT_FIND_SUBJECT_STR_W, L"Albireo Local Authority", nullptr);
        if (c) CertFreeCertificateContext(c);
        CertCloseStore(store, 0);
        return c != nullptr;
    }

    bool trust_authority(std::string& error)
    {
        const std::string der = certificate_der(certificate_paths(join_path(data_dir(), "certificates")).ca_cert);
        if (der.empty())
        {
            error = "the certificate of the local authority cannot be read";
            return false;
        }
        if (!local_authority(der))
        {
            error = "the certificate of the local authority is not one that Albireo makes; it is not trusted";
            return false;
        }
        HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_SYSTEM_W, 0, 0,
            CERT_SYSTEM_STORE_LOCAL_MACHINE | CERT_STORE_OPEN_EXISTING_FLAG, L"ROOT");
        if (!store)
        {
            error = last_error("cannot open the trusted roots of the machine");
            return false;
        }
        remove_authority_certificates(store);
        const bool ok = CertAddEncodedCertificateToStore(store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING,
            reinterpret_cast<const BYTE*>(der.data()), static_cast<DWORD>(der.size()), CERT_STORE_ADD_REPLACE_EXISTING,
            nullptr) != FALSE;
        if (!ok) error = last_error("cannot add the local authority to the trusted roots");
        CertCloseStore(store, 0);
        return ok;
    }

    void untrust_authority()
    {
        HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_SYSTEM_W, 0, 0,
            CERT_SYSTEM_STORE_LOCAL_MACHINE | CERT_STORE_OPEN_EXISTING_FLAG, L"ROOT");
        if (!store) return;
        remove_authority_certificates(store);
        CertCloseStore(store, 0);
    }

    bool register_uninstall(const std::string& program_dir, uint64_t size_bytes, std::string& error)
    {
        HKEY k = nullptr;
        if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, uninstall_key, 0, nullptr, 0, KEY_WRITE | KEY_WOW64_64KEY, nullptr, &k,
                nullptr) != ERROR_SUCCESS)
        {
            error = "cannot write the registry";
            return false;
        }
        auto str = [&](const wchar_t* name, const std::wstring& v) {
            RegSetValueExW(k, name, 0, REG_SZ, reinterpret_cast<const BYTE*>(v.c_str()),
                static_cast<DWORD>((v.size() + 1) * sizeof(wchar_t)));
        };
        auto dword = [&](const wchar_t* name, DWORD v) {
            RegSetValueExW(k, name, 0, REG_DWORD, reinterpret_cast<const BYTE*>(&v), sizeof(v));
        };
        const std::wstring dir = widen(program_dir);
        const std::wstring setup = L"\"" + widen(join_path(program_dir, "albireo-setup.exe")) + L"\"";
        str(L"DisplayName", widen(ALBIREO_FULL_NAME));
        str(L"DisplayVersion", widen(ALBIREO_VERSION));
        str(L"Publisher", widen(ALBIREO_COMPANY_NAME));
        str(L"DisplayIcon", widen(join_path(program_dir, "albireo-tray.exe")) + L",0");
        str(L"InstallLocation", dir);
        str(L"UninstallString", setup + L" /uninstall");
        str(L"QuietUninstallString", setup + L" /uninstall /silent");
        str(L"URLInfoAbout", widen(ALBIREO_PROJECT_URL));
        str(L"HelpLink", widen(ALBIREO_PROJECT_URL));
        str(L"Comments", L"Local language models, built on Dlib-Transformer-extensions");
        time_t now = time(nullptr);
        tm t;
        gmtime_s(&t, &now);
        wchar_t date[16];
        swprintf(date, 16, L"%04d%02d%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday);
        str(L"InstallDate", date);
        dword(L"EstimatedSize", static_cast<DWORD>(size_bytes / 1024));
        dword(L"NoModify", 1);
        dword(L"NoRepair", 1);
        dword(L"VersionMajor", ALBIREO_VERSION_MAJOR);
        dword(L"VersionMinor", ALBIREO_VERSION_MINOR);
        // The content of the data folder was checked at the start of this installation (prepare_data_dir()).
        dword(L"DataFolderChecked", 1);
        RegCloseKey(k);
        return true;
    }

    void remove_uninstall()
    {
        RegDeleteKeyExW(HKEY_LOCAL_MACHINE, uninstall_key, KEY_WOW64_64KEY, 0);
    }

    void set_autostart(const std::string& program_dir, bool on)
    {
        HKEY k = nullptr;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, run_key, 0, KEY_WRITE | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return;
        if (on)
        {
            const std::wstring v = L"\"" + widen(join_path(program_dir, "albireo-tray.exe")) + L"\"";
            RegSetValueExW(k, L"Albireo", 0, REG_SZ, reinterpret_cast<const BYTE*>(v.c_str()),
                static_cast<DWORD>((v.size() + 1) * sizeof(wchar_t)));
        }
        else
        {
            RegDeleteValueW(k, L"Albireo");
        }
        RegCloseKey(k);
    }

    void create_shortcuts(const std::string& program_dir, bool desktop)
    {
        const std::wstring tray = widen(join_path(program_dir, "albireo-tray.exe"));
        const std::wstring desc = L"Albireo";
        create_link(widen(start_menu_link()), tray, L"--open", desc);
        if (desktop) create_link(widen(desktop_link()), tray, L"--open", desc);
        else remove_file(desktop_link());
    }

    void remove_shortcuts()
    {
        remove_file(start_menu_link());
        remove_file(desktop_link());
    }

    bool launch_unelevated(const std::string& program, const std::string& args)
    {
        /*
            The desktop runs with the rights of the user: its shell starts the program, so
            that the tray does not inherit the rights of the installer.
        */
        bool ok = false;
        IShellWindows* windows = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_ShellWindows, nullptr, CLSCTX_LOCAL_SERVER, IID_IShellWindows,
                reinterpret_cast<void**>(&windows))))
        {
            VARIANT empty;
            VariantInit(&empty);
            long hwnd = 0;
            IDispatch* disp = nullptr;
            if (SUCCEEDED(windows->FindWindowSW(&empty, &empty, SWC_DESKTOP, &hwnd, SWFO_NEEDDISPATCH, &disp)) && disp)
            {
                IServiceProvider* sp = nullptr;
                if (SUCCEEDED(disp->QueryInterface(IID_IServiceProvider, reinterpret_cast<void**>(&sp))))
                {
                    IShellBrowser* browser = nullptr;
                    if (SUCCEEDED(sp->QueryService(SID_STopLevelBrowser, IID_IShellBrowser,
                        reinterpret_cast<void**>(&browser))))
                    {
                        IShellView* view = nullptr;
                        if (SUCCEEDED(browser->QueryActiveShellView(&view)))
                        {
                            IDispatch* bg = nullptr;
                            if (SUCCEEDED(view->GetItemObject(SVGIO_BACKGROUND, IID_IDispatch,
                                reinterpret_cast<void**>(&bg))))
                            {
                                IShellFolderViewDual* dual = nullptr;
                                if (SUCCEEDED(bg->QueryInterface(IID_IShellFolderViewDual,
                                    reinterpret_cast<void**>(&dual))))
                                {
                                    IDispatch* app = nullptr;
                                    if (SUCCEEDED(dual->get_Application(&app)))
                                    {
                                        IShellDispatch2* shell = nullptr;
                                        if (SUCCEEDED(app->QueryInterface(IID_IShellDispatch2,
                                            reinterpret_cast<void**>(&shell))))
                                        {
                                            BSTR file = SysAllocString(widen(program).c_str());
                                            VARIANT vargs, vdir, vop, vshow;
                                            VariantInit(&vargs);
                                            vargs.vt = VT_BSTR;
                                            vargs.bstrVal = SysAllocString(widen(args).c_str());
                                            VariantInit(&vdir);
                                            VariantInit(&vop);
                                            VariantInit(&vshow);
                                            vshow.vt = VT_I4;
                                            vshow.lVal = SW_SHOWNORMAL;
                                            ok = SUCCEEDED(shell->ShellExecute(file, vargs, vdir, vop, vshow));
                                            VariantClear(&vargs);
                                            SysFreeString(file);
                                            shell->Release();
                                        }
                                        app->Release();
                                    }
                                    dual->Release();
                                }
                                bg->Release();
                            }
                            view->Release();
                        }
                        browser->Release();
                    }
                    sp->Release();
                }
                disp->Release();
            }
            windows->Release();
        }
        // Started from here, the program would receive the rights of the installer.
        if (!ok) log("the shell of the desktop did not start " + program);
        return ok;
    }

    bool wait_for_server(int port, unsigned timeout_ms)
    {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        bool ok = false;
        while (!ok && std::chrono::steady_clock::now() < deadline)
        {
            SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (s != INVALID_SOCKET)
            {
                sockaddr_in a = {};
                a.sin_family = AF_INET;
                a.sin_port = htons(static_cast<u_short>(port));
                a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
                ok = connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0;
                closesocket(s);
            }
            if (!ok) std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        WSACleanup();
        return ok;
    }

    namespace
    {
        // Removes a link alone, without what it leads to; its attributes are its own.
        void remove_link(const std::string& p)
        {
            const std::wstring w = widen(p);
            if (GetFileAttributesW(w.c_str()) & FILE_ATTRIBUTE_DIRECTORY) RemoveDirectoryW(w.c_str());
            else DeleteFileW(w.c_str());
        }

        const char* const file_list_name = "albireo-files.json";

        // The files the list of an installation names, relative to its folder; a name that
        // would lead out of the folder is left out.
        std::vector<std::string> listed_files(const std::string& program_dir, bool& found)
        {
            std::string text;
            found = read_file(join_path(program_dir, file_list_name), text, 16u << 20);
            const json list = found ? json::parse_or_null(text) : json();
            std::vector<std::string> out;
            for (const auto& f : list["files"].elements())
            {
                const std::string rel = f.as_string();
                if (!rel.empty() && rel[0] != '\\' && rel[0] != '/' && rel.find(':') == std::string::npos
                    && rel.find("..") == std::string::npos)
                    out.push_back(rel);
            }
            return out;
        }

        // The files under base\rel, named relative to base, without following the links.
        void collect_files(const std::string& base, const std::string& rel, std::vector<std::string>& out)
        {
            for (const auto& name : list_dir(join_path(base, rel)))
            {
                const std::string r = join_path(rel, name);
                const std::string p = join_path(base, r);
                if (!is_link(p) && dir_exists(p)) collect_files(base, r, out);
                else out.push_back(r);
            }
        }

        // The files of Albireo in a folder of programs, known by their places: its programs, and
        // the files of its engines, web application, licenses and models.
        std::vector<std::string> albireo_files(const std::string& program_dir)
        {
            std::vector<std::string> files;
            for (const char* name : { "albireo-service.exe", "albireo-tray.exe", "albireo-setup.exe" })
                if (file_exists(join_path(program_dir, name))) files.push_back(name);
            for (const char* folder : { "engines", "web", "licenses", "models" })
                if (!is_link(join_path(program_dir, folder))) collect_files(program_dir, folder, files);
            return files;
        }

        // The uninstaller that runs, renamed aside, and its folder, removed once it ends.
        std::string running_aside, running_dir;
    }

    void remove_folder(const std::string& path)
    {
        if (is_link(path))
        {
            remove_link(path);
            return;
        }
        if (!dir_exists(path)) return;
        for (const auto& name : list_dir(path))
        {
            const std::string p = join_path(path, name);
            if (is_link(p)) remove_link(p);
            else if (dir_exists(p)) remove_folder(p);
            else delete_file(p);
        }
        if (!RemoveDirectoryW(widen(path).c_str()))
            MoveFileExW(widen(path).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
    }

    void write_file_list(const std::string& program_dir, const std::vector<std::string>& files)
    {
        bool found = false;
        std::vector<std::string> before = listed_files(program_dir, found);
        // The first list also names what an earlier installation left in the places of Albireo.
        if (!found) before = albireo_files(program_dir);
        std::set<std::string> names;
        for (const auto& rel : before)
            if (file_exists(join_path(program_dir, rel))) names.insert(rel);
        // The files given are paths in program_dir, written as the installation writes it.
        std::string prefix = program_dir;
        if (!prefix.empty() && prefix.back() != '\\' && prefix.back() != '/') prefix += '\\';
        for (const auto& f : files)
            if (f.size() > prefix.size() && f.compare(0, prefix.size(), prefix) == 0)
                names.insert(f.substr(prefix.size()));
        json list = json::array();
        for (const auto& n : names) list.push_back(n);
        json doc = json::object();
        doc["files"] = list;
        if (!write_file_atomic(join_path(program_dir, file_list_name), doc.dump_formatted(2) + "\n"))
            log("cannot write the list of the files of " + program_dir);
    }

    void remove_installation(const std::string& program_dir)
    {
        bool found = false;
        std::vector<std::string> files = listed_files(program_dir, found);
        // An installation older than the list: the files of Albireo, known by their places.
        if (!found) files = albireo_files(program_dir);
        files.push_back(file_list_name);
        remove_old_files(program_dir);
        const std::wstring self = widen(executable_path());
        std::set<std::string> folders;
        for (const auto& rel : files)
        {
            for (std::string d = parent_path(rel); !d.empty(); d = parent_path(d)) folders.insert(d);
            const std::string p = join_path(program_dir, rel);
            if (is_link(p))
            {
                remove_link(p);
            }
            else if (_wcsicmp(widen(p).c_str(), self.c_str()) == 0)
            {
                // The running uninstaller goes aside, then away once it ends (finish_removal()).
                const std::string aside = p + ".old-" + random_hex(8);
                if (!MoveFileExW(widen(p).c_str(), widen(aside).c_str(), 0)) continue;
                MoveFileExW(widen(aside).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
                running_aside = aside;
                running_dir = program_dir;
            }
            else
            {
                delete_file(p);
            }
        }
        // The folders left empty, the deepest first, then the folder of the programs.
        std::vector<std::string> ordered(folders.begin(), folders.end());
        std::sort(ordered.begin(), ordered.end(),
            [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
        for (const auto& d : ordered) RemoveDirectoryW(widen(join_path(program_dir, d)).c_str());
        if (!RemoveDirectoryW(widen(program_dir).c_str()) && running_dir.empty())
            log("the folder " + program_dir + " keeps files that Albireo did not install");
    }

    void finish_removal()
    {
        if (running_aside.empty()) return;
        // The folder goes at the next restart of Windows if it is empty then.
        MoveFileExW(widen(running_dir).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
        wchar_t buf[MAX_PATH];
        const UINT n = GetSystemDirectoryW(buf, MAX_PATH);
        const std::wstring system(buf, n < MAX_PATH ? n : 0);
        const std::wstring file = widen(running_aside), dir = widen(running_dir);
        // cmd expands %name% even between quotes: such paths wait for the restart.
        if (system.empty() || (file + dir).find(L'%') != std::wstring::npos) return;
        /*
            cmd and ping of System32, by their full paths and from System32, try once a
            second for half a minute to delete the uninstaller, which ends right after this
            call, then remove its folder when it is empty.
        */
        const std::wstring cmd_exe = system + L"\\cmd.exe";
        std::wstring cmd = L"\"" + cmd_exe + L"\" /d /s /c \"for /l %i in (1,1,30) do (del /f /q \"" + file
            + L"\" 2>nul & if not exist \"" + file + L"\" (rd \"" + dir + L"\" 2>nul & exit) & \"" + system
            + L"\\PING.EXE\" -n 2 127.0.0.1 >nul)\"";
        STARTUPINFOW si = { sizeof(si) };
        PROCESS_INFORMATION pi = {};
        if (CreateProcessW(cmd_exe.c_str(), &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr,
                system.c_str(), &si, &pi))
        {
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
        }
    }

    void log(const std::string& line)
    {
        std::lock_guard<std::mutex> lock(log_m);
        append_log(iso_time(now_unix()) + " " + line + "\n");
    }

}
}
