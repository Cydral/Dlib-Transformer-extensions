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
#include <tlhelp32.h>
#include <wincrypt.h>
#include <iphlpapi.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <cwctype>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "../common/file_kinds.h"
#include "../common/json.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "../host/certificates.h"
#include "../host/crypto.h"
#include "resource.h"

namespace pseudalis
{
namespace setup
{

    namespace
    {
        const wchar_t uninstall_key[] = L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Pseudalis";
        const wchar_t run_key[] = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";

        // Why the service cannot be registered while it waits for its deletion.
        const char* const service_marked_text = "the service Pseudalis waits for its deletion: close the Services "
            "console and the programs that use the service, or restart Windows, then run the installer again";

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
            return join_path(known_folder(FOLDERID_CommonPrograms, fallback), "Pseudalis.lnk");
        }

        std::string desktop_link()
        {
            return join_path(known_folder(FOLDERID_PublicDesktop, "C:\\Users\\Public\\Desktop"), "Pseudalis.lnk");
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

        // The full path of a folder, without "\" at its end.
        std::wstring full_folder(const std::string& dir)
        {
            std::wstring full(32768, L'\0');
            const DWORD n = GetFullPathNameW(widen(dir).c_str(), static_cast<DWORD>(full.size()), &full[0], nullptr);
            full.resize(n > 0 && n < full.size() ? n : 0);
            if (full.empty()) full = widen(dir);
            while (full.size() > 3 && (full.back() == L'\\' || full.back() == L'/')) full.pop_back();
            return full;
        }

        // The folders of Windows and of the users, which no folder of Pseudalis may be or hold.
        std::vector<std::string> system_folders()
        {
            static const KNOWNFOLDERID* const ids[] = { &FOLDERID_Windows, &FOLDERID_ProgramFiles,
                &FOLDERID_ProgramFilesX86, &FOLDERID_ProgramFilesCommon, &FOLDERID_ProgramData, &FOLDERID_UserProfiles,
                &FOLDERID_Public, &FOLDERID_Profile, &FOLDERID_Desktop, &FOLDERID_Documents, &FOLDERID_Downloads,
                &FOLDERID_LocalAppData, &FOLDERID_RoamingAppData };
            std::vector<std::string> folders;
            for (const KNOWNFOLDERID* id : ids) folders.push_back(known_folder(*id, ""));
            return folders;
        }

        // Whether folder (a full path) is a drive root, or is or holds a folder of Windows or of the users.
        bool system_or_root(const std::wstring& folder)
        {
            if (folder.size() <= 3) return true;
            for (const auto& f : system_folders())
                if (!f.empty() && same_or_inside(full_folder(f), folder)) return true;
            return false;
        }

        // Whether folder (a full path) lies in the folder of Windows or in the folders of the users.
        bool inside_windows_or_users(const std::wstring& folder)
        {
            for (const KNOWNFOLDERID* id : { &FOLDERID_Windows, &FOLDERID_UserProfiles, &FOLDERID_Profile })
            {
                const std::string f = known_folder(*id, "");
                if (!f.empty() && same_or_inside(folder, full_folder(f))) return true;
            }
            return false;
        }

        // SYSTEM, the administrators, TrustedInstaller, and the account that runs this program.
        bool trusted_account(PSID sid)
        {
            if (!sid || !IsValidSid(sid)) return false;
            if (IsWellKnownSid(sid, WinLocalSystemSid) || IsWellKnownSid(sid, WinBuiltinAdministratorsSid)) return true;
            LPWSTR text = nullptr;
            bool installer = false;
            if (ConvertSidToStringSidW(sid, &text))
            {
                installer = std::wcscmp(text, L"S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464") == 0;
                LocalFree(text);
            }
            if (installer) return true;
            HANDLE token = nullptr;
            if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
            alignas(TOKEN_USER) BYTE user[sizeof(TOKEN_USER) + SECURITY_MAX_SID_SIZE];
            DWORD size = 0;
            const bool same = GetTokenInformation(token, TokenUser, user, sizeof(user), &size)
                && EqualSid(reinterpret_cast<TOKEN_USER*>(user)->User.Sid, sid);
            CloseHandle(token);
            return same;
        }

        /*
            Whether another account than those of trusted_account() may rename folder (a full
            path), or what it holds, and put a folder of its own in its place: as its owner,
            or through a right over its permissions, its deletion or the deletion of what it
            holds.  For a drive root, only the deletion of what it holds counts.
        */
        bool movable_by_others(const std::wstring& folder)
        {
            const bool root = folder.size() <= 3;
            PSID owner = nullptr;
            PACL dacl = nullptr;
            PSECURITY_DESCRIPTOR sd = nullptr;
            if (GetNamedSecurityInfoW(folder.c_str(), SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION
                    | DACL_SECURITY_INFORMATION, &owner, nullptr, &dacl, nullptr, &sd) != ERROR_SUCCESS)
                return true;
            // A missing list of permissions gives every right to every account.
            bool others = !dacl || (!root && !trusted_account(owner));
            const ACCESS_MASK moves = root ? FILE_DELETE_CHILD | GENERIC_ALL
                : DELETE | FILE_DELETE_CHILD | WRITE_DAC | WRITE_OWNER | GENERIC_ALL;
            for (WORD i = 0; !others && i < dacl->AceCount; ++i)
            {
                void* ace = nullptr;
                if (!GetAce(dacl, i, &ace))
                {
                    others = true;
                    break;
                }
                // The entries that pass only to what the folder holds, and the denials, change nothing here.
                const ACE_HEADER* h = static_cast<const ACE_HEADER*>(ace);
                if (h->AceType != ACCESS_ALLOWED_ACE_TYPE || (h->AceFlags & INHERIT_ONLY_ACE)) continue;
                ACCESS_ALLOWED_ACE* allowed = static_cast<ACCESS_ALLOWED_ACE*>(ace);
                others = (allowed->Mask & moves) != 0 && !trusted_account(&allowed->SidStart);
            }
            LocalFree(sd);
            return others;
        }

        // Whether a folder that exists on the path of folder (a full path), its drive root
        // included, lets another account put a folder of its own in the place of the next one.
        bool movable_parents(const std::wstring& folder)
        {
            if (movable_by_others(folder.substr(0, 3))) return true;
            for (size_t at = 3; at < folder.size(); ++at)
            {
                if (folder[at] != L'\\') continue;
                const std::wstring parent = folder.substr(0, at);
                // The folders to come are made as secure_parents() makes them.
                if (!dir_exists(narrow(parent))) break;
                if (movable_by_others(parent)) return true;
            }
            return false;
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
        return join_path(known_folder(FOLDERID_ProgramFiles, "C:\\Program Files"), "Pseudalis");
    }

    std::string folder_problem(const std::string& dir, bool program, const std::string& program_dir)
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
        // The folders of Windows and of the users, the data folder, and those that hold them.
        std::vector<std::string> folders = system_folders();
        folders.push_back(data_dir());
        for (const auto& f : folders)
            if (!f.empty() && same_or_inside(widen(f), full)) return "folder_system";
        // The programs of the service stay out of the folder of Windows and of the folders of the users.
        if (program && inside_windows_or_users(full)) return "folder_system";
        // The data stays apart from the programs, whose folders an update replaces.
        if (!program && !program_dir.empty())
        {
            const std::wstring programs = full_folder(program_dir);
            if (same_or_inside(full, programs) || same_or_inside(programs, full)) return "folder_overlap";
        }
        // The links on its path, as secure_folder() accepts them.
        for (size_t at = 3; at <= full.size(); ++at)
        {
            if (at < full.size() && full[at] != L'\\') continue;
            const std::string prefix = narrow(full.substr(0, at));
            if (is_link(prefix) && !owned_by_system(prefix)) return "folder_foreign";
        }
        // Another account that may rename a folder on its path could put programs of its own in their place.
        if (program && movable_parents(full)) return "folder_parent_unsafe";
        const std::string path = narrow(full);
        if (!dir_exists(path)) return file_exists(path) ? "folder_not_empty" : std::string();
        const bool empty = list_dir(path).empty();
        // An installation of Pseudalis, or what one left: its list of files.
        const bool pseudalis = file_exists(join_path(path, "pseudalis-service.exe"))
            || file_exists(join_path(path, "pseudalis-files.json"));
        if (program && !empty && !pseudalis) return "folder_not_empty";
        if (!empty && !owned_by_system(final_path(path))) return "folder_foreign";
        return std::string();
    }

    std::string data_folder_problem(const std::string& dir, const std::string& program_dir)
    {
        if (dir.size() < 3 || !std::isalpha(static_cast<unsigned char>(dir[0])) || dir[1] != ':'
            || (dir[2] != '\\' && dir[2] != '/'))
            return "folder_not_local";
        // The data stays on a fixed drive: neither a removable drive nor a network share.
        const UINT drive = GetDriveTypeW(widen(dir.substr(0, 2) + "\\").c_str());
        if (drive != DRIVE_FIXED) return drive == DRIVE_NO_ROOT_DIR ? "folder_not_local" : "folder_not_fixed";
        const std::wstring full = full_folder(dir);
        if (system_or_root(full) || inside_windows_or_users(full)) return "folder_system";
        if (!program_dir.empty())
        {
            const std::wstring programs = full_folder(program_dir);
            if (same_or_inside(full, programs) || same_or_inside(programs, full)) return "folder_overlap";
        }
        // The links on its path, as secure_folder() accepts them.
        for (size_t at = 3; at <= full.size(); ++at)
        {
            if (at < full.size() && full[at] != L'\\') continue;
            const std::string prefix = narrow(full.substr(0, at));
            if (is_link(prefix) && !owned_by_system(prefix)) return "folder_foreign";
        }
        // Another account that may rename a folder on its path could put its own data in its place.
        if (movable_parents(full)) return "folder_parent_unsafe";
        const std::string path = narrow(full);
        if (!dir_exists(path)) return file_exists(path) ? "folder_not_empty" : std::string();
        if (list_dir(path).empty()) return std::string();
        // The data of an earlier installation, kept for this one.
        const bool pseudalis = file_exists(join_path(path, "settings.json"))
            || file_exists(join_path(path, "pseudalis.db"));
        if (!pseudalis) return "folder_not_empty";
        if (!owned_by_system(final_path(path))) return "folder_foreign";
        return std::string();
    }

    bool secure_parents(const std::string& dir, std::string& error)
    {
        const std::wstring full = full_folder(dir);
        for (size_t at = 3; at < full.size(); ++at)
        {
            if (full[at] != L'\\') continue;
            const std::string parent = narrow(full.substr(0, at));
            if (!dir_exists(parent) && !secure_folder(parent, error)) return false;
        }
        return true;
    }

    namespace
    {
        const wchar_t pseudalis_key[] = L"Software\\Pseudalis";

        bool write_registry_string(const wchar_t* name, const std::wstring& value)
        {
            HKEY k = nullptr;
            if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, pseudalis_key, 0, nullptr, 0, KEY_WRITE | KEY_WOW64_64KEY, nullptr,
                    &k, nullptr) != ERROR_SUCCESS)
                return false;
            const bool ok = RegSetValueExW(k, name, 0, REG_SZ, reinterpret_cast<const BYTE*>(value.c_str()),
                static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t))) == ERROR_SUCCESS;
            RegCloseKey(k);
            return ok;
        }
    }

    std::string recorded_data_dir()
    {
        return narrow(registry_string(HKEY_LOCAL_MACHINE, pseudalis_key, L"DataDir"));
    }

    std::string default_data_dir()
    {
        return join_path(known_folder(FOLDERID_ProgramData, "C:\\ProgramData"), "Pseudalis");
    }

    bool record_data_dir(const std::string& dir, std::string& error)
    {
        if (write_registry_string(L"DataDir", full_folder(dir))) return true;
        error = last_error("cannot record the data folder in the registry");
        return false;
    }

    void forget_folders()
    {
        RegDeleteKeyExW(HKEY_LOCAL_MACHINE, pseudalis_key, KEY_WOW64_64KEY, 0);
    }

    bool removable_folder(const std::string& dir)
    {
        if (dir.size() < 3 || !std::isalpha(static_cast<unsigned char>(dir[0])) || dir[1] != ':'
            || (dir[2] != '\\' && dir[2] != '/'))
            return false;
        const UINT drive = GetDriveTypeW(widen(dir.substr(0, 2) + "\\").c_str());
        if (drive != DRIVE_FIXED && drive != DRIVE_REMOVABLE && drive != DRIVE_RAMDISK) return false;
        return !system_or_root(full_folder(dir));
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
        // The data folder recorded by an installation, in place or kept by its uninstallation.
        o.data_dir = data_dir();
        std::string text;
        if (read_file(join_path(data_dir(), "settings.json"), text, 1u << 20))
        {
            const json s = json::parse_or_null(text);
            const long long port = s["port"].as_int(0);
            if (port >= 1024 && port <= 65535) o.port = static_cast<int>(port);
            if (s.contains("drive_letter")) o.drive_letter = s["drive_letter"].str();
            /* The drive of restoration came with the drives without driver: an installation of the
               version before takes the first free letter from R on. */
            if (s.contains("restore_letter")) o.restore_letter = s["restore_letter"].str();
            else o.restore_letter = default_restore_letter(o.drive_letter);
            const std::string language = s["language"].str("auto");
            if (language == "en" || language == "fr" || language == "de" || language == "it" || language == "es")
                o.language = language;
            o.trust_certificate = s["trust_authority"].as_bool(true);
        }
        if (!dir.empty())
        {
            o.autostart = !registry_string(HKEY_LOCAL_MACHINE, run_key, L"Pseudalis").empty();
            o.desktop_shortcut = file_exists(desktop_link());
            o.context_menu = context_menu_registered();
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

    bool service_marked_for_deletion(std::string& error)
    {
        service_handles h;
        if (!open_service(h, SERVICE_CHANGE_CONFIG)) return false;
        // A change that changes nothing fails for a service that waits for its deletion.
        if (ChangeServiceConfigW(h.svc, SERVICE_NO_CHANGE, SERVICE_NO_CHANGE, SERVICE_NO_CHANGE, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr)
            || GetLastError() != ERROR_SERVICE_MARKED_FOR_DELETE)
            return false;
        error = service_marked_text;
        return true;
    }

    bool service_disabled()
    {
        service_handles h;
        if (!open_service(h, SERVICE_QUERY_CONFIG)) return false;
        DWORD size = 0;
        QueryServiceConfigW(h.svc, nullptr, 0, &size);
        if (size == 0) return false;
        std::vector<unsigned char> buf(size);
        QUERY_SERVICE_CONFIGW* config = reinterpret_cast<QUERY_SERVICE_CONFIGW*>(buf.data());
        return QueryServiceConfigW(h.svc, config, size, &size) && config->dwStartType == SERVICE_DISABLED;
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

    bool register_service(const std::string& program_dir, bool keep_start_type, std::string& error)
    {
        SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
        if (!scm)
        {
            error = last_error("cannot open the service manager");
            return false;
        }
        const std::wstring name = widen(service_name);
        const std::wstring path = L"\"" + widen(join_path(program_dir, "pseudalis-service.exe")) + L"\"";
        SC_HANDLE svc = OpenServiceW(scm, name.c_str(), SERVICE_ALL_ACCESS);
        // The start of a service in place stays as an administrator set it, on an update.
        const bool keep = svc && keep_start_type;
        if (svc)
        {
            const DWORD start = keep ? SERVICE_NO_CHANGE : SERVICE_AUTO_START;
            if (!ChangeServiceConfigW(svc, SERVICE_WIN32_OWN_PROCESS, start, SERVICE_ERROR_NORMAL, path.c_str(),
                    nullptr, nullptr, nullptr, nullptr, nullptr, L"Pseudalis"))
            {
                error = GetLastError() == ERROR_SERVICE_MARKED_FOR_DELETE ? std::string(service_marked_text)
                    : last_error("cannot configure the service");
                CloseServiceHandle(svc);
                CloseServiceHandle(scm);
                return false;
            }
        }
        else
        {
            svc = CreateServiceW(scm, name.c_str(), L"Pseudalis", SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS,
                SERVICE_AUTO_START, SERVICE_ERROR_NORMAL, path.c_str(), nullptr, nullptr, nullptr, nullptr, nullptr);
        }
        if (!svc)
        {
            error = last_error("cannot register the service");
            CloseServiceHandle(scm);
            return false;
        }
        std::wstring text = L"Protects the personal data of documents: the drive, the engine and the local "
            L"interface of Pseudalis.";
        SERVICE_DESCRIPTIONW desc = { &text[0] };
        ChangeServiceConfig2W(svc, SERVICE_CONFIG_DESCRIPTION, &desc);
        if (!keep)
        {
            // The drive is there when the users sign in: the service starts with Windows, without delay.
            SERVICE_DELAYED_AUTO_START_INFO delayed = { FALSE };
            ChangeServiceConfig2W(svc, SERVICE_CONFIG_DELAYED_AUTO_START_INFO, &delayed);
        }
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
        while ((w = FindWindowExW(nullptr, w, L"PseudalisTrayWindow", nullptr)) != nullptr)
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
        // The processes, of every session, whose program lies in folder (or is folder\program), this one aside.
        std::vector<DWORD> processes_in(const std::wstring& folder, const std::string& program)
        {
            std::vector<DWORD> out;
            HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if (snap == INVALID_HANDLE_VALUE) return out;
            const std::wstring file = program.empty() ? std::wstring() : folder + L"\\" + widen(program);
            PROCESSENTRY32W e = {};
            e.dwSize = sizeof(e);
            for (BOOL more = Process32FirstW(snap, &e); more; more = Process32NextW(snap, &e))
            {
                if (e.th32ProcessID == 0 || e.th32ProcessID == GetCurrentProcessId()) continue;
                HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, e.th32ProcessID);
                if (!p) continue;
                std::wstring image(32768, L'\0');
                DWORD n = static_cast<DWORD>(image.size());
                if (QueryFullProcessImageNameW(p, 0, &image[0], &n))
                {
                    image.resize(n);
                    if (file.empty() ? same_or_inside(image, folder) : same_text(image, file))
                        out.push_back(e.th32ProcessID);
                }
                CloseHandle(p);
            }
            CloseHandle(snap);
            return out;
        }

        // The right of the administrators to end the processes of the other accounts.
        void enable_debug_privilege()
        {
            HANDLE token = nullptr;
            if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) return;
            TOKEN_PRIVILEGES tp = {};
            tp.PrivilegeCount = 1;
            tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            if (LookupPrivilegeValueW(nullptr, L"SeDebugPrivilege", &tp.Privileges[0].Luid))
                AdjustTokenPrivileges(token, FALSE, &tp, sizeof(tp), nullptr, nullptr);
            CloseHandle(token);
        }
    }

    bool end_processes(const std::string& folder, unsigned timeout_ms, const std::string& program)
    {
        const std::wstring dir = full_folder(folder);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        std::vector<DWORD> left = processes_in(dir, program);
        while (!left.empty() && std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            left = processes_in(dir, program);
        }
        if (left.empty()) return true;
        enable_debug_privilege();
        for (DWORD pid : left)
        {
            HANDLE p = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid);
            if (!p) continue;
            log("the process " + std::to_string(pid) + " of " + narrow(dir) + " is ended");
            TerminateProcess(p, 1);
            WaitForSingleObject(p, 5000);
            CloseHandle(p);
        }
        left = processes_in(dir, program);
        for (DWORD pid : left) log("the process " + std::to_string(pid) + " of " + narrow(dir) + " still runs");
        return left.empty();
    }

    std::string program_version(const std::string& path)
    {
        const DWORD flags = LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE;
        HMODULE m = LoadLibraryExW(widen(path).c_str(), nullptr, flags);
        if (!m) return std::string();
        HRSRC r = FindResourceW(m, MAKEINTRESOURCEW(VS_VERSION_INFO), RT_VERSION);
        HGLOBAL g = r ? LoadResource(m, r) : nullptr;
        const BYTE* data = g ? static_cast<const BYTE*>(LockResource(g)) : nullptr;
        const DWORD size = r ? SizeofResource(m, r) : 0;
        std::string version;
        // The fixed part of the version starts with its signature, on a boundary of 4 bytes.
        for (DWORD at = 0; data && at + sizeof(VS_FIXEDFILEINFO) <= size && version.empty(); at += 4)
        {
            VS_FIXEDFILEINFO info;
            std::memcpy(&info, data + at, sizeof(info));
            if (info.dwSignature != 0xFEEF04BD) continue;
            version = std::to_string(HIWORD(info.dwFileVersionMS)) + "." + std::to_string(LOWORD(info.dwFileVersionMS))
                + "." + std::to_string(HIWORD(info.dwFileVersionLS));
        }
        FreeLibrary(m);
        return version;
    }

    namespace
    {
        /*
            Renames a file, trying again for five seconds: a file that a scanner or an
            indexer reads is released soon.  A running program may be renamed.  With
            replace, a file of the name to stands in the way of nothing.
        */
        bool move_file(const std::string& from, const std::string& to, bool replace = false)
        {
            const DWORD flags = MOVEFILE_WRITE_THROUGH | (replace ? MOVEFILE_REPLACE_EXISTING : 0);
            DWORD e = ERROR_SUCCESS;
            for (int attempt = 0; attempt < 20; ++attempt)
            {
                if (MoveFileExW(widen(from).c_str(), widen(to).c_str(), flags)) return true;
                e = GetLastError();
                if (e != ERROR_SHARING_VIOLATION && e != ERROR_ACCESS_DENIED && e != ERROR_LOCK_VIOLATION) return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            // The error of the last attempt, which the waits may clear, for the message of the caller.
            SetLastError(e);
            return false;
        }

        // Deletes a file now, or under a name of its own at the next restart of Windows when it
        // is in use, so that a file installed meanwhile under its name stays.
        void delete_file(const std::string& p)
        {
            if (DeleteFileW(widen(p).c_str()) || GetLastError() == ERROR_FILE_NOT_FOUND) return;
            // A file marked read only loses the mark first.
            const DWORD attributes = GetFileAttributesW(widen(p).c_str());
            if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_READONLY)
                && SetFileAttributesW(widen(p).c_str(), attributes & ~FILE_ATTRIBUTE_READONLY)
                && DeleteFileW(widen(p).c_str()))
                return;
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
        bool ok = body([h](const void* data, size_t n) {
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
        // On the disk before the renames of commit(): a file renamed into place is never left empty.
        if (ok && !FlushFileBuffers(h))
        {
            ok = false;
            error = last_error(("cannot write " + fresh).c_str());
        }
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
                    // A file the change removes stays where it is, for finish().
                    if (e.removal)
                    {
                        e.kept = true;
                        continue;
                    }
                    error = last_error(("cannot replace " + e.target).c_str());
                    if (!restore(i, unrestored_)) error += "; not put back: " + unrestored_;
                    return false;
                }
                e.aside = aside;
            }
            if (!e.removal)
            {
                if (!move_file(e.fresh, e.target))
                {
                    error = last_error(("cannot install " + e.target).c_str());
                    if (!restore(i + 1, unrestored_)) error += "; not put back: " + unrestored_;
                    return false;
                }
                e.placed = true;
            }
        }
        committed_ = true;
        return true;
    }

    bool staged_files::restore(size_t count, std::string& failed)
    {
        bool ok = true;
        for (size_t i = count; i-- > 0;)
        {
            entry& e = entries_[i];
            // The new file goes back to its own name, which the destructor deletes.
            if (e.placed) move_file(e.target, e.fresh, true);
            // The previous file takes its place again, over the new one when that could not move.
            if (!e.aside.empty() && !move_file(e.aside, e.target, true))
            {
                ok = false;
                failed += (failed.empty() ? "" : ", ") + e.target + " (kept as " + e.aside + ")";
                log("cannot put back " + e.target + ": it stays as " + e.aside);
            }
            e.placed = false;
            e.aside.clear();
            e.kept = false;
        }
        return ok;
    }

    bool staged_files::rollback(std::string& error)
    {
        if (!committed_)
        {
            error = unrestored_;
            return unrestored_.empty();
        }
        committed_ = false;
        std::string failed;
        const bool ok = restore(entries_.size(), failed);
        if (!ok) error = failed;
        return ok;
    }

    void staged_files::finish()
    {
        // The change stays: rollback() has nothing left to put back.
        for (auto& e : entries_)
        {
            if (!e.aside.empty()) delete_file(e.aside);
            // A file to remove that could not be set aside goes now, or at the next restart of Windows.
            if (e.kept) delete_file(e.target);
            e.aside.clear();
            e.placed = false;
            e.kept = false;
        }
    }

    std::vector<std::string> staged_files::targets() const
    {
        std::vector<std::string> out;
        for (const auto& e : entries_)
            if (!e.removal) out.push_back(e.target);
        return out;
    }

    namespace
    {
        // Whether name ends with tag followed by the 8 hexadecimal digits of random_hex(8).
        bool ends_with_mark(const std::string& name, const char* tag)
        {
            const size_t n = std::strlen(tag);
            if (name.size() < n + 8 || name.compare(name.size() - n - 8, n, tag) != 0) return false;
            for (size_t i = name.size() - 8; i < name.size(); ++i)
                if (!std::isxdigit(static_cast<unsigned char>(name[i]))) return false;
            return true;
        }

        /*
            The files the programs make in their folder: those an update sets aside
            (name.old-<random>) or leaves unfinished (name.new), and the temporary files of
            a write (name.tmp-<random>).
        */
        bool program_made(const std::string& name)
        {
            const bool unfinished = name.size() > 4 && name.compare(name.size() - 4, 4, ".new") == 0;
            return unfinished || ends_with_mark(name, ".old-") || ends_with_mark(name, ".tmp-");
        }
    }

    size_t remove_old_files(const std::string& program_dir, bool now_only)
    {
        size_t left = 0;
        for (const auto& name : list_dir(program_dir))
        {
            const std::string p = join_path(program_dir, name);
            if (is_link(p)) continue;
            if (dir_exists(p))
            {
                left += remove_old_files(p, now_only);
                continue;
            }
            if (!program_made(name)) continue;
            if (!now_only) delete_file(p);
            else DeleteFileW(widen(p).c_str());
            if (file_exists(p)) ++left;
        }
        return left;
    }

    namespace
    {
        HANDLE own_program = INVALID_HANDLE_VALUE;
        std::string own_program_problem = "the file of the installer was not opened";

        bool same_file(HANDLE a, HANDLE b)
        {
            BY_HANDLE_FILE_INFORMATION x, y;
            return GetFileInformationByHandle(a, &x) && GetFileInformationByHandle(b, &y) &&
                x.dwVolumeSerialNumber == y.dwVolumeSerialNumber && x.nFileIndexHigh == y.nFileIndexHigh &&
                x.nFileIndexLow == y.nFileIndexLow;
        }

        // Copies the whole content of an open file into a new file.
        bool copy_from_handle(HANDLE from, const std::wstring& to, std::string& error)
        {
            LARGE_INTEGER zero = {};
            if (!SetFilePointerEx(from, zero, nullptr, FILE_BEGIN))
            {
                error = last_error("cannot read the file of the installer");
                return false;
            }
            HANDLE out = CreateFileW(to.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                nullptr);
            if (out == INVALID_HANDLE_VALUE)
            {
                error = last_error("cannot write the uninstaller");
                return false;
            }
            std::vector<char> buffer(1u << 20);
            bool ok = true;
            for (;;)
            {
                DWORD got = 0, put = 0;
                if (!ReadFile(from, buffer.data(), static_cast<DWORD>(buffer.size()), &got, nullptr))
                {
                    ok = false;
                    error = last_error("cannot read the file of the installer");
                    break;
                }
                if (got == 0) break;
                if (!WriteFile(out, buffer.data(), got, &put, nullptr) || put != got)
                {
                    ok = false;
                    error = last_error("cannot write the uninstaller");
                    break;
                }
            }
            ok = ok && FlushFileBuffers(out);
            CloseHandle(out);
            if (!ok) DeleteFileW(to.c_str());
            return ok;
        }
    }

    void hold_own_program()
    {
        if (own_program != INVALID_HANDLE_VALUE) return;
        // Read only, and shared for reading only: no rename, no deletion, no write until the end.
        HANDLE h = CreateFileW(widen(executable_path()).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE)
        {
            own_program_problem = last_error("cannot open the file of the installer");
            return;
        }
        /*
            The file of the image of this process, by its device path, which follows the
            renames: the file opened above must be that one, not a file put in its place
            between the start of the process and its opening.
        */
        typedef DWORD (WINAPI* mapped_name)(HANDLE, LPVOID, LPWSTR, DWORD);
        const mapped_name name_of = reinterpret_cast<mapped_name>(reinterpret_cast<void*>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32GetMappedFileNameW")));
        wchar_t device_path[2048];
        const DWORD n = name_of ? name_of(GetCurrentProcess(), reinterpret_cast<LPVOID>(GetModuleHandleW(nullptr)),
            device_path, 2048) : 0;
        const std::wstring image_path = L"\\\\?\\GLOBALROOT" + std::wstring(device_path, n < 2048 ? n : 0);
        HANDLE image = n > 0 && n < 2048 ? CreateFileW(image_path.c_str(), FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr)
            : INVALID_HANDLE_VALUE;
        if (image == INVALID_HANDLE_VALUE)
        {
            // A system that cannot name the file of the image (Wine, for one): the file held is the reference.
            log("the file of the installer is held, its image could not be compared with it");
            own_program = h;
            return;
        }
        const bool same = same_file(h, image);
        CloseHandle(image);
        if (!same)
        {
            CloseHandle(h);
            own_program_problem = "the file of the installer is no longer the program that runs";
            log(own_program_problem);
            return;
        }
        own_program = h;
    }

    bool stage_uninstaller(const std::string& program_dir, staged_files& files, std::string& error)
    {
        const std::string self = executable_path();
        const std::string target = join_path(program_dir, "pseudalis-setup.exe");
        if (_wcsicmp(widen(self).c_str(), widen(target).c_str()) == 0) return true;
        if (own_program == INVALID_HANDLE_VALUE)
        {
            if (error.empty()) error = own_program_problem;
            return false;
        }
        const std::wstring tmp = widen(files.add(target));
        if (!copy_from_handle(own_program, tmp, error)) return false;
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
        // A data folder that another account made, or its link, is no folder of Pseudalis.
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
        if (!secure_parents(dir, error) || !secure_folder(dir, error, !content_checked)) return false;
        for (const char* sub : { "logs", "certificates" }) make_dirs(join_path(dir, sub));
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
        s["drive_letter"] = o.drive_letter;
        s["restore_letter"] = o.restore_letter;
        s["language"] = o.language;
        // The service puts its authority in the trusted roots according to this choice.
        s["trust_authority"] = o.trust_certificate;
        if (!write_file_atomic(path, s.dump_formatted(2) + "\n"))
        {
            error = "cannot write " + path;
            return false;
        }
        return true;
    }

    bool add_missing_settings(const install_options& o, std::string& error)
    {
        const std::string path = join_path(data_dir(), "settings.json");
        std::string text;
        if (!read_file(path, text, 1u << 20)) return true;
        json s = json::parse_or_null(text);
        if (!s.is_object() || s.contains("restore_letter")) return true;
        s["restore_letter"] = o.restore_letter;
        if (!write_file_atomic(path, s.dump_formatted(2) + "\n"))
        {
            error = "cannot write " + path;
            return false;
        }
        log("the drive of restoration takes " + (o.restore_letter.empty() ? std::string("no letter")
                                                                          : o.restore_letter + ":"));
        return true;
    }

    void remove_drive_labels(const std::string& program_dir)
    {
        const std::wstring base = L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\DriveIcons\\";
        const std::wstring icon = widen(join_path(program_dir, "pseudalis-tray.exe"));
        for (wchar_t c = L'A'; c <= L'Z'; ++c)
        {
            const std::wstring key = base + c;
            const std::wstring value = registry_string(HKEY_LOCAL_MACHINE, (key + L"\\DefaultIcon").c_str(), L"");
            // Only the names that the service of these programs gave.
            if (value.size() >= icon.size() && _wcsnicmp(value.c_str(), icon.c_str(), icon.size()) == 0)
                RegDeleteTreeW(HKEY_LOCAL_MACHINE, key.c_str());
        }
    }

    bool prepare_certificates(bool new_installation, std::string& error)
    {
        // A new installation lets the service make its own authority: none found in the folder stays.
        const std::string dir = join_path(data_dir(), "certificates");
        if (new_installation && !remove_certificate_files(dir))
        {
            error = last_error(("cannot replace the certificates of " + dir).c_str());
            return false;
        }
        return true;
    }

    bool authority_trusted()
    {
        return pseudalis::authority_trusted();
    }

    bool trust_authority(std::string& error)
    {
        return pseudalis::trust_authority(join_path(data_dir(), "certificates"), error);
    }

    void untrust_authority()
    {
        pseudalis::untrust_authority();
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
        const std::wstring setup = L"\"" + widen(join_path(program_dir, "pseudalis-setup.exe")) + L"\"";
        str(L"DisplayName", widen(PSEUDALIS_FULL_NAME));
        str(L"DisplayVersion", widen(PSEUDALIS_VERSION));
        str(L"Publisher", widen(PSEUDALIS_PUBLISHER));
        str(L"DisplayIcon", widen(join_path(program_dir, "pseudalis-tray.exe")) + L",0");
        str(L"InstallLocation", dir);
        str(L"UninstallString", setup + L" /uninstall");
        str(L"QuietUninstallString", setup + L" /uninstall /silent");
        str(L"URLInfoAbout", widen(PSEUDALIS_PROJECT_URL));
        str(L"HelpLink", widen(PSEUDALIS_PROJECT_URL));
        str(L"Comments", L"Protection of the personal data of documents, written by " + widen(PSEUDALIS_AUTHOR_NAME) +
            L", based on the open source Dlib-Transformer-extensions library");
        str(L"Contact", widen(PSEUDALIS_AUTHOR_EMAIL));
        time_t now = time(nullptr);
        tm t;
        gmtime_s(&t, &now);
        wchar_t date[16];
        swprintf(date, 16, L"%04d%02d%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday);
        str(L"InstallDate", date);
        dword(L"EstimatedSize", static_cast<DWORD>(size_bytes / 1024));
        dword(L"NoModify", 1);
        dword(L"NoRepair", 1);
        dword(L"VersionMajor", PSEUDALIS_VERSION_MAJOR);
        dword(L"VersionMinor", PSEUDALIS_VERSION_MINOR);
        // The content of the data folder was checked at the start of this installation (prepare_data_dir()).
        dword(L"DataFolderChecked", 1);
        RegCloseKey(k);
        return true;
    }

    void remove_uninstall()
    {
        RegDeleteKeyExW(HKEY_LOCAL_MACHINE, uninstall_key, KEY_WOW64_64KEY, 0);
    }

    uninstall_entry save_uninstall_entry()
    {
        uninstall_entry entry;
        HKEY k = nullptr;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, uninstall_key, 0, KEY_READ | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS)
            return entry;
        entry.present = true;
        for (DWORD i = 0;; ++i)
        {
            wchar_t name[16384];
            DWORD name_size = 16384, type = 0, data_size = 0;
            if (RegEnumValueW(k, i, name, &name_size, nullptr, &type, nullptr, &data_size) != ERROR_SUCCESS) break;
            uninstall_entry::value v;
            v.name.assign(name, name_size);
            v.type = type;
            v.data.resize(data_size);
            name_size = 16384;
            if (RegEnumValueW(k, i, name, &name_size, nullptr, &type, v.data.empty() ? nullptr : &v.data[0],
                    &data_size) != ERROR_SUCCESS)
                break;
            v.data.resize(data_size);
            entry.values.push_back(v);
        }
        RegCloseKey(k);
        return entry;
    }

    void restore_uninstall_entry(const uninstall_entry& entry)
    {
        remove_uninstall();
        if (!entry.present) return;
        HKEY k = nullptr;
        if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, uninstall_key, 0, nullptr, 0, KEY_WRITE | KEY_WOW64_64KEY, nullptr, &k,
                nullptr) != ERROR_SUCCESS)
        {
            log(last_error("cannot write back the entry of the installed programs"));
            return;
        }
        for (const auto& v : entry.values)
            RegSetValueExW(k, v.name.c_str(), 0, v.type, v.data.empty() ? nullptr : v.data.data(),
                static_cast<DWORD>(v.data.size()));
        RegCloseKey(k);
    }

    void set_autostart(const std::string& program_dir, bool on)
    {
        HKEY k = nullptr;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, run_key, 0, KEY_WRITE | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return;
        if (on)
        {
            const std::wstring v = L"\"" + widen(join_path(program_dir, "pseudalis-tray.exe")) + L"\"";
            RegSetValueExW(k, L"Pseudalis", 0, REG_SZ, reinterpret_cast<const BYTE*>(v.c_str()),
                static_cast<DWORD>((v.size() + 1) * sizeof(wchar_t)));
        }
        else
        {
            RegDeleteValueW(k, L"Pseudalis");
        }
        RegCloseKey(k);
    }

    void create_shortcuts(const std::string& program_dir, bool desktop)
    {
        const std::wstring tray = widen(join_path(program_dir, "pseudalis-tray.exe"));
        const std::wstring desc = L"Pseudalis";
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

    namespace
    {
        // Whether a server accepts a connection on the port of the loopback address; WSAStartup() was called.
        bool port_accepts(int port)
        {
            SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (s == INVALID_SOCKET) return false;
            sockaddr_in a = {};
            a.sin_family = AF_INET;
            a.sin_port = htons(static_cast<u_short>(port));
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            const bool ok = connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0;
            closesocket(s);
            return ok;
        }

        // The port of settings.json, which the service rewrites when it falls back on its last port.
        int settings_port()
        {
            std::string text;
            if (!read_file(join_path(data_dir(), "settings.json"), text, 1u << 20)) return 8444;
            const long long port = json::parse_or_null(text)["port"].as_int(8444);
            return port >= 1024 && port <= 65535 ? static_cast<int>(port) : 8444;
        }

        typedef DWORD(WINAPI* tcp_table_function)(PVOID, PDWORD, BOOL, ULONG, TCP_TABLE_CLASS, ULONG);

        /*
            Whether the process pid listens on the TCP port of the IPv4 addresses: 1 when it
            does, 0 when no process does or another one, -1 when the table of the listening
            sockets cannot be read or does not name the process of a socket on that port.
            iphlpapi.dll comes from System32 only, at the first call.
        */
        int listened_by(DWORD pid, int port)
        {
            static const tcp_table_function table_of = []() {
                HMODULE m = LoadLibraryExW(L"iphlpapi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
                return m ? reinterpret_cast<tcp_table_function>(
                    reinterpret_cast<void*>(GetProcAddress(m, "GetExtendedTcpTable"))) : nullptr;
            }();
            if (!table_of) return -1;
            std::vector<unsigned char> buffer;
            ULONG size = 16384;
            DWORD r = ERROR_INSUFFICIENT_BUFFER;
            // The table grows between the two calls when sockets open meanwhile.
            for (int attempt = 0; attempt < 4 && r == ERROR_INSUFFICIENT_BUFFER; ++attempt)
            {
                buffer.resize(size + 4096);
                size = static_cast<ULONG>(buffer.size());
                r = table_of(buffer.data(), &size, FALSE, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0);
            }
            if (r != NO_ERROR) return -1;
            const MIB_TCPTABLE_OWNER_PID* table = reinterpret_cast<const MIB_TCPTABLE_OWNER_PID*>(buffer.data());
            int found = 0;
            for (DWORD i = 0; i < table->dwNumEntries; ++i)
            {
                const MIB_TCPROW_OWNER_PID& row = table->table[i];
                if (ntohs(static_cast<u_short>(row.dwLocalPort)) != port) continue;
                if (row.dwOwningPid == pid) return 1;
                if (row.dwOwningPid == 0) found = -1;
            }
            return found;
        }
    }

    bool wait_for_service(unsigned timeout_ms, std::string& error)
    {
        service_handles h;
        if (!open_service(h, SERVICE_QUERY_STATUS))
        {
            error = last_error("cannot open the service");
            return false;
        }
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
        {
            error = "cannot use the network interface of Windows";
            return false;
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        bool ok = false;
        for (;;)
        {
            SERVICE_STATUS_PROCESS sp = {};
            DWORD needed = 0;
            if (!QueryServiceStatusEx(h.svc, SC_STATUS_PROCESS_INFO, reinterpret_cast<LPBYTE>(&sp), sizeof(sp),
                    &needed))
            {
                error = last_error("cannot query the service");
                break;
            }
            if (sp.dwCurrentState == SERVICE_STOPPED)
            {
                const DWORD code = sp.dwWin32ExitCode == ERROR_SERVICE_SPECIFIC_ERROR ? sp.dwServiceSpecificExitCode
                    : sp.dwWin32ExitCode;
                // A stop without an error is the one of the service itself: for an update it started, or when asked.
                if (code == NO_ERROR)
                {
                    log("the service stopped by itself after its start, without an error");
                    ok = true;
                    break;
                }
                error = "the service stopped at its start (exit code " + std::to_string(code) + ")";
                break;
            }
            if (sp.dwCurrentState == SERVICE_RUNNING)
            {
                /*
                    The server of the service listens on the port, not another program that
                    took it meanwhile; a connection to the port tells it only when the table
                    of the sockets does not name their processes.
                */
                const int port = settings_port();
                const int listener = listened_by(sp.dwProcessId, port);
                if (listener == 1 || (listener < 0 && port_accepts(port)))
                {
                    ok = true;
                    break;
                }
            }
            if (std::chrono::steady_clock::now() > deadline)
            {
                error = "the service did not open its port " + std::to_string(settings_port()) + " within "
                    + std::to_string(timeout_ms / 1000) + " seconds";
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        WSACleanup();
        return ok;
    }

    void log_service_tail()
    {
        std::string text;
        if (!read_file(join_path(join_path(data_dir(), "logs"), "service.log"), text, 64u << 20) || text.empty())
            return;
        // The last lines, those of the start that failed.
        const size_t keep = 4096;
        if (text.size() > keep)
        {
            const size_t from = text.find('\n', text.size() - keep);
            text = text.substr(from == std::string::npos ? text.size() - keep : from + 1);
        }
        log("the end of the log of the service:\n" + text);
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

        const char* const file_list_name = "pseudalis-files.json";

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

        // The folders and the programs of Pseudalis at the top of a folder of programs; winfsp held the
        // installer of WinFsp that the versions before carried.
        const char* const pseudalis_folders[] = { "web", "licenses", "winfsp" };
        const char* const pseudalis_programs[] = { "pseudalis-service.exe", "pseudalis-tray.exe", "pseudalis.exe",
            "pseudalis-setup.exe", "pseudalis-engine.dll", "pseudalis-engine-avx2.dll" };

        bool pseudalis_entry(const std::string& name)
        {
            for (const char* f : pseudalis_folders)
                if (_stricmp(name.c_str(), f) == 0) return true;
            for (const char* f : pseudalis_programs)
                if (_stricmp(name.c_str(), f) == 0) return true;
            return _stricmp(name.c_str(), file_list_name) == 0;
        }

        /*
            Whether an entry at the top of the folder of the programs stays after its removal:
            a folder of keep or a folder that holds one, and, when the folder does not go
            whole (all false), what is neither Pseudalis's nor made by its programs.
        */
        bool stays(const std::string& name, const std::wstring& path, bool all, const std::vector<std::wstring>& keep)
        {
            for (const auto& k : keep)
                if (same_or_inside(k, path)) return true;
            return !all && !program_made(name) && !pseudalis_entry(name);
        }

        // Whether the folder of the programs holds an entry that stays, or one that does not.
        bool holds(const std::string& folder, bool all, const std::vector<std::wstring>& keep, bool staying)
        {
            for (const auto& name : list_dir(folder))
                if (stays(name, full_folder(join_path(folder, name)), all, keep) == staying) return true;
            return false;
        }

        // The files of Pseudalis in a folder of programs, known by their places: its programs, its
        // engines, and the files of its web application, its licenses and, from the versions before,
        // the installer of WinFsp.
        std::vector<std::string> pseudalis_files(const std::string& program_dir)
        {
            std::vector<std::string> files;
            for (const char* name : pseudalis_programs)
                if (file_exists(join_path(program_dir, name))) files.push_back(name);
            for (const char* folder : pseudalis_folders)
                if (!is_link(join_path(program_dir, folder))) collect_files(program_dir, folder, files);
            return files;
        }

        // Removes the folders under dir that hold nothing, deepest first, without following the links.
        void remove_empty_folders(const std::string& dir)
        {
            for (const auto& name : list_dir(dir))
            {
                const std::string p = join_path(dir, name);
                if (is_link(p) || !dir_exists(p)) continue;
                remove_empty_folders(p);
                RemoveDirectoryW(widen(p).c_str());
            }
        }

        // A path as Windows compares it: "\" between its parts.
        std::wstring windows_path(const std::string& p)
        {
            std::wstring w = widen(p);
            std::replace(w.begin(), w.end(), L'/', L'\\');
            return w;
        }

        /*
            Removes what folder holds by the rules of remove_installation(): everything with
            all, otherwise the files the programs make, and at the top of the folder of the
            programs the content of the folders of Pseudalis.  The folders of keep, and the
            file skip, stay; the folders left empty go.  With last, a file in use goes at the
            next restart of Windows; otherwise it stays for a later pass.
        */
        void clear_folder(const std::string& folder, bool all, bool top, const std::vector<std::wstring>& keep,
            const std::wstring& skip, bool last)
        {
            for (const auto& name : list_dir(folder))
            {
                const std::string p = join_path(folder, name);
                const std::wstring w = full_folder(p);
                if (same_text(w, skip)
                    || std::any_of(keep.begin(), keep.end(), [&](const std::wstring& k) { return same_text(w, k); }))
                    continue;
                if (is_link(p))
                {
                    if (all) remove_link(p);
                    continue;
                }
                if (dir_exists(p))
                {
                    bool pseudalis = false;
                    for (const char* f : pseudalis_folders)
                        pseudalis = pseudalis || (top && _stricmp(name.c_str(), f) == 0);
                    // A folder of the user, in a folder of the programs that does not go whole, stays.
                    if (!all && !pseudalis) continue;
                    clear_folder(p, true, false, keep, skip, last);
                    RemoveDirectoryW(w.c_str());
                    continue;
                }
                if (!all && !program_made(name) && !(top && pseudalis_entry(name))) continue;
                if (last) delete_file(p);
                else if (!DeleteFileW(w.c_str()) && GetLastError() == ERROR_ACCESS_DENIED)
                {
                    SetFileAttributesW(w.c_str(), FILE_ATTRIBUTE_NORMAL);
                    DeleteFileW(w.c_str());
                }
            }
        }

        /*
            The removal that the copy of the uninstaller finishes once the uninstaller ends:
            the folder of the programs, the uninstaller in it, whether the folder goes whole,
            and the folders that stay.
        */
        struct removal_plan
        {
            std::string program_dir;
            std::string uninstaller;
            bool all = false;
            std::vector<std::string> keep;
        };
        removal_plan pending_removal;
        bool removal_pending = false;

        // The folder of the system programs, from which the helper processes start.
        std::wstring system_folder()
        {
            wchar_t buf[MAX_PATH];
            const UINT n = GetSystemDirectoryW(buf, MAX_PATH);
            return std::wstring(buf, n < MAX_PATH ? n : 0);
        }

        /*
            Deletes file and then folder once the process that runs file ends: cmd and ping
            of System32, by their full paths and from System32, try once a second for half a
            minute.  cmd expands %name% even between quotes: such paths wait for the restart
            of Windows, for which both are recorded first.
        */
        void delete_after_exit(const std::string& file, const std::string& folder)
        {
            MoveFileExW(widen(file).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
            MoveFileExW(widen(folder).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
            const std::wstring system = system_folder();
            const std::wstring f = widen(file), d = widen(folder);
            if (system.empty() || (f + d).find(L'%') != std::wstring::npos) return;
            const std::wstring cmd_exe = system + L"\\cmd.exe";
            std::wstring cmd = L"\"" + cmd_exe + L"\" /d /s /c \"for /l %i in (1,1,30) do (del /f /q \"" + f
                + L"\" 2>nul & if not exist \"" + f + L"\" (rd \"" + d + L"\" 2>nul & exit) & \"" + system
                + L"\\PING.EXE\" -n 2 127.0.0.1 >nul)\"";
            STARTUPINFOW si = {};
            si.cb = sizeof(si);
            PROCESS_INFORMATION pi = {};
            if (CreateProcessW(cmd_exe.c_str(), &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr,
                    system.c_str(), &si, &pi))
            {
                CloseHandle(pi.hThread);
                CloseHandle(pi.hProcess);
            }
        }
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
        // The first list also names what an earlier installation left in the places of Pseudalis.
        if (!found) before = pseudalis_files(program_dir);
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

    void stage_stale_files(const std::string& program_dir, staged_files& files)
    {
        std::vector<std::wstring> placed;
        for (const auto& f : files.targets()) placed.push_back(windows_path(f));
        // The folders of the user that an earlier version accepted inside these folders keep their files.
        std::vector<std::wstring> kept = { full_folder(data_dir()) };
        for (const char* folder : { "web", "licenses", "winfsp" })
        {
            const std::string top = join_path(program_dir, folder);
            if (is_link(top) || !dir_exists(top)) continue;
            std::vector<std::string> found;
            collect_files(program_dir, folder, found);
            for (const auto& rel : found)
            {
                const std::string p = join_path(program_dir, rel);
                // remove_old_files() deletes the files an update set aside or left unfinished.
                if (is_link(p) || program_made(file_name(p))) continue;
                const std::wstring w = windows_path(p);
                if (std::any_of(placed.begin(), placed.end(), [&](const std::wstring& t) { return same_text(t, w); }))
                    continue;
                const std::wstring full = full_folder(p);
                const auto inside = [&](const std::wstring& k) { return same_or_inside(full, k); };
                if (std::any_of(kept.begin(), kept.end(), inside)) continue;
                files.remove(p);
                log("no longer part of Pseudalis, set aside with the change: " + rel);
            }
        }
    }

    void remove_empty_program_folders(const std::string& program_dir)
    {
        for (const char* folder : { "web", "licenses", "winfsp" })
        {
            const std::string top = join_path(program_dir, folder);
            if (!is_link(top) && dir_exists(top)) remove_empty_folders(top);
        }
    }

    bool remove_installation(const std::string& program_dir, const std::vector<std::string>& keep)
    {
        bool found = false;
        std::vector<std::string> files = listed_files(program_dir, found);
        // An installation older than the list: the files of Pseudalis, known by their places.
        if (!found) files = pseudalis_files(program_dir);
        files.push_back(file_list_name);
        const std::wstring dir = full_folder(program_dir);
        // The folder goes whole, unless it is a drive root or a folder of Windows or of the users.
        bool all = removable_folder(program_dir);
        if (!all) log("the folder " + program_dir + " is a folder of the system: only the files of Pseudalis go");
        std::vector<std::wstring> kept;
        for (const auto& k : keep)
        {
            const std::wstring w = full_folder(k);
            if (k.empty() || !same_or_inside(w, dir)) continue;
            // The folder of the programs kept whole keeps what Pseudalis did not install.
            if (same_text(w, dir)) all = false;
            else kept.push_back(w);
        }
        const std::wstring self = full_folder(executable_path());
        const bool inside = same_or_inside(self, dir);
        // A copy of the uninstaller of the folder finishes the removal once it ends: a file in use
        // goes at the restart of Windows only when nothing comes after this pass.
        const bool last = !inside;
        log("removal of " + program_dir);
        for (const auto& rel : files)
        {
            const std::string p = join_path(program_dir, rel);
            if (same_text(full_folder(p), self)) continue;
            if (is_link(p)) remove_link(p);
            else if (file_exists(p)) delete_file(p);
        }
        clear_folder(program_dir, all, true, kept, self, last);
        const bool removed = RemoveDirectoryW(dir.c_str()) != 0 || !dir_exists(program_dir);
        const bool others = !removed && holds(program_dir, all, kept, true);
        if (others) log("the folder " + program_dir + " stays with the folders kept, or with other files");
        if (!removed && inside)
        {
            pending_removal.program_dir = program_dir;
            pending_removal.uninstaller = narrow(self);
            pending_removal.all = all;
            pending_removal.keep.clear();
            for (const auto& k : kept) pending_removal.keep.push_back(narrow(k));
            removal_pending = true;
        }
        else if (!removed)
        {
            // The last resort: what stays in use goes at the restart of Windows, then the folder if empty.
            MoveFileExW(dir.c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
        }
        return !others;
    }

    void finish_removal()
    {
        if (!removal_pending) return;
        removal_pending = false;
        const removal_plan& plan = pending_removal;
        /*
            The copy of the uninstaller and the removal it finishes, in a folder of its own of
            %TEMP% that only SYSTEM and the administrators change: the copy, started from
            there, loads nothing that another account placed beside it.
        */
        std::wstring temp(MAX_PATH + 1, L'\0');
        const DWORD n = GetTempPathW(static_cast<DWORD>(temp.size()), &temp[0]);
        temp.resize(n > 0 && n < temp.size() ? n : 0);
        const std::string folder = temp.empty() ? std::string() : join_path(narrow(temp), "pseudalis-uninstall-"
            + random_hex(8));
        const std::string copy = join_path(folder, "pseudalis-setup.exe");
        json doc = json::object();
        doc["program_dir"] = plan.program_dir;
        doc["uninstaller"] = plan.uninstaller;
        doc["all"] = plan.all;
        json keep = json::array();
        for (const auto& k : plan.keep) keep.push_back(k);
        doc["keep"] = keep;
        doc["parent"] = static_cast<long long>(GetCurrentProcessId());
        std::string error;
        bool started = false;
        if (!folder.empty() && !dir_exists(folder) && secure_folder(folder, error)
            && CopyFileW(widen(executable_path()).c_str(), widen(copy).c_str(), TRUE)
            && write_file_atomic(join_path(folder, "removal.json"), doc.dump()))
        {
            const std::wstring system = system_folder();
            std::wstring cmd = L"\"" + widen(copy) + L"\" /finish-removal";
            STARTUPINFOW si = {};
            si.cb = sizeof(si);
            PROCESS_INFORMATION pi = {};
            const wchar_t* from = system.empty() ? nullptr : system.c_str();
            started = CreateProcessW(widen(copy).c_str(), &cmd[0], nullptr, nullptr, FALSE,
                    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB, nullptr, from, &si, &pi)
                || CreateProcessW(widen(copy).c_str(), &cmd[0], nullptr, nullptr, FALSE,
                    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, nullptr, from, &si, &pi);
            if (started)
            {
                CloseHandle(pi.hThread);
                CloseHandle(pi.hProcess);
            }
        }
        if (started)
        {
            log("the removal of " + plan.program_dir + " ends after the uninstaller, from " + folder);
            return;
        }
        // The last resort: the files in use, the uninstaller and the folder go at the restart of Windows.
        log("the removal of " + plan.program_dir + " ends at the next restart of Windows" + (error.empty() ? ""
            : ": " + error));
        if (!folder.empty() && dir_exists(folder)) remove_folder(folder);
        std::vector<std::wstring> kept;
        for (const auto& k : plan.keep) kept.push_back(widen(k));
        clear_folder(plan.program_dir, plan.all, true, kept, widen(plan.uninstaller), true);
        MoveFileExW(widen(plan.uninstaller).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
        MoveFileExW(widen(plan.program_dir).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
    }

    int complete_removal()
    {
        // The plan lies beside this copy, in a folder that only SYSTEM and the administrators change.
        const std::string folder = executable_dir();
        const std::string plan_path = join_path(folder, "removal.json");
        std::string text;
        if (!owned_by_system(folder) || !read_file(plan_path, text, 1u << 20)) return 1;
        const json plan = json::parse_or_null(text);
        const std::string program_dir = plan["program_dir"].as_string();
        const std::string uninstaller = plan["uninstaller"].as_string();
        const std::wstring dir = full_folder(program_dir);
        if (program_dir.empty() || !same_or_inside(full_folder(uninstaller), dir) || same_text(full_folder(uninstaller),
                dir))
            return 1;
        {
            // The lines go to the log of the data folder when it is still there.
            std::lock_guard<std::mutex> lock(log_m);
            log_ready = dir_exists(join_path(data_dir(), "logs"));
        }
        const DWORD parent = static_cast<DWORD>(plan["parent"].as_int(0));
        HANDLE p = parent ? OpenProcess(SYNCHRONIZE, FALSE, parent) : nullptr;
        if (p)
        {
            WaitForSingleObject(p, 120000);
            CloseHandle(p);
        }
        std::vector<std::wstring> kept;
        for (const auto& k : plan["keep"].elements()) kept.push_back(full_folder(k.as_string()));
        const bool all = plan["all"].as_bool(false) && removable_folder(program_dir);
        // Once a second for half a minute, while a file stays in use; then the restart of Windows.
        bool removed = false;
        for (int attempt = 0; attempt < 30 && !removed; ++attempt)
        {
            if (attempt > 0) std::this_thread::sleep_for(std::chrono::seconds(1));
            const bool last = attempt == 29;
            if (file_exists(uninstaller))
            {
                if (last) delete_file(uninstaller);
                else DeleteFileW(widen(uninstaller).c_str());
            }
            clear_folder(program_dir, all, true, kept, std::wstring(), last);
            removed = RemoveDirectoryW(dir.c_str()) != 0 || !dir_exists(program_dir);
            // Only what stays for good is left: the folders kept, or the files of a folder of the system.
            if (!removed && !holds(program_dir, all, kept, false)) break;
        }
        if (removed) log("the folder " + program_dir + " is removed");
        else if (holds(program_dir, all, kept, true))
            log("the folder " + program_dir + " stays with the folders kept, or with other files");
        else
        {
            log("the folder " + program_dir + " goes at the next restart of Windows");
            MoveFileExW(dir.c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
        }
        // This copy and its folder go once it ends.
        remove_file(plan_path);
        delete_after_exit(executable_path(), folder);
        return 0;
    }

    void write_result(const std::string& operation, const std::string& previous, bool ok, bool rolled_back,
        const std::string& error)
    {
        const std::string logs = join_path(data_dir(), "logs");
        if (!dir_exists(logs)) return;
        json r = json::object();
        r["time"] = static_cast<long long>(now_unix());
        r["operation"] = operation;
        r["version"] = PSEUDALIS_VERSION;
        r["previous"] = previous;
        r["ok"] = ok;
        if (!ok)
        {
            r["rolled_back"] = rolled_back;
            r["error"] = error;
        }
        write_file_atomic(join_path(logs, "setup-result.json"), r.dump_formatted(2) + "\n");
    }

    void log(const std::string& line)
    {
        std::lock_guard<std::mutex> lock(log_m);
        append_log(iso_time(now_unix()) + " " + line + "\n");
    }

    void flush_log()
    {
        std::lock_guard<std::mutex> lock(log_m);
        if (log_ready || log_waiting.empty()) return;
        wchar_t temp[MAX_PATH + 1];
        const DWORD n = GetTempPathW(MAX_PATH + 1, temp);
        if (n == 0 || n > MAX_PATH) return;
        std::ofstream out(widen(join_path(narrow(std::wstring(temp, n)), "pseudalis-setup.log")).c_str(),
            std::ios::app | std::ios::binary);
        out << log_waiting;
        log_waiting.clear();
    }

    void detach_log()
    {
        std::lock_guard<std::mutex> lock(log_m);
        if (!log_ready) return;
        std::string text;
        read_file(join_path(join_path(data_dir(), "logs"), "setup.log"), text, 16u << 20);
        log_waiting = text + log_waiting;
        log_ready = false;
    }

// ----------------------------------------------------------------------------------------
// The WinFsp of the earlier versions, the control of the service by the users, the commands of
// the Explorer and the letters of the drives.

    namespace
    {
        // The upgrade code that the installers of WinFsp keep from one version to the next.
        const wchar_t* const winfsp_upgrade_code = L"{5466A3D8-3AA1-4240-B6A0-3A051940A3EC}";

        /*
            The product code of the WinFsp its installer installed, empty without one.  msi.dll
            comes from the folder of the system alone, loaded when it serves: the installer
            runs with the rights of an administrator from any folder, a download folder
            included, where another msi.dll could lie.
        */
        std::wstring installed_winfsp_product()
        {
            typedef UINT (WINAPI* enum_related)(LPCWSTR, DWORD, DWORD, LPWSTR);
            const HMODULE msi = LoadLibraryExW(L"msi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (!msi) return std::wstring();
            const enum_related related =
                reinterpret_cast<enum_related>(reinterpret_cast<void*>(GetProcAddress(msi, "MsiEnumRelatedProductsW")));
            wchar_t code[39] = {};
            const bool found = related && related(winfsp_upgrade_code, 0, 0, code) == ERROR_SUCCESS;
            FreeLibrary(msi);
            return found ? std::wstring(code) : std::wstring();
        }

        // The WinFsp that Pseudalis installed: its product code, beside the record of the data folder.
        std::wstring owned_winfsp_product()
        {
            return registry_string(HKEY_LOCAL_MACHINE, pseudalis_key, L"WinFspProduct");
        }

        void forget_owned_winfsp()
        {
            HKEY k = nullptr;
            if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, pseudalis_key, 0, KEY_SET_VALUE | KEY_WOW64_64KEY, &k)
                    != ERROR_SUCCESS)
                return;
            RegDeleteValueW(k, L"WinFspProduct");
            RegCloseKey(k);
        }

        /*
            The file systems that other programs registered with the launcher of WinFsp, in the
            32-bit view of the registry; memfs, the sample its installer registers, does not count.
        */
        std::vector<std::string> other_winfsp_file_systems()
        {
            std::vector<std::string> found;
            HKEY k = nullptr;
            if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"Software\\WinFsp\\Services", 0, KEY_READ | KEY_WOW64_32KEY, &k)
                    != ERROR_SUCCESS)
                return found;
            for (DWORD i = 0;; ++i)
            {
                wchar_t name[256];
                DWORD size = 256;
                if (RegEnumKeyExW(k, i, name, &size, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break;
                const std::wstring n(name, size);
                if (n == L"memfs-a64" || n == L"memfs64" || n == L"memfs32" || n == L"memfs-dotnet") continue;
                found.push_back(narrow(n));
            }
            RegCloseKey(k);
            return found;
        }

        // Shows again among the installed programs a WinFsp that Pseudalis hid there.
        void show_winfsp(const std::wstring& product)
        {
            const std::wstring key = L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\" + product;
            const REGSAM views[] = { KEY_WOW64_32KEY, KEY_WOW64_64KEY };
            for (REGSAM view : views)
            {
                HKEY k = nullptr;
                if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, key.c_str(), 0, KEY_SET_VALUE | view, &k) != ERROR_SUCCESS)
                    continue;
                RegDeleteValueW(k, L"SystemComponent");
                RegCloseKey(k);
            }
        }

        std::string msiexec_program(std::string& error)
        {
            wchar_t system[MAX_PATH];
            const UINT n = GetSystemDirectoryW(system, MAX_PATH);
            if (n == 0 || n >= MAX_PATH)
            {
                error = "the folder of Windows is not found";
                return std::string();
            }
            return join_path(narrow(system), "msiexec.exe");
        }

        /*
            Runs msiexec with args.  Another installation running answers 1618: msiexec runs
            again after a pause, attempts times at most.
        */
        int run_msiexec(const std::string& msiexec, const std::vector<std::string>& args, int attempts, int pause_s)
        {
            int code = -1;
            for (int attempt = 0; attempt < attempts; ++attempt)
            {
                code = run_process(msiexec, args, nullptr, 10 * 60 * 1000);
                if (code != 1618) break;
                std::this_thread::sleep_for(std::chrono::seconds(pause_s));
            }
            return code;
        }
    }

    bool remove_owned_winfsp(std::string& error)
    {
        const std::wstring owned = owned_winfsp_product();
        if (owned.empty()) return true;
        forget_owned_winfsp();
        if (owned != installed_winfsp_product())
        {
            log("WinFsp was replaced since Pseudalis installed it: it stays");
            return true;
        }
        const std::vector<std::string> others = other_winfsp_file_systems();
        if (!others.empty())
        {
            std::string names;
            for (const std::string& n : others) names += (names.empty() ? "" : ", ") + n;
            log("WinFsp stays for the file systems of other programs (" + names + "), among the installed programs");
            show_winfsp(owned);
            return true;
        }
        const std::string msiexec = msiexec_program(error);
        if (msiexec.empty())
        {
            show_winfsp(owned);
            return false;
        }
        // The log of msiexec lies in the temporary folder: the data folder may go.
        wchar_t temp[MAX_PATH + 1];
        const DWORD n = GetTempPathW(MAX_PATH + 1, temp);
        const std::string uninstall_log = n == 0 || n > MAX_PATH ? std::string()
            : join_path(narrow(std::wstring(temp, n)), "pseudalis-winfsp-uninstall.log");
        std::vector<std::string> args = { "/x", narrow(owned), "/qn", "/norestart" };
        if (!uninstall_log.empty()) args.insert(args.end(), { "/l*v", uninstall_log });
        log("uninstallation of WinFsp, installed with an earlier version of Pseudalis");
        const int code = run_msiexec(msiexec, args, 4, 15);
        // 1605: the product is no longer installed.
        if (code == 0 || code == 3010 || code == 1605)
        {
            log(std::string("WinFsp is uninstalled") + (code == 3010 ? ", Windows must restart to finish" : ""));
            return true;
        }
        show_winfsp(owned);
        error = "msiexec ended with code " + std::to_string(code);
        log("WinFsp cannot be uninstalled (" + error + "): it stays, among the installed programs");
        return false;
    }

    bool allow_service_control(std::string& error)
    {
        /*
            SYSTEM and the administrators as by default; the interactive users may also start
            and stop the service, which the icon does when the service does not answer.
        */
        const wchar_t* const sddl = L"D:(A;;CCLCSWRPWPDTLOCRRC;;;SY)(A;;CCDCLCSWRPWPDTLOCRSDRCWDWO;;;BA)"
            L"(A;;CCLCSWRPWPLOCRRC;;;IU)(A;;CCLCSWLOCRRC;;;SU)";
        PSECURITY_DESCRIPTOR sd = nullptr;
        if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, SDDL_REVISION_1, &sd, nullptr))
        {
            error = last_error("cannot build the permissions of the service");
            return false;
        }
        service_handles h;
        const bool ok = open_service(h, READ_CONTROL | WRITE_DAC) &&
            SetServiceObjectSecurity(h.svc, DACL_SECURITY_INFORMATION, sd);
        if (!ok) error = last_error("cannot set the permissions of the service");
        LocalFree(sd);
        return ok;
    }

    namespace
    {
        // The extensions that receive the commands of the Explorer: ".tar.gz" is ".gz" for Windows.
        std::vector<std::wstring> menu_extensions()
        {
            std::vector<std::wstring> out;
            for (const std::string& e : supported_extensions())
            {
                const std::wstring w = widen(e == ".tar.gz" ? std::string(".gz") : e);
                if (std::find(out.begin(), out.end(), w) == out.end()) out.push_back(w);
            }
            return out;
        }

        std::wstring menu_key(const std::wstring& extension)
        {
            return L"Software\\Classes\\SystemFileAssociations\\" + extension + L"\\shell\\Pseudalis";
        }

        bool set_string(HKEY k, const wchar_t* name, const std::wstring& value)
        {
            return RegSetValueExW(k, name, 0, REG_SZ, reinterpret_cast<const BYTE*>(value.c_str()),
                static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t))) == ERROR_SUCCESS;
        }

        void notify_associations()
        {
            SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
        }
    }

    bool context_menu_registered()
    {
        HKEY k = nullptr;
        const LSTATUS r = RegOpenKeyExW(HKEY_LOCAL_MACHINE, menu_key(L".pdf").c_str(), 0, KEY_READ | KEY_WOW64_64KEY,
            &k);
        if (r != ERROR_SUCCESS) return false;
        RegCloseKey(k);
        return true;
    }

    bool register_context_menu(const std::string& program_dir, const std::map<std::string, std::wstring>& labels,
                               bool each_user)
    {
        const std::wstring program = widen(join_path(program_dir, "pseudalis.exe"));
        // The text of a command: an indirect string, which the Explorer loads in the language of
        // its user from the resources of pseudalis.exe, or the text in the language of the
        // installation.
        auto label = [&](const char* key, const wchar_t* fallback, int resource) {
            if (each_user && resource > 0) return L"@" + program + L",-" + std::to_wstring(resource);
            const auto it = labels.find(key);
            return it == labels.end() || it->second.empty() ? std::wstring(fallback) : it->second;
        };
        const std::wstring icon = program + L",0";
        struct verb
        {
            const wchar_t* key;
            std::wstring text;
            const wchar_t* arguments;
            DWORD flags;
        };
        // ECF_SEPARATORBEFORE (0x40) sets the restorations apart from the protections.
        const verb verbs[] = {
            { L"1protect", label("protect", L"Anonymize", IDS_VERB_PROTECT), L"protect", 0 },
            { L"2protectcopy", label("protect_copy", L"Anonymize as a copy", IDS_VERB_PROTECT_COPY), L"protect --copy",
              0 },
            { L"3restore", label("restore", L"Restore the original", IDS_VERB_RESTORE), L"restore", 0x40 },
            { L"4restorecopy", label("restore_copy", L"Restore as a copy", IDS_VERB_RESTORE_COPY), L"restore --copy",
              0 },
        };
        bool ok = true;
        for (const std::wstring& extension : menu_extensions())
        {
            const std::wstring root = menu_key(extension);
            RegDeleteTreeW(HKEY_LOCAL_MACHINE, root.c_str());
            HKEY k = nullptr;
            if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, root.c_str(), 0, nullptr, 0, KEY_WRITE | KEY_WOW64_64KEY, nullptr,
                    &k, nullptr) != ERROR_SUCCESS)
            {
                ok = false;
                continue;
            }
            // A cascade whose commands are the subkeys of its own shell key.
            set_string(k, L"MUIVerb", label("menu", L"Pseudalis", 0));
            set_string(k, L"Icon", icon);
            set_string(k, L"SubCommands", L"");
            RegCloseKey(k);
            for (const verb& v : verbs)
            {
                const std::wstring key = root + L"\\shell\\" + v.key;
                HKEY vk = nullptr;
                if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, key.c_str(), 0, nullptr, 0, KEY_WRITE | KEY_WOW64_64KEY,
                        nullptr, &vk, nullptr) != ERROR_SUCCESS)
                {
                    ok = false;
                    continue;
                }
                set_string(vk, L"MUIVerb", v.text);
                if (v.flags) RegSetValueExW(vk, L"CommandFlags", 0, REG_DWORD, reinterpret_cast<const BYTE*>(&v.flags),
                    sizeof(v.flags));
                RegCloseKey(vk);
                HKEY ck = nullptr;
                if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, (key + L"\\command").c_str(), 0, nullptr, 0,
                        KEY_WRITE | KEY_WOW64_64KEY, nullptr, &ck, nullptr) != ERROR_SUCCESS)
                {
                    ok = false;
                    continue;
                }
                set_string(ck, nullptr, L"\"" + program + L"\" " + v.arguments + L" \"%1\"");
                RegCloseKey(ck);
            }
        }
        notify_associations();
        if (!ok) log("the commands of the Explorer are not all registered");
        return ok;
    }

    void remove_context_menu()
    {
        for (const std::wstring& extension : menu_extensions())
            RegDeleteTreeW(HKEY_LOCAL_MACHINE, menu_key(extension).c_str());
        notify_associations();
    }

    namespace
    {
        bool letter_syntax(const std::string& letter)
        {
            if (letter.size() != 1) return false;
            const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(letter[0])));
            return c >= 'A' && c <= 'Z' && c != 'C';
        }

        /*
            The letters in use: the drives of the machine, and the network drives the user of
            the installer keeps (HKCU\Network), which a drive of the machine would hide.
        */
        std::set<char> letters_in_use()
        {
            std::set<char> used;
            const DWORD mask = GetLogicalDrives();
            for (int i = 0; i < 26; ++i)
                if (mask & (1u << i)) used.insert(static_cast<char>('A' + i));
            HKEY k = nullptr;
            if (RegOpenKeyExW(HKEY_CURRENT_USER, L"Network", 0, KEY_READ, &k) == ERROR_SUCCESS)
            {
                for (DWORD i = 0;; ++i)
                {
                    wchar_t name[256];
                    DWORD size = 256;
                    if (RegEnumKeyExW(k, i, name, &size, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break;
                    if (size == 1 && std::iswalpha(name[0])) used.insert(static_cast<char>(std::towupper(name[0])));
                }
                RegCloseKey(k);
            }
            return used;
        }
    }

    std::vector<std::string> free_drive_letters(const std::string& current, const std::string& also)
    {
        const std::set<char> used = letters_in_use();
        auto upper = [](const std::string& l) {
            return letter_syntax(l) ? static_cast<char>(std::toupper(static_cast<unsigned char>(l[0]))) : '\0';
        };
        const char mine = upper(current), other = upper(also);
        std::vector<std::string> out;
        for (char c = 'A'; c <= 'Z'; ++c)
        {
            if (c == 'C') continue;
            if (c == mine || c == other || !used.count(c)) out.push_back(std::string(1, c));
        }
        return out;
    }

    bool valid_drive_choice(const std::string& letter)
    {
        if (!letter_syntax(letter)) return false;
        const install_options o = installed_options();
        const std::vector<std::string> free = free_drive_letters(o.drive_letter, o.restore_letter);
        const std::string up(1, static_cast<char>(std::toupper(static_cast<unsigned char>(letter[0]))));
        return std::find(free.begin(), free.end(), up) != free.end();
    }

    std::string default_restore_letter(const std::string& drive_letter)
    {
        const std::vector<std::string> free = free_drive_letters(std::string());
        for (char c = 'R'; c <= 'Z'; ++c)
        {
            const std::string l(1, c);
            if (l != drive_letter && std::find(free.begin(), free.end(), l) != free.end()) return l;
        }
        for (const std::string& l : free)
            if (l != drive_letter && l != "A" && l != "B") return l;
        return std::string();
    }

}
}
