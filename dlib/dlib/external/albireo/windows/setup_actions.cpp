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
#include <aclapi.h>
#include <exdisp.h>
#include <sddl.h>
#include <shellapi.h>
#include <shldisp.h>
#include <shlguid.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <wincrypt.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <thread>
#include <vector>

#include "../common/json.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "../host/certificates.h"
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

        bool apply_dacl(const std::string& path, const wchar_t* sddl)
        {
            PSECURITY_DESCRIPTOR sd = nullptr;
            if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, SDDL_REVISION_1, &sd, nullptr))
                return false;
            BOOL present = FALSE, defaulted = FALSE;
            PACL dacl = nullptr;
            GetSecurityDescriptorDacl(sd, &present, &dacl, &defaulted);
            std::wstring w = widen(path);
            const DWORD rc = SetNamedSecurityInfoW(&w[0], SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, nullptr, nullptr, dacl, nullptr);
            LocalFree(sd);
            return rc == ERROR_SUCCESS;
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
    }

    std::string default_program_dir()
    {
        return join_path(known_folder(FOLDERID_ProgramFiles, "C:\\Program Files"), "Albireo");
    }

    std::string default_models_dir()
    {
        return join_path(data_dir(), "models");
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
        // Restarted after a failure, twice, then left stopped; the counter resets daily.
        SC_ACTION actions[3] = { { SC_ACTION_RESTART, 10000 }, { SC_ACTION_RESTART, 60000 }, { SC_ACTION_NONE, 0 } };
        SERVICE_FAILURE_ACTIONSW fa = {};
        fa.dwResetPeriod = 86400;
        fa.cActions = 3;
        fa.lpsaActions = actions;
        ChangeServiceConfig2W(svc, SERVICE_CONFIG_FAILURE_ACTIONS, &fa);
        SERVICE_FAILURE_ACTIONS_FLAG flag = { TRUE };
        ChangeServiceConfig2W(svc, SERVICE_CONFIG_FAILURE_ACTIONS_FLAG, &flag);
        SERVICE_PRESHUTDOWN_INFO pre = { 20000 };
        ChangeServiceConfig2W(svc, SERVICE_CONFIG_PRESHUTDOWN_INFO, &pre);
        CloseServiceHandle(svc);
        CloseServiceHandle(scm);
        return true;
    }

    bool remove_service(std::string& error)
    {
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

    void close_trays(unsigned timeout_ms)
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
    }

    bool write_program_file(const std::string& path, const std::function<bool(const chunk_writer&)>& body,
        bool& reboot_needed, std::string& error)
    {
        make_dirs(parent_path(path));
        const std::wstring target = widen(path);
        const std::wstring tmp = target + L".new";
        HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE)
        {
            error = last_error(("cannot create " + path).c_str());
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
        if (!ok)
        {
            DeleteFileW(tmp.c_str());
            if (error.empty()) error = "cannot write " + path;
            return false;
        }
        if (MoveFileExW(tmp.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return true;
        // In use: the running file is renamed aside, which Windows allows, and deleted later.
        const std::wstring aside = target + L".old-" + widen(random_hex(8));
        if (MoveFileExW(target.c_str(), aside.c_str(), 0)
            && MoveFileExW(tmp.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH))
        {
            MoveFileExW(aside.c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
            return true;
        }
        if (MoveFileExW(tmp.c_str(), target.c_str(), MOVEFILE_DELAY_UNTIL_REBOOT | MOVEFILE_REPLACE_EXISTING))
        {
            reboot_needed = true;
            return true;
        }
        error = last_error(("cannot replace " + path).c_str());
        return false;
    }

    void remove_old_files(const std::string& program_dir)
    {
        for (const auto& name : list_dir(program_dir))
        {
            const std::string p = join_path(program_dir, name);
            const bool aside = name.find(".old-") != std::string::npos;
            const bool unfinished = name.size() > 4 && name.substr(name.size() - 4) == ".new";
            if (dir_exists(p)) remove_old_files(p);
            else if (aside || unfinished) remove_file(p);
        }
    }

    bool install_uninstaller(const std::string& program_dir, std::string& error)
    {
        const std::string self = executable_path();
        const std::string target = join_path(program_dir, "albireo-setup.exe");
        if (_wcsicmp(widen(self).c_str(), widen(target).c_str()) == 0) return true;
        const std::wstring tmp = widen(target) + L".tmp";
        if (!CopyFileW(widen(self).c_str(), tmp.c_str(), FALSE))
        {
            error = last_error("cannot copy the uninstaller");
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
        bool reboot = false;
        std::string data;
        const bool ok = write_program_file(target, [&](const std::function<bool(const void*, size_t)>& write) {
            std::string content;
            if (!read_file(narrow(tmp), content, 1ull << 34)) return false;
            return write(content.data(), content.size());
        }, reboot, error);
        DeleteFileW(tmp.c_str());
        return ok;
    }

    bool prepare_data_dir(std::string& error)
    {
        const std::string dir = data_dir();
        if (!make_dirs(dir))
        {
            error = "cannot create " + dir;
            return false;
        }
        make_dirs(join_path(dir, "logs"));
        make_dirs(join_path(dir, "import"));
        if (!apply_dacl(dir, L"D:PAI(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;0x1200a9;;;BU)"))
            log("the permissions of the data folder could not be set");
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
        if (!o.models_dir.empty()) make_dirs(o.models_dir);
        if (!write_file_atomic(path, s.dump_formatted(2) + "\n"))
        {
            error = "cannot write " + path;
            return false;
        }
        return true;
    }

    int stage_bundled_models(const std::function<void(const std::string&)>& progress)
    {
        const std::string source = join_path(executable_dir(), "models");
        if (!dir_exists(source)) return 0;
        const std::string inbox = join_path(data_dir(), "import");
        make_dirs(inbox);
        int n = 0;
        for (const auto& name : list_dir(source))
        {
            const bool gguf = name.size() > 5 && name.substr(name.size() - 5) == ".gguf";
            const bool side = name.size() > 5 && name.substr(name.size() - 5) == ".json";
            if (!gguf && !side) continue;
            if (progress && gguf) progress(name);
            const bool copied = CopyFileW(widen(join_path(source, name)).c_str(),
                widen(join_path(inbox, name)).c_str(), FALSE) != FALSE;
            if (copied && gguf) ++n;
        }
        return n;
    }

    bool create_certificates(std::string& error)
    {
        return ensure_certificates(join_path(data_dir(), "certificates"), error);
    }

    bool trust_authority(std::string& error)
    {
        const std::string der = certificate_der(certificate_paths(join_path(data_dir(), "certificates")).ca_cert);
        if (der.empty())
        {
            error = "the certificate of the local authority cannot be read";
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
        str(L"DisplayName", L"Albireo");
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
        if (!ok)
        {
            // Without a desktop shell (a remote session, a test machine) the program
            // starts from here.
            ok = reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", widen(program).c_str(), widen(args).c_str(),
                nullptr, SW_SHOWNORMAL)) > 32;
        }
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

    void remove_folder(const std::string& path)
    {
        if (!dir_exists(path)) return;
        for (const auto& name : list_dir(path))
        {
            const std::string p = join_path(path, name);
            if (dir_exists(p)) remove_folder(p);
            else if (!DeleteFileW(widen(p).c_str()))
                MoveFileExW(widen(p).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
        }
        if (!RemoveDirectoryW(widen(path).c_str()))
            MoveFileExW(widen(path).c_str(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);
    }

    void log(const std::string& line)
    {
        const std::string dir = join_path(data_dir(), "logs");
        make_dirs(dir);
        std::ofstream out(widen(join_path(dir, "setup.log")).c_str(), std::ios::app | std::ios::binary);
        out << iso_time(now_unix()) << " " << line << "\n";
    }

}
}
