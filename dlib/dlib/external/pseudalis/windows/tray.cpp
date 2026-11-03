// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    pseudalis-tray: the icon of Pseudalis in the notification area of each user.

        pseudalis-tray                  shows the icon (started at sign in)
        pseudalis-tray --open           opens the web application, starting the icon if needed
        pseudalis-tray --about          opens the About window, starting the icon if needed
        pseudalis-tray --setup-result   started again by the installer after an installation
                                        made without its window: notifies its failure

    The icon follows the service: its state (running, stopped, update ready), its tooltip
    (the drives, the originals kept), the files the drives protect, restore or refuse, the
    results of the commands of the Explorer (pseudalis.exe sends them here, WM_COPYDATA),
    and the updates.  It tells the Explorer of the drives that come and go.  Its menu opens
    the web application, the drives and the folder of the files the drive did not process,
    checks for updates, restarts the service, opens the documentation of the interface and
    the About window.

    While the service does not answer, the icon asks the service manager of Windows what
    the service does, and warns only when the service stays stopped, or stays silent while
    it runs, beyond the time a start or a restart takes.
*/

#include "ui.h"

#include <objbase.h>
#include <sddl.h>
#include <shellapi.h>
#include <shlobj.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <set>
#include <string>
#include <vector>

#include "../common/platform.h"
#include "../common/version.h"
#include "resource.h"
#include "service_client.h"
#include "strings.h"

namespace pseudalis
{

    namespace
    {
        const wchar_t window_class[] = L"PseudalisTrayWindow";
        const wchar_t about_class[] = L"PseudalisAboutWindow";
        const UINT WM_TRAY = WM_APP + 1;
        const UINT WM_STATUS = WM_APP + 2;
        const UINT WM_ACTION_DONE = WM_APP + 3;
        const UINT WM_OPEN_APP = WM_APP + 4;
        const UINT WM_SHOW_ABOUT = WM_APP + 5;
        const UINT_PTR TIMER_POLL = 1;
        const UINT_PTR TIMER_RESULTS = 2;
        // The mark of the results that pseudalis.exe sends (WM_COPYDATA).
        const ULONG_PTR results_mark = 0x50534431;

        enum command_id
        {
            CMD_OPEN = 100, CMD_DRIVE, CMD_CHECK_UPDATES, CMD_INSTALL_UPDATE, CMD_RESTART, CMD_START_SERVICE, CMD_DOCS,
            CMD_ABOUT, CMD_QUIT, CMD_RESTORE_DRIVE, CMD_REFUSED
        };

        enum action_tag { ACT_CHECK = 1, ACT_APPLY_UPDATE, ACT_RESTART, ACT_SCM };

        struct shell_result
        {
            bool ok = false;
            std::wstring text;
            std::string name, error;
        };

        struct tray_state
        {
            HWND hwnd = nullptr;
            HWND about = nullptr;
            UINT taskbar_created = 0;
            NOTIFYICONDATAW nid = {};
            service_client client;
            json info;
            json updates;
            bool online = false;
            // The service answered since the icon started.
            bool seen_online = false;
            // When the service stopped answering (GetTickCount64), and whether the icon warned of it.
            ULONGLONG silent_since = 0;
            bool warned = false;
            // The web application opens once the service answers.
            bool open_pending = false;
            bool polling = false;
            int icon_state = -1;
            std::string notified_update;
            bool check_requested = false;
            bool relaunch_on_restart = false;
            ULONGLONG version_checked = 0;
            UINT poll_ms = 2000;
            /*
                The last event of the drive seen, -1 before the first answer.  The events in place
                when the icon starts, or when the service answers again, are read without being
                notified, a hundred at a time, until a batch holds fewer.
            */
            long long last_event = -1;
            bool catching_up = true;
            // The errors of the drives notified last, and the letters the Explorer was told of.
            std::string drive_error, restore_error;
            std::set<std::string> letters_shown;
            // The results of the commands of the Explorer, notified together.
            std::vector<shell_result> results;
            // When the icon started (seconds since 1970), and the result of the installer read last.
            int64_t started = 0;
            int64_t setup_result_time = 0;
            ULONGLONG setup_result_checked = 0;
            bool report_setup_result = false;
        } tray;

        /*
            The state of the service for the service manager: "stopped", "starting",
            "running", "stopping", "other" or "missing", and whether it starts with the system.
        */
        json service_manager_state()
        {
            json out = json::object();
            out["state"] = "missing";
            SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT);
            if (!scm) return out;
            SC_HANDLE svc = OpenServiceW(scm, L"Pseudalis", SERVICE_QUERY_STATUS | SERVICE_QUERY_CONFIG);
            if (svc)
            {
                SERVICE_STATUS_PROCESS sp = {};
                DWORD needed = 0;
                if (QueryServiceStatusEx(svc, SC_STATUS_PROCESS_INFO, reinterpret_cast<LPBYTE>(&sp), sizeof(sp),
                        &needed))
                {
                    const DWORD s = sp.dwCurrentState;
                    out["state"] = s == SERVICE_STOPPED ? "stopped" : s == SERVICE_START_PENDING ? "starting"
                        : s == SERVICE_RUNNING ? "running" : s == SERVICE_STOP_PENDING ? "stopping" : "other";
                }
                DWORD size = 0;
                QueryServiceConfigW(svc, nullptr, 0, &size);
                if (size > 0)
                {
                    std::vector<unsigned char> buf(size);
                    QUERY_SERVICE_CONFIGW* config = reinterpret_cast<QUERY_SERVICE_CONFIGW*>(buf.data());
                    if (QueryServiceConfigW(svc, config, size, &size))
                        out["auto"] = config->dwStartType == SERVICE_AUTO_START;
                }
                CloseServiceHandle(svc);
            }
            CloseServiceHandle(scm);
            return out;
        }

        /*
            Starts the service, after stopping it when restart is true, through the service
            manager: the installer lets the interactive users do it (allow_service_control()).
            Without that right, the installer does it, with the consent of an administrator.
        */
        json control_service(bool restart)
        {
            json out = json::object();
            SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT);
            SC_HANDLE svc = scm ? OpenServiceW(scm, L"Pseudalis", SERVICE_START | SERVICE_STOP | SERVICE_QUERY_STATUS)
                : nullptr;
            bool ok = svc != nullptr;
            if (ok && restart)
            {
                SERVICE_STATUS st = {};
                ControlService(svc, SERVICE_CONTROL_STOP, &st);
                for (int i = 0; i < 300; ++i)
                {
                    if (!QueryServiceStatus(svc, &st) || st.dwCurrentState == SERVICE_STOPPED) break;
                    Sleep(100);
                }
            }
            if (ok) ok = StartServiceW(svc, 0, nullptr) || GetLastError() == ERROR_SERVICE_ALREADY_RUNNING;
            if (svc) CloseServiceHandle(svc);
            if (scm) CloseServiceHandle(scm);
            if (!ok)
            {
                // The consent of an administrator, through the installer, which needs COM on this thread.
                CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
                const std::wstring setup = widen(join_path(executable_dir(), "pseudalis-setup.exe"));
                ShellExecuteW(nullptr, L"runas", setup.c_str(), restart ? L"/service restart" : L"/service start",
                    nullptr, SW_HIDE);
                CoUninitialize();
            }
            return out;
        }

        HICON load_icon(int id, int size)
        {
            return static_cast<HICON>(LoadImageW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(id), IMAGE_ICON,
                size, size, LR_DEFAULTCOLOR));
        }

        // 0: the service answers, 1: it does not, 2: an update is ready.
        void set_icon(int state)
        {
            if (state == tray.icon_state) return;
            tray.icon_state = state;
            static const int ids[] = { IDI_TRAY, IDI_TRAY_OFF, IDI_TRAY_UPDATE };
            HICON old = tray.nid.hIcon;
            tray.nid.hIcon = load_icon(ids[state], GetSystemMetrics(SM_CXSMICON));
            tray.nid.uFlags = NIF_ICON;
            Shell_NotifyIconW(NIM_MODIFY, &tray.nid);
            if (old) DestroyIcon(old);
        }

        void set_tooltip(const std::wstring& text)
        {
            const std::wstring t = text.substr(0, 127);
            if (t == tray.nid.szTip) return;
            wcsncpy(tray.nid.szTip, t.c_str(), 127);
            tray.nid.szTip[127] = 0;
            tray.nid.uFlags = NIF_TIP | NIF_SHOWTIP;
            Shell_NotifyIconW(NIM_MODIFY, &tray.nid);
        }

        void notify(const std::wstring& title, const std::wstring& text, bool warning = false)
        {
            NOTIFYICONDATAW n = tray.nid;
            n.uFlags = NIF_INFO;
            wcsncpy(n.szInfoTitle, title.substr(0, 63).c_str(), 63);
            n.szInfoTitle[63] = 0;
            wcsncpy(n.szInfo, text.substr(0, 255).c_str(), 255);
            n.szInfo[255] = 0;
            n.dwInfoFlags = (warning ? NIIF_WARNING : NIIF_INFO) | NIIF_RESPECT_QUIET_TIME;
            Shell_NotifyIconW(NIM_MODIFY, &n);
        }

        void add_icon()
        {
            tray.nid = NOTIFYICONDATAW();
            tray.nid.cbSize = sizeof(tray.nid);
            tray.nid.hWnd = tray.hwnd;
            tray.nid.uID = 1;
            tray.nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_SHOWTIP;
            tray.nid.uCallbackMessage = WM_TRAY;
            tray.nid.hIcon = load_icon(IDI_TRAY_OFF, GetSystemMetrics(SM_CXSMICON));
            wcscpy(tray.nid.szTip, L"Pseudalis");
            Shell_NotifyIconW(NIM_ADD, &tray.nid);
            tray.nid.uVersion = NOTIFYICON_VERSION_4;
            Shell_NotifyIconW(NIM_SETVERSION, &tray.nid);
            tray.icon_state = 1;
        }

        // The error of the updates: the text of its code, else the English text of the service.
        std::wstring update_error_text(const json& up)
        {
            const std::string code = up["code"].as_string();
            if (!code.empty() && has_text("err_" + code)) return error_text(up);
            return tr("notify_update_error", { { "e", up["error"].as_string() } });
        }

        /*
            The result of the last run of the installer, which it writes for the updates it
            makes out of sight (logs\setup-result.json in the data folder): an update that
            failed after the start of the icon is notified, with the version put back.
        */
        void check_setup_result()
        {
            const ULONGLONG now = GetTickCount64();
            if (tray.setup_result_checked && now - tray.setup_result_checked < 10000) return;
            tray.setup_result_checked = now;
            std::string text;
            const std::string path = join_path(join_path(tray.client.data_dir(), "logs"), "setup-result.json");
            if (!read_file(path, text, 1u << 16)) return;
            const json r = json::parse_or_null(text);
            const int64_t time = r["time"].as_int(0);
            if (time <= tray.setup_result_time) return;
            tray.setup_result_time = time;
            if ((time < tray.started && !tray.report_setup_result) || r["ok"].as_bool(true)) return;
            const char* key = r["rolled_back"].as_bool(false) ? "notify_update_rolled_back" : "notify_update_failed";
            notify(tr("notify_update_failed_title"), tr(key, { { "v", r["version"].as_string() },
                { "o", r["previous"].as_string() }, { "e", r["error"].as_string() } }), true);
        }

        void poll()
        {
            if (tray.polling) return;
            tray.polling = true;
            service_client c = tray.client;
            const long long since = tray.last_event;
            // The polls share a connection kept open between them (service_client::poll()).
            ui::run_async(tray.hwnd, WM_STATUS, 0, [c, since]() {
                json r = json::object();
                std::string error;
                json info, updates, events;
                if (!c.poll("/v1/info", info, error, 5))
                {
                    r["error"] = error;
                    r["service"] = service_manager_state();
                    return r;
                }
                r["info"] = info;
                if (c.poll("/v1/updates", updates, error, 5)) r["updates"] = updates;
                if (c.poll("/v1/events?since=" + std::to_string(since < 0 ? 0 : since), events, error, 5))
                    r["events"] = events["events"];
                return r;
            });
        }

        void set_poll_period(UINT ms)
        {
            if (ms == tray.poll_ms) return;
            tray.poll_ms = ms;
            SetTimer(tray.hwnd, TIMER_POLL, ms, nullptr);
        }

        /*
            Whether the program file of the icon is newer than the icon that runs, read once
            a minute at most: an icon whose file an update did not replace keeps running.
        */
        bool program_replaced()
        {
            const ULONGLONG now = GetTickCount64();
            if (tray.version_checked && now - tray.version_checked < 60000) return false;
            tray.version_checked = now;
            const DWORD flags = LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE;
            HMODULE m = LoadLibraryExW(widen(executable_path()).c_str(), nullptr, flags);
            if (!m) return false;
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
                version = std::to_string(HIWORD(info.dwFileVersionMS)) + "."
                    + std::to_string(LOWORD(info.dwFileVersionMS)) + "." + std::to_string(HIWORD(info.dwFileVersionLS));
            }
            FreeLibrary(m);
            return compare_versions(version, PSEUDALIS_VERSION) > 0;
        }

        void relaunch(const wchar_t* args)
        {
            const std::wstring exe = widen(executable_path());
            std::wstring cmd = L"\"" + exe + L"\" " + args;
            STARTUPINFOW si = { sizeof(si) };
            PROCESS_INFORMATION pi = {};
            if (CreateProcessW(exe.c_str(), &cmd[0], nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi))
            {
                CloseHandle(pi.hThread);
                CloseHandle(pi.hProcess);
            }
        }

        /*
            The address of a page of the service, with the administration token in its fragment,
            which the browser never sends: the web application and the documentation of the
            interface read it there, then take it out of the address.
        */
        std::wstring page_url(const std::string& page)
        {
            const std::string& token = tray.client.token();
            return widen(tray.client.base_url() + page + (token.empty() ? std::string() : "#admin=" + token));
        }

        void open_app()
        {
            ui::open_url(page_url(""));
        }

        // The letter of the drive of anonymization, or of restoration, while it is given.
        std::string drive_letter(bool restoration = false)
        {
            const json& d = restoration ? tray.info["drive"]["restore"] : tray.info["drive"];
            return d["mounted"].as_bool(false) ? d["letter"].as_string() : std::string();
        }

        // The folder of the files of this user that the drive did not process, out of the drive.
        std::string refused_folder()
        {
            const std::string root = tray.info["drive"]["refused_folder"].as_string();
            HANDLE token = nullptr;
            if (root.empty() || !OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return std::string();
            std::vector<BYTE> buffer(256);
            DWORD size = 0;
            std::string sid;
            if (GetTokenInformation(token, TokenUser, buffer.data(), static_cast<DWORD>(buffer.size()), &size) ||
                (GetLastError() == ERROR_INSUFFICIENT_BUFFER && (buffer.resize(size), true) &&
                 GetTokenInformation(token, TokenUser, buffer.data(), static_cast<DWORD>(buffer.size()), &size)))
            {
                LPWSTR text = nullptr;
                if (ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(buffer.data())->User.Sid, &text))
                {
                    sid = narrow(text);
                    LocalFree(text);
                }
            }
            CloseHandle(token);
            const std::string folder = sid.empty() ? std::string() : join_path(root, sid);
            return !folder.empty() && dir_exists(folder) ? folder : std::string();
        }

        // Tells the Explorer of the drives that came or went, which it shows at once.
        void show_letters()
        {
            std::set<std::string> now;
            for (bool r : { false, true })
            {
                const std::string l = drive_letter(r);
                if (!l.empty()) now.insert(l);
            }
            for (const std::string& l : now)
                if (!tray.letters_shown.count(l))
                    SHChangeNotify(SHCNE_DRIVEADD, SHCNF_PATHW, widen(l + ":\\").c_str(), nullptr);
            for (const std::string& l : tray.letters_shown)
                if (!now.count(l)) SHChangeNotify(SHCNE_DRIVEREMOVED, SHCNF_PATHW, widen(l + ":\\").c_str(), nullptr);
            tray.letters_shown = now;
        }

        // The name of a file of the drive from its path in the drive: "\folder\name" gives "name".
        std::string event_name(const json& e)
        {
            const std::string path = e["path"].as_string();
            const size_t slash = path.find_last_of("\\/");
            return slash == std::string::npos ? path : path.substr(slash + 1);
        }

        // The files the drives processed or refused since the last poll, notified together.
        void notify_events(const json& events)
        {
            if (!events.is_array()) return;
            long long last = tray.last_event;
            std::vector<const json*> protected_files, refused, restored, not_restored;
            for (const auto& e : events.elements())
            {
                last = std::max(last, e["id"].as_int(0));
                const std::string kind = e["kind"].as_string();
                if (kind == "protected") protected_files.push_back(&e);
                else if (kind == "refused") refused.push_back(&e);
                else if (kind == "restored") restored.push_back(&e);
                else if (kind == "not_restored") not_restored.push_back(&e);
            }
            tray.last_event = std::max<long long>(last, 0);
            if (tray.catching_up)
            {
                tray.catching_up = events.size() >= 100;
                return;
            }
            // The drive of restoration: what it gave back, or could not.
            const std::wstring restore_title =
                tr("notify_restore_title", { { "l", tray.info["drive"]["restore"]["letter"].as_string() } });
            if (!not_restored.empty())
            {
                const json& e = *not_restored.front();
                notify(restore_title, not_restored.size() == 1
                    ? tr("notify_not_restored", { { "f", event_name(e) }, { "e", e["message"].as_string() } })
                    : tr("notify_not_restored_many", { { "n", std::to_string(not_restored.size()) },
                        { "f", event_name(e) }, { "e", e["message"].as_string() } }), true);
            }
            else if (restored.size() == 1)
                notify(restore_title, tr("notify_restored_one", { { "f", event_name(*restored.front()) } }));
            else if (restored.size() > 1)
                notify(restore_title, tr("notify_restored_many", { { "n", std::to_string(restored.size()) } }));
            const std::string letter = tray.info["drive"]["letter"].as_string();
            const std::wstring title = tr("notify_drive_title", { { "l", letter } });
            if (!refused.empty())
            {
                const json& e = *refused.front();
                if (refused.size() == 1)
                    notify(title, tr("notify_refused", { { "f", event_name(e) }, { "e", e["message"].as_string() } }),
                        true);
                else
                    notify(title, tr("notify_refused_many", { { "n", std::to_string(refused.size()) },
                        { "f", event_name(e) }, { "e", e["message"].as_string() } }), true);
                return;
            }
            if (protected_files.size() == 1)
            {
                // "12 values, 0 entries left out": the number of values comes first.
                const std::string message = protected_files.front()->operator[]("message").as_string();
                const std::string values = message.substr(0, message.find(' '));
                notify(title, tr("notify_protected_one", { { "f", event_name(*protected_files.front()) },
                    { "n", values } }));
            }
            else if (protected_files.size() > 1)
            {
                notify(title, tr("notify_protected_many", { { "n", std::to_string(protected_files.size()) } }));
            }
        }

        void on_status(json* result)
        {
            tray.polling = false;
            const bool online = !result->contains("error");
            if (!online)
            {
                tray.online = false;
                const ULONGLONG now = GetTickCount64();
                if (!tray.silent_since) tray.silent_since = now;
                const ULONGLONG silent = now - tray.silent_since;
                const json& scm = (*result)["service"];
                const std::string st = scm["state"].as_string();
                const ULONGLONG minute = 60000;
                // How long each state may last before the icon warns.
                bool starting = false;
                ULONGLONG grace = minute;
                const char* warning = "notify_service_down";
                const char* tip = "tip_off";
                if (st == "starting")
                {
                    starting = true;
                    grace = 5 * minute;
                }
                else if (st == "running")
                {
                    starting = !tray.seen_online;
                    grace = tray.seen_online ? minute / 2 : 3 * minute;
                }
                else if (st == "stopping")
                {
                    grace = 2 * minute;
                    tip = "tip_stopping";
                }
                else if (st == "stopped")
                {
                    // A service that starts with the system may not run yet when the icon starts at sign in.
                    starting = scm["auto"].as_bool(false) && !tray.seen_online && now < 10 * minute;
                    grace = starting ? 5 * minute : minute;
                    warning = "notify_service_stopped";
                    tip = "tip_stopped";
                }
                else if (!tray.seen_online)
                {
                    // A service the manager does not know, as a console run: a long wait.
                    grace = 5 * minute;
                }
                const bool waiting = starting && silent < grace;
                if (waiting) tip = "tip_starting";
                if (!tray.warned && silent >= grace)
                {
                    tray.warned = true;
                    notify(L"Pseudalis", tr(warning), true);
                }
                set_icon(1);
                set_tooltip(tr(tip));
                // The events of the drive are read again once the service answers, its database perhaps a new one.
                tray.last_event = -1;
                tray.catching_up = true;
                tray.client.reload();
                set_poll_period(2000);
                check_setup_result();
                return;
            }
            tray.silent_since = 0;
            tray.warned = false;
            tray.online = true;
            tray.seen_online = true;
            tray.info = (*result)["info"];
            if (result->contains("updates")) tray.updates = (*result)["updates"];
            if (tray.open_pending)
            {
                tray.open_pending = false;
                open_app();
            }
            const json& s = tray.info;

            // A newer service means the programs were updated: the icon starts again, once,
            // when its program file is the new one.
            if (compare_versions(s["version"].as_string(), PSEUDALIS_VERSION) > 0 && !tray.relaunch_on_restart &&
                program_replaced())
            {
                tray.relaunch_on_restart = true;
                relaunch(L"--relaunched");
                PostMessageW(tray.hwnd, WM_CLOSE, 0, 0);
                return;
            }

            // Tooltip: version, drives, originals kept.
            const json& d = s["drive"];
            const json& rd = d["restore"];
            const std::string letter = d["letter"].as_string(), restore_letter = rd["letter"].as_string();
            std::wstring tip = widen(PSEUDALIS_FULL_NAME) + L" " + widen(s["version"].as_string()) + L"\n";
            if (letter.empty()) tip += tr("tip_no_drive");
            else if (d["mounted"].as_bool(false)) tip += tr("tip_drive", { { "l", letter } });
            else tip += tr("tip_drive_error", { { "l", letter } });
            if (!restore_letter.empty())
                tip += L" \xB7 " + tr(rd["mounted"].as_bool(false) ? "tip_restore_drive" : "tip_restore_drive_error",
                                      { { "l", restore_letter } });
            tip += L"\n" + tr("tip_records", { { "n", std::to_string(s["records"].as_int(0)) } });
            set_tooltip(tip);
            show_letters();

            // A drive that cannot take its letter is notified once for each of its errors.
            const std::string drive_error = letter.empty() || d["mounted"].as_bool(false) ? std::string()
                : d["error"].as_string();
            if (!drive_error.empty() && drive_error != tray.drive_error)
                notify(L"Pseudalis", tr("notify_drive_error", { { "l", letter }, { "e", drive_error } }), true);
            tray.drive_error = drive_error;
            const std::string restore_error = restore_letter.empty() || rd["mounted"].as_bool(false) ? std::string()
                : rd["error"].as_string();
            if (!restore_error.empty() && restore_error != tray.restore_error)
                notify(L"Pseudalis",
                       tr("notify_drive_error", { { "l", restore_letter }, { "e", restore_error } }), true);
            tray.restore_error = restore_error;

            notify_events((*result)["events"]);

            // Updates: notified once per version; a check asked from the menu is answered.
            const json& up = tray.updates;
            const std::string state = up["state"].as_string();
            const bool ready = state == "ready";
            if (ready && tray.notified_update != up["latest"].as_string())
            {
                tray.notified_update = up["latest"].as_string();
                notify(tr("notify_update_title"), tr("notify_update_text", { { "v", tray.notified_update } }));
            }
            if (tray.check_requested && state != "checking" && state != "downloading")
            {
                tray.check_requested = false;
                if (state == "up_to_date") notify(L"Pseudalis", tr("notify_uptodate"));
                else if (state == "error") notify(L"Pseudalis", update_error_text(up), true);
            }
            check_setup_result();
            set_icon(ready ? 2 : 0);
            // Every two seconds while something changes, every five seconds otherwise.
            const bool changing = tray.check_requested || state == "checking" || state == "downloading";
            set_poll_period(changing ? 2000 : 5000);
        }

        void post_action(const std::string& path, action_tag tag, int timeout)
        {
            service_client c = tray.client;
            ui::run_async(tray.hwnd, WM_ACTION_DONE, tag, [c, path, timeout]() {
                json out;
                std::string error;
                if (!c.send("POST", path, json::object(), out, error, timeout)) out = failed_answer(out, error);
                return out;
            });
        }

        void on_action_done(WPARAM tag, json* r)
        {
            set_poll_period(2000);
            if (tag == ACT_RESTART && (*r)["code"].as_string() == "service")
            {
                // The service does not answer: the service manager restarts it.
                ui::run_async(tray.hwnd, WM_ACTION_DONE, ACT_SCM, []() { return control_service(true); });
                return;
            }
            if (r->contains("error") && !(*r)["error"].as_string().empty())
            {
                ui::message(nullptr, error_text(*r), MB_OK | MB_ICONWARNING);
                return;
            }
            switch (tag)
            {
            case ACT_CHECK:
                tray.check_requested = true;
                break;
            case ACT_APPLY_UPDATE:
            case ACT_RESTART:
            case ACT_SCM:
                notify(L"Pseudalis", tr("notify_restarting"));
                break;
            default:
                break;
            }
        }

        // A text of a menu, whose & are doubled: the menus take & for the mark of a shortcut.
        std::wstring menu_label(const std::wstring& text)
        {
            std::wstring out;
            for (wchar_t c : text)
            {
                if (c == L'&') out += L'&';
                out += c;
            }
            return out;
        }

        // The About window: the banner, the version, the author, the license.
        enum { IDC_A_TITLE = 3000, IDC_A_TEXT, IDC_A_PROJECT, IDC_A_CLOSE };

        struct about_state
        {
            HWND title = nullptr, text = nullptr, project = nullptr, close = nullptr;
            HBITMAP art = nullptr;
            int art_w = 0, art_h = 0;
        } about;

        void about_layout(HWND w)
        {
            RECT rc;
            GetClientRect(w, &rc);
            const int cw = rc.right, ch = rc.bottom;
            const int m = ui::px(w, 18);
            const SIZE s = ui::png_size(IDB_ABOUT);
            const int ah = s.cx > 0 ? MulDiv(cw, s.cy, s.cx) : 0;
            if (cw != about.art_w || ah != about.art_h)
            {
                if (about.art) DeleteObject(about.art);
                about.art = ui::load_png(IDB_ABOUT, cw, ah);
                about.art_w = cw;
                about.art_h = ah;
                InvalidateRect(w, nullptr, TRUE);
            }
            int y = ah + ui::px(w, 12);
            ui::place(about.title, m, y, cw - 2 * m, ui::px(w, 36));
            y += ui::px(w, 40);
            const int bh = ui::px(w, 30);
            ui::place(about.text, m, y, cw - 2 * m, ch - y - bh - 2 * m);
            const int bw = ui::px(w, 110);
            ui::place(about.close, cw - m - bw, ch - m - bh, bw, bh);
            ui::place(about.project, m, ch - m - bh, ui::px(w, 150), bh);
        }

        LRESULT CALLBACK about_proc(HWND w, UINT msg, WPARAM wp, LPARAM lp)
        {
            switch (msg)
            {
            case WM_CREATE:
            {
                about.title = ui::label(w, L"Pseudalis", IDC_A_TITLE, 0, ui::font_kind::title);
                const std::wstring body = tr("about_version", { { "p", PSEUDALIS_FULL_NAME },
                    { "v", PSEUDALIS_VERSION } }) + L"\n"
                    + tr("about_purpose") + L"\n\n"
                    + tr("about_written", { { "a", PSEUDALIS_AUTHOR_NAME " (" PSEUDALIS_AUTHOR_EMAIL ")" } }) + L"\n"
                    + tr("about_built") + L"\n\n"
                    + widen(PSEUDALIS_COPYRIGHT) + L"\n" + tr("about_license");
                about.text = ui::label(w, body, IDC_A_TEXT);
                about.project = ui::button(w, tr("btn_project"), IDC_A_PROJECT);
                about.close = ui::button(w, tr("btn_close"), IDC_A_CLOSE, BS_DEFPUSHBUTTON);
                about_layout(w);
                return 0;
            }
            case WM_SIZE:
                about_layout(w);
                return 0;
            case WM_DPICHANGED:
            {
                for (HWND c : { about.text, about.project, about.close })
                    SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(ui::font(w, ui::font_kind::normal)), TRUE);
                SendMessageW(about.title, WM_SETFONT, reinterpret_cast<WPARAM>(ui::font(w, ui::font_kind::title)),
                    TRUE);
                const RECT* r = reinterpret_cast<RECT*>(lp);
                SetWindowPos(w, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top,
                    SWP_NOZORDER | SWP_NOACTIVATE);
                return 0;
            }
            case WM_ERASEBKGND:
            {
                RECT rc;
                GetClientRect(w, &rc);
                FillRect(reinterpret_cast<HDC>(wp), &rc, ui::background_brush());
                return 1;
            }
            case WM_PAINT:
            {
                PAINTSTRUCT ps;
                HDC dc = BeginPaint(w, &ps);
                ui::draw_bitmap(dc, about.art, 0, 0);
                EndPaint(w, &ps);
                return 0;
            }
            case WM_CTLCOLORSTATIC:
            case WM_CTLCOLORBTN:
            {
                HDC dc = reinterpret_cast<HDC>(wp);
                SetBkColor(dc, ui::color_background());
                SetTextColor(dc, reinterpret_cast<HWND>(lp) == about.title ? ui::color_accent() : ui::color_text());
                return reinterpret_cast<LRESULT>(ui::background_brush());
            }
            case WM_COMMAND:
                switch (LOWORD(wp))
                {
                case IDC_A_PROJECT: ui::open_url(widen(PSEUDALIS_PROJECT_URL)); break;
                case IDC_A_CLOSE:
                case IDCANCEL: DestroyWindow(w); break;
                default: break;
                }
                return 0;
            case WM_CLOSE:
                DestroyWindow(w);
                return 0;
            case WM_DESTROY:
                if (about.art) DeleteObject(about.art);
                about = about_state();
                tray.about = nullptr;
                return 0;
            default:
                return DefWindowProcW(w, msg, wp, lp);
            }
        }

        void show_about()
        {
            if (tray.about)
            {
                ShowWindow(tray.about, SW_RESTORE);
                SetForegroundWindow(tray.about);
                return;
            }
            static bool registered = false;
            HINSTANCE inst = GetModuleHandleW(nullptr);
            if (!registered)
            {
                WNDCLASSEXW wc = { sizeof(wc) };
                wc.style = CS_HREDRAW | CS_VREDRAW;
                wc.lpfnWndProc = about_proc;
                wc.hInstance = inst;
                wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
                wc.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(IDI_APP));
                wc.lpszClassName = about_class;
                RegisterClassExW(&wc);
                registered = true;
            }
            const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
            const int dpi = static_cast<int>(ui::dpi_of(nullptr));
            RECT r = { 0, 0, MulDiv(600, dpi, 96), MulDiv(560, dpi, 96) };
            AdjustWindowRectEx(&r, style, FALSE, WS_EX_CONTROLPARENT);
            RECT work;
            SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
            const int ww = r.right - r.left, wh = r.bottom - r.top;
            const int x = work.left + std::max(0L, (work.right - work.left - ww) / 2);
            const int y = work.top + std::max(0L, (work.bottom - work.top - wh) / 2);
            tray.about = CreateWindowExW(WS_EX_CONTROLPARENT, about_class, tr("about_title").c_str(), style, x, y, ww,
                wh, nullptr, nullptr, inst, nullptr);
            if (!tray.about) return;
            ShowWindow(tray.about, SW_SHOWNORMAL);
            SetForegroundWindow(tray.about);
        }

        void show_menu()
        {
            HMENU m = CreatePopupMenu();
            const bool on = tray.online;
            AppendMenuW(m, MF_STRING | (on ? 0 : MF_GRAYED), CMD_OPEN, tr("tray_open").c_str());
            SetMenuDefaultItem(m, CMD_OPEN, FALSE);
            const std::string letter = on ? drive_letter() : std::string();
            if (!letter.empty())
                AppendMenuW(m, MF_STRING, CMD_DRIVE, menu_label(tr("tray_drive", { { "l", letter } })).c_str());
            const std::string restore_letter = on ? drive_letter(true) : std::string();
            if (!restore_letter.empty())
                AppendMenuW(m, MF_STRING, CMD_RESTORE_DRIVE,
                    menu_label(tr("tray_restore_drive", { { "l", restore_letter } })).c_str());
            if (on && !refused_folder().empty())
                AppendMenuW(m, MF_STRING, CMD_REFUSED, tr("tray_refused").c_str());
            AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
            AppendMenuW(m, MF_STRING | (on ? 0 : MF_GRAYED), CMD_CHECK_UPDATES, tr("tray_updates").c_str());
            const json& up = tray.updates;
            if (on && up["state"].as_string() == "error")
            {
                // The error of the last check or of the last download, under the command.
                std::wstring why = update_error_text(up);
                if (why.size() > 160) why = why.substr(0, 159) + L"\x2026";
                AppendMenuW(m, MF_STRING | MF_GRAYED, 0, menu_label(why).c_str());
            }
            if (on && up["state"].as_string() == "ready")
                AppendMenuW(m, MF_STRING, CMD_INSTALL_UPDATE,
                    tr("tray_install_update", { { "v", up["latest"].as_string() } }).c_str());
            if (on) AppendMenuW(m, MF_STRING, CMD_RESTART, tr("tray_restart").c_str());
            else AppendMenuW(m, MF_STRING, CMD_START_SERVICE, tr("tray_start_service").c_str());
            AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
            AppendMenuW(m, MF_STRING | (on ? 0 : MF_GRAYED), CMD_DOCS, tr("tray_docs").c_str());
            AppendMenuW(m, MF_STRING, CMD_ABOUT, tr("tray_about").c_str());
            AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
            AppendMenuW(m, MF_STRING, CMD_QUIT, tr("tray_quit").c_str());

            POINT p;
            GetCursorPos(&p);
            SetForegroundWindow(tray.hwnd);
            const UINT flags = TPM_RETURNCMD | TPM_RIGHTBUTTON | TPM_NONOTIFY;
            const UINT cmd = TrackPopupMenuEx(m, flags, p.x, p.y, tray.hwnd, nullptr);
            PostMessageW(tray.hwnd, WM_NULL, 0, 0);
            DestroyMenu(m);
            if (cmd) SendMessageW(tray.hwnd, WM_COMMAND, cmd, 0);
        }

        void on_command(int id)
        {
            switch (id)
            {
            case CMD_OPEN: open_app(); break;
            case CMD_DRIVE:
            case CMD_RESTORE_DRIVE:
            {
                const std::string letter = drive_letter(id == CMD_RESTORE_DRIVE);
                if (!letter.empty()) ui::open_url(widen(letter + ":\\"));
                break;
            }
            case CMD_REFUSED:
            {
                const std::string folder = refused_folder();
                if (!folder.empty()) ui::open_url(widen(folder));
                break;
            }
            case CMD_CHECK_UPDATES: post_action("/v1/updates/check", ACT_CHECK, 30); break;
            case CMD_INSTALL_UPDATE: post_action("/v1/updates/apply", ACT_APPLY_UPDATE, 30); break;
            case CMD_RESTART: post_action("/v1/restart", ACT_RESTART, 15); break;
            case CMD_START_SERVICE:
                ui::run_async(tray.hwnd, WM_ACTION_DONE, ACT_SCM, []() { return control_service(false); });
                break;
            case CMD_DOCS: ui::open_url(page_url("docs/")); break;
            case CMD_ABOUT: show_about(); break;
            case CMD_QUIT: DestroyWindow(tray.hwnd); break;
            default: break;
            }
        }

        /*
            A result of a command of the Explorer, sent by pseudalis.exe: {"ok", "text", "name",
            "error"}.  The results that come within a second and a half are notified together.
        */
        void on_shell_result(const COPYDATASTRUCT* cd)
        {
            if (!cd || cd->dwData != results_mark || !cd->lpData || cd->cbData > 65536) return;
            const json r = json::parse_or_null(std::string(static_cast<const char*>(cd->lpData), cd->cbData));
            if (!r.is_object()) return;
            shell_result s;
            s.ok = r["ok"].as_bool(false);
            s.text = widen(r["text"].as_string());
            s.name = r["name"].as_string();
            s.error = r["error"].as_string();
            tray.results.push_back(s);
            SetTimer(tray.hwnd, TIMER_RESULTS, 1500, nullptr);
        }

        void notify_results()
        {
            KillTimer(tray.hwnd, TIMER_RESULTS);
            if (tray.results.empty()) return;
            size_t failed = 0;
            const shell_result* first_failure = nullptr;
            for (const auto& r : tray.results)
                if (!r.ok)
                {
                    ++failed;
                    if (!first_failure) first_failure = &r;
                }
            if (tray.results.size() == 1)
                notify(L"Pseudalis", tray.results.front().text, failed > 0);
            else if (failed == 0)
                notify(L"Pseudalis", tr("shell_many_done", { { "n", std::to_string(tray.results.size()) } }));
            else
                notify(L"Pseudalis", tr("shell_many_failed", { { "n", std::to_string(failed) },
                    { "f", first_failure->name }, { "e", first_failure->error } }), true);
            tray.results.clear();
        }

        LRESULT CALLBACK proc(HWND w, UINT msg, WPARAM wp, LPARAM lp)
        {
            if (msg == tray.taskbar_created && msg != 0)
            {
                // Explorer started again: the icon is added again.
                add_icon();
                tray.icon_state = -1;
                return 0;
            }
            switch (msg)
            {
            case WM_TRAY:
                switch (LOWORD(lp))
                {
                case WM_LBUTTONUP:
                case NIN_SELECT:
                case NIN_KEYSELECT:
                    if (tray.online) open_app();
                    else show_menu();
                    return 0;
                case WM_CONTEXTMENU:
                case WM_RBUTTONUP:
                    show_menu();
                    return 0;
                default:
                    return 0;
                }
            case WM_TIMER:
                if (wp == TIMER_POLL) poll();
                else if (wp == TIMER_RESULTS) notify_results();
                return 0;
            case WM_STATUS:
            {
                json* r = reinterpret_cast<json*>(lp);
                on_status(r);
                delete r;
                return 0;
            }
            case WM_ACTION_DONE:
            {
                json* r = reinterpret_cast<json*>(lp);
                on_action_done(wp, r);
                delete r;
                return 0;
            }
            case WM_COPYDATA:
                on_shell_result(reinterpret_cast<const COPYDATASTRUCT*>(lp));
                return TRUE;
            case WM_OPEN_APP:
                // Asked at the end of an installation, before the service answers.
                if (tray.online) open_app();
                else tray.open_pending = true;
                return 0;
            case WM_SHOW_ABOUT:
                show_about();
                return 0;
            case WM_COMMAND:
                on_command(LOWORD(wp));
                return 0;
            case WM_DESTROY:
                if (tray.about) DestroyWindow(tray.about);
                Shell_NotifyIconW(NIM_DELETE, &tray.nid);
                PostQuitMessage(0);
                return 0;
            default:
                return DefWindowProcW(w, msg, wp, lp);
            }
        }

        std::string language_setting()
        {
            std::string text;
            if (read_file(join_path(data_dir(), "settings.json"), text, 1u << 20))
                return json::parse_or_null(text)["language"].str("auto");
            return "auto";
        }
    }

}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR command_line, int)
{
    // The libraries loaded from here on come from System32 only.
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32);
    using namespace pseudalis;
    const std::wstring args = command_line ? command_line : L"";
    const bool open = args.find(L"--open") != std::wstring::npos;
    const bool about = args.find(L"--about") != std::wstring::npos;

    // One icon per session: a second start passes its request to the first one, and a
    // relaunch waits for the previous icon to close.
    const bool relaunched = args.find(L"--relaunched") != std::wstring::npos;
    HANDLE mutex = CreateMutexW(nullptr, FALSE, L"Local\\PseudalisTray");
    const DWORD owned = mutex ? WaitForSingleObject(mutex, relaunched ? 15000 : 0) : WAIT_FAILED;
    if (owned != WAIT_OBJECT_0 && owned != WAIT_ABANDONED)
    {
        HWND other = FindWindowW(window_class, nullptr);
        if (other && open) PostMessageW(other, WM_OPEN_APP, 0, 0);
        if (other && about)
        {
            AllowSetForegroundWindow(ASFW_ANY);
            PostMessageW(other, WM_SHOW_ABOUT, 0, 0);
        }
        return 0;
    }

    set_language(language_setting());
    ui::init();
    tray.started = now_unix();
    tray.report_setup_result = args.find(L"--setup-result") != std::wstring::npos;

    WNDCLASSEXW wc = { sizeof(wc) };
    wc.lpfnWndProc = proc;
    wc.hInstance = instance;
    wc.lpszClassName = window_class;
    wc.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(IDI_APP));
    RegisterClassExW(&wc);
    tray.hwnd = CreateWindowExW(0, window_class, L"Pseudalis", WS_OVERLAPPED, 0, 0, 0, 0, nullptr, nullptr, instance,
        nullptr);
    tray.taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    add_icon();
    SetTimer(tray.hwnd, TIMER_POLL, 2000, nullptr);
    poll();
    if (open) PostMessageW(tray.hwnd, WM_OPEN_APP, 0, 0);
    if (about) PostMessageW(tray.hwnd, WM_SHOW_ABOUT, 0, 0);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0)
    {
        if (tray.about && IsDialogMessageW(tray.about, &msg)) continue;
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    if (mutex)
    {
        ReleaseMutex(mutex);
        CloseHandle(mutex);
    }
    return 0;
}
