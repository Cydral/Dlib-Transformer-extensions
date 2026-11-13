// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    albireo-tray: the icon of Albireo in the notification area of each user.

        albireo-tray                    shows the icon (started at sign in)
        albireo-tray --open             opens the web application, starting the icon if needed
        albireo-tray --show <window>    opens a window of the icon: models, host or about
        albireo-tray --setup-result     started again by the installer after an installation
                                        made without its window: notifies its failure

    The icon follows the service: its state (running, generating, stopped, update ready),
    its tooltip (model, engine, speed), and the notifications of the updates and of the
    downloads.  Its menu opens the web application, the windows of the models, of this
    computer and the About window, chooses the accelerator, checks for updates and
    restarts Albireo.

    While the service does not answer, the icon asks the service manager of Windows what
    the service does.  A service that starts (the delayed start after the system starts,
    the engines and the models of a slow machine), restarts or stops for an update is
    shown as starting, and the icon warns only when the service stays stopped, or stays
    silent while it runs, beyond the time such a step takes.
*/

#include "ui.h"

#include <shellapi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>
#include <utility>
#include <vector>

#include "../common/platform.h"
#include "../common/version.h"
#include "resource.h"
#include "service_client.h"
#include "strings.h"
#include "tray_windows.h"

namespace albireo
{

    namespace
    {
        const wchar_t window_class[] = L"AlbireoTrayWindow";
        const UINT WM_TRAY = WM_APP + 1;
        const UINT WM_STATUS = WM_APP + 2;
        const UINT WM_ACTION_DONE = WM_APP + 3;
        const UINT WM_OPEN_APP = WM_APP + 4;
        const UINT WM_SHOW_WINDOW = WM_APP + 5;
        const UINT_PTR TIMER_POLL = 1;

        // The choices of the devices of the menu take CMD_ACCEL_CHOICE and the identifiers after it.
        enum command_id
        {
            CMD_OPEN = 100, CMD_MODELS, CMD_HOST, CMD_ACCEL_AUTO, CMD_ACCEL_NONE, CMD_MEASURE, CMD_CHECK_UPDATES,
            CMD_INSTALL_UPDATE, CMD_RESTART, CMD_START_SERVICE, CMD_DOCS, CMD_ABOUT, CMD_QUIT, CMD_ACCEL_CHOICE = 300
        };

        enum action_tag { ACT_SETTINGS = 1, ACT_MEASURE, ACT_CHECK, ACT_APPLY_UPDATE, ACT_RESTART };

        struct tray_state
        {
            HWND hwnd = nullptr;
            UINT taskbar_created = 0;
            NOTIFYICONDATAW nid = {};
            service_client client;
            json status;
            json engines;
            bool online = false;
            // The service answered since the icon started.
            bool seen_online = false;
            // When the service stopped answering (GetTickCount64), and whether the icon
            // warned of it.
            ULONGLONG silent_since = 0;
            bool warned = false;
            // The web application opens once the service answers.
            bool open_pending = false;
            /*
                Whether the shell took the version 4 of the notification icons, which reports a
                click of the left button as NIN_SELECT and one of the right button as
                WM_CONTEXTMENU besides the messages of the mouse; whether the menu is shown; and
                when the web application was last opened (GetTickCount64).
            */
            bool version4 = false;
            bool menu_shown = false;
            ULONGLONG opened_at = 0;
            bool polling = false;
            int icon_state = -1;
            std::string notified_update;
            bool check_requested = false;
            json download_states = json::object();
            bool relaunch_on_restart = false;
            // When the version of the program file was last read (GetTickCount64).
            ULONGLONG version_checked = 0;
            // When the engines were last asked for, 0 to ask at the next poll.
            ULONGLONG engines_at = 0;
            // The period of the polls: short while something changes, long otherwise.
            UINT poll_ms = 2000;
            // The settings {engine, accelerator} of the choices of devices of the menu shown last.
            std::vector<std::pair<std::string, std::string>> choices;
            // When the icon started (seconds since 1970), and the result of the installer it read
            // last: an update that failed after the start of the icon is notified.
            int64_t started = 0;
            int64_t setup_result_time = 0;
            ULONGLONG setup_result_checked = 0;
            // Started again by the installer that closed it (--setup-result): the failure of
            // that run is notified, although the installer wrote it before the icon started.
            bool report_setup_result = false;
        } tray;

        /*
            The state of the service for the service manager: "stopped", "starting",
            "running", "stopping", "other" or "missing", whether it starts with the system,
            and the code of its last stop.  Read while the service does not answer.
        */
        json service_manager_state()
        {
            json out = json::object();
            out["state"] = "missing";
            SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT);
            if (!scm) return out;
            SC_HANDLE svc = OpenServiceW(scm, L"Albireo", SERVICE_QUERY_STATUS | SERVICE_QUERY_CONFIG);
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
                    out["exit_code"] = static_cast<long long>(sp.dwWin32ExitCode == ERROR_SERVICE_SPECIFIC_ERROR
                        ? sp.dwServiceSpecificExitCode : sp.dwWin32ExitCode);
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

        HICON load_icon(int id, int size)
        {
            return static_cast<HICON>(LoadImageW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(id), IMAGE_ICON,
                size, size, LR_DEFAULTCOLOR));
        }

        void set_icon(int state)
        {
            if (state == tray.icon_state) return;
            tray.icon_state = state;
            static const int ids[] = { IDI_TRAY, IDI_TRAY_BUSY, IDI_TRAY_OFF, IDI_TRAY_UPDATE };
            const int size = GetSystemMetrics(SM_CXSMICON);
            HICON old = tray.nid.hIcon;
            tray.nid.hIcon = load_icon(ids[state], size);
            tray.nid.uFlags = NIF_ICON;
            Shell_NotifyIconW(NIM_MODIFY, &tray.nid);
            if (old) DestroyIcon(old);
        }

        void set_tooltip(const std::wstring& text)
        {
            std::wstring t = text.substr(0, 127);
            if (t == tray.nid.szTip) return;
            wcsncpy(tray.nid.szTip, t.c_str(), 127);
            tray.nid.szTip[127] = 0;
            tray.nid.uFlags = NIF_TIP | NIF_SHOWTIP;
            Shell_NotifyIconW(NIM_MODIFY, &tray.nid);
        }

        void notify(const std::wstring& title, const std::wstring& text)
        {
            NOTIFYICONDATAW n = tray.nid;
            n.uFlags = NIF_INFO;
            wcsncpy(n.szInfoTitle, title.substr(0, 63).c_str(), 63);
            n.szInfoTitle[63] = 0;
            wcsncpy(n.szInfo, text.substr(0, 255).c_str(), 255);
            n.szInfo[255] = 0;
            n.dwInfoFlags = NIIF_INFO | NIIF_RESPECT_QUIET_TIME;
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
            wcscpy(tray.nid.szTip, L"Albireo");
            Shell_NotifyIconW(NIM_ADD, &tray.nid);
            tray.nid.uVersion = NOTIFYICON_VERSION_4;
            tray.version4 = Shell_NotifyIconW(NIM_SETVERSION, &tray.nid) != FALSE;
            tray.icon_state = 2;
        }

        std::string model_name(const std::string& id)
        {
            for (const auto& m : tray.status["loaded_models"].elements())
                if (m["id"].as_string() == id && m["name"].is_string() && !m["name"].as_string().empty())
                    return m["name"].as_string();
            return id;
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
                { "o", r["previous"].as_string() }, { "e", r["error"].as_string() } }));
        }

        void poll()
        {
            if (tray.polling) return;
            tray.polling = true;
            service_client c = tray.client;
            // The engines change rarely: they are asked for every thirty seconds, and after a change.
            const ULONGLONG now = GetTickCount64();
            const bool with_engines = !tray.online || tray.engines_at == 0 || now - tray.engines_at >= 30000;
            if (with_engines) tray.engines_at = now;
            // The polls share a connection kept open between them (service_client::poll()).
            ui::run_async(tray.hwnd, WM_STATUS, 0, [c, with_engines]() {
                json r = json::object();
                std::string error;
                json status;
                if (!c.poll("/albireo/v1/status", status, error, 5))
                {
                    r["error"] = error;
                    r["service"] = service_manager_state();
                    return r;
                }
                r["status"] = status;
                json engines;
                if (with_engines && c.poll("/albireo/v1/engines", engines, error, 5)) r["engines"] = engines;
                return r;
            });
        }

        void set_poll_period(UINT ms)
        {
            if (ms == tray.poll_ms) return;
            tray.poll_ms = ms;
            SetTimer(tray.hwnd, TIMER_POLL, ms, nullptr);
        }

        // The version of the program file on disk, which an update replaces while this one runs.
        std::string file_version(const std::string& path)
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
                version = std::to_string(HIWORD(info.dwFileVersionMS)) + "."
                    + std::to_string(LOWORD(info.dwFileVersionMS)) + "." + std::to_string(HIWORD(info.dwFileVersionLS));
            }
            FreeLibrary(m);
            return version;
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
            return compare_versions(file_version(executable_path()), ALBIREO_VERSION) > 0;
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
                /*
                    How long each state may last before the icon warns.  The delayed start of
                    the service comes about two minutes after the system starts, then a slow
                    machine describes its engines and loads its models; a restart and an
                    update stop the service for a while.
                */
                bool starting = false;
                ULONGLONG grace = minute;
                const char* warning = "notify_service_down";
                const char* tip = "tip_off";
                if (st == "starting")
                {
                    starting = true;
                    grace = 10 * minute;
                }
                else if (st == "running")
                {
                    starting = !tray.seen_online;
                    grace = tray.seen_online ? minute / 2 : 5 * minute;
                }
                else if (st == "stopping")
                {
                    grace = 2 * minute;
                    tip = "tip_stopping";
                }
                else if (st == "stopped")
                {
                    // Before its delayed start, a service that starts with the system is
                    // stopped: the icon, started at sign in, waits for it.
                    starting = scm["auto"].as_bool(false) && !tray.seen_online && now < 15 * minute;
                    grace = starting ? 15 * minute : minute;
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
                    notify(L"Albireo", tr(warning));
                }
                set_icon(waiting ? 1 : 2);
                set_tooltip(tr(tip));
                tray.client.reload();
                set_poll_period(2000);
                check_setup_result();
                return;
            }
            tray.silent_since = 0;
            tray.warned = false;
            tray.online = true;
            tray.seen_online = true;
            if (tray.open_pending)
            {
                tray.open_pending = false;
                ui::open_url(widen(tray.client.base_url()));
            }
            tray.status = (*result)["status"];
            if ((*result).contains("engines")) tray.engines = (*result)["engines"];
            const json& s = tray.status;

            // A newer service means the programs were updated: the icon starts again, once,
            // when its program file is the new one.
            const std::string version = s["version"].as_string();
            if (compare_versions(version, ALBIREO_VERSION) > 0 && !tray.relaunch_on_restart && program_replaced())
            {
                tray.relaunch_on_restart = true;
                relaunch(L"--relaunched");
                PostMessageW(tray.hwnd, WM_CLOSE, 0, 0);
                return;
            }

            // Tooltip: model, engine, speed, requests.
            const std::string general = s["active"]["general"].as_string();
            const json& stats = s["engine"]["stats"];
            bool loading = false;
            for (const auto& kv : s["models_state"].items()) loading = loading || kv.second.as_string() == "loading";
            std::wstring tip = L"Albireo";
            if (!general.empty()) tip += L" · " + widen(model_name(general));
            tip += L"\n" + widen(s["engine"]["variant"].as_string());
            if (s["offline"].as_bool(false)) tip += L" · " + tr("tip_offline");
            const long long active = stats["active"].as_int(0);
            const double tps = stats["generation_tokens_per_second"].as_double(0);
            if (general.empty()) tip += L" · " + tr("tip_no_model");
            else if (loading) tip += L" · " + tr("tip_loading");
            else if (active > 0) tip += L" · " + tr("tip_active", { { "n", std::to_string(active) } });
            else tip += L" · " + tr("tip_idle");
            if (tps > 0.05) tip += L"\n" + tr("tokens_s", { { "n", ui::format_decimal(tps, 1) } });
            // The server answers while the engines start and the models load.
            const bool starting = s["state"].as_string() == "starting";
            if (starting) tip = tr("tip_starting");
            set_tooltip(tip);

            // Updates: notified once per version; a check asked from the menu is answered.
            const json& up = s["updates"];
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
                if (state == "up_to_date") notify(L"Albireo", tr("notify_uptodate"));
                else if (state == "error") notify(L"Albireo", update_error_text(up));
            }
            check_setup_result();

            // Downloads that finished since the last poll.
            bool downloading = false;
            for (const auto& d : s["downloads"].elements())
            {
                const std::string id = d["id"].as_string();
                const std::string st = d["state"].as_string();
                const std::string before = tray.download_states[id].as_string();
                if (before != st && (before == "downloading" || before == "verifying" || before == "queued"))
                {
                    if (st == "done") notify(L"Albireo", tr("notify_installed", { { "m", id } }));
                    else if (st == "error")
                    {
                        // The text of the code of the failure, or the error of the transfer itself.
                        std::string why = d["code"].as_string() == "download_failed" ? d["args"]["e"].as_string()
                            : narrow(error_text(d));
                        if (why.empty()) why = d["error"].as_string();
                        notify(L"Albireo", tr("notify_failed", { { "m", id }, { "e", why } }));
                    }
                }
                tray.download_states[id] = st;
                downloading = downloading || st == "downloading" || st == "verifying" || st == "queued";
            }

            set_icon(ready ? 3 : active > 0 || loading || starting ? 1 : 0);
            // Every two seconds while something changes, every ten seconds otherwise.
            const bool changing = starting || loading || active > 0 || downloading || tray.check_requested
                || state == "checking" || state == "downloading";
            set_poll_period(changing ? 2000 : 10000);
        }

        void put_settings(const json& changes)
        {
            service_client c = tray.client;
            ui::run_async(tray.hwnd, WM_ACTION_DONE, ACT_SETTINGS, [c, changes]() {
                json out;
                std::string error;
                if (!c.send("PUT", "/albireo/v1/settings", changes, out, error)) out = failed_answer(out, error);
                return out;
            });
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
            // The service changes after an action: it is followed closely, its engines included.
            tray.engines_at = 0;
            set_poll_period(2000);
            if (r->contains("error") && !(*r)["error"].as_string().empty() && tag != ACT_MEASURE)
            {
                // The errors of the updates and of the settings come with a code, explained in the
                // language of the icon.
                ui::message(nullptr, error_text(*r), MB_OK | MB_ICONWARNING);
                return;
            }
            switch (tag)
            {
            case ACT_SETTINGS:
                if ((*r)["restart"].as_bool(false)) notify(L"Albireo", tr("notify_restarting"));
                break;
            case ACT_MEASURE:
            {
                std::string text;
                for (const auto& kv : (*r)["results"].items())
                {
                    const double v = kv.second["tokens_per_second"].as_double(0);
                    if (!text.empty()) text += ", ";
                    text += kv.first + " " + (v > 0 ? narrow(tr("tokens_s", { { "n", ui::format_decimal(v, 1) } }))
                        : std::string("-"));
                }
                if (r->contains("error")) text = (*r)["error"].as_string();
                notify(L"Albireo", tr("notify_measures", { { "r", text } }));
                break;
            }
            case ACT_CHECK:
                tray.check_requested = true;
                break;
            case ACT_APPLY_UPDATE:
            case ACT_RESTART:
                notify(L"Albireo", tr("notify_restarting"));
                if (tag == ACT_RESTART)
                {
                    // Everything restarts: the service by itself, the icon from its program.
                    relaunch(L"--relaunched");
                    PostMessageW(tray.hwnd, WM_CLOSE, 0, 0);
                }
                break;
            default:
                break;
            }
        }

        // Opens the web application once for a click the shell reports twice.
        void open_app()
        {
            const ULONGLONG now = GetTickCount64();
            if (tray.opened_at != 0 && now - tray.opened_at < 1500) return;
            tray.opened_at = now;
            ui::open_url(widen(tray.client.base_url()));
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

        /*
            The menu of the accelerator: the automatic choice, the processor alone, one choice
            per device the engines offer, named by its devices and its runtime, then, grayed,
            why an engine of a device cannot run here and which devices the runtime of an
            engine left out.  The check mark follows the settings; settings that name none of
            the choices, an accelerator with the automatic engine for instance, mark the choice
            in use.
        */
        HMENU accelerator_menu(bool on)
        {
            HMENU acc = CreatePopupMenu();
            const json& eng = tray.engines;
            const std::string engine = eng["setting"].str("auto");
            const std::string accelerator = eng["accelerator"].str("auto");
            const bool automatic = engine == "auto" || engine == "best";
            tray.choices.clear();
            std::vector<std::wstring> labels;
            int marked = -1, in_use = -1;
            for (const auto& o : eng["options"].elements())
            {
                const std::string kind = o["kind"].as_string();
                // The devices, and the processor through the graph of OpenVINO.
                if (kind != "gpu" && kind != "npu" && kind != "both" && o["key"].as_string() != "openvino:cpu")
                    continue;
                const std::pair<std::string, std::string> settings = choice_settings(o);
                const int index = static_cast<int>(tray.choices.size());
                if (settings.first == engine && settings.second == accelerator) marked = index;
                if (o["current"].as_bool(false)) in_use = index;
                tray.choices.push_back(settings);
                labels.push_back(choice_label(o));
            }
            const bool auto_marked = automatic && accelerator == "auto";
            const bool none_marked = (automatic && accelerator == "none") || engine.compare(0, 3, "cpu") == 0;
            if (marked < 0 && !auto_marked && !none_marked) marked = in_use;
            const UINT enabled = on ? 0 : MF_GRAYED;
            AppendMenuW(acc, MF_STRING | enabled | (auto_marked ? MF_CHECKED : 0), CMD_ACCEL_AUTO,
                tr("accel_auto").c_str());
            AppendMenuW(acc, MF_STRING | enabled | (none_marked ? MF_CHECKED : 0), CMD_ACCEL_NONE,
                tr("accel_none").c_str());
            if (!labels.empty()) AppendMenuW(acc, MF_SEPARATOR, 0, nullptr);
            for (size_t i = 0; i < labels.size(); ++i)
                AppendMenuW(acc, MF_STRING | enabled | (static_cast<int>(i) == marked ? MF_CHECKED : 0),
                    CMD_ACCEL_CHOICE + static_cast<UINT>(i), menu_label(labels[i]).c_str());
            // Why the engines of the devices offer no choice, or fewer choices than their devices.
            std::vector<std::wstring> notes;
            for (const auto& x : eng["engines"].elements())
            {
                const std::string variant = x["variant"].as_string();
                if (variant != "cuda" && variant.compare(0, 8, "openvino") != 0) continue;
                std::wstring note;
                const std::wstring why = reason_text(x);
                if (!x["usable"].as_bool(false) && !why.empty())
                    note = tr("accel_cannot_run", { { "r", engine_name(variant) }, { "why", narrow(why) } });
                else if (x["usable"].as_bool(false) && !x["missing"].as_string().empty())
                    note = tr("accel_left_out", { { "r", runtime_name(variant) }, { "m", x["missing"].as_string() } });
                if (!note.empty() && std::find(notes.begin(), notes.end(), note) == notes.end()) notes.push_back(note);
            }
            if (notes.empty() && eng["accelerators_reason"].as_string() == "not_installed")
                notes.push_back(tr("accel_not_installed"));
            if (!notes.empty() && labels.empty()) AppendMenuW(acc, MF_SEPARATOR, 0, nullptr);
            // A long reason ends with an ellipsis; This computer gives it whole.
            for (const auto& n : notes)
            {
                const std::wstring shown = n.size() > 110 ? n.substr(0, 109) + L"…" : n;
                AppendMenuW(acc, MF_STRING | MF_GRAYED, 0, menu_label(shown).c_str());
            }
            AppendMenuW(acc, MF_SEPARATOR, 0, nullptr);
            AppendMenuW(acc, MF_STRING | enabled, CMD_MEASURE, tr("accel_measure").c_str());
            return acc;
        }

        void show_menu()
        {
            // A second message of the same click finds the menu shown and leaves it alone.
            if (tray.menu_shown) return;
            tray.menu_shown = true;
            HMENU m = CreatePopupMenu();
            const bool on = tray.online;
            AppendMenuW(m, MF_STRING | (on ? 0 : MF_GRAYED), CMD_OPEN, tr("tray_open").c_str());
            SetMenuDefaultItem(m, CMD_OPEN, FALSE);
            AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
            AppendMenuW(m, MF_STRING | (on ? 0 : MF_GRAYED), CMD_MODELS, tr("tray_models").c_str());
            AppendMenuW(m, MF_STRING | (on ? 0 : MF_GRAYED), CMD_HOST, tr("tray_host").c_str());

            HMENU acc = accelerator_menu(on);
            AppendMenuW(m, MF_POPUP | (on ? 0 : MF_GRAYED), reinterpret_cast<UINT_PTR>(acc), tr("tray_accel").c_str());
            AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
            // Offline, an update comes from a folder (Models, Import).
            const json& status = tray.status;
            const bool offline = status["offline"].as_bool(false);
            AppendMenuW(m, MF_STRING | (on && !offline ? 0 : MF_GRAYED), CMD_CHECK_UPDATES, tr("tray_updates").c_str());
            const json& up = status["updates"];
            if (on && up["state"].as_string() == "error")
            {
                // The error of the last check or of the last download, under the command.
                std::wstring why = update_error_text(up);
                if (why.size() > 160) why = why.substr(0, 159) + L"…";
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
            tray.menu_shown = false;
            if (cmd) SendMessageW(tray.hwnd, WM_COMMAND, cmd, 0);
        }

        void on_command(int id)
        {
            json changes = json::object();
            switch (id)
            {
            case CMD_OPEN: open_app(); break;
            case CMD_MODELS: show_models_window(tray.client); break;
            case CMD_HOST: show_host_window(tray.client); break;
            case CMD_ACCEL_AUTO:
            case CMD_ACCEL_NONE:
                // The automatic engine, with every device or with the processor alone.
                changes["engine"] = "auto";
                changes["accelerator"] = id == CMD_ACCEL_AUTO ? "auto" : "none";
                put_settings(changes);
                break;
            case CMD_MEASURE: post_action("/albireo/v1/engines/benchmark", ACT_MEASURE, 3600); break;
            case CMD_CHECK_UPDATES: post_action("/albireo/v1/updates/check", ACT_CHECK, 30); break;
            case CMD_INSTALL_UPDATE: post_action("/albireo/v1/updates/apply", ACT_APPLY_UPDATE, 30); break;
            case CMD_RESTART: post_action("/albireo/v1/restart", ACT_RESTART, 30); break;
            case CMD_START_SERVICE:
            {
                const std::wstring setup = widen(join_path(executable_dir(), "albireo-setup.exe"));
                ShellExecuteW(nullptr, L"runas", setup.c_str(), L"/service start", nullptr, SW_HIDE);
                break;
            }
            case CMD_DOCS: ui::open_url(widen(tray.client.base_url() + "docs/")); break;
            case CMD_ABOUT: show_about_window(tray.client); break;
            case CMD_QUIT: DestroyWindow(tray.hwnd); break;
            default:
                // A device of an engine: its engine and its accelerator.
                if (id >= CMD_ACCEL_CHOICE && id < CMD_ACCEL_CHOICE + static_cast<int>(tray.choices.size()))
                {
                    const auto& choice = tray.choices[static_cast<size_t>(id - CMD_ACCEL_CHOICE)];
                    changes["engine"] = choice.first;
                    changes["accelerator"] = choice.second;
                    put_settings(changes);
                }
                break;
            }
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
            {
                /*
                    With the version 4 of the notification icons, the shell reports a click of
                    the left button as NIN_SELECT and a click of the right button as
                    WM_CONTEXTMENU, besides the messages of the mouse, which a shell without that
                    version sends alone; the icon acts on one message of each click.
                */
                const UINT event = LOWORD(lp);
                const bool select = event == NIN_SELECT || event == NIN_KEYSELECT
                    || (event == WM_LBUTTONUP && !tray.version4);
                const bool menu = event == WM_CONTEXTMENU || (event == WM_RBUTTONUP && !tray.version4);
                if (select)
                {
                    if (tray.online) open_app();
                    else show_menu();
                }
                else if (menu)
                {
                    show_menu();
                }
                return 0;
            }
            case WM_TIMER:
                if (wp == TIMER_POLL) poll();
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
            case WM_OPEN_APP:
                // Asked at the end of an installation, before the service answers.
                if (tray.online) open_app();
                else tray.open_pending = true;
                return 0;
            case WM_SHOW_WINDOW:
                if (wp == 1) show_models_window(tray.client);
                else if (wp == 2) show_host_window(tray.client);
                else if (wp == 3) show_about_window(tray.client);
                return 0;
            case WM_COMMAND:
                on_command(LOWORD(wp));
                return 0;
            case WM_DESTROY:
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
    using namespace albireo;
    const std::wstring args = command_line ? command_line : L"";
    const bool open = args.find(L"--open") != std::wstring::npos;
    WPARAM show = 0;
    if (args.find(L"--show models") != std::wstring::npos) show = 1;
    else if (args.find(L"--show host") != std::wstring::npos) show = 2;
    else if (args.find(L"--show about") != std::wstring::npos) show = 3;

    // One icon per session: a second start passes its request to the first one, and a
    // relaunch waits for the previous icon to close.
    const bool relaunched = args.find(L"--relaunched") != std::wstring::npos;
    HANDLE mutex = CreateMutexW(nullptr, FALSE, L"Local\\AlbireoTray");
    const DWORD owned = mutex ? WaitForSingleObject(mutex, relaunched ? 15000 : 0) : WAIT_FAILED;
    if (owned != WAIT_OBJECT_0 && owned != WAIT_ABANDONED)
    {
        HWND other = FindWindowW(window_class, nullptr);
        if (other && open) PostMessageW(other, WM_OPEN_APP, 0, 0);
        if (other && show)
        {
            AllowSetForegroundWindow(ASFW_ANY);
            PostMessageW(other, WM_SHOW_WINDOW, show, 0);
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
    tray.hwnd = CreateWindowExW(0, window_class, L"Albireo", WS_OVERLAPPED, 0, 0, 0, 0,
        nullptr, nullptr, instance, nullptr);
    tray.taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    add_icon();
    SetTimer(tray.hwnd, TIMER_POLL, 2000, nullptr);
    poll();
    if (open) PostMessageW(tray.hwnd, WM_OPEN_APP, 0, 0);
    if (show) PostMessageW(tray.hwnd, WM_SHOW_WINDOW, show, 0);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0)
    {
        if (!dispatch_dialog_message(&msg))
        {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    if (mutex)
    {
        ReleaseMutex(mutex);
        CloseHandle(mutex);
    }
    return 0;
}
