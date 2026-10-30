// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    albireo-tray: the icon of Albireo in the notification area of each user.

        albireo-tray                    shows the icon (started at sign in)
        albireo-tray --open             opens the web application, starting the icon if needed
        albireo-tray --show <window>    opens a window of the icon: models, host or about

    The icon follows the service: its state (running, generating, stopped, update ready),
    its tooltip (model, engine, speed), and the notifications of the updates and of the
    downloads.  Its menu opens the web application, the windows of the models, of this
    computer and the About window, chooses the accelerator, checks for updates and
    restarts Albireo.
*/

#include "ui.h"

#include <shellapi.h>

#include <atomic>
#include <cstdlib>
#include <cwchar>
#include <string>

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

        enum command_id
        {
            CMD_OPEN = 100, CMD_MODELS, CMD_HOST, CMD_ACCEL_AUTO, CMD_ACCEL_NONE, CMD_ACCEL_NPU, CMD_ACCEL_GPU,
            CMD_ACCEL_BOTH, CMD_MEASURE, CMD_CHECK_UPDATES, CMD_INSTALL_UPDATE, CMD_RESTART, CMD_START_SERVICE,
            CMD_DOCS, CMD_ABOUT, CMD_QUIT
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
            bool was_online = true;
            bool polling = false;
            int icon_state = -1;
            std::string notified_update;
            bool check_requested = false;
            json download_states = json::object();
            bool relaunch_on_restart = false;
            int failures = 0;
        } tray;

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
            Shell_NotifyIconW(NIM_SETVERSION, &tray.nid);
            tray.icon_state = 2;
        }

        std::string model_name(const std::string& id)
        {
            for (const auto& m : tray.status["loaded_models"].elements())
                if (m["id"].as_string() == id && m["name"].is_string() && !m["name"].as_string().empty())
                    return m["name"].as_string();
            return id;
        }

        bool cpu_engine()
        {
            const std::string key = tray.status["engine"]["key"].as_string();
            return key.compare(0, 3, "cpu") == 0 || key.empty();
        }

        void poll()
        {
            if (tray.polling) return;
            tray.polling = true;
            service_client c = tray.client;
            ui::run_async(tray.hwnd, WM_STATUS, 0, [c]() {
                json r = json::object();
                std::string error;
                json status;
                if (!c.get("/albireo/v1/status", status, error, 5))
                {
                    r["error"] = error;
                    return r;
                }
                r["status"] = status;
                json engines;
                if (c.get("/albireo/v1/engines", engines, error, 5)) r["engines"] = engines;
                return r;
            });
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

        int compare_version(const std::string& a, const std::string& b)
        {
            int x[3] = { 0, 0, 0 }, y[3] = { 0, 0, 0 };
            sscanf(a.c_str(), "%d.%d.%d", &x[0], &x[1], &x[2]);
            sscanf(b.c_str(), "%d.%d.%d", &y[0], &y[1], &y[2]);
            for (int i = 0; i < 3; ++i)
                if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
            return 0;
        }

        void on_status(json* result)
        {
            tray.polling = false;
            const bool online = !result->contains("error");
            if (!online)
            {
                tray.online = false;
                // A restart of the service makes it silent a few seconds: no notification
                // before several failed polls.
                if (tray.was_online && ++tray.failures >= 4)
                {
                    tray.was_online = false;
                    notify(L"Albireo", tr("notify_service_down"));
                }
                set_icon(2);
                set_tooltip(tr("tip_off"));
                tray.client.reload();
                return;
            }
            tray.failures = 0;
            tray.online = true;
            tray.was_online = true;
            tray.status = (*result)["status"];
            if ((*result).contains("engines")) tray.engines = (*result)["engines"];
            const json& s = tray.status;

            // A newer service means the programs were updated: the icon starts again from
            // the new program, once.
            const std::string version = s["version"].as_string();
            if (compare_version(version, ALBIREO_VERSION) > 0 && !tray.relaunch_on_restart)
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
            if (tps > 0.05)
            {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.1f", tps);
                tip += L"\n" + tr("tokens_s", { { "n", buf } });
            }
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
                else if (state == "error")
                    notify(L"Albireo", tr("notify_update_error", { { "e", up["error"].as_string() } }));
            }

            // Downloads that finished since the last poll.
            for (const auto& d : s["downloads"].elements())
            {
                const std::string id = d["id"].as_string();
                const std::string st = d["state"].as_string();
                const std::string before = tray.download_states[id].as_string();
                if (before != st && (before == "downloading" || before == "verifying" || before == "queued"))
                {
                    if (st == "done") notify(L"Albireo", tr("notify_installed", { { "m", id } }));
                    else if (st == "error")
                        notify(L"Albireo", tr("notify_failed", { { "m", id }, { "e", d["error"].as_string() } }));
                }
                tray.download_states[id] = st;
            }

            set_icon(ready ? 3 : active > 0 || loading ? 1 : 0);
        }

        void put_settings(const json& changes)
        {
            service_client c = tray.client;
            ui::run_async(tray.hwnd, WM_ACTION_DONE, ACT_SETTINGS, [c, changes]() {
                json out;
                std::string error;
                if (!c.send("PUT", "/albireo/v1/settings", changes, out, error))
                {
                    out = json::object();
                    out["error"] = error;
                }
                return out;
            });
        }

        void post_action(const std::string& path, action_tag tag, int timeout)
        {
            service_client c = tray.client;
            ui::run_async(tray.hwnd, WM_ACTION_DONE, tag, [c, path, timeout]() {
                json out;
                std::string error;
                if (!c.send("POST", path, json::object(), out, error, timeout))
                {
                    out = json::object();
                    out["error"] = error;
                }
                return out;
            });
        }

        void on_action_done(WPARAM tag, json* r)
        {
            if (r->contains("error") && !(*r)["error"].as_string().empty() && tag != ACT_MEASURE)
            {
                ui::message(nullptr, widen((*r)["error"].as_string()), MB_OK | MB_ICONWARNING);
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
                    char buf[64];
                    const double v = kv.second["tokens_per_second"].as_double(0);
                    if (v > 0) snprintf(buf, sizeof(buf), "%.1f", v);
                    else snprintf(buf, sizeof(buf), "-");
                    if (!text.empty()) text += ", ";
                    text += kv.first + " " + buf;
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

        void open_app()
        {
            ui::open_url(widen(tray.client.base_url()));
        }

        void show_menu()
        {
            HMENU m = CreatePopupMenu();
            const bool on = tray.online;
            AppendMenuW(m, MF_STRING | (on ? 0 : MF_GRAYED), CMD_OPEN, tr("tray_open").c_str());
            SetMenuDefaultItem(m, CMD_OPEN, FALSE);
            AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
            AppendMenuW(m, MF_STRING | (on ? 0 : MF_GRAYED), CMD_MODELS, tr("tray_models").c_str());
            AppendMenuW(m, MF_STRING | (on ? 0 : MF_GRAYED), CMD_HOST, tr("tray_host").c_str());

            // The accelerators may be forced while the processor engine runs.
            HMENU acc = CreatePopupMenu();
            const std::string setting = tray.engines["accelerator"].str("auto");
            const bool available = tray.engines["accelerators_available"].as_bool(false);
            const bool allowed = on && available && (cpu_engine() || setting != "auto");
            struct item { int id; const char* key; const char* value; };
            const item items[] = { { CMD_ACCEL_AUTO, "accel_auto", "auto" }, { CMD_ACCEL_NONE, "accel_none", "none" },
                { CMD_ACCEL_NPU, "accel_npu", "npu" }, { CMD_ACCEL_GPU, "accel_gpu", "gpu" },
                { CMD_ACCEL_BOTH, "accel_both", "both" } };
            for (const auto& it : items)
            {
                bool has = true;
                const std::string v = it.value;
                if (v == "npu" || v == "gpu" || v == "both")
                {
                    has = false;
                    for (const auto& o : tray.engines["options"].elements())
                    {
                        const std::string k = o["key"].as_string();
                        const bool npu = k == "openvino:npu";
                        const bool gpu = k == "cuda" || k == "openvino:gpu";
                        if ((v == "npu" && npu) || (v == "gpu" && gpu) || (v == "both" && (npu || gpu)))
                            has = true;
                    }
                }
                AppendMenuW(acc, MF_STRING | (allowed && has ? 0 : MF_GRAYED) | (setting == v ? MF_CHECKED : 0), it.id,
                    tr(it.key).c_str());
            }
            if (on && !available)
            {
                // Why the choices are grayed: no engine of a device, or none that runs here.
                const bool unusable = tray.engines["accelerators_reason"].as_string() == "unusable";
                AppendMenuW(acc, MF_SEPARATOR, 0, nullptr);
                const char* why = unusable ? "accel_unusable" : "accel_not_installed";
                AppendMenuW(acc, MF_STRING | MF_GRAYED, 0, tr(why).c_str());
            }
            AppendMenuW(acc, MF_SEPARATOR, 0, nullptr);
            AppendMenuW(acc, MF_STRING | (on ? 0 : MF_GRAYED), CMD_MEASURE, tr("accel_measure").c_str());
            if (!cpu_engine() && setting == "auto")
            {
                AppendMenuW(acc, MF_SEPARATOR, 0, nullptr);
                AppendMenuW(acc, MF_STRING | MF_GRAYED, 0, tr("accel_cpu_only").c_str());
            }
            AppendMenuW(m, MF_POPUP | (on ? 0 : MF_GRAYED), reinterpret_cast<UINT_PTR>(acc), tr("tray_accel").c_str());
            AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
            // Offline, an update comes from a folder (Models, Import).
            const bool offline = tray.status["offline"].as_bool(false);
            AppendMenuW(m, MF_STRING | (on && !offline ? 0 : MF_GRAYED), CMD_CHECK_UPDATES, tr("tray_updates").c_str());
            const json& up = tray.status["updates"];
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
            json changes = json::object();
            switch (id)
            {
            case CMD_OPEN: open_app(); break;
            case CMD_MODELS: show_models_window(tray.client); break;
            case CMD_HOST: show_host_window(tray.client); break;
            case CMD_ACCEL_AUTO: changes["accelerator"] = "auto"; put_settings(changes); break;
            case CMD_ACCEL_NONE: changes["accelerator"] = "none"; put_settings(changes); break;
            case CMD_ACCEL_NPU: changes["accelerator"] = "npu"; put_settings(changes); break;
            case CMD_ACCEL_GPU: changes["accelerator"] = "gpu"; put_settings(changes); break;
            case CMD_ACCEL_BOTH: changes["accelerator"] = "both"; put_settings(changes); break;
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
            default: break;
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
                open_app();
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
