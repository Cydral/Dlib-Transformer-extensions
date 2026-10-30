// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    albireo-setup: the installer, the uninstaller and the updater of Albireo.

        albireo-setup                       the installation wizard, or the update wizard
                                            when Albireo is installed
        albireo-setup /silent [/dir <folder>] [/models <folder>] [/port <port>] [/offline]
                                            installation without questions; over an
                                            installation, an update that keeps everything but
                                            the options given; /offline for a computer
                                            without Internet access
        albireo-setup /update /silent       update started by the service (keeps everything)
        albireo-setup /uninstall [/silent] [/purge]
                                            removal; /purge also deletes models and settings
        albireo-setup /models-dir <folder>  moves the models folder (asked by the tray)
        albireo-setup /service start|stop|restart
        albireo-setup /lang <code>          language of the wizard: en, fr, de, it, es

    The programs travel inside this program as a ZIP payload (a resource); the copy left
    in the installation folder, the uninstaller, has no payload.

    Over an installation, the programs are replaced in their folder and everything else
    stays: the settings, the models, the certificates and their trust, the token of the
    tray, the shortcuts and the start of the icon.  A newer version in place is left as
    it is.  An offline bundle
    (tools/make_offline_bundle.py) puts the installer next to a signed catalog and a
    "models" folder, on a memory card for instance: they are imported at the first start
    of the service, for the computers without network.
*/

#include "ui.h"

#include <objbase.h>
#include <shellapi.h>

#include <algorithm>
#include <atomic>
#include <cwchar>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common/platform.h"
#include "../common/version.h"
#include "resource.h"
#include "setup_actions.h"
#include "strings.h"
#include "zip_reader.h"

namespace albireo
{

    namespace
    {
        const UINT WM_STEP = WM_APP + 1;
        const UINT WM_DONE = WM_APP + 2;

        struct payload_view
        {
            const void* data = nullptr;
            uint64_t size = 0;
        };

        payload_view payload()
        {
            payload_view p;
            HMODULE m = GetModuleHandleW(nullptr);
            HRSRC r = FindResourceW(m, MAKEINTRESOURCEW(IDR_PAYLOAD), RT_RCDATA);
            if (!r) return p;
            HGLOBAL g = LoadResource(m, r);
            if (!g) return p;
            p.data = LockResource(g);
            p.size = SizeofResource(m, r);
            if (p.size < 22) p.data = nullptr;
            return p;
        }

        std::string resource_text(int id)
        {
            HMODULE m = GetModuleHandleW(nullptr);
            HRSRC r = FindResourceW(m, MAKEINTRESOURCEW(id), RT_RCDATA);
            if (!r) return std::string();
            HGLOBAL g = LoadResource(m, r);
            if (!g) return std::string();
            return std::string(static_cast<const char*>(LockResource(g)), SizeofResource(m, r));
        }

        // Progress of an operation: thousandths, and a text naming a step or, with detail, a file.
        typedef std::function<void(int permille, const std::wstring& text, bool detail)> progress_fn;

        // Copies the programs of the payload into the installation folder.
        bool extract_payload(const std::string& dir, const progress_fn& progress, uint64_t& total, bool& reboot,
            std::string& error)
        {
            const payload_view pv = payload();
            if (!pv.data)
            {
                error = "this program has no payload";
                return false;
            }
            zip_reader zip;
            if (!zip.open(pv.data, pv.size, error)) return false;
            total = zip.total_size();
            uint64_t done = 0;
            for (const auto& e : zip.entries())
            {
                std::string rel = e.name;
                for (char& c : rel)
                    if (c == '/') c = '\\';
                if (rel.find("..") != std::string::npos || rel.empty() || rel[0] == '\\'
                    || rel.find(':') != std::string::npos)
                {
                    error = "invalid path in the payload: " + e.name;
                    return false;
                }
                const std::string target = join_path(dir, rel);
                if (e.directory())
                {
                    make_dirs(target);
                    continue;
                }
                if (progress) progress(static_cast<int>(total ? 1000 * done / total : 0),
                    tr("step_files", { { "f", file_name(target) } }), true);
                auto content = [&](const setup::chunk_writer& write) { return zip.extract(e, write, error); };
                if (!setup::write_program_file(target, content, reboot, error)) return false;
                done += e.size;
            }
            return true;
        }

        /*
            Installs the programs, or updates them when update is true: the settings, the
            certificates and their trust, the shortcuts and the start of the icon then stay
            as they are.  close_icons closes the icons of the users first; the update started
            by the service leaves them open, and each one starts again from the new program
            when it sees the new version of the service.
        */
        bool perform_install(const setup::install_options& o, bool update, bool close_icons,
            const progress_fn& progress, bool& reboot, std::string& error)
        {
            const std::string what = update ? "update" : "installation";
            setup::log(what + " of " + ALBIREO_VERSION + " in " + o.program_dir + ", port " + std::to_string(o.port));
            progress(0, tr("step_stop"), false);
            setup::stop_service(60000, error);
            error.clear();
            if (close_icons) setup::close_trays(10000);
            make_dirs(o.program_dir);
            setup::remove_old_files(o.program_dir);
            uint64_t total = 0;
            progress(0, tr("progress_title"), false);
            // The programs, then the models that come with the complete installer.
            const bool models = setup::read_builtin_overlay(executable_path()).present();
            const int programs_end = models ? 300 : 800;
            auto copying = [&](int p, const std::wstring& t, bool d) { progress(p * programs_end / 1000, t, d); };
            if (!extract_payload(o.program_dir, copying, total, reboot, error)) return false;
            if (models)
            {
                auto placing = [&](uint64_t done, uint64_t all, const std::string& f) {
                    const int p = programs_end + static_cast<int>(all ? (800 - programs_end) * done / all : 0);
                    progress(p, tr("step_models", { { "f", f } }), true);
                };
                if (!setup::install_builtin_models(o.program_dir, placing, error)) return false;
            }
            if (!setup::install_uninstaller(o.program_dir, error)) return false;
            // The size of the installation, with the models that came with it, now or before.
            const std::string models_dir = join_path(o.program_dir, "models");
            for (const auto& name : list_dir(models_dir)) total += file_size(join_path(models_dir, name));
            if (!setup::prepare_data_dir(error)) return false;
            if (!update && !setup::write_settings(o, error)) return false;
            // The models and the catalog of an offline bundle next to this program.
            setup::stage_bundle([&](const std::string& f) {
                progress(820, tr("step_files", { { "f", f } }), true);
            });
            progress(850, tr("step_certificates"), false);
            if (!setup::create_certificates(error)) return false;
            if (!update && o.trust_certificate && !setup::trust_authority(error))
            {
                setup::log("trust of the local authority: " + error);
                error.clear();
            }
            progress(900, tr("step_service"), false);
            if (!setup::register_service(o.program_dir, error)) return false;
            if (!setup::register_uninstall(o.program_dir, total, error)) return false;
            if (!update)
            {
                progress(930, tr("step_shortcuts"), false);
                setup::set_autostart(o.program_dir, o.autostart);
                setup::create_shortcuts(o.program_dir, o.desktop_shortcut);
            }
            progress(950, tr("step_start"), false);
            if (!setup::start_service(error)) return false;
            setup::wait_for_server(o.port, 30000);
            progress(1000, tr("step_done"), false);
            setup::log("done");
            return true;
        }

        void perform_uninstall(bool purge, const progress_fn& progress)
        {
            std::string error;
            const std::string dir = setup::installed_dir().empty() ? executable_dir() : setup::installed_dir();
            setup::log("uninstallation from " + dir);
            progress(100, tr("step_stop"), false);
            setup::remove_service(error);
            setup::close_trays(10000);
            progress(400, tr("uninstall_progress"), false);
            setup::set_autostart(dir, false);
            setup::remove_shortcuts();
            setup::untrust_authority();
            setup::remove_uninstall();
            const bool inside = _wcsnicmp(widen(executable_dir()).c_str(), widen(dir).c_str(), dir.size()) == 0;
            if (!inside) setup::remove_folder(dir);
            else
            {
                for (const auto& name : list_dir(dir))
                {
                    const std::string p = join_path(dir, name);
                    if (_wcsicmp(widen(p).c_str(), widen(executable_path()).c_str()) == 0) continue;
                    if (dir_exists(p)) setup::remove_folder(p);
                    else remove_file(p);
                }
                // The folder and this program go once it has ended.
                std::wstring cmd = L"cmd.exe /c ping -n 3 127.0.0.1 >nul & rd /s /q \"" + widen(dir) + L"\"";
                STARTUPINFOW si = { sizeof(si) };
                PROCESS_INFORMATION pi = {};
                const DWORD flags = CREATE_NO_WINDOW | DETACHED_PROCESS;
                if (CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, flags, nullptr, nullptr, &si, &pi))
                {
                    CloseHandle(pi.hThread);
                    CloseHandle(pi.hProcess);
                }
            }
            progress(800, tr("uninstall_progress"), false);
            if (purge) setup::remove_folder(data_dir());
            progress(1000, tr("step_done"), false);
        }

        // The wizard.
        enum page { P_WELCOME, P_LICENSE, P_FOLDERS, P_OPTIONS, P_PROGRESS, P_FINISH, P_UNINSTALL, P_REMOVED };
        enum
        {
            IDC_TITLE = 100, IDC_TEXT, IDC_LANG_LABEL, IDC_LANG, IDC_LICENSE, IDC_ACCEPT, IDC_DIR_LABEL, IDC_DIR,
            IDC_DIR_BROWSE, IDC_MODELS_LABEL, IDC_MODELS, IDC_MODELS_BROWSE, IDC_SPACE, IDC_SPACE_NOTE, IDC_AUTOSTART,
            IDC_DESKTOP, IDC_TRUST, IDC_DEFAULT_MODEL, IDC_PORT_LABEL, IDC_PORT, IDC_STEP, IDC_PROGRESS, IDC_DETAIL,
            IDC_LAUNCH, IDC_PURGE, IDC_BACK, IDC_NEXT, IDC_CANCEL, IDC_OFFLINE
        };

        struct wizard
        {
            HWND hwnd = nullptr;
            bool uninstall = false;
            page current = P_WELCOME;
            std::map<int, HWND> c;
            std::map<HWND, ui::font_kind> fonts;
            HBITMAP side = nullptr;
            int side_w = 0, side_h = 0;
            uint64_t required = 0;
            bool working = false;
            bool succeeded = false;
            bool reboot = false;
            std::string error;
            std::string previous_version;
            // Albireo is installed: the wizard updates it, or leaves a newer version alone.
            bool update = false;
            bool newer_installed = false;
            std::string installed_dir;
            // The size of the models this installer brings, 0 for the installer without models.
            uint64_t builtin_size = 0;
        } wz;

        HWND ctl(int id) { return wz.c[id]; }

        HWND add(int id, HWND h, ui::font_kind f = ui::font_kind::normal)
        {
            wz.c[id] = h;
            wz.fonts[h] = f;
            SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(ui::font(wz.hwnd, f)), TRUE);
            return h;
        }

        int px(int v) { return ui::px(wz.hwnd, v); }

        void update_space()
        {
            const std::string dir = narrow(ui::text_of(ctl(IDC_DIR)));
            const uint64_t avail = free_disk_space(dir.empty() ? std::string("C:\\") : dir);
            std::wstring s = tr("space_required", { { "s", ui::format_size(wz.required) } }) + L"\n" +
                tr("space_available", { { "s", ui::format_size(avail) } });
            if (avail < wz.required + (64ull << 20)) s += L"\n" + tr("space_insufficient");
            ui::set_text(ctl(IDC_SPACE), s);
            ui::enable(ctl(IDC_NEXT), avail >= wz.required + (64ull << 20));
        }

        void apply_texts()
        {
            SetWindowTextW(wz.hwnd, (wz.uninstall ? tr("uninstall_title") : wz.update ? tr("update_title")
                : tr("setup_title")).c_str());
            ui::set_text(ctl(IDC_LANG_LABEL), tr("lbl_language"));
            ui::set_text(ctl(IDC_ACCEPT), tr("license_accept"));
            ui::set_text(ctl(IDC_DIR_LABEL), tr("folder_programs"));
            ui::set_text(ctl(IDC_MODELS_LABEL), tr("folder_models"));
            ui::set_text(ctl(IDC_DIR_BROWSE), tr("browse"));
            ui::set_text(ctl(IDC_MODELS_BROWSE), tr("browse"));
            ui::set_text(ctl(IDC_SPACE_NOTE), tr("space_models"));
            ui::set_text(ctl(IDC_AUTOSTART), tr("opt_autostart"));
            ui::set_text(ctl(IDC_DESKTOP), tr("opt_desktop"));
            ui::set_text(ctl(IDC_TRUST), tr("opt_trust"));
            ui::set_text(ctl(IDC_DEFAULT_MODEL), tr("opt_default_model"));
            ui::set_text(ctl(IDC_OFFLINE), tr("opt_offline"));
            ui::set_text(ctl(IDC_PORT_LABEL), tr("opt_port"));
            ui::set_text(ctl(IDC_LAUNCH), tr("opt_launch"));
            ui::set_text(ctl(IDC_PURGE), tr("uninstall_data"));
            ui::set_text(ctl(IDC_BACK), tr("btn_back"));
            ui::set_text(ctl(IDC_CANCEL), tr("btn_cancel"));
        }

        void show_page(page p)
        {
            wz.current = p;
            const std::vector<int> all = { IDC_TEXT, IDC_LANG_LABEL, IDC_LANG, IDC_LICENSE, IDC_ACCEPT, IDC_DIR_LABEL,
                IDC_DIR, IDC_DIR_BROWSE, IDC_MODELS_LABEL, IDC_MODELS, IDC_MODELS_BROWSE, IDC_SPACE, IDC_SPACE_NOTE,
                IDC_AUTOSTART, IDC_DESKTOP, IDC_TRUST, IDC_DEFAULT_MODEL, IDC_OFFLINE, IDC_PORT_LABEL, IDC_PORT,
                IDC_STEP, IDC_PROGRESS, IDC_DETAIL, IDC_LAUNCH, IDC_PURGE };
            std::vector<int> visible;
            std::wstring title, text;
            switch (p)
            {
            case P_WELCOME:
            {
                title = wz.update ? tr("update_title") : tr("welcome_title");
                const std::vector<std::pair<std::string, std::string>> values = { { "v", ALBIREO_VERSION },
                    { "o", wz.previous_version }, { "d", wz.installed_dir } };
                if (wz.newer_installed) text = tr("update_newer", values);
                else if (wz.update) text = tr("update_text", values);
                else text = tr("welcome_text", values);
                if (wz.builtin_size && !wz.newer_installed)
                    text += L"\n\n" + tr("welcome_models", { { "s", ui::format_size(wz.builtin_size) } });
                visible = { IDC_TEXT, IDC_LANG_LABEL, IDC_LANG };
                break;
            }
            case P_LICENSE:
                title = tr("license_title");
                visible = { IDC_LICENSE, IDC_ACCEPT };
                break;
            case P_FOLDERS:
                title = tr("folder_title");
                visible = { IDC_DIR_LABEL, IDC_DIR, IDC_DIR_BROWSE, IDC_MODELS_LABEL, IDC_MODELS, IDC_MODELS_BROWSE,
                    IDC_SPACE, IDC_SPACE_NOTE };
                break;
            case P_OPTIONS:
                title = tr("options_title");
                // The models of the complete installer answer at once: no default model to download.
                visible = { IDC_AUTOSTART, IDC_DESKTOP, IDC_TRUST, IDC_OFFLINE, IDC_PORT_LABEL, IDC_PORT };
                if (!wz.builtin_size) visible.push_back(IDC_DEFAULT_MODEL);
                break;
            case P_PROGRESS:
                title = wz.uninstall ? tr("uninstall_progress") : wz.update ? tr("update_progress")
                    : tr("progress_title");
                visible = { IDC_STEP, IDC_PROGRESS, IDC_DETAIL };
                break;
            case P_FINISH:
                if (wz.update)
                {
                    title = wz.succeeded ? tr("update_done_title") : tr("update_title");
                    text = wz.succeeded
                        ? tr("update_done_text", { { "v", ALBIREO_VERSION }, { "o", wz.previous_version } })
                        : tr("update_failed", { { "e", wz.error } });
                }
                else
                {
                    title = wz.succeeded ? tr("finish_title") : tr("error_title");
                    text = wz.succeeded ? tr("finish_text") : tr("error_failed", { { "e", wz.error } });
                }
                if (wz.succeeded && wz.reboot) text += L"\n\n" + tr("reboot_note");
                visible = { IDC_TEXT };
                if (wz.succeeded) visible.push_back(IDC_LAUNCH);
                break;
            case P_UNINSTALL:
                title = tr("uninstall_title");
                text = tr("uninstall_text") + L"\n\n" + tr("uninstall_browser");
                visible = { IDC_TEXT, IDC_PURGE };
                break;
            case P_REMOVED:
                title = tr("uninstall_title");
                text = tr("uninstall_done");
                visible = { IDC_TEXT };
                break;
            }
            ui::set_text(ctl(IDC_TITLE), title);
            ui::set_text(ctl(IDC_TEXT), text);
            for (int id : all)
                ShowWindow(ctl(id), std::find(visible.begin(), visible.end(), id) != visible.end() ? SW_SHOW : SW_HIDE);

            const bool first = p == P_WELCOME || p == P_UNINSTALL;
            const bool last = p == P_FINISH || p == P_REMOVED;
            ShowWindow(ctl(IDC_BACK), first || last || p == P_PROGRESS ? SW_HIDE : SW_SHOW);
            ShowWindow(ctl(IDC_CANCEL), last ? SW_HIDE : SW_SHOW);
            ui::enable(ctl(IDC_CANCEL), p != P_PROGRESS);
            std::wstring next = tr("btn_next");
            if (p == P_OPTIONS) next = tr("btn_install");
            else if (p == P_LICENSE && wz.update) next = tr("btn_update");
            else if (p == P_UNINSTALL) next = tr("btn_uninstall");
            else if (last) next = tr("btn_finish");
            ui::set_text(ctl(IDC_NEXT), next);
            ui::enable(ctl(IDC_NEXT), p != P_PROGRESS && (p != P_LICENSE || ui::checked(ctl(IDC_ACCEPT)))
                && !(p == P_WELCOME && wz.newer_installed));
            if (p == P_FOLDERS) update_space();
            InvalidateRect(wz.hwnd, nullptr, TRUE);
        }

        void layout()
        {
            RECT rc;
            GetClientRect(wz.hwnd, &rc);
            const int w = rc.right, h = rc.bottom;
            const int sw = px(220);
            if (wz.side_w != sw || wz.side_h != h)
            {
                if (wz.side) DeleteObject(wz.side);
                // The image fills the height of the panel, cropped on its width.
                const SIZE s = ui::png_size(IDB_SIDE);
                const int iw = s.cy > 0 ? MulDiv(h, s.cx, s.cy) : sw;
                wz.side = ui::load_png(IDB_SIDE, std::max(iw, sw), h);
                wz.side_w = sw;
                wz.side_h = h;
            }
            const int m = px(24), x = sw + m, cw = w - x - m;
            const int bh = px(30), by = h - px(16) - bh;
            ui::place(ctl(IDC_TITLE), x, px(22), cw, px(40));
            int y = px(76);
            ui::place(ctl(IDC_TEXT), x, y, cw, px(170));
            ui::place(ctl(IDC_LANG_LABEL), x, y + px(190), px(120), px(22));
            ui::place(ctl(IDC_LANG), x + px(126), y + px(186), px(200), px(200));
            ui::place(ctl(IDC_PURGE), x, y + px(150), cw, px(40));
            ui::place(ctl(IDC_LAUNCH), x, y + px(150), cw, px(24));
            ui::place(ctl(IDC_LICENSE), x, y, cw, by - px(56) - y);
            ui::place(ctl(IDC_ACCEPT), x, by - px(44), cw, px(24));
            ui::place(ctl(IDC_DIR_LABEL), x, y, cw, px(20));
            ui::place(ctl(IDC_DIR), x, y + px(22), cw - px(110), px(26));
            ui::place(ctl(IDC_DIR_BROWSE), x + cw - px(100), y + px(21), px(100), px(28));
            ui::place(ctl(IDC_MODELS_LABEL), x, y + px(66), cw, px(20));
            ui::place(ctl(IDC_MODELS), x, y + px(88), cw - px(110), px(26));
            ui::place(ctl(IDC_MODELS_BROWSE), x + cw - px(100), y + px(87), px(100), px(28));
            ui::place(ctl(IDC_SPACE), x, y + px(136), cw, px(64));
            ui::place(ctl(IDC_SPACE_NOTE), x, y + px(204), cw, px(40));
            int oy = y;
            for (int id : { IDC_AUTOSTART, IDC_DESKTOP, IDC_TRUST, IDC_DEFAULT_MODEL, IDC_OFFLINE })
            {
                ui::place(ctl(id), x, oy, cw, px(40));
                oy += px(44);
            }
            ui::place(ctl(IDC_PORT_LABEL), x, oy + px(10), px(200), px(22));
            ui::place(ctl(IDC_PORT), x + px(206), oy + px(6), px(90), px(26));
            ui::place(ctl(IDC_STEP), x, y + px(10), cw, px(22));
            ui::place(ctl(IDC_PROGRESS), x, y + px(40), cw, px(18));
            ui::place(ctl(IDC_DETAIL), x, y + px(66), cw, px(40));
            ui::place(ctl(IDC_CANCEL), w - m - px(100), by, px(100), bh);
            ui::place(ctl(IDC_NEXT), w - m - px(208), by, px(100), bh);
            ui::place(ctl(IDC_BACK), w - m - px(316), by, px(100), bh);
        }

        void build()
        {
            add(IDC_TITLE, ui::label(wz.hwnd, L"", IDC_TITLE), ui::font_kind::title);
            add(IDC_TEXT, ui::label(wz.hwnd, L"", IDC_TEXT));
            add(IDC_LANG_LABEL, ui::label(wz.hwnd, L"", IDC_LANG_LABEL));
            HWND lang = add(IDC_LANG, ui::combo(wz.hwnd, IDC_LANG));
            const wchar_t* names[] = { L"English", L"Français", L"Deutsch", L"Italiano", L"Español" };
            const char* codes[] = { "en", "fr", "de", "it", "es" };
            for (int i = 0; i < 5; ++i)
            {
                SendMessageW(lang, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(names[i]));
                if (current_language() == codes[i]) SendMessageW(lang, CB_SETCURSEL, static_cast<WPARAM>(i), 0);
            }
            std::wstring license = widen(resource_text(IDR_LICENSE));
            std::wstring crlf;
            for (wchar_t ch : license)
            {
                if (ch == L'\n' && (crlf.empty() || crlf.back() != L'\r')) crlf += L'\r';
                crlf += ch;
            }
            const DWORD reader = ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL;
            add(IDC_LICENSE, ui::edit(wz.hwnd, crlf, IDC_LICENSE, reader), ui::font_kind::note);
            add(IDC_ACCEPT, ui::checkbox(wz.hwnd, L"", IDC_ACCEPT, false));

            // Over an installation, the fields start from its current options.
            const setup::install_options current = setup::installed_options();
            add(IDC_DIR_LABEL, ui::label(wz.hwnd, L"", IDC_DIR_LABEL), ui::font_kind::bold);
            add(IDC_DIR, ui::edit(wz.hwnd, widen(current.program_dir), IDC_DIR));
            add(IDC_DIR_BROWSE, ui::button(wz.hwnd, L"", IDC_DIR_BROWSE));
            if (!wz.previous_version.empty())
            {
                // An update replaces the programs in their folder.
                SendMessageW(ctl(IDC_DIR), EM_SETREADONLY, TRUE, 0);
                ui::enable(ctl(IDC_DIR_BROWSE), false);
            }
            add(IDC_MODELS_LABEL, ui::label(wz.hwnd, L"", IDC_MODELS_LABEL), ui::font_kind::bold);
            add(IDC_MODELS, ui::edit(wz.hwnd, widen(current.models_dir), IDC_MODELS));
            add(IDC_MODELS_BROWSE, ui::button(wz.hwnd, L"", IDC_MODELS_BROWSE));
            add(IDC_SPACE, ui::label(wz.hwnd, L"", IDC_SPACE));
            add(IDC_SPACE_NOTE, ui::label(wz.hwnd, L"", IDC_SPACE_NOTE), ui::font_kind::note);
            add(IDC_AUTOSTART, ui::checkbox(wz.hwnd, L"", IDC_AUTOSTART, current.autostart));
            add(IDC_DESKTOP, ui::checkbox(wz.hwnd, L"", IDC_DESKTOP, current.desktop_shortcut));
            add(IDC_TRUST, ui::checkbox(wz.hwnd, L"", IDC_TRUST, current.trust_certificate));
            add(IDC_DEFAULT_MODEL, ui::checkbox(wz.hwnd, L"", IDC_DEFAULT_MODEL, current.default_model));
            add(IDC_OFFLINE, ui::checkbox(wz.hwnd, L"", IDC_OFFLINE, current.offline));
            // Offline, no model is downloaded.
            ui::enable(ctl(IDC_DEFAULT_MODEL), !current.offline);
            add(IDC_PORT_LABEL, ui::label(wz.hwnd, L"", IDC_PORT_LABEL));
            add(IDC_PORT, ui::edit(wz.hwnd, std::to_wstring(current.port), IDC_PORT, ES_NUMBER));
            add(IDC_STEP, ui::label(wz.hwnd, L"", IDC_STEP), ui::font_kind::bold);
            add(IDC_PROGRESS, ui::progress(wz.hwnd, IDC_PROGRESS));
            add(IDC_DETAIL, ui::label(wz.hwnd, L"", IDC_DETAIL, SS_PATHELLIPSIS), ui::font_kind::note);
            add(IDC_LAUNCH, ui::checkbox(wz.hwnd, L"", IDC_LAUNCH, true));
            add(IDC_PURGE, ui::checkbox(wz.hwnd, L"", IDC_PURGE, false));
            add(IDC_BACK, ui::button(wz.hwnd, L"", IDC_BACK));
            add(IDC_NEXT, ui::button(wz.hwnd, L"", IDC_NEXT, BS_DEFPUSHBUTTON));
            add(IDC_CANCEL, ui::button(wz.hwnd, L"", IDC_CANCEL));
            apply_texts();
            layout();
            show_page(wz.uninstall ? P_UNINSTALL : P_WELCOME);
        }

        void start_work()
        {
            wz.working = true;
            show_page(P_PROGRESS);
            HWND target = wz.hwnd;
            auto progress = [target](int permille, const std::wstring& text, bool detail) {
                std::wstring* t = new std::wstring(text);
                if (!PostMessageW(target, WM_STEP, static_cast<WPARAM>(permille) | (detail ? 0x100000 : 0),
                        reinterpret_cast<LPARAM>(t)))
                    delete t;
            };
            if (wz.uninstall)
            {
                const bool purge = ui::checked(ctl(IDC_PURGE));
                std::thread([target, purge, progress]() {
                    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
                    perform_uninstall(purge, progress);
                    CoUninitialize();
                    PostMessageW(target, WM_DONE, 1, 0);
                }).detach();
                return;
            }
            setup::install_options o = setup::installed_options();
            const bool update = wz.update;
            if (!update)
            {
                o.program_dir = narrow(ui::text_of(ctl(IDC_DIR)));
                o.models_dir = narrow(ui::text_of(ctl(IDC_MODELS)));
                o.autostart = ui::checked(ctl(IDC_AUTOSTART));
                o.desktop_shortcut = ui::checked(ctl(IDC_DESKTOP));
                o.trust_certificate = ui::checked(ctl(IDC_TRUST));
                o.offline = ui::checked(ctl(IDC_OFFLINE));
                o.default_model = ui::checked(ctl(IDC_DEFAULT_MODEL)) && !o.offline;
                const long port = std::wcstol(ui::text_of(ctl(IDC_PORT)).c_str(), nullptr, 10);
                if (port >= 1024 && port <= 65535) o.port = static_cast<int>(port);
            }
            std::thread([target, o, update, progress]() {
                CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
                std::string* error = new std::string();
                bool reboot = false;
                const bool ok = perform_install(o, update, true, progress, reboot, *error);
                if (!ok) setup::log("failed: " + *error);
                CoUninitialize();
                const WPARAM result = (ok ? 1 : 0) | (reboot ? 2 : 0);
                if (!PostMessageW(target, WM_DONE, result, reinterpret_cast<LPARAM>(error))) delete error;
            }).detach();
        }

        void finish()
        {
            if (!wz.uninstall && wz.succeeded)
            {
                const std::string dir = narrow(ui::text_of(ctl(IDC_DIR)));
                const std::string tray = join_path(dir, "albireo-tray.exe");
                setup::launch_unelevated(tray, ui::checked(ctl(IDC_LAUNCH)) ? "--open" : "");
            }
            DestroyWindow(wz.hwnd);
        }

        LRESULT CALLBACK proc(HWND w, UINT msg, WPARAM wp, LPARAM lp)
        {
            switch (msg)
            {
            case WM_CREATE:
                wz.hwnd = w;
                build();
                return 0;
            case WM_SIZE:
                layout();
                return 0;
            case WM_DPICHANGED:
            {
                for (const auto& kv : wz.fonts)
                    SendMessageW(kv.first, WM_SETFONT, reinterpret_cast<WPARAM>(ui::font(w, kv.second)), TRUE);
                const RECT* r = reinterpret_cast<RECT*>(lp);
                SetWindowPos(w, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER);
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
                if (wz.side)
                {
                    // The center of the image, on the width of the panel.
                    BITMAP b;
                    GetObjectW(wz.side, sizeof(b), &b);
                    HDC mem = CreateCompatibleDC(dc);
                    HGDIOBJ old = SelectObject(mem, wz.side);
                    const int sx = std::max(0, (static_cast<int>(b.bmWidth) - wz.side_w) / 2);
                    BitBlt(dc, 0, 0, wz.side_w, b.bmHeight, mem, sx, 0, SRCCOPY);
                    SelectObject(mem, old);
                    DeleteDC(mem);
                }
                RECT rc;
                GetClientRect(w, &rc);
                RECT line = { wz.side_w, rc.bottom - px(62), rc.right, rc.bottom - px(61) };
                HBRUSH b = CreateSolidBrush(RGB(226, 229, 238));
                FillRect(dc, &line, b);
                DeleteObject(b);
                EndPaint(w, &ps);
                return 0;
            }
            case WM_CTLCOLORSTATIC:
            case WM_CTLCOLORBTN:
            {
                HDC dc = reinterpret_cast<HDC>(wp);
                SetBkColor(dc, ui::color_background());
                SetTextColor(dc, reinterpret_cast<HWND>(lp) == ctl(IDC_TITLE) ? RGB(28, 36, 82) : ui::color_text());
                if (reinterpret_cast<HWND>(lp) == ctl(IDC_LICENSE))
                    return reinterpret_cast<LRESULT>(GetStockObject(WHITE_BRUSH));
                return reinterpret_cast<LRESULT>(ui::background_brush());
            }
            case WM_COMMAND:
                switch (LOWORD(wp))
                {
                case IDC_LANG:
                    if (HIWORD(wp) == CBN_SELCHANGE)
                    {
                        const char* codes[] = { "en", "fr", "de", "it", "es" };
                        const LRESULT i = SendMessageW(ctl(IDC_LANG), CB_GETCURSEL, 0, 0);
                        if (i >= 0 && i < 5) set_language(codes[i]);
                        apply_texts();
                        show_page(wz.current);
                    }
                    return 0;
                case IDC_ACCEPT:
                    ui::enable(ctl(IDC_NEXT), ui::checked(ctl(IDC_ACCEPT)));
                    return 0;
                case IDC_OFFLINE:
                    ui::enable(ctl(IDC_DEFAULT_MODEL), !ui::checked(ctl(IDC_OFFLINE)));
                    return 0;
                case IDC_DIR:
                    if (HIWORD(wp) == EN_CHANGE && wz.current == P_FOLDERS) update_space();
                    return 0;
                case IDC_DIR_BROWSE:
                case IDC_MODELS_BROWSE:
                {
                    const int target = LOWORD(wp) == IDC_DIR_BROWSE ? IDC_DIR : IDC_MODELS;
                    std::wstring d = ui::choose_folder(w, ui::text_of(ctl(target)));
                    if (!d.empty())
                    {
                        // A program folder chosen by the user gets its own Albireo subfolder.
                        if (target == IDC_DIR && file_name(narrow(d)) != "Albireo")
                            d = widen(join_path(narrow(d), "Albireo"));
                        ui::set_text(ctl(target), d);
                        update_space();
                    }
                    return 0;
                }
                case IDC_BACK:
                    if (wz.current > P_WELCOME && wz.current <= P_OPTIONS) show_page(static_cast<page>(wz.current - 1));
                    return 0;
                case IDC_NEXT:
                    switch (wz.current)
                    {
                    case P_WELCOME: show_page(P_LICENSE); break;
                    // An update keeps the folders and the options of the installation.
                    case P_LICENSE:
                        if (wz.update) start_work();
                        else show_page(P_FOLDERS);
                        break;
                    case P_FOLDERS: show_page(P_OPTIONS); break;
                    case P_OPTIONS: start_work(); break;
                    case P_UNINSTALL: start_work(); break;
                    case P_FINISH:
                    case P_REMOVED: finish(); break;
                    default: break;
                    }
                    return 0;
                case IDC_CANCEL:
                    if (!wz.working && ui::message(w, tr("cancel_confirm"), MB_YESNO | MB_ICONQUESTION) == IDYES)
                        DestroyWindow(w);
                    return 0;
                default:
                    return 0;
                }
            case WM_STEP:
            {
                std::wstring* t = reinterpret_cast<std::wstring*>(lp);
                SendMessageW(ctl(IDC_PROGRESS), PBM_SETPOS, wp & 0xFFFFF, 0);
                if (wp & 0x100000) ui::set_text(ctl(IDC_DETAIL), *t);
                else
                {
                    ui::set_text(ctl(IDC_STEP), *t);
                    ui::set_text(ctl(IDC_DETAIL), L"");
                }
                delete t;
                return 0;
            }
            case WM_DONE:
            {
                std::string* e = reinterpret_cast<std::string*>(lp);
                wz.working = false;
                wz.succeeded = (wp & 1) != 0;
                wz.reboot = (wp & 2) != 0;
                if (e)
                {
                    wz.error = *e;
                    delete e;
                }
                show_page(wz.uninstall ? P_REMOVED : P_FINISH);
                return 0;
            }
            case WM_CLOSE:
                if (wz.working) return 0;
                if (wz.current == P_FINISH || wz.current == P_REMOVED) DestroyWindow(w);
                else if (ui::message(w, tr("cancel_confirm"), MB_YESNO | MB_ICONQUESTION) == IDYES) DestroyWindow(w);
                return 0;
            case WM_DESTROY:
                PostQuitMessage(0);
                return 0;
            default:
                return DefWindowProcW(w, msg, wp, lp);
            }
        }

        /*
            The installation in place: its version and its folder, when its programs are
            there.  A registration whose folder lost its programs is installed again.
        */
        bool installed_in_place(std::string& version, std::string& dir)
        {
            version = setup::installed_version();
            dir = setup::installed_dir();
            return !version.empty() && !dir.empty() && file_exists(join_path(dir, "albireo-service.exe"));
        }

        int run_wizard(HINSTANCE instance, bool uninstall)
        {
            wz.uninstall = uninstall;
            if (!uninstall && installed_in_place(wz.previous_version, wz.installed_dir))
            {
                wz.update = true;
                wz.newer_installed = compare_versions(wz.previous_version, ALBIREO_VERSION) > 0;
            }
            else
            {
                wz.previous_version.clear();
            }
            const payload_view pv = payload();
            if (pv.data)
            {
                zip_reader z;
                std::string e;
                if (z.open(pv.data, pv.size, e)) wz.required = z.total_size();
            }
            wz.builtin_size = setup::read_builtin_overlay(executable_path()).total;
            wz.required += wz.builtin_size;
            WNDCLASSEXW wc = { sizeof(wc) };
            wc.lpfnWndProc = proc;
            wc.hInstance = instance;
            wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
            wc.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(IDI_APP));
            wc.lpszClassName = L"AlbireoSetupWindow";
            RegisterClassExW(&wc);
            const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
            const int dpi = static_cast<int>(ui::dpi_of(nullptr));
            RECT r = { 0, 0, MulDiv(780, dpi, 96), MulDiv(520, dpi, 96) };
            AdjustWindowRectEx(&r, style, FALSE, WS_EX_CONTROLPARENT);
            RECT work;
            SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
            const int ww = r.right - r.left, wh = r.bottom - r.top;
            const int x = work.left + (work.right - work.left - ww) / 2;
            const int y = work.top + (work.bottom - work.top - wh) / 2;
            HWND w = CreateWindowExW(WS_EX_CONTROLPARENT, wc.lpszClassName, L"Albireo", style, x, y, ww, wh,
                nullptr, nullptr, instance, nullptr);
            ShowWindow(w, SW_SHOWNORMAL);
            SetForegroundWindow(w);
            MSG msg;
            while (GetMessageW(&msg, nullptr, 0, 0) > 0)
            {
                if (IsDialogMessageW(w, &msg)) continue;
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            return wz.succeeded || uninstall ? 0 : 1;
        }

        std::wstring arg_after(const std::vector<std::wstring>& args, const wchar_t* name)
        {
            for (size_t i = 0; i + 1 < args.size(); ++i)
                if (_wcsicmp(args[i].c_str(), name) == 0) return args[i + 1];
            return std::wstring();
        }

        bool has_arg(const std::vector<std::wstring>& args, const wchar_t* name)
        {
            for (const auto& a : args)
                if (_wcsicmp(a.c_str(), name) == 0) return true;
            return false;
        }
    }

}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int)
{
    using namespace albireo;
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::vector<std::wstring> args;
    for (int i = 1; argv && i < argc; ++i) args.push_back(argv[i]);
    if (argv) LocalFree(argv);

    const std::wstring lang = arg_after(args, L"/lang");
    set_language(lang.empty() ? "auto" : narrow(lang));
    ui::init();
    const bool silent = has_arg(args, L"/silent");
    std::string error;

    const std::wstring svc = arg_after(args, L"/service");
    if (!svc.empty())
    {
        bool ok = true;
        if (svc == L"stop" || svc == L"restart") ok = setup::stop_service(60000, error);
        if (svc == L"start" || svc == L"restart") ok = setup::start_service(error) && ok;
        return ok ? 0 : 1;
    }

    const std::wstring models = arg_after(args, L"/models-dir");
    if (!models.empty())
    {
        // A new models folder: the service records it at its next start.
        const std::string dir = narrow(models);
        make_dirs(dir);
        const std::string path = join_path(data_dir(), "settings.json");
        std::string text;
        json s = read_file(path, text, 1u << 20) ? json::parse_or_null(text) : json::object();
        if (!s.is_object()) s = json::object();
        s["models_dir"] = dir == setup::default_models_dir() ? std::string() : dir;
        write_file_atomic(path, s.dump_formatted(2) + "\n");
        setup::log("models folder: " + dir);
        setup::stop_service(60000, error);
        return setup::start_service(error) ? 0 : 1;
    }

    if (has_arg(args, L"/update"))
    {
        // Started by the service, which stops for it; everything else is kept.
        const setup::install_options o = setup::installed_options();
        bool reboot = false;
        const bool ok = perform_install(o, true, false, [](int, const std::wstring&, bool) {}, reboot, error);
        if (!ok)
        {
            setup::log("update failed: " + error);
            // The service starts again on its previous programs, or what replaced them.
            std::string e2;
            setup::start_service(e2);
        }
        return ok ? 0 : 1;
    }

    if (has_arg(args, L"/uninstall") || !payload().data)
    {
        if (silent)
        {
            perform_uninstall(has_arg(args, L"/purge"), [](int, const std::wstring&, bool) {});
            return 0;
        }
        return run_wizard(instance, true);
    }

    if (silent)
    {
        // The options not given keep the values of the installation in place.
        setup::install_options o = setup::installed_options();
        std::string version, installed;
        const bool update = installed_in_place(version, installed);
        if (update && compare_versions(version, ALBIREO_VERSION) > 0)
        {
            setup::log("Albireo " + version + " is installed, newer than " + ALBIREO_VERSION + ": nothing done");
            return 0;
        }
        bool options = false;
        const std::wstring dir = arg_after(args, L"/dir");
        // An update replaces the programs in their folder.
        if (!dir.empty() && !update) o.program_dir = narrow(dir);
        const std::wstring m = arg_after(args, L"/models");
        if (!m.empty())
        {
            o.models_dir = narrow(m);
            options = true;
        }
        if (has_arg(args, L"/offline"))
        {
            o.offline = true;
            o.default_model = false;
            options = true;
        }
        const std::wstring port = arg_after(args, L"/port");
        if (!port.empty())
        {
            const long p = std::wcstol(port.c_str(), nullptr, 10);
            if (p < 1024 || p > 65535)
            {
                setup::log("invalid port: " + narrow(port));
                return 1;
            }
            o.port = static_cast<int>(p);
            options = true;
        }
        // Over an installation, the options given join its settings, which stay otherwise.
        if (update && options && !setup::write_settings(o, error))
        {
            setup::log("failed: " + error);
            return 1;
        }
        bool reboot = false;
        const bool ok = perform_install(o, update, true, [](int, const std::wstring&, bool) {}, reboot, error);
        if (!ok) setup::log("failed: " + error);
        return ok ? 0 : 1;
    }
    return run_wizard(instance, false);
}
