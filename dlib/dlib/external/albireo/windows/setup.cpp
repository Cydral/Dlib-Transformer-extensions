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
    in the installation folder, the uninstaller, has no payload: it uninstalls, started
    with /uninstall or without options, and does nothing else.

    Over an installation, the programs are replaced in their folder and everything else
    stays: the settings, the models, the certificates and their trust, the token of the
    tray, the shortcuts and the start of the icon.  The new files are written beside the
    files in place, then take their places together; when a step fails, the previous
    files come back and the service starts again.  A newer version in place is left as
    it is.  An offline bundle
    (tools/make_offline_bundle.py) puts the installer next to a signed catalog and a
    "models" folder, on a memory card for instance: the models the catalog lists are
    imported at the first start of the service, for the computers without network.
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
        // The space of the drive is read again once the folder typed stays the same for a moment.
        const UINT_PTR TIMER_SPACE = 1;

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

        // The bytes of the programs of the payload, once extracted.
        uint64_t payload_size()
        {
            const payload_view pv = payload();
            zip_reader zip;
            std::string error;
            return pv.data && zip.open(pv.data, pv.size, error) ? zip.total_size() : 0;
        }

        // Writes the programs of the payload into the set of the new files of the installation folder.
        bool stage_payload(const std::string& dir, setup::staged_files& files, const progress_fn& progress,
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
            const uint64_t total = zip.total_size();
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
                if (!files.write(target, content, error)) return false;
                done += e.size;
            }
            return true;
        }

        // How an installation went, for the wizard: its error, and what failed without stopping it.
        struct install_report
        {
            std::string error;
            std::string notice;
            bool icons_closed = false;   // icons of this session closed for the copy
        };

        // The text of a problem of a folder (setup::folder_problem()), in the language of the wizard.
        std::string folder_text(const std::string& problem, const std::string& dir)
        {
            return narrow(tr(problem.c_str(), { { "d", dir } }));
        }

        /*
            The folder of the programs, and the models folder chosen for a new installation,
            made so that only SYSTEM and the administrators change them, before anything
            stops.  The folder of an installation in place keeps its permissions when they
            cannot be made so, which the log says; the service secures its models folder
            itself at its start.
        */
        bool prepare_folders(const setup::install_options& o, bool update, std::string& error)
        {
            const std::string problem = setup::folder_problem(o.program_dir, true);
            if (update)
            {
                std::string e;
                if (!problem.empty()) e = folder_text(problem, o.program_dir);
                else secure_folder(o.program_dir, e);
                if (!e.empty()) setup::log("the folder of the programs keeps its permissions: " + e);
                return true;
            }
            if (!problem.empty())
            {
                error = folder_text(problem, o.program_dir);
                return false;
            }
            if (!secure_folder(o.program_dir, error)) return false;
            if (o.models_dir.empty() || o.models_dir == setup::default_models_dir()) return true;
            const std::string models_problem = setup::folder_problem(o.models_dir, false);
            if (!models_problem.empty())
            {
                error = folder_text(models_problem, o.models_dir);
                return false;
            }
            return secure_folder(o.models_dir, error);
        }

        // The programs, the models that come with the complete installer and the uninstaller, staged.
        bool stage_programs(const std::string& program_dir, bool models, setup::staged_files& files,
            const progress_fn& progress, std::string& error)
        {
            setup::remove_old_files(program_dir);
            progress(0, tr("progress_title"), false);
            const int programs_end = models ? 300 : 800;
            auto copying = [&](int p, const std::wstring& t, bool d) { progress(p * programs_end / 1000, t, d); };
            if (!stage_payload(program_dir, files, copying, error)) return false;
            if (models)
            {
                auto placing = [&](uint64_t done, uint64_t all, const std::string& f) {
                    const int p = programs_end + static_cast<int>(all ? (800 - programs_end) * done / all : 0);
                    progress(p, tr("step_models", { { "f", f } }), true);
                };
                if (!setup::stage_builtin_models(program_dir, files, placing, error)) return false;
            }
            return setup::stage_uninstaller(program_dir, files, error);
        }

        // The steps after the copy: settings, offline bundle, certificates, service, entry of the programs.
        bool configure(const setup::install_options& o, bool update, const progress_fn& progress,
            install_report& report)
        {
            std::string& error = report.error;
            if (!update && !setup::write_settings(o, error)) return false;
            // The models and the catalog of an offline bundle next to this program.
            auto copying = [&](uint64_t done, uint64_t total, const std::string& f) {
                progress(800 + static_cast<int>(total ? 50 * done / total : 0), tr("step_files", { { "f", f } }), true);
            };
            if (!setup::stage_bundle(copying, report.notice)) setup::log("offline bundle: " + report.notice);
            progress(850, tr("step_certificates"), false);
            bool renewed = false;
            if (!setup::create_certificates(!update, renewed, error)) return false;
            // An update trusts the authority made anew in place of one that Windows trusted.
            const bool trust = update ? renewed && setup::authority_trusted() : o.trust_certificate;
            if (trust && !setup::trust_authority(error))
            {
                setup::log("trust of the local authority: " + error);
                error.clear();
            }
            progress(900, tr("step_service"), false);
            if (!setup::register_service(o.program_dir, error)) return false;
            // The size of the installation, with the models that came with it, now or before.
            uint64_t total = payload_size();
            const std::string models_dir = join_path(o.program_dir, "models");
            for (const auto& name : list_dir(models_dir)) total += file_size(join_path(models_dir, name));
            if (!setup::register_uninstall(o.program_dir, total, error)) return false;
            if (!update)
            {
                progress(930, tr("step_shortcuts"), false);
                setup::set_autostart(o.program_dir, o.autostart);
                setup::create_shortcuts(o.program_dir, o.desktop_shortcut);
            }
            return true;
        }

        /*
            Installs the programs, or updates them when update is true: the settings, the
            certificates and their trust, the shortcuts and the start of the icon then stay
            as they are.  close_icons closes the icons of the users first; the update started
            by the service leaves them open, and each one starts again from the new program
            when it sees the new version of the service.  The free space is checked before
            anything stops; when a later step fails, the previous programs come back and the
            service starts again if it was running.
        */
        bool perform_install(const setup::install_options& o, bool update, bool close_icons,
            const progress_fn& progress, install_report& report)
        {
            std::string& error = report.error;
            if (!setup::prepare_data_dir(error)) return false;
            const std::string what = update ? "update" : "installation";
            setup::log(what + " of " + ALBIREO_VERSION + " in " + o.program_dir + ", port " + std::to_string(o.port));
            if (!prepare_folders(o, update, error)) return false;
            // The new files stay beside those they replace until the end of the copy.
            const setup::builtin_overlay overlay = setup::read_builtin_overlay(executable_path());
            const uint64_t program = overlay.present() ? overlay.start : file_size(executable_path());
            const uint64_t needed = payload_size() + setup::builtin_bytes_to_copy(o.program_dir) + program
                + (64ull << 20);
            const uint64_t available = free_disk_space(o.program_dir);
            if (available < needed)
            {
                error = "not enough free space for " + o.program_dir + ": " + std::to_string(needed >> 20)
                    + " MB needed, " + std::to_string(available >> 20) + " MB available";
                return false;
            }

            progress(0, tr("step_stop"), false);
            const bool was_running = setup::service_running();
            const bool had_service = setup::service_exists();
            setup::suspend_recovery();
            setup::stop_service(60000, error);
            error.clear();
            if (close_icons) report.icons_closed = setup::close_trays(10000);
            setup::staged_files files;
            const bool copied = stage_programs(o.program_dir, overlay.present(), files, progress, error)
                && files.commit(error);
            if (!copied || !configure(o, update, progress, report))
            {
                // The previous programs come back, and the service as it was.
                setup::log("the previous programs are put back: " + error);
                files.rollback();
                std::string e;
                if (!had_service) setup::remove_service(e);
                setup::restore_recovery();
                if (was_running) setup::start_service(e);
                return false;
            }
            files.finish();
            setup::write_file_list(o.program_dir, files.targets());
            progress(950, tr("step_start"), false);
            // A service started meanwhile runs the previous programs: it stops first.
            setup::stop_service(60000, error);
            error.clear();
            if (!setup::start_service(error)) return false;
            setup::wait_for_server(o.port, 30000);
            progress(1000, tr("step_done"), false);
            setup::log("done");
            return true;
        }

        // Removes the installation registered on this computer; false when there is none.
        bool perform_uninstall(bool purge, const progress_fn& progress, std::string& error)
        {
            const std::string dir = setup::installed_dir();
            if (dir.empty())
            {
                error = narrow(tr("not_installed"));
                return false;
            }
            std::string e;
            if (dir_exists(data_dir())) setup::prepare_data_dir(e);
            setup::log("uninstallation from " + dir);
            progress(100, tr("step_stop"), false);
            setup::remove_service(e);
            setup::close_trays(10000);
            progress(400, tr("uninstall_progress"), false);
            setup::set_autostart(dir, false);
            setup::remove_shortcuts();
            setup::untrust_authority();
            setup::remove_uninstall();
            // The files of the installation only, and its folder once empty.
            setup::remove_installation(dir);
            progress(800, tr("uninstall_progress"), false);
            if (purge) setup::remove_folder(data_dir());
            progress(1000, tr("step_done"), false);
            return true;
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
            std::string error;
            // What failed without stopping the installation: the models of an offline bundle.
            std::string notice;
            // The icons of this session were closed for the copy.
            bool icons_closed = false;
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

        // Whether the folders typed or chosen can serve; otherwise a message says why.
        bool folders_accepted()
        {
            const std::string programs = narrow(ui::text_of(ctl(IDC_DIR)));
            const std::string models = narrow(ui::text_of(ctl(IDC_MODELS)));
            std::string dir = programs;
            std::string problem = setup::folder_problem(programs, true);
            if (problem.empty() && !models.empty() && models != setup::default_models_dir())
            {
                dir = models;
                problem = setup::folder_problem(models, false);
            }
            if (problem.empty()) return true;
            ui::message(wz.hwnd, tr(problem.c_str(), { { "d", dir } }), MB_OK | MB_ICONWARNING);
            return false;
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
                if (wz.succeeded && !wz.notice.empty()) text += L"\n\n" + tr("bundle_notice", { { "e", wz.notice } });
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
                text = wz.succeeded ? tr("uninstall_done") : widen(wz.error);
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
                    install_report* r = new install_report();
                    const bool ok = perform_uninstall(purge, progress, r->error);
                    CoUninitialize();
                    if (!PostMessageW(target, WM_DONE, ok ? 1 : 0, reinterpret_cast<LPARAM>(r))) delete r;
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
                install_report* r = new install_report();
                const bool ok = perform_install(o, update, true, progress, *r);
                if (!ok) setup::log("failed: " + r->error);
                CoUninitialize();
                if (!PostMessageW(target, WM_DONE, ok ? 1 : 0, reinterpret_cast<LPARAM>(r))) delete r;
            }).detach();
        }

        void finish()
        {
            // The icon starts with the rights of its user: after an installation, and after an
            // update that failed once it had closed the icons.
            if (!wz.uninstall && (wz.succeeded || wz.icons_closed))
            {
                const std::string dir = narrow(ui::text_of(ctl(IDC_DIR)));
                const std::string tray = join_path(dir, "albireo-tray.exe");
                const bool open = wz.succeeded && ui::checked(ctl(IDC_LAUNCH));
                if (file_exists(tray)) setup::launch_unelevated(tray, open ? "--open" : "");
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
                    if (HIWORD(wp) == EN_CHANGE && wz.current == P_FOLDERS) SetTimer(w, TIMER_SPACE, 400, nullptr);
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
                    case P_FOLDERS:
                        if (folders_accepted()) show_page(P_OPTIONS);
                        break;
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
                install_report* r = reinterpret_cast<install_report*>(lp);
                wz.working = false;
                wz.succeeded = (wp & 1) != 0;
                if (r)
                {
                    wz.error = r->error;
                    wz.notice = r->notice;
                    wz.icons_closed = r->icons_closed;
                    delete r;
                }
                show_page(wz.uninstall ? P_REMOVED : P_FINISH);
                return 0;
            }
            case WM_TIMER:
                if (wp == TIMER_SPACE)
                {
                    KillTimer(w, TIMER_SPACE);
                    if (wz.current == P_FOLDERS) update_space();
                }
                return 0;
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
            // The uninstaller removes the installation the registry names, and nothing else.
            if (uninstall && setup::installed_dir().empty())
            {
                ui::message(nullptr, tr("not_installed"), MB_OK | MB_ICONINFORMATION);
                return 1;
            }
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
    // The libraries loaded from here on come from System32, never from the folder of this
    // program, which may be a folder of downloads.
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32);
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
        /*
            A new models folder, which only SYSTEM and the administrators may change, since
            the service writes there with its rights: the service records it at its next
            start.  A folder that cannot serve is explained, and nothing changes.
        */
        const std::string dir = narrow(models);
        const bool custom = dir != setup::default_models_dir();
        std::string reason;
        if (!setup::prepare_data_dir(error))
        {
            reason = error;
        }
        else if (custom)
        {
            const std::string problem = setup::folder_problem(dir, false);
            if (!problem.empty()) reason = folder_text(problem, dir);
            else if (!secure_folder(dir, error)) reason = error;
        }
        if (!reason.empty())
        {
            setup::log("models folder " + dir + " refused: " + reason);
            if (!silent) ui::message(nullptr, widen(reason), MB_OK | MB_ICONWARNING);
            return 1;
        }
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
        install_report report;
        const bool ok = perform_install(o, true, false, [](int, const std::wstring&, bool) {}, report);
        if (!ok)
        {
            setup::log("update failed: " + report.error);
            // The service starts again on its previous programs.
            std::string e2;
            setup::start_service(e2);
        }
        return ok ? 0 : 1;
    }

    /*
        The uninstaller, which has no payload, uninstalls when asked to, or started without
        options.  The copy kept in the folder of the installation is the uninstaller, also
        when its payload could not be taken out of it.
    */
    const std::string install_dir = setup::installed_dir();
    const std::wstring uninstaller = widen(join_path(install_dir, "albireo-setup.exe"));
    const bool no_payload = !payload().data
        || (!install_dir.empty() && _wcsicmp(widen(executable_path()).c_str(), uninstaller.c_str()) == 0);
    const size_t wizard_args = lang.empty() ? 0 : 2;
    if (has_arg(args, L"/uninstall") || (no_payload && args.size() == wizard_args))
    {
        int code = 0;
        if (silent)
        {
            if (!perform_uninstall(has_arg(args, L"/purge"), [](int, const std::wstring&, bool) {}, error))
            {
                setup::log("uninstallation: " + error);
                code = 1;
            }
        }
        else
        {
            code = run_wizard(instance, true);
        }
        setup::finish_removal();
        return code;
    }
    if (no_payload)
    {
        // Any other command does nothing here: with /silent it would have removed Albireo.
        if (!silent) ui::message(nullptr, tr("no_payload"), MB_OK | MB_ICONINFORMATION);
        return 1;
    }

    if (silent)
    {
        setup::prepare_data_dir(error);
        error.clear();
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
        // The folders given must serve before anything changes.
        std::string folder = o.program_dir;
        std::string problem = update ? std::string() : setup::folder_problem(folder, true);
        if (problem.empty() && !m.empty() && o.models_dir != setup::default_models_dir())
        {
            folder = o.models_dir;
            problem = setup::folder_problem(folder, false);
        }
        if (!problem.empty())
        {
            setup::log("failed: " + folder_text(problem, folder));
            return 1;
        }
        // Over an installation, the options given join its settings, which stay otherwise.
        if (update && options && !setup::write_settings(o, error))
        {
            setup::log("failed: " + error);
            return 1;
        }
        install_report report;
        const bool ok = perform_install(o, update, true, [](int, const std::wstring&, bool) {}, report);
        if (!ok) setup::log("failed: " + report.error);
        // The icons closed for the copy start again, with the rights of their user.
        const std::string tray = join_path(o.program_dir, "albireo-tray.exe");
        if (report.icons_closed && file_exists(tray)) setup::launch_unelevated(tray, "");
        return ok ? 0 : 1;
    }
    return run_wizard(instance, false);
}
