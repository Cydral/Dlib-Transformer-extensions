// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    pseudalis-setup: the installer, the uninstaller and the updater of Pseudalis.

        pseudalis-setup                     the installation wizard, or the update wizard
                                            when Pseudalis is installed
        pseudalis-setup /silent [/dir <folder>] [/data-dir <folder>] [/drive <letter>|none]
                        [/restore-drive <letter>|none] [/port <port>] [/no-context-menu]
                                            installation without questions; over an
                                            installation, an update that keeps everything but
                                            the options given
        pseudalis-setup /update /silent     update started by the service (keeps everything)
        pseudalis-setup /uninstall [/silent] [/purge]
                                            removal of the programs; the key, the originals
                                            and the settings stay for a later installation,
                                            unless /purge removes them too (/keep-data,
                                            accepted, is the default)
        pseudalis-setup /service start|stop|restart
        pseudalis-setup /lang <code>        language of the wizard: en, fr, de, it, es

    The programs travel inside this program as a ZIP payload (a resource).  The drives need
    no driver: the service shows two folders of its data folder under their letters.  The
    copy left in the installation folder, the uninstaller, has no payload: it uninstalls,
    started with /uninstall or without options, and does nothing else.

    Over an installation, the programs are replaced in their folder and everything else
    stays: the settings, the key and the originals, the certificates and their trust, the
    token of the tray, the shortcuts, the commands of the Explorer and the start of the
    icon.  The new files are written beside the files in place, then take their places
    together; the previous files go once the new service answers on its port, and when a
    step fails, the start of the new service included, they come back and the service
    starts again.  A newer version in place is left as it is.  The WinFsp that an earlier
    version installed for its drive leaves once this version is in place, unless another
    program registered a file system with it since (setup_actions.h).
*/

#include "ui.h"

#include <objbase.h>
#include <shellapi.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cwchar>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common/platform.h"
#include "../common/version.h"
#include "../host/certificates.h"
#include "resource.h"
#include "setup_actions.h"
#include "strings.h"
#include "zip_reader.h"

namespace pseudalis
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
            bool rolled_back = false;    // the previous programs were put back after a failure
        };

        // The text of a problem of a folder (setup::folder_problem()), in the language of the wizard.
        std::string folder_text(const std::string& problem, const std::string& dir)
        {
            return narrow(tr(problem.c_str(), { { "d", dir } }));
        }

        /*
            The folder of the programs, made so that only SYSTEM and the administrators change
            it, before anything stops.  The folder of an installation in place keeps its
            permissions when they cannot be made so, which the log says.
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
            return setup::secure_parents(o.program_dir, error) && secure_folder(o.program_dir, error);
        }

        /*
            The data folder of the installation, recorded in the registry for every program
            before anything is written there: the folder chosen for a new installation, which
            must serve (setup::data_folder_problem()), and the folder in use for an update.
        */
        bool record_data_folder(const setup::install_options& o, bool update, std::string& error)
        {
            if (update) return !setup::recorded_data_dir().empty() || setup::record_data_dir(data_dir(), error);
            const std::string dir = o.data_dir.empty() ? setup::default_data_dir() : o.data_dir;
            const std::string problem = setup::data_folder_problem(dir, o.program_dir);
            if (!problem.empty())
            {
                error = folder_text(problem, dir);
                return false;
            }
            return setup::record_data_dir(dir, error);
        }

        // The programs and the uninstaller, staged.
        bool stage_programs(const std::string& program_dir, setup::staged_files& files, const progress_fn& progress,
            std::string& error)
        {
            setup::remove_old_files(program_dir);
            progress(0, tr("progress_title"), false);
            auto copying = [&](int p, const std::wstring& t, bool d) { progress(p * 700 / 1000, t, d); };
            if (!stage_payload(program_dir, files, copying, error)) return false;
            if (!setup::stage_uninstaller(program_dir, files, error)) return false;
            // What an earlier version placed and this one no longer carries leaves with the change.
            setup::stage_stale_files(program_dir, files);
            return true;
        }

        // The steps after the copy: settings, certificates, service, commands, entry of the programs.
        bool configure(const setup::install_options& o, bool update, const progress_fn& progress,
            install_report& report)
        {
            std::string& error = report.error;
            if (!update && !setup::write_settings(o, error)) return false;
            // An update from a version without drive of restoration gives it a free letter.
            if (update && !setup::add_missing_settings(o, error)) return false;
            /*
                The service makes the certificates at its start, under its own account, and puts
                its authority in the trusted roots as settings.json says (write_settings()); an
                update keeps the certificates and their trust.
            */
            progress(850, tr("step_certificates"), false);
            if (!setup::prepare_certificates(!update, error)) return false;
            if (!update && !o.trust_certificate) setup::untrust_authority();
            progress(900, tr("step_service"), false);
            if (!setup::register_service(o.program_dir, update, error)) return false;
            // The users of the console may restart the service from the icon.
            std::string e;
            if (!setup::allow_service_control(e)) setup::log("the users cannot restart the service: " + e);
            if (!setup::register_uninstall(o.program_dir, payload_size(), error)) return false;
            /*
                The commands of the Explorer, in the language of the installation: an update
                renews those in place (installed_options()), or removes them when asked to.
            */
            if (o.context_menu)
                setup::register_context_menu(o.program_dir, {
                    { "menu", tr("verb_menu") }, { "protect", tr("verb_protect") },
                    { "protect_copy", tr("verb_protect_copy") }, { "restore", tr("verb_restore") },
                    { "restore_copy", tr("verb_restore_copy") } }, o.language == "auto");
            else if (update)
                setup::remove_context_menu();
            if (!update)
            {
                progress(930, tr("step_shortcuts"), false);
                setup::set_autostart(o.program_dir, o.autostart);
                setup::create_shortcuts(o.program_dir, o.desktop_shortcut);
            }
            return true;
        }

        /*
            Whether the programs placed in dir are those of this installer: an installer built
            from other sources would otherwise record its version over other programs.
        */
        bool programs_match(const std::string& dir, std::string& error)
        {
            for (const char* name : { "pseudalis-service.exe", "pseudalis-tray.exe", "pseudalis.exe" })
            {
                const std::string version = setup::program_version(join_path(dir, name));
                if (version.empty() || version == PSEUDALIS_VERSION) continue;
                error = std::string(name) + " of this installer is version " + version + ", not " + PSEUDALIS_VERSION
                    + ": the installer was built from other sources";
                return false;
            }
            return true;
        }

        // The time the new service takes to run and to open its port before the change is undone.
        const unsigned service_start_ms = 120000;

        /*
            Starts the service of the programs just placed in program_dir and waits until it
            answers on its port.  A service that an administrator disabled stays stopped, and
            counts as started.
        */
        bool start_new_service(const std::string& program_dir, std::string& error)
        {
            if (setup::service_disabled())
            {
                setup::log("the service is disabled: it is not started");
                return true;
            }
            // A service started meanwhile runs the previous programs: it stops first.
            std::string e;
            setup::stop_service(60000, e);
            if (setup::start_service(error) && setup::wait_for_service(service_start_ms, error)) return true;
            setup::log("the new service does not start: " + error);
            setup::log_service_tail();
            // The service that failed stops, with its children, before its files go back.
            setup::stop_service(60000, e);
            setup::end_processes(program_dir, 5000, "pseudalis-service.exe");
            return false;
        }

        /*
            Installs the programs, or updates them when update is true: the settings, the
            certificates and their trust, the shortcuts and the start of the icon then stay
            as they are.  close_icons closes the icons of the users first; the update started
            by the service leaves them open, and each one starts again from the new program
            when it sees the new version of the service.  The free space is checked before
            anything stops.  The previous files stay aside until the new service answers on
            its port: when a step fails, the new service included, the previous programs come
            back with the entry of the installed programs, and the service starts again if it
            was running.
        */
        bool install_programs(const setup::install_options& o, bool update, bool close_icons,
            const progress_fn& progress, install_report& report)
        {
            std::string& error = report.error;
            if (!record_data_folder(o, update, error) || !setup::prepare_data_dir(error)) return false;
            const std::string what = update ? "update" : "installation";
            setup::log(what + " of " + PSEUDALIS_VERSION + " in " + o.program_dir + ", data in " + data_dir() +
                ", port " + std::to_string(o.port) + ", drives " +
                (o.drive_letter.empty() ? std::string("none") : o.drive_letter + ":") + " and " +
                (o.restore_letter.empty() ? std::string("none") : o.restore_letter + ":"));
            if (!prepare_folders(o, update, error)) return false;
            // The new files stay beside those they replace until the end of the copy.
            const uint64_t needed = payload_size() + file_size(executable_path()) + (64ull << 20);
            const uint64_t available = free_disk_space(o.program_dir);
            if (available < needed)
            {
                error = "not enough free space for " + o.program_dir + ": " + std::to_string(needed >> 20)
                    + " MB needed, " + std::to_string(available >> 20) + " MB available";
                return false;
            }
            // A service that waits for its deletion would refuse its new configuration once the files are copied.
            if (setup::service_marked_for_deletion(error)) return false;

            progress(0, tr("step_stop"), false);
            const bool was_running = setup::service_running();
            const bool had_service = setup::service_exists();
            const setup::uninstall_entry entry = setup::save_uninstall_entry();
            setup::suspend_recovery();
            setup::stop_service(60000, error);
            error.clear();
            if (close_icons) report.icons_closed = setup::close_trays(10000);
            // The children of the service that outlive it, the probe of an engine for instance.
            setup::end_processes(o.program_dir, 5000, "pseudalis-service.exe");
            setup::staged_files files;
            const bool copied = stage_programs(o.program_dir, files, progress, error)
                && files.commit(error) && programs_match(o.program_dir, error);
            const bool configured = copied && configure(o, update, progress, report);
            if (configured) progress(950, tr("step_start"), false);
            if (!configured || !start_new_service(o.program_dir, error))
            {
                // The previous programs come back, with the entry of the installed programs and the service.
                setup::log("the previous programs are put back: " + error);
                std::string missing;
                report.rolled_back = files.rollback(missing);
                if (!report.rolled_back)
                {
                    error += "; the previous files could not all be put back: " + missing;
                    setup::log("the previous files could not all be put back: " + missing);
                }
                std::string e;
                if (!had_service) setup::remove_service(e);
                setup::restore_uninstall_entry(entry);
                if (!update)
                {
                    setup::set_autostart(o.program_dir, false);
                    setup::remove_shortcuts();
                    setup::remove_context_menu();
                }
                setup::restore_recovery();
                if (was_running) setup::start_service(e);
                return false;
            }
            files.finish();
            /*
                The drives need no driver: the WinFsp that an earlier version installed for its
                drive leaves, unless another program uses it.
            */
            std::string w;
            if (!setup::remove_owned_winfsp(w))
            {
                setup::log("WinFsp: " + w);
                report.notice = narrow(tr("winfsp_kept", { { "e", w } }));
            }
            setup::remove_empty_program_folders(o.program_dir);
            setup::write_file_list(o.program_dir, files.targets());
            progress(1000, tr("step_done"), false);
            setup::log("done");
            return true;
        }

        /*
            The folders that a new installation finds missing or empty: when it fails, those it
            made go with what they hold, the parents it made with them when nothing else came
            into them, and those it found empty are emptied again.
        */
        class new_folders
        {
        public:
            // Notes dir before the installation, as it is then.
            void note(const std::string& dir)
            {
                std::string d = dir;
                while (d.size() > 3 && (d.back() == '\\' || d.back() == '/')) d.pop_back();
                if (d.size() <= 3) return;
                if (dir_exists(d))
                {
                    if (!is_link(d) && list_dir(d).empty()) emptied_.push_back(d);
                    return;
                }
                // The folder, then the parents missing up to the first folder that exists.
                bool whole = true;
                for (std::string p = d; p.size() > 3 && !dir_exists(p); p = parent_path(p))
                {
                    made_.push_back({ p, whole });
                    whole = false;
                }
            }

            // Whether path lies in a folder that undo() removes or empties.
            bool covers(const std::string& path) const
            {
                for (const auto& m : made_)
                    if (path_inside(path, m.path)) return true;
                for (const auto& e : emptied_)
                    if (path_inside(path, e)) return true;
                return false;
            }

            // Removes the folders made and empties those found empty.
            void undo() const
            {
                for (const auto& e : emptied_)
                {
                    if (is_link(e) || !dir_exists(e)) continue;
                    setup::log("the folder " + e + " is emptied");
                    for (const auto& name : list_dir(e))
                    {
                        const std::string p = join_path(e, name);
                        if (dir_exists(p)) setup::remove_folder(p);
                        else remove_file(p);
                    }
                }
                // The deepest folders first: each folder noted comes before its parents.
                for (const auto& m : made_)
                {
                    if (!dir_exists(m.path)) continue;
                    if (m.whole)
                    {
                        setup::log("the folder " + m.path + " is removed");
                        setup::remove_folder(m.path);
                    }
                    else
                    {
                        RemoveDirectoryW(widen(m.path).c_str());
                    }
                }
            }

        private:
            struct made_folder
            {
                std::string path;
                bool whole;   // the folder noted, which goes with what it holds; otherwise a parent that goes empty
            };
            std::vector<made_folder> made_;
            std::vector<std::string> emptied_;
        };

        /*
            install_programs(), and the record of its result in the log and for the tray
            (setup::write_result()), which tells the user of an update made out of sight that
            failed.  A new installation that fails removes the folders it made and empties
            those it found empty, its log included, which then goes to the temporary folder.
        */
        bool perform_install(const setup::install_options& o, bool update, bool close_icons,
            const progress_fn& progress, install_report& report)
        {
            const std::string previous = setup::installed_version();
            const std::string recorded = setup::recorded_data_dir();
            const bool fresh = !update && previous.empty();
            new_folders folders;
            if (fresh)
            {
                folders.note(o.program_dir);
                folders.note(o.data_dir.empty() ? setup::default_data_dir() : o.data_dir);
            }
            const bool ok = install_programs(o, update, close_icons, progress, report);
            const std::string what = update ? "update" : "installation";
            if (!ok)
            {
                setup::log(what + " failed: " + report.error);
                if (report.rolled_back && update) setup::log("version " + previous + " is in place again");
                else if (report.rolled_back) setup::log("the files copied are removed");
            }
            setup::write_result(what, previous, ok, report.rolled_back, report.error);
            /*
                A new installation that failed leaves the folders and their record as it found
                them, unless its programs stay registered: they keep the folders they use, which
                their uninstaller removes.
            */
            if (!ok && fresh && setup::installed_version().empty())
            {
                // The log leaves the data folder before it goes.
                if (folders.covers(join_path(data_dir(), "logs"))) setup::detach_log();
                folders.undo();
                std::string e;
                if (recorded.empty()) setup::forget_folders();
                else setup::record_data_dir(recorded, e);
            }
            return ok;
        }

        // The keys and the certificates of the local server, which leave with the programs, then their folder.
        void remove_certificates()
        {
            const std::string dir = join_path(data_dir(), "certificates");
            if (!remove_certificate_files(dir)) setup::log("a certificate or a key of " + dir + " cannot be deleted");
            RemoveDirectoryW(widen(dir).c_str());
        }

        /*
            Removes the installation registered on this computer; false when there is none.
            The folder of the programs and the data folder go whole; with keep_data, the data
            (the key, the originals, the database and the settings) stays for a later
            installation on this computer, without the certificates, the token of the tray and
            the updates, and the files of the drives, which lie in the data folder, follow it.
            The WinFsp that an earlier version installed goes, unless another program uses it
            (setup::remove_owned_winfsp()).
        */
        bool perform_uninstall(bool keep_data, const progress_fn& progress, install_report& report)
        {
            std::string& error = report.error;
            const std::string dir = setup::installed_dir();
            if (dir.empty())
            {
                error = narrow(tr("not_installed"));
                return false;
            }
            // The folders read before anything goes: the record of the data folder leaves last.
            const std::string data = data_dir();
            std::string e;
            if (dir_exists(data)) setup::prepare_data_dir(e);
            setup::log("uninstallation from " + dir + (keep_data ? ", the data kept in " : ", with the data of ")
                + data);
            progress(100, tr("step_stop"), false);
            setup::remove_service(e);
            setup::close_trays(10000);
            // The other programs of the folder: the icons of the other sessions, the children of the service.
            setup::end_processes(dir, 5000);
            progress(400, tr("uninstall_progress"), false);
            setup::set_autostart(dir, false);
            setup::remove_shortcuts();
            setup::remove_context_menu();
            setup::untrust_authority();
            remove_certificates();
            // The letters of the drives left with the service; their names in the Explorer go too.
            setup::remove_drive_labels(dir);
            std::string w;
            if (!setup::remove_owned_winfsp(w)) report.notice = narrow(tr("winfsp_kept", { { "e", w } }));
            std::vector<std::string> keep;
            if (keep_data) keep = { data };
            if (!setup::remove_installation(dir, keep))
            {
                const std::string kept = narrow(tr("uninstall_kept", { { "d", dir } }));
                report.notice = report.notice.empty() ? kept : kept + "\n\n" + report.notice;
            }
            progress(800, tr("uninstall_progress"), false);
            if (keep_data)
            {
                // The secrets and the caches go: a new installation makes them again.
                remove_file(join_path(data, "admin.token"));
                setup::remove_folder(join_path(data, "updates"));
            }
            else if (setup::removable_folder(data))
            {
                /*
                    The log leaves the data folder before it goes: written there afterwards, it
                    would follow whatever another account placed at that path meanwhile.
                */
                setup::log("removal of the data folder " + data);
                setup::detach_log();
                setup::remove_folder(data);
                setup::forget_folders();
            }
            else
            {
                setup::log("the data folder " + data + " is a folder of the system: it stays");
                setup::forget_folders();
            }
            // The entry of the installed programs goes last: an uninstallation cut short can run again.
            setup::remove_uninstall();
            progress(1000, tr("step_done"), false);
            return true;
        }

        // The wizard.
        enum page { P_WELCOME, P_LICENSE, P_FOLDERS, P_OPTIONS, P_PROGRESS, P_FINISH, P_UNINSTALL, P_REMOVED };
        enum
        {
            IDC_TITLE = 100, IDC_TEXT, IDC_LANG_LABEL, IDC_LANG, IDC_LICENSE, IDC_ACCEPT, IDC_DIR_LABEL, IDC_DIR,
            IDC_DIR_BROWSE, IDC_SPACE, IDC_AUTOSTART, IDC_DESKTOP, IDC_TRUST, IDC_CONTEXT_MENU, IDC_PORT_LABEL,
            IDC_PORT, IDC_STEP, IDC_PROGRESS, IDC_DETAIL, IDC_LAUNCH, IDC_KEEP_DATA, IDC_BACK, IDC_NEXT, IDC_CANCEL,
            IDC_DATA_LABEL, IDC_DATA, IDC_DATA_BROWSE, IDC_DRIVE_LABEL, IDC_DRIVE, IDC_DRIVE_NOTE,
            IDC_RESTORE_LABEL, IDC_RESTORE
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
            // What failed without stopping the installation: the removal of a WinFsp of an earlier version.
            std::string notice;
            // The icons of this session were closed for the copy.
            bool icons_closed = false;
            // The previous programs were put back after a failure.
            bool rolled_back = false;
            std::string previous_version;
            // Pseudalis is installed: the wizard updates it, or leaves a newer version alone.
            bool update = false;
            bool newer_installed = false;
            std::string installed_dir;
            // The letters the drives may take, "" first for no drive.
            std::vector<std::string> letters;
            std::vector<std::string> restore_letters;
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
            const std::string data = narrow(ui::text_of(ctl(IDC_DATA)));
            std::string dir = programs;
            std::string problem = setup::folder_problem(programs, true);
            if (problem.empty())
            {
                dir = data;
                problem = setup::data_folder_problem(data, programs);
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
            ui::set_text(ctl(IDC_DATA_LABEL), tr("folder_data"));
            ui::set_text(ctl(IDC_DIR_BROWSE), tr("browse"));
            ui::set_text(ctl(IDC_DATA_BROWSE), tr("browse"));
            ui::set_text(ctl(IDC_DRIVE_LABEL), tr("opt_drive"));
            ui::set_text(ctl(IDC_RESTORE_LABEL), tr("opt_restore_drive"));
            ui::set_text(ctl(IDC_DRIVE_NOTE), tr("opt_drive_note"));
            ui::set_text(ctl(IDC_AUTOSTART), tr("opt_autostart"));
            ui::set_text(ctl(IDC_DESKTOP), tr("opt_desktop"));
            ui::set_text(ctl(IDC_TRUST), tr("opt_trust"));
            ui::set_text(ctl(IDC_CONTEXT_MENU), tr("opt_context_menu"));
            ui::set_text(ctl(IDC_PORT_LABEL), tr("opt_port"));
            // The first choice of each drive: no drive.
            for (int id : { IDC_DRIVE, IDC_RESTORE })
            {
                if ((id == IDC_DRIVE ? wz.letters : wz.restore_letters).empty()) continue;
                const LRESULT sel = SendMessageW(ctl(id), CB_GETCURSEL, 0, 0);
                SendMessageW(ctl(id), CB_DELETESTRING, 0, 0);
                const std::wstring none = tr("opt_drive_none");
                SendMessageW(ctl(id), CB_INSERTSTRING, 0, reinterpret_cast<LPARAM>(none.c_str()));
                SendMessageW(ctl(id), CB_SETCURSEL, static_cast<WPARAM>(sel < 0 ? 0 : sel), 0);
            }
            ui::set_text(ctl(IDC_LAUNCH), tr("opt_launch"));
            ui::set_text(ctl(IDC_KEEP_DATA), tr("uninstall_keep_data"));
            ui::set_text(ctl(IDC_BACK), tr("btn_back"));
            ui::set_text(ctl(IDC_CANCEL), tr("btn_cancel"));
        }

        // The text of the uninstallation page follows the choice of keeping the key and the originals.
        std::wstring uninstall_text()
        {
            return tr(ui::checked(ctl(IDC_KEEP_DATA)) ? "uninstall_text_keep" : "uninstall_text");
        }

        void show_page(page p)
        {
            wz.current = p;
            const std::vector<int> all = { IDC_TEXT, IDC_LANG_LABEL, IDC_LANG, IDC_LICENSE, IDC_ACCEPT, IDC_DIR_LABEL,
                IDC_DIR, IDC_DIR_BROWSE, IDC_DATA_LABEL, IDC_DATA, IDC_DATA_BROWSE, IDC_SPACE, IDC_DRIVE_LABEL,
                IDC_DRIVE, IDC_DRIVE_NOTE, IDC_AUTOSTART, IDC_DESKTOP, IDC_TRUST, IDC_CONTEXT_MENU, IDC_PORT_LABEL,
                IDC_PORT, IDC_STEP, IDC_PROGRESS, IDC_DETAIL, IDC_LAUNCH, IDC_KEEP_DATA, IDC_RESTORE_LABEL,
                IDC_RESTORE };
            std::vector<int> visible;
            std::wstring title, text;
            switch (p)
            {
            case P_WELCOME:
            {
                title = wz.update ? tr("update_title") : tr("welcome_title");
                const std::vector<std::pair<std::string, std::string>> values = { { "v", PSEUDALIS_VERSION },
                    { "o", wz.previous_version }, { "d", wz.installed_dir } };
                if (wz.newer_installed) text = tr("update_newer", values);
                else if (wz.update) text = tr("update_text", values);
                else text = tr("welcome_text", values);
                visible = { IDC_TEXT, IDC_LANG_LABEL, IDC_LANG };
                break;
            }
            case P_LICENSE:
                title = tr("license_title");
                visible = { IDC_LICENSE, IDC_ACCEPT };
                break;
            case P_FOLDERS:
                title = tr("folder_title");
                visible = { IDC_DIR_LABEL, IDC_DIR, IDC_DIR_BROWSE, IDC_DATA_LABEL, IDC_DATA, IDC_DATA_BROWSE,
                    IDC_SPACE };
                break;
            case P_OPTIONS:
                title = tr("options_title");
                visible = { IDC_DRIVE_LABEL, IDC_DRIVE, IDC_RESTORE_LABEL, IDC_RESTORE, IDC_DRIVE_NOTE,
                    IDC_CONTEXT_MENU, IDC_AUTOSTART, IDC_DESKTOP, IDC_TRUST, IDC_PORT_LABEL, IDC_PORT };
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
                        ? tr("update_done_text", { { "v", PSEUDALIS_VERSION }, { "o", wz.previous_version } })
                        : tr("update_failed", { { "e", wz.error } });
                    // A failed update says that the previous version is back in place.
                    if (!wz.succeeded && wz.rolled_back)
                        text += L"\n\n" + tr("update_rolled_back", { { "o", wz.previous_version } });
                }
                else
                {
                    title = wz.succeeded ? tr("finish_title") : tr("error_title");
                    text = wz.succeeded ? tr("finish_text") : tr("error_failed", { { "e", wz.error } });
                    if (!wz.succeeded && wz.rolled_back) text += L"\n\n" + tr("install_rolled_back");
                }
                if (wz.succeeded && !wz.notice.empty()) text += L"\n\n" + widen(wz.notice);
                visible = { IDC_TEXT };
                if (wz.succeeded) visible.push_back(IDC_LAUNCH);
                break;
            case P_UNINSTALL:
                title = tr("uninstall_title");
                text = uninstall_text();
                visible = { IDC_TEXT, IDC_KEEP_DATA };
                break;
            case P_REMOVED:
                title = tr("uninstall_title");
                text = wz.succeeded ? tr("uninstall_done") : widen(wz.error);
                // The folder of the programs that stays with the folders kept.
                if (wz.succeeded && !wz.notice.empty()) text += L"\n\n" + widen(wz.notice);
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
            ui::place(ctl(IDC_KEEP_DATA), x, y + px(150), cw, px(40));
            ui::place(ctl(IDC_LAUNCH), x, y + px(150), cw, px(24));
            ui::place(ctl(IDC_LICENSE), x, y, cw, by - px(56) - y);
            ui::place(ctl(IDC_ACCEPT), x, by - px(44), cw, px(24));
            // The folders of the programs and of the data, then the space of the drive.
            const int fields[] = { IDC_DIR_LABEL, IDC_DIR, IDC_DIR_BROWSE, IDC_DATA_LABEL, IDC_DATA, IDC_DATA_BROWSE };
            for (int i = 0; i < 2; ++i)
            {
                const int fy = y + px(60) * i;
                ui::place(ctl(fields[3 * i]), x, fy, cw, px(20));
                ui::place(ctl(fields[3 * i + 1]), x, fy + px(22), cw - px(110), px(26));
                ui::place(ctl(fields[3 * i + 2]), x + cw - px(100), fy + px(21), px(100), px(28));
            }
            ui::place(ctl(IDC_SPACE), x, y + px(130), cw, px(64));
            // The drives, their note, then the options.
            ui::place(ctl(IDC_DRIVE_LABEL), x, y + px(4), px(260), px(22));
            ui::place(ctl(IDC_DRIVE), x + px(266), y, px(140), px(200));
            ui::place(ctl(IDC_RESTORE_LABEL), x, y + px(36), px(260), px(22));
            ui::place(ctl(IDC_RESTORE), x + px(266), y + px(32), px(140), px(200));
            ui::place(ctl(IDC_DRIVE_NOTE), x, y + px(64), cw, px(52));
            int oy = y + px(118);
            for (int id : { IDC_CONTEXT_MENU, IDC_AUTOSTART, IDC_DESKTOP, IDC_TRUST })
            {
                ui::place(ctl(id), x, oy, cw, px(32));
                oy += px(34);
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
            add(IDC_DATA_LABEL, ui::label(wz.hwnd, L"", IDC_DATA_LABEL), ui::font_kind::bold);
            add(IDC_DATA, ui::edit(wz.hwnd, widen(current.data_dir), IDC_DATA));
            add(IDC_DATA_BROWSE, ui::button(wz.hwnd, L"", IDC_DATA_BROWSE));
            if (!wz.previous_version.empty())
            {
                // An update replaces the programs in their folder, and keeps the data in its own.
                for (int id : { IDC_DIR, IDC_DATA }) SendMessageW(ctl(id), EM_SETREADONLY, TRUE, 0);
                for (int id : { IDC_DIR_BROWSE, IDC_DATA_BROWSE }) ui::enable(ctl(id), false);
            }
            add(IDC_SPACE, ui::label(wz.hwnd, L"", IDC_SPACE));
            // The letters free on this computer, A: first when it is, and the letters of the drives in place.
            add(IDC_DRIVE_LABEL, ui::label(wz.hwnd, L"", IDC_DRIVE_LABEL));
            HWND drive = add(IDC_DRIVE, ui::combo(wz.hwnd, IDC_DRIVE));
            add(IDC_RESTORE_LABEL, ui::label(wz.hwnd, L"", IDC_RESTORE_LABEL));
            HWND restore = add(IDC_RESTORE, ui::combo(wz.hwnd, IDC_RESTORE));
            for (int k = 0; k < 2; ++k)
            {
                std::vector<std::string>& letters = k == 0 ? wz.letters : wz.restore_letters;
                const std::string& wanted = k == 0 ? current.drive_letter : current.restore_letter;
                HWND box = k == 0 ? drive : restore;
                letters = setup::free_drive_letters(current.drive_letter, current.restore_letter);
                letters.insert(letters.begin(), std::string());
                int chosen = 0;
                for (size_t i = 0; i < letters.size(); ++i)
                {
                    const std::wstring item = letters[i].empty() ? std::wstring(L"-") : widen(letters[i] + ":");
                    SendMessageW(box, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(item.c_str()));
                    if (letters[i] == wanted && !letters[i].empty()) chosen = static_cast<int>(i);
                }
                // The letter of the drive of anonymization defaults to the first free one.
                if (k == 0 && chosen == 0 && letters.size() > 1 && !wanted.empty()) chosen = 1;
                SendMessageW(box, CB_SETCURSEL, static_cast<WPARAM>(chosen), 0);
            }
            add(IDC_DRIVE_NOTE, ui::label(wz.hwnd, L"", IDC_DRIVE_NOTE), ui::font_kind::note);
            add(IDC_CONTEXT_MENU, ui::checkbox(wz.hwnd, L"", IDC_CONTEXT_MENU, current.context_menu));
            add(IDC_AUTOSTART, ui::checkbox(wz.hwnd, L"", IDC_AUTOSTART, current.autostart));
            add(IDC_DESKTOP, ui::checkbox(wz.hwnd, L"", IDC_DESKTOP, current.desktop_shortcut));
            add(IDC_TRUST, ui::checkbox(wz.hwnd, L"", IDC_TRUST, current.trust_certificate));
            add(IDC_PORT_LABEL, ui::label(wz.hwnd, L"", IDC_PORT_LABEL));
            add(IDC_PORT, ui::edit(wz.hwnd, std::to_wstring(current.port), IDC_PORT, ES_NUMBER));
            add(IDC_STEP, ui::label(wz.hwnd, L"", IDC_STEP), ui::font_kind::bold);
            add(IDC_PROGRESS, ui::progress(wz.hwnd, IDC_PROGRESS));
            add(IDC_DETAIL, ui::label(wz.hwnd, L"", IDC_DETAIL, SS_PATHELLIPSIS), ui::font_kind::note);
            add(IDC_LAUNCH, ui::checkbox(wz.hwnd, L"", IDC_LAUNCH, true));
            // The key and the originals stay unless the user unticks the box: without them,
            // no document anonymized here can be restored.
            add(IDC_KEEP_DATA, ui::checkbox(wz.hwnd, L"", IDC_KEEP_DATA, true));
            add(IDC_BACK, ui::button(wz.hwnd, L"", IDC_BACK));
            add(IDC_NEXT, ui::button(wz.hwnd, L"", IDC_NEXT, BS_DEFPUSHBUTTON));
            add(IDC_CANCEL, ui::button(wz.hwnd, L"", IDC_CANCEL));
            apply_texts();
            layout();
            show_page(wz.uninstall ? P_UNINSTALL : P_WELCOME);
        }

        // The letter chosen in a list of the drives, empty for no drive.
        std::string chosen_letter(int id, const std::vector<std::string>& letters)
        {
            const LRESULT d = SendMessageW(ctl(id), CB_GETCURSEL, 0, 0);
            const size_t chosen = d > 0 ? static_cast<size_t>(d) : 0;
            return chosen > 0 && chosen < letters.size() ? letters[chosen] : std::string();
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
                const bool keep_data = ui::checked(ctl(IDC_KEEP_DATA));
                std::thread([target, keep_data, progress]() {
                    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
                    install_report* r = new install_report();
                    const bool ok = perform_uninstall(keep_data, progress, *r);
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
                o.data_dir = narrow(ui::text_of(ctl(IDC_DATA)));
                o.drive_letter = chosen_letter(IDC_DRIVE, wz.letters);
                o.restore_letter = chosen_letter(IDC_RESTORE, wz.restore_letters);
                // The language of the system, the one of the wizard when nothing else was chosen,
                // follows each user; a language chosen in its place is kept for every user.
                o.language = current_language() == system_language() ? "auto" : current_language();
                o.context_menu = ui::checked(ctl(IDC_CONTEXT_MENU));
                o.autostart = ui::checked(ctl(IDC_AUTOSTART));
                o.desktop_shortcut = ui::checked(ctl(IDC_DESKTOP));
                o.trust_certificate = ui::checked(ctl(IDC_TRUST));
                const long port = std::wcstol(ui::text_of(ctl(IDC_PORT)).c_str(), nullptr, 10);
                if (port >= 1024 && port <= 65535) o.port = static_cast<int>(port);
            }
            std::thread([target, o, update, progress]() {
                CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
                install_report* r = new install_report();
                const bool ok = perform_install(o, update, true, progress, *r);
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
                const std::string tray = join_path(dir, "pseudalis-tray.exe");
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
                case IDC_KEEP_DATA:
                    if (wz.current == P_UNINSTALL) ui::set_text(ctl(IDC_TEXT), uninstall_text());
                    return 0;
                case IDC_DIR:
                    if (HIWORD(wp) == EN_CHANGE && wz.current == P_FOLDERS) SetTimer(w, TIMER_SPACE, 400, nullptr);
                    return 0;
                case IDC_DIR_BROWSE:
                case IDC_DATA_BROWSE:
                {
                    const int target = LOWORD(wp) == IDC_DIR_BROWSE ? IDC_DIR : IDC_DATA;
                    std::wstring d = ui::choose_folder(w, ui::text_of(ctl(target)));
                    if (!d.empty())
                    {
                        // A folder of the programs or of the data chosen by the user gets its own Pseudalis subfolder.
                        if (file_name(narrow(d)) != "Pseudalis") d = widen(join_path(narrow(d), "Pseudalis"));
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
                    case P_OPTIONS:
                    {
                        // Two drives, two letters.
                        const std::string a = chosen_letter(IDC_DRIVE, wz.letters);
                        if (!a.empty() && a == chosen_letter(IDC_RESTORE, wz.restore_letters))
                        {
                            ui::message(wz.hwnd, tr("opt_drive_same", { { "l", a } }), MB_OK | MB_ICONWARNING);
                            break;
                        }
                        start_work();
                        break;
                    }
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
                    wz.rolled_back = r->rolled_back;
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
            return !version.empty() && !dir.empty() && file_exists(join_path(dir, "pseudalis-service.exe"));
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
                wz.newer_installed = compare_versions(wz.previous_version, PSEUDALIS_VERSION) > 0;
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
            WNDCLASSEXW wc = { sizeof(wc) };
            wc.lpfnWndProc = proc;
            wc.hInstance = instance;
            wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
            wc.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(IDI_APP));
            wc.lpszClassName = L"PseudalisSetupWindow";
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
            HWND w = CreateWindowExW(WS_EX_CONTROLPARENT, wc.lpszClassName, L"Pseudalis", style, x, y, ww, wh,
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
    using namespace pseudalis;
    // The current folder never holds the folder of the programs, which the uninstaller removes.
    wchar_t system[MAX_PATH];
    const UINT length = GetSystemDirectoryW(system, MAX_PATH);
    if (length > 0 && length < MAX_PATH) SetCurrentDirectoryW(system);
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::vector<std::wstring> args;
    for (int i = 1; argv && i < argc; ++i) args.push_back(argv[i]);
    if (argv) LocalFree(argv);

    // The copy of the uninstaller that finishes the removal of the folder of the programs.
    if (has_arg(args, L"/finish-removal")) return setup::complete_removal();
    // The file of an installer, held from now on: the uninstaller is copied from it.
    if (payload().data) setup::hold_own_program();

    const std::wstring lang = arg_after(args, L"/lang");
    set_language(lang.empty() ? "auto" : narrow(lang));
    ui::init();
    const bool silent = has_arg(args, L"/silent");
    std::string error;
    // The lines of the log that no data folder received go to the temporary folder at the end.
    struct log_flush
    {
        ~log_flush() { setup::flush_log(); }
    } flush_at_exit;

    const std::wstring svc = arg_after(args, L"/service");
    if (!svc.empty())
    {
        bool ok = true;
        if (svc == L"stop" || svc == L"restart") ok = setup::stop_service(60000, error);
        if (svc == L"start" || svc == L"restart") ok = setup::start_service(error) && ok;
        return ok ? 0 : 1;
    }

    if (has_arg(args, L"/update"))
    {
        // Started by the service, which stops for it; everything else is kept.
        std::string version, installed, e;
        const bool in_place = installed_in_place(version, installed);
        if (in_place) setup::prepare_data_dir(e);
        // Only an installation in place is updated, and never by an older version.
        if (!in_place || compare_versions(version, PSEUDALIS_VERSION) > 0)
        {
            setup::log(in_place ? "update: Pseudalis " + version + " is installed, newer than " + PSEUDALIS_VERSION
                + ": nothing done" : std::string("update: Pseudalis is not installed in place: nothing done"));
            // The service that stopped for the update starts again on its programs.
            if (in_place) setup::start_service(e);
            return 1;
        }
        const setup::install_options o = setup::installed_options();
        install_report report;
        const bool ok = perform_install(o, true, false, [](int, const std::wstring&, bool) {}, report);
        if (!ok)
        {
            // The service starts again on its previous programs.
            std::string e2;
            setup::start_service(e2);
            return 1;
        }
        /*
            The icons stay open during this update: the programs they run are set aside and
            go once each icon starts again from its new program, which it does when it sees
            the new version of the service, within two minutes.
        */
        size_t left = setup::remove_old_files(o.program_dir, true);
        for (int i = 0; i < 120 && left > 0; ++i)
        {
            Sleep(1000);
            left = setup::remove_old_files(o.program_dir, true);
        }
        if (left > 0)
            setup::log(std::to_string(left) + " files set aside stay in use, they go at the next restart of Windows");
        return 0;
    }

    /*
        The uninstaller, which has no payload, uninstalls when asked to, or started without
        options.  The copy kept in the folder of the installation is the uninstaller, also
        when its payload could not be taken out of it.
    */
    const std::string install_dir = setup::installed_dir();
    const std::wstring uninstaller = widen(join_path(install_dir, "pseudalis-setup.exe"));
    const bool no_payload = !payload().data
        || (!install_dir.empty() && _wcsicmp(widen(executable_path()).c_str(), uninstaller.c_str()) == 0);
    const size_t wizard_args = lang.empty() ? 0 : 2;
    if (has_arg(args, L"/uninstall") || (no_payload && args.size() == wizard_args))
    {
        int code = 0;
        if (silent)
        {
            /*
                The data stays, unless /purge is given: the silent removal also serves to replace an
                installation (a deployment tool), which must not take the key and the originals.
            */
            install_report report;
            if (!perform_uninstall(!has_arg(args, L"/purge"), [](int, const std::wstring&, bool) {}, report))
            {
                setup::log("uninstallation: " + report.error);
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
        // Any other command does nothing here: with /silent it would have removed Pseudalis.
        if (!silent) ui::message(nullptr, tr("no_payload"), MB_OK | MB_ICONINFORMATION);
        return 1;
    }

    if (silent)
    {
        std::string version, installed;
        const bool update = installed_in_place(version, installed);
        // An update keeps its data folder; a new installation takes the folder given, else the default one.
        const std::wstring data = arg_after(args, L"/data-dir");
        const bool chosen_data = !data.empty() && !update;
        /*
            The log starts in the data folder in place; a new installation makes its data folder
            once the folders given serve (install_programs()), so that a refusal leaves nothing.
        */
        if (update) setup::prepare_data_dir(error);
        error.clear();
        // The options not given keep the values of the installation in place.
        setup::install_options o = setup::installed_options();
        if (chosen_data) o.data_dir = narrow(data);
        if (update && compare_versions(version, PSEUDALIS_VERSION) > 0)
        {
            setup::log("Pseudalis " + version + " is installed, newer than " + PSEUDALIS_VERSION + ": nothing done");
            return 0;
        }
        bool options = false;
        const std::wstring dir = arg_after(args, L"/dir");
        // An update replaces the programs in their folder.
        if (!dir.empty() && !update) o.program_dir = narrow(dir);
        // The letter of each drive, or none.
        for (int k = 0; k < 2; ++k)
        {
            const std::wstring drive = arg_after(args, k == 0 ? L"/drive" : L"/restore-drive");
            if (drive.empty()) continue;
            std::string& target = k == 0 ? o.drive_letter : o.restore_letter;
            const std::string letter = narrow(drive).substr(0, 1);
            if (_wcsicmp(drive.c_str(), L"none") == 0) target.clear();
            else if (setup::valid_drive_choice(letter)) target = std::string(1,
                static_cast<char>(std::toupper(static_cast<unsigned char>(letter[0]))));
            else
            {
                setup::log("invalid drive letter: " + narrow(drive));
                return 1;
            }
            options = true;
        }
        if (!o.drive_letter.empty() && o.drive_letter == o.restore_letter)
        {
            setup::log("the two drives need two letters: " + o.drive_letter + ":");
            return 1;
        }
        if (has_arg(args, L"/no-context-menu")) o.context_menu = false;
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
        if (problem.empty() && !update)
        {
            folder = o.data_dir;
            problem = setup::data_folder_problem(folder, o.program_dir);
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
        // The icons closed for the copy start again, with the rights of their user; after a
        // failure, the icon tells it, since no window of the installer does.
        const std::string tray = join_path(o.program_dir, "pseudalis-tray.exe");
        if (report.icons_closed && file_exists(tray)) setup::launch_unelevated(tray, ok ? "" : "--setup-result");
        return ok ? 0 : 1;
    }
    return run_wizard(instance, false);
}
