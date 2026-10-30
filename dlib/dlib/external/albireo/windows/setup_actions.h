// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_WINDOWS_SETUP_ACTIONS_H_
#define ALBIREO_WINDOWS_SETUP_ACTIONS_H_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "../common/json.h"

namespace albireo
{
namespace setup
{

    /*
        The operations of the installer, each one usable by the wizard, by the silent
        update the service starts, and by the uninstaller.  Paths are UTF-8.
    */

    const char* const service_name = "Albireo";

    struct install_options
    {
        std::string program_dir;      // C:\Program Files\Albireo by default
        std::string models_dir;       // empty: <data>\models
        bool autostart = true;        // the tray at the sign in of every user
        bool desktop_shortcut = false;
        bool trust_certificate = true;
        bool default_model = true;
        bool offline = false;         // no Internet access: nothing is fetched, files are imported
        int port = 8443;
    };

    // Folders of the machine.
    std::string default_program_dir();
    std::string default_models_dir();

    /*
        Why a folder cannot receive the programs (program true) or the models: the key of
        the text that says it (strings.cpp, with {d} for the folder), or an empty string
        when it can.  The folder must be on a local drive, below its root, and neither a
        folder of Windows or of the users nor one that holds them (it may lie inside
        them).  The links on its path must belong to SYSTEM or the administrators.  An
        existing folder of the programs must be empty or hold an installation of Albireo;
        an existing models folder must belong to SYSTEM or the administrators, or be empty,
        and hold nothing of another account.
    */
    std::string folder_problem(const std::string& dir, bool program);

    // The installation recorded in the registry: folder and version, empty when none.
    std::string installed_dir();
    std::string installed_version();

    /*
        The options of the installation in place, for a new run of the installer over it:
        its folder, the port, the models folder, the default model and the offline mode
        recorded in settings.json, the start of the tray and the shortcut of the desktop as
        found in the registry and on the desktop.  The defaults without an installation.
    */
    install_options installed_options();

    /*
        Service control; the wait is bounded by timeout_ms.  A service that does not stop
        in time is ended.  service_running() is true for a service in any state but
        stopped.
    */
    bool service_exists();
    bool service_running();
    bool stop_service(unsigned timeout_ms, std::string& error);
    bool start_service(std::string& error);
    // Registers the service, with the restarts the service manager makes after a failure.
    bool register_service(const std::string& program_dir, std::string& error);
    /*
        Turns off the restarts after a failure while the programs are replaced, so that a
        service ended by stop_service() does not start again on the files being copied;
        restore_recovery() and register_service() turn them on again.
    */
    void suspend_recovery();
    void restore_recovery();
    bool remove_service(std::string& error);

    // Asks the trays of this session to close and waits for them; true when one was open.
    bool close_trays(unsigned timeout_ms);

    // Receives the successive pieces of a file; returns false to stop.
    typedef std::function<bool(const void* data, size_t size)> chunk_writer;

    class staged_files
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The files an installation writes, as one change.  Each new file is
                written beside the file it replaces (name.new); commit() then moves each
                file replaced aside (name.old-<random>) and each new file into its place.
                rollback() puts the previous files back and finish() deletes them; the
                files in use, such as a program running in another session, are deleted
                at the next restart of Windows.  Destroyed before commit(), the set
                deletes its new files.
        !*/
    public:
        staged_files() = default;
        staged_files(const staged_files&) = delete;
        staged_files& operator=(const staged_files&) = delete;
        ~staged_files();

        // Writes the new content of target, which body gives to the writer it receives.
        bool write(const std::string& target, const std::function<bool(const chunk_writer&)>& body,
            std::string& error);
        // The path of the new file of target, which the caller writes; the file joins the set.
        std::string add(const std::string& target);
        // A file of the installation that the change removes.
        void remove(const std::string& target);
        // Moves the new files into place; on failure, puts every file back and returns false.
        bool commit(std::string& error);
        // After commit(): puts the previous files back.
        void rollback();
        // After commit(): deletes the previous files.
        void finish();
        // The files the set installs.
        std::vector<std::string> targets() const;

    private:
        struct entry
        {
            std::string target, fresh, aside;
            bool removal = false;
            bool placed = false;
        };
        void restore(size_t count);
        std::vector<entry> entries_;
        bool committed_ = false;
    };

    // Deletes the files left aside by an earlier update.
    void remove_old_files(const std::string& program_dir);

    // Writes this program into the set, without its payload and its models, as the uninstaller.
    bool stage_uninstaller(const std::string& program_dir, staged_files& files, std::string& error);

    /*
        The models that come with Albireo, which the complete installer carries after its
        program (common/builtin_models.h): the manifest, with the position of each file in
        the installer, and the size of their files.  The installer without models has none.
    */
    struct builtin_overlay
    {
        uint64_t start = 0;     // position of the overlay, the size of the program before it
        uint64_t total = 0;     // bytes of the files of the models
        json manifest;
        bool present() const { return manifest["models"].size() > 0; }
    };
    builtin_overlay read_builtin_overlay(const std::string& installer);

    /*
        Writes the models that come with this installer into the set, for the "models"
        folder of program_dir: each file copied and checked against its hash, unless the
        same file is there already; the files of the models that no longer come with
        Albireo removed; the manifest builtin.json, which the service reads at its start,
        placed last.  progress receives the bytes done and the name of the file.
        builtin_bytes_to_copy() gives the bytes these new files take.
    */
    bool stage_builtin_models(const std::string& program_dir, staged_files& files,
        const std::function<void(uint64_t done, uint64_t total, const std::string& file)>& progress,
        std::string& error);
    uint64_t builtin_bytes_to_copy(const std::string& program_dir);

    /*
        The data folder, secured before the installer writes anything in it: SYSTEM and
        the administrators full control, the users read (the tray reads the certificate
        of the authority and the token), and nothing that another account placed there
        (secure_folder()).  A data folder that another account made, or a link of another
        account, is renamed aside first.  The log of the installer starts there.
    */
    bool prepare_data_dir(std::string& error);

    // Merges the options into settings.json, keeping the other settings.
    bool write_settings(const install_options& o, std::string& error);

    /*
        Copies the offline bundle the installer comes with, on a medium, into the import
        folder of the service, which installs it at its start: the catalog next to the
        installer (catalog.json and catalog.json.sig) when its signature is that of
        Albireo, and the models of its "models" folder that this catalog lists, each
        checked against its hash.  progress receives the bytes copied, their total and the
        name of the file.  Returns false, with the reason in error, when the bundle cannot
        be copied whole (an unsigned catalog, too little space, a damaged model).
    */
    bool stage_bundle(const std::function<void(uint64_t done, uint64_t total, const std::string& file)>& progress,
        std::string& error);

    /*
        The certificates of the local server: with new_authority, a new installation, the
        authority found in the data folder is replaced by a new one; renewed tells whether
        the authority changed.  trust_authority() adds the authority to the trusted roots
        of the machine once it has checked that it is the one Albireo makes: signed by its
        own key, named "Albireo Local Authority" and limited to localhost and the loopback
        addresses.  authority_trusted() tells whether the trusted roots hold an authority
        of Albireo.
    */
    bool create_certificates(bool new_authority, bool& renewed, std::string& error);
    bool trust_authority(std::string& error);
    bool authority_trusted();
    void untrust_authority();

    // Registry: the entry of the installed programs and the start of the tray.
    bool register_uninstall(const std::string& program_dir, uint64_t size_bytes, std::string& error);
    void remove_uninstall();
    void set_autostart(const std::string& program_dir, bool on);

    // Shortcuts of the Start menu and of the desktop.
    void create_shortcuts(const std::string& program_dir, bool desktop);
    void remove_shortcuts();

    /*
        Starts a program with the rights of the user of the desktop, from an elevated
        program, through the shell of the desktop; without one (a remote session, a
        service), the program is not started.
    */
    bool launch_unelevated(const std::string& program, const std::string& args);

    // Waits until the local server accepts connections.
    bool wait_for_server(int port, unsigned timeout_ms);

    /*
        Removes a folder and its content; a link goes alone, without what it leads to.
        What is in use is renamed aside and deleted at the next restart of Windows.
    */
    void remove_folder(const std::string& path);

    /*
        The list of the files of an installation, kept in its folder: write_file_list()
        adds the files given to those it names already; the first list also names the
        programs, engines, web application, licenses and models that an earlier
        installation left.  remove_installation() removes the files the list names (the
        same places for an installation older than the list), then the folders left
        empty, the folder of the programs included.  The running uninstaller is renamed
        aside; finish_removal() starts, as this program ends, the deletion of that file
        and of the folder left empty.
    */
    void write_file_list(const std::string& program_dir, const std::vector<std::string>& files);
    void remove_installation(const std::string& program_dir);
    void finish_removal();

    // Log of the installer in the data folder; the lines wait until prepare_data_dir().
    void log(const std::string& line);

}
}

#endif // ALBIREO_WINDOWS_SETUP_ACTIONS_H_
