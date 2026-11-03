// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_WINDOWS_SETUP_ACTIONS_H_
#define PSEUDALIS_WINDOWS_SETUP_ACTIONS_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "../common/json.h"

namespace pseudalis
{
namespace setup
{

    /*
        The operations of the installer, each one usable by the wizard, by the silent
        update the service starts, and by the uninstaller.  Paths are UTF-8.
    */

    const char* const service_name = "Pseudalis";

    struct install_options
    {
        std::string program_dir;      // C:\Program Files\Pseudalis by default
        std::string data_dir;         // C:\ProgramData\Pseudalis by default
        std::string drive_letter = "A";   // the letter of the drive of anonymization, empty without drive
        std::string restore_letter = "R"; // the letter of the drive of restoration, empty without drive
        std::string language = "auto";    // the language of the programs and of the descriptions
        bool autostart = true;        // the tray at the sign in of every user
        bool desktop_shortcut = false;
        bool context_menu = true;     // the commands of the Explorer
        bool trust_certificate = true;
        int port = 8444;
    };

    // Folders of the machine.
    std::string default_program_dir();

    /*
        Why a folder cannot receive the programs (program true) or other files: the key of
        the text that says it (strings.cpp, with {d} for the folder), or an empty string
        when it can.  The folder must be on a local drive, below its root, and neither a
        folder of Windows or of the users nor one that holds them.  The links on its path
        must belong to SYSTEM or the administrators, and no other account may rename a
        folder on the path of the programs (folder_parent_unsafe).  An existing folder of
        the programs must be empty or hold an installation of Pseudalis.  Another folder
        lies apart from the folder of the programs program_dir, when given
        (folder_overlap): an update replaces what that folder holds.
    */
    std::string folder_problem(const std::string& dir, bool program, const std::string& program_dir = std::string());

    /*
        Why a folder cannot hold the data (the settings, the key, the database, the vault
        of the originals, the logs and the certificates): the key of the text that says it, or an
        empty string when it can.  The folder must be on a local fixed drive, below its
        root, outside the folder of Windows and the folders of the users, holding none of
        them, and apart from the folder of the programs program_dir; no other account may
        rename a folder on its path.  An existing folder must be empty, or hold the data of
        an earlier installation (settings.json, pseudalis.db) and belong to SYSTEM or the
        administrators.
    */
    std::string data_folder_problem(const std::string& dir, const std::string& program_dir);

    /*
        Makes the missing folders on the path of dir, below the folder that exists, with
        the permissions of secure_folder(), so that no other account may rename them and
        put a folder of its own in the place of dir.
    */
    bool secure_parents(const std::string& dir, std::string& error);

    /*
        The data folder of an installation recorded in HKLM\Software\Pseudalis, which only
        the administrators change, and which every program reads first (data_dir()).
        recorded_data_dir() is empty without a record; default_data_dir() is the folder of
        the data without a choice, %ProgramData%\Pseudalis.  forget_folders() deletes the record.
    */
    std::string recorded_data_dir();
    std::string default_data_dir();
    bool record_data_dir(const std::string& dir, std::string& error);
    void forget_folders();

    /*
        Whether the uninstaller may remove a folder whole: a folder on a local drive, below
        its root, that is neither a folder of Windows or of the users nor one that holds
        them.
    */
    bool removable_folder(const std::string& dir);


    // The installation recorded in the registry: folder and version, empty when none.
    std::string installed_dir();
    std::string installed_version();

    /*
        The options of the installation in place, for a new run of the installer over it:
        its folder, the port, the drive letter and the language
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
    // Whether the service waits for its deletion, until the programs that hold it close it; error says what to do.
    bool service_marked_for_deletion(std::string& error);
    // Whether the service is disabled, by an administrator for instance.
    bool service_disabled();
    bool stop_service(unsigned timeout_ms, std::string& error);
    bool start_service(std::string& error);
    /*
        Waits until the service runs and its process listens on the port of settings.json,
        which the service rewrites when it falls back on its last port; without the table
        of the sockets, a connection to the port serves.  A service that stops without an
        error code meanwhile counts as started: it stops so for an update it starts, or
        when asked to.  Returns false, with the reason in error, when the service stops
        with an error code or does not listen within timeout_ms.
    */
    bool wait_for_service(unsigned timeout_ms, std::string& error);
    // Writes the last lines of the log of the service (logs\service.log) into the log of the installer.
    void log_service_tail();
    /*
        Registers the service, with the restarts the service manager makes after a failure.
        With keep_start_type, a service in place keeps the start chosen for it (an update);
        otherwise it starts with Windows, after a delay.
    */
    bool register_service(const std::string& program_dir, bool keep_start_type, std::string& error);
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

    /*
        Waits until no process runs a program of folder, this one aside, for timeout_ms at
        most, then ends those that remain, in every session, and waits for their end: the
        trays of the other users, the commands of the Explorer, and the children of the
        service (its restart).  With program, only the
        processes of that file of folder count.  Returns false when one still runs.
    */
    bool end_processes(const std::string& folder, unsigned timeout_ms, const std::string& program = std::string());

    // The version of a program file (its version resource, "1.2.3"), empty without one.
    std::string program_version(const std::string& path);

    // Receives the successive pieces of a file; returns false to stop.
    typedef std::function<bool(const void* data, size_t size)> chunk_writer;

    class staged_files
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The files an installation writes, as one change.  Each new file is
                written beside the file it replaces (name.new) and flushed to the disk;
                commit() then moves each file replaced aside (name.old-<random>) and each
                new file into its place.  rollback() puts the previous files back and
                finish() deletes them; the files in use, such as a program running in
                another session, are deleted at the next restart of Windows.  Destroyed
                before commit(), the set deletes its new files.  The moves are tried again
                for a few seconds, the time a scanner takes to release a file.
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
        /*
            A file of the installation that the change removes: commit() sets it aside with
            the files replaced, so that rollback() puts it back.  A file that cannot be set
            aside stays, and finish() deletes it, at the next restart of Windows when in use.
        */
        void remove(const std::string& target);
        /*
            Moves the new files into place; on failure, puts every file back and returns
            false, error naming the files that could not be put back, if any.
        */
        bool commit(std::string& error);
        /*
            After commit(): puts the previous files back.  Returns true when every previous
            file is in place again (also without a commit, when commit() put them back), else
            false with the files that stay aside in error.
        */
        bool rollback(std::string& error);
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
            bool kept = false;   // a removed file that could not be set aside
        };
        bool restore(size_t count, std::string& failed);
        std::vector<entry> entries_;
        bool committed_ = false;
        // The previous files that a failed commit() could not put back.
        std::string unrestored_;
    };

    /*
        Deletes the files an earlier update left aside or unfinished, and the temporary files
        of a write.  A file in use goes at the next restart of Windows, or, with now_only,
        stays for a later call.  Returns the number of files that stay.
    */
    size_t remove_old_files(const std::string& program_dir, bool now_only = false);

    /*
        Adds to the set, as files it removes, the files of the web application, the licenses
        and the installer of WinFsp that an earlier installation placed in program_dir and
        that the set does not install: commit() sets them aside with the files it replaces,
        rollback() puts them back and finish() deletes them.  The paths compare without case,
        as Windows names its files; the links stay, with what they lead to, and so do the
        files of the data folder.
    */
    void stage_stale_files(const std::string& program_dir, staged_files& files);

    // Removes the folders of the web application, the licenses and the WinFsp of the versions before that hold nothing.
    void remove_empty_program_folders(const std::string& program_dir);

    /*
        Opens the file of this program at its start and holds it, so that no account renames,
        replaces or changes it while the installer runs, after checking that it is the very
        file Windows runs (the file of the image of the process, which follows its renames).
        stage_uninstaller() copies that file, never the path the program was started from,
        which another account may have replaced since.
    */
    void hold_own_program();

    // Writes this program into the set, without its payload, as the uninstaller.
    bool stage_uninstaller(const std::string& program_dir, staged_files& files, std::string& error);


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
    // Adds to settings.json the settings a version before did not write: the letter of the drive of restoration.
    bool add_missing_settings(const install_options& o, std::string& error);

    // Removes the names and the icons of the drives of the programs of program_dir from the Explorer.
    void remove_drive_labels(const std::string& program_dir);


    /*
        The certificates of the local server, which the service makes and renews under its
        own account (host/certificates.h), so that the key of the server is encrypted for
        LocalSystem and the key of the authority is never written: a new installation
        deletes those found in the data folder, an update keeps them.  The choice of trust
        reaches the service through the settings (trust_authority).  trust_authority() adds
        the authority of the data folder to the trusted roots of the machine once it has
        checked that it is the one Pseudalis makes; authority_trusted() tells whether the
        trusted roots hold an authority of Pseudalis; untrust_authority() removes them.
    */
    bool prepare_certificates(bool new_installation, std::string& error);
    bool trust_authority(std::string& error);
    bool authority_trusted();
    void untrust_authority();

    // Registry: the entry of the installed programs and the start of the tray.
    bool register_uninstall(const std::string& program_dir, uint64_t size_bytes, std::string& error);
    void remove_uninstall();
    void set_autostart(const std::string& program_dir, bool on);

    /*
        The values of the entry of the installed programs, saved before an installation
        changes it and written back by restore_uninstall_entry() when the installation fails;
        a snapshot taken without an entry removes the entry.
    */
    struct uninstall_entry
    {
        struct value
        {
            std::wstring name;
            uint32_t type = 0;
            std::vector<uint8_t> data;
        };
        bool present = false;
        std::vector<value> values;
    };
    uninstall_entry save_uninstall_entry();
    void restore_uninstall_entry(const uninstall_entry& entry);

    /*
        The WinFsp that an earlier version of Pseudalis installed for its drive, recorded in
        the registry and hidden from the installed programs: the drives need no driver any
        more, and remove_owned_winfsp() uninstalls it, at the installation of this version and
        at the uninstallation.  It stays when another program registered a file system with
        it since, or replaced it, shown again among the installed programs.  A WinFsp that
        Pseudalis did not install is never touched.  Returns false, with the reason in error,
        when msiexec could not uninstall it.
    */
    bool remove_owned_winfsp(std::string& error);

    // Lets the interactive users start and stop the service, from the icon.
    bool allow_service_control(std::string& error);

    /*
        The commands of the Explorer on the files Pseudalis protects (Show more options on
        Windows 11): a cascade "Pseudalis" with Anonymize, Anonymize as a copy, Restore the
        original and Restore as a copy, which run pseudalis.exe of program_dir.  labels gives
        their texts in the language of the installation: "menu", "protect", "protect_copy",
        "restore" and "restore_copy".  With each_user, the language of the installation
        follows the system, and the commands take the texts of pseudalis.exe in the language
        of each user instead.
    */
    bool context_menu_registered();
    bool register_context_menu(const std::string& program_dir, const std::map<std::string, std::wstring>& labels,
                               bool each_user);
    void remove_context_menu();

    /*
        The letters the drives may take: A, B and D to Z that no drive of the computer nor
        network drive of the user takes, and current and also, the letters of the drives in
        place.  valid_drive_choice() tells whether a letter given to the silent installation
        is one.  default_restore_letter() is the first free letter from R on that is not
        drive_letter, else another free one, else none.
    */
    std::vector<std::string> free_drive_letters(const std::string& current, const std::string& also = std::string());
    bool valid_drive_choice(const std::string& letter);
    std::string default_restore_letter(const std::string& drive_letter);

    // Shortcuts of the Start menu and of the desktop.
    void create_shortcuts(const std::string& program_dir, bool desktop);
    void remove_shortcuts();

    /*
        Starts a program with the rights of the user of the desktop, from an elevated
        program, through the shell of the desktop; without one (a remote session, a
        service), the program is not started.
    */
    bool launch_unelevated(const std::string& program, const std::string& args);

    /*
        Removes a folder and its content; a link goes alone, without what it leads to.
        What is in use is renamed aside and deleted at the next restart of Windows.
    */
    void remove_folder(const std::string& path);

    /*
        The list of the files of an installation, kept in its folder: write_file_list()
        adds the files given to those it names already; the first list also names the
        programs, engines, web application and licenses that an earlier installation left.

        remove_installation() removes the folder of the programs whole, with what the
        programs or the user added there, the installer refusing a folder that holds other
        files: links go alone, without what they lead to, and the folders of keep that lie
        in it stay, with their parents (the data kept for a later installation).  A folder
        that removable_folder() refuses only loses the files the list names and the files
        the programs make there (the files an update set aside or left unfinished, the
        temporary files of a write).  The running uninstaller stays:
        finish_removal() starts, as this program ends, a copy of it in a folder of its own
        of %TEMP%, which waits for the end of this program, deletes the uninstaller and
        what the folder still holds by the same rules, then the folder, and leaves for the
        restart of Windows only what stays in use.  remove_installation() returns false
        when the folder stays, with the folders of keep or the files of a refused folder.

        complete_removal() is that copy, started with /finish-removal; it reads what to do
        from the file its folder holds, which only an administrator writes.
    */
    void write_file_list(const std::string& program_dir, const std::vector<std::string>& files);
    bool remove_installation(const std::string& program_dir, const std::vector<std::string>& keep);
    void finish_removal();
    int complete_removal();

    /*
        The result of the last run of the installer, logs\setup-result.json in the data
        folder, which the tray reads to tell the user of an update that failed out of sight:
        operation "installation" or "update", the version installed or kept, ok, and when it
        failed, the error and whether the previous programs were put back.
    */
    void write_result(const std::string& operation, const std::string& previous, bool ok, bool rolled_back,
        const std::string& error);

    // Log of the installer in the data folder; the lines wait until prepare_data_dir().
    void log(const std::string& line);

    /*
        The lines of the log that still wait for a data folder, which a refused folder or a
        removal of a copy that was not installed never makes, go to pseudalis-setup.log of
        the temporary folder of the account.
    */
    void flush_log();

    /*
        Takes back the lines this run wrote in the log of the data folder, which then wait
        again for flush_log(): called before a data folder that a failed installation made
        is removed, so that pseudalis-setup.log of the temporary folder keeps them.
    */
    void detach_log();

}
}

#endif // PSEUDALIS_WINDOWS_SETUP_ACTIONS_H_
