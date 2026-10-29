// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_WINDOWS_SETUP_ACTIONS_H_
#define ALBIREO_WINDOWS_SETUP_ACTIONS_H_

#include <cstdint>
#include <functional>
#include <string>

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

    // Service control; the wait is bounded by timeout_ms.
    bool service_exists();
    bool stop_service(unsigned timeout_ms, std::string& error);
    bool start_service(std::string& error);
    bool register_service(const std::string& program_dir, std::string& error);
    bool remove_service(std::string& error);

    // Asks the trays of this session to close and waits for them.
    void close_trays(unsigned timeout_ms);

    // Receives the successive pieces of a file; returns false to stop.
    typedef std::function<bool(const void* data, size_t size)> chunk_writer;

    /*
        Writes a file of the installation, whose content body gives to the writer it
        receives: when the target is in use (a program that runs in another session), the
        old file is renamed aside and deleted at the next restart of Windows; reboot_needed
        then becomes true.
    */
    bool write_program_file(const std::string& path, const std::function<bool(const chunk_writer&)>& body,
        bool& reboot_needed, std::string& error);

    // Deletes the files left aside by an earlier update.
    void remove_old_files(const std::string& program_dir);

    // Copies this program into the installation, without its payload, as the uninstaller.
    bool install_uninstaller(const std::string& program_dir, std::string& error);

    // The data folder with its permissions: SYSTEM and administrators full control,
    // users read (the tray reads the certificate of the authority and the token).
    bool prepare_data_dir(std::string& error);

    // Merges the options into settings.json, keeping the other settings.
    bool write_settings(const install_options& o, std::string& error);

    /*
        Copies the offline bundle the installer comes with, on a medium, into the import
        folder of the service, which installs it at its start: the signed catalog next to
        the installer (catalog.json and catalog.json.sig) and the models of its "models"
        folder.  Returns the number of models; progress receives the name of each one.
    */
    int stage_bundle(const std::function<void(const std::string&)>& progress);

    // The certificates of the local server, and the trust of the authority by Windows.
    bool create_certificates(std::string& error);
    bool trust_authority(std::string& error);
    void untrust_authority();

    // Registry: the entry of the installed programs and the start of the tray.
    bool register_uninstall(const std::string& program_dir, uint64_t size_bytes, std::string& error);
    void remove_uninstall();
    void set_autostart(const std::string& program_dir, bool on);

    // Shortcuts of the Start menu and of the desktop.
    void create_shortcuts(const std::string& program_dir, bool desktop);
    void remove_shortcuts();

    // Starts a program with the rights of the user of the desktop, from an elevated program.
    bool launch_unelevated(const std::string& program, const std::string& args);

    // Waits until the local server accepts connections.
    bool wait_for_server(int port, unsigned timeout_ms);

    // Removes a folder and its content; what is in use is deleted at the next restart.
    void remove_folder(const std::string& path);

    // Log of the installer in the data folder.
    void log(const std::string& line);

}
}

#endif // ALBIREO_WINDOWS_SETUP_ACTIONS_H_
