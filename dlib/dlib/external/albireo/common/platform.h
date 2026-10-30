// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_COMMON_PLATFORM_H_
#define ALBIREO_COMMON_PLATFORM_H_

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

/*
    Files, folders and processes, with UTF-8 paths on every platform.  On Windows the
    paths are converted to UTF-16 at the system calls, so that a user name or a folder
    with accents, or in another script, works everywhere.
*/

namespace albireo
{

#if defined(_WIN32)
    std::wstring widen(const std::string& utf8);
    std::string narrow(const std::wstring& utf16);
#endif

    // The program running, and its folder.
    std::string executable_path();
    std::string executable_dir();

    /*
        The folder of the data shared by the programs: the settings, the certificates,
        the database, the logs and, unless the settings say otherwise, the models.
        $ALBIREO_HOME when it is set, otherwise %ProgramData%\Albireo on Windows and
        ~/.local/share/albireo elsewhere.
    */
    std::string data_dir();

    std::string join_path(const std::string& a, const std::string& b);
    std::string parent_path(const std::string& p);
    std::string file_name(const std::string& p);

    bool file_exists(const std::string& p);
    bool dir_exists(const std::string& p);
    bool make_dirs(const std::string& p);
    uint64_t file_size(const std::string& p);
    // Whether another program still writes a file: held open for writing on Windows,
    // modified during the last 30 seconds elsewhere.
    bool file_being_written(const std::string& p);
    bool remove_file(const std::string& p);
    // Moves a file, replacing the destination.
    bool rename_file(const std::string& from, const std::string& to);
    // Copies a file, replacing the destination; a copy that cancel ends leaves no
    // destination behind.
    bool copy_file(const std::string& from, const std::string& to, const std::atomic<bool>* cancel = nullptr);
    // Removes a folder and everything under it.
    bool remove_tree(const std::string& p);
    std::vector<std::string> list_dir(const std::string& p);

    bool read_file(const std::string& p, std::string& data, uint64_t max_bytes = 256u << 20);
    // Writes through a temporary file renamed over the target, so that a crash never
    // leaves a half written file behind.
    bool write_file_atomic(const std::string& p, const std::string& data);

    /*
        Writes a file that only the account of the program and the administrators can
        read (mode 0600 elsewhere than on Windows), through a temporary file renamed over
        the target.  With interactive_readers, the users of the console may read it too on
        Windows: the token of the tray is written so.
    */
    bool write_file_restricted(const std::string& p, const std::string& data, bool interactive_readers = false);

    // Bytes available to the user on the volume of a path.
    uint64_t free_disk_space(const std::string& p);

    // Seconds since 1970, and the ISO 8601 form of a time in UTC.
    int64_t now_unix();
    std::string iso_time(int64_t t);

    // A random hexadecimal string of n characters, from the generator of the system.
    std::string random_hex(size_t n);

    // Compares versions "1.2.3": negative, zero or positive as a is older, equal or newer.
    int compare_versions(const std::string& a, const std::string& b);

    // Runs a program with arguments and waits for it; returns its exit code, or -1.
    int run_process(const std::string& program, const std::vector<std::string>& args, std::string* output = nullptr,
        unsigned timeout_ms = 60000);

}

#endif // ALBIREO_COMMON_PLATFORM_H_
