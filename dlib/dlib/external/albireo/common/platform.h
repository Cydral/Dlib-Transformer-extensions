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
        ~/.local/share/albireo elsewhere.  On Windows the variable is read from the
        environment of the machine, the one the services receive: the variable of a user
        steers neither the installer nor the icon away from the service.  The folder the
        installer records (DataDir of HKLM\Software\Albireo) comes first.
    */
    std::string data_dir();

    /*
        Makes data_dir() return p in this process from then on.  The test programs call it, so
        that they never write in the data folder of an installed service; the programs of
        Albireo never do, and no variable of the environment of a user can do it.
    */
    void set_data_dir(const std::string& p);

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
    // Removes a folder and everything under it; a link goes alone, what it leads to stays.
    bool remove_tree(const std::string& p);
    std::vector<std::string> list_dir(const std::string& p);

    // Whether p itself is a link: a symbolic link, a junction or a mount point.
    bool is_link(const std::string& p);

    /*
        The path of p with its links resolved, as the system opens it: a path on a drive
        letter, or \\server\share\... on a network share, on Windows.  An empty string
        when p cannot be opened.
    */
    std::string final_path(const std::string& p);

    // Whether p reaches its file or folder through no link: its final path is p itself.
    bool direct_path(const std::string& p);

    // Whether path, links resolved, is folder or lies under it; path may not exist yet.
    bool path_inside(const std::string& path, const std::string& folder);

    /*
        Whether the owner of p itself, a link included, is SYSTEM, the administrators, the
        installer of Windows (TrustedInstaller) or the account of this program, which has
        their rights where the function serves (the installer, the service).  Always true
        elsewhere than on Windows.
    */
    bool owned_by_system(const std::string& p);

    /*
        Makes p a folder that only SYSTEM and the administrators change, and that the
        users read: the folders where the service and the installer write with their
        rights.  On Windows:
            - a missing folder is created so;
            - the links on its path must belong to SYSTEM or the administrators, and the
              folder they lead to must be on a local drive;
            - the folder must belong to them too, or be empty: it is then taken over;
            - unless it has these permissions already, it receives them, and what other
              accounts placed in it, links included, is renamed with the suffix
              ".untrusted-<random>", which no program reads; with check_content, what it
              holds is checked even when it had these permissions.
        Returns false, with the reason in error, when the folder cannot be made so.
        Elsewhere the folder is only created.
    */
    bool secure_folder(const std::string& p, std::string& error, bool check_content = false);

    /*
        Whether only SYSTEM and the administrators can change what the folder p holds: p is
        a folder and no link, its owner is one of the accounts of owned_by_system(), and no
        entry of its permissions lets another account write in it, delete in it or change
        its permissions, whether the entry applies to p or to what it holds.  Always true
        elsewhere than on Windows.
    */
    bool folder_admin_only(const std::string& p);

    class link_free_file
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A file opened for reading through no link.  Each folder on the way from a
                trusted folder to the file is opened within the folder before it, and the
                file within the last one; a symbolic link, a junction or a mount point among
                them, the file included, fails the opening, so that a link placed or swapped
                while the file opens never leads elsewhere.  On Windows the file is shared
                for reading only while it is open: what is read is what was checked.
        !*/
    public:
        link_free_file() {}
        ~link_free_file() { close(); }
        link_free_file(const link_free_file&) = delete;
        link_free_file& operator=(const link_free_file&) = delete;

        /*!
            ensures
                - Opens path, which lies under folder.  folder itself is opened as its path
                  leads, links included; an empty folder stands for the root of the drive of
                  path.
                - Returns false otherwise, with why: "link" (a link on the way, the file
                  included), "missing", or "invalid" (path outside folder, a name "." or "..",
                  an entry that is no regular file, or another error).
        !*/
        bool open(const std::string& path, const std::string& folder, std::string& why);
        bool is_open() const;
        uint64_t size() const { return size_; }
        // The number of names of the file: more than one for a file with hard links.
        uint32_t links() const { return links_; }
        // The volume and the index of the file, equal for two names of the same file.
        const std::string& identity() const { return identity_; }
        // Reads up to n bytes; returns the bytes read, 0 at the end of the file, -1 on an error.
        long long read(char* data, size_t n);
        bool seek(uint64_t offset);
        void close();

    private:
#if defined(_WIN32)
        void* h_ = nullptr;
#else
        int fd_ = -1;
#endif
        uint64_t size_ = 0;
        uint32_t links_ = 0;
        std::string identity_;
    };

    // Reads a file of at most max_bytes through no link, as link_free_file opens it.
    bool read_file_link_free(const std::string& path, const std::string& folder, std::string& data,
        uint64_t max_bytes);

    // Whether the folder path, under folder, is reached through no link, as link_free_file opens a file.
    bool link_free_folder(const std::string& path, const std::string& folder);

    // The identity of the file p itself, a link not followed, as link_free_file::identity() gives it, or
    // an empty string for a link, a folder or a missing entry.
    std::string file_identity(const std::string& p);

    // Whether an open file, a handle on Windows and a descriptor elsewhere, lies at path reached through
    // no link: a folder of path replaced by a link after a check shows.
#if defined(_WIN32)
    bool handle_at(void* handle, const std::string& path);
#else
    bool handle_at(int fd, const std::string& path);
#endif

    class new_file
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A file created for writing that must not exist yet: a name taken meanwhile,
                by a link for instance, fails the creation, so that the writes stay in the
                file created.  A file that close() does not complete is left to the caller,
                which removes it.
        !*/
    public:
        new_file() {}
        ~new_file() { close(); }
        new_file(const new_file&) = delete;
        new_file& operator=(const new_file&) = delete;

        bool create(const std::string& path);
        bool write(const char* data, size_t n);
        // Writes the data to the disk and closes the file; false when a write or the flush failed.
        bool close();

    private:
#if defined(_WIN32)
        void* h_ = nullptr;
#else
        int fd_ = -1;
#endif
        bool failed_ = false;
    };

    bool read_file(const std::string& p, std::string& data, uint64_t max_bytes = 256u << 20);
    /*
        Writes through a temporary file whose data reaches the disk before it is renamed over
        the target, so that a crash or a power cut leaves the old file or the new one, never
        an empty or half written one.
    */
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

    struct process_options
    {
        // Written to the standard input of the program, which closes once it is written.
        const std::string* input = nullptr;
        /*
            On Windows, the program runs without console, with a copy of the token of this
            process without its privileges and of low integrity, alone in a job that ends it
            beyond memory_limit bytes and when the job closes, and that refuses it any child
            process; it does not run when Windows refuses one of these steps.  Elsewhere its
            address space is bounded by memory_limit, it writes no file and leaves no core
            file, and on Linux it gains no privilege by running another program.
        */
        bool restricted = false;
        size_t memory_limit = 0;
        // The folder the program starts in, that of this process when empty.
        std::string directory;
    };

    /*
        Runs a program with arguments and waits for it; returns its exit code, or -1.  The
        program is ended when it runs beyond timeout_ms, or as soon as stop becomes true.
        output receives what the program writes to its standard output and error.
    */
    int run_process(const std::string& program, const std::vector<std::string>& args, std::string* output = nullptr,
        unsigned timeout_ms = 60000, const std::atomic<bool>* stop = nullptr, const process_options* options = nullptr);

    /*
        Ends the process with code once its standard outputs are flushed, without unloading
        its libraries and without the destructors of its static objects.  A process that
        loaded the engine of a device ends this way: the runtime of OpenVINO or of CUDA would
        otherwise be torn down while its library unloads, which Windows does under its loader
        lock, where the runtime waits for threads that cannot end.
    */
    [[noreturn]] void end_process(int code);

}

#endif // ALBIREO_COMMON_PLATFORM_H_
