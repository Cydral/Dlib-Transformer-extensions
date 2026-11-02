// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_STORE_H_
#define PSEUDALIS_HOST_STORE_H_

#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "../common/json.h"

struct sqlite3;

namespace pseudalis
{
    struct file_record
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A file the service protected and keeps the original of: the protected
                file is found again by the SHA-256 of its bytes, which every copy shares,
                or by the NTFS object identifier its file received (a file replaced in
                place, renamed or moved on its volume since), whatever its name.
        !*/
        long long id = 0;
        int64_t created = 0;
        std::string source;             // "drive", "shell", "web" or "api"
        std::string original_name;
        std::string output_name;
        std::string original_sha256;
        std::string output_sha256;
        std::string original_blob;      // the original in the vault
        long long original_size = 0;
        long long output_size = 0;
        std::string kind;
        std::string object_id;          // 32 hexadecimal digits, or empty
        std::string path;               // where the protected file was written, when known
        int64_t removed = 0;            // when its file left the drive (Unix time), else 0
    };

    struct picture_record
    {
        std::string id;                 // 16 hexadecimal digits, the start of sha256
        std::string sha256;
        std::string media_type;
        std::string blob;               // the picture in the vault
        std::string description;        // protected
        int64_t created = 0;
    };

    struct drive_node
    {
        long long id = 0;               // 1 is the root
        long long parent = 0;
        std::string name;
        bool directory = false;
        long long file = 0;             // the record of the original, 0 for a folder or a file without one
        std::string blob;               // the protected content in the vault
        long long size = 0;
        uint32_t attributes = 0;
        int64_t created = 0, accessed = 0, written = 0, changed = 0;   // FILETIME units
    };

    class store
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The database of the service, pseudalis.db in a folder of SYSTEM and the
                administrators: the files protected and their originals, the pictures and
                their descriptions, the tree of the drive and the recent events.  Every
                method takes the lock of the connection, so that the threads of the server
                and of the drive share it.
            THROWS
                std::runtime_error when the database cannot be opened or written.
        !*/
    public:
        explicit store(const std::string& path);
        ~store();
        store(const store&) = delete;
        store& operator=(const store&) = delete;

        long long add_file(const file_record& r);
        bool get_file(long long id, file_record& r);
        bool find_by_output(const std::string& sha256, file_record& r);
        bool find_by_object(const std::string& object_id, file_record& r);
        std::vector<file_record> list_files(long long limit, long long offset,
            const std::string& source = std::string());
        long long count_files();
        // The blobs the records, the pictures and the files of the drive name.
        std::set<std::string> referenced_blobs();
        void set_file_location(long long id, const std::string& object_id, const std::string& path);
        // Records the time the file of a record left the drive, deleted or replaced.
        void set_file_removed(long long id, int64_t when);
        // The records whose file left the drive at when or before.
        std::vector<long long> files_removed_before(int64_t when);
        /*!
            ensures
                - Removes the record and its links to its pictures; blobs receives the blobs
                  nothing refers to any more: its original, and the pictures no other file
                  shows.
                - With a reason ("forgotten", "expired"), the protected file of the record is
                  remembered as forgotten (find_forgotten()), in the same transaction.
        !*/
        bool remove_file(long long id, std::vector<std::string>& blobs, const std::string& reason = std::string());
        // The last original forgotten of a protected file, by its SHA-256 or its object identifier.
        bool find_forgotten(const std::string& sha256, const std::string& object_id, int64_t& time,
            std::string& reason);

        bool get_picture(const std::string& id, picture_record& r);
        bool find_picture(const std::string& sha256, picture_record& r);
        void add_picture(const picture_record& r);
        void link_picture(long long file, const std::string& picture);
        std::vector<std::string> pictures_of(long long file);

        bool node(long long id, drive_node& n);
        bool child(long long parent, const std::string& name, drive_node& n);
        std::vector<drive_node> children(long long parent);
        long long add_node(const drive_node& n);
        void update_node(const drive_node& n);
        // Writes n and deletes the node replaced, which n takes the place of, both or neither.
        void replace_node(const drive_node& n, long long replaced);
        void remove_node(long long id);
        // The files and the bytes of the drive.
        void drive_usage(long long& files, long long& bytes);

        void add_event(const std::string& kind, const std::string& path, const std::string& message);
        json events(long long since, int limit = 100);

        std::string value(const std::string& key);
        void set_value(const std::string& key, const std::string& value);

    private:
        void exec(const char* sql);
        void write_node(const drive_node& n);   // m_ held
        std::mutex m_;
        sqlite3* db_ = nullptr;
    };
}

#endif // PSEUDALIS_HOST_STORE_H_
