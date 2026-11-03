// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_STORE_H_
#define PSEUDALIS_HOST_STORE_H_

#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <utility>
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

    struct entry_record
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                An entry of a protected archive that the database records: a file extracted
                from that archive is known by the SHA-256 of its bytes, and its original is the
                entry original of the archive kept in the vault.  The paths of the entries of an
                archive inside the archive follow its own and '!' (engine_api.h).
        !*/
        long long record = 0;           // the record of the archive
        std::string name;               // its path in the protected archive
        std::string original;           // its path in the original archive
        std::string sha256;             // of the entry written
        std::string kind;
    };

    struct picture_record
    {
        std::string id;                 // 16 hexadecimal digits of a keyed digest (picture_found)
        std::string sha256;
        std::string media_type;
        std::string blob;               // the picture in the vault
        std::string description;        // the description of Albireo encrypted in the vault, or empty
        int64_t created = 0;
    };

    struct folder_file
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A file of the folder of the drive (drive.h) as the service left it: processed or
                left as it is, with the size and the time of last write that tell whether it
                changed since.  A folder of the drive is recorded too, in the state "folder", once
                its name is protected.
        !*/
        std::string folder;             // "drive"
        std::string path;               // within the folder, '/' between its parts
        long long size = 0;
        int64_t written = 0;            // the time of its last write, as the file system gives it
        std::string sha256;
        long long record = 0;           // the record of its original
        std::string state;              // "protected", "kept" or "folder"
    };

    class store
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The database of the service, pseudalis.db in a folder of SYSTEM and the
                administrators: the files protected and their originals, the entries of the
                protected archives, the pictures and their descriptions, the protected files
                whose original was forgotten, the files of the drive and the recent events.
                Every method takes the lock of the connection, so that the threads of the
                server and of the drive share it.
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
        // The record of a protected file; among several of the same content, the one whose protected name is name.
        bool find_by_output(const std::string& sha256, file_record& r, const std::string& name = std::string());
        bool find_by_object(const std::string& object_id, file_record& r);
        // Whether a protected file of this size is recorded, which a file must be to be one.
        bool output_size_known(long long size);
        std::vector<file_record> list_files(long long limit, long long offset,
            const std::string& source = std::string());
        long long count_files();
        // The blobs the records and the pictures name.
        std::set<std::string> referenced_blobs();
        void set_file_location(long long id, const std::string& object_id, const std::string& path);
        // Records the time the file of a record left the drive, deleted or replaced.
        void set_file_removed(long long id, int64_t when);
        // The records whose file left the drive at when or before.
        std::vector<long long> files_removed_before(int64_t when);
        /*!
            ensures
                - Removes the record, its entries and its links to its pictures; blobs receives
                  the blobs nothing refers to any more: its original, and the pictures no other
                  file shows.
                - With a reason ("forgotten", "expired"), the protected file of the record and
                  the entries of a protected archive are remembered as forgotten
                  (find_forgotten()), in the same transaction: a copy extracted from the archive
                  is not restored either, unless a record of its own holds it.
        !*/
        bool remove_file(long long id, std::vector<std::string>& blobs, const std::string& reason = std::string());
        // The last original forgotten of a protected file, by its SHA-256 or its object identifier.
        bool find_forgotten(const std::string& sha256, const std::string& object_id, int64_t& time,
            std::string& reason);

        // The entries of the protected archive of record (entry_record).
        void add_entries(long long record, const std::vector<entry_record>& entries);
        // The entry written with this SHA-256, of the latest record that holds one.
        bool find_entry(const std::string& sha256, entry_record& e);
        // The entries inside the entry e of its record, an archive, their paths taken from inside it.
        std::vector<entry_record> entries_inside(const entry_record& e);

        bool get_picture(const std::string& id, picture_record& r);
        void add_picture(const picture_record& r);
        /*
            Gives the picture id the description blob description, before receiving the blob it
            named; false when no picture has this identifier.
        */
        bool set_picture_description(const std::string& id, const std::string& description, std::string& before);
        // Removes the pictures that no file names, the blobs of which the vault then drops; their count.
        size_t remove_unlinked_pictures();
        void link_picture(long long file, const std::string& picture);
        std::vector<std::string> pictures_of(long long file);

        // The files of the folder of the drive as the service left them; their paths compare without case.
        std::vector<folder_file> folder_files(const std::string& folder);
        void set_folder_file(const folder_file& f);
        void remove_folder_file(const std::string& folder, const std::string& path);
        // The files and the bytes of the drive, its folders left out.
        void drive_usage(long long& files, long long& bytes);

        void add_event(const std::string& kind, const std::string& path, const std::string& message);
        json events(long long since, int limit = 100);

        std::string value(const std::string& key);
        void set_value(const std::string& key, const std::string& value);

        /*!
            ensures
                - Records the substitutes the engine wrote in the names and the properties of
                  the files, each (substitute, tag), a tag and never a value in clear, so
                  that their values come back at the restoration of any copy of the file.
        !*/
        void add_substitutes(const std::vector<std::pair<std::string, std::string>>& made);
        // The tags recorded for a substitute, the most recent first, eight at most.
        std::vector<std::string> tags_for(const std::string& substitute);

    private:
        void exec(const char* sql);
        std::mutex m_;
        sqlite3* db_ = nullptr;
    };
}

#endif // PSEUDALIS_HOST_STORE_H_
