// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "store.h"

#include <sqlite3.h>

#include <stdexcept>

#include "../common/platform.h"

namespace pseudalis
{
    namespace
    {
        const char* const schema =
            "PRAGMA journal_mode=WAL;"
            // Each commit reaches the disk, as the blobs of the vault it names do before it.
            "PRAGMA synchronous=FULL;"
            "PRAGMA foreign_keys=ON;"
            // AUTOINCREMENT: the identifier of a record deleted is never given again, since the drive
            // and the copies of the files may still name it.
            "CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY AUTOINCREMENT, created INTEGER NOT NULL,"
            " source TEXT NOT NULL, original_name TEXT NOT NULL, output_name TEXT NOT NULL,"
            " original_sha256 TEXT NOT NULL, output_sha256 TEXT NOT NULL, original_blob TEXT NOT NULL,"
            " original_size INTEGER NOT NULL, output_size INTEGER NOT NULL, kind TEXT NOT NULL,"
            " object_id TEXT NOT NULL DEFAULT '', path TEXT NOT NULL DEFAULT '', removed INTEGER NOT NULL DEFAULT 0);"
            "CREATE INDEX IF NOT EXISTS files_output ON files(output_sha256);"
            "CREATE INDEX IF NOT EXISTS files_object ON files(object_id);"
            "CREATE INDEX IF NOT EXISTS files_output_size ON files(output_size);"
            "CREATE TABLE IF NOT EXISTS pictures(id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, media_type TEXT NOT NULL,"
            " blob TEXT NOT NULL, description TEXT NOT NULL, created INTEGER NOT NULL);"
            "CREATE INDEX IF NOT EXISTS pictures_sha ON pictures(sha256);"
            "CREATE TABLE IF NOT EXISTS file_pictures(file INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,"
            " picture TEXT NOT NULL, PRIMARY KEY(file, picture));"
            // The entries of the protected archives, by which a file extracted from one is known.
            "CREATE TABLE IF NOT EXISTS entries(record INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,"
            " name TEXT NOT NULL, original TEXT NOT NULL, sha256 TEXT NOT NULL, kind TEXT NOT NULL,"
            " PRIMARY KEY(record, name));"
            "CREATE INDEX IF NOT EXISTS entries_sha ON entries(sha256);"
            // The originals forgotten, by the protected files that showed them: restored from their tags since.
            "CREATE TABLE IF NOT EXISTS forgotten(id INTEGER PRIMARY KEY AUTOINCREMENT, time INTEGER NOT NULL,"
            " reason TEXT NOT NULL, output_sha256 TEXT NOT NULL, object_id TEXT NOT NULL DEFAULT '');"
            "CREATE INDEX IF NOT EXISTS forgotten_output ON forgotten(output_sha256);"
            "CREATE INDEX IF NOT EXISTS forgotten_object ON forgotten(object_id);"
            "CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, time INTEGER NOT NULL, kind TEXT NOT NULL,"
            " path TEXT NOT NULL, message TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            /* The substitutes written in the names and the properties of the files, values without
               their tags, with the tags they stand for: tags, never values in clear. */
            "CREATE TABLE IF NOT EXISTS substitutes(substitute TEXT NOT NULL, tag TEXT NOT NULL,"
            " created INTEGER NOT NULL, PRIMARY KEY(substitute, tag)) WITHOUT ROWID;"
            /* The substitutes written in the text of a file protected without tags, with the tags
               they stand for and the number of places that hold each, and those of its original that a
               protection without tags wrote before (inherited): they leave with the record. */
            "CREATE TABLE IF NOT EXISTS text_substitutes(record INTEGER NOT NULL REFERENCES files(id)"
            " ON DELETE CASCADE, substitute TEXT NOT NULL, tag TEXT NOT NULL, places INTEGER NOT NULL,"
            " inherited INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(record, substitute, tag)) WITHOUT ROWID;"
            // The files of the folder of the drive as the service left them.
            "CREATE TABLE IF NOT EXISTS folder_files(folder TEXT NOT NULL, path TEXT NOT NULL COLLATE NOCASE,"
            " size INTEGER NOT NULL, written INTEGER NOT NULL, sha256 TEXT NOT NULL, record INTEGER NOT NULL,"
            " state TEXT NOT NULL, PRIMARY KEY(folder, path));";

        // A prepared statement, finalized at the end of its scope.
        class statement
        {
        public:
            statement(sqlite3* db, const char* sql) : db_(db)
            {
                if (sqlite3_prepare_v2(db, sql, -1, &s_, nullptr) != SQLITE_OK)
                    throw std::runtime_error(std::string("database: ") + sqlite3_errmsg(db));
            }
            ~statement() { sqlite3_finalize(s_); }
            statement& bind(int i, const std::string& v)
            {
                sqlite3_bind_text(s_, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT);
                return *this;
            }
            statement& bind(int i, long long v)
            {
                sqlite3_bind_int64(s_, i, v);
                return *this;
            }
            // True with a row, false at the end.
            bool step()
            {
                const int rc = sqlite3_step(s_);
                if (rc == SQLITE_ROW) return true;
                if (rc == SQLITE_DONE) return false;
                throw std::runtime_error(std::string("database: ") + sqlite3_errmsg(db_));
            }
            std::string text(int i)
            {
                const unsigned char* t = sqlite3_column_text(s_, i);
                return t ? std::string(reinterpret_cast<const char*>(t),
                                       static_cast<size_t>(sqlite3_column_bytes(s_, i)))
                         : std::string();
            }
            long long integer(int i) { return sqlite3_column_int64(s_, i); }
            // Ready to run again with other values.
            void reset()
            {
                sqlite3_reset(s_);
                sqlite3_clear_bindings(s_);
            }

        private:
            sqlite3* db_;
            sqlite3_stmt* s_ = nullptr;
        };

        const char* const file_columns = "id, created, source, original_name, output_name, original_sha256,"
                                         " output_sha256, original_blob, original_size, output_size, kind,"
                                         " object_id, path, removed, tags";

        file_record read_file_row(statement& s)
        {
            file_record r;
            r.id = s.integer(0);
            r.created = s.integer(1);
            r.source = s.text(2);
            r.original_name = s.text(3);
            r.output_name = s.text(4);
            r.original_sha256 = s.text(5);
            r.output_sha256 = s.text(6);
            r.original_blob = s.text(7);
            r.original_size = s.integer(8);
            r.output_size = s.integer(9);
            r.kind = s.text(10);
            r.object_id = s.text(11);
            r.path = s.text(12);
            r.removed = s.integer(13);
            r.tags = s.integer(14) != 0;
            return r;
        }
    }

    store::store(const std::string& path)
    {
        if (sqlite3_open_v2(path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                            nullptr) != SQLITE_OK)
        {
            const std::string e = db_ ? sqlite3_errmsg(db_) : "out of memory";
            sqlite3_close(db_);
            db_ = nullptr;
            throw std::runtime_error("database " + path + ": " + e);
        }
        sqlite3_busy_timeout(db_, 5000);
        try
        {
            exec(schema);
            /*
                The version of the schema, from which the migrations start.  2: the description of
                a picture names a blob of the vault (32 hexadecimal digits), where version 1 kept it
                in clear under an identifier that the SHA-256 alone gave; nothing reads those any
                more, the pictures of this version taking keyed identifiers, and they are erased.
            */
            long long version = 0;
            {
                statement v(db_, "PRAGMA user_version");
                if (v.step()) version = v.integer(0);
            }
            if (version < 2)
            {
                exec("BEGIN IMMEDIATE");
                try
                {
                    exec("UPDATE pictures SET description = '' WHERE description <> ''"
                         " AND (length(description) <> 32 OR description GLOB '*[^0-9a-f]*');"
                         "PRAGMA user_version = 2;"
                         "COMMIT;");
                }
                catch (...)
                {
                    sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
                    throw;
                }
            }
            /*
                3: whether the protected file of a record, or of an original forgotten, holds the tags
                of its values, false for a file protected without tags; every file of the versions
                before held them.
            */
            if (version < 3)
            {
                // A column already there, a database a later version wrote and then marked older, stays.
                auto has_column = [this](const char* table, const char* column) {
                    statement c(db_, (std::string("PRAGMA table_info(") + table + ")").c_str());
                    while (c.step())
                        if (c.text(1) == column) return true;
                    return false;
                };
                exec("BEGIN IMMEDIATE");
                try
                {
                    if (!has_column("files", "tags"))
                        exec("ALTER TABLE files ADD COLUMN tags INTEGER NOT NULL DEFAULT 1;");
                    if (!has_column("forgotten", "tags"))
                        exec("ALTER TABLE forgotten ADD COLUMN tags INTEGER NOT NULL DEFAULT 1;");
                    exec("PRAGMA user_version = 3;"
                         "COMMIT;");
                }
                catch (...)
                {
                    sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
                    throw;
                }
            }
        }
        catch (...)
        {
            // The destructor of an object whose constructor failed never runs.
            sqlite3_close(db_);
            db_ = nullptr;
            throw;
        }
    }

    store::~store()
    {
        sqlite3_close(db_);
    }

    void store::exec(const char* sql)
    {
        char* error = nullptr;
        if (sqlite3_exec(db_, sql, nullptr, nullptr, &error) != SQLITE_OK)
        {
            const std::string e = error ? error : "error";
            sqlite3_free(error);
            throw std::runtime_error("database: " + e);
        }
    }

    long long store::add_file(const file_record& r)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_,
            "INSERT INTO files(created, source, original_name, output_name, original_sha256, output_sha256,"
                         " original_blob, original_size, output_size, kind, object_id, path, tags)"
                         " VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)");
        s.bind(1, static_cast<long long>(r.created ? r.created : now_unix())).bind(2, r.source).bind(3, r.original_name)
            .bind(4, r.output_name).bind(5, r.original_sha256).bind(6, r.output_sha256).bind(7, r.original_blob)
            .bind(8, r.original_size).bind(9, r.output_size).bind(10, r.kind).bind(11, r.object_id).bind(12, r.path)
            .bind(13, static_cast<long long>(r.tags ? 1 : 0));
        s.step();
        return sqlite3_last_insert_rowid(db_);
    }

    bool store::get_file(long long id, file_record& r)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, (std::string("SELECT ") + file_columns + " FROM files WHERE id = ?1").c_str());
        s.bind(1, id);
        if (!s.step()) return false;
        r = read_file_row(s);
        return true;
    }

    bool store::find_by_output(const std::string& sha256, file_record& r, const std::string& name)
    {
        std::lock_guard<std::mutex> lock(m_);
        // Several files of the same content: the one that bears the name, else the last one.
        statement s(db_, (std::string("SELECT ") + file_columns +
                          " FROM files WHERE output_sha256 = ?1 ORDER BY output_name = ?2 COLLATE NOCASE DESC,"
                          " id DESC LIMIT 1").c_str());
        s.bind(1, sha256).bind(2, name);
        if (!s.step()) return false;
        r = read_file_row(s);
        return true;
    }

    bool store::output_size_known(long long size)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT 1 FROM files WHERE output_size = ?1 LIMIT 1");
        s.bind(1, size);
        return s.step();
    }

    bool store::find_by_object(const std::string& object_id, file_record& r)
    {
        if (object_id.empty()) return false;
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, (std::string("SELECT ") + file_columns +
                          " FROM files WHERE object_id = ?1 ORDER BY id DESC LIMIT 1").c_str());
        s.bind(1, object_id);
        if (!s.step()) return false;
        r = read_file_row(s);
        return true;
    }

    std::vector<file_record> store::list_files(long long limit, long long offset, const std::string& source)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, (std::string("SELECT ") + file_columns +
                          " FROM files WHERE ?3 = '' OR source = ?3 ORDER BY id DESC LIMIT ?1 OFFSET ?2").c_str());
        s.bind(1, limit).bind(2, offset).bind(3, source);
        std::vector<file_record> out;
        while (s.step()) out.push_back(read_file_row(s));
        return out;
    }

    long long store::count_files(const std::string& source)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT COUNT(*) FROM files WHERE ?1 = '' OR source = ?1");
        s.bind(1, source);
        return s.step() ? s.integer(0) : 0;
    }

    long long store::originals_bytes()
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT COALESCE(SUM(original_size), 0) FROM files");
        return s.step() ? s.integer(0) : 0;
    }

    std::vector<std::pair<long long, long long>> store::oldest_files(long long limit)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT id, original_size FROM files ORDER BY created, id LIMIT ?1");
        s.bind(1, limit);
        std::vector<std::pair<long long, long long>> out;
        while (s.step()) out.push_back(std::make_pair(s.integer(0), s.integer(1)));
        return out;
    }

    std::set<std::string> store::referenced_blobs()
    {
        std::lock_guard<std::mutex> lock(m_);
        std::set<std::string> found;
        statement s(db_, "SELECT original_blob FROM files UNION SELECT blob FROM pictures"
                         " UNION SELECT description FROM pictures WHERE description <> ''");
        while (s.step()) found.insert(s.text(0));
        return found;
    }

    void store::set_file_location(long long id, const std::string& object_id, const std::string& path)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "UPDATE files SET object_id = ?2, path = ?3 WHERE id = ?1");
        s.bind(1, id).bind(2, object_id).bind(3, path);
        s.step();
    }

    void store::set_file_removed(long long id, int64_t when)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "UPDATE files SET removed = ?2 WHERE id = ?1");
        s.bind(1, id).bind(2, static_cast<long long>(when));
        s.step();
    }

    std::vector<long long> store::files_removed_before(int64_t when)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT id FROM files WHERE removed > 0 AND removed <= ?1 ORDER BY id");
        s.bind(1, static_cast<long long>(when));
        std::vector<long long> out;
        while (s.step()) out.push_back(s.integer(0));
        return out;
    }

    bool store::remove_file(long long id, std::vector<std::string>& blobs, const std::string& reason)
    {
        std::lock_guard<std::mutex> lock(m_);
        std::string original;
        {
            statement s(db_, "SELECT original_blob FROM files WHERE id = ?1");
            s.bind(1, id);
            if (!s.step()) return false;
            original = s.text(0);
        }
        std::vector<std::string> pictures;
        {
            statement s(db_, "SELECT picture FROM file_pictures WHERE file = ?1");
            s.bind(1, id);
            while (s.step()) pictures.push_back(s.text(0));
        }
        exec("BEGIN IMMEDIATE");
        try
        {
            if (!reason.empty())
            {
                statement f(db_, "INSERT INTO forgotten(time, reason, output_sha256, object_id, tags)"
                                 " SELECT ?2, ?3, output_sha256, object_id, tags FROM files WHERE id = ?1");
                f.bind(1, id).bind(2, static_cast<long long>(now_unix())).bind(3, reason);
                f.step();
                statement fe(db_, "INSERT INTO forgotten(time, reason, output_sha256, object_id, tags)"
                                  " SELECT DISTINCT ?2, ?3, e.sha256, '', f.tags FROM entries e, files f"
                                  " WHERE e.record = ?1 AND f.id = ?1");
                fe.bind(1, id).bind(2, static_cast<long long>(now_unix())).bind(3, reason);
                fe.step();
            }
            statement de(db_, "DELETE FROM entries WHERE record = ?1");
            de.bind(1, id);
            de.step();
            statement dt(db_, "DELETE FROM text_substitutes WHERE record = ?1");
            dt.bind(1, id);
            dt.step();
            statement d(db_, "DELETE FROM files WHERE id = ?1");
            d.bind(1, id);
            d.step();
            statement l(db_, "DELETE FROM file_pictures WHERE file = ?1");
            l.bind(1, id);
            l.step();
            for (const std::string& p : pictures)
            {
                statement used(db_, "SELECT COUNT(*) FROM file_pictures WHERE picture = ?1");
                used.bind(1, p);
                if (used.step() && used.integer(0) > 0) continue;
                statement b(db_, "SELECT blob, description FROM pictures WHERE id = ?1");
                b.bind(1, p);
                if (b.step())
                {
                    blobs.push_back(b.text(0));
                    if (!b.text(1).empty()) blobs.push_back(b.text(1));
                }
                statement dp(db_, "DELETE FROM pictures WHERE id = ?1");
                dp.bind(1, p);
                dp.step();
            }
            exec("COMMIT");
        }
        catch (...)
        {
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
            throw;
        }
        if (!original.empty()) blobs.push_back(original);
        return true;
    }

    bool store::find_forgotten(const std::string& sha256, const std::string& object_id, int64_t& time,
                               std::string& reason, bool* tags)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT time, reason, tags FROM forgotten WHERE output_sha256 = ?1"
                         " OR (?2 != '' AND object_id = ?2) ORDER BY id DESC LIMIT 1");
        s.bind(1, sha256).bind(2, object_id);
        if (!s.step()) return false;
        time = s.integer(0);
        reason = s.text(1);
        if (tags) *tags = s.integer(2) != 0;
        return true;
    }

    void store::add_text_substitutes(long long record, const std::vector<std::pair<std::string, std::string>>& made)
    {
        if (made.empty()) return;
        std::lock_guard<std::mutex> lock(m_);
        exec("BEGIN IMMEDIATE");
        try
        {
            statement s(db_, "INSERT INTO text_substitutes(record, substitute, tag, places) VALUES(?1, ?2, ?3, 1)"
                             " ON CONFLICT(record, substitute, tag) DO UPDATE SET places = places + 1,"
                             " inherited = 0");
            for (const auto& m : made)
            {
                s.reset();
                s.bind(1, record).bind(2, m.first).bind(3, m.second);
                s.step();
            }
            exec("COMMIT");
        }
        catch (...)
        {
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
            throw;
        }
    }

    std::vector<std::pair<std::string, std::string>> store::text_substitutes(long long record, bool inherited_only)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT substitute, tag FROM text_substitutes WHERE record = ?1"
                         " AND (?2 = 0 OR inherited = 1)");
        s.bind(1, record).bind(2, static_cast<long long>(inherited_only ? 1 : 0));
        std::vector<std::pair<std::string, std::string>> out;
        while (s.step()) out.push_back(std::make_pair(s.text(0), s.text(1)));
        return out;
    }

    void store::copy_text_substitutes(long long from, long long to, bool inherited)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "INSERT OR IGNORE INTO text_substitutes(record, substitute, tag, places, inherited)"
                         " SELECT ?2, substitute, tag, places, ?3 FROM text_substitutes WHERE record = ?1");
        s.bind(1, from).bind(2, to).bind(3, static_cast<long long>(inherited ? 1 : 0));
        s.step();
    }

    void store::add_entries(long long record, const std::vector<entry_record>& entries)
    {
        if (entries.empty()) return;
        std::lock_guard<std::mutex> lock(m_);
        exec("BEGIN IMMEDIATE");
        try
        {
            statement s(db_, "INSERT OR REPLACE INTO entries(record, name, original, sha256, kind)"
                             " VALUES(?1, ?2, ?3, ?4, ?5)");
            for (const entry_record& e : entries)
            {
                s.reset();
                s.bind(1, record).bind(2, e.name).bind(3, e.original).bind(4, e.sha256).bind(5, e.kind);
                s.step();
            }
            exec("COMMIT");
        }
        catch (...)
        {
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
            throw;
        }
    }

    bool store::find_entry(const std::string& sha256, entry_record& e)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT record, name, original, sha256, kind FROM entries WHERE sha256 = ?1"
                         " ORDER BY record DESC LIMIT 1");
        s.bind(1, sha256);
        if (!s.step()) return false;
        e.record = s.integer(0);
        e.name = s.text(1);
        e.original = s.text(2);
        e.sha256 = s.text(3);
        e.kind = s.text(4);
        return true;
    }

    std::vector<entry_record> store::entries_inside(const entry_record& e)
    {
        std::lock_guard<std::mutex> lock(m_);
        const std::string name = e.name + "!", original = e.original + "!";
        statement s(db_, "SELECT name, original, sha256, kind FROM entries WHERE record = ?1"
                         " AND substr(name, 1, length(?2)) = ?2 AND substr(original, 1, length(?3)) = ?3");
        s.bind(1, e.record).bind(2, name).bind(3, original);
        std::vector<entry_record> out;
        while (s.step())
        {
            entry_record r;
            r.name = s.text(0).substr(name.size());
            r.original = s.text(1).substr(original.size());
            r.sha256 = s.text(2);
            r.kind = s.text(3);
            out.push_back(r);
        }
        return out;
    }

    bool store::get_picture(const std::string& id, picture_record& r)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT id, sha256, media_type, blob, description, created FROM pictures WHERE id = ?1");
        s.bind(1, id);
        if (!s.step()) return false;
        r.id = s.text(0);
        r.sha256 = s.text(1);
        r.media_type = s.text(2);
        r.blob = s.text(3);
        r.description = s.text(4);
        r.created = s.integer(5);
        return true;
    }

    bool store::set_picture_description(const std::string& id, const std::string& description, std::string& before)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT description FROM pictures WHERE id = ?1");
        s.bind(1, id);
        if (!s.step()) return false;
        before = s.text(0);
        statement u(db_, "UPDATE pictures SET description = ?2 WHERE id = ?1");
        u.bind(1, id).bind(2, description);
        u.step();
        return true;
    }

    size_t store::remove_unlinked_pictures()
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "DELETE FROM pictures WHERE id NOT IN (SELECT picture FROM file_pictures)");
        s.step();
        return static_cast<size_t>(sqlite3_changes(db_));
    }

    void store::add_picture(const picture_record& r)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "INSERT OR IGNORE INTO pictures(id, sha256, media_type, blob, description, created)"
                         " VALUES(?1, ?2, ?3, ?4, ?5, ?6)");
        s.bind(1, r.id).bind(2, r.sha256).bind(3, r.media_type).bind(4, r.blob).bind(5, r.description)
            .bind(6, static_cast<long long>(r.created ? r.created : now_unix()));
        s.step();
    }

    void store::link_picture(long long file, const std::string& picture)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "INSERT OR IGNORE INTO file_pictures(file, picture) VALUES(?1, ?2)");
        s.bind(1, file).bind(2, picture);
        s.step();
    }

    void store::drive_usage(long long& files, long long& bytes)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM folder_files"
                         " WHERE folder = 'drive' AND state != 'folder'");
        files = bytes = 0;
        if (s.step())
        {
            files = s.integer(0);
            bytes = s.integer(1);
        }
    }

    void store::add_event(const std::string& kind, const std::string& path, const std::string& message)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "INSERT INTO events(time, kind, path, message) VALUES(?1, ?2, ?3, ?4)");
        s.bind(1, static_cast<long long>(now_unix())).bind(2, kind).bind(3, path).bind(4, message);
        s.step();
        // The last thousand events are kept.
        exec("DELETE FROM events WHERE id <= (SELECT MAX(id) - 1000 FROM events)");
    }

    json store::events(long long since, int limit)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT id, time, kind, path, message FROM events WHERE id > ?1 ORDER BY id LIMIT ?2");
        s.bind(1, since).bind(2, static_cast<long long>(limit));
        json out = json::array();
        while (s.step())
        {
            json e = json::object();
            e["id"] = s.integer(0);
            e["time"] = iso_time(s.integer(1));
            e["kind"] = s.text(2);
            e["path"] = s.text(3);
            e["message"] = s.text(4);
            out.push_back(e);
        }
        return out;
    }

    std::vector<folder_file> store::folder_files(const std::string& folder)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT path, size, written, sha256, record, state FROM folder_files WHERE folder = ?1");
        s.bind(1, folder);
        std::vector<folder_file> out;
        while (s.step())
        {
            folder_file f;
            f.folder = folder;
            f.path = s.text(0);
            f.size = s.integer(1);
            f.written = s.integer(2);
            f.sha256 = s.text(3);
            f.record = s.integer(4);
            f.state = s.text(5);
            out.push_back(f);
        }
        return out;
    }

    void store::set_folder_file(const folder_file& f)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "INSERT INTO folder_files(folder, path, size, written, sha256, record, state)"
                         " VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7) ON CONFLICT(folder, path) DO UPDATE SET"
                         " path = ?2, size = ?3, written = ?4, sha256 = ?5, record = ?6, state = ?7");
        s.bind(1, f.folder).bind(2, f.path).bind(3, f.size).bind(4, static_cast<long long>(f.written)).bind(5, f.sha256)
            .bind(6, f.record).bind(7, f.state);
        s.step();
    }

    void store::remove_folder_file(const std::string& folder, const std::string& path)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "DELETE FROM folder_files WHERE folder = ?1 AND path = ?2");
        s.bind(1, folder).bind(2, path);
        s.step();
    }

    void store::add_substitutes(const std::vector<std::pair<std::string, std::string>>& made)
    {
        if (made.empty()) return;
        std::lock_guard<std::mutex> lock(m_);
        exec("BEGIN IMMEDIATE");
        try
        {
            const long long now = static_cast<long long>(now_unix());
            for (const auto& m : made)
            {
                if (m.first.empty() || m.second.empty()) continue;
                statement s(db_, "INSERT INTO substitutes(substitute, tag, created) VALUES(?1, ?2, ?3)"
                                 " ON CONFLICT(substitute, tag) DO UPDATE SET created = ?3");
                s.bind(1, m.first).bind(2, m.second).bind(3, now);
                s.step();
            }
            exec("COMMIT");
        }
        catch (...)
        {
            exec("ROLLBACK");
            throw;
        }
    }

    std::vector<std::string> store::tags_for(const std::string& substitute)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT tag FROM substitutes WHERE substitute = ?1 ORDER BY created DESC LIMIT 8");
        s.bind(1, substitute);
        std::vector<std::string> tags;
        while (s.step()) tags.push_back(s.text(0));
        return tags;
    }

    std::string store::value(const std::string& key)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "SELECT value FROM kv WHERE key = ?1");
        s.bind(1, key);
        return s.step() ? s.text(0) : std::string();
    }

    void store::set_value(const std::string& key, const std::string& value)
    {
        std::lock_guard<std::mutex> lock(m_);
        statement s(db_, "INSERT INTO kv(key, value) VALUES(?1, ?2) ON CONFLICT(key) DO UPDATE SET value = ?2");
        s.bind(1, key).bind(2, value);
        s.step();
    }
}
