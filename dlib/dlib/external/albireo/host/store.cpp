// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "store.h"

#include <algorithm>
#include <ctime>
#include <stdexcept>

#include <dlib/sqlite.h>

#include "../common/log.h"
#include "../common/platform.h"

namespace albireo
{

    namespace
    {
        // The columns of a model, in the order model_of() reads them.
        const char* const model_columns = "id, name, file, size, sha256, source, installed_at, verified_at, "
            "descriptor, projector, projector_size, projector_sha256";

        installed_model model_of(dlib::statement& st)
        {
            installed_model m;
            m.id = st.get_column_as_text(0);
            m.name = st.get_column_as_text(1);
            m.file = st.get_column_as_text(2);
            m.size = static_cast<uint64_t>(st.get_column_as_int64(3));
            m.sha256 = st.get_column_as_text(4);
            m.source = st.get_column_as_text(5);
            m.installed_at = st.get_column_as_int64(6);
            m.verified_at = st.get_column_as_int64(7);
            m.descriptor = json::parse_or_null(st.get_column_as_text(8));
            m.projector = st.get_column_as_text(9);
            m.projector_size = static_cast<uint64_t>(st.get_column_as_int64(10));
            m.projector_sha256 = st.get_column_as_text(11);
            return m;
        }
    }

    store::store(const std::string& path) : path_(path)
    {
        make_dirs(parent_path(path));
        connect();
        /*
            Every page of the database is checked once at the start: a database damaged beyond
            its header fails here, with "malformed" in its error, where the service sets it
            aside and starts a new one, rather than in the queries that follow.
        */
        {
            dlib::statement st(*db_, "PRAGMA quick_check");
            st.exec();
            const std::string result = st.move_next() ? st.get_column_as_text(0) : std::string("no answer");
            if (result != "ok") throw std::runtime_error("database disk image is malformed (" + result + ")");
        }
        db_->exec(
            "CREATE TABLE IF NOT EXISTS models ("
            " id TEXT PRIMARY KEY, name TEXT NOT NULL, file TEXT NOT NULL, size INTEGER NOT NULL,"
            " sha256 TEXT NOT NULL, source TEXT NOT NULL, installed_at INTEGER NOT NULL,"
            " verified_at INTEGER NOT NULL DEFAULT 0, descriptor TEXT NOT NULL DEFAULT '{}',"
            " projector TEXT NOT NULL DEFAULT '', projector_size INTEGER NOT NULL DEFAULT 0,"
            " projector_sha256 TEXT NOT NULL DEFAULT '')");
        // The columns of the projectors, which the databases of the earlier versions lack.
        std::vector<std::string> columns;
        {
            dlib::statement st(*db_, "PRAGMA table_info(models)");
            st.exec();
            while (st.move_next()) columns.push_back(st.get_column_as_text(1));
        }
        const char* const added[][2] = { { "projector", "TEXT NOT NULL DEFAULT ''" },
            { "projector_size", "INTEGER NOT NULL DEFAULT 0" }, { "projector_sha256", "TEXT NOT NULL DEFAULT ''" } };
        for (const auto& c : added)
            if (std::find(columns.begin(), columns.end(), c[0]) == columns.end())
            {
                db_->exec(std::string("ALTER TABLE models ADD COLUMN ") + c[0] + " " + c[1]);
                log_info(std::string("store: the column ") + c[0] + " is added to the models of " + path_);
            }
        db_->exec(
            "CREATE TABLE IF NOT EXISTS projectors ("
            " sha256 TEXT PRIMARY KEY, file TEXT NOT NULL, size INTEGER NOT NULL, name TEXT NOT NULL,"
            " descriptor TEXT NOT NULL DEFAULT '{}', received_at INTEGER NOT NULL)");
        db_->exec(
            "CREATE TABLE IF NOT EXISTS downloads ("
            " id TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at INTEGER NOT NULL)");
        db_->exec(
            "CREATE TABLE IF NOT EXISTS usage ("
            " day TEXT PRIMARY KEY, prompt_tokens INTEGER NOT NULL DEFAULT 0,"
            " generated_tokens INTEGER NOT NULL DEFAULT 0, requests INTEGER NOT NULL DEFAULT 0)");
        db_->exec("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)");
    }

    store::~store() {}

    void store::connect()
    {
        db_.reset(new dlib::database(path_));
        /*
            A database that another program holds a moment, a backup or a copy of the file say,
            is waited for up to five seconds before an operation fails, and the next
            operations find it free again.
        */
        db_->exec("PRAGMA busy_timeout=5000");
        db_->exec("PRAGMA journal_mode=WAL");
        db_->exec("PRAGMA synchronous=NORMAL");
        db_->exec("PRAGMA foreign_keys=ON");
    }

    void store::writing(const std::function<void()>& work)
    {
        try
        {
            work();
        }
        catch (const std::exception& e)
        {
            // A connection that SQLite opened read-only, once the file can be written again.
            if (std::string(e.what()).find("readonly") == std::string::npos) throw;
            connect();
            work();
            log_info("store: " + path_ + " is written again");
        }
    }

    std::vector<installed_model> store::models()
    {
        std::lock_guard<std::mutex> lock(m_);
        std::vector<installed_model> out;
        dlib::statement st(*db_, std::string("SELECT ") + model_columns + " FROM models ORDER BY name");
        st.exec();
        while (st.move_next()) out.push_back(model_of(st));
        return out;
    }

    bool store::find_model(const std::string& id, installed_model& m)
    {
        // The row of the model alone, by its key, rather than every model with its description.
        std::lock_guard<std::mutex> lock(m_);
        dlib::statement st(*db_, std::string("SELECT ") + model_columns + " FROM models WHERE id = ?");
        st.bind(1, id);
        st.exec();
        if (!st.move_next()) return false;
        m = model_of(st);
        return true;
    }

    void store::put_model(const installed_model& m)
    {
        std::lock_guard<std::mutex> lock(m_);
        writing([&]() {
            dlib::statement st(*db_, "INSERT OR REPLACE INTO models (id, name, file, size, sha256, source, "
                "installed_at, verified_at, descriptor, projector, projector_size, projector_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
            st.bind(1, m.id);
            st.bind(2, m.name);
            st.bind(3, m.file);
            st.bind_int64(4, static_cast<dlib::int64>(m.size));
            st.bind(5, m.sha256);
            st.bind(6, m.source);
            st.bind_int64(7, m.installed_at);
            st.bind_int64(8, m.verified_at);
            st.bind(9, m.descriptor.dump());
            st.bind(10, m.projector);
            st.bind_int64(11, static_cast<dlib::int64>(m.projector_size));
            st.bind(12, m.projector_sha256);
            st.exec();
        });
    }

    std::vector<pending_projector> store::projectors()
    {
        std::lock_guard<std::mutex> lock(m_);
        std::vector<pending_projector> out;
        dlib::statement st(*db_, "SELECT sha256, file, size, name, descriptor, received_at FROM projectors "
            "ORDER BY received_at");
        st.exec();
        while (st.move_next())
        {
            pending_projector p;
            p.sha256 = st.get_column_as_text(0);
            p.file = st.get_column_as_text(1);
            p.size = static_cast<uint64_t>(st.get_column_as_int64(2));
            p.name = st.get_column_as_text(3);
            p.descriptor = json::parse_or_null(st.get_column_as_text(4));
            p.received_at = st.get_column_as_int64(5);
            out.push_back(p);
        }
        return out;
    }

    void store::put_projector(const pending_projector& p)
    {
        std::lock_guard<std::mutex> lock(m_);
        writing([&]() {
            dlib::statement st(*db_, "INSERT OR REPLACE INTO projectors (sha256, file, size, name, descriptor, "
                "received_at) VALUES (?, ?, ?, ?, ?, ?)");
            st.bind(1, p.sha256);
            st.bind(2, p.file);
            st.bind_int64(3, static_cast<dlib::int64>(p.size));
            st.bind(4, p.name);
            st.bind(5, p.descriptor.dump());
            st.bind_int64(6, p.received_at);
            st.exec();
        });
    }

    void store::remove_projector(const std::string& sha256)
    {
        std::lock_guard<std::mutex> lock(m_);
        writing([&]() {
            dlib::statement st(*db_, "DELETE FROM projectors WHERE sha256 = ?");
            st.bind(1, sha256);
            st.exec();
        });
    }

    void store::remove_model(const std::string& id)
    {
        std::lock_guard<std::mutex> lock(m_);
        writing([&]() {
            dlib::statement st(*db_, "DELETE FROM models WHERE id = ?");
            st.bind(1, id);
            st.exec();
        });
    }

    void store::set_verified(const std::string& id, int64_t when)
    {
        std::lock_guard<std::mutex> lock(m_);
        writing([&]() {
            dlib::statement st(*db_, "UPDATE models SET verified_at = ? WHERE id = ?");
            st.bind_int64(1, when);
            st.bind(2, id);
            st.exec();
        });
    }

    json store::download_state(const std::string& id)
    {
        std::lock_guard<std::mutex> lock(m_);
        dlib::statement st(*db_, "SELECT state FROM downloads WHERE id = ?");
        st.bind(1, id);
        st.exec();
        if (st.move_next()) return json::parse_or_null(st.get_column_as_text(0));
        return json();
    }

    void store::set_download_state(const std::string& id, const json& state)
    {
        std::lock_guard<std::mutex> lock(m_);
        writing([&]() {
            dlib::statement st(*db_, "INSERT OR REPLACE INTO downloads (id, state, updated_at) VALUES (?, ?, ?)");
            st.bind(1, id);
            st.bind(2, state.dump());
            st.bind_int64(3, now_unix());
            st.exec();
        });
    }

    void store::remove_download_state(const std::string& id)
    {
        std::lock_guard<std::mutex> lock(m_);
        writing([&]() {
            dlib::statement st(*db_, "DELETE FROM downloads WHERE id = ?");
            st.bind(1, id);
            st.exec();
        });
    }

    std::vector<std::string> store::pending_downloads()
    {
        std::lock_guard<std::mutex> lock(m_);
        std::vector<std::string> out;
        dlib::statement st(*db_, "SELECT id FROM downloads ORDER BY updated_at");
        st.exec();
        while (st.move_next()) out.push_back(st.get_column_as_text(0));
        return out;
    }

    void store::add_usage(int64_t prompt_tokens, int64_t generated_tokens, int64_t requests)
    {
        const std::string day = iso_time(now_unix()).substr(0, 10);
        std::lock_guard<std::mutex> lock(m_);
        writing([&]() {
            dlib::statement st(*db_,
                "INSERT INTO usage (day, prompt_tokens, generated_tokens, requests) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(day) DO UPDATE SET prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
                "generated_tokens = generated_tokens + excluded.generated_tokens, "
                "requests = requests + excluded.requests");
            st.bind(1, day);
            st.bind_int64(2, prompt_tokens);
            st.bind_int64(3, generated_tokens);
            st.bind_int64(4, requests);
            st.exec();
        });
    }

    json store::usage(int days)
    {
        std::lock_guard<std::mutex> lock(m_);
        json out = json::array();
        dlib::statement st(*db_,
            "SELECT day, prompt_tokens, generated_tokens, requests FROM usage ORDER BY day DESC LIMIT ?");
        st.bind(1, days);
        st.exec();
        while (st.move_next())
        {
            json d = json::object();
            d["day"] = st.get_column_as_text(0);
            d["prompt_tokens"] = static_cast<long long>(st.get_column_as_int64(1));
            d["generated_tokens"] = static_cast<long long>(st.get_column_as_int64(2));
            d["requests"] = static_cast<long long>(st.get_column_as_int64(3));
            out.push_back(d);
        }
        return out;
    }

    std::string store::value(const std::string& key, const std::string& def)
    {
        std::lock_guard<std::mutex> lock(m_);
        dlib::statement st(*db_, "SELECT value FROM settings WHERE key = ?");
        st.bind(1, key);
        st.exec();
        if (st.move_next()) return st.get_column_as_text(0);
        return def;
    }

    void store::set_value(const std::string& key, const std::string& v)
    {
        std::lock_guard<std::mutex> lock(m_);
        writing([&]() {
            dlib::statement st(*db_, "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)");
            st.bind(1, key);
            st.bind(2, v);
            st.exec();
        });
    }

}
