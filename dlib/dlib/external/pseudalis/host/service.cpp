// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
#include <dlib/external/httplib/httplib.h>

#include "service.h"

#include <cctype>
#include <chrono>
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "../common/log.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "certificates.h"
#include "crypto.h"
#include "drive.h"
#include "http_fetch.h"
#include "keys.h"
#include "public_link.h"

namespace pseudalis
{
    namespace
    {
        std::string trim(const std::string& s)
        {
            const size_t b = s.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return std::string();
            return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
        }

        // The identifier of a text file that holds a picture marker alone: "[Image@<16 digits> ...]".
        std::string marker_id(const std::string& bytes)
        {
            const std::string t = trim(bytes.size() > 4096 ? std::string() : bytes);
            if (t.size() < 24 || t.compare(0, 7, "[Image@") != 0 || t.back() != ']') return std::string();
            const std::string id = t.substr(7, 16);
            for (char c : id)
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return std::string();
            return t.size() == 24 || t.compare(23, 3, " : ") == 0 ? id : std::string();
        }

        std::string without_suffix(const std::string& name, const char* suffix)
        {
            const std::string s(suffix);
            if (name.size() > s.size())
            {
                std::string tail = name.substr(name.size() - s.size());
                for (char& c : tail) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                if (tail == s) return name.substr(0, name.size() - s.size());
            }
            return name;
        }
    }

    service::service() {}

    service::~service()
    {
        stop();
        if (background_.joinable()) background_.join();
        wipe(master_);
    }

    void service::stop()
    {
        stopping_ = true;
        wake_.notify_all();
        std::lock_guard<std::mutex> lock(server_m_);
        if (server_) server_->stop();
    }

    void service::request_restart()
    {
        restart_ = true;
        stop();
    }

    std::string service::language() const
    {
        const std::string l = settings_ ? settings_->get().language : std::string("auto");
        if (l != "auto") return l;
#if defined(_WIN32)
        switch (PRIMARYLANGID(GetSystemDefaultUILanguage()))
        {
            case LANG_FRENCH: return "fr";
            case LANG_GERMAN: return "de";
            case LANG_ITALIAN: return "it";
            case LANG_SPANISH: return "es";
            default: return "en";
        }
#else
        const char* lang = std::getenv("LANG");
        const std::string s = lang ? std::string(lang).substr(0, 2) : std::string();
        return s == "fr" || s == "de" || s == "it" || s == "es" ? s : "en";
#endif
    }

    bool service::start_engine(std::string& error)
    {
        library_ = std::make_shared<engine_library>();
        const char* force = std::getenv("PSEUDALIS_ENGINE");
        if (!library_->load(program_dir_, force ? force : "", error)) return false;
        json options = json::object();
        options["date_order"] = settings_->get().date_order;
        const key_set keys = derive_keys(master_);
        std::string key = keys.pseudonymization;
        engine_.reset(new engine_client(library_, key, options, error));
        wipe(key);
        if (!engine_->ok()) return false;
        key_fingerprint_ = engine_->key_fingerprint();
        log_info("engine: " + library_->path() + " (" + engine_->variant() + "), key " + key_fingerprint_);
        return true;
    }

    bool service::run(std::string& error)
    {
        started_ = now_unix();
        program_dir_ = executable_dir();
        data_dir_ = pseudalis::data_dir();
        std::string e;
        if (!secure_folder(data_dir_, e))
        {
            error = "the data folder " + data_dir_ + " cannot be prepared: " + e;
            return false;
        }
        make_dirs(join_path(data_dir_, "logs"));
        log_file::instance().open(join_path(join_path(data_dir_, "logs"), "service.log"));
        log_info("service: " + std::string(PSEUDALIS_PRODUCT_NAME) + " " + PSEUDALIS_VERSION + " starts, data in " +
                 data_dir_);
        settings_.reset(new settings_store(join_path(data_dir_, "settings.json")));
        const host_settings st = settings_->get();
        port_ = st.port;

        // The token of the administration interface, which the tray and the shell read.
        const std::string token_path = join_path(data_dir_, "admin.token");
        std::string token;
        if (!read_file(token_path, token, 4096) || trim(token).size() < 32)
        {
            token = random_hex(48);
            if (!write_file_restricted(token_path, token, true))
            {
                error = "cannot write " + token_path;
                return false;
            }
        }
        admin_token_ = trim(token);

        // The keys, the vault and the database, in folders of SYSTEM and the administrators.
        bool created = false;
        if (!load_master_key(join_path(data_dir_, "keys"), master_, created, error)) return false;
        if (created) log_info("keys: a new master key was made");
        const key_set keys = derive_keys(master_);
        if (!private_folder(join_path(data_dir_, "vault"), error) || !private_folder(join_path(data_dir_, "db"), error))
            return false;
        vault_.reset(new vault(join_path(data_dir_, "vault"), keys.vault_encryption, keys.vault_authentication));
        try
        {
            db_.reset(new store(join_path(join_path(data_dir_, "db"), "pseudalis.db")));
        }
        catch (const std::exception& x)
        {
            error = x.what();
            return false;
        }
        if (!start_engine(error)) return false;

        // The certificates of the local server, and the trust of a new authority.
        const std::string cert_dir = join_path(data_dir_, "certificates");
        bool renewed = false;
#if defined(_WIN32)
        const bool was_trusted = authority_trusted();
#endif
        if (!ensure_certificates(cert_dir, renewed, error)) return false;
#if defined(_WIN32)
        if (renewed && (st.trust_authority || was_trusted))
        {
            std::string te;
            if (trust_authority(cert_dir, te))
                log_info("certificates: the new local authority is trusted by this computer");
            else
                log_warning("certificates: " + te);
        }
#endif

        // The updates: one downloaded earlier is applied before anything is served.
        set_http_proxy(st.proxy);
        set_public_link(st.public_link);
        updater_.reset(new updater(st.feed_url, join_path(data_dir_, "updates")));
        if (!updater_->staged_installer().empty())
        {
            user_error ue;
            if (updater_->can_apply(ue) && updater_->launch_staged(ue))
            {
                log_info("updates: the installer of the update starts");
                update_started_ = true;
                return true;
            }
            log_warning("updates: " + ue.text);
        }
        albireo_.set_enabled(st.albireo != "off");

        // The drive.
        drive_.reset(new drive_host(*this));
        if (!st.drive_letter.empty())
        {
            std::string de;
            if (drive_->start(st.drive_letter, de)) log_info("drive: " + st.drive_letter + ": mounted");
            else log_warning("drive: " + de);
        }

        // The server.
        std::string cert_pem, key_pem;
        if (!read_file(certificate_paths(cert_dir).server_cert, cert_pem) || !read_server_key(cert_dir, key_pem))
        {
            wipe_secret(key_pem);
            error = "the certificate or the key of the local server cannot be read in " + cert_dir;
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(server_m_);
            if (stopping_)
            {
                wipe_secret(key_pem);
                return true;
            }
            httplib::SSLServer::PemMemory pem = {};
            pem.cert_pem = cert_pem.c_str();
            pem.cert_pem_len = cert_pem.size();
            pem.key_pem = key_pem.c_str();
            pem.key_pem_len = key_pem.size();
            server_.reset(new httplib::SSLServer(pem));
        }
        wipe_secret(key_pem);
        if (!server_->is_valid())
        {
            error = "the TLS configuration of the server failed";
            return false;
        }
        server_->set_payload_max_length(static_cast<size_t>(st.max_file_mb + 1) << 20);
        server_->set_keep_alive_timeout(5);
        server_->set_read_timeout(60);
        server_->set_write_timeout(60);
        server_->new_task_queue = []() { return new httplib::ThreadPool(8, 32, 64); };
        server_->set_socket_options([](socket_t sock) {
#if defined(_WIN32)
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#else
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
#endif
        });
        if (!server_->bind_to_port("127.0.0.1", port_))
        {
            error = "the port " + std::to_string(port_) + " is in use or cannot be opened";
            if (drive_) drive_->stop();
            return false;
        }
        install_routes(*server_, *this);
        {
            std::lock_guard<std::mutex> lock(server_m_);
            if (stopping_) server_->stop();
        }
        log_info("server: https://localhost:" + std::to_string(port_) + "/");
        background_ = std::thread([this]() {
            try
            {
                background_loop();
            }
            catch (const std::exception& x)
            {
                log_error("service: the background tasks stopped: " + std::string(x.what()));
            }
        });
        server_->listen_after_bind();

        stop();
        updater_->cancel();
        http_cancel_all("the service stops");
        if (background_.joinable()) background_.join();
        if (drive_) drive_->stop();
        log_info("service: stopped");
        return true;
    }

    void service::background_loop()
    {
        // The first check of the updates comes five minutes after the start, then every day;
        // the originals of the files removed from the drive long enough go every hour.
        int64_t next_check = now_unix() + 300;
        int64_t next_purge = 0;
        while (!stopping_)
        {
            {
                std::unique_lock<std::mutex> lock(wake_m_);
                wake_.wait_for(lock, std::chrono::seconds(60), [this]() { return stopping_.load(); });
            }
            if (stopping_) break;
            if (now_unix() >= next_purge)
            {
                purge_removed();
                next_purge = now_unix() + 3600;
            }
            if (settings_->get().check_updates && now_unix() >= next_check)
            {
                updater_->check_async();
                next_check = now_unix() + 24 * 3600;
            }
        }
    }

    json service::status()
    {
        json j = json::object();
        j["product"] = PSEUDALIS_PRODUCT_NAME;
        j["version"] = PSEUDALIS_VERSION;
        j["copyright"] = PSEUDALIS_COPYRIGHT;
        j["author"] = PSEUDALIS_AUTHOR_NAME;
        j["started"] = iso_time(started_);
        j["port"] = port_;
        j["language"] = language();
        json engine = json::object();
        engine["variant"] = engine_ ? engine_->variant() : std::string();
        engine["library"] = library_ ? file_name(library_->path()) : std::string();
        engine["models"] = engine_ ? engine_->models() : json();
        engine["avx2"] = cpu_has_avx2();
        j["engine"] = engine;
        j["key_fingerprint"] = key_fingerprint_;
        j["records"] = db_ ? db_->count_files() : 0;
        j["drive"] = drive_ ? drive_->status() : json::object();
        j["albireo"] = albireo_.status();
        j["updates"] = updater_ ? updater_->status() : json::object();
        return j;
    }

    bool service::protect(const std::string& bytes, const std::string& name, const std::string& source, bool record,
                          protect_result& out, user_error& error)
    {
        if (!engine_)
        {
            error.set("the engine is not ready", "engine");
            return false;
        }
        const host_settings st = settings_->get();
        if (bytes.size() > (static_cast<uint64_t>(st.max_file_mb) << 20))
        {
            error.set("the file is larger than " + std::to_string(st.max_file_mb) + " MB", "file_too_large")
                .arg("mb", std::to_string(st.max_file_mb));
            return false;
        }
        const std::string lang = language();
        // Each picture: kept, described once.
        auto pictures = [&](const picture_request& p) -> std::string {
            picture_record rec;
            if (db_->find_picture(p.sha256, rec)) return rec.description;
            std::string description, why;
            if (st.albireo != "off")
            {
                description = albireo_.describe(p.bytes, p.media_type, lang, why);
                if (description.empty()) log_info("pictures: " + p.id + " without description: " + why);
            }
            if (record)
            {
                rec.id = p.id;
                rec.sha256 = p.sha256;
                rec.media_type = p.media_type;
                rec.blob = vault_->put(p.bytes);
                rec.description = description;
                if (!rec.blob.empty()) db_->add_picture(rec);
            }
            return description;
        };
        json request = json::object();
        request["operation"] = "anonymize";
        request["name"] = name;
        request["pdf_layout"] = st.pdf_layout;
        json result;
        std::string e;
        if (!engine_->process(request, bytes, result, out.bytes, pictures, e))
        {
            error.set(e, "protect_failed").arg("error", e);
            return false;
        }
        out.name = result["name"].str(name);
        out.kind = result["kind"].str();
        out.report = result["report"];
        out.sha256 = sha256_hex(out.bytes);
        out.record = 0;
        if (record)
        {
            const std::string blob = vault_->put(bytes);
            if (blob.empty())
            {
                error.set("the original cannot be kept in the vault", "vault");
                return false;
            }
            file_record r;
            r.source = source;
            r.original_name = name;
            r.output_name = out.name;
            r.original_sha256 = sha256_hex(bytes);
            r.output_sha256 = out.sha256;
            r.original_blob = blob;
            r.original_size = static_cast<long long>(bytes.size());
            r.output_size = static_cast<long long>(out.bytes.size());
            r.kind = out.kind;
            try
            {
                out.record = db_->add_file(r);
                for (const json& id : result["pictures"].elements()) db_->link_picture(out.record, id.str());
            }
            catch (const std::exception& x)
            {
                vault_->remove(blob);
                error.set(x.what(), "database");
                return false;
            }
        }
        return true;
    }

    bool service::restore(const std::string& bytes, const std::string& name, const std::string& object_id,
                          restore_result& out, user_error& error)
    {
        file_record r;
        if (db_->find_by_output(sha256_hex(bytes), r))
        {
            if (!vault_->get(r.original_blob, out.bytes))
            {
                error.set("the original of this file can no longer be read in the vault", "original_missing");
                return false;
            }
            out.name = r.original_name;
            out.how = "original";
            out.record = r.id;
            out.report = json::object();
            /* The original of a protected file edited on the drive is that file with its edits:
               the tags it kept are restored too. */
            if (engine_ && out.bytes.find("[[") != std::string::npos)
            {
                json request = json::object(), result;
                request["operation"] = "restore";
                request["name"] = r.original_name;
                std::string restored, e;
                if (engine_->process(request, out.bytes, result, restored, nullptr, e) &&
                    result["report"]["tags_restored"].as_int(0) > 0)
                {
                    out.bytes = restored;
                    out.report = result["report"];
                }
            }
            return true;
        }
        const std::string id = marker_id(bytes);
        picture_record pic;
        if (!id.empty() && db_->get_picture(id, pic))
        {
            if (!vault_->get(pic.blob, out.bytes))
            {
                error.set("the picture can no longer be read in the vault", "original_missing");
                return false;
            }
            out.name = without_suffix(name, ".txt");
            out.how = "picture";
            out.report = json::object();
            return true;
        }
        if (!engine_)
        {
            error.set("the engine is not ready", "engine");
            return false;
        }
        json request = json::object();
        request["operation"] = "restore";
        request["name"] = name;
        json result;
        std::string e;
        if (!engine_->process(request, bytes, result, out.bytes, nullptr, e))
        {
            error.set(e, "restore_failed").arg("error", e);
            return false;
        }
        out.report = result["report"];
        if (out.report["tags_restored"].as_int(0) == 0)
        {
            error.set("this file holds no value that Pseudalis protected with this key", "not_protected");
            return false;
        }
        out.name = name;
        out.how = "tags";
        if (db_->find_by_object(object_id, r)) out.record = r.id;
        return true;
    }

    bool service::text(const std::string& operation, const std::string& text, json& out, user_error& error)
    {
        if (!engine_)
        {
            error.set("the engine is not ready", "engine");
            return false;
        }
        json request = json::object();
        request["operation"] = operation;
        request["name"] = "text";
        json result;
        std::string output, e;
        if (!engine_->process(request, text, result, output, nullptr, e))
        {
            error.set(e, "text_failed");
            return false;
        }
        out = json::object();
        if (operation == "detect")
        {
            out["spans"] = result["spans"];
        }
        else
        {
            out["text"] = output;
            out["report"] = result["report"];
        }
        return true;
    }

    bool service::forget(long long record)
    {
        std::vector<std::string> blobs;
        try
        {
            if (!db_->remove_file(record, blobs)) return false;
        }
        catch (const std::exception& x)
        {
            log_error("records: " + std::string(x.what()));
            return false;
        }
        for (const std::string& b : blobs) vault_->remove(b);
        return true;
    }

    void service::mark_removed(long long record)
    {
        try
        {
            db_->set_file_removed(record, now_unix());
        }
        catch (const std::exception& x)
        {
            log_error("records: " + std::string(x.what()));
        }
    }

    void service::purge_removed()
    {
        const int days = settings_->get().removed_days;
        std::vector<long long> records;
        try
        {
            records = db_->files_removed_before(now_unix() - static_cast<int64_t>(days) * 86400);
        }
        catch (const std::exception& x)
        {
            log_error("records: " + std::string(x.what()));
            return;
        }
        size_t forgotten = 0;
        for (long long r : records)
            if (forget(r)) ++forgotten;
        if (forgotten)
            log_info("records: " + std::to_string(forgotten) + " original(s) of files removed from the drive " +
                     std::to_string(days) + " days ago or more forgotten");
    }

    bool service::export_key(const std::string& passphrase, std::string& exported, user_error& error)
    {
        if (passphrase.size() < 12)
        {
            error.set("the passphrase must have 12 characters at least", "passphrase_short");
            return false;
        }
        std::lock_guard<std::mutex> lock(key_m_);
        exported = export_master_key(master_, passphrase);
        log_info("keys: the master key was exported");
        return true;
    }

    bool service::import_key(const std::string& exported, const std::string& passphrase, user_error& error)
    {
        long long files = 0, bytes = 0;
        db_->drive_usage(files, bytes);
        if (db_->count_files() > 0 || files > 0)
        {
            error.set("the key of an earlier installation can only be imported while no file is recorded",
                      "import_not_empty");
            return false;
        }
        std::string key, e;
        if (!import_master_key(exported, passphrase, key, e))
        {
            error.set(e, "import_failed");
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(key_m_);
            if (!save_master_key(join_path(data_dir_, "keys"), key, e))
            {
                wipe(key);
                error.set(e, "import_failed");
                return false;
            }
        }
        wipe(key);
        log_info("keys: a master key was imported, the service restarts");
        request_restart();
        return true;
    }
}
