// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
#include <dlib/external/httplib/httplib.h>

#include "service.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <set>

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

        /*
            The code of a failure of the engine: "unsupported" for a file of a kind it does not
            process, "kind_mismatch" for a file whose content is not the kind its name gives,
            "encoded_content" for a text that holds encoded parts (attachments, base64, RTF...),
            which the clients tell apart from a file that cannot be read.
        */
        std::string engine_code(const std::string& e, const char* code)
        {
            if (e.find(": this kind of file is not supported") != std::string::npos) return "unsupported";
            if (e.find(": the content of this file does not match its name") != std::string::npos)
                return "kind_mismatch";
            if (e.find(": this file holds encoded parts that the detector cannot read") != std::string::npos)
                return "encoded_content";
            return code;
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

    void work_slots::init(int slots)
    {
        std::lock_guard<std::mutex> lock(m_);
        size_ = free_ = std::max(1, slots);
    }

    bool work_slots::acquire(std::chrono::milliseconds wait)
    {
        std::unique_lock<std::mutex> lock(m_);
        if (!cv_.wait_for(lock, wait, [this]() { return free_ > 0; })) return false;
        --free_;
        return true;
    }

    void work_slots::release()
    {
        {
            std::lock_guard<std::mutex> lock(m_);
            ++free_;
        }
        cv_.notify_one();
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
        // The imposed terms that the database keeps: a list it no longer takes is left aside.
        {
            std::vector<imposed_term> kept;
            std::string message, code;
            long long index = -1;
            const json list = json::parse_or_null(db_->value("imposed_terms"));
            if (!list.is_null() && !parse_imposed_terms(list, kept, message, code, index))
            {
                log_warning("terms: the imposed terms of the database are left aside: " + message);
                kept.clear();
            }
            std::string te;
            if (!engine_->set_terms(imposed_terms_json(kept), te))
            {
                error = "the engine refuses the imposed terms: " + te;
                return false;
            }
            std::lock_guard<std::mutex> lock(terms_m_);
            terms_ = kept;
            if (!kept.empty()) log_info("terms: " + std::to_string(kept.size()) + " imposed term(s)");
        }
        // The imposed columns that the database keeps, read the same way.
        {
            std::vector<imposed_column> kept;
            std::string message, code;
            long long index = -1;
            const json list = json::parse_or_null(db_->value("imposed_columns"));
            if (!list.is_null() && !parse_imposed_columns(list, kept, message, code, index))
            {
                log_warning("columns: the imposed columns of the database are left aside: " + message);
                kept.clear();
            }
            std::string ce;
            if (!engine_->set_columns(imposed_columns_json(kept), ce))
            {
                error = "the engine refuses the imposed columns: " + ce;
                return false;
            }
            std::lock_guard<std::mutex> lock(terms_m_);
            columns_ = kept;
            if (!kept.empty()) log_info("columns: " + std::to_string(kept.size()) + " imposed column(s)");
        }
        log_info("engine: " + library_->path() + " (" + engine_->variant() + ", " + engine_->build() + "), key " +
                 key_fingerprint_);
        return true;
    }

    bool service::run(std::string& error)
    {
        started_ = now_unix();
        // A previous run of this process cancelled the downloads as it stopped.
        http_restore();
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
        log_info("service: " + std::string(PSEUDALIS_FULL_NAME) + " " + PSEUDALIS_VERSION + " starts, data in " +
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
            /*
                The blobs that nothing names any more go before anything is written: a deletion
                that failed or was cut short, a picture kept twice by two files at once.
            */
            // The pictures of a protection that failed, or of an entry left out, no file names.
            const size_t unlinked = db_->remove_unlinked_pictures();
            if (unlinked) log_info("pictures: " + std::to_string(unlinked) + " that no record names removed");
            const std::set<std::string> named = db_->referenced_blobs();
            size_t orphans = 0;
            for (const std::string& id : vault_->list_and_clean())
                if (!named.count(id) && vault_->remove(id)) ++orphans;
            if (orphans) log_info("vault: " + std::to_string(orphans) + " blob(s) that no record names removed");
        }
        catch (const std::exception& x)
        {
            error = x.what();
            return false;
        }
        if (!start_engine(error)) return false;
        /*
            The files of the interface processed at once: half the processors, from 2 to 8, and
            no more than the memory holds, each file standing about six times in memory.
        */
        {
            const unsigned cores = std::max(2u, std::thread::hardware_concurrency());
            unsigned long long by_memory = 8;
#if defined(_WIN32)
            const unsigned long long file_bytes = static_cast<unsigned long long>(st.max_file_mb) << 20;
            MEMORYSTATUSEX memory;
            memory.dwLength = sizeof(memory);
            if (GlobalMemoryStatusEx(&memory)) by_memory = memory.ullTotalPhys / (6 * std::max(file_bytes, 1ull));
#endif
            const int slots = static_cast<int>(std::min<unsigned long long>(by_memory, std::min(8u, cores / 2)));
            work_.init(std::max(2, slots));
        }

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
        // The bound of the files of the settings at most (the routes check the current one); a
        // connection left idle a second goes, so that the polls of the icons hold no worker.
        server_->set_payload_max_length(static_cast<size_t>(2048 + 1) << 20);
        server_->set_keep_alive_timeout(1);
        server_->set_keep_alive_max_count(20);
        server_->set_read_timeout(30);
        server_->set_write_timeout(60);
        server_->new_task_queue = []() { return new httplib::ThreadPool(8, 32, 64); };
        /*
            On Windows the socket keeps its default options: a port bound to 127.0.0.1 that way
            cannot be taken by a program of another account, and a restart binds it again at
            once, where SO_EXCLUSIVEADDRUSE would refuse it while the connections of the previous
            run end.  Elsewhere SO_REUSEADDR gives the same restart.
        */
        server_->set_socket_options([](socket_t sock) {
#if defined(_WIN32)
            (void)sock;
#else
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
#endif
        });
        // The previous run, or the previous process, may still release the port: tried again
        // for ten seconds.
        bool bound = false;
        for (int attempt = 0; attempt < 40 && !bound && !stopping_; ++attempt)
        {
            if (attempt > 0) std::this_thread::sleep_for(std::chrono::milliseconds(250));
            bound = server_->bind_to_port("127.0.0.1", port_);
        }
        if (!bound)
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
                wake_.wait_for(lock, std::chrono::seconds(60),
                               [this]() { return stopping_.load() || cleanup_.load(); });
            }
            if (stopping_) break;
            if (cleanup_.exchange(false) || now_unix() >= next_purge)
            {
                purge_removed();
                make_room(0);
                next_purge = now_unix() + 3600;
            }
            if (settings_->get().check_updates && now_unix() >= next_check)
            {
                updater_->check_async();
                next_check = now_unix() + 24 * 3600;
            }
        }
    }

    void service::counts_changed()
    {
        std::lock_guard<std::mutex> lock(counts_m_);
        counts_at_ = std::chrono::steady_clock::time_point();
    }

    json service::status()
    {
        json j = json::object();
        j["product"] = PSEUDALIS_PRODUCT_NAME;
        j["edition"] = PSEUDALIS_EDITION;
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
        // Whether the files take their tags unless a request says otherwise.
        j["tags"] = settings_ ? settings_->get().tags : true;
        {
            std::lock_guard<std::mutex> lock(counts_m_);
            const auto now = std::chrono::steady_clock::now();
            if (counts_at_.time_since_epoch().count() == 0 || now - counts_at_ > std::chrono::seconds(2))
            {
                records_count_ = db_ ? db_->count_files() : 0;
                drive_status_ = drive_ ? drive_->status() : json::object();
                unsigned long long used = 0, limit = 0;
                if (db_) history_space(used, limit);
                history_bytes_ = used;
                history_limit_ = limit;
                counts_at_ = now;
            }
            j["records"] = records_count_;
            j["drive"] = drive_status_;
            json history = json::object();
            history["bytes"] = history_bytes_;
            history["limit"] = history_limit_;
            j["history"] = history;
        }
        j["albireo"] = albireo_.status();
        j["updates"] = updater_ ? updater_->status() : json::object();
        return j;
    }

    bool service::protect(const std::string& bytes, const std::string& name, const std::string& source, bool record,
                          bool tags, protect_result& out, user_error& error, long long previous)
    {
        std::shared_lock<std::shared_timed_mutex> work(work_m_);
        if (key_replaced_)
        {
            error.set("the service restarts with the key imported", "stopping");
            return false;
        }
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
        /*
            Each picture: kept, described once.  The descriptions of a file take three minutes
            at most, and none starts once the service stops: the pictures that follow keep
            their marker alone, as every picture does while the setting albireo is off.  A
            description may name what the picture shows: it is kept encrypted in the vault, as
            the picture, and the engine protects it each time it writes it.  A picture kept
            without description, or whose description can no longer be read, receives the one
            Albireo gives now.
        */
        const auto describe_until = std::chrono::steady_clock::now() + std::chrono::minutes(3);
        auto pictures = [&](const picture_request& p) -> std::string {
            picture_record rec;
            const bool known = db_->get_picture(p.id, rec) && rec.sha256 == p.sha256;
            std::string description, why;
            if (known && st.albireo == "off") return description;
            if (known && !rec.description.empty() && vault_->get(rec.description, description)) return description;
            if (st.albireo != "off" && !stopping_ && std::chrono::steady_clock::now() < describe_until)
            {
                description = albireo_.describe(p.bytes, p.media_type, lang, why);
                if (description.empty()) log_info("pictures: " + p.id + " without description: " + why);
            }
            if (!record) return description;
            const std::string text = description.empty() ? std::string() : vault_->put(description);
            if (known)
            {
                // The description the record lacked; the blob it named before leaves the vault.
                std::string before;
                if (!text.empty() && db_->set_picture_description(rec.id, text, before) && !before.empty())
                    vault_->remove(before);
                return description;
            }
            rec.id = p.id;
            rec.sha256 = p.sha256;
            rec.media_type = p.media_type;
            rec.blob = vault_->put(p.bytes);
            rec.description = text;
            if (!rec.blob.empty() && (description.empty() || !text.empty())) db_->add_picture(rec);
            return description;
        };
        json request = json::object();
        request["operation"] = "anonymize";
        request["name"] = name;
        request["tags"] = tags;
        /* A file protected here without tags, or with imposed columns, protected again, by a newer
           engine for instance, or edited on the drive: the substitutes that its record keeps stay as
           they are, and the new record inherits them, which give back the values of its original. */
        std::vector<long long> sources;
        {
            file_record before;
            if (db_->find_by_output(sha256_hex(bytes), before)) sources.push_back(before.id);
            if (previous > 0 && db_->get_file(previous, before) &&
                std::find(sources.begin(), sources.end(), previous) == sources.end())
                sources.push_back(previous);
            json known = json::array();
            for (long long id : sources)
            {
                const json list = text_substitutes_of(id);
                for (const json& one : list.elements()) known.push_back(one);
            }
            if (!known.empty()) request["substitutions"] = known;
        }
        json result;
        std::string e;
        if (!engine_->process(request, bytes, result, out.bytes, pictures, e, substitute_lookup()))
        {
            error.set(e, engine_code(e, "protect_failed")).arg("error", e);
            return false;
        }
        out.name = result["name"].str(name);
        out.kind = result["kind"].str();
        out.report = result["report"];
        out.sha256 = sha256_hex(out.bytes);
        out.record = 0;
        /* A protected file that holds nothing more to protect comes back as it is: the record that keeps
           its original stays the only one of its content, so that its restoration still reaches it. */
        if (record && out.bytes == bytes)
        {
            file_record known;
            if (db_->find_by_output(out.sha256, known))
            {
                out.record = known.id;
                record = false;
            }
        }
        // The substitutes of the names and the properties come back through their tags, kept here.
        if (!record_substitutes(result, error)) return false;
        // Without tags, the substitutes of the text, which the record of the file alone keeps.
        std::vector<std::pair<std::string, std::string>> text_made;
        for (const json& one : result["substitutions"].elements())
            if (one["text"].as_bool(false))
                text_made.push_back(std::make_pair(one["substitute"].str(), one["tag"].str()));
        // An original larger than the space of the originals is not kept: the file is protected without record.
        if (record && !make_room(bytes.size())) record = false;
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
            // A protected file protected again, edited on the drive for instance: its name in clear.
            r.original_name = restore_name(name);
            r.output_name = out.name;
            r.original_sha256 = sha256_hex(bytes);
            r.output_sha256 = out.sha256;
            r.original_blob = blob;
            r.original_size = static_cast<long long>(bytes.size());
            r.output_size = static_cast<long long>(out.bytes.size());
            r.kind = out.kind;
            r.tags = tags;
            std::vector<entry_record> entries;
            for (const json& w : result["entries"].elements())
            {
                entry_record en;
                en.name = w["name"].str();
                en.original = w["original"].str();
                en.sha256 = w["sha256"].str();
                en.kind = w["kind"].str();
                if (!en.name.empty() && en.sha256.size() == 64) entries.push_back(en);
            }
            try
            {
                out.record = db_->add_file(r);
                for (const json& id : result["pictures"].elements()) db_->link_picture(out.record, id.str());
                db_->add_entries(out.record, entries);
                db_->add_text_substitutes(out.record, text_made);
                for (long long id : sources) db_->copy_text_substitutes(id, out.record, true);
                counts_changed();
            }
            catch (const std::exception& x)
            {
                discard(out.record);
                out.record = 0;
                vault_->remove(blob);
                error.set(x.what(), "database");
                return false;
            }
        }
        // Without tags, or with substitutes in imposed columns, and without record, nothing gives them back.
        out.report["tags"] = tags;
        out.report["restorable"] = (tags && text_made.empty()) || out.record > 0;
        return true;
    }

    void service::discard(long long record)
    {
        if (record <= 0) return;
        try
        {
            std::vector<std::string> blobs;
            db_->remove_file(record, blobs);
        }
        catch (const std::exception& x)
        {
            log_error("records: record " + std::to_string(record) + " half written stays: " + x.what());
        }
    }

    bool service::recorded_original(const file_record& r, restore_result& out, user_error& error, int depth)
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
        /* The original of a file protected again, by a newer engine for instance, is a protected file
           itself: the original that its own record keeps comes back in its place, so that the values
           protected before come back with the others, with or without tags. */
        file_record before;
        if (depth < 8 && db_->find_by_output(sha256_hex(out.bytes), before) && before.id != r.id)
        {
            restore_result inner;
            user_error unused;
            if (recorded_original(before, inner, unused, depth + 1))
            {
                wipe(out.bytes);
                out.bytes = std::move(inner.bytes);
                out.name = inner.name;
                out.report = inner.report;
                return true;
            }
        }
        /* The original of a protected file edited on the drive, or of a file written elsewhere
           from protected values, is that file: the tags it kept are restored too, with the
           substitutes of its properties, and the markers of its pictures leave it, as from any
           restoration without the original.  The parts of an Office document, of a PDF or of an
           archive are compressed: only a text shows its tags to a search. */
        const bool text = r.kind == "txt" || r.kind == "csv" || r.kind == "xml" || r.kind == "text";
        /* The substitutes that a protection without tags wrote in the original before, which the record
           inherited: those of its own protection are not in its original, which they would change. */
        const json inherited = text_substitutes_of_inherited(r.id);
        if (engine_ && r.kind != "image" &&
            (!text || !inherited.empty() || out.bytes.find("[[") != std::string::npos ||
             out.bytes.find("[Image@") != std::string::npos))
        {
            json request = json::object(), result;
            request["operation"] = "restore";
            request["name"] = r.original_name;
            if (!inherited.empty()) request["substitutions"] = inherited;
            std::string restored, e;
            const bool done =
                engine_->process(request, out.bytes, result, restored, nullptr, e, substitute_lookup());
            const long long put_back = done ? result["report"]["tags_restored"].as_int(0) +
                                              result["report"]["substitutes_restored"].as_int(0) +
                                              result["report"]["pictures_removed"].as_int(0) : 0;
            if (put_back > 0)
            {
                out.bytes = restored;
                out.report = result["report"];
            }
        }
        return true;
    }

    int service::extract_entry(const entry_record& e, file_record& archive, std::string& bytes, user_error& error)
    {
        if (!db_->get_file(e.record, archive)) return 0;
        std::string original;
        if (!vault_->get(archive.original_blob, original))
        {
            error.set("the original of the archive can no longer be read in the vault", "original_missing");
            return -2;
        }
        if (!engine_)
        {
            error.set("the engine is not ready", "engine");
            return -1;
        }
        json request = json::object(), result;
        request["operation"] = "extract";
        request["name"] = archive.original_name;
        request["entry"] = e.original;
        std::string why;
        const bool done = engine_->process(request, original, result, bytes, nullptr, why, substitute_lookup());
        wipe(original);
        if (!done)
        {
            error.set(why, "original_missing");
            return -2;
        }
        return 1;
    }

    int service::adopt(const std::string& sha256, const std::string& name, long long size, long long& record)
    {
        record = 0;
        entry_record e;
        file_record archive;
        std::string bytes;
        user_error error;
        if (!db_->find_entry(sha256, e)) return 0;
        const int extracted = extract_entry(e, archive, bytes, error);
        if (extracted <= 0)
        {
            if (extracted == -2) log_warning("records: an entry of record " + std::to_string(e.record) +
                                             " cannot be extracted from its original: " + error.text);
            return extracted == -1 ? -1 : 0;
        }
        // An entry larger than the space of the originals: the file stays as it is, protected, without record.
        if (!make_room(bytes.size()))
        {
            wipe(bytes);
            return 1;
        }
        const std::string blob = vault_->put(bytes);
        if (blob.empty())
        {
            wipe(bytes);
            return -1;
        }
        file_record r;
        r.source = "drive";
        const std::size_t cut = e.original.find_last_of("!/");
        r.original_name = cut == std::string::npos ? e.original : e.original.substr(cut + 1);
        r.output_name = name;
        r.original_sha256 = sha256_hex(bytes);
        r.output_sha256 = sha256;
        r.original_blob = blob;
        r.original_size = static_cast<long long>(bytes.size());
        r.output_size = size;
        r.kind = e.kind;
        r.tags = archive.tags;
        wipe(bytes);
        try
        {
            record = db_->add_file(r);
            // An archive extracted from the archive keeps the entries inside it, for the files extracted from it.
            db_->add_entries(record, db_->entries_inside(e));
            // Without tags, or with imposed columns, the substitutes of the archive, which hold those of the entry.
            db_->copy_text_substitutes(archive.id, record);
            counts_changed();
        }
        catch (const std::exception& x)
        {
            log_error("records: " + std::string(x.what()));
            discard(record);
            record = 0;
            vault_->remove(blob);
            return -1;
        }
        log_info("records: an entry of record " + std::to_string(e.record) + " extracted on the drive, recorded as " +
                 std::to_string(record));
        return 1;
    }

    bool service::restore(const std::string& bytes, const std::string& name, const std::string& object_id,
                          restore_result& out, user_error& error)
    {
        file_record r;
        const std::string sha = sha256_hex(bytes);
        if (db_->find_by_output(sha, r, name)) return recorded_original(r, out, error);
        /* The original was forgotten, at the request of a user, at the end of its time or to make
           room: the file is restored from its tags, which the key still opens, as a file written
           elsewhere from protected values is; what its protection removed, its pictures first,
           does not come back.  Its object identifier tells it only while no live record holds it
           since. */
        int64_t when = 0;
        std::string reason;
        bool forgotten_tags = true;
        const bool tracked = db_->find_by_object(object_id, r);
        const bool forgotten = db_->find_forgotten(sha, tracked ? std::string() : object_id, when, reason,
                                                   &forgotten_tags);
        const std::string day = forgotten ? iso_time(when).substr(0, 10) : std::string();
        // A file extracted from a protected archive: the entry of the original archive.
        entry_record entry;
        if (db_->find_entry(sha, entry))
        {
            file_record archive;
            const int extracted = extract_entry(entry, archive, out.bytes, error);
            if (extracted == -1) return false;
            if (extracted > 0)
            {
                const std::size_t cut = entry.original.find_last_of("!/");
                out.name = cut == std::string::npos ? entry.original : entry.original.substr(cut + 1);
                out.how = "original";
                out.record = archive.id;
                out.report = json::object();
                return true;
            }
            // An original that cannot be read: the tags of the file are restored rather.
            if (extracted == -2) log_warning("records: an entry of record " + std::to_string(entry.record) +
                                             " cannot be extracted from its original: " + error.text);
            error = user_error();
            out.bytes.clear();
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
            out.name = restore_name(without_suffix(name, ".txt"));
            out.how = "picture";
            out.report = json::object();
            return true;
        }
        // The text file of a picture that the vault no longer holds: its tags give nothing back.
        if (!id.empty())
        {
            if (forgotten)
                error.set("the picture of this file was forgotten on " + day + " and cannot be given back",
                          "picture_forgotten").arg("date", day);
            else
                error.set("the picture of this file is not in the vault of this computer", "picture_missing");
            return false;
        }
        if (!engine_)
        {
            error.set("the engine is not ready", "engine");
            return false;
        }
        json request = json::object();
        request["operation"] = "restore";
        request["name"] = name;
        /* A file protected without tags, or with imposed columns, and edited since, known by its object
           identifier: its substitutes. */
        if (tracked)
        {
            const json known = text_substitutes_of(r.id);
            if (!known.empty()) request["substitutions"] = known;
        }
        json result;
        std::string e;
        if (!engine_->process(request, bytes, result, out.bytes, nullptr, e, substitute_lookup()))
        {
            error.set(e, engine_code(e, "restore_failed")).arg("error", e);
            return false;
        }
        out.report = result["report"];
        const bool values =
            out.report["tags_restored"].as_int(0) > 0 || out.report["substitutes_restored"].as_int(0) > 0;
        /* Protected without tags, a file whose record was forgotten gives no value back: its substitutes
           alone are left, which nothing reads any more. */
        if (forgotten && !forgotten_tags && !values)
        {
            wipe(out.bytes);
            error.set("this file was protected without tags and its original was forgotten on " + day +
                      ": its values cannot come back", "tagless_forgotten").arg("date", day);
            return false;
        }
        if (!forgotten && !values && out.report["pictures_removed"].as_int(0) == 0)
        {
            error.set("this file holds no value that Pseudalis protected with this key", "not_protected");
            return false;
        }
        // The day and the reason of the forgetting, which the clients tell with the restoration.
        if (forgotten)
        {
            json f = json::object();
            f["date"] = day;
            f["reason"] = reason;
            out.report["original_forgotten"] = f;
        }
        // The name with its substitutes given back.
        out.name = result["name"].str(name);
        out.how = "tags";
        if (db_->find_by_object(object_id, r)) out.record = r.id;
        return true;
    }

    std::function<std::vector<std::string>(const std::string&)> service::substitute_lookup()
    {
        /* A candidate of four characters or fewer, a year for instance, is too common in a name or a
           property edited since to be taken for a substitute. */
        return [this](const std::string& candidate) -> std::vector<std::string> {
            if (candidate.size() < 5 || !db_) return std::vector<std::string>();
            try
            {
                return db_->tags_for(candidate);
            }
            catch (const std::exception& x)
            {
                log_error("substitutes: " + std::string(x.what()));
                return std::vector<std::string>();
            }
        };
    }

    json service::text_substitutes_of(long long record, bool inherited_only)
    {
        json list = json::array();
        try
        {
            for (const auto& t : db_->text_substitutes(record, inherited_only))
            {
                json one = json::object();
                one["substitute"] = t.first;
                one["tag"] = t.second;
                list.push_back(one);
            }
        }
        catch (const std::exception& x)
        {
            log_error("substitutes: " + std::string(x.what()));
        }
        return list;
    }

    bool service::record_substitutes(const json& result, user_error& error)
    {
        std::vector<std::pair<std::string, std::string>> made;
        // Those of the text, without tags, belong to the record of their file alone.
        for (const json& s : result["substitutions"].elements())
            if (!s["text"].as_bool(false)) made.push_back(std::make_pair(s["substitute"].str(), s["tag"].str()));
        try
        {
            db_->add_substitutes(made);
            return true;
        }
        catch (const std::exception& x)
        {
            error.set(x.what(), "database");
            return false;
        }
    }

    bool service::protect_name(const std::string& name, std::string& out, user_error& error)
    {
        std::shared_lock<std::shared_timed_mutex> work(work_m_);
        if (key_replaced_)
        {
            error.set("the service restarts with the key imported", "stopping");
            return false;
        }
        if (!engine_)
        {
            error.set("the engine is not ready", "engine");
            return false;
        }
        json request = json::object(), result;
        request["operation"] = "anonymize_name";
        request["name"] = "name";
        std::string e;
        if (!engine_->process(request, name, result, out, nullptr, e, substitute_lookup()))
        {
            error.set(e, "protect_failed").arg("error", e);
            return false;
        }
        return record_substitutes(result, error);
    }

    std::string service::restore_name(const std::string& name)
    {
        if (!engine_) return name;
        json request = json::object(), result;
        request["operation"] = "restore_name";
        request["name"] = "name";
        std::string out, e;
        if (!engine_->process(request, name, result, out, nullptr, e, substitute_lookup()) || out.empty())
            return name;
        return out;
    }

    bool service::text(const std::string& operation, const std::string& text, json& out, user_error& error,
                       bool tags, const json& substitutions)
    {
        if (!engine_)
        {
            error.set("the engine is not ready", "engine");
            return false;
        }
        json request = json::object();
        request["operation"] = operation;
        request["name"] = "text";
        request["tags"] = tags;
        if (substitutions.is_array()) request["substitutions"] = substitutions;
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
            if (result.contains("tags")) out["tags"] = result["tags"];
            if (result.contains("restored")) out["restored"] = result["restored"];
            if (result.contains("substitutions")) out["substitutions"] = result["substitutions"];
        }
        return true;
    }

    bool service::forget(long long record, const std::string& reason)
    {
        std::vector<std::string> blobs;
        try
        {
            if (!db_->remove_file(record, blobs, reason)) return false;
        }
        catch (const std::exception& x)
        {
            log_error("records: " + std::string(x.what()));
            return false;
        }
        counts_changed();
        // A blob that cannot be deleted now goes at the next start, nothing naming it any more.
        for (const std::string& b : blobs)
            if (!vault_->remove(b) && vault_->exists(b)) log_warning("vault: a blob of record " +
                std::to_string(record) + " cannot be deleted now, it goes at the next start");
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
        // 0: the originals of the files removed from the drive stay, within the space of the history.
        if (days <= 0) return;
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
            if (forget(r, "expired")) ++forgotten;
        if (forgotten)
            log_info("records: " + std::to_string(forgotten) + " original(s) of files removed from the drive " +
                     std::to_string(days) + " days ago or more forgotten");
    }

    void service::history_space(unsigned long long& used, unsigned long long& limit)
    {
        used = 0;
        try
        {
            const long long bytes = db_->originals_bytes();
            used = bytes > 0 ? static_cast<unsigned long long>(bytes) : 0;
        }
        catch (const std::exception& x)
        {
            log_error("records: " + std::string(x.what()));
        }
        limit = static_cast<unsigned long long>(settings_->get().history_mb) << 20;
    }

    bool service::make_room(unsigned long long needed)
    {
        std::lock_guard<std::mutex> lock(room_m_);
        const host_settings st = settings_->get();
        const unsigned long long limit = static_cast<unsigned long long>(st.history_mb) << 20;
        // The free space the disk of the data folder keeps beyond a new original.
        const unsigned long long reserve = 256ull << 20;
        unsigned long long used = 0, ignored = 0;
        history_space(used, ignored);
        // An original larger than the space of all of them: forgetting the others would not make it fit.
        if (needed > limit)
        {
            log_warning("records: an original of " + std::to_string(needed >> 20) + " MB is larger than the space " +
                        "of the originals (" + std::to_string(limit >> 20) + " MB), it is not kept");
            return false;
        }
        /*
            For a new original, the disk keeps 256 MB free once it is written: the originals make room
            for it when forgetting them would reach that, and stay when other files fill the disk or
            when its free space cannot be known (the vault then reports the lack of space).  The
            free space counts what each original forgotten gives back, its blob deleted at once.
        */
        uint64_t free_at_start = 0;
        bool disk = false;
        if (needed > 0 && free_disk_space(data_dir_, free_at_start) && free_at_start < reserve + needed)
        {
            disk = free_at_start + used >= reserve + needed;
            if (!disk)
                log_warning("records: the disk of the data folder has " + std::to_string(free_at_start >> 20) +
                            " MB free, the originals kept would not give the room an original needs");
        }
        unsigned long long freed = 0;
        auto full = [&]() {
            return used + needed > limit || (disk && free_at_start + freed < reserve + needed);
        };
        size_t forgotten = 0;
        while (full())
        {
            std::vector<std::pair<long long, long long>> oldest;
            try
            {
                oldest = db_->oldest_files(64);
            }
            catch (const std::exception& x)
            {
                log_error("records: " + std::string(x.what()));
                break;
            }
            if (oldest.empty()) break;
            bool any = false;
            for (const auto& r : oldest)
            {
                if (!full()) break;
                if (!forget(r.first, "space")) continue;
                any = true;
                ++forgotten;
                const unsigned long long size = r.second > 0 ? static_cast<unsigned long long>(r.second) : 0;
                used = used > size ? used - size : 0;
                freed += size;
            }
            if (!any) break;
        }
        if (forgotten)
            log_info("records: " + std::to_string(forgotten) + " original(s) first kept forgotten to make room, " +
                     std::to_string(used >> 20) + " MB kept of " + std::to_string(limit >> 20) + " MB");
        return true;
    }

    void service::request_cleanup()
    {
        cleanup_ = true;
        std::lock_guard<std::mutex> lock(wake_m_);
        wake_.notify_all();
    }

    json service::terms()
    {
        std::lock_guard<std::mutex> lock(terms_m_);
        return imposed_terms_json(terms_);
    }

    bool service::set_terms(const json& list, user_error& error)
    {
        std::vector<imposed_term> parsed;
        std::string message, code;
        long long index = -1;
        if (!parse_imposed_terms(list, parsed, message, code, index))
        {
            error.set(message, code);
            if (index >= 0) error.arg("index", std::to_string(index));
            return false;
        }
        const json canonical = imposed_terms_json(parsed);
        std::lock_guard<std::mutex> lock(terms_m_);
        std::string e;
        if (!engine_ || !engine_->set_terms(canonical, e))
        {
            error.set(e.empty() ? "the engine is not ready" : e, "engine");
            return false;
        }
        try
        {
            db_->set_value("imposed_terms", canonical.dump());
        }
        catch (const std::exception& x)
        {
            // The engine keeps the terms it had: those of the database, which the next start reads.
            std::string ignored;
            engine_->set_terms(imposed_terms_json(terms_), ignored);
            error.set(x.what(), "database");
            return false;
        }
        terms_ = parsed;
        log_info("terms: " + std::to_string(parsed.size()) + " imposed term(s) set");
        return true;
    }

    json service::columns()
    {
        std::lock_guard<std::mutex> lock(terms_m_);
        return imposed_columns_json(columns_);
    }

    bool service::set_columns(const json& list, user_error& error)
    {
        std::vector<imposed_column> parsed;
        std::string message, code;
        long long index = -1;
        if (!parse_imposed_columns(list, parsed, message, code, index))
        {
            error.set(message, code);
            if (index >= 0) error.arg("index", std::to_string(index));
            return false;
        }
        const json canonical = imposed_columns_json(parsed);
        std::lock_guard<std::mutex> lock(terms_m_);
        std::string e;
        if (!engine_ || !engine_->set_columns(canonical, e))
        {
            error.set(e.empty() ? "the engine is not ready" : e, "engine");
            return false;
        }
        try
        {
            db_->set_value("imposed_columns", canonical.dump());
        }
        catch (const std::exception& x)
        {
            // The engine keeps the columns it had: those of the database, which the next start reads.
            std::string ignored;
            engine_->set_columns(imposed_columns_json(columns_), ignored);
            error.set(x.what(), "database");
            return false;
        }
        columns_ = parsed;
        log_info("columns: " + std::to_string(parsed.size()) + " imposed column(s) set");
        return true;
    }

    bool service::export_key(const std::string& passphrase, std::string& exported, user_error& error)
    {
        if (key_replaced_)
        {
            error.set("the service restarts with the key imported", "stopping");
            return false;
        }
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
        // No protection runs from the check below to the restart (work_m_, key_replaced_).
        std::unique_lock<std::shared_timed_mutex> work(work_m_);
        if (key_replaced_)
        {
            error.set("the service restarts with the key imported", "stopping");
            return false;
        }
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
        key_replaced_ = true;
        log_info("keys: a master key was imported, the service restarts");
        request_restart();
        return true;
    }
}
