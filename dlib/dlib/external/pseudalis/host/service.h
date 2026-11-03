// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_SERVICE_H_
#define PSEUDALIS_HOST_SERVICE_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "../common/engine_loader.h"
#include "../common/json.h"
#include "albireo_client.h"
#include "settings.h"
#include "store.h"
#include "updater.h"
#include "user_error.h"
#include "vault.h"

namespace httplib { class SSLServer; class Server; }

namespace pseudalis
{
    class drive_host;

    struct protect_result
    {
        std::string bytes;      // the protected file
        std::string name;       // its name: the name given, or <name>.txt for a picture
        std::string kind;
        json report;
        long long record = 0;   // the record of the original, 0 when none was kept
        std::string sha256;     // of bytes
    };

    struct restore_result
    {
        std::string bytes;
        std::string name;       // the original name, when the original comes from the vault
        std::string how;        // "original": the bytes kept; "picture": a picture from its marker; "tags"
        json report;
        long long record = 0;
    };

    class work_slots
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The files and texts of the interface processed at once.  Each one stands in
                memory several times while it is read, processed and answered, so their
                number is bounded by the processors and the memory of the computer; a request
                waits for a slot before its body is read.  The drive keeps its own bound.
        !*/
    public:
        void init(int slots);
        // Waits for a slot, until wait; false when none came free in time.
        bool acquire(std::chrono::milliseconds wait);
        void release();
        int size() const { return size_; }

    private:
        std::mutex m_;
        std::condition_variable cv_;
        int free_ = 0;
        int size_ = 0;
    };

    // A slot of work_slots held by its scope, after a wait of one minute at most.
    class work_slot
    {
    public:
        explicit work_slot(work_slots& slots) : slots_(slots), held_(slots.acquire(std::chrono::seconds(60))) {}
        ~work_slot() { if (held_) slots_.release(); }
        work_slot(const work_slot&) = delete;
        work_slot& operator=(const work_slot&) = delete;
        bool held() const { return held_; }

    private:
        work_slots& slots_;
        bool held_;
    };

    class service
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The Pseudalis service: the engine and its key, the vault and the database of
                the originals, the local HTTPS server (the web application, its interface and
                its documentation), the drive, the descriptions of the pictures by Albireo,
                and the updates.  run() starts everything and blocks until stop() or a
                restart request; the caller then starts a new service when
                restart_requested() says so, which applies a change of port or of drive.
                When an update downloaded earlier is ready, run() starts its installer
                instead and returns with update_started() true.

            PROTECTION
                protect() runs the engine on a file: each picture is kept in the vault,
                described by Albireo when it answers (the description is kept with it, for
                the next files that show it), and with record, the original is kept in the
                vault and recorded with the SHA-256 of the protected file, so that
                restore() gives back the original itself, from any copy of the protected
                file.  restore() of a file the database does not know puts back the values
                of its tags, and a text file holding the marker of a known picture gives
                that picture back.
        !*/
    public:
        service();
        ~service();

        bool run(std::string& error);
        void stop();
        bool stopping() const { return stopping_; }
        bool restart_requested() const { return restart_; }
        bool update_started() const { return update_started_; }
        void request_restart();

        std::string program_dir() const { return program_dir_; }
        std::string data_dir() const { return data_dir_; }
        std::string admin_token() const { return admin_token_; }
        int port() const { return port_; }

        settings_store& settings() { return *settings_; }
        store& db() { return *db_; }
        vault& blobs() { return *vault_; }
        updater& updates() { return *updater_; }
        albireo_client& albireo() { return albireo_; }
        work_slots& work() { return work_; }

        json status();
        // The language of the descriptions and of the messages: the setting, else the one of the system.
        std::string language() const;

        bool protect(const std::string& bytes, const std::string& name, const std::string& source, bool record,
            protect_result& out, user_error& error);
        bool restore(const std::string& bytes, const std::string& name, const std::string& object_id,
            restore_result& out, user_error& error);
        // A name alone, a folder of the drive for instance: its values become substitutes, recorded.
        bool protect_name(const std::string& name, std::string& out, user_error& error);
        // A name with the substitutes the database knows given back, or the name as it is.
        std::string restore_name(const std::string& name);
        // operation: "anonymize_text", "restore_text" or "detect".
        bool text(const std::string& operation, const std::string& text, json& out, user_error& error);
        /*!
            Deletes a record, its original and the pictures no other record shows.  With a
            reason, "forgotten" when a user asked for it or "expired" when its time is over, the
            protected file of the record is remembered, and its restoration refused from then on.
        !*/
        bool forget(long long record, const std::string& reason = std::string());
        // Marks a record whose file left the drive, deleted or replaced.  A move out of the drive
        // is a copy followed by a deletion: the original stays restorable from the copies of the
        // file for removed_days days (settings), then purge_removed() forgets it.
        void mark_removed(long long record);
        // Forgets the records whose file left the drive more than removed_days days ago.
        void purge_removed();

        bool export_key(const std::string& passphrase, std::string& exported, user_error& error);
        // Allowed while the database records no file: the key of an earlier installation.
        bool import_key(const std::string& exported, const std::string& passphrase, user_error& error);

    private:
        bool start_engine(std::string& error);
        void background_loop();
        // The lookup of the engine: the tags recorded for a substitute.
        std::function<std::vector<std::string>(const std::string&)> substitute_lookup();
        bool record_substitutes(const json& result, user_error& error);

        std::string program_dir_, data_dir_, admin_token_;
        int port_ = 8444;
        std::unique_ptr<settings_store> settings_;
        std::unique_ptr<store> db_;
        std::unique_ptr<vault> vault_;
        std::unique_ptr<updater> updater_;
        std::shared_ptr<engine_library> library_;
        std::unique_ptr<engine_client> engine_;
        std::string key_fingerprint_;
        albireo_client albireo_;
        std::unique_ptr<drive_host> drive_;
        std::unique_ptr<httplib::SSLServer> server_;
        std::mutex server_m_;
        std::thread background_;
        std::mutex wake_m_;
        std::condition_variable wake_;
        std::atomic<bool> stopping_{false};
        std::atomic<bool> restart_{false};
        std::atomic<bool> update_started_{false};
        std::mutex key_m_;
        std::string master_;
        int64_t started_ = 0;
        work_slots work_;
        /*
            The protections hold work_m_ shared; the import of a key holds it alone from its
            check to the save of the key, after which no protection starts until the restart
            (key_replaced_): a protection made meanwhile would use the key that goes.
        */
        std::shared_timed_mutex work_m_;
        std::atomic<bool> key_replaced_{false};
        // The counts of status(), which every icon asks every few seconds, kept two seconds.
        std::mutex counts_m_;
        std::chrono::steady_clock::time_point counts_at_;
        long long records_count_ = 0;
        json drive_status_ = json::object();

        friend class drive_host;
    };

    // The routes of the server (server.cpp).
    void install_routes(httplib::Server& srv, service& svc);
}

#endif // PSEUDALIS_HOST_SERVICE_H_
