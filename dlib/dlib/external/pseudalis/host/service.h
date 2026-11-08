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
#include "../common/imposed_terms.h"
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
                file.  restore() of a file the database does not know, written elsewhere from
                protected values or protected here before its original was forgotten, puts
                back the values of its tags, its picture markers removed; a text file holding
                the marker of a known picture gives that picture back.

                Without tags, the values of a file take their substitutes alone, and its
                record keeps them with the tags they stand for: the copies of the file give
                back their original, and a copy edited since, known by its NTFS identifier,
                its values from those substitutes.  Once the record forgotten, its values
                cannot come back, and a file protected without record never gives them back.
                A protected file protected again, by a newer engine for instance, keeps the
                values protected before as they are: its restoration gives back the original
                of each record of the chain that still keeps it, and the values of the tags
                of the others.
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

        /*
            Protects a file; with record, keeps its original (out.record), unless it is larger than
            history_mb; without tags, its values take their substitutes, which the record keeps.
            out.report["tags"] tells which, out.report["restorable"] whether the file can be restored.
            A file protected without tags and protected again, a copy of its protected file or, previous,
            the record of the version it replaces on the drive, keeps the substitutes of that record as
            they are, and its record inherits them, which give back the values of its original.
        */
        bool protect(const std::string& bytes, const std::string& name, const std::string& source, bool record,
            bool tags, protect_result& out, user_error& error, long long previous = 0);
        /*!
            The original of a protected file: the original of its record, that of an entry of
            a protected archive when it was extracted from one, a picture from its marker, else
            the file with the values of its tags put back and its picture markers removed, the
            day and the reason of a forgetting in out.report["original_forgotten"].
        !*/
        bool restore(const std::string& bytes, const std::string& name, const std::string& object_id,
            restore_result& out, user_error& error);
        /*
            The original of record r as restore() gives it: the original of its own record when it is
            a protected file that a record keeps, eight records deep at most, else its tags and the
            substitutes that r inherited restored when it holds some.
        */
        bool recorded_original(const file_record& r, restore_result& out, user_error& error, int depth = 0);
        /*!
            A file of the drive whose content is an entry of a protected archive that the
            database records, extracted from it, is recorded with the original of that entry,
            which the vault keeps apart from the original of the archive: each one is restored
            and forgotten alone.  Returns 1 with its new record (0 for an entry larger than
            history_mb, kept as it is without record), 0 when the content is no such entry or its
            original cannot be read, -1 when the engine or the vault cannot do it now.
        !*/
        int adopt(const std::string& sha256, const std::string& name, long long size, long long& record);
        // A name alone, a folder of the drive for instance: its values become substitutes, recorded.
        bool protect_name(const std::string& name, std::string& out, user_error& error);
        // A name with the substitutes the database knows given back, or the name as it is.
        std::string restore_name(const std::string& name);
        /*
            operation: "anonymize_text", "restore_text" or "detect".  Without tags, the text keeps no
            tag and out["substitutions"] lists the substitutes written, which restore_text gives back
            when substitutions holds them: nothing else keeps them.
        */
        bool text(const std::string& operation, const std::string& text, json& out, user_error& error,
            bool tags = true, const json& substitutions = json());
        /*!
            Deletes a record, its original and the pictures no other record shows.  With a
            reason, "forgotten" when a user asked for it, "expired" when its time is over or
            "space" when the oldest originals make room (make_room()), the protected file of
            the record is remembered: its restoration puts back the values of its tags from
            then on, and tells the day and the reason.
        !*/
        bool forget(long long record, const std::string& reason = std::string());
        // Marks a record whose file left the drive, deleted or replaced.  A move out of the drive
        // is a copy followed by a deletion: the original stays restorable from the copies of the
        // file for removed_days days (settings, 0 for always), then purge_removed() forgets it.
        void mark_removed(long long record);
        // Forgets the records whose file left the drive more than removed_days days ago, none for 0.
        void purge_removed();
        /*!
            Makes room for an original of needed bytes: the originals first kept are forgotten,
            the oldest first, while those kept with it would take more than history_mb
            (settings), and, for a new original, while the disk of the data folder would keep
            less than 256 MB free once it is written.  Forgetting starts only when it reaches its
            end: the originals stay when other files fill the disk, or when its free space cannot
            be known.  Returns false, forgetting none, for an original larger than history_mb,
            which is not kept.
        !*/
        bool make_room(unsigned long long needed);
        // Runs purge_removed() and make_room(0) in the background, at once: the settings changed.
        void request_cleanup();
        // The space the originals kept take, in bytes, and the one they may take.
        void history_space(unsigned long long& used, unsigned long long& limit);

        /*!
            The imposed terms (common/imposed_terms.h), kept in the database and given to the
            engine: set_terms() takes a list [{"term", "type", "case_sensitive"}] whole, or
            refuses it with its reason (codes terms_invalid, terms_too_many, terms_type,
            terms_duplicate, with the position of the term at fault in "index").
        !*/
        json terms();
        bool set_terms(const json& list, user_error& error);

        bool export_key(const std::string& passphrase, std::string& exported, user_error& error);
        // Allowed while the database records no file: the key of an earlier installation.
        bool import_key(const std::string& exported, const std::string& passphrase, user_error& error);

    private:
        bool start_engine(std::string& error);
        /*
            The original of the entry e, from the original of its archive, in bytes: 1, 0 when the
            record of the archive is gone, -1 when the engine cannot do it now, -2 when the original
            cannot be read.
        */
        int extract_entry(const entry_record& e, file_record& archive, std::string& bytes, user_error& error);
        // Removes a record half written, its original being removed by the caller.
        void discard(long long record);
        void background_loop();
        // The lookup of the engine: the tags recorded for a substitute.
        std::function<std::vector<std::string>(const std::string&)> substitute_lookup();
        // Records the substitutes of the names and the properties of a result of the engine.
        bool record_substitutes(const json& result, user_error& error);
        // The substitutes of the text of record, or those it inherited alone, as the engine reads them.
        json text_substitutes_of(long long record, bool inherited_only = false);
        json text_substitutes_of_inherited(long long record) { return text_substitutes_of(record, true); }

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
        std::atomic<bool> cleanup_{false};
        // make_room() runs for one caller at a time.
        std::mutex room_m_;
        std::mutex terms_m_;
        std::vector<imposed_term> terms_;
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
        // The counts of status(), which every icon asks every few seconds, kept two seconds, or until
        // a record is added or forgotten (counts_changed()).
        void counts_changed();
        std::mutex counts_m_;
        std::chrono::steady_clock::time_point counts_at_;
        long long records_count_ = 0;
        unsigned long long history_bytes_ = 0, history_limit_ = 0;
        json drive_status_ = json::object();

        friend class drive_host;
    };

    // The routes of the server (server.cpp).
    void install_routes(httplib::Server& srv, service& svc);
}

#endif // PSEUDALIS_HOST_SERVICE_H_
