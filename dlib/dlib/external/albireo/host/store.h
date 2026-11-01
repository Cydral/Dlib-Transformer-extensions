// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_STORE_H_
#define ALBIREO_HOST_STORE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../common/json.h"

namespace dlib { class database; }

namespace albireo
{

    struct installed_model
    {
        std::string id;              // identifier of the catalog, or of the import
        std::string name;
        std::string file;            // absolute path of the GGUF file
        uint64_t size = 0;
        std::string sha256;          // of the file as installed
        std::string source;          // "catalog", "import" or "builtin"
        int64_t installed_at = 0;
        int64_t verified_at = 0;     // last successful check of the hashes
        json descriptor;             // the catalog entry, or what the import learned
        // The projector of a model that reads images (an mmproj file, general.architecture
        // "clip", which holds the vision encoder): its absolute path, its size and its SHA-256.
        // Empty for a model that reads text only.
        std::string projector;
        uint64_t projector_size = 0;
        std::string projector_sha256;
    };

    // A projector that arrived without its decoder, kept in the models folder until a
    // decoder it goes with is installed.
    struct pending_projector
    {
        std::string sha256;
        std::string file;            // absolute path, in the models folder
        uint64_t size = 0;
        std::string name;            // the name of the file it came from
        json descriptor;             // describe_gguf()
        int64_t received_at = 0;
    };

    class store
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The database of the service (SQLite, in the data folder): the installed
                models with their projectors, the projectors waiting for their decoders,
                the state of the downloads, so that an interrupted one resumes where it
                stopped, the daily usage figures and a few named values.

                A database of an earlier version receives the columns of the projectors
                when it opens.  The constructor checks every page of the database (PRAGMA
                quick_check) and throws an error that names it "malformed" when one fails.

                Every function locks the database for its duration, so that the threads
                of the server, of the downloads and of the engine may share it.

                A failure throws the error of SQLite, which the callers report.  The
                database held by another program is waited for five seconds.  SQLite opens a
                file it cannot write read-only, and its connection stays so: a write that
                fails on it opens the connection again, so that the writes work again as
                soon as the file can be written.
        !*/
    public:
        explicit store(const std::string& path);
        ~store();

        std::vector<installed_model> models();
        bool find_model(const std::string& id, installed_model& m);
        void put_model(const installed_model& m);
        void remove_model(const std::string& id);
        void set_verified(const std::string& id, int64_t when);

        // The projectors waiting for their decoders, oldest first.
        std::vector<pending_projector> projectors();
        void put_projector(const pending_projector& p);
        void remove_projector(const std::string& sha256);

        // Download jobs: the parts received so far are recorded by index.
        json download_state(const std::string& id);
        void set_download_state(const std::string& id, const json& state);
        void remove_download_state(const std::string& id);
        std::vector<std::string> pending_downloads();

        void add_usage(int64_t prompt_tokens, int64_t generated_tokens, int64_t requests);
        json usage(int days);

        std::string value(const std::string& key, const std::string& def = std::string());
        void set_value(const std::string& key, const std::string& v);

    private:
        // Opens the connection and sets its options; m_ is held, or the store is being built.
        void connect();
        // Runs a write; a connection that SQLite opened read-only is opened again once.  m_ is held.
        void writing(const std::function<void()>& work);

        std::mutex m_;
        std::string path_;
        std::unique_ptr<dlib::database> db_;
    };

}

#endif // ALBIREO_HOST_STORE_H_
