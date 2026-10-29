// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_STORE_H_
#define ALBIREO_HOST_STORE_H_

#include <cstdint>
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
        std::string source;          // "catalog" or "import"
        int64_t installed_at = 0;
        int64_t verified_at = 0;     // last successful check of the hash
        json descriptor;             // the catalog entry, or what the import learned
    };

    class store
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The database of the service (SQLite, in the data folder): the installed
                models, the state of the downloads, so that an interrupted one resumes
                where it stopped, the daily usage figures and a few named values.

                Every function locks the database for its duration, so that the threads
                of the server, of the downloads and of the engine may share it.
        !*/
    public:
        explicit store(const std::string& path);
        ~store();

        std::vector<installed_model> models();
        bool find_model(const std::string& id, installed_model& m);
        void put_model(const installed_model& m);
        void remove_model(const std::string& id);
        void set_verified(const std::string& id, int64_t when);

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
        std::mutex m_;
        std::unique_ptr<dlib::database> db_;
    };

}

#endif // ALBIREO_HOST_STORE_H_
