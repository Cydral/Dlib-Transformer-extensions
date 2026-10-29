// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_MODELS_H_
#define ALBIREO_HOST_MODELS_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../common/json.h"
#include "store.h"

namespace albireo
{

    class model_manager
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The models of the machine and of the catalog: the catalog published on
                GitHub (a signed JSON document listing the models, their descriptions in
                each language, their files and their hashes), the installed models, the
                downloads in progress and the imports from removable media.

            DOWNLOADS
                A model is downloaded as the parts the catalog lists (a large file is
                published in several parts, compressed or not), each cut into segments
                fetched by several connections at once and written at their place in a
                single file.  The progress is saved in the database, so that a download
                interrupted by a restart resumes where it stopped.  The file is checked
                against the hash of the catalog before it is installed; when it does not
                match, the parts whose own hash fails are fetched again, once.

            OFFLINE INSTALLATION
                import_path() installs the GGUF files of a folder or a single file, such as
                a memory card or a disk brought to a machine without network.  A file whose
                hash is in the catalog takes its entry; another one is described from its
                own metadata.
        !*/
    public:
        model_manager(store& db, const std::string& models_dir, const std::string& catalog_url);
        ~model_manager();

        // Fetches the catalog and checks its signature; keeps the last valid one.
        bool refresh_catalog(std::string& error);
        json catalog() const;
        bool catalog_entry(const std::string& id, json& entry) const;

        // The catalog merged with the installed models and the downloads, for the windows.
        json overview(const std::string& language) const;

        bool start_download(const std::string& id, std::string& error);
        void cancel_download(const std::string& id);
        void resume_pending();
        json downloads() const;

        bool remove(const std::string& id, std::string& error);
        // Recomputes the hash of an installed model; compares it with the recorded one and
        // with the catalog.
        json verify(const std::string& id);
        json import_path(const std::string& path);
        // Installs a file received by the service (a transfer from the tray), moved into
        // the models folder; sha is the hash computed while it was received.
        json import_received(const std::string& file, const std::string& original_name, const std::string& sha);

        // Called when a download completes or an import installs a model.
        void on_installed(std::function<void(const installed_model&)> f) { installed_cb_ = std::move(f); }

        const std::string& models_dir() const { return dir_; }

    private:
        struct job;
        void run_job(std::shared_ptr<job> j);
        bool install_file(const std::string& id, const std::string& file, const json& descriptor,
            const std::string& sha, const std::string& source, std::string& error);
        bool import_file(const std::string& f, const std::string& original_name, const std::string& known_sha,
            bool move, std::string& id, std::string& error);

        store& db_;
        std::string dir_;
        std::string catalog_url_;
        mutable std::mutex m_;
        json catalog_;
        std::map<std::string, std::shared_ptr<job>> jobs_;
        std::function<void(const installed_model&)> installed_cb_;
    };

    // A short description of a GGUF file read from its metadata: name, architecture,
    // parameters, quantization, context.
    json describe_gguf(const std::string& path);

}

#endif // ALBIREO_HOST_MODELS_H_
