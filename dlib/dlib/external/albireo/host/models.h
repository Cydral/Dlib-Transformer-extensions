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
#include "user_error.h"

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
                import_path() installs the GGUF files of the import folder of the data
                folder, which only SYSTEM and the administrators fill: the files of a
                folder (and of its "models" subfolder) or a single file.  A file whose hash
                is in the catalog takes its entry; another one is described from its own
                metadata, the identity of a model of Albireo included (albireo.id,
                albireo.name, albireo.roles, albireo.inputs).  import_bundle() installs the
                models of a signed offline bundle, wherever it lies: only the files its
                signed catalog lists, found by their names and checked against their sizes
                and hashes before anything is copied.  import_catalog() installs a signed
                catalog brought the same way.

                The files are read through no link (link_free_file), with a single name,
                so that no import makes the service read the file of another account.  A
                file whose content is that of an installed model, a model that comes with
                Albireo included, is not copied again: it is listed under "already".  A
                file never takes the identifier of a model that comes with Albireo: it is
                installed as <identifier>-import.

            THE MODELS FOLDER
                Every file the service writes there (a download, its partial file, an
                import, a transfer from the tray) is created in the folder reached through
                no link (direct_path()), checked again right before the file is created,
                renamed or deleted, and opened without following a link at its own name;
                a renamed file is checked to be the one written.

            MODELS THAT COME WITH ALBIREO
                register_builtin() records the models of the folder of the programs, which
                the complete installer places there.  They are listed like the others and
                take the functions no model holds; they cannot be deleted, only replaced.

            THE LISTS OF THE WINDOWS
                overview() lists the language models only: the embedding model serves the
                document search, which always has one, and is given apart ("embedding").
        !*/
    public:
        model_manager(store& db, const std::string& models_dir, const std::string& catalog_url);
        ~model_manager();

        // Fetches the catalog and checks its signature; keeps the last valid one, and the
        // one in use when the fetched one was published before it.
        bool refresh_catalog(std::string& error);
        // The same on a thread of its own, one refresh at a time; the caller never waits.
        void refresh_catalog_async();
        /*!
            ensures
                - Installs the catalog of folder (catalog.json and catalog.json.sig), read
                  through no link from base (an empty base: the root of the drive), its
                  signature checked as for a downloaded one, unless the catalog in use was
                  published later.
                - Returns {"models": number of models} when installed, {"state": "kept"}
                  when the catalog in use is more recent, {"error"} otherwise.
        !*/
        json import_catalog(const std::string& folder, const std::string& base);
        // The catalog in use: "state" (idle, checking, ok, error), "error", "updated",
        // "models" (their number) and "time" (of the last installation, Unix seconds).
        json catalog_status() const;
        // Changes each time a catalog is installed.
        unsigned catalog_generation() const { return generation_; }
        json catalog() const;
        bool catalog_entry(const std::string& id, json& entry) const;

        // The catalog merged with the installed models and the downloads, for the windows.
        json overview(const std::string& language) const;

        bool start_download(const std::string& id, user_error& error);
        // Shows a download of id that could not start, with its error, until one starts.
        void note_failure(const std::string& id, const user_error& error);
        void cancel_download(const std::string& id);
        void resume_pending();
        json downloads() const;

        bool remove(const std::string& id, std::string& error);
        // Recomputes the hash of an installed model; compares it with the recorded one and
        // with the catalog.
        json verify(const std::string& id);
        /*!
            requires
                - path is base, the import folder, or lies under it.
            ensures
                - Installs the GGUF files of path: a file, or a folder with its "models"
                  subfolder, each read through no link from base.  With move, the files
                  installed, and those installed already, leave path: moved into the models
                  folder, or copied then deleted when it is on another volume.  A file that
                  cannot be installed stays where it is.
                - Returns {"imported": identifiers, "already": identifiers of the installed
                  models whose content the files repeat, "errors": messages}.
        !*/
        json import_path(const std::string& path, const std::string& base, bool move = false);
        /*!
            ensures
                - Installs the models of the offline bundle in folder that catalog, its
                  signed catalog, lists: models/<name> or <name>, the name of its single file
                  or <identifier>.gguf, read through no link from the root of its drive, with
                  the size and the hash of the catalog.  The files stay in the bundle.
                - Returns {"imported", "already", "errors"} as import_path().
        !*/
        json import_bundle(const std::string& folder, const json& catalog);
        // Installs a file received by the service (a transfer from the tray), moved into
        // the models folder; sha is the hash computed while it was received.
        json import_received(const std::string& file, const std::string& original_name, const std::string& sha);

        /*!
            ensures
                - Registers the models that come with Albireo, listed by the manifest
                  builtin.json of dir, the "models" folder of the programs
                  (common/builtin_models.h): source "builtin", their files where they lie,
                  their hashes as the installer checked them.  The models of an earlier
                  manifest that this one no longer lists, or whose file is gone, leave the
                  database; their identifiers are added to removed.
                - Returns the models this call registers for the first time.
        !*/
        std::vector<installed_model> register_builtin(const std::string& dir, std::vector<std::string>& removed);

        // Called when a download completes or an import installs a model.
        void on_installed(std::function<void(const installed_model&)> f) { installed_cb_ = std::move(f); }

        // Ends the downloads and the refresh in progress and refuses the following ones:
        // the service stops.  The progress of the downloads is kept for the next start.
        void stop();

        const std::string& models_dir() const { return dir_; }

    private:
        struct job;
        void run_job(std::shared_ptr<job> j);
        bool install_catalog(const std::string& body, const std::string& sig, bool keep_newer, json& result,
            std::string& error);
        bool install_file(const std::string& id, const std::string& file, const json& descriptor,
            const std::string& sha, const std::string& source, std::string& error);
        /*!
            ensures
                - Installs the file f, read through no link from base.  Returns true with id
                  set, and already true when an installed model has its content (no copy is
                  made, and with move f is deleted).
        !*/
        bool import_file(const std::string& f, const std::string& base, const std::string& original_name,
            const std::string& known_sha, bool move, std::string& id, bool& already, std::string& error);
        // The installed model whose file has the content of hash sha, the one that comes with Albireo
        // first, or false.
        bool installed_with_hash(const std::string& sha, installed_model& m);
        // The identifier a new file takes: id, or id-import when a model that comes with Albireo holds id.
        std::string free_identifier(const std::string& id);
        // Moves the file from to target, a name of the models folder: false when the folder is no longer
        // reached directly, when the move fails, or when target then names another file than the one moved.
        bool place(const std::string& from, const std::string& target);

        store& db_;
        std::string dir_;
        std::string catalog_url_;
        mutable std::mutex m_;
        json catalog_;
        std::string catalog_state_ = "idle";
        std::string catalog_error_;
        std::atomic<unsigned> generation_{0};
        std::thread refresher_;
        std::mutex refresher_m_;            // the start of a refresh, from a route or the service
        std::atomic<bool> refreshing_{false};
        std::atomic<bool> stopped_{false};
        std::map<std::string, std::shared_ptr<job>> jobs_;
        std::function<void(const installed_model&)> installed_cb_;
    };

    // A short description of a GGUF file read from its metadata: name, architecture,
    // parameters, quantization, context.
    json describe_gguf(const std::string& path);

}

#endif // ALBIREO_HOST_MODELS_H_
