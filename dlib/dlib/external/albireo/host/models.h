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
    class link_free_file;


    class model_manager
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The models of the machine and of the catalog: the catalog published in the
                folder dist of the public link (public_link.h), a signed JSON document
                listing the models, their descriptions in each language, their files and
                their hashes, the official source of each model among them, read from the
                link alone; the installed models, the downloads in progress and the imports
                from removable media.

            MODELS THAT READ IMAGES
                Such a model comes as two GGUF files: its decoder, and a projector
                (mmproj-*.gguf, general.architecture "clip") that holds the vision encoder.
                A catalog entry names it in "projector": {"files", "size", "sha256",
                "quantization"}, and its "inputs" are ["text", "image"].  Installed, the two
                files are <identifier>.gguf and <identifier>.mmproj.gguf in the models
                folder, recorded together (installed_model::projector); the engine receives
                the path of the projector with the model (options "projector").  A decoder
                installed without its projector reads text only.

            DOWNLOADS
                A model is downloaded as the parts the catalog lists (a large file is
                published in several parts, compressed or not), each cut into segments
                fetched by several connections at once and written at their place in its
                partial file: one job fetches the decoder and its projector, each into its
                own partial file, with one progress over both.  The progress is saved in
                the database, so that a download interrupted by a restart resumes where it
                stopped.  Each file is checked against its own hash of the catalog; when
                one does not match, the parts whose own hash fails are fetched again, once.
                The model is installed only once both files are in place: a cancellation,
                a hash that still fails or a model the database cannot record leaves
                neither file under its final name.

                Each segment comes from the copy of its part in the public link alone,
                models/<identifier>/<file name of the official address of the part>: the
                first attempt with the address of the cache, the next ones with a new one.
                The official address the catalog gives is never read.  A link that is not
                set, or that does not hold the part at its size in a new listing, ends the
                download in error; a part the link does not hold has the code
                download_not_published, which names it ("file"), since the files of the
                models reach the link while they are published.  The hashes of the catalog
                check the bytes.

            OFFLINE INSTALLATION
                import_path() installs the GGUF files of the import folder of the data
                folder, which only SYSTEM and the administrators fill: the files of a
                folder (and of its "models" subfolder) or a single file.  A file whose hash
                is in the catalog takes its entry; another one is described from its own
                metadata, the identity of a model of Albireo included (albireo.id,
                albireo.name, albireo.roles, albireo.inputs).  import_bundle() installs the
                models of a signed offline bundle, wherever it lies: only the files its
                signed catalog lists, decoders and projectors, found by their names and
                checked against their sizes and hashes before anything is copied.
                import_catalog() installs a signed catalog brought the same way.

                The projectors of a folder go with its decoders as pair_projectors() says:
                by the hashes of the catalog, else by the widths of their headers and by
                their names.  A projector without a decoder in its folder, or received
                alone (import_received()), is given to an installed decoder that lacks one
                and goes with it; otherwise it waits in the models folder as a pending
                projector (.pending-<hash>.mmproj.gguf), which the next decoder it goes
                with takes.  A decoder of an architecture that reads images installed
                without a projector reads text only until one comes.

                The files are read through no link (link_free_file), with a single name,
                so that no import makes the service read the file of another account.  A
                file whose content is that of an installed model or projector, a model that
                comes with Albireo included, or of a pending projector, is not copied
                again: it is listed under "already".  A file never takes the identifier of
                another installed model, unless it is a newer version from the catalog of a
                model of the catalog: it is installed as <identifier>-import, or
                <identifier>-import-2 and so on when that one is taken.

            THE MODELS FOLDER
                Every file the service writes there (a download, its partial files, an
                import, a projector, a transfer from the tray) is created in the folder
                reached through no link (direct_path()), checked again right before the
                file is created, renamed or deleted, and opened without following a link
                at its own name; a renamed file is checked to be the one written.

            MODELS THAT COME WITH ALBIREO
                register_builtin() records the models of the folder of the programs, which
                the complete installer places there, with their projectors.  They are
                listed like the others and take the functions no model holds; they cannot
                be deleted, only replaced.

            THE LISTS OF THE WINDOWS
                overview() lists the language models only: the embedding model serves the
                document search, which always has one, and is given apart ("embedding").
                Each model gives its "inputs" and its "publisher", the company that made
                it.
        !*/
    public:
        model_manager(store& db, const std::string& models_dir);
        ~model_manager();

        // Fetches the catalog from the public link and checks its signature; keeps the last
        // valid one, and the one in use when the fetched one was published before it.
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

        /*!
            ensures
                - Returns the catalog merged with the installed models and the downloads,
                  for the windows: "models" (the language models, each with "inputs",
                  "publisher", empty when unknown, "total_size", the bytes of the model and
                  of its projector, and "projector" for a model that has one),
                  "embedding_models", "pending_projectors" ({"name", "size", "sha256",
                  "projector_type", "projection_width"} of the projectors waiting for their
                  decoders), "downloads" and "catalog".
                - A catalog entry whose file a model that comes with Albireo carries (the
                  source_sha256 of its manifest) is not listed, unless it is installed under
                  its own identifier: the model that comes with Albireo stands for it.
        !*/
        json overview(const std::string& language) const;

        /*!
            ensures
                - Starts the download of the catalog entry id, its projector included.
                - Refuses, with error set, an entry that is not in the catalog, a model that
                  comes with Albireo, and an entry whose file such a model carries
                  (builtin_installed, naming that model).
        !*/
        bool start_download(const std::string& id, user_error& error);
        // Shows a download of id that could not start, with its error, until one starts.
        void note_failure(const std::string& id, const user_error& error);
        void cancel_download(const std::string& id);
        void resume_pending();
        json downloads() const;

        // Deletes the files of an installed model, its projector included, and its record.
        bool remove(const std::string& id, std::string& error);
        /*!
            ensures
                - Recomputes the hashes of an installed model and of its projector, and
                  compares them with the recorded ones ("local_match", true when both
                  match) and with the catalog ("catalog_match", "projector_match",
                  "catalog_projector_match").
        !*/
        json verify(const std::string& id);
        /*!
            requires
                - path is base, the import folder, or lies under it.
            ensures
                - Installs the GGUF files of path: a file, or a folder with its "models"
                  subfolder, each read through no link from base, the projectors with the
                  decoders they go with (see MODELS THAT READ IMAGES).  With move, the files
                  installed, those installed already and the projectors kept pending leave
                  path: moved into the models folder, or copied then deleted when it is on
                  another volume.  A file that cannot be installed stays where it is.
                - Returns {"imported": identifiers of the models installed, and of the
                  installed models that received their projector, "already": identifiers of
                  the installed models whose content the files repeat, "pending": names of
                  the projectors kept until their decoder comes, "errors": messages,
                  "files": [{"file", "result": imported, already, pending or error, "id"}]
                  for each file read}.
        !*/
        json import_path(const std::string& path, const std::string& base, bool move = false);
        /*!
            ensures
                - Installs the models of the offline bundle in folder that catalog, its
                  signed catalog, lists: models/<name> or <name>, the name of its single file
                  or <identifier>.gguf, read through no link from the root of its drive, with
                  the size and the hash of the catalog, and the same for the projector of an
                  entry that has one (<name> of its single file or <identifier>.mmproj.gguf).
                  The files stay in the bundle.
                - Returns {"imported", "already", "pending", "errors"} as import_path().
        !*/
        json import_bundle(const std::string& folder, const json& catalog);
        /*!
            ensures
                - Installs a file received by the service (a transfer from the tray), moved
                  into the models folder; sha is the hash computed while it was received.  A
                  decoder takes a pending projector it goes with; a projector goes to an
                  installed decoder that lacks one and goes with it, or waits as a pending
                  projector.
                - Returns {"imported", "already", "pending", "errors"} as import_path().
        !*/
        json import_received(const std::string& file, const std::string& original_name, const std::string& sha);

        /*!
            ensures
                - Registers the models that come with Albireo, listed by the manifest
                  builtin.json of dir, the "models" folder of the programs
                  (common/builtin_models.h): source "builtin", their files and their
                  projectors where they lie, their hashes as the installer checked them.
                  The models of an earlier manifest that this one no longer lists, or whose
                  file is gone, leave the database and are added to removed.
                - Returns the models this call registers for the first time.
        !*/
        std::vector<installed_model> register_builtin(const std::string& dir, std::vector<installed_model>& removed);

        // Called when a download completes, an import installs a model or gives a projector
        // to an installed model.
        void on_installed(std::function<void(const installed_model&)> f) { installed_cb_ = std::move(f); }

        /*!
            ensures
                - f(id, place) is called when new files take the names of those of the
                  installed model id, a newer version downloaded or imported: f runs place,
                  which renames them, while no program holds the files of id open (the engine
                  holds those of a loaded model, which Windows refuses to replace), and returns
                  its result.  Without f, place runs directly.
        !*/
        typedef std::function<bool(const std::string&, const std::function<bool()>&)> replace_function;
        void on_replace(replace_function f) { replace_cb_ = std::move(f); }

        // Ends the downloads and the refresh in progress and refuses the following ones:
        // the service stops.  The progress of the downloads is kept for the next start.
        void stop();

        const std::string& models_dir() const { return dir_; }

    private:
        struct job;
        // A GGUF file of an import, opened through no link, hashed and described.
        struct import_item;
        // How put_file() placed a file: moved, copied, or found at its name already.
        struct placement
        {
            bool moved = false;
            bool copied = false;
            bool in_place = false;
        };
        void run_job(std::shared_ptr<job> j);
        bool install_catalog(const std::string& body, const std::string& sig, bool keep_newer, json& result,
            std::string& error);
        // Records the model id of file, with its projector when projector is not empty.
        bool install_file(const std::string& id, const std::string& file, const json& descriptor,
            const std::string& sha, const std::string& source, const std::string& projector,
            const std::string& projector_sha, std::string& error);
        /*!
            ensures
                - Opens item.path through no link from item.base, with a single name, hashes it
                  (known_sha when given) and describes it; false with item.error otherwise.
        !*/
        bool inspect(import_item& item, const std::string& known_sha);
        /*!
            ensures
                - Installs the files of items, inspected, as import_path() describes: the
                  decoders with the projectors they go with, the other projectors given to
                  installed decoders or kept pending.  Sets the result of each item.
        !*/
        void import_items(std::vector<import_item>& items, bool move);
        // The answer of an import: imported, already, pending, errors and files, from the results of items.
        static json import_result(const std::vector<import_item>& items);
        /*!
            ensures
                - Installs the decoder d with the projector p, or with the pending projector
                  pending, or alone.  Returns true with d.id set.
        !*/
        bool import_decoder(import_item& d, import_item* p, const pending_projector* pending, bool move,
            std::string& error);
        // Gives the projector p to the installed model m, which lies in the models folder and lacks one.
        bool attach_projector(import_item& p, const installed_model& m, bool move, std::string& error);
        // Keeps the projector p in the models folder until a decoder it goes with comes.
        bool keep_pending(import_item& p, bool move, std::string& error);
        // The pending projector the decoder d goes with (pair_projectors()), or false.
        bool pending_for(import_item& d, pending_projector& found);
        // The installed decoder of the models folder, without a projector, that p goes with, or false.
        bool installed_for(const import_item& p, installed_model& found);
        // Renames the pending projector to target, its record removed; false when it stays pending.
        bool take_pending(const pending_projector& pending, const std::string& target);
        // Gives back to the pending projectors the one take_pending() renamed to target.
        void return_pending(const pending_projector& pending, const std::string& target);
        // Deletes the pending projector of content sha, once a model is installed with the same projector.
        void drop_pending_copy(const std::string& sha);
        /*!
            ensures
                - Opens out at models/<name> or <name> of folder, name the name of the single
                  part of parts or fallback, through no link from the root of its drive, and
                  checks its size and its hash.  False when no such file is there, or with
                  out.error when it is refused.
        !*/
        bool bundle_file(const std::string& folder, const json& parts, const std::string& fallback,
            const std::string& sha, uint64_t size, import_item& out);
        /*!
            ensures
                - Places the file source, opened through no link at path under base, at target
                  in the models folder: moved when move and when it lies on the volume of the
                  folder, copied otherwise from the open file and checked against sha as it is
                  written.  Closes source; returns false with error when it cannot be placed.
                - target is a file of the installed model id, when id is not empty: the
                  renaming then runs through replacing().
        !*/
        bool put_file(link_free_file& source, const std::string& path, const std::string& base,
            const std::string& sha, bool move, const std::string& target, placement& how, std::string& error,
            const std::string& id);
        /*
            Runs work, which renames new files over those of the model id, through the function
            of on_replace() when id names an installed model.
        */
        bool replacing(const std::string& id, const std::function<bool()>& work);
        // Takes back what put_file() did: the file moved goes back to path, the copy is deleted.
        void undo_put(const std::string& path, const std::string& target, const placement& how);
        // The installed model whose file has the content of hash sha, the one that comes with Albireo
        // first, or false.
        bool installed_with_hash(const std::string& sha, installed_model& m);
        /*!
            ensures
                - Returns true when a projector of content sha is installed, with holder the
                  model that has it, or waits as a pending projector, with holder empty and
                  pending set; path receives its file.
        !*/
        bool projector_with_hash(const std::string& sha, std::string& holder, std::string& path);
        // The SHA-256 of the projector the catalog entry of the decoder of hash sha names, or an empty string.
        std::string catalog_projector(const std::string& decoder_sha) const;
        /*
            The identifier a new file takes: id when no model holds it, or when update is true and
            the model of the catalog that holds it receives the file as its newer version;
            otherwise id-import, id-import-2, id-import-3..., the first one free, so that a file
            never takes the place of another model.
        */
        std::string free_identifier(const std::string& id, bool update);
        // Moves the file from to target, a name of the models folder: false when the folder is no longer
        // reached directly, when the move fails, or when target then names another file than the one moved.
        bool place(const std::string& from, const std::string& target);

        store& db_;
        std::string dir_;
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
        replace_function replace_cb_;
    };

    /*!
        ensures
            - Returns a short description of a GGUF file read from its metadata: "name",
              "arch", "parameters", "quantization", "gguf_name" (general.name), and
              "publisher" when the header names the company that made the model
              (general.organization, else general.author).
            - For a decoder: "context", "experts", "embedding", "roles", "reasoning",
              "tools", "inputs", the identity of a file of Albireo ("albireo_id") and
              "reads_images", true for an architecture that reads images (reads_images())
              and for a vocabulary that holds the token <image> (LLaVA); for these,
              "projection_width", the width of the vision tokens it takes:
              embedding_length * (1 + n_deepstack_layers).
            - For a projector (general.architecture "clip"): "projector" true,
              "projector_type" (clip.projector_type), "projection_width", the width of the
              vision tokens it gives: clip.vision.projection_dim * (1 + the layers of
              clip.vision.is_deepstack_layers), or the output width of its last layer when
              the header names none (LLaVA), "image_size" and "vision" (whether it holds a
              vision encoder).
            - "error" when the file cannot be read.
    !*/
    json describe_gguf(const std::string& path);

    // Whether the decoders of an architecture take the vision tokens of a projector: qwen3vl, qwen3vlmoe.
    bool reads_images(const std::string& arch);

    // A GGUF file as the pairing of the imports sees it.
    struct pairing_file
    {
        std::string name;                 // the name of the file
        std::string sha256;
        json description;                 // describe_gguf()
        std::string expected_projector;   // a decoder: the SHA-256 of the projector its catalog entry names
    };

    /*!
        ensures
            - Returns, for each projector, the index in decoders of the decoder it goes with,
              or -1:
                - a decoder takes the projector whose SHA-256 its catalog entry names
                  (expected_projector);
                - the other projectors go with a decoder of an architecture that reads images
                  whose projection width is theirs ("projection_width" of describe_gguf());
                  among several, the one whose name is closest to that of the projector: the
                  same general.name in both headers first, then the same file name once the
                  prefix "mmproj", the quantization and the extension are set aside, then the
                  longest common beginning of these names; none when two are equally close.
            - Each decoder takes one projector at most.
    !*/
    std::vector<int> pair_projectors(const std::vector<pairing_file>& decoders,
        const std::vector<pairing_file>& projectors);

    // The inputs of an installed model: those its entry names, "text" at least, with "image" when it has
    // a projector and without it otherwise.
    json model_inputs(const installed_model& m);

    // The inputs of a catalog entry: those it names, with "image" only when it names a projector.
    json entry_inputs(const json& entry);

}

#endif // ALBIREO_HOST_MODELS_H_
