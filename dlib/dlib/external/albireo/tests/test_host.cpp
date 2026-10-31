// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the files the service writes in its models folder and reads in its imports, in a
    folder of its own, with a local server for the downloads.

        test_host <scratch folder> [port]

    The downloads create, open and rename their files in the models folder reached through
    no link: a partial file or a final name that is a link, a partial file with a second
    name, and a models folder replaced by a link are refused, and the files they lead to stay
    as they were.  The imports read through no link and a single name, an offline bundle
    gives only the files its catalog lists, and a file whose content is that of an installed
    model, one that comes with Albireo included, is not copied again nor recorded over it.
    The links need a system that lets every account make them: they are skipped on Windows.
*/

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <dlib/external/httplib/httplib.h>

#include "../common/builtin_models.h"
#include "../common/platform.h"
#include "../host/crypto.h"
#include "../host/models.h"
#include "../host/store.h"

namespace
{
    using albireo::json;

    int failures = 0;

    void check(bool ok, const std::string& what)
    {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
        if (!ok) ++failures;
    }

    void write_text(const std::string& path, const std::string& text)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << text;
    }

    std::string read_text(const std::string& path)
    {
        std::string data;
        albireo::read_file(path, data);
        return data;
    }

    // A content of size bytes that differs with seed.
    std::string blob(size_t size, char seed)
    {
        std::string b(size, '\0');
        for (size_t i = 0; i < size; ++i) b[i] = static_cast<char>(seed + i * 31 + (i >> 9));
        return b;
    }

    json catalog_entry(const std::string& id, const std::string& content, const std::string& url)
    {
        json f = json::object();
        f["url"] = url;
        f["size"] = static_cast<unsigned long long>(content.size());
        f["sha256"] = albireo::sha256_hex(content);
        json files = json::array();
        files.push_back(f);
        json e = json::object();
        e["id"] = id;
        e["name"] = id;
        e["size"] = static_cast<unsigned long long>(content.size());
        e["sha256"] = albireo::sha256_hex(content);
        e["files"] = files;
        json roles = json::array();
        roles.push_back("general");
        e["roles"] = roles;
        return e;
    }

    // Waits for the download of id to leave its running states; returns its last state.
    json wait_download(albireo::model_manager& m, const std::string& id)
    {
        for (int i = 0; i < 600; ++i)
        {
            // The list held by a variable, which the loop reads after the call ends.
            const json downloads = m.downloads();
            for (const auto& d : downloads.elements())
                if (d["id"].as_string() == id)
                {
                    const std::string s = d["state"].as_string();
                    if (s != "queued" && s != "downloading" && s != "verifying") return d;
                }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return json();
    }

#if !defined(_WIN32)
    bool make_symlink(const std::string& target, const std::string& link)
    {
        return symlink(target.c_str(), link.c_str()) == 0;
    }

    bool make_hard_link(const std::string& target, const std::string& link)
    {
        return ::link(target.c_str(), link.c_str()) == 0;
    }
#endif

    bool installed(albireo::store& db, const std::string& id, albireo::installed_model& m)
    {
        return db.find_model(id, m);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: test_host <scratch folder> [port]\n");
        return 2;
    }
    const int port = argc >= 3 ? std::atoi(argv[2]) : 18487;
    const std::string root = albireo::final_path(argv[1]).empty() ? argv[1] : albireo::final_path(argv[1]);
    const std::string work = albireo::join_path(root, "test_host-" + albireo::random_hex(8));
    const std::string models_dir = albireo::join_path(work, "models");
    const std::string outside = albireo::join_path(work, "outside");
    albireo::make_dirs(models_dir);
    albireo::make_dirs(outside);
#if !defined(_WIN32)
    setenv("ALBIREO_HOME", albireo::join_path(work, "data").c_str(), 1);
#endif
    const std::string inbox = albireo::join_path(albireo::data_dir(), "import");
    albireo::make_dirs(inbox);

    // The files the local server gives, with ranges.
    const std::string served = blob(300000, 'a');
    const std::string other = blob(200000, 'k');
    httplib::Server server;
    server.Get("/model.bin", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(served, "application/octet-stream");
    });
    server.Get("/other.bin", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(other, "application/octet-stream");
    });
    if (!server.bind_to_port("127.0.0.1", port))
    {
        std::printf("the port %d cannot be opened\n", port);
        return 2;
    }
    std::thread listener([&]() { server.listen_after_bind(); });
    const std::string base_url = "http://127.0.0.1:" + std::to_string(port);

    {
        albireo::store db(albireo::join_path(work, "albireo.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-10-30T00:00:00Z";
        json list = json::array();
        list.push_back(catalog_entry("tiny", served, base_url + "/model.bin"));
        list.push_back(catalog_entry("small", other, base_url + "/other.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        albireo::model_manager m(db, models_dir, "");
        const std::string dir = m.models_dir();
        const std::string partial = albireo::join_path(dir, "tiny.gguf.partial");
        const std::string target = albireo::join_path(dir, "tiny.gguf");
        albireo::user_error error;

#if !defined(_WIN32)
        // The partial file a link to a file outside: refused, the file outside unchanged.
        const std::string victim = albireo::join_path(outside, "victim.txt");
        write_text(victim, "kept");
        make_symlink(victim, partial);
        check(m.start_download("tiny", error), "a download starts");
        json d = wait_download(m, "tiny");
        check(d["state"].as_string() == "error" && d["code"].as_string() == "download_file",
            "a partial file that is a link is refused (" + d["state"].as_string() + ")");
        check(read_text(victim) == "kept", "the file the link leads to is unchanged");
        albireo::remove_file(partial);

        // The partial file a second name of a file outside: refused, the file unchanged.
        make_hard_link(victim, partial);
        check(m.start_download("tiny", error), "a download starts again");
        d = wait_download(m, "tiny");
        check(d["state"].as_string() == "error" && d["code"].as_string() == "download_file",
            "a partial file with a second name is refused");
        check(read_text(victim) == "kept", "the file of the second name is unchanged");
        albireo::remove_file(partial);

        // The final name a link: the download replaces the link, never writes through it.
        make_symlink(victim, target);
        check(m.start_download("tiny", error), "a download starts over a final name that is a link");
        d = wait_download(m, "tiny");
        albireo::installed_model im;
        check(d["state"].as_string() == "done" && !albireo::is_link(target) && read_text(target) == served,
            "the model replaces the link at its final name (" + d["state"].as_string() + ")");
        check(read_text(victim) == "kept", "the file the final link led to is unchanged");
        check(installed(db, "tiny", im) && im.file == target, "the model is recorded at its name");
        check(!albireo::file_exists(partial), "no partial file is left");

        // The models folder replaced by a link: the download writes nothing where it leads.
        const std::string moved = dir + ".moved";
        const std::string elsewhere = albireo::join_path(outside, "elsewhere");
        albireo::make_dirs(elsewhere);
        albireo::rename_file(dir, moved);
        make_symlink(elsewhere, dir);
        check(m.start_download("small", error), "a download starts into a folder replaced by a link");
        d = wait_download(m, "small");
        check(d["state"].as_string() == "error", "the download into a folder replaced by a link fails");
        check(albireo::list_dir(elsewhere).empty(), "nothing is written where the link leads");
        albireo::remove_file(dir);
        albireo::rename_file(moved, dir);
#else
        check(m.start_download("tiny", error), "a download starts");
        json d = wait_download(m, "tiny");
        check(d["state"].as_string() == "done", "the download completes");
        std::printf("skip the links, which this system lets only some accounts make\n");
#endif

        // A cancelled download leaves no partial file and comes back whole afterwards.
        albireo::installed_model small;
        check(m.start_download("small", error), "a download of small starts");
        m.cancel_download("small");
        d = wait_download(m, "small");
        check(d["state"].as_string() == "cancelled" || d["state"].as_string() == "done",
            "the cancelled download ends (" + d["state"].as_string() + ")");
        if (d["state"].as_string() == "cancelled")
            check(!albireo::file_exists(albireo::join_path(dir, "small.gguf.partial")), "its partial file is removed");
        check(m.start_download("small", error), "the download starts again");
        d = wait_download(m, "small");
        check(d["state"].as_string() == "done" && installed(db, "small", small), "the new download installs it");
        m.stop();
    }

    {
        // The models that come with Albireo, and the imports of the same content.
        albireo::store db(albireo::join_path(work, "albireo-2.db"));
        const std::string builtin = albireo::join_path(work, "builtin");
        albireo::make_dirs(builtin);
        const std::string own = blob(150000, 'q');
        write_text(albireo::join_path(builtin, "base.gguf"), own);
        json entry = json::object();
        entry["id"] = "base-albireo";
        entry["name"] = "Base";
        entry["file"] = "base.gguf";
        entry["sha256"] = albireo::sha256_hex(own);
        json listed = json::array();
        listed.push_back(entry);
        json manifest = json::object();
        manifest["models"] = listed;
        write_text(albireo::join_path(builtin, albireo::builtin_manifest_name), manifest.dump());
        // The catalog in use knows another content under the identifier of the model that comes with Albireo.
        const std::string newer = blob(160000, 'z');
        json catalog = json::object();
        catalog["updated"] = "2026-10-30T00:00:00Z";
        json list = json::array();
        list.push_back(catalog_entry("base-albireo", newer, base_url + "/none.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        const std::string models2 = albireo::join_path(work, "models-2");
        albireo::model_manager m(db, models2, "");
        std::vector<std::string> removed;
        m.register_builtin(builtin, removed);

        // The same content in the import folder: not copied, not recorded again, removed from the folder.
        const std::string copy = albireo::join_path(inbox, "base-albireo.gguf");
        write_text(copy, own);
        const json r = m.import_path(copy, inbox, true);
        albireo::installed_model im;
        check(r["imported"].size() == 0 && r["already"].size() == 1 && r["already"][0].as_string() == "base-albireo",
            "a copy of a model that comes with Albireo is installed already");
        check(albireo::list_dir(m.models_dir()).empty(), "no second copy enters the models folder");
        check(installed(db, "base-albireo", im) && im.source == "builtin", "the model stays the one of Albireo");
        check(!albireo::file_exists(copy), "the copy leaves the import folder");

        // Another content under its identifier: installed under its own name, never over it.
        const std::string variant = albireo::join_path(inbox, "variant.gguf");
        write_text(variant, newer);
        const json v = m.import_path(variant, inbox, false);
        albireo::installed_model renamed;
        check(v["imported"].size() == 1 && v["imported"][0].as_string() == "base-albireo-import",
            "a file with the identifier of a model of Albireo takes the suffix -import");
        check(installed(db, "base-albireo", im) && im.source == "builtin", "the model of Albireo keeps its entry");
        check(installed(db, "base-albireo-import", renamed) && renamed.source == "catalog",
            "the other file has an entry of its own");
        check(albireo::file_exists(variant), "a file imported without move stays in place");

        // A download of its identifier is refused.
        albireo::user_error error;
        check(!m.start_download("base-albireo", error) && error.code == "builtin_installed",
            "a download of a model that comes with Albireo is refused");

        // The import folder reads through no link and a single name.
#if !defined(_WIN32)
        const std::string secret = albireo::join_path(outside, "secret.gguf");
        write_text(secret, blob(1000, 's'));
        const std::string linked = albireo::join_path(inbox, "linked.gguf");
        make_symlink(secret, linked);
        const json l = m.import_path(linked, inbox, true);
        check(l["imported"].size() == 0 && l["errors"].size() == 1, "a link in the import folder is refused");
        const std::string hard = albireo::join_path(inbox, "hard.gguf");
        make_hard_link(secret, hard);
        const json h = m.import_path(hard, inbox, true);
        check(h["imported"].size() == 0 && h["errors"].size() == 1, "a second name in the import folder is refused");
        check(albireo::file_exists(secret), "the file they lead to stays");
        albireo::remove_file(linked);
        albireo::remove_file(hard);
        // A folder of the import folder replaced by a link: nothing read through it.
        const std::string sub = albireo::join_path(inbox, "models");
        make_symlink(outside, sub);
        const json s = m.import_path(inbox, inbox, false);
        check(s["imported"].size() == 0, "the models subfolder that is a link is not read");
        albireo::remove_file(sub);
        std::string why;
        albireo::link_free_file f;
        check(!f.open(albireo::join_path(albireo::join_path(work, "via"), "secret.gguf"), work, why)
            && (why == "missing" || why == "link"), "a missing entry is reported");
        make_symlink(outside, albireo::join_path(work, "via"));
        check(!f.open(albireo::join_path(albireo::join_path(work, "via"), "secret.gguf"), work, why) && why == "link",
            "a folder that is a link stops a link-free opening");
#endif

        // An offline bundle: only the files of its catalog, with their sizes and hashes.
        const std::string bundle = albireo::join_path(work, "bundle");
        albireo::make_dirs(albireo::join_path(bundle, "models"));
        const std::string listed_content = blob(120000, 'm');
        json bundle_catalog = json::object();
        json models = json::array();
        models.push_back(catalog_entry("bundled", listed_content, "https://example.org/files/bundled-q4.gguf"));
        models.push_back(catalog_entry("damaged", blob(50000, 'd'), "https://example.org/files/damaged.gguf"));
        bundle_catalog["models"] = models;
        write_text(albireo::join_path(albireo::join_path(bundle, "models"), "bundled-q4.gguf"), listed_content);
        write_text(albireo::join_path(albireo::join_path(bundle, "models"), "damaged.gguf"), blob(50000, 'x'));
        write_text(albireo::join_path(albireo::join_path(bundle, "models"), "unlisted.gguf"), blob(1000, 'u'));
        const json b = m.import_bundle(bundle, bundle_catalog);
        check(b["imported"].size() == 1 && b["imported"][0].as_string() == "bundled", "the listed model is imported");
        check(b["errors"].size() == 1, "the file that does not match the catalog is refused");
        bool leftover = false;
        for (const auto& n : albireo::list_dir(m.models_dir()))
            leftover = leftover || n.find(".partial") != std::string::npos;
        check(!leftover, "no temporary file is left in the models folder");
        const json again = m.import_bundle(bundle, bundle_catalog);
        check(again["imported"].size() == 0 && again["already"].size() == 1, "an imported model is not copied twice");
        m.stop();
    }

    server.stop();
    listener.join();
    if (failures == 0) albireo::remove_tree(work);
    std::printf(failures ? "%d checks FAILED\n" : "all checks passed\n", failures);
    return failures ? 1 : 0;
}
