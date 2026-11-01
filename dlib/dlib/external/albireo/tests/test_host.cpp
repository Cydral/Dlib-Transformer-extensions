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

    A file never takes the name of another installed model; a newer version of an installed
    model takes its names through the function that keeps them from the engine, and a
    replacement it refuses leaves the model as it was, the next one completing from the
    files received.  The ranges of the downloads bring the bytes asked for and no more, a
    damaged database fails its opening as malformed, and the files written atomically leave
    no temporary file.  The public link answers through a local API of pCloud: its listing and
    addresses are kept, a model comes from its copy in the link when the official address
    fails, and from the official address when the copy has another size, and its errors never
    show the query of a request.  The test works in a data folder of its own (set_data_dir()), never in
    the one of an installed service.
*/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
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
#include "../host/http_fetch.h"
#include "../host/models.h"
#include "../host/public_link.h"
#include "../host/store.h"
#include "tiny_gguf.h"

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
    // A data folder of its own, on every system: the import folder of an installed service is never touched.
    const std::string data = albireo::join_path(work, "data");
    albireo::set_data_dir(data);
    check(albireo::data_dir() == data, "the test works in a data folder of its own");
    const std::string inbox = albireo::join_path(albireo::data_dir(), "import");
    albireo::make_dirs(inbox);

    // The files the local server gives, with ranges, and a file it gives whole whatever the range asked.
    const std::string served = blob(300000, 'a');
    const std::string other = blob(200000, 'k');
    const std::string newer_tiny = blob(300000, 'b');
    httplib::Server server;
    server.Get("/model.bin", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(served, "application/octet-stream");
    });
    server.Get("/other.bin", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(other, "application/octet-stream");
    });
    server.Get("/model-2.bin", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(newer_tiny, "application/octet-stream");
    });
    server.Get("/whole.bin", [&](const httplib::Request&, httplib::Response& res) {
        res.set_chunked_content_provider("application/octet-stream", [&](size_t offset, httplib::DataSink& sink) {
            if (offset >= served.size())
            {
                sink.done();
                return true;
            }
            return sink.write(served.data() + offset, std::min<size_t>(65536, served.size() - offset));
        });
    });
    /*
        The public link of the tests: the API of pCloud on the local server (showpublink,
        getpublinkdownload) and the files of its folder, dist/feed.json, the copy of
        models/linked/gone.bin, whose official address answers 404, and a copy of
        models/short/other.bin of another size than the file of the catalog.  protected_link
        makes the API answer as a link whose password is refused.
    */
    const std::string feed_copy = "{\"feed\": 1}";
    const std::string linked = blob(250000, 'l');
    const std::string short_copy = blob(1000, 'w');
    std::atomic<int> listings{0}, addresses{0}, password_mismatch{0};
    std::atomic<bool> protected_link{false};
    auto api_refusal = [&](const httplib::Request& req, httplib::Response& res) {
        if (req.has_param("linkpassword") != albireo::public_link_has_password()) ++password_mismatch;
        if (req.get_param_value("code") != "TESTLINK1")
        {
            res.set_content("{\"result\": 7001, \"error\": \"Invalid link 'code'.\"}", "application/json");
            return true;
        }
        if (protected_link)
        {
            res.set_content("{\"result\": 1125, \"error\": \"Invalid 'linkpassword' provided.\"}", "application/json");
            return true;
        }
        return false;
    };
    server.Get("/showpublink", [&](const httplib::Request& req, httplib::Response& res) {
        if (api_refusal(req, res)) return;
        ++listings;
        auto file = [](const std::string& name, long long id, size_t size) {
            json f = json::object();
            f["name"] = name;
            f["isfolder"] = false;
            f["fileid"] = id;
            f["size"] = static_cast<unsigned long long>(size);
            return f;
        };
        auto folder = [](const std::string& name, const json& contents) {
            json f = json::object();
            f["name"] = name;
            f["isfolder"] = true;
            f["contents"] = contents;
            return f;
        };
        json dist = json::array(), models = json::array(), in_linked = json::array(), in_short = json::array();
        dist.push_back(file("feed.json", 11, feed_copy.size()));
        in_linked.push_back(file("gone.bin", 21, linked.size()));
        in_short.push_back(file("other.bin", 31, short_copy.size()));
        models.push_back(folder("linked", in_linked));
        models.push_back(folder("short", in_short));
        json root_contents = json::array();
        root_contents.push_back(folder("dist", dist));
        root_contents.push_back(folder("models", models));
        json answer = json::object();
        answer["result"] = 0;
        answer["metadata"] = folder("Albireo", root_contents);
        res.set_content(answer.dump(), "application/json");
    });
    server.Get("/getpublinkdownload", [&](const httplib::Request& req, httplib::Response& res) {
        if (api_refusal(req, res)) return;
        ++addresses;
        const std::string id = req.get_param_value("fileid");
        const std::string path = id == "11" ? "/content/feed.json" : id == "21" ? "/content/gone.bin"
            : id == "31" ? "/content/other.bin" : "";
        json answer = json::object();
        if (path.empty())
        {
            answer["result"] = 2009;
            answer["error"] = "File not found.";
        }
        else
        {
            json hosts = json::array();
            hosts.push_back("127.0.0.1:" + std::to_string(port));
            answer["result"] = 0;
            answer["hosts"] = hosts;
            answer["path"] = path;
        }
        res.set_content(answer.dump(), "application/json");
    });
    server.Get("/content/feed.json", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(feed_copy, "application/octet-stream");
    });
    server.Get("/content/gone.bin", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(linked, "application/octet-stream");
    });
    server.Get("/content/other.bin", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(short_copy, "application/octet-stream");
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

        // An unknown file named after an installed model: installed under a name of its own, the model unchanged.
        const albireo_test::tiny_gguf tiny_model = albireo_test::tiny_gguf_writer().build();
        const std::string unknown = albireo::join_path(inbox, "tiny.gguf");
        tiny_model.save(unknown, tiny_model.bytes);
        const json u = m.import_path(unknown, inbox, false);
        albireo::installed_model kept;
        check(u["imported"].size() == 1 && u["imported"][0].as_string() == "tiny-import",
            "an unknown file named after an installed model takes the suffix -import (" + u.dump() + ")");
        check(read_text(target) == served && installed(db, "tiny", kept) && kept.sha256 == albireo::sha256_hex(served)
            && kept.source == "catalog", "the installed model keeps its file and its record");
        // Another unknown file of that name takes the next free name.
        std::vector<unsigned char> changed = tiny_model.bytes;
        changed.back() = static_cast<unsigned char>(changed.back() ^ 0x5a);
        tiny_model.save(unknown, changed);
        const json u2 = m.import_path(unknown, inbox, false);
        check(u2["imported"].size() == 1 && u2["imported"][0].as_string() == "tiny-import-2",
            "a second unknown file of that name takes the next free name (" + u2.dump() + ")");
        albireo::remove_file(unknown);
        m.stop();
    }

    {
        /*
            A newer version of the installed model tiny: its file takes its name through the
            function of on_replace(), which keeps the files from the engine.  A replacement it
            refuses leaves the model as it was; the next download completes from the files
            received, without a new transfer.
        */
        albireo::store db(albireo::join_path(work, "albireo.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-10-31T00:00:00Z";
        json list = json::array();
        list.push_back(catalog_entry("tiny", newer_tiny, base_url + "/model-2.bin"));
        list.push_back(catalog_entry("small", other, base_url + "/other.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        albireo::model_manager m(db, models_dir, "");
        const std::string target = albireo::join_path(m.models_dir(), "tiny.gguf");
        std::atomic<bool> allow{false};
        std::atomic<int> asked{0};
        m.on_replace([&](const std::string& id, const std::function<bool()>& place) {
            if (id == "tiny") ++asked;
            return allow && place();
        });
        albireo::user_error error;
        check(m.start_download("tiny", error), "the download of a newer version of an installed model starts");
        json d = wait_download(m, "tiny");
        check(d["state"].as_string() == "error" && d["code"].as_string() == "download_install" && asked == 1,
            "a replacement refused ends the download in error (" + d.dump() + ")");
        albireo::installed_model im;
        check(read_text(target) == served && installed(db, "tiny", im) && im.sha256 == albireo::sha256_hex(served),
            "the model keeps its file and its record when its replacement is refused");
        allow = true;
        check(m.start_download("tiny", error), "the download starts again");
        d = wait_download(m, "tiny");
        check(d["state"].as_string() == "done" && asked == 2, "the next replacement completes it (" + d.dump() + ")");
        check(read_text(target) == newer_tiny && installed(db, "tiny", im)
            && im.sha256 == albireo::sha256_hex(newer_tiny), "the newer version takes the name and the record of tiny");
        check(!albireo::file_exists(target + ".partial"), "no partial file is left");
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
        // The file of the catalog the build made it from.
        const std::string source = blob(140000, 'o');
        entry["source_sha256"] = albireo::sha256_hex(source);
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
        list.push_back(catalog_entry("base", source, base_url + "/source.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        const std::string models2 = albireo::join_path(work, "models-2");
        albireo::model_manager m(db, models2, "");
        std::vector<albireo::installed_model> removed;
        m.register_builtin(builtin, removed);

        // The entry of the file the model of Albireo was made from: the model of Albireo stands for it.
        const json view = m.overview("en");
        bool listed_source = false, listed_builtin = false;
        for (const auto& x : view["models"].elements())
        {
            listed_source = listed_source || x["id"].as_string() == "base";
            listed_builtin = listed_builtin || x["id"].as_string() == "base-albireo";
        }
        check(!listed_source && listed_builtin, "the catalog entry a model of Albireo stands for is not listed");
        albireo::user_error carried;
        check(!m.start_download("base", carried) && carried.code == "builtin_installed",
            "the download of the file a model of Albireo carries is refused");

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

    {
        // The ranges of the downloads bring the bytes asked for and no more.
        std::string got, error;
        auto keep = [&](const char* data, size_t n) {
            got.append(data, n);
            return true;
        };
        check(albireo::http_get_range(base_url + "/model.bin", 1000, 1999, keep, error)
            && got == served.substr(1000, 1000), "a part of a file arrives as asked (" + error + ")");
        got.clear();
        check(albireo::http_get_range(base_url + "/whole.bin", 0, 999, keep, error) && got == served.substr(0, 1000),
            "a server that ignores the range of a request from 0 gives the bytes asked for alone (" + error + ")");
        got.clear();
        check(!albireo::http_get_range(base_url + "/whole.bin", 1000, 1999, keep, error) && got.empty(),
            "a server that ignores the range of a request from elsewhere is refused");
    }

    {
        // The public link: the links it accepts, its listing and addresses kept, the files it lacks.
        check(albireo::valid_public_link("https://u.pcloud.link/publink/show?code=kZeNhSJZpcMygkcCvGXmq1MTvkCfBuSsx2GV")
            && albireo::valid_public_link("https://e.pcloud.link/publink/show?code=XZabcdef12")
            && !albireo::valid_public_link("https://example.org/publink/show?code=kZeNhSJZpcMygkcC")
            && !albireo::valid_public_link("http://u.pcloud.link/publink/show?code=kZeNhSJZpcMygkcC")
            && !albireo::valid_public_link("https://u.pcloud.link/publink/show?code=kZeN%2FhSJZpcM"),
            "the public links of pCloud are told from the other addresses");
        albireo::set_public_link("");
        check(!albireo::public_link_enabled(), "an empty link turns the reads of the link off");
        albireo::set_public_link(base_url + "/publink/show?code=TESTLINK1");
        check(albireo::public_link_enabled(), "a local link is used");
        std::string body, error;
        check(albireo::public_link_get("dist/feed.json", body, error, 1u << 20) && body == feed_copy,
            "a file of the link is read (" + error + ")");
        check(albireo::public_link_get("dist/feed.json", body, error, 1u << 20) && listings == 1 && addresses == 1,
            "the listing and the address of the link are kept for the next reads");
        std::string url;
        uint64_t size = 0;
        check(!albireo::public_link_file("dist/catalog.json", url, size, error)
            && error.find("holds no dist/catalog.json") != std::string::npos, "a file the link lacks is reported");
        check(password_mismatch == 0, "the password goes with each request when the program carries one");
        check(albireo::public_link_model_path("tiny", "https://hub.example/a/resolve/abc/Tiny-Q4.gguf?download=1")
            == "models/tiny/Tiny-Q4.gguf", "the copy of a file of a model lies in the folder of its identifier");
        check(albireo::without_queries("https://api.pcloud.com/showpublink?code=X&linkpassword=Y: status 500")
            == "https://api.pcloud.com/showpublink: status 500",
            "the queries of the addresses stay out of the messages");

        // The downloads: the copy of the link, and the official address for a copy of another size.
        albireo::store db(albireo::join_path(work, "albireo-3.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-11-01T00:00:00Z";
        json list = json::array();
        list.push_back(catalog_entry("linked", linked, base_url + "/gone.bin"));
        list.push_back(catalog_entry("short", other, base_url + "/other.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        albireo::model_manager m(db, albireo::join_path(work, "models-3"), "");
        albireo::user_error started;
        albireo::installed_model im;
        check(m.start_download("linked", started), "the download of a model the link holds starts");
        json d = wait_download(m, "linked");
        check(d["state"].as_string() == "done" && installed(db, "linked", im) && read_text(im.file) == linked,
            "a model whose official address fails comes from the copy of the link (" + d.dump() + ")");
        check(m.start_download("short", started), "the download of a model whose copy has another size starts");
        d = wait_download(m, "short");
        check(d["state"].as_string() == "done" && installed(db, "short", im) && read_text(im.file) == other,
            "a copy of another size gives way to the official address (" + d.dump() + ")");
        m.stop();

        // A link that refuses the password: the message names the answer, never the query.
        protected_link = true;
        albireo::set_public_link(base_url + "/publink/show?code=TESTLINK1");
        check(!albireo::public_link_get("dist/feed.json", body, error, 1u << 20)
            && error.find("1125") != std::string::npos && error.find("linkpassword=") == std::string::npos
            && error.find("code=") == std::string::npos, "a refused password is reported without the query (" + error
            + ")");
        albireo::set_public_link("");
    }

    {
        // A database damaged beyond its header fails its opening as malformed.
        const std::string path = albireo::join_path(work, "damaged.db");
        {
            albireo::store db(path);
            for (int i = 0; i < 200; ++i) db.set_value("key-" + std::to_string(i), blob(2000, static_cast<char>(i)));
        }
        for (const char* part : { "-wal", "-shm" }) albireo::remove_file(path + part);
        std::string bytes = read_text(path);
        check(bytes.size() > 8 * 4096, "the database spans several pages");
        // Its first page, the header and the schema, stays; the pages of its tables are erased.
        for (size_t i = 4096; i < bytes.size(); ++i) bytes[i] = 0;
        write_text(path, bytes);
        bool malformed = false;
        try
        {
            albireo::store damaged(path);
        }
        catch (const std::exception& e)
        {
            malformed = std::string(e.what()).find("malformed") != std::string::npos;
        }
        check(malformed, "a damaged database fails its opening as malformed");
    }

    {
        // A file written atomically replaces the one in place and leaves no temporary file.
        const std::string file = albireo::join_path(work, "atomic.txt");
        check(albireo::write_file_atomic(file, "first") && albireo::write_file_atomic(file, "second")
            && read_text(file) == "second", "a file written atomically replaces its content");
        bool temporary = false;
        for (const auto& n : albireo::list_dir(work)) temporary = temporary || n.find("atomic.txt.tmp-") == 0;
        check(!temporary, "no temporary file is left");
    }

    server.stop();
    listener.join();
    if (failures == 0) albireo::remove_tree(work);
    std::printf(failures ? "%d checks FAILED\n" : "all checks passed\n", failures);
    return failures ? 1 : 0;
}
