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
    no temporary file.  The files of the models come from their official addresses alone, which
    lie on the local server under /official/: the address of a message loses its query, a file of
    another size or whose bytes fail its hash, a server that answers a web page, a status of
    failure or nothing fail the download with the address and its failure, the progress kept, so
    that a model of eight parts, one of them compressed, resumes with the part it lacked, and a
    cancellation ends a download at once.  An address that answers twice that it no longer publishes
    its file (404, 410 or 451, or 401 or 403 with the error headers of the hub, by itself or after a
    redirection of its own server, and without a web page) ends the download with the code
    download_unavailable, its progress and its partial files dropped and its address kept for seven
    days; the same statuses from a web page, from another server a redirection led to, and 401 or
    403 without these headers stay failures of the transfer.  The catalog and the feed of the
    updates come from the folder of the versions alone,
    which the tests serve under /dist/ (distribution_for_tests()): the signed catalog and the
    signed feed of the repository are read from it, their signatures checked, and a file the
    folder lacks, a signature that fails, a web page with 200 and with 403, a proxy asking for its
    authentication, a connection reset or refused, and a server that stalls or sends its answer
    byte by byte beyond the bounds the tests shorten each fail with the address and the failure,
    let no exception out and leave the local server answering.  The installer of an update is
    written through a new partial file and takes its name once it has the size and the hash the
    feed names.  The settings of an earlier version (public_link, the providers and the key of
    the search) are read without error and leave settings.json at the start.  The test works in a
    data folder of its own (set_data_dir()), never in the one of an installed service.

    The web access reads public addresses only: the addresses on both sides of each range that
    is not public, the addresses refused before any connection (user information, schemes, names
    of a single label, the names and the numeric forms of the loopback), then, the loopback
    allowed for the tests, the pages of the local server: a page, a text cut at 2 MB, a type
    refused, the PDF files above 20 MB refused, a redirection to a private address refused.  The
    search of Wikipedia answers from the same server with its documents.  Both routes refuse to
    work offline or with the web access turned off, and a search without provider is refused.
    The relay keeps its secret in relay.secret, never in settings.json, for the provider, the
    address and the user it was given for; it restores the substitutes of an answer cut at any
    place, at the boundaries of words, its memory of the pseudonymization serves a text again
    without Pseudalis and forgets the oldest and the idle values, and its requests reach a local
    server that answers as an engine of the interface of OpenAI, as Ollama and as Claude, and as
    Pseudalis: the texts and the descriptions of the images leave pseudonymized, the answers come
    back with their values, and a refusal of the key names neither the key nor the address.
*/

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <zlib.h>

#include <dlib/external/httplib/httplib.h>

#include "../common/builtin_models.h"
#include "../common/engine_api.h"
#include "../common/log.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "../host/crypto.h"
#include "../host/distribution.h"
#include "../host/http_fetch.h"
#include "../host/models.h"
#include "../host/pseudonyms.h"
#include "../host/relay.h"
#include "../host/settings.h"
#include "../host/store.h"
#include "../host/updater.h"
#include "../host/web_access.h"
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

    // The bytes of content compressed by zlib, as the address of a compressed part gives them.
    std::string deflated(const std::string& content)
    {
        uLongf size = compressBound(static_cast<uLong>(content.size()));
        std::string out(size, '\0');
        if (compress2(reinterpret_cast<Bytef*>(&out[0]), &size, reinterpret_cast<const Bytef*>(content.data()),
                static_cast<uLong>(content.size()), 6) != Z_OK)
            return std::string();
        out.resize(size);
        return out;
    }

    /*
        A catalog entry of several parts, each at its address in urls: contents gives the bytes of each part,
        and packed, when it is not empty for a part, the compressed bytes its address gives ("compression":
        "gzip").
    */
    json parts_entry(const std::string& id, const std::vector<std::string>& contents,
        const std::vector<std::string>& urls, const std::vector<std::string>& packed)
    {
        json files = json::array();
        std::string whole;
        for (size_t i = 0; i < contents.size(); ++i)
        {
            json f = json::object();
            f["url"] = urls[i];
            f["size"] = static_cast<unsigned long long>(contents[i].size());
            f["sha256"] = albireo::sha256_hex(contents[i]);
            if (!packed[i].empty())
            {
                f["compression"] = "gzip";
                f["compressed_size"] = static_cast<unsigned long long>(packed[i].size());
            }
            files.push_back(f);
            whole += contents[i];
        }
        json e = catalog_entry(id, whole, urls[0]);
        e["files"] = files;
        return e;
    }

    /*
        A file of the sources the test is built from, found from the path of this source and empty when the
        sources are not there: below names its folder from the folder of the tests.
    */
    std::string source_file(const std::string& below, const std::string& name)
    {
        const std::string source = __FILE__;
        const size_t slash = source.find_last_of("/\\");
        std::string data;
        if (slash != std::string::npos)
            albireo::read_file(source.substr(0, slash) + "/" + below + "/" + name, data, 16u << 20);
        return data;
    }

    // A file of dist/albireo in the repository, which keeps the signed catalog.
    std::string repository_file(const std::string& name)
    {
        return source_file("../../../../../dist/albireo", name);
    }

    // A file of tests/data: the signed feed of version 1.0.24, which the tests of the updates serve.
    std::string data_file(const std::string& name)
    {
        return source_file("data", name);
    }

    // The number of lines of the recent log that hold text.
    int logged(const std::string& text)
    {
        int n = 0;
        for (const auto& line : albireo::log_file::instance().recent())
            if (line.find(text) != std::string::npos) ++n;
        return n;
    }

    // Waits for the download of id to leave its running states; returns its last state.
    json wait_download(albireo::model_manager& m, const std::string& id)
    {
        for (int i = 0; i < 1200; ++i)
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

    /*
        A server that accepts each connection, reads its request, and resets it (SO_LINGER at
        0), as a filter of the network that cuts the connections to a site.
    */
    class reset_server
    {
    public:
        ~reset_server() { stop(); }

        bool start(int port)
        {
            listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
            if (listener_ == INVALID_SOCKET) return false;
            int yes = 1;
            setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
            sockaddr_in a = {};
            a.sin_family = AF_INET;
            a.sin_port = htons(static_cast<unsigned short>(port));
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            if (::bind(listener_, reinterpret_cast<const sockaddr*>(&a), sizeof(a)) != 0 || ::listen(listener_, 8) != 0)
            {
                httplib::detail::close_socket(listener_);
                listener_ = INVALID_SOCKET;
                return false;
            }
            thread_ = std::thread([this]() {
                while (!stopped_)
                {
                    if (!readable(listener_)) continue;
                    const socket_t c = ::accept(listener_, nullptr, nullptr);
                    if (c == INVALID_SOCKET) continue;
                    char request[4096];
                    if (readable(c)) ::recv(c, request, sizeof(request), 0);
                    linger reset;
                    reset.l_onoff = 1;
                    reset.l_linger = 0;
                    setsockopt(c, SOL_SOCKET, SO_LINGER, reinterpret_cast<const char*>(&reset), sizeof(reset));
                    httplib::detail::close_socket(c);
                }
            });
            return true;
        }

        void stop()
        {
            stopped_ = true;
            if (thread_.joinable()) thread_.join();
            if (listener_ != INVALID_SOCKET) httplib::detail::close_socket(listener_);
            listener_ = INVALID_SOCKET;
        }

    private:
        // Whether a socket has data or a connection to give within a tenth of a second.
        static bool readable(socket_t sock)
        {
            fd_set set;
            FD_ZERO(&set);
            FD_SET(sock, &set);
            timeval wait = { 0, 100000 };
            return select(static_cast<int>(sock + 1), &set, nullptr, nullptr, &wait) > 0;
        }

        socket_t listener_ = INVALID_SOCKET;
        std::atomic<bool> stopped_{false};
        std::thread thread_;
    };
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
        The contents of the files the local server gives, the signed catalog of the repository, the signed
        feed of the tests, and the faults a route answers with, as a proxy, a filter of the network or a
        damaged server would.
    */
    const std::string linked = blob(250000, 'l');
    const std::string short_copy = blob(1000, 'w');
    const std::string first_part = blob(120000, 'f'), second_part = blob(80000, 'g');
    const std::string stale_copy = blob(150000, 's'), fresh = blob(150000, 'r');
    const std::string repository_catalog = repository_file("catalog.json");
    const std::string repository_signature = repository_file("catalog.json.sig");
    const std::string signed_feed = data_file("feed.json");
    const std::string signed_feed_signature = data_file("feed.json.sig");
    // A signature of the form of the others that signs nothing of the tests.
    const std::string other_signature = "MEQCIAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gAiABAgMEBQYHCAkKCwwNDg8QERIT"
        "FBUWFxgZGhscHR4fIA==";
    enum fault { no_fault, html_200, html_403, html_404, proxy_407, stall, trickle };
    // The answer of a fault; false without one.
    auto fault_answer = [](int f, httplib::Response& res) {
        const std::string blocked = "<html><body><h1>Access denied</h1><p>The filter of this network blocks the "
            "site.</p></body></html>";
        switch (f)
        {
        case html_200:
            res.set_content(blocked, "text/html; charset=utf-8");
            return true;
        case html_403:
            res.status = 403;
            res.set_content(blocked, "text/html");
            return true;
        case html_404:
            res.status = 404;
            res.set_content("<html><body><h1>Page not found</h1></body></html>", "text/html; charset=utf-8");
            return true;
        case proxy_407:
            res.status = 407;
            res.set_header("Proxy-Authenticate", "Basic realm=\"filter\"");
            res.set_content(blocked, "text/html");
            return true;
        case stall:
            std::this_thread::sleep_for(std::chrono::seconds(3));
            res.set_content("{}", "application/json");
            return true;
        case trickle:
            // A byte every 0.3 s, below the bound of each wait and beyond the bound of the whole request.
            res.set_chunked_content_provider("application/json", [](size_t offset, httplib::DataSink& sink) {
                if (offset >= 20)
                {
                    sink.done();
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                return sink.write(" ", 1);
            });
            return true;
        default:
            return false;
        }
    };
    server.Get("/alive", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("alive", "text/plain");
    });
    /*
        A set of files the local server gives under a folder, with their ranges: those of files, the names that
        begin with "slow" a kilobyte every 20 ms, those of statuses answered with their status and a line of
        text, as the hub of the models answers for a file it no longer publishes, the others with 404.  The
        refusals carry the error headers of the hub while hub is set.  fault makes every name answer as a
        proxy or a filter of the network would, asked counts the requests of each name, and encoding keeps
        the Accept-Encoding of the last request.
    */
    struct served_folder
    {
        std::mutex m;
        std::map<std::string, std::string> files;
        std::map<std::string, int> statuses;
        std::map<std::string, int> asked;
        std::string encoding;
        std::atomic<int> fault{no_fault};
        std::atomic<int> requests{0};
        std::atomic<bool> hub{true};

        void serve(const std::string& name, const std::string& content)
        {
            std::lock_guard<std::mutex> lock(m);
            files[name] = content;
            statuses.erase(name);
        }
        void refuse(const std::string& name, int status)
        {
            std::lock_guard<std::mutex> lock(m);
            statuses[name] = status;
        }
        void remove(const std::string& name)
        {
            std::lock_guard<std::mutex> lock(m);
            files.erase(name);
            statuses.erase(name);
        }
        int count(const std::string& name)
        {
            std::lock_guard<std::mutex> lock(m);
            const auto it = asked.find(name);
            return it == asked.end() ? 0 : it->second;
        }
        std::string last_encoding()
        {
            std::lock_guard<std::mutex> lock(m);
            return encoding;
        }
    };
    auto serve_folder = [&](const std::string& pattern, served_folder& folder) {
        server.Get(pattern, [&folder, fault_answer](const httplib::Request& req, httplib::Response& res) {
            ++folder.requests;
            const std::string name = req.matches[1].str();
            auto content = std::make_shared<std::string>();
            bool found = false;
            int status = 0;
            {
                std::lock_guard<std::mutex> lock(folder.m);
                ++folder.asked[name];
                folder.encoding = req.get_header_value("Accept-Encoding");
                const auto refused = folder.statuses.find(name);
                if (refused != folder.statuses.end()) status = refused->second;
                const auto it = folder.files.find(name);
                found = it != folder.files.end();
                if (found) *content = it->second;
            }
            if (fault_answer(folder.fault, res)) return;
            if (status != 0 || !found)
            {
                res.status = status != 0 ? status : 404;
                const char* const text = status == 401 ? "Invalid username or password." : "Entry not found";
                if (folder.hub)
                {
                    res.set_header("X-Error-Message", text);
                    if (res.status == 404) res.set_header("X-Error-Code", "EntryNotFound");
                }
                res.set_content(text, "text/plain");
                return;
            }
            if (name.compare(0, 4, "slow") != 0)
            {
                res.set_content(*content, "application/octet-stream");
                return;
            }
            res.set_content_provider(content->size(), "application/octet-stream",
                [content](size_t offset, size_t length, httplib::DataSink& sink) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    return sink.write(content->data() + offset, std::min<size_t>(length, 1024));
                });
        });
    };
    // The official addresses of the catalogs of the tests, under /official/.
    served_folder official_files;
    serve_folder(R"(/official/(.*))", official_files);
    // The folder of the versions, which the website serves to the programs, under /dist/.
    served_folder dist_files;
    serve_folder(R"(/dist/(.*))", dist_files);
    /*
        An official address that sends to a server of contents, another server, on the port after the two
        the failures of the network take: it answers as the hub does for a file it lacks, which tells nothing
        of the address asked.  An address that sends to another address of its own server, as the hub does
        for a repository that took another name.
    */
    const int contents_port = port + 3;
    server.Get("/moved/denied.bin", [&](const httplib::Request&, httplib::Response& res) {
        res.set_redirect("http://127.0.0.1:" + std::to_string(contents_port) + "/contents/denied.bin");
    });
    std::atomic<int> contents_requests{0};
    httplib::Server contents_server;
    contents_server.Get("/contents/denied.bin", [&](const httplib::Request&, httplib::Response& res) {
        ++contents_requests;
        res.status = 404;
        res.set_header("X-Error-Code", "EntryNotFound");
        res.set_header("X-Error-Message", "Entry not found");
        res.set_content("Entry not found", "text/plain");
    });
    server.Get(R"(/renamed/(.*))", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_redirect("/official/" + req.matches[1].str(), 307);
    });
    /*
        The pages of the web access: a page, a text of 3 MB, an image, a small PDF file, two PDF
        files of 21 MB (one announced by its length, one sent in chunks), a page that answers
        404, redirections to a private address, to a relative address and to themselves, and a
        page that sends back the headers it received.  The search answers as the interface of
        Wikipedia, and keeps what it received, which the checks read under seen_m.
    */
    const std::string page = "<html><body><p>Albireo &amp; the web</p></body></html>";
    const std::string large_text = blob(3u << 20, 't');
    const size_t large_pdf = size_t(21) << 20;
    server.Get("/web/page.html", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(page, "text/html; charset=UTF-8");
    });
    server.Get("/web/large.txt", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(large_text, "text/plain");
    });
    server.Get("/web/image.png", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string("\x89PNG\r\n\x1a\n", 8), "image/png");
    });
    server.Get("/web/small.pdf", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content("%PDF-1.4\n%%EOF\n", "application/pdf");
    });
    server.Get("/web/large.pdf", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content_provider(large_pdf, "application/pdf", [](size_t, size_t length, httplib::DataSink& sink) {
            const std::string chunk(65536, '%');
            return sink.write(chunk.data(), std::min(length, chunk.size()));
        });
    });
    server.Get("/web/stream.pdf", [&](const httplib::Request&, httplib::Response& res) {
        res.set_chunked_content_provider("application/pdf", [&](size_t offset, httplib::DataSink& sink) {
            if (offset >= large_pdf)
            {
                sink.done();
                return true;
            }
            const std::string chunk(65536, '%');
            return sink.write(chunk.data(), chunk.size());
        });
    });
    server.Get("/web/missing", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 404;
        res.set_content("<p>no such page</p>", "text/html");
    });
    server.Get("/web/to-private", [&](const httplib::Request&, httplib::Response& res) {
        res.set_redirect("http://192.168.1.10/admin");
    });
    server.Get("/web/relative", [&](const httplib::Request&, httplib::Response& res) {
        res.set_redirect("./sub/../page.html?from=relative");
    });
    server.Get("/web/loop", [&](const httplib::Request&, httplib::Response& res) {
        res.set_redirect("/web/loop");
    });
    server.Get("/web/headers", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_content(req.get_header_value("User-Agent") + "\n" + req.get_header_value("Accept-Language") + "\n"
            + req.get_header_value("Cookie"), "text/plain; charset=\"utf-8\"");
    });
    std::mutex seen_m;
    std::string seen_agent, seen_query, seen_count;
    server.Get("/w/rest.php/v1/search/page", [&](const httplib::Request& req, httplib::Response& res) {
        {
            std::lock_guard<std::mutex> lock(seen_m);
            seen_agent = req.get_header_value("User-Agent");
            seen_query = req.get_param_value("q");
            seen_count = req.get_param_value("limit");
        }
        if (req.get_param_value("q") == "too many")
        {
            res.status = 429;
            res.set_header("Retry-After", "30");
            res.set_content("You are making too many requests to the API.", "text/plain");
            return;
        }
        res.set_content("{\"pages\": [{\"id\": 736, \"key\": \"Albert_Einstein\", \"title\": \"Albert Einstein\", "
            "\"excerpt\": \"<span class=\\\"searchmatch\\\">Albert</span> Einstein was a &quot;physicist&quot;\", "
            "\"matched_title\": null, \"description\": \"German-born physicist\"}, {\"id\": 9411, \"key\": "
            "\"Einstein_(crater)\", \"title\": \"Einstein (crater)\", \"excerpt\": \"\", \"description\": "
            "\"Lunar crater\"}]}", "application/json");
    });
    if (!server.bind_to_port("127.0.0.1", port))
    {
        std::printf("the port %d cannot be opened\n", port);
        return 2;
    }
    std::thread listener([&]() { server.listen_after_bind(); });
    const bool contents_listening = contents_server.bind_to_port("127.0.0.1", contents_port);
    std::thread contents_listener([&]() { if (contents_listening) contents_server.listen_after_bind(); });
    check(contents_listening, "the server of the contents listens on the port " + std::to_string(contents_port));
    const std::string base_url = "http://127.0.0.1:" + std::to_string(port);
    const std::string page_text = "a web page comes in place of the answer, as a proxy or a filter of the network "
        "sends one";
    /*
        The folder of the versions is the one of the website; the tests read theirs on the local server, where
        the official addresses of their catalogs lie as well.
    */
    check(albireo::distribution_address("feed.json") == std::string(ALBIREO_DISTRIBUTION_URL) + "/feed.json"
        && std::string(ALBIREO_DISTRIBUTION_URL).compare(0, 8, "https://") == 0,
        "the folder of the versions is an address of the website, over https");
    const std::string dist = base_url + "/dist";
    albireo::distribution_for_tests(dist);
    check(albireo::distribution_address("catalog.json.sig") == dist + "/catalog.json.sig",
        "the tests read the folder of the versions on the local server");
    const std::string official = base_url + "/official/";
    official_files.serve("model.bin", served);
    official_files.serve("other.bin", other);
    official_files.serve("model-2.bin", newer_tiny);

    // An entry of the catalog for a later version of Albireo is left out, and stays in the database.
    {
        albireo::store db(albireo::join_path(work, "since.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-10-30T00:00:00Z";
        json list = json::array();
        list.push_back(catalog_entry("tiny", served, base_url + "/official/model.bin"));
        json current = catalog_entry("current", other, base_url + "/official/other.bin");
        current["since"] = ALBIREO_VERSION;
        list.push_back(current);
        json later = catalog_entry("later", other, base_url + "/official/other.bin");
        later["since"] = "999.0.0";
        list.push_back(later);
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        albireo::model_manager m(db, albireo::join_path(work, "models-since"));
        json entry;
        check(m.catalog()["models"].size() == 2 && m.catalog_status()["models"].as_int() == 2,
            "the catalog serves the models of this version");
        check(m.catalog_entry("tiny", entry) && m.catalog_entry("current", entry) && !m.catalog_entry("later", entry),
            "a model of a later version has no entry");
        albireo::user_error error;
        check(!m.start_download("later", error) && error.code == "not_in_catalog",
            "a model of a later version is not downloaded (" + error.code + ")");
        check(json::parse_or_null(db.value("catalog"))["models"].size() == 3,
            "the database keeps the catalog as it was signed");
    }

    {
        albireo::store db(albireo::join_path(work, "albireo.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-10-30T00:00:00Z";
        json list = json::array();
        list.push_back(catalog_entry("tiny", served, base_url + "/official/model.bin"));
        list.push_back(catalog_entry("small", other, base_url + "/official/other.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        albireo::model_manager m(db, models_dir);
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
        /*
            A hybrid decoder described from its header: its context is the window of its
            attentions, and its template, which writes the answers in channels with a reasoning
            effort, gives it the reasoning on request and the tools.
        */
        const albireo_test::tiny_gguf hybrid_model = albireo_test::tiny_gguf_writer().build_hybrid();
        const std::string hybrid_path = albireo::join_path(work, "hybrid-described.gguf");
        hybrid_model.save(hybrid_path, hybrid_model.bytes);
        const json h = albireo::describe_gguf(hybrid_path);
        check(h["arch"].as_string() == "olala" && h["context"].as_int() == 2048 && h["experts"].as_int() == 8
            && h["reasoning"].as_bool(false) && !h["reasoning_always"].as_bool(true) && h["tools"].as_bool(false)
            && h["quantization"].as_string() == "F32", "a hybrid decoder described from its header (" + h.dump() + ")");
        albireo::remove_file(hybrid_path);
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
        list.push_back(catalog_entry("tiny", newer_tiny, base_url + "/official/model-2.bin"));
        list.push_back(catalog_entry("small", other, base_url + "/official/other.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        albireo::model_manager m(db, models_dir);
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
        albireo::model_manager m(db, models2);
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
        // What answered a failed request tells whether its address withdrew the file.
        albireo::http_answer a;
        bool told = !albireo::http_withdrawn(a);
        for (int status : { 401, 403, 404, 410, 451 })
        {
            a = albireo::http_answer();
            a.status = status;
            // The server states 401 and 403 as its own with the error headers of the hub.
            const bool stated = status != 401 && status != 403;
            told = told && albireo::http_withdrawn(a) == stated;
            a.hub_error = true;
            told = told && albireo::http_withdrawn(a);
            a.page = true;
            told = told && !albireo::http_withdrawn(a);
            a.page = false;
            a.redirected = true;
            told = told && !albireo::http_withdrawn(a);
        }
        for (int status : { 200, 206, 302, 400, 407, 429, 500, 502, 503 })
        {
            a = albireo::http_answer();
            a.status = status;
            a.hub_error = true;
            told = told && !albireo::http_withdrawn(a);
        }
        check(told, "an address withdrew its file when it answers 404, 410 or 451, or 401 or 403 with the error "
            "headers of the hub, by itself and with no web page");
        check(albireo::without_queries("https://example.org/search?q=X&key=Y: status 500")
            == "https://example.org/search: status 500", "the queries of the addresses stay out of the messages");
        check(albireo::without_queries("\"https://a.example/x?k=1\" then 'https://b.example/y?k=2'\t"
            "https://c.example/z?k=3\r\nhttps://d.example/w?k=4, https://e.example/v")
            == "\"https://a.example/x\" then 'https://b.example/y'\thttps://c.example/z\r\nhttps://d.example/w, "
            "https://e.example/v",
            "each address of a message loses its query, whatever ends it: a quotation mark, a tab or an end of line");

        /*
            The downloads: each part from its official address, and from nowhere else.  A file of another size
            than the catalog gives and a file whose bytes fail its hash fail the download; an address that no
            longer publishes its file ends it at once, with a code of its own.
        */
        albireo::store db(albireo::join_path(work, "albireo-3.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-11-01T00:00:00Z";
        json list = json::array();
        const std::string absent = blob(5000, 'n');
        list.push_back(catalog_entry("linked", linked, official + "gone.bin?download=1&token=abc"));
        list.push_back(catalog_entry("short", other, official + "short.bin"));
        list.push_back(catalog_entry("absent", absent, official + "absent.bin"));
        list.push_back(parts_entry("mixed", { first_part, second_part }, { official + "first.bin",
            official + "second.bin" }, { std::string(), std::string() }));
        list.push_back(catalog_entry("stale", fresh, official + "stale.bin"));
        list.push_back(catalog_entry("moved", absent, base_url + "/moved/denied.bin"));
        list.push_back(catalog_entry("renamed", absent, base_url + "/renamed/lost.bin"));
        // A model of two parts: the first one arrives in about twenty seconds, the hub lacks the second.
        const std::string slow_part = blob(1u << 20, 'w');
        list.push_back(parts_entry("halved", { slow_part, absent }, { official + "slow-part.bin",
            official + "lost-part.bin" }, { std::string(), std::string() }));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        official_files.serve("gone.bin", linked);
        official_files.serve("short.bin", short_copy);
        official_files.serve("first.bin", first_part);
        official_files.serve("second.bin", second_part);
        official_files.serve("stale.bin", stale_copy);
        official_files.serve("slow-part.bin", slow_part);
        albireo::model_manager m(db, albireo::join_path(work, "models-3"));
        albireo::user_error started;
        albireo::installed_model im;
        auto complete = [&](const json& d, const std::string& id, const std::string& content) {
            return d["state"].as_string() == "done" && installed(db, id, im) && read_text(im.file) == content
                && im.sha256 == albireo::sha256_hex(content) && d["received"].as_int(-1) == d["total"].as_int(-2);
        };
        auto failed_with = [&](const json& d, const std::string& code, const std::string& text) {
            return d["state"].as_string() == "error" && d["code"].as_string() == code && d["error"].as_string() == text;
        };
        check(m.start_download("linked", started), "the download of a model starts");
        json d = wait_download(m, "linked");
        check(complete(d, "linked", linked) && official_files.count("gone.bin") == 1
            && official_files.last_encoding() == "identity" && logged("models: linked installed") == 1,
            "a model comes from its official address, asked as it is published (" + d.dump() + ")");
        check(m.start_download("short", started), "the download of a model whose file has another size starts");
        d = wait_download(m, "short");
        const std::string other_size = official + "short.bin cannot be downloaded: the file it gives does not have the "
            "size the catalog gives";
        check(failed_with(d, "download_failed", other_size) && d["args"]["e"].as_string() == other_size
            && official_files.count("short.bin") == 3 && !installed(db, "short", im),
            "a file of another size than the catalog gives fails the download after three attempts (" + d.dump()
            + ")");
        check(m.start_download("mixed", started), "the download of a model of two parts starts");
        d = wait_download(m, "mixed");
        check(complete(d, "mixed", first_part + second_part) && official_files.count("first.bin") == 1
            && official_files.count("second.bin") == 1, "each part of a model comes from its own address ("
            + d.dump() + ")");
        check(m.start_download("stale", started), "the download of a model whose file has other bytes starts");
        d = wait_download(m, "stale");
        check(failed_with(d, "download_hash", "the downloaded file does not match the hash of the catalog")
            && official_files.count("stale.bin") == 2
            && logged("models: part 0 of stale is corrupt, fetched again") == 1
            && !albireo::file_exists(albireo::join_path(m.models_dir(), "stale.gguf.partial"))
            && !db.download_state("stale").is_object() && !installed(db, "stale", im),
            "a file whose bytes fail the hash is fetched again once, then refused and removed (" + d.dump() + ")");
        official_files.serve("stale.bin", fresh);
        check(m.start_download("stale", started), "the download of the model starts again");
        d = wait_download(m, "stale");
        check(complete(d, "stale", fresh), "the file of the catalog at its address installs the model (" + d.dump()
            + ")");

        /*
            An address that no longer publishes its file: a second request confirms the first, then the code
            download_unavailable with the model, the address and the status, the progress and the partial file
            dropped.
        */
        const std::string partial = albireo::join_path(m.models_dir(), "absent.gguf.partial");
        for (int status : { 404, 401, 403, 410, 451 })
        {
            if (status != 404) official_files.refuse("absent.bin", status);
            const int before = official_files.count("absent.bin");
            check(m.start_download("absent", started), "the download of a model its address withdrew starts");
            d = wait_download(m, "absent");
            const std::string gone = "the model 'absent' is no longer published at its official address " + official
                + "absent.bin (status " + std::to_string(status) + "): it can be downloaded again once the catalog "
                "names another address";
            check(failed_with(d, "download_unavailable", gone) && d["args"]["m"].as_string() == "absent"
                && d["args"]["url"].as_string() == official + "absent.bin"
                && d["args"]["status"].as_string() == std::to_string(status)
                && official_files.count("absent.bin") == before + 2 && !db.download_state("absent").is_object()
                && !albireo::file_exists(partial) && !installed(db, "absent", im),
                "an address that answers " + std::to_string(status) + " twice for its file ends the download, "
                "nothing kept (" + d.dump() + ")");
        }
        /*
            The database keeps the address that withdrew the file, for the downloads the service starts by itself:
            the entry that names it waits, an entry that names another address does not, and no other model waits.
        */
        json withdrawn_entry, served_entry;
        albireo::user_error told_again;
        check(m.catalog_entry("absent", withdrawn_entry) && m.withdrawn(withdrawn_entry, &told_again)
            && told_again.code == "download_unavailable"
            && told_again.args["url"].as_string() == official + "absent.bin"
            && told_again.args["status"].as_string() == "451" && told_again.args["m"].as_string() == "absent"
            && m.catalog_entry("linked", served_entry) && !m.withdrawn(served_entry),
            "the model whose address withdrew its file is told with its error, and no other");
        check(!m.withdrawn(catalog_entry("absent", absent, official + "elsewhere.bin"))
            && m.withdrawn(catalog_entry("absent", absent, official + "absent.bin?download=1")),
            "an entry that names another address no longer waits, whatever the query of the address");
        {
            // The address is kept seven days: the service then asks for a default model again by itself.
            const std::string kept = db.value("withdrawn:absent");
            json older = json::parse_or_null(kept);
            older["time"] = older["time"].as_int(0) - 7 * 24 * 3600;
            db.set_value("withdrawn:absent", older.dump());
            const bool lapsed = !m.withdrawn(withdrawn_entry);
            older["time"] = older["time"].as_int(0) + 60;
            db.set_value("withdrawn:absent", older.dump());
            check(lapsed && m.withdrawn(withdrawn_entry), "an address withdrawn seven days ago is asked again");
            db.set_value("withdrawn:absent", kept);
        }
        // The statuses 401 and 403 of a server that does not state them as the hub does: a proxy, a filter.
        official_files.hub = false;
        for (int status : { 401, 403 })
        {
            official_files.refuse("absent.bin", status);
            const int before_refusal = official_files.count("absent.bin");
            check(m.start_download("absent", started), "the download of a model behind a refusal starts");
            d = wait_download(m, "absent");
            const std::string refusal = official + "absent.bin cannot be downloaded: the request is answered with "
                "the status " + std::to_string(status);
            check(failed_with(d, "download_failed", refusal)
                && official_files.count("absent.bin") == before_refusal + 3
                && db.download_state("absent").is_object(), "a status " + std::to_string(status) + " without the "
                "error headers of the hub is a failure of the transfer, the progress kept (" + d.dump() + ")");
        }
        official_files.hub = true;
        // An address that sends to another address of its own server, which lacks the file: withdrawn as well.
        {
            const int before_renamed = official_files.count("lost.bin");
            check(m.start_download("renamed", started), "the download of a model whose server renamed it starts");
            d = wait_download(m, "renamed");
            json renamed_entry;
            check(d["state"].as_string() == "error" && d["code"].as_string() == "download_unavailable"
                && d["args"]["url"].as_string() == base_url + "/renamed/lost.bin"
                && d["args"]["status"].as_string() == "404" && official_files.count("lost.bin") == before_renamed + 2
                && m.catalog_entry("renamed", renamed_entry) && m.withdrawn(renamed_entry),
                "a file its own server lacks after a redirection is withdrawn (" + d.dump() + ")");
        }
        // The same status from a web page comes from a filter of the network: three attempts, the progress kept.
        official_files.fault = html_403;
        int before = official_files.count("absent.bin");
        check(m.start_download("absent", started), "the download of a model behind a filter of the network starts");
        d = wait_download(m, "absent");
        official_files.fault = no_fault;
        const std::string filtered = official + "absent.bin cannot be downloaded: the request is answered with the "
            "status 403";
        check(failed_with(d, "download_failed", filtered) && official_files.count("absent.bin") == before + 3
            && db.download_state("absent").is_object(),
            "a web page with the status 403 is a failure of the transfer, the progress kept (" + d.dump() + ")");
        // The other transfers of a download stop once an address withdrew its file, their bytes dropped.
        {
            const auto began = std::chrono::steady_clock::now();
            check(m.start_download("halved", started), "the download of a model the hub holds in part starts");
            d = wait_download(m, "halved");
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            check(d["state"].as_string() == "error" && d["code"].as_string() == "download_unavailable"
                && d["args"]["url"].as_string() == official + "lost-part.bin" && seconds < 12.0
                && !albireo::file_exists(albireo::join_path(m.models_dir(), "halved.gguf.partial"))
                && !db.download_state("halved").is_object(),
                "a part the hub lacks ends the download without waiting for the part that arrives ("
                + std::to_string(seconds) + " s, " + d.dump() + ")");
        }
        // The refusal of another server, which a redirection led to, is a failure of the transfer as well.
        check(m.start_download("moved", started), "the download of a model whose address redirects starts");
        d = wait_download(m, "moved");
        json moved_entry;
        check(failed_with(d, "download_failed", base_url + "/moved/denied.bin cannot be downloaded: the request is "
            "answered with the status 404") && contents_requests == 3 && db.download_state("moved").is_object()
            && m.catalog_entry("moved", moved_entry) && !m.withdrawn(moved_entry),
            "a status 404 from the server of contents a redirection led to is a failure of the transfer ("
            + d.dump() + ")");
        // The file back at its address: the model comes.
        official_files.serve("absent.bin", absent);
        check(m.start_download("absent", started), "the download of the model starts once its address serves it");
        d = wait_download(m, "absent");
        check(complete(d, "absent", absent) && !m.withdrawn(withdrawn_entry),
            "a model whose address serves it again is installed, and its address forgotten (" + d.dump() + ")");

        // The catalog comes from the folder of the versions alone, which does not hold it yet.
        std::string refreshed;
        const bool refreshed_ok = m.refresh_catalog(refreshed);
        check(!refreshed_ok && refreshed == "the catalog cannot be downloaded from " + dist + "/catalog.json: the "
            "request is answered with the status 404", "a catalog the folder of the versions lacks fails with its "
            "address (" + refreshed + ")");
        m.stop();

        // The updates: the feed and its signature come from the folder of the versions alone.
        const int official_before_updates = official_files.requests;
        int folder = 0;
        auto checked = [&]() {
            albireo::updater u(albireo::join_path(work, "updates-" + std::to_string(++folder)));
            u.check_async();
            json update;
            for (int i = 0; i < 600; ++i)
            {
                update = u.status();
                const std::string state = update["state"].as_string();
                if (state != "idle" && state != "checking") break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return update;
        };
        auto unpublished = [&](const json& update, const std::string& address) {
            return update["state"].as_string() == "error" && update["code"].as_string() == "update_not_published"
                && update["args"]["file"].as_string() == address
                && update["error"].as_string() == "the website does not serve " + address + " yet";
        };
        json update = checked();
        check(unpublished(update, dist + "/feed.json"), "a feed the folder of the versions lacks fails as not "
            "published yet (" + update.dump() + ")");
        // The website answers a file it lacks with its own page.
        dist_files.fault = html_404;
        update = checked();
        dist_files.fault = no_fault;
        check(unpublished(update, dist + "/feed.json"), "a feed the website answers with its page for a missing "
            "file fails as not published yet (" + update.dump() + ")");
        if (signed_feed.empty() || signed_feed_signature.empty())
        {
            std::printf("skip the signed feed, which the sources of this build do not give\n");
        }
        else
        {
            dist_files.serve("feed.json", signed_feed);
            update = checked();
            check(unpublished(update, dist + "/feed.json.sig"), "a feed whose signature the folder lacks fails as "
                "not published yet (" + update.dump() + ")");
            dist_files.serve("feed.json.sig", other_signature);
            update = checked();
            check(update["state"].as_string() == "error" && update["code"].as_string() == "update_signature"
                && update["args"]["file"].as_string() == dist + "/feed.json", "a feed with another signature is "
                "refused (" + update.dump() + ")");
            dist_files.serve("feed.json.sig", signed_feed_signature);
            const std::string published = json::parse_or_null(signed_feed)["channels"]["stable"]["version"]
                .as_string();
            if (albireo::compare_versions(published, ALBIREO_VERSION) > 0)
            {
                std::printf("skip the feed read up to date: it announces %s, later than these sources\n",
                    published.c_str());
            }
            else
            {
                update = checked();
                check(update["state"].as_string() == "up_to_date" && update["latest"].as_string() == published
                    && !update.contains("error") && dist_files.last_encoding() == "identity",
                    "the signed feed of the tests, served by the folder of the versions, is read ("
                    + update.dump() + ")");
            }
            // A filter of the network that answers a web page in place of the feed.
            dist_files.fault = html_403;
            update = checked();
            dist_files.fault = no_fault;
            const std::string refused = dist + "/feed.json cannot be downloaded: the request is answered with the "
                "status 403";
            check(update["state"].as_string() == "error" && update["code"].as_string() == "update_download"
                && update["args"]["e"].as_string() == refused && update["error"].as_string() == refused,
                "a feed refused by a filter of the network fails with its address and the status (" + update.dump()
                + ")");
            dist_files.remove("feed.json");
            dist_files.remove("feed.json.sig");
        }
        check(official_files.requests == official_before_updates, "the updates send no request to an official "
            "address (" + std::to_string(official_files.requests - official_before_updates) + ")");

        // The installer of an update: a new partial file, named once it has the size and the hash the feed names.
        const std::string updates = albireo::join_path(work, "updates-installer");
        albireo::make_dirs(updates);
        const std::string installer = blob(180000, 'i');
        const std::string name = "albireo-setup-online-9.9.9.exe";
        const std::string file = albireo::join_path(updates, "albireo-setup-9.9.9.exe");
        const std::string address = albireo::distribution_address(name);
        const std::string sha = albireo::sha256_hex(installer);
        std::atomic<bool> cancelled{false};
        uint64_t seen = 0;
        auto progress = [&](uint64_t got) { seen = got; };
        albireo::user_error why;
        auto nothing_left = [&]() { return !albireo::file_exists(file) && !albireo::file_exists(file + ".partial"); };
        bool ok = albireo::download_installer(address, file, installer.size(), sha, "9.9.9", progress, cancelled, why);
        check(!ok && why.code == "update_not_published" && why.args["file"].as_string() == address && nothing_left(),
            "an installer the folder of the versions lacks is not published yet (" + why.text + ")");
        dist_files.serve(name, blob(180000, 'j'));
        ok = albireo::download_installer(address, file, installer.size(), sha, "9.9.9", progress, cancelled, why);
        check(!ok && why.code == "update_installer_mismatch" && why.args["version"].as_string() == "9.9.9"
            && nothing_left(), "an installer of other bytes than the feed names is refused (" + why.text + ")");
        dist_files.serve(name, installer + "x");
        ok = albireo::download_installer(address, file, installer.size(), sha, "9.9.9", progress, cancelled, why);
        check(!ok && why.code == "update_installer_mismatch" && nothing_left(),
            "an installer larger than the feed names is refused (" + why.text + ")");
        dist_files.fault = html_200;
        ok = albireo::download_installer(address, file, installer.size(), sha, "9.9.9", progress, cancelled, why);
        dist_files.fault = no_fault;
        check(!ok && why.code == "update_download" && why.args["e"].as_string() == address + " cannot be downloaded: "
            + page_text && nothing_left(), "a web page in place of the installer fails its download (" + why.text
            + ")");
        dist_files.serve(name, installer);
        write_text(file + ".partial", "left by an earlier attempt");
        ok = albireo::download_installer(address, file, installer.size(), sha, "9.9.9", progress, cancelled, why);
        check(ok && why.code.empty() && read_text(file) == installer && !albireo::file_exists(file + ".partial")
            && seen == installer.size(), "the installer the feed names takes its name, over the partial file of an "
            "earlier attempt (" + why.text + ")");
        albireo::remove_file(file);
        cancelled = true;
        ok = albireo::download_installer(address, file, installer.size(), sha, "9.9.9", progress, cancelled, why);
        check(!ok && nothing_left(), "a download of the installer ended by the service leaves no file");
        dist_files.remove(name);
    }

    {
        /*
            The failures of the folder of the versions and of the official addresses, each with the bounds of
            the waits shortened to 1 s and of a whole request to 2 s: the operation fails with the address and
            the failure, no exception leaves it, and the local server still answers.
        */
        reset_server resetting;
        check(resetting.start(port + 1), "a server that resets its connections listens on the port "
            + std::to_string(port + 1));
        albireo::http_bounds_for_tests(1, 2);
        auto answers = [&]() {
            std::string alive, e;
            return albireo::http_get(base_url + "/alive", alive, e, 64) && alive == "alive";
        };
        albireo::store db(albireo::join_path(work, "albireo-4.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-11-01T00:00:00Z";
        json list = json::array();
        list.push_back(catalog_entry("tiny", served, official + "model.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        albireo::model_manager m(db, albireo::join_path(work, "models-4"));
        const std::string from = "the catalog cannot be downloaded from " + dist + "/catalog.json: ";
        struct failure_case
        {
            const char* what;
            int f;
            std::string expected;
        };
        const failure_case cases[] = {
            {"a catalog answered by a web page with 200", html_200, from + page_text},
            {"a catalog answered by a web page with 403", html_403, from + "the request is answered with the status "
                "403"},
            {"a catalog answered by a proxy that asks for its authentication", proxy_407,
                from + "the proxy asks for an authentication (status 407)"},
            {"a catalog from a server that stalls", stall, from + "no data arrives within 1 s"},
            {"a catalog sent byte by byte", trickle, from + "no complete answer arrives within 2 s"}
        };
        for (const auto& c : cases)
        {
            dist_files.fault = c.f;
            std::string error;
            bool ok = true, escaped = false;
            const auto begun = std::chrono::steady_clock::now();
            try
            {
                ok = m.refresh_catalog(error);
            }
            catch (...)
            {
                escaped = true;
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
            dist_files.fault = no_fault;
            check(!ok && !escaped && error == c.expected && seconds < 10 && answers(), std::string(c.what)
                + " fails with its message, in " + std::to_string(static_cast<int>(seconds)) + " s (" + error + ")");
        }
        // A folder of the versions whose server resets its connections, or does not listen.
        for (int to : { port + 1, port + 2 })
        {
            const std::string elsewhere = "http://127.0.0.1:" + std::to_string(to) + "/dist";
            albireo::distribution_for_tests(elsewhere);
            const std::string beginning = "the catalog cannot be downloaded from " + elsewhere + "/catalog.json: ";
            std::string error;
            bool ok = true, escaped = false;
            try
            {
                ok = m.refresh_catalog(error);
            }
            catch (...)
            {
                escaped = true;
            }
            check(!ok && !escaped && error.compare(0, beginning.size(), beginning) == 0
                && error.size() > beginning.size() && answers(), std::string(to == port + 1
                ? "a server that resets its connections" : "a server that does not listen")
                + " fails the catalog with its message (" + error + ")");
        }
        albireo::distribution_for_tests(dist);

        // The ranges of the downloads: a web page, a stall and a reset fail before any byte reaches the file.
        std::string written, error;
        auto keep = [&](const char* data, size_t n) {
            written.append(data, n);
            return true;
        };
        bool escaped = false;
        official_files.fault = html_200;
        const bool from_page = albireo::http_get_range(official + "model.bin", 0, 999, keep, error);
        const std::string page_error = error;
        official_files.fault = stall;
        const bool from_stall = albireo::http_get_range(official + "model.bin", 0, 999, keep, error);
        const std::string stall_error = error;
        official_files.fault = no_fault;
        const bool from_reset = albireo::http_get_range("http://127.0.0.1:" + std::to_string(port + 1)
            + "/official/model.bin", 1000, 1999, keep, error);
        // A receiver that throws: the exception becomes the error of the download.
        try
        {
            albireo::http_get_range(official + "model.bin", 0, 999, [](const char*, size_t) -> bool {
                throw std::runtime_error("the disk is full");
            }, error);
        }
        catch (...)
        {
            escaped = true;
        }
        check(!from_page && !from_stall && !from_reset && written.empty() && page_error == page_text
            && stall_error == "no data arrives within 1 s" && !escaped
            && error == "the transfer fails: the disk is full", "a range answered by a web page, a stall or a reset "
            "writes nothing, and an exception of its receiver becomes its error (" + page_error + "; " + stall_error
            + "; " + error + ")");

        // A model whose official address answers a web page, then the file.
        albireo::user_error started;
        official_files.fault = html_200;
        check(m.start_download("tiny", started), "the download of a model whose address answers a web page starts");
        json d = wait_download(m, "tiny");
        official_files.fault = no_fault;
        const std::string partial = albireo::join_path(m.models_dir(), "tiny.gguf.partial");
        const std::string paged = official + "model.bin cannot be downloaded: " + page_text;
        check(d["state"].as_string() == "error" && d["code"].as_string() == "download_failed"
            && d["error"].as_string() == paged && d["args"]["e"].as_string() == paged
            && read_text(partial).find("Access denied") == std::string::npos && answers(),
            "a model whose address answers a web page fails with it, its partial file free of the page ("
            + d.dump() + ")");
        check(m.start_download("tiny", started), "the download of the model starts again");
        d = wait_download(m, "tiny");
        albireo::installed_model im;
        check(d["state"].as_string() == "done" && installed(db, "tiny", im) && read_text(im.file) == served
            && im.sha256 == albireo::sha256_hex(served) && answers(), "the model comes once its address serves the "
            "file (" + d.dump() + ")");
        m.stop();
        albireo::http_bounds_for_tests(0, 0);
        resetting.stop();
    }

    {
        /*
            A model of eight parts, one of them compressed, the address of the last one failing at first: the
            download ends in error with its progress, then resumes with that part alone.  The catalog comes
            from the folder of the versions, its signature checked, and a download cancelled while its file
            arrives ends at once.
        */
        albireo::store db(albireo::join_path(work, "albireo-5.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-11-01T00:00:00Z";
        json list = json::array();
        std::vector<std::string> contents, urls, packed;
        for (int i = 0; i < 8; ++i)
        {
            const std::string name = "part-" + std::to_string(i) + ".bin";
            contents.push_back(blob(20000 + 1000 * i, static_cast<char>('A' + i)));
            urls.push_back(official + name);
            packed.push_back(i == 3 ? deflated(contents.back()) : std::string());
            official_files.serve(name, packed.back().empty() ? contents.back() : packed.back());
        }
        // The address of the last part answers that its server fails.
        official_files.refuse("part-7.bin", 500);
        list.push_back(parts_entry("parts", contents, urls, packed));
        const std::string slow = blob(300000, 'v');
        list.push_back(catalog_entry("slow", slow, official + "slow.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        official_files.serve("slow.bin", slow);
        albireo::model_manager m(db, albireo::join_path(work, "models-5"));
        albireo::user_error started;
        albireo::installed_model im;
        check(m.start_download("parts", started), "the download of a model of eight parts starts");
        json d = wait_download(m, "parts");
        const std::string last = official + "part-7.bin cannot be downloaded: the request is answered with the status "
            "500";
        check(d["state"].as_string() == "error" && d["code"].as_string() == "download_failed"
            && d["error"].as_string() == last && db.download_state("parts").is_object(),
            "a part whose address fails ends the download in error, its progress kept (" + d.dump() + ")");
        official_files.serve("part-7.bin", contents[7]);
        check(m.start_download("parts", started), "the download of the model of eight parts starts again");
        d = wait_download(m, "parts");
        std::string whole;
        bool each_once = true;
        for (int i = 0; i < 8; ++i)
        {
            whole += contents[static_cast<size_t>(i)];
            each_once = each_once && official_files.count("part-" + std::to_string(i) + ".bin") == (i < 7 ? 1 : 4);
        }
        check(d["state"].as_string() == "done" && installed(db, "parts", im) && read_text(im.file) == whole
            && im.sha256 == albireo::sha256_hex(whole) && d["received"].as_int(-1) == d["total"].as_int(-2)
            && each_once, "a model of eight parts, one compressed, resumes with the part it lacked (" + d.dump()
            + ")");

        // The catalog: the folder of the versions serves the signed catalog of the repository.
        if (repository_catalog.empty() || repository_signature.empty())
        {
            std::printf("skip the signed catalog, which the sources of this build do not give\n");
        }
        else
        {
            const json signed_catalog = json::parse_or_null(repository_catalog);
            const long long models = static_cast<long long>(signed_catalog["models"].size());
            const std::string address = dist + "/catalog.json";
            albireo::store cdb(albireo::join_path(work, "albireo-6.db"));
            albireo::model_manager cm(cdb, albireo::join_path(work, "models-6"));
            std::string error;
            dist_files.serve("catalog.json", repository_catalog);
            bool ok = cm.refresh_catalog(error);
            check(!ok && error == "the signature of the catalog cannot be downloaded from " + address + ".sig: the "
                "request is answered with the status 404", "a catalog whose signature the folder lacks fails with "
                "the address of the signature (" + error + ")");
            dist_files.serve("catalog.json.sig", other_signature);
            ok = cm.refresh_catalog(error);
            check(!ok && error == "the catalog of " + address + " is refused: the signature of the catalog is not "
                "valid" && cm.catalog_status()["models"].as_int() == 0, "a catalog with another signature is "
                "refused (" + error + ")");
            dist_files.serve("catalog.json.sig", repository_signature);
            const int lines = logged("models: catalog read from " + address);
            error.clear();
            ok = cm.refresh_catalog(error);
            const json s = cm.catalog_status();
            check(ok && s["models"].as_int() == models && s["state"].as_string() == "ok"
                && s["updated"].as_string() == signed_catalog["updated"].as_string()
                && logged("models: catalog read from " + address) == lines + 1,
                "the signed catalog of the repository, served by the folder of the versions, is installed (" + error
                + ")");
            dist_files.remove("catalog.json");
            dist_files.remove("catalog.json.sig");
            cm.stop();
        }

        // A download cancelled while its file arrives ends at once, its partial file removed.
        check(m.start_download("slow", started), "the download of a model whose address is slow starts");
        long long received = 0;
        for (int i = 0; i < 300 && received == 0; ++i)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const json downloads = m.downloads();
            for (const auto& x : downloads.elements())
                if (x["id"].as_string() == "slow") received = x["received"].as_int(0);
        }
        const auto cancelled_at = std::chrono::steady_clock::now();
        m.cancel_download("slow");
        d = wait_download(m, "slow");
        const double cancel_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()
            - cancelled_at).count();
        check(received > 0 && received < static_cast<long long>(slow.size()) && d["state"].as_string() == "cancelled"
            && cancel_seconds < 5 && official_files.count("slow.bin") == 1 && !installed(db, "slow", im)
            && !albireo::file_exists(albireo::join_path(m.models_dir(), "slow.gguf.partial"))
            && !db.download_state("slow").is_object(), "a download cancelled while its file arrives ends in "
            + std::to_string(static_cast<int>(cancel_seconds)) + " s, its partial file and progress removed ("
            + d.dump() + ")");
        m.stop();
    }

    {
        // The settings of an earlier version: read without error, written again without what this version lacks.
        const std::string folder = albireo::join_path(work, "settings");
        albireo::make_dirs(folder);
        const std::string settings_file = albireo::join_path(folder, "settings.json");
        const std::string key_file = albireo::join_path(folder, "web_search.secret");
        write_text(settings_file, "{\"port\": 9123, \"web_search\": \"brave\", \"web_search_url\": "
            "\"https://searx.example.org\", \"web_search_key_set\": true, \"web_search_key\": \"BSA-by-hand-42\", "
            "\"public_link\": \"https://storage.example/link?code=0123456789\"}");
        write_text(key_file, "BSA-secret-0123456789");
        {
            albireo::settings_store store(settings_file);
            const std::string text = read_text(settings_file), copy = read_text(settings_file + ".bak");
            check(store.get().port == 9123 && store.get().web_search == "wikipedia"
                && text.find("storage.example") == std::string::npos
                && copy.find("storage.example") == std::string::npos && text.find("web_search_url") == std::string::npos
                && text.find("web_search_key") == std::string::npos && text.find("public_link") == std::string::npos
                && text.find("BSA-") == std::string::npos && !albireo::file_exists(key_file),
                "the settings of an earlier version are read, and written again without what this version lacks");
        }
        albireo::host_settings s;
        const char* const providers[] = { "auto", "brave", "searxng", "wikipedia" };
        bool wikipedia = s.web_search == "wikipedia";
        for (const char* p : providers)
        {
            json j = json::object();
            j["web_search"] = p;
            s.from_json(j);
            wikipedia = wikipedia && s.web_search == "wikipedia";
        }
        json none = json::object();
        none["web_search"] = "none";
        s.from_json(none);
        const bool off = s.web_search == "none";
        json number = json::object();
        number["web_search"] = 3;
        s.from_json(number);
        const json document = s.to_json();
        check(wikipedia && off && s.web_search == "wikipedia" && !document.contains("public_link")
            && !document.contains("web_search_url") && !document.contains("web_search_key_set"),
            "the search of the settings is wikipedia or none, and the documents give no link and no key");

        // The graph of OpenVINO: auto, the default before version 2 of the document, reads off.
        write_text(settings_file, "{\"port\": 9124, \"graph\": \"auto\"}");
        bool before = false;
        {
            albireo::settings_store store(settings_file);
            const json written = json::parse_or_null(read_text(settings_file));
            before = store.get().port == 9124 && store.get().graph == "off"
                && written["settings_version"].as_int() == 3 && written["graph"].as_string() == "off";
        }
        // The accelerator cpu, the graph on the processor chosen before version 2, keeps it.
        write_text(settings_file, "{\"port\": 9126, \"graph\": \"auto\", \"accelerator\": \"cpu\"}");
        bool kept = false;
        {
            albireo::settings_store store(settings_file);
            kept = store.get().port == 9126 && store.get().graph == "auto" && store.get().accelerator == "cpu";
        }
        write_text(settings_file, "{\"port\": 9125, \"graph\": \"auto\", \"settings_version\": 2}");
        bool chosen = false;
        {
            albireo::settings_store store(settings_file);
            chosen = store.get().port == 9125 && store.get().graph == "auto";
        }
        // A document of a later version keeps its version when this one writes it.
        write_text(settings_file, "{\"port\": 9127, \"graph\": \"auto\", \"settings_version\": 4}");
        bool later = false;
        {
            albireo::settings_store store(settings_file);
            store.update([](albireo::host_settings& s) { s.port = 9128; });
            const json written = json::parse_or_null(read_text(settings_file));
            later = store.get().graph == "auto" && written["port"].as_int() == 9128
                && written["settings_version"].as_int() == 4;
        }
        // The 8192 tokens of an answer of the relay, the default of version 2, read 16384; chosen since, they stay.
        write_text(settings_file, "{\"port\": 9129, \"relay\": {\"max_output\": 8192}, \"settings_version\": 2}");
        bool raised = false;
        {
            albireo::settings_store store(settings_file);
            const json written = json::parse_or_null(read_text(settings_file));
            raised = store.get().relay.max_output == 16384 && written["relay"]["max_output"].as_int() == 16384
                && written["settings_version"].as_int() == 3;
        }
        write_text(settings_file, "{\"port\": 9130, \"relay\": {\"max_output\": 8192}, \"settings_version\": 3}");
        bool output_kept = false;
        {
            albireo::settings_store store(settings_file);
            output_kept = store.get().port == 9130 && store.get().relay.max_output == 8192;
        }
        check(raised && output_kept && albireo::relay_settings().max_output == 16384,
            "the relay gives 16384 tokens to an answer by default, 8192 of a document of version 2 reads 16384, and "
            "8192 chosen since stays");
        json change = albireo::host_settings().to_json();
        change["graph"] = "auto";
        albireo::host_settings changed;
        changed.from_json(change);
        check(before && kept && chosen && later && albireo::host_settings().graph == "off" && changed.graph == "auto"
                && !change.contains("settings_version"),
            "the graph of OpenVINO is off by default, auto of a document before version 2 reads off unless the "
            "accelerator is cpu, auto chosen since then stays, and a document of a later version keeps its version");
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

    {
        // The public addresses: those that follow and precede each range that is not public, and its bounds.
        const char* const public_addresses[] = {
            "1.0.0.0", "9.255.255.255", "11.0.0.0", "100.63.255.255", "100.128.0.0", "126.255.255.255",
            "128.0.0.0", "169.253.255.255", "169.255.0.0", "172.15.255.255", "172.32.0.0", "191.255.255.255",
            "192.0.1.0", "192.0.1.255", "192.0.3.0", "192.88.98.255", "192.88.100.0", "192.167.255.255",
            "192.169.0.0", "198.17.255.255", "198.20.0.0", "198.51.99.255", "198.51.101.0", "203.0.112.255",
            "203.0.114.0", "223.255.255.255", "8.8.8.8", "::1:0:0", "ff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "100:0:0:1::", "2001:db7:ffff:ffff:ffff:ffff:ffff:ffff", "2001:db9::", "2000:ffff:ffff:ffff::",
            "2001:1::", "64:ff9b:0:ffff:ffff:ffff:ffff:ffff", "64:ff9b:2::",
            "fbff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", "fe00::", "fe7f:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "::fffe:ffff:ffff", "::1:0:0:0", "::ffff:8.8.8.8", "64:ff9a:ffff:ffff:ffff:ffff:ffff:ffff",
            "64:ff9b::1:0:0", "64:ff9b::808:808", "2001:ffff:ffff:ffff:ffff:ffff:ffff:ffff", "2003::",
            "2002:808:808::", "2606:4700:4700::1111"
        };
        const char* const private_addresses[] = {
            "0.0.0.0", "0.255.255.255", "10.0.0.0", "10.255.255.255", "100.64.0.0", "100.127.255.255",
            "127.0.0.0", "127.255.255.255", "169.254.0.0", "169.254.255.255", "172.16.0.0", "172.31.255.255",
            "192.0.0.0", "192.0.0.255", "192.0.2.0", "192.0.2.255", "192.88.99.0", "192.88.99.255",
            "192.168.0.0", "192.168.255.255", "198.18.0.0", "198.19.255.255", "198.51.100.0", "198.51.100.255",
            "203.0.113.0", "203.0.113.255", "224.0.0.0", "239.255.255.255", "240.0.0.0", "255.255.255.255",
            "::", "::1", "100::", "100::ffff:ffff:ffff:ffff", "2001:db8::", "2001:db8:ffff:ffff:ffff:ffff:ffff:ffff",
            "fc00::", "fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", "fe80::", "febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "fec0::", "feff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", "ff00::", "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            "::ffff:0.0.0.0", "::ffff:127.0.0.1", "::ffff:10.1.2.3", "::ffff:192.168.1.1", "64:ff9b::7f00:1",
            "64:ff9b::a00:1", "2002:7f00:1::", "2002:c0a8:101::1", "2002:a9fe:1::", "::2", "::7f00:1",
            "::ffff:ffff", "64:ff9b:1::", "64:ff9b:1::a00:1", "64:ff9b:1:ffff:ffff:ffff:ffff:ffff", "2001::",
            "2001::1", "2001:0:ffff:ffff:ffff:ffff:ffff:ffff"
        };
        std::string wrong;
        for (const char* a : public_addresses)
            if (!albireo::public_address(a)) wrong += std::string(" ") + a;
        check(wrong.empty(), "the addresses beside the ranges that are not public are public" + wrong);
        wrong.clear();
        for (const char* a : private_addresses)
            if (albireo::public_address(a)) wrong += std::string(" ") + a;
        check(wrong.empty(), "the addresses of the ranges that are not public are not" + wrong);
        check(!albireo::public_address("example.org") && !albireo::public_address("127.1")
            && !albireo::public_address(""), "a text that is no address is not public");

        // The addresses refused before any connection: 400 for an invalid one, 403 for a local or private one.
        albireo::host_settings web;
        auto fetch = [&](const std::string& url, const albireo::host_settings& s, const std::string& language) {
            json request = json::object();
            request["url"] = url;
            if (!language.empty()) request["language"] = language;
            return albireo::web_fetch(request, s);
        };
        struct refusal
        {
            std::string url;
            int status;
        };
        const refusal refusals[] = {
            {"ftp://example.org/file", 400}, {"file:///etc/passwd", 400}, {"javascript:alert(1)", 400},
            {"http://user:secret@example.org/", 400}, {"http://user@example.org/", 400}, {"http:///no-host", 400},
            {"http://exa mple.org/", 400}, {"http://999.1.1.1/", 400}, {"http://example.org:99999/", 400},
            {"https://intranet/", 403}, {"http://localhost/", 403}, {"http://LOCALHOST./", 403},
            {"http://app.localhost:8080/", 403}, {"http://printer.local/", 403}, {"http://db.internal/", 403},
            {"http://nas.home.arpa/", 403}, {"http://2130706433/", 403}, {"http://0x7f.1/", 403},
            {"http://127.1/", 403}, {"http://017700000001/", 403}, {"http://[::1]/", 403},
            {"http://[::ffff:127.0.0.1]/", 403}, {"http://[::ffff:7f00:1]/", 403}, {"http://10.0.0.1/", 403},
            {"http://[fe80::1]/", 403}, {"http://169.254.169.254/latest/meta-data/", 403},
            {base_url + "/web/page.html", 403}
        };
        for (const auto& r : refusals)
        {
            const albireo::web_reply reply = fetch(r.url, web, "");
            check(reply.status == r.status, "the address " + r.url + " is refused with " + std::to_string(r.status)
                + " (" + std::to_string(reply.status) + " " + reply.code + ": " + reply.message + ")");
        }
        const albireo::web_reply local = fetch("http://[::ffff:127.0.0.1]/", web, "");
        check(local.code == "address_refused" && local.message == "the address of [::ffff:127.0.0.1] is local or "
            "private, which the web access of Albireo does not read",
            "a refusal names the host (" + local.message + ")");

        // The service offline and the web access turned off refuse both routes; the capability follows.
        albireo::host_settings offline = web, off = web, none = web;
        offline.offline = true;
        off.web_access = false;
        none.web_search = "none";
        json search_request = json::object();
        search_request["query"] = "Einstein relativit\xc3\xa9";
        albireo::web_reply reply = fetch("https://example.org/", offline, "");
        check(reply.status == 403 && reply.code == "offline", "a page is refused offline (" + reply.message + ")");
        reply = albireo::web_search(search_request, offline);
        check(reply.status == 403 && reply.code == "offline", "a search is refused offline");
        reply = fetch("https://example.org/", off, "");
        check(reply.status == 403 && reply.code == "web_access_off",
            "a page is refused with the web access turned off");
        reply = albireo::web_search(search_request, off);
        check(reply.status == 403 && reply.code == "web_access_off", "a search is refused with the web access off");
        reply = albireo::web_search(search_request, none);
        check(reply.status == 403 && reply.code == "no_search_provider" && reply.message == "no search provider is set",
            "a search without provider is refused");
        check(albireo::web_capability(web).dump() == "{\"available\":true,\"search\":\"wikipedia\"}"
            && albireo::web_capability(offline).dump() == "{\"available\":false,\"search\":\"none\"}"
            && albireo::web_capability(off).dump() == "{\"available\":false,\"search\":\"none\"}"
            && albireo::web_capability(none).dump() == "{\"available\":true,\"search\":\"none\"}"
            && albireo::web_search_provider(web) == "wikipedia" && albireo::web_search_provider(none) == "none",
            "the capability gives the web access and its provider of search, wikipedia or none");

        // The pages of the local server, the loopback allowed for the tests.
        albireo::web_allow_loopback_for_tests(true);
        reply = fetch(base_url + "/web/page.html", web, "");
        const json& b = reply.body;
        check(reply.status == 200 && b["status"].as_int() == 200 && b["content_type"].as_string() == "text/html"
            && b["charset"].as_string() == "utf-8" && !b["truncated"].as_bool()
            && b["size"].as_int() == static_cast<long long>(page.size())
            && albireo::base64_decode(b["data"].as_string()) == page && b["url"].as_string() == base_url
            + "/web/page.html" && b["final_url"].as_string() == base_url + "/web/page.html",
            "a page is read with its status, its type and its bytes (" + reply.message + ")");
        reply = fetch(base_url + "/web/large.txt", web, "");
        check(reply.status == 200 && reply.body["truncated"].as_bool() && reply.body["size"].as_int() == (2 << 20)
            && albireo::base64_decode(reply.body["data"].as_string()) == large_text.substr(0, 2u << 20),
            "a text is cut at 2 MB (" + reply.message + ")");
        reply = fetch(base_url + "/web/image.png", web, "");
        check(reply.status == 415 && reply.code == "unsupported_type" && reply.message.find("image/png")
            != std::string::npos, "a type the web access does not read is refused (" + reply.message + ")");
        reply = fetch(base_url + "/web/small.pdf", web, "");
        check(reply.status == 200 && reply.body["content_type"].as_string() == "application/pdf"
            && albireo::base64_decode(reply.body["data"].as_string()) == "%PDF-1.4\n%%EOF\n", "a PDF file is read");
        reply = fetch(base_url + "/web/large.pdf", web, "");
        check(reply.status == 413 && reply.code == "too_large", "a PDF file announced above 20 MB is refused ("
            + reply.message + ")");
        reply = fetch(base_url + "/web/stream.pdf", web, "");
        check(reply.status == 413 && reply.code == "too_large", "a PDF file that passes 20 MB while it is read is "
            "refused (" + reply.message + ")");
        reply = fetch(base_url + "/web/missing", web, "");
        check(reply.status == 200 && reply.body["status"].as_int() == 404
            && albireo::base64_decode(reply.body["data"].as_string()) == "<p>no such page</p>",
            "a page that answers 404 is given with its status and its body");
        reply = fetch(base_url + "/web/to-private", web, "");
        check(reply.status == 403 && reply.code == "address_refused" && reply.message.find("192.168.1.10")
            != std::string::npos, "a redirection to a private address is refused (" + reply.message + ")");
        reply = fetch(base_url + "/web/relative", web, "");
        check(reply.status == 200 && reply.body["final_url"].as_string() == base_url + "/web/page.html?from=relative"
            && albireo::base64_decode(reply.body["data"].as_string()) == page,
            "a relative redirection is followed to its address (" + reply.message + ")");
        reply = fetch(base_url + "/web/loop", web, "");
        check(reply.status == 502 && reply.code == "too_many_redirects",
            "redirections are followed five times at most");
        reply = fetch(base_url + "/web/headers", web, "fr-FR");
        check(reply.status == 200 && albireo::base64_decode(reply.body["data"].as_string()) == std::string("Albireo/")
            + ALBIREO_VERSION + " (+https://github.com/Cydral/Dlib-Transformer-extensions)\nfr-FR\n",
            "a page receives the User-Agent of Albireo, the language asked and no cookie");
        albireo::http_cancel_all("the service stops");
        reply = fetch(base_url + "/web/page.html", web, "");
        check(reply.status == 503 && reply.code == "cancelled", "a service that stops ends the reading of pages");
        albireo::http_restore();

        // The search of Wikipedia, answered by the local server.
        albireo::web_provider_origin_for_tests(base_url);
        search_request["count"] = 5;
        search_request["language"] = "fr";
        reply = albireo::web_search(search_request, web);
        const json results = reply.body["results"];
        check(reply.status == 200 && reply.body["provider"].as_string() == "wikipedia" && results.size() == 2
            && results[0]["title"].as_string() == "Albert Einstein"
            && results[0]["url"].as_string() == "https://fr.wikipedia.org/wiki/Albert_Einstein"
            && results[0]["snippet"].as_string() == "Albert Einstein was a \"physicist\""
            && results[1]["url"].as_string() == "https://fr.wikipedia.org/wiki/Einstein_(crater)"
            && results[1]["snippet"].as_string() == "Lunar crater",
            "Wikipedia gives its pages (" + reply.message + ")");
        {
            std::lock_guard<std::mutex> lock(seen_m);
            check(seen_query == "Einstein relativit\xc3\xa9" && seen_count == "5"
                && seen_agent == std::string("Albireo/") + ALBIREO_VERSION
                + " (https://github.com/Cydral/Dlib-Transformer-extensions; contact@cydral-technology.eu)",
                "Wikipedia receives the query, the count and the User-Agent with a contact (" + seen_agent + ")");
        }
        search_request["query"] = "too many";
        reply = albireo::web_search(search_request, web);
        check(reply.status == 429 && reply.code == "rate_limited" && reply.retry_after == 30
            && reply.message.find("Wikipedia limits the rate") != std::string::npos,
            "a limit of the rate of Wikipedia is reported with its delay (" + reply.message + ")");
        json invalid = json::object();
        invalid["query"] = "   ";
        const bool empty_refused = albireo::web_search(invalid, web).code == "invalid_query";
        invalid["query"] = std::string(401, 'q');
        const bool long_refused = albireo::web_search(invalid, web).code == "invalid_query";
        std::string four_hundred;
        for (int i = 0; i < 400; ++i) four_hundred += "\xc3\xa9";
        invalid["query"] = four_hundred;
        const bool characters_counted = albireo::web_search(invalid, web).status == 200;
        invalid["query"] = "ok";
        invalid["count"] = 11;
        const bool count_refused = albireo::web_search(invalid, web).code == "invalid_count";
        invalid["count"] = 2;
        invalid["language"] = "french";
        const bool language_refused = albireo::web_search(invalid, web).code == "invalid_language";
        check(empty_refused && long_refused && characters_counted && count_refused && language_refused,
            "a query of 1 to 400 characters, a count of 1 to 10 and a code of language are required");

        albireo::web_provider_origin_for_tests("");
        albireo::web_allow_loopback_for_tests(false);

        // The log names the hosts read and the provider, never a path or a query.
        bool leaked = false, read_line = false, search_line = false;
        for (const auto& line : albireo::log_file::instance().recent())
        {
            leaked = leaked || line.find("relativit") != std::string::npos || line.find("/web/") != std::string::npos
                || line.find("q=") != std::string::npos;
            read_line = read_line || line.find("web: 127.0.0.1 read (200, ") != std::string::npos;
            search_line = search_line || line.find("web: search through wikipedia (2 results)") != std::string::npos;
        }
        check(!leaked && read_line && search_line, "the log names the hosts and the provider, never a path or a "
            "query");
    }

    {
        // The relay: its secret in the settings, the restoration of the substitutes, the memory of the
        // pseudonymization, and the requests of its three providers to a local server that answers as each.
        const std::string folder = albireo::join_path(work, "relay-settings");
        albireo::make_dirs(folder);
        const std::string file = albireo::join_path(folder, "settings.json");
        const std::string secret_file = albireo::join_path(folder, "relay.secret");
        const std::string key = "sk-ant-test-0123456789";
        {
            albireo::settings_store store(file);
            store.update([&](albireo::host_settings& s) {
                s.relay.enabled = true;
                s.relay.provider = "claude";
                s.relay.model = "claude-opus-5-5";
                s.relay.secret = key;
            });
        }
        const std::string text = read_text(file);
        bool reread = false;
        {
            albireo::settings_store store(file);
            reread = store.get().relay.secret == key && store.get().relay.to_json()["secret"].as_string() == "********";
        }
        // On Windows, relay.secret holds the key encrypted for the account (DPAPI).
#if defined(_WIN32)
        const bool sealed = read_text(secret_file).find(key) == std::string::npos;
#else
        const bool sealed = true;
#endif
        check(text.find(key) == std::string::npos && text.find("\"secret\": \"********\"") != std::string::npos
            && read_text(file + ".bak").find(key) == std::string::npos && albireo::file_exists(secret_file) && reread
            && sealed, "the key of the relay stays out of settings.json and its copy, kept in relay.secret and read "
            "again");
        json edited = json::parse_or_null(text);
        edited["relay"]["provider"] = "ollama";
        write_text(file, edited.dump());
        bool elsewhere = false;
        {
            albireo::settings_store store(file);
            elsewhere = store.get().relay.secret.empty();
            store.update([](albireo::host_settings& s) { s.relay.enabled = false; });
        }
        check(elsewhere && !albireo::file_exists(secret_file),
            "a settings.json that names another provider reads no key, and relay.secret goes with the key");

        // The values come back in an answer cut at any place, at the boundaries of words only.
        std::map<std::string, std::string> values;
        values["Xoydoeh Nuggwdfsorn"] = "Jean Dupont";
        values["Qbspm"] = "Paris";
        values["[[PERSON:Xoydoeh Nuggwdfsorn#EKR]]"] = "Jean Dupont";
        {
            albireo::stream_restorer r(values);
            std::string out;
            for (const char* chunk : { "Bonjour Xoy", "doeh Nugg", "wdfsorn, de Qbs", "pm et Qbspmx. ",
                     "M. Nuggwdfsorn", " et [[PERSON:Xoydoeh", " Nuggwdfsorn#EKR]], d\xE2\x80\x99Xoydoeh" })
                out += r.feed(chunk);
            out += r.finish();
            check(out == "Bonjour Jean Dupont, de Paris et Qbspmx. M. Dupont et Jean Dupont, d\xE2\x80\x99Jean",
                "the substitutes of an answer cut anywhere come back as their values, their words alone too ("
                + out + ")");
            std::map<std::string, std::string> both;
            both["Mxrtin"] = "Durand";
            both["Jxan Mxrtin"] = "Paul Dupont";
            albireo::stream_restorer w(both);
            const std::string kept = w.feed("Ask Mxrtin, then Jxan Mxrtin.") + w.finish();
            check(kept == "Ask Durand, then Paul Dupont.", "a whole substitute keeps its value when it is also a word "
                "of another one (" + kept + ")");
        }

        // A local server that answers as Pseudalis and as the providers.
        struct relay_mock
        {
            std::mutex m;
            int anonymized = 0;
            std::string mode = "ok";
            json body;
            httplib::Headers headers;
        } mock;
        const std::vector<std::pair<std::string, std::string>> people = { { "Jean Dupont", "Xoydoeh Nuggwdfsorn" },
            { "Paris", "Qbspm" }, { "Marie Curie", "Lsyiq Bfksz" }, { "Lyon", "Wpzq" } };
        auto tag_of = [](const std::string& substitute) { return "[[PERSON:" + substitute + "#EKR]]"; };
        httplib::Server relay_server;
        relay_server.Post("/v1/text/anonymize", [&](const httplib::Request& req, httplib::Response& res) {
            const json in = json::parse_or_null(req.body);
            const bool tags = in["tags"].as_bool(true);
            std::string text = in["text"].as_string(), out;
            json spans = json::array(), subs = json::array();
            for (size_t i = 0; i < text.size();)
            {
                bool hit = false;
                for (const auto& p : people)
                    if (text.compare(i, p.first.size(), p.first) == 0)
                    {
                        const std::string written = tags ? tag_of(p.second) : p.second;
                        json span = json::object();
                        span["begin"] = static_cast<long long>(out.size());
                        out += written;
                        span["end"] = static_cast<long long>(out.size());
                        span["type"] = "PERSON";
                        spans.push_back(span);
                        json sub = json::object();
                        sub["substitute"] = p.second;
                        sub["tag"] = tag_of(p.second);
                        subs.push_back(sub);
                        i += p.first.size();
                        hit = true;
                        break;
                    }
                if (!hit) out += text[i++];
            }
            json o = json::object();
            o["text"] = out;
            o["tags"] = spans;
            if (!tags) o["substitutions"] = subs;
            {
                std::lock_guard<std::mutex> lock(mock.m);
                ++mock.anonymized;
            }
            res.set_content(o.dump(), "application/json");
        });
        relay_server.Post("/v1/text/restore", [&](const httplib::Request& req, httplib::Response& res) {
            const json in = json::parse_or_null(req.body);
            const std::string text = in["text"].as_string();
            std::string out;
            json spans = json::array();
            for (size_t i = 0; i < text.size();)
            {
                bool hit = false;
                for (const auto& p : people)
                    for (const std::string& token : { tag_of(p.second), p.second })
                        if (!hit && text.compare(i, token.size(), token) == 0)
                        {
                            json span = json::object();
                            span["begin"] = static_cast<long long>(out.size());
                            out += p.first;
                            span["end"] = static_cast<long long>(out.size());
                            spans.push_back(span);
                            i += token.size();
                            hit = true;
                        }
                if (!hit) out += text[i++];
            }
            json o = json::object();
            o["text"] = out;
            o["restored"] = spans;
            res.set_content(o.dump(), "application/json");
        });
        auto event = [](const json& data) { return "data: " + data.dump() + "\n\n"; };
        auto record = [&](const httplib::Request& req) {
            std::lock_guard<std::mutex> lock(mock.m);
            mock.body = json::parse_or_null(req.body);
            mock.headers = req.headers;
            return mock.mode;
        };
        relay_server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string mode = record(req);
            if (mode == "unauthorized")
            {
                // A gateway that echoes the header of the authentication in its refusal.
                json refusal = json::object(), inner = json::object();
                inner["message"] = "invalid key: " + req.get_header_value("Authorization");
                refusal["error"] = inner;
                res.status = 401;
                res.set_content(refusal.dump(), "application/json");
                return;
            }
            if (mode == "credit")
            {
                res.status = 402;
                res.set_content("{\"error\": {\"message\": \"insufficient credits\"}}", "application/json");
                return;
            }
            if (mode == "whole")
            {
                res.set_content("{\"choices\": [{\"message\": {\"content\": \"Xoydoeh Nuggwdfsorn habite Qbspm.\"}, "
                    "\"finish_reason\": \"stop\"}], \"usage\": {\"prompt_tokens\": 5, \"completion_tokens\": 6}}",
                    "application/json");
                return;
            }
            if (mode == "page")
            {
                res.set_content("<html><body>Sign in</body></html>", "text/html");
                return;
            }
            std::string sse;
            for (const char* r : { "Xoydoeh vit \xC3\xA0 Qbs", "pm." })
            {
                json d = json::object(), c = json::object(), e = json::object();
                d["reasoning"] = r;
                c["delta"] = d;
                e["choices"] = json::array();
                e["choices"].push_back(c);
                sse += event(e);
            }
            for (const char* t : { "Xoydoeh Nugg", "wdfsorn habite Qbspm." })
            {
                json d = json::object(), c = json::object(), e = json::object();
                d["content"] = t;
                c["delta"] = d;
                e["choices"] = json::array();
                e["choices"].push_back(c);
                sse += event(e);
            }
            json c = json::object(), e = json::object(), u = json::object(), last = json::object();
            c["delta"] = json::object();
            c["finish_reason"] = "stop";
            e["choices"] = json::array();
            e["choices"].push_back(c);
            sse += event(e);
            u["prompt_tokens"] = 12;
            u["completion_tokens"] = 7;
            last["choices"] = json::array();
            last["usage"] = u;
            sse += event(last) + "data: [DONE]\n\n";
            res.set_content(sse, "text/event-stream");
        });
        relay_server.Post("/v1/messages", [&](const httplib::Request& req, httplib::Response& res) {
            record(req);
            // A search of the web: its query in parts, its results, then a text that cites one of them.
            if (json::parse_or_null(req.body)["tools"].is_array())
            {
                const char* const searched[] = {
                    "{\"type\": \"message_start\", \"message\": {\"usage\": {\"input_tokens\": 30}}}",
                    "{\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": "
                        "\"server_tool_use\", \"id\": \"srvtoolu_1\", \"name\": \"web_search\"}}",
                    "{\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"input_json_delta\", "
                        "\"partial_json\": \"{\\\"query\\\": \\\"capitale\"}}",
                    "{\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"input_json_delta\", "
                        "\"partial_json\": \" Italie\\\"}\"}}",
                    "{\"type\": \"content_block_stop\", \"index\": 0}",
                    "{\"type\": \"content_block_start\", \"index\": 1, \"content_block\": {\"type\": "
                        "\"web_search_tool_result\", \"tool_use_id\": \"srvtoolu_1\", \"content\": [{\"type\": "
                        "\"web_search_result\", \"title\": \"Rome\", \"url\": "
                        "\"https://fr.wikipedia.org/wiki/Rome\"}]}}",
                    "{\"type\": \"content_block_stop\", \"index\": 1}",
                    "{\"type\": \"content_block_start\", \"index\": 2, \"content_block\": {\"type\": \"text\", "
                        "\"text\": \"\"}}",
                    "{\"type\": \"content_block_delta\", \"index\": 2, \"delta\": {\"type\": \"citations_delta\", "
                        "\"citation\": {\"type\": \"web_search_result_location\", \"url\": "
                        "\"https://www.fr.wikipedia.org/wiki/Rome_(ville)\", \"title\": \"Rome\"}}}",
                    "{\"type\": \"content_block_delta\", \"index\": 2, \"delta\": {\"type\": \"text_delta\", "
                        "\"text\": \"Rome est la capitale.\"}}",
                    "{\"type\": \"content_block_stop\", \"index\": 2}",
                    "{\"type\": \"message_delta\", \"delta\": {\"stop_reason\": \"end_turn\"}, \"usage\": "
                        "{\"output_tokens\": 12, \"server_tool_use\": {\"web_search_requests\": 1}}}",
                    "{\"type\": \"message_stop\"}" };
                std::string sse;
                for (const char* e : searched)
                    sse += "event: " + json::parse_or_null(e)["type"].as_string() + "\ndata: " + e + "\n\n";
                res.set_content(sse, "text/event-stream");
                return;
            }
            const char* const events[] = {
                "{\"type\": \"message_start\", \"message\": {\"usage\": {\"input_tokens\": 20, "
                    "\"cache_read_input_tokens\": 5}}}",
                "{\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": \"thinking\"}}",
                "{\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"thinking_delta\", "
                    "\"thinking\": \"Let me think.\"}}",
                "{\"type\": \"content_block_delta\", \"index\": 0, \"delta\": {\"type\": \"signature_delta\", "
                    "\"signature\": \"abc\"}}",
                "{\"type\": \"content_block_stop\", \"index\": 0}",
                "{\"type\": \"content_block_delta\", \"index\": 1, \"delta\": {\"type\": \"text_delta\", "
                    "\"text\": \"Bonjour\"}}",
                "{\"type\": \"content_block_delta\", \"index\": 1, \"delta\": {\"type\": \"text_delta\", "
                    "\"text\": \" !\"}}",
                "{\"type\": \"message_delta\", \"delta\": {\"stop_reason\": \"end_turn\"}, \"usage\": "
                    "{\"output_tokens\": 9}}",
                "{\"type\": \"message_stop\"}" };
            std::string sse;
            for (const char* e : events)
                sse += "event: " + json::parse_or_null(e)["type"].as_string() + "\ndata: " + e + "\n\n";
            res.set_content(sse, "text/event-stream");
        });
        relay_server.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
            if (req.get_header_value("Authorization") != "Bearer sk-test-secret")
            {
                res.status = 401;
                res.set_content("{\"error\": \"unauthorized\"}", "application/json");
                return;
            }
            res.set_content("{\"data\": [{\"id\": \"m1\"}, {\"id\": \"m2\"}]}", "application/json");
        });
        const int relay_port = port + 7;
        const bool relay_listening = relay_server.bind_to_port("127.0.0.1", relay_port);
        std::thread relay_listener([&]() { if (relay_listening) relay_server.listen_after_bind(); });
        const std::string origin = "http://127.0.0.1:" + std::to_string(relay_port);
        check(relay_listening, "the server of the relay listens on the port " + std::to_string(relay_port));
        albireo::pseudalis_client::endpoint_for_tests(origin);
        albireo::relay_origin_for_tests("claude", origin);
        albireo::relay_origin_for_tests("ollama", origin);

        // The memory of the pseudonymization: a text served again without Pseudalis, its values forgotten.
        {
            albireo::pseudalis_client client;
            albireo::pseudonym_memory::limits limits;
            limits.values = 3;
            limits.max_idle = 100;
            albireo::pseudonym_memory memory(limits);
            memory.set_time_for_tests(1000);
            std::map<std::string, std::string> found;
            std::string out, error;
            const bool first = memory.protect("Jean Dupont habite Paris.", false, client, nullptr, out, found, error);
            const int calls = mock.anonymized;
            std::map<std::string, std::string> again;
            std::string out2;
            const bool second = memory.protect("Jean Dupont habite Paris.", false, client, nullptr, out2, again, error);
            check(first && out == "Xoydoeh Nuggwdfsorn habite Qbspm." && found["Xoydoeh Nuggwdfsorn"] == "Jean Dupont"
                && found["Qbspm"] == "Paris" && second && out2 == out && again == found && mock.anonymized == calls,
                "a text is protected once, its substitutes known with their values, and served again from memory ("
                + out + ", " + error + ")");
            memory.protect("Marie Curie habite Lyon.", false, client, nullptr, out, found, error);
            const bool bounded = memory.stats()["values"].as_int() == 3;
            memory.protect("Jean Dupont habite Paris.", false, client, nullptr, out, found, error);
            const bool again_asked = mock.anonymized == calls + 2;
            memory.set_time_for_tests(1000 + 200);
            const bool idle = memory.stats()["values"].as_int() == 0 && memory.stats()["texts"].as_int() == 0;
            check(bounded && again_asked && idle, "the memory keeps its bounds, protects again a text whose values "
                "went, and forgets what stays unused");
        }

        // The relay with each provider.
        struct answer
        {
            std::mutex m;
            std::condition_variable cv;
            std::string content, reasoning, done, error;
            bool finished = false;
        };
        std::mutex settings_m;
        albireo::host_settings hs;
        int described = 0;
        albireo::relay_service relay(
            [&]() {
                std::lock_guard<std::mutex> lock(settings_m);
                return hs;
            },
            [&](const std::string&, const std::atomic<bool>&, std::string& description, std::string&) {
                ++described;
                description = "Une photo de Jean Dupont devant la tour Eiffel.";
                return true;
            });
        auto ask = [&](const json& request, std::shared_ptr<answer>& a, int& status, std::string& code) {
            a = std::make_shared<answer>();
            const std::shared_ptr<answer> target = a;
            uint64_t id = 0;
            std::string error;
            if (!relay.submit(request, [target](int kind, const std::string& data) {
                    std::lock_guard<std::mutex> lock(target->m);
                    if (kind == ALBIREO_EVENT_CONTENT) target->content += data;
                    if (kind == ALBIREO_EVENT_REASONING) target->reasoning += data;
                    if (kind == ALBIREO_EVENT_DONE) target->done = data;
                    if (kind == ALBIREO_EVENT_ERROR) target->error = data;
                    if (kind == ALBIREO_EVENT_DONE || kind == ALBIREO_EVENT_ERROR)
                    {
                        target->finished = true;
                        target->cv.notify_all();
                    }
                }, id, error, status, code))
                return false;
            std::unique_lock<std::mutex> lock(a->m);
            return a->cv.wait_for(lock, std::chrono::seconds(30), [&]() { return a->finished; });
        };
        auto set_relay = [&](const std::function<void(albireo::relay_settings&)>& change) {
            std::lock_guard<std::mutex> lock(settings_m);
            hs.relay = albireo::relay_settings();
            hs.relay.enabled = true;
            change(hs.relay);
        };
        const json picture = json::parse_or_null("{\"type\": \"image_url\", \"image_url\": {\"url\": "
            "\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==\"}}");

        // An engine of the interface of OpenAI, the texts pseudonymized, the image described here.
        set_relay([&](albireo::relay_settings& r) {
            r.provider = "openai";
            r.url = origin + "/v1/";
            r.model = "local-test";
            r.secret = "sk-test-secret";
            r.pseudonymize = true;
        });
        json request = json::object();
        request["model"] = albireo::relay_model_id;
        request["stream"] = true;
        request["messages"] = json::parse_or_null("[{\"role\": \"system\", \"content\": \"Tu aides Jean Dupont.\"}, "
            "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": "
            "\"O\u00f9 habite Jean Dupont, Paris ?\"}]}]");
        request["messages"].at(1)["content"].push_back(picture);
        request["stop"] = "\nJean Dupont:";
        request["temperature"] = "Jean Dupont";
        request["top_p"] = 0.9;
        std::shared_ptr<answer> a;
        int status = 0;
        std::string code;
        bool answered = ask(request, a, status, code);
        std::string sent;
        std::string authorization;
        {
            std::lock_guard<std::mutex> lock(mock.m);
            sent = mock.body.dump();
            const auto h = mock.headers.find("Authorization");
            authorization = h == mock.headers.end() ? std::string() : h->second;
        }
        check(answered && a->error.empty() && a->content == "Jean Dupont habite Paris."
            && a->reasoning == "Jean vit \xC3\xA0 Paris."
            && json::parse_or_null(a->done)["usage"]["prompt_tokens"].as_int() == 12,
            "the answer of an engine of the interface of OpenAI streams back with its values (" + a->content + " | "
            + a->reasoning + " | " + a->error + ")");
        check(sent.find("Jean") == std::string::npos && sent.find("Paris") == std::string::npos
            && sent.find("\"stop\"") == std::string::npos && sent.find("\"temperature\"") == std::string::npos
            && sent.find("\"top_p\":0.9") != std::string::npos
            && sent.find("Xoydoeh Nuggwdfsorn") != std::string::npos
            && sent.find("image_url") == std::string::npos && sent.find("[Picture 1, described") != std::string::npos
            && described == 1 && authorization == "Bearer sk-test-secret" && sent.find("sk-test") == std::string::npos,
            "the texts, the description of the image and no stop sequence leave pseudonymized, the sampling as "
            "numbers alone, the key in its header alone");
        ask(request, a, status, code);
        check(described == 1 && a->content == "Jean Dupont habite Paris.", "the description of an image serves again");

        // A refused key: the error names neither the key nor the address.
        {
            std::lock_guard<std::mutex> lock(mock.m);
            mock.mode = "unauthorized";
        }
        answered = ask(request, a, status, code);
        check(answered && json::parse_or_null(a->error)["code"].as_string() == "relay_refused"
            && a->error.find("refuses the key") != std::string::npos && a->error.find("sk-test") == std::string::npos
            && a->error.find("127.0.0.1") == std::string::npos, "a refused key ends the answer with its reason, the "
            "key that the engine echoes masked (" + a->error + ")");
        // An answer sent whole despite the stream, then a web page in place of an answer.
        {
            std::lock_guard<std::mutex> lock(mock.m);
            mock.mode = "whole";
        }
        const bool whole = ask(request, a, status, code) && a->content == "Jean Dupont habite Paris."
            && json::parse_or_null(a->done)["usage"]["completion_tokens"].as_int() == 6;
        {
            std::lock_guard<std::mutex> lock(mock.m);
            mock.mode = "page";
        }
        const bool page = ask(request, a, status, code)
            && json::parse_or_null(a->error)["code"].as_string() == "relay_failed";
        check(whole && page, "an answer sent whole comes back with its values, a web page ends the answer with an "
            "error");
        {
            std::lock_guard<std::mutex> lock(mock.m);
            mock.mode = "ok";
        }
        albireo::relay_settings tested = hs.relay;
        const json listed = relay.test(tested);
        tested.secret = "wrong";
        const json refused_test = relay.test(tested);
        check(listed["ok"].as_bool(false) && listed["models"].size() == 2 && !refused_test["ok"].as_bool(true),
            "the test of the relay lists the models of the engine, and fails with a wrong key");

        // An account without credit, then the substitutes left in the answer, a check of what left.
        {
            std::lock_guard<std::mutex> lock(mock.m);
            mock.mode = "credit";
        }
        const bool credit = ask(request, a, status, code)
            && json::parse_or_null(a->error)["message"].as_string().find("no credit left") != std::string::npos
            && a->error.find("insufficient credits") != std::string::npos;
        {
            std::lock_guard<std::mutex> lock(mock.m);
            mock.mode = "ok";
        }
        set_relay([&](albireo::relay_settings& r) {
            r.provider = "openai";
            r.url = origin + "/v1/";
            r.model = "local-test";
            r.secret = "sk-test-secret";
            r.pseudonymize = true;
            r.keep_substitutes = true;
        });
        const bool kept = ask(request, a, status, code) && a->content == "Xoydoeh Nuggwdfsorn habite Qbspm."
            && a->reasoning == "Xoydoeh vit \xC3\xA0 Qbspm."
            && albireo::relay_description(hs.relay)["keep_substitutes"].as_bool(false);
        check(credit && kept, "an account without credit ends the answer with its reason, and the check of what "
            "left keeps the substitutes in the answer (" + a->content + ")");

        // An address without /v1, which the relay completes, one with the path of an operation, and their failures.
        set_relay([&](albireo::relay_settings& r) {
            r.provider = "openai";
            r.url = origin;
            r.model = "local-test";
            r.secret = "sk-test-secret";
        });
        json plain = json::object();
        plain["model"] = albireo::relay_model_id;
        plain["stream"] = true;
        plain["messages"] = json::parse_or_null("[{\"role\": \"user\", \"content\": \"Bonjour\"}]");
        const bool completed = ask(plain, a, status, code) && a->error.empty()
            && a->content == "Xoydoeh Nuggwdfsorn habite Qbspm.";
        albireo::relay_settings bare = hs.relay;
        const json found = relay.test(bare);
        bare.url = origin + "/v1/chat/completions";
        const json pasted = relay.test(bare);
        bare.url = origin + "/nothing";
        const json nowhere = relay.test(bare);
        bare.url = origin + "/v1";
        bare.secret = "wrong";
        const json refused_key = relay.test(bare);
        check(completed && found["ok"].as_bool(false) && found["url"].as_string() == origin + "/v1"
            && pasted["ok"].as_bool(false) && pasted["url"].as_string() == origin + "/v1"
            && !nowhere["ok"].as_bool(true) && nowhere["code"].as_string() == "relay_http_address"
            && nowhere["args"]["s"].as_string() == "404" && refused_key["code"].as_string() == "relay_http_key"
            && refused_key["args"]["s"].as_string() == "401" && refused_key["detail"].as_string() == "unauthorized"
            && !refused_key.contains("url"),
            "an address without /v1 or with the path of an operation reaches the interface at /v1, and the failures "
            "of the test give their code, their status and the message of the engine (" + nowhere.dump() + ")");
        // A user, a query or a fragment beside the host is refused; an engine out of reach is named by its code.
        bare.url = "http://evil.example?.localhost/v1";
        const json hidden = relay.test(bare);
        bare.url = "http://user@127.0.0.1:" + std::to_string(relay_port) + "/v1";
        const json with_user = relay.test(bare);
        bare.url = "http://127.0.0.1:1/v1";
        const json closed = relay.test(bare);
        check(hidden["code"].as_string() == "relay_address" && with_user["code"].as_string() == "relay_address"
            && albireo::valid_relay_url("http://gpu_box.lan:8000/v1") && albireo::valid_relay_url("http://[::1]:8080")
            && !albireo::valid_relay_url("http://host:70000/v1") && !albireo::valid_relay_url("http://host#x/v1")
            && closed["code"].as_string() == "relay_unreachable",
            "an address whose host carries a user, a query or a fragment is refused, and an engine out of reach gives "
            "its code (" + closed.dump() + ")");

        // Claude: the system apart, the reasoning summarized with its effort, no temperature, the image as is.
        set_relay([&](albireo::relay_settings& r) {
            r.provider = "claude";
            r.model = "claude-opus-5-5";
            r.secret = key;
        });
        json claude = json::object();
        claude["model"] = albireo::relay_model_id;
        claude["temperature"] = 0.7;
        claude["reasoning_effort"] = "high";
        claude["messages"] = json::parse_or_null("[{\"role\": \"system\", \"content\": \"Be brief.\"}, "
            "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Hello\"}]}]");
        claude["messages"].at(1)["content"].push_back(picture);
        answered = ask(claude, a, status, code);
        json body;
        std::string api_key;
        {
            std::lock_guard<std::mutex> lock(mock.m);
            body = mock.body;
            const auto h = mock.headers.find("x-api-key");
            api_key = h == mock.headers.end() ? std::string() : h->second;
        }
        const json done = json::parse_or_null(a->done);
        check(answered && a->content == "Bonjour !" && a->reasoning == "Let me think."
            && done["usage"]["prompt_tokens"].as_int() == 25 && done["usage"]["completion_tokens"].as_int() == 9
            && done["finish_reason"].as_string() == "stop", "the answer of Claude streams back with its reasoning and "
            "its usage (" + a->content + " | " + a->error + ")");
        check(body["system"].as_string() == "Be brief." && body["thinking"]["type"].as_string() == "adaptive"
            && body["thinking"]["display"].as_string() == "summarized"
            && body["output_config"]["effort"].as_string() == "high" && !body.contains("temperature")
            && static_cast<const json&>(body)["messages"][0]["content"][1]["source"]["media_type"].as_string()
                == "image/png"
            && body["max_tokens"].as_int() == 64000 && api_key == key,
            "the request of Claude has its system, its reasoning, its effort, its image and 64000 tokens of answer "
            "without a bound of the request, no temperature, the key in x-api-key");

        // The search of the web of Claude: its tool, its query with the reasoning, its source after the text.
        json searching = claude;
        searching["web_search"] = true;
        answered = ask(searching, a, status, code);
        {
            std::lock_guard<std::mutex> lock(mock.m);
            body = mock.body;
        }
        const json searched_done = json::parse_or_null(a->done);
        const json tool = static_cast<const json&>(body)["tools"][0];
        const std::string searched_text = a->content + " | " + a->reasoning;
        const bool searched = answered && tool["type"].as_string() == "web_search_20250305"
            && tool["max_uses"].as_int() == 5
            && a->content
                == "Rome est la capitale. ([fr.wikipedia.org](https://www.fr.wikipedia.org/wiki/Rome_%28ville%29))"
            && a->reasoning.find("web_search: capitale Italie") != std::string::npos
            && searched_done["usage"]["web_search_requests"].as_int() == 1;
        {
            std::lock_guard<std::mutex> lock(settings_m);
            hs.web_access = false;
        }
        const bool web_off = !ask(searching, a, status, code) && status == 403 && code == "web_access_off";
        {
            std::lock_guard<std::mutex> lock(settings_m);
            hs.web_access = true;
        }
        // Pseudalis replaces the names before they leave: the searches would look for its substitutes.
        set_relay([&](albireo::relay_settings& r) {
            r.provider = "claude";
            r.model = "claude-opus-5-5";
            r.secret = key;
            r.pseudonymize = true;
        });
        const bool pseudonymized = !ask(searching, a, status, code) && status == 400
            && code == "web_search_pseudonymized";
        check(searched && web_off && pseudonymized, "Claude searches the web when the request asks for it, its query "
            "shown with the reasoning and its source as a link after the text, and the web access turned off or the "
            "pseudonymization of the relay refuses it (" + searched_text + ")");

        // Ollama: the reasoning of a model that turns it on or off, the image as is.
        set_relay([&](albireo::relay_settings& r) {
            r.provider = "ollama";
            r.model = "gemma4:31b";
            r.secret = "sk-test-secret";
        });
        json ollama = request;
        ollama["reasoning"] = false;
        answered = ask(ollama, a, status, code);
        {
            std::lock_guard<std::mutex> lock(mock.m);
            body = mock.body;
        }
        check(answered && body["reasoning_effort"].as_string() == "none" && body["model"].as_string() == "gemma4:31b"
            && body.dump().find("image_url") != std::string::npos
            && body.dump().find("Jean Dupont") != std::string::npos,
            "Ollama receives the reasoning turned off and the image itself without pseudonymization");
        json ollama_web = ollama;
        ollama_web["web_search"] = true;
        check(!ask(ollama_web, a, status, code) && status == 400 && code == "web_search_not_supported",
            "the search of the web of the relay takes Claude alone");

        // The refusals before any request.
        json tools = request;
        tools["tools"] = json::parse_or_null("[{\"type\": \"function\"}]");
        const bool no_tools = !ask(tools, a, status, code) && status == 400 && code == "tools_not_supported";
        {
            std::lock_guard<std::mutex> lock(settings_m);
            hs.offline = true;
        }
        const bool offline = !ask(request, a, status, code) && status == 503 && code == "offline";
        set_relay([&](albireo::relay_settings& r) {
            r.provider = "claude";
            r.model = "claude-opus-5-5";
        });
        const bool no_key = !ask(request, a, status, code) && status == 404 && code == "relay_unavailable";
        check(no_tools && offline && no_key, "the relay refuses the tools, the offline mode and a missing key");

        /*
            Offline, an engine of this computer answers, reached directly whatever the cancellation
            of the downloads, which the offline mode holds, Pseudalis included; the providers of the
            Internet and a public address are refused before any connection.
        */
        albireo::http_cancel_all("offline mode");
        {
            // The names of the offline relay resolve whatever the cancellation of the downloads.
            std::vector<std::string> local_addresses, download_addresses;
            std::string local_error, download_error;
            const bool local = albireo::http_resolve_local("localhost", local_addresses, local_error);
            const bool download = albireo::http_resolve("localhost", download_addresses, download_error);
            check(local && !local_addresses.empty() && !download,
                "offline, the names of the relay resolve while the downloads are cancelled (" + local_error + " | "
                + download_error + ")");
        }
        set_relay([&](albireo::relay_settings& r) {
            r.provider = "openai";
            r.url = origin + "/v1";
            r.model = "local-test";
            r.secret = "sk-test-secret";
            r.pseudonymize = true;
        });
        const bool local = ask(request, a, status, code) && a->error.empty()
            && a->content == "Jean Dupont habite Paris.";
        const std::string local_error = a ? a->error : std::string();
        std::string why;
        albireo::relay_settings remote = hs.relay;
        remote.url = "http://93.184.215.14/v1";
        const bool remote_refused = !albireo::relay_offline_ready(remote, why)
            && why.find("neither") != std::string::npos;
        const json remote_test = relay.test(remote);
        remote.url = "http://[2606:4700:4700::1111]/v1";
        const bool remote6_refused = !albireo::relay_offline_ready(remote, why);
        albireo::relay_settings loopback6 = hs.relay;
        loopback6.url = "http://[::1]:8080/v1";
        albireo::relay_settings cloud = hs.relay;
        cloud.provider = "claude";
        cloud.model = "claude-opus-5-5";
        cloud.secret = key;
        const bool cloud_refused = !albireo::relay_offline_ready(cloud, why)
            && why.find("Internet") != std::string::npos;
        const json cloud_test = relay.test(cloud);
        albireo::relay_settings named = hs.relay;
        named.url = "http://localhost:" + std::to_string(relay_port) + "/v1";
        const json named_test = relay.test(named);
        /*
            A name of the local network leads the request to the address checked, a name the
            resolver of the system does not know here; a name with a public address among its
            own is refused.
        */
        albireo::relay_name_for_tests("engine.albireo-test", { "127.0.0.1" });
        albireo::relay_name_for_tests("mixed.albireo-test", { "10.0.0.5", "93.184.215.14" });
        albireo::relay_settings pinned = hs.relay;
        pinned.url = "http://Engine.Albireo-Test:" + std::to_string(relay_port) + "/v1";
        const json pinned_test = relay.test(pinned);
        albireo::relay_settings mixed = hs.relay;
        mixed.url = "http://mixed.albireo-test:" + std::to_string(relay_port) + "/v1";
        const json mixed_test = relay.test(mixed);
        albireo::relay_name_for_tests("engine.albireo-test", {});
        albireo::relay_name_for_tests("mixed.albireo-test", {});
        // A name under .localhost reaches 127.0.0.1 whatever the resolver; the local network alone stays in reach.
        albireo::relay_settings dotted = hs.relay;
        dotted.url = "http://engine.localhost:" + std::to_string(relay_port) + "/v1";
        const json dotted_test = relay.test(dotted);
        albireo::relay_settings nearby = hs.relay;
        nearby.url = "http://100.64.1.1:8080/v1";
        const bool shared_refused = !albireo::relay_offline_ready(nearby, why);
        nearby.url = "http://[64:ff9b:1::1]:8080/v1";
        const bool translated_refused = !albireo::relay_offline_ready(nearby, why);
        nearby.url = "http://192.168.1.20:8080/v1";
        const bool lan_ready = albireo::relay_offline_ready(nearby, why);
        nearby.url = "http://[fd00::1]:8080/v1";
        const bool unique_local_ready = albireo::relay_offline_ready(nearby, why);
        check(pinned_test["ok"].as_bool(false) && mixed_test["code"].as_string() == "relay_offline_remote"
            && dotted_test["ok"].as_bool(false) && shared_refused && translated_refused && lan_ready
            && unique_local_ready,
            "offline, a name of the local network leads the request to the address checked, a name under .localhost "
            "to 127.0.0.1, and the addresses outside the local network are refused (" + pinned_test.dump() + " | "
            + mixed_test.dump() + " | " + dotted_test.dump() + ")");
        albireo::http_restore();
        check(local && remote_refused && remote6_refused && remote_test["code"].as_string() == "relay_offline_remote"
            && albireo::relay_offline_ready(loopback6, why) && cloud_refused
            && cloud_test["code"].as_string() == "relay_offline" && named_test["ok"].as_bool(false),
            "offline, the relay answers through an engine of this computer and Pseudalis whatever the cancellation of "
            "the downloads, and refuses the providers of the Internet and a public address before any connection ("
            + local_error + " | " + named_test.dump() + ")");
        bool leaked = false;
        for (const auto& line : albireo::log_file::instance().recent())
            leaked = leaked || line.find("sk-test") != std::string::npos || line.find(key) != std::string::npos
                || line.find("Jean Dupont") != std::string::npos;
        check(!leaked, "the log names no key and no value of the discussions");
        relay.stop();
        albireo::pseudalis_client::endpoint_for_tests("");
        // The state of Pseudalis names by a code why it is not reached, whether this computer holds it or not.
        {
            albireo::pseudalis_client client;
            const json s = client.status();
            const std::string code = s["code"].as_string();
            const bool coherent = s["found"].as_bool(false) ? code.empty() && s["reason"].as_string().empty()
                : (code == "not_installed" || code == "no_authority" || code == "unprotected" || code == "failed")
                    && !s["reason"].as_string().empty();
            check(coherent, "the state of Pseudalis names by a code why it is not reached (" + s.dump() + ")");
        }
        albireo::relay_origin_for_tests("claude", "");
        albireo::relay_origin_for_tests("ollama", "");
        relay_server.stop();
        relay_listener.join();
    }

    contents_server.stop();
    contents_listener.join();
    server.stop();
    listener.join();
    if (failures == 0) albireo::remove_tree(work);
    std::printf(failures ? "%d checks FAILED\n" : "all checks passed\n", failures);
    return failures ? 1 : 0;
}
