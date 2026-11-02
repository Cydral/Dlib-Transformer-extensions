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
    no temporary file.  The downloads come from the publication space alone, whose link of the
    tests answers through a local interface of pCloud: its listing and addresses are kept, every
    model comes from its copy in the space while no official address of the catalog receives a
    request, and a copy of another size, a file the space lacks and a space that is not set fail
    with a message that names the file in the space; the catalog and the feed of the updates come
    from the space alone too.  Each failure of the space fails its operation with its message,
    lets no exception out, leaves the local server answering, and names neither pCloud, nor the
    link, nor its code, nor an address: a web page with 200 and with 403 and a proxy asking for
    its authentication, from each method of the interface and from a content server, a
    connection reset or refused, a JSON cut short or of other types, sizes of another type,
    negative, beyond the integers or above the file read, and a server that stalls or sends its
    answer byte by byte beyond the bounds the tests shorten.  The log passes its lines, and its
    files of an earlier version, through the last guard (common/log.h), as the errors of the
    routes.  The settings of an earlier version (public_link, the providers and the key of the
    search) are read without error and leave settings.json at the start.  The test works in a
    data folder of its own (set_data_dir()), never in the one of an installed service.

    The web access reads public addresses only: the addresses on both sides of each range that
    is not public, the addresses refused before any connection (user information, schemes, names
    of a single label, the names and the numeric forms of the loopback), then, the loopback
    allowed for the tests, the pages of the local server: a page, a text cut at 2 MB, a type
    refused, the PDF files above 20 MB refused, a redirection to a private address refused.  The
    search of Wikipedia answers from the same server with its documents.  Both routes refuse to
    work offline or with the web access turned off, and a search without provider is refused.
*/

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
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

#include <dlib/external/httplib/httplib.h>

#include "../common/builtin_models.h"
#include "../common/log.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "../host/crypto.h"
#include "../host/http_fetch.h"
#include "../host/models.h"
#include "../host/public_link.h"
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

    // Whether text holds word, its ASCII letters compared whatever their case (word in lowercase).
    bool holds_folded(const std::string& text, const std::string& word)
    {
        std::string folded = text;
        for (auto& c : folded) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return folded.find(word) != std::string::npos;
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
        The publication space of the tests: the interface of pCloud on the local server
        (showpublink, getpublinkdownload) and the files of its folder, each served at
        /content/<identifier>: dist/feed.json without its signature, the copies of the models the
        downloads take, models/linked/gone.bin, and models/short/other.bin of another size than
        the file of the catalog.  The official addresses of the catalogs lie under /official/,
        which counts the requests it receives and answers 404: the downloads never send one.
        protected_link makes the interface answer as a link whose password is refused, and the
        faults make a method or the content server answer as a proxy, a filter of the network or
        a damaged server would; reset_port is the server that resets its connections.
    */
    const std::string feed_copy = "{\"feed\": 1}";
    const std::string linked = blob(250000, 'l');
    const std::string short_copy = blob(1000, 'w');
    const std::map<std::string, const std::string*> link_content = {
        {"11", &feed_copy}, {"21", &linked}, {"31", &short_copy}, {"41", &served}, {"42", &other},
        {"43", &newer_tiny}
    };
    std::atomic<int> listings{0}, addresses{0}, password_mismatch{0}, official_requests{0};
    std::atomic<bool> protected_link{false};
    enum fault
    {
        no_fault, html_200, html_403, proxy_407, truncated_json, stall, trickle, metadata_array, result_string,
        size_string, size_negative, size_beyond, size_huge, hosts_object, reset
    };
    std::atomic<int> listing_fault{no_fault}, address_fault{no_fault}, content_fault{no_fault};
    std::atomic<int> reset_port{0};
    // The answer of a fault that any method gives; false for the faults of the documents.
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
        case proxy_407:
            res.status = 407;
            res.set_header("Proxy-Authenticate", "Basic realm=\"filter\"");
            res.set_content(blocked, "text/html");
            return true;
        case truncated_json:
            res.set_content("{\"result\": 0, \"metadata\": {\"name\": \"Albireo\", \"contents\": [{\"na",
                "application/json");
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
    server.Get("/alive", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("alive", "text/plain");
    });
    server.Get("/showpublink", [&](const httplib::Request& req, httplib::Response& res) {
        if (api_refusal(req, res) || fault_answer(listing_fault, res)) return;
        ++listings;
        auto file = [](const std::string& name, long long id, const json& size) {
            json f = json::object();
            f["name"] = name;
            f["isfolder"] = false;
            f["fileid"] = id;
            f["size"] = size;
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
        json in_tiny = json::array(), in_small = json::array();
        // The size of dist/feed.json: of another type, negative, beyond the integers, or above any file read.
        const int f = listing_fault;
        const json feed_size = f == size_string ? json("11") : f == size_negative ? json(-11)
            : f == size_beyond ? json(1e30) : f == size_huge ? json(9000000000000000000ll)
            : json(static_cast<unsigned long long>(feed_copy.size()));
        dist.push_back(file("feed.json", 11, feed_size));
        in_linked.push_back(file("gone.bin", 21, json(static_cast<unsigned long long>(linked.size()))));
        in_short.push_back(file("other.bin", 31, json(static_cast<unsigned long long>(short_copy.size()))));
        in_tiny.push_back(file("model.bin", 41, json(static_cast<unsigned long long>(served.size()))));
        in_tiny.push_back(file("model-2.bin", 43, json(static_cast<unsigned long long>(newer_tiny.size()))));
        in_small.push_back(file("other.bin", 42, json(static_cast<unsigned long long>(other.size()))));
        models.push_back(folder("linked", in_linked));
        models.push_back(folder("short", in_short));
        models.push_back(folder("tiny", in_tiny));
        models.push_back(folder("small", in_small));
        json root_contents = json::array();
        root_contents.push_back(folder("dist", dist));
        root_contents.push_back(folder("models", models));
        json answer = json::object();
        answer["result"] = f == result_string ? json("0") : json(0);
        answer["metadata"] = f == metadata_array ? json::array() : folder("Albireo", root_contents);
        res.set_content(answer.dump(), "application/json");
    });
    server.Get("/getpublinkdownload", [&](const httplib::Request& req, httplib::Response& res) {
        if (api_refusal(req, res) || fault_answer(address_fault, res)) return;
        ++addresses;
        const std::string id = req.get_param_value("fileid");
        const std::string path = link_content.count(id) ? "/content/" + id : std::string();
        json answer = json::object();
        if (path.empty())
        {
            answer["result"] = 2009;
            answer["error"] = "File not found.";
        }
        else
        {
            // The content server, or the server that resets its connections.
            json hosts = json::array();
            hosts.push_back("127.0.0.1:" + std::to_string(content_fault == reset ? reset_port.load() : port));
            answer["result"] = 0;
            answer["hosts"] = address_fault == hosts_object ? json::object() : hosts;
            answer["path"] = path;
        }
        res.set_content(answer.dump(), "application/json");
    });
    server.Get(R"(/content/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        if (fault_answer(content_fault, res)) return;
        const auto it = link_content.find(req.matches[1].str());
        if (it == link_content.end()) res.status = 404;
        else res.set_content(*it->second, "application/octet-stream");
    });
    server.Get(R"(/official/.*)", [&](const httplib::Request&, httplib::Response& res) {
        ++official_requests;
        res.status = 404;
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
    const std::string base_url = "http://127.0.0.1:" + std::to_string(port);
    // The link of the build sets the publication space; the downloads of the tests come from their own link alone.
    check(albireo::public_link_enabled(), "the link of the build sets the publication space");
    albireo::public_link_for_tests(base_url + "/publink/show?code=TESTLINK1");

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
        // The publication space: the links the tests may give, its listing and addresses kept, the files it lacks.
        const std::string link = base_url + "/publink/show?code=TESTLINK1";
        albireo::public_link_for_tests("https://example.org/publink/show?code=kZeNhSJZpcMygkcC");
        const bool other_site = albireo::public_link_enabled();
        albireo::public_link_for_tests("http://u.pcloud.link/publink/show?code=kZeNhSJZpcMygkcC");
        const bool without_tls = albireo::public_link_enabled();
        albireo::public_link_for_tests("");
        check(!other_site && !without_tls && !albireo::public_link_enabled(),
            "a link of another site, without TLS or empty sets no publication space");
        std::string body, error, url;
        uint64_t size = 0;
        albireo::link_status status = albireo::link_status::ok;
        bool ok = albireo::public_link_file("dist/feed.json", url, size, error, false, &status);
        check(!ok && status == albireo::link_status::unset && error == "dist/feed.json cannot be downloaded: no "
            "publication space is set", "without a space, a file is refused with its name (" + error + ")");
        albireo::public_link_for_tests(link);
        check(albireo::public_link_enabled(), "a local link is used");
        const int listed_before = listings, addressed_before = addresses;
        error.clear();
        ok = albireo::public_link_get("dist/feed.json", body, error, 1u << 20);
        check(ok && body == feed_copy, "a file of the space is read (" + error + ")");
        check(albireo::public_link_get("dist/feed.json", body, error, 1u << 20) && listings - listed_before == 1
            && addresses - addressed_before == 1,
            "the listing and the address of the space are kept for the next reads");
        ok = albireo::public_link_file("dist/catalog.json", url, size, error, false, &status);
        check(!ok && status == albireo::link_status::missing
            && error == "the publication space of Albireo does not hold dist/catalog.json yet",
            "a file the space lacks is reported as not published yet (" + error + ")");
        check(password_mismatch == 0, "the password goes with each request when the program carries one");
        check(albireo::public_link_model_path("tiny", "https://hub.example/a/resolve/abc/Tiny-Q4.gguf?download=1")
            == "models/tiny/Tiny-Q4.gguf", "the copy of a file of a model lies in the folder of its identifier");
        check(albireo::without_queries("https://example.org/search?q=X&key=Y: status 500")
            == "https://example.org/search: status 500", "the queries of the addresses stay out of the messages");

        // The downloads: the copy of the space alone, a copy of another size or a file the space lacks failing.
        albireo::store db(albireo::join_path(work, "albireo-3.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-11-01T00:00:00Z";
        json list = json::array();
        list.push_back(catalog_entry("linked", linked, base_url + "/official/gone.bin"));
        list.push_back(catalog_entry("short", other, base_url + "/official/other.bin"));
        list.push_back(catalog_entry("absent", blob(5000, 'n'), base_url + "/official/absent.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        albireo::model_manager m(db, albireo::join_path(work, "models-3"));
        albireo::user_error started;
        albireo::installed_model im;
        check(m.start_download("linked", started), "the download of a model the space holds starts");
        json d = wait_download(m, "linked");
        check(d["state"].as_string() == "done" && installed(db, "linked", im) && read_text(im.file) == linked,
            "a model comes from its copy in the space (" + d.dump() + ")");
        check(m.start_download("short", started), "the download of a model whose copy has another size starts");
        d = wait_download(m, "short");
        check(d["state"].as_string() == "error" && d["code"].as_string() == "download_failed"
            && d["error"].as_string() == "the copy of models/short/other.bin in the publication space of Albireo does "
            "not have the size the catalog gives" && !installed(db, "short", im),
            "a copy of another size than the catalog fails, naming the file in the space (" + d.dump() + ")");
        check(m.start_download("absent", started), "the download of a model the space lacks starts");
        d = wait_download(m, "absent");
        check(d["state"].as_string() == "error" && d["code"].as_string() == "download_not_published"
            && d["args"]["file"].as_string() == "models/absent/absent.bin"
            && d["error"].as_string() == "the publication space of Albireo does not hold models/absent/absent.bin yet",
            "a model the space lacks fails as not published yet, naming its file (" + d.dump() + ")");
        std::string refreshed;
        check(!m.refresh_catalog(refreshed)
            && refreshed == "the publication space of Albireo does not hold dist/catalog.json yet",
            "a catalog the space lacks fails as not published yet (" + refreshed + ")");
        // Without a space, a download fails at once, naming its file.
        albireo::public_link_for_tests("");
        check(m.start_download("absent", started), "a download starts without a space");
        d = wait_download(m, "absent");
        check(d["state"].as_string() == "error" && d["code"].as_string() == "download_failed"
            && d["error"].as_string() == "models/absent/absent.bin cannot be downloaded: no publication space is set",
            "without a space, a download fails, naming its file (" + d.dump() + ")");
        m.stop();

        // The updates: the feed and its signature come from the space alone; the space holds no signature.
        albireo::public_link_for_tests(link);
        {
            albireo::updater u(albireo::join_path(work, "updates"));
            u.check_async();
            json update;
            for (int i = 0; i < 600; ++i)
            {
                update = u.status();
                const std::string state = update["state"].as_string();
                if (state != "idle" && state != "checking") break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            check(update["state"].as_string() == "error" && update["code"].as_string() == "update_not_published"
                && update["args"]["file"].as_string() == "dist/feed.json.sig"
                && update["error"].as_string() == "the publication space of Albireo does not hold dist/feed.json.sig "
                "yet", "a feed whose signature the space lacks fails as not published yet (" + update.dump() + ")");
        }
        check(official_requests == 0, "no official address of the catalog receives a request ("
            + std::to_string(official_requests) + ")");

        // A link that refuses the password: the message names the answer, never the query.
        protected_link = true;
        albireo::public_link_for_tests(link);
        ok = albireo::public_link_get("dist/feed.json", body, error, 1u << 20);
        check(!ok && error == "the publication space of Albireo answers \"Invalid 'linkpassword' provided.\" (1125) "
            "for dist/feed.json", "a refused password is reported without the query (" + error + ")");
        protected_link = false;
        albireo::public_link_for_tests("");
    }

    {
        /*
            The failures of the publication space, each with the bounds of the waits shortened to 1 s
            and of a whole request to 2 s: the operation fails with its message, no exception leaves it,
            the local server still answers, and the message names neither pCloud, nor the link, nor its
            code, nor an address.
        */
        reset_server resetting;
        check(resetting.start(port + 1), "a server that resets its connections listens on the port "
            + std::to_string(port + 1));
        reset_port = port + 1;
        const std::string link = base_url + "/publink/show?code=TESTLINK1";
        albireo::http_bounds_for_tests(1, 2);
        auto clean = [&](const std::string& text) {
            return !holds_folded(text, "pcloud") && !holds_folded(text, "publink") && text.find("TESTLINK1")
                == std::string::npos && text.find("://") == std::string::npos && text.find("127.0.0.1")
                == std::string::npos;
        };
        auto answers = [&]() {
            std::string alive, e;
            return albireo::http_get(base_url + "/alive", alive, e, 64) && alive == "alive";
        };
        const std::string unreachable = "the publication space of Albireo cannot be reached for dist/feed.json: ";
        const std::string undownloadable = "the copy of dist/feed.json in the publication space of Albireo cannot be "
            "downloaded: ";
        const std::string page_text = "a web page comes in place of the answer, as a proxy or a filter of the network "
            "sends one";
        const std::string other_answer = "the publication space of Albireo gives an answer that is not the one of its "
            "interface for dist/feed.json";
        const std::string invalid_entry = "the publication space of Albireo lists dist/feed.json without a valid "
            "identifier or size";
        struct failure_case
        {
            const char* what;
            std::atomic<int>* where;
            int f;
            std::string expected;    // the message, or its beginning when prefix is true
            bool prefix;
        };
        const failure_case cases[] = {
            {"a listing answered by a web page with 200", &listing_fault, html_200, unreachable + page_text, false},
            {"a listing answered by a web page with 403", &listing_fault, html_403,
                unreachable + "the request is answered with the status 403", false},
            {"a listing answered by a proxy that asks for its authentication", &listing_fault, proxy_407,
                unreachable + "the proxy asks for an authentication (status 407)", false},
            {"a listing cut short", &listing_fault, truncated_json, other_answer, false},
            {"a listing whose folder is an array", &listing_fault, metadata_array, "the publication space of "
                "Albireo gives a listing that is not the one of its interface for dist/feed.json", false},
            {"a listing whose result is a string", &listing_fault, result_string, other_answer, false},
            {"a listing whose size is a string", &listing_fault, size_string, invalid_entry, false},
            {"a listing whose size is negative", &listing_fault, size_negative, invalid_entry, false},
            {"a listing whose size is beyond the integers", &listing_fault, size_beyond, invalid_entry, false},
            {"a listing whose size is above the file read", &listing_fault, size_huge, "the copy of dist/feed.json in "
                "the publication space of Albireo is larger than expected", false},
            {"a listing that stalls", &listing_fault, stall, unreachable + "no data arrives within 1 s", false},
            {"a listing sent byte by byte", &listing_fault, trickle,
                unreachable + "no complete answer arrives within 2 s", false},
            {"an address answered by a web page with 200", &address_fault, html_200, unreachable + page_text, false},
            {"an address answered by a web page with 403", &address_fault, html_403,
                unreachable + "the request is answered with the status 403", false},
            {"an address answered by a proxy that asks for its authentication", &address_fault, proxy_407,
                unreachable + "the proxy asks for an authentication (status 407)", false},
            {"an address cut short", &address_fault, truncated_json, other_answer, false},
            {"an address whose servers are an object", &address_fault, hosts_object, "the publication space of "
                "Albireo gives no address for dist/feed.json", false},
            {"an address that stalls", &address_fault, stall, unreachable + "no data arrives within 1 s", false},
            {"a content server that answers a web page with 200", &content_fault, html_200, undownloadable + page_text,
                false},
            {"a content server that answers a web page with 403", &content_fault, html_403,
                undownloadable + "the request is answered with the status 403", false},
            {"a content server that stalls", &content_fault, stall, undownloadable + "no data arrives within 1 s",
                false},
            {"a content server that sends the file byte by byte", &content_fault, trickle,
                undownloadable + "no complete answer arrives within 2 s", false},
            {"a content server that resets the connection", &content_fault, reset, undownloadable, true}
        };
        for (const auto& c : cases)
        {
            albireo::public_link_for_tests(link);
            *c.where = c.f;
            std::string body, error;
            bool ok = true, escaped = false;
            const auto begun = std::chrono::steady_clock::now();
            try
            {
                ok = albireo::public_link_get("dist/feed.json", body, error, 1u << 20);
            }
            catch (...)
            {
                escaped = true;
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
            *c.where = no_fault;
            const bool told = c.prefix ? error.compare(0, c.expected.size(), c.expected) == 0
                && error.size() > c.expected.size() : error == c.expected;
            check(!ok && !escaped && told && clean(error) && seconds < 10 && answers(), std::string(c.what)
                + " fails with its message, in " + std::to_string(static_cast<int>(seconds)) + " s (" + error + ")");
        }
        // A space whose interface resets its connections, or does not listen.
        for (int to : { port + 1, port + 2 })
        {
            albireo::public_link_for_tests("http://127.0.0.1:" + std::to_string(to) + "/publink/show?code=TESTLINK1");
            std::string body, error;
            bool ok = true, escaped = false;
            try
            {
                ok = albireo::public_link_get("dist/feed.json", body, error, 1u << 20);
            }
            catch (...)
            {
                escaped = true;
            }
            check(!ok && !escaped && error.compare(0, unreachable.size(), unreachable) == 0 && clean(error)
                && answers(), std::string(to == port + 1 ? "an interface that resets its connections"
                : "an interface that does not listen") + " fails with its message (" + error + ")");
        }
        albireo::public_link_for_tests(link);

        // The ranges of the downloads: a web page, a stall and a reset fail before any byte reaches the file.
        std::string written, error;
        auto keep = [&](const char* data, size_t n) {
            written.append(data, n);
            return true;
        };
        bool escaped = false;
        content_fault = html_200;
        const bool from_page = albireo::http_get_range(base_url + "/content/41", 0, 999, keep, error);
        const std::string page_error = error;
        content_fault = stall;
        const bool from_stall = albireo::http_get_range(base_url + "/content/41", 0, 999, keep, error);
        const std::string stall_error = error;
        content_fault = no_fault;
        const bool from_reset = albireo::http_get_range("http://127.0.0.1:" + std::to_string(port + 1) + "/content/41",
            1000, 1999, keep, error);
        // A receiver that throws: the exception becomes the error of the download.
        try
        {
            albireo::http_get_range(base_url + "/content/41", 0, 999, [](const char*, size_t) -> bool {
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

        // A model whose copy is a web page, an update whose feed is refused, a catalog from a stalled listing.
        albireo::store db(albireo::join_path(work, "albireo-4.db"));
        json catalog = json::object();
        catalog["updated"] = "2026-11-01T00:00:00Z";
        json list = json::array();
        list.push_back(catalog_entry("tiny", served, base_url + "/official/model.bin"));
        catalog["models"] = list;
        db.set_value("catalog", catalog.dump());
        {
            albireo::model_manager m(db, albireo::join_path(work, "models-4"));
            albireo::user_error started;
            content_fault = html_200;
            check(m.start_download("tiny", started), "the download of a model whose copy is a web page starts");
            const json d = wait_download(m, "tiny");
            content_fault = no_fault;
            const std::string partial = albireo::join_path(m.models_dir(), "tiny.gguf.partial");
            check(d["state"].as_string() == "error" && d["code"].as_string() == "download_failed"
                && d["error"].as_string() == "the copy of models/tiny/model.bin in the publication space of Albireo "
                "cannot be downloaded: " + page_text && clean(d.dump())
                && read_text(partial).find("Access denied") == std::string::npos && answers(),
                "a model whose copy is a web page fails, its partial file free of the page (" + d.dump() + ")");
            listing_fault = stall;
            std::string refreshed;
            const bool refreshed_ok = m.refresh_catalog(refreshed);
            check(!refreshed_ok && refreshed == "the publication space of Albireo cannot be reached for "
                "dist/catalog.json: no data arrives within 1 s" && clean(refreshed) && answers(),
                "a catalog from a listing that stalls fails with its message (" + refreshed + ")");
            listing_fault = no_fault;
            m.stop();
        }
        {
            albireo::public_link_for_tests(link);
            listing_fault = html_403;
            albireo::updater u(albireo::join_path(work, "updates-4"));
            u.check_async();
            json update;
            for (int i = 0; i < 600; ++i)
            {
                update = u.status();
                const std::string state = update["state"].as_string();
                if (state != "idle" && state != "checking") break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            listing_fault = no_fault;
            check(update["state"].as_string() == "error" && update["code"].as_string() == "update_download"
                && update["args"]["e"].as_string() == "the publication space of Albireo cannot be reached for "
                "dist/feed.json: the request is answered with the status 403" && clean(update.dump()) && answers(),
                "a feed refused by a filter of the network fails with its message (" + update.dump() + ")");
        }
        albireo::http_bounds_for_tests(0, 0);
        resetting.stop();

        // The log of these failures names neither pCloud, nor the link, nor its code.
        bool named = false;
        for (const auto& line : albireo::log_file::instance().recent())
            named = named || holds_folded(line, "pcloud") || holds_folded(line, "publink")
                || line.find("TESTLINK1") != std::string::npos;
        check(!named, "the log of the failures names neither pCloud, nor the link, nor its code");
        albireo::public_link_for_tests("");
    }

    {
        /*
            The last guard: the words that name the publication space, whatever their case, and the
            code given to hide_in_messages(), in a text, in the errors of a document, in the log and
            in its files of an earlier version.
        */
        const std::string space = "the publication space of Albireo";
        albireo::hide_in_messages("TESTLINK1");
        albireo::hide_in_messages("abc");
        check(albireo::without_publication_space("public link: https://u.pcloud.link/publink/show?code=kZeNhSJZpcMyg"
            "kcCvGXmq1MTvkCfBuSsx2GV, with the password") == "public link: " + space + ", with the password"
            && albireo::without_publication_space("api.pcloud.com:443: no answer") == space + ": no answer"
            && albireo::without_publication_space("(c123.PCLOUD.com) and \"eapi.pcloud.com\".") == "(" + space
                + ") and \"" + space + "\"."
            && albireo::without_publication_space("the code TESTLINK1 is refused") == "the code " + space
                + " is refused"
            && albireo::without_publication_space("abc and abcdef stay") == "abc and abcdef stay"
            && albireo::without_publication_space("no address here: dist/feed.json") == "no address here: "
                "dist/feed.json", "the guard replaces the words that name the publication space");
        json document = json::object();
        document["error"] = "see https://api.pcloud.com/showpublink";
        json args = json::object();
        args["e"] = "content server c4.pcloud.com";
        document["args"] = args;
        document["text"] = "pcloud";
        json errors = json::array();
        errors.push_back("eapi.pcloud.com");
        document["errors"] = errors;
        const bool named = albireo::errors_name_publication_space(document);
        albireo::hide_publication_space_in_errors(document);
        const json& hidden = document;
        check(named && hidden["error"].as_string() == "see " + space && hidden["args"]["e"].as_string()
            == "content server " + space && hidden["errors"][0].as_string() == space
            && hidden["text"].as_string() == "pcloud" && !albireo::errors_name_publication_space(hidden),
            "the errors of a document leave the publication space unnamed, its other members unchanged");
        const std::string old_log = albireo::join_path(work, "logs/service.log");
        albireo::make_dirs(albireo::join_path(work, "logs"));
        write_text(old_log, "2026-10-30 10:00:00.000 [info] public link: https://u.pcloud.link/publink/show?code="
            "kZeNhSJZpcMygkcCvGXmq1MTvkCfBuSsx2GV, with the password\n2026-10-30 10:00:01.000 [info] kept\n");
        write_text(old_log + ".1", "2026-10-29 09:00:00.000 [warning] updates: https://api.pcloud.com/showpublink: "
            "status 403\n");
        albireo::log_file::instance().open(old_log);
        albireo::log_warning("models: c12.pcloud.com does not answer");
        const std::string now = read_text(old_log), before = read_text(old_log + ".1");
        check(!holds_folded(now, "pcloud") && !holds_folded(before, "pcloud")
            && now.find("[info] kept") != std::string::npos
            && now.find("public link: " + space + ", with the password") != std::string::npos
            && now.find("models: " + space + " does not answer") != std::string::npos,
            "the log and its files of an earlier version leave the publication space unnamed");
    }

    {
        // The settings of an earlier version: read without error, written again without what this version lacks.
        const std::string folder = albireo::join_path(work, "settings");
        albireo::make_dirs(folder);
        const std::string settings_file = albireo::join_path(folder, "settings.json");
        const std::string key_file = albireo::join_path(folder, "web_search.secret");
        write_text(settings_file, "{\"port\": 9123, \"web_search\": \"brave\", \"web_search_url\": "
            "\"https://searx.example.org\", \"web_search_key_set\": true, \"web_search_key\": \"BSA-by-hand-42\", "
            "\"public_link\": \"https://u.pcloud.link/publink/show?code=kZeNhSJZpcMygkcCvGXmq1MTvkCfBuSsx2GV\"}");
        write_text(key_file, "BSA-secret-0123456789");
        {
            albireo::settings_store store(settings_file);
            const std::string text = read_text(settings_file), copy = read_text(settings_file + ".bak");
            check(store.get().port == 9123 && store.get().web_search == "wikipedia" && !holds_folded(text, "pcloud")
                && !holds_folded(copy, "pcloud") && text.find("web_search_url") == std::string::npos
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
                + " (https://github.com/Cydral/Dlib-Transformer-extensions; cydraltechnology@gmail.com)",
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

    server.stop();
    listener.join();
    if (failures == 0) albireo::remove_tree(work);
    std::printf(failures ? "%d checks FAILED\n" : "all checks passed\n", failures);
    return failures ? 1 : 0;
}
