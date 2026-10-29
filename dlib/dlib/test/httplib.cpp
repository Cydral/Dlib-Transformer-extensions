// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#include <dlib/external/httplib/httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef CPPHTTPLIB_MBEDTLS_SUPPORT
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/entropy.h>
#include <mbedtls/pk.h>
#include <mbedtls/x509_crt.h>
#include <psa/crypto.h>
#endif

#include <dlib/sockets.h>

#include "tester.h"

namespace
{
    using namespace test;
    using namespace std;

    dlib::logger dlog("test.httplib");

// ----------------------------------------------------------------------------------------
//                                      Helpers
// ----------------------------------------------------------------------------------------

    /* A server listening on an ephemeral loopback port, stopped and joined when the
       object goes out of scope, so that a failing check never leaves a thread behind. */
    template <typename server_type>
    class running_server
    {
    public:
        explicit running_server(server_type& srv) : srv_(srv)
        {
            port_ = srv_.bind_to_any_port("127.0.0.1");
            DLIB_CASSERT(port_ > 0, "the test server could not bind a loopback port");
            thread_ = std::thread([this]() { srv_.listen_after_bind(); });
            srv_.wait_until_ready();
        }
        ~running_server()
        {
            srv_.stop();
            if (thread_.joinable()) thread_.join();
        }
        int port() const { return port_; }
    private:
        server_type& srv_;
        int port_ = -1;
        std::thread thread_;
    };

    /* Sends raw bytes on a fresh connection and returns everything the server answers
       before closing the connection or staying silent for timeout_ms. */
    std::string raw_exchange(int port, const std::string& request, unsigned long timeout_ms = 2000)
    {
        std::unique_ptr<dlib::connection> con(dlib::connect("127.0.0.1", static_cast<unsigned short>(port)));
        if (!request.empty())
            con->write(request.data(), static_cast<long>(request.size()));
        std::string answer;
        char buf[4096];
        for (;;)
        {
            const long n = con->read(buf, sizeof(buf), timeout_ms);
            if (n <= 0) break;
            answer.append(buf, static_cast<size_t>(n));
        }
        return answer;
    }

    int status_of(const std::string& raw_response)
    {
        if (raw_response.compare(0, 9, "HTTP/1.1 ") != 0 || raw_response.size() < 12) return -1;
        return std::atoi(raw_response.substr(9, 3).c_str());
    }

    bool make_dir(const std::string& path)
    {
#if defined(_WIN32)
        return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
        return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
    }

    void write_file(const std::string& path, const std::string& content)
    {
        std::ofstream out(path, std::ios::binary);
        out << content;
    }

    std::string all_bytes(size_t n)
    {
        std::string s(n, '\0');
        for (size_t i = 0; i < n; ++i) s[i] = static_cast<char>((i * 131 + (i >> 8)) & 0xFF);
        return s;
    }

// ----------------------------------------------------------------------------------------
//                                      Tests
// ----------------------------------------------------------------------------------------

    void test_routes_and_methods()
    {
        print_spinner();
        httplib::Server srv;
        srv.Get("/hello", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("Hello", "text/plain");
            res.set_header("X-Test", "yes");
        });
        srv.Post("/echo", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content(req.body, req.get_header_value("Content-Type"));
        });
        srv.Put("/put", [](const httplib::Request& req, httplib::Response& res) {
            res.status = 201;
            res.set_content("put:" + req.body, "text/plain");
        });
        srv.Patch("/patch", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content("patch:" + req.body, "text/plain");
        });
        srv.Delete("/item", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
        srv.Options("/item", [](const httplib::Request&, httplib::Response& res) {
            res.set_header("Allow", "GET, DELETE, OPTIONS");
        });
        srv.Get("/users/:id/files/:name", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content(req.path_params.at("id") + "|" + req.path_params.at("name"), "text/plain");
        });
        srv.Get(R"(/numbers/(\d+))", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content(req.matches[1].str(), "text/plain");
        });
        srv.Get("/query", [](const httplib::Request& req, httplib::Response& res) {
            std::string out = req.get_param_value("a") + "," + std::to_string(req.get_param_value_count("b"));
            for (size_t i = 0; i < req.get_param_value_count("b"); ++i) out += "," + req.get_param_value("b", i);
            out += "," + req.get_header_value("x-custom");
            res.set_content(out, "text/plain");
        });

        running_server<httplib::Server> run(srv);
        httplib::Client cli("127.0.0.1", run.port());
        cli.set_connection_timeout(5);

        auto r = cli.Get("/hello");
        DLIB_TEST(r && r->status == 200);
        DLIB_TEST(r->body == "Hello");
        DLIB_TEST(r->get_header_value("X-Test") == "yes");
        DLIB_TEST(r->get_header_value("Content-Type") == "text/plain");

        r = cli.Head("/hello");
        DLIB_TEST(r && r->status == 200 && r->body.empty());
        DLIB_TEST(r->get_header_value("Content-Length") == "5");

        const std::string payload = all_bytes(70000);
        r = cli.Post("/echo", payload, "application/octet-stream");
        DLIB_TEST(r && r->status == 200);
        DLIB_TEST(r->body == payload);

        r = cli.Put("/put", "abc", "text/plain");
        DLIB_TEST(r && r->status == 201 && r->body == "put:abc");
        r = cli.Patch("/patch", "xyz", "text/plain");
        DLIB_TEST(r && r->status == 200 && r->body == "patch:xyz");
        r = cli.Delete("/item");
        DLIB_TEST(r && r->status == 204);
        r = cli.Options("/item");
        DLIB_TEST(r && r->get_header_value("Allow") == "GET, DELETE, OPTIONS");

        r = cli.Get("/users/42/files/report%20final.pdf");
        DLIB_TEST(r && r->status == 200);
        DLIB_TEST_MSG(r->body == "42|report final.pdf", r->body);
        r = cli.Get("/numbers/12345");
        DLIB_TEST(r && r->body == "12345");
        r = cli.Get("/numbers/abc");
        DLIB_TEST(r && r->status == 404);

        httplib::Headers headers = { {"X-Custom", "case insensitive"} };
        r = cli.Get("/query?a=%C3%A9t%C3%A9&b=1&b=two&b=3", headers);
        DLIB_TEST(r && r->status == 200);
        DLIB_TEST_MSG(r->body == "\xC3\xA9t\xC3\xA9,3,1,two,3,case insensitive", r->body);

        r = cli.Get("/missing");
        DLIB_TEST(r && r->status == 404);
        r = cli.Post("/hello", "x", "text/plain");
        DLIB_TEST(r && (r->status == 404 || r->status == 405));
    }

// ----------------------------------------------------------------------------------------

    void test_chunked_bodies_and_streaming()
    {
        print_spinner();
        httplib::Server srv;

        // A request body sent with chunked encoding, read whole by the handler.
        srv.Post("/upload", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content(std::to_string(req.body.size()) + ":" + req.get_header_value("Transfer-Encoding"),
                "text/plain");
        });

        // A response produced chunk by chunk.
        srv.Get("/chunks", [](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider("text/plain", [](size_t offset, httplib::DataSink& sink) {
                const size_t index = offset / 6;
                if (index >= 10)
                {
                    sink.done();
                    return true;
                }
                char piece[7];
                std::snprintf(piece, sizeof(piece), "part%02d", static_cast<int>(index));
                sink.write(piece, 6);
                return true;
            });
        });

        // Server-sent events, the second event only once the client confirmed the first,
        // which proves that nothing buffers the stream between the two ends.
        std::mutex m;
        std::condition_variable cv;
        bool first_seen = false;
        srv.Get("/events", [&](const httplib::Request&, httplib::Response& res) {
            res.set_header("Cache-Control", "no-cache");
            auto step = std::make_shared<int>(0);
            res.set_chunked_content_provider("text/event-stream", [&, step](size_t, httplib::DataSink& sink) {
                if (*step == 0)
                {
                    const std::string ev = "data: {\"n\":1}\n\n";
                    sink.write(ev.data(), ev.size());
                    *step = 1;
                    return true;
                }
                if (*step == 1)
                {
                    std::unique_lock<std::mutex> lock(m);
                    cv.wait_for(lock, std::chrono::seconds(5), [&]() { return first_seen; });
                    const std::string ev = first_seen ? "data: {\"n\":2}\n\ndata: [DONE]\n\n" : "data: timeout\n\n";
                    sink.write(ev.data(), ev.size());
                    *step = 2;
                    return true;
                }
                sink.done();
                return true;
            });
        });

        running_server<httplib::Server> run(srv);
        httplib::Client cli("127.0.0.1", run.port());

        // Content provider without a length: the client sends chunks.
        const std::string blob = all_bytes(300000);
        auto r = cli.Post("/upload", [&](size_t offset, httplib::DataSink& sink) {
            const size_t n = std::min<size_t>(65536, blob.size() - offset);
            sink.write(blob.data() + offset, n);
            if (offset + n == blob.size()) sink.done();
            return true;
        }, "application/octet-stream");
        DLIB_TEST(r && r->status == 200);
        DLIB_TEST_MSG(r->body == "300000:chunked", r->body);

        std::string received;
        int calls = 0;
        r = cli.Get("/chunks", [&](const char* data, size_t len) {
            received.append(data, len);
            ++calls;
            return true;
        });
        DLIB_TEST(r && r->status == 200);
        DLIB_TEST(calls >= 1);
        DLIB_TEST(received == "part00part01part02part03part04part05part06part07part08part09");

        std::string stream;
        r = cli.Get("/events", [&](const char* data, size_t len) {
            stream.append(data, len);
            if (stream.find("\"n\":1") != std::string::npos)
            {
                std::lock_guard<std::mutex> lock(m);
                first_seen = true;
                cv.notify_all();
            }
            return true;
        });
        DLIB_TEST(r && r->status == 200);
        DLIB_TEST(r->get_header_value("Content-Type") == "text/event-stream");
        DLIB_TEST_MSG(stream == "data: {\"n\":1}\n\ndata: {\"n\":2}\n\ndata: [DONE]\n\n", stream);

        // A receiver that stops early ends the transfer on the client side.
        size_t got = 0;
        r = cli.Get("/chunks", [&](const char*, size_t len) {
            got += len;
            return false;
        });
        DLIB_TEST(!r);
        DLIB_TEST(r.error() == httplib::Error::Canceled);
        DLIB_TEST(got > 0 && got < 60);
    }

// ----------------------------------------------------------------------------------------

    void test_multipart_and_ranges()
    {
        print_spinner();
        httplib::Server srv;
        srv.Post("/form", [](const httplib::Request& req, httplib::Response& res) {
            std::string out;
            if (req.form.has_field("title")) out += "title=" + req.form.get_field("title") + ";";
            if (req.form.has_file("doc"))
            {
                const auto f = req.form.get_file("doc");
                out += "file=" + f.filename + ";type=" + f.content_type + ";size=" + std::to_string(f.content.size());
                out += f.content == all_bytes(5000) ? ";ok" : ";corrupt";
            }
            res.set_content(out, "text/plain");
        });
        const std::string data = all_bytes(100);
        srv.Get("/data", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content(data, "application/octet-stream");
        });

        running_server<httplib::Server> run(srv);
        httplib::Client cli("127.0.0.1", run.port());

        httplib::UploadFormDataItems items = {
            { "title", "Rapport \xC3\xA9t\xC3\xA9", "", "" },
            { "doc", all_bytes(5000), "notes.bin", "application/octet-stream" },
        };
        auto r = cli.Post("/form", items);
        DLIB_TEST(r && r->status == 200);
        DLIB_TEST_MSG(r->body == "title=Rapport \xC3\xA9t\xC3\xA9;file=notes.bin;type=application/octet-stream;size=5000;ok", r->body);

        r = cli.Get("/data", httplib::Headers{ httplib::make_range_header({ {10, 19} }) });
        DLIB_TEST(r && r->status == 206);
        DLIB_TEST(r->body == data.substr(10, 10));
        DLIB_TEST(r->get_header_value("Content-Range") == "bytes 10-19/100");

        r = cli.Get("/data", httplib::Headers{ httplib::make_range_header({ {90, -1} }) });
        DLIB_TEST(r && r->status == 206);
        DLIB_TEST(r->body == data.substr(90));

        r = cli.Get("/data", httplib::Headers{ httplib::make_range_header({ {0, 4}, {50, 54} }) });
        DLIB_TEST(r && r->status == 206);
        DLIB_TEST(r->get_header_value("Content-Type").find("multipart/byteranges") == 0);

        r = cli.Get("/data", httplib::Headers{ httplib::make_range_header({ {200, 300} }) });
        DLIB_TEST(r && r->status == 416);
    }

// ----------------------------------------------------------------------------------------

    void test_static_files()
    {
        print_spinner();
        const std::string base = "dtest_httplib_www";
        make_dir(base);
        make_dir(base + "/sub");
        make_dir("dtest_httplib_secret");
        write_file(base + "/index.html", "<p>index</p>");
        write_file(base + "/sub/page.txt", "page");
        write_file("dtest_httplib_secret/secret.txt", "secret");
#if !defined(_WIN32)
        // A link out of the mounted directory must not be followed.
        std::remove((base + "/escape.txt").c_str());
        const bool linked = symlink("../dtest_httplib_secret/secret.txt", (base + "/escape.txt").c_str()) == 0;
#endif

        httplib::Server srv;
        DLIB_TEST(srv.set_mount_point("/static", base));
        DLIB_TEST(srv.set_mount_point("/missing", "dtest_httplib_does_not_exist") == false);

        running_server<httplib::Server> run(srv);
        httplib::Client cli("127.0.0.1", run.port());

        auto r = cli.Get("/static/sub/page.txt");
        DLIB_TEST(r && r->status == 200 && r->body == "page");
        DLIB_TEST(r->get_header_value("Content-Type") == "text/plain");
        r = cli.Get("/static/");
        DLIB_TEST(r && r->status == 200 && r->body == "<p>index</p>");
        r = cli.Get("/static/sub");
        DLIB_TEST(r && r->status == 301);
        r = cli.Get("/staticx/index.html");
        DLIB_TEST(r && r->status == 404);
#if !defined(_WIN32)
        if (linked)
        {
            r = cli.Get("/static/escape.txt");
            DLIB_TEST(r && r->status == 403);
            DLIB_TEST(r->body.find("secret") == std::string::npos);
        }
#endif

        // Traversal attempts, sent raw so that no client normalizes them first.
        const char* attempts[] = {
            "/static/../dtest_httplib_secret/secret.txt",
            "/static/%2e%2e/dtest_httplib_secret/secret.txt",
            "/static/..%2fdtest_httplib_secret/secret.txt",
            "/static/sub/..\\..\\dtest_httplib_secret/secret.txt",
            "/static//../dtest_httplib_secret/secret.txt",
            "/static/%00/index.html",
        };
        for (const char* path : attempts)
        {
            const std::string answer = raw_exchange(run.port(),
                std::string("GET ") + path + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
            DLIB_TEST_MSG(answer.find("secret") == std::string::npos || status_of(answer) >= 400,
                "served outside the mount point for " << path);
            DLIB_TEST_MSG(status_of(answer) != 200, path << " -> " << answer.substr(0, 40));
        }

#if !defined(_WIN32)
        std::remove((base + "/escape.txt").c_str());
#endif
    }

// ----------------------------------------------------------------------------------------

    void test_limits_and_malformed_requests()
    {
        print_spinner();
        httplib::Server srv;
        srv.set_payload_max_length(1024);
        srv.Post("/small", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content(std::to_string(req.body.size()), "text/plain");
        });
        srv.Get("/ok", [](const httplib::Request&, httplib::Response& res) { res.set_content("ok", "text/plain"); });

        running_server<httplib::Server> run(srv);
        httplib::Client cli("127.0.0.1", run.port());

        auto r = cli.Post("/small", std::string(1024, 'a'), "text/plain");
        DLIB_TEST(r && r->status == 200 && r->body == "1024");
        r = cli.Post("/small", std::string(4096, 'a'), "text/plain");
        DLIB_TEST(r && r->status == 413);

        const int p = run.port();
        const std::string host = "Host: localhost\r\nConnection: close\r\n";

        DLIB_TEST(status_of(raw_exchange(p, "GET /ok HTTP/1.1\r\n" + host + "\r\n")) == 200);
        // A request line or a URI beyond the limits.
        DLIB_TEST(status_of(raw_exchange(p, "GET /" + std::string(CPPHTTPLIB_REQUEST_URI_MAX_LENGTH + 10, 'a')
            + " HTTP/1.1\r\n" + host + "\r\n")) == 414);
        // More headers than allowed, and a header longer than allowed.
        {
            std::string many = "GET /ok HTTP/1.1\r\n" + host;
            for (int i = 0; i < CPPHTTPLIB_HEADER_MAX_COUNT + 5; ++i) many += "X-H" + std::to_string(i) + ": v\r\n";
            const int st = status_of(raw_exchange(p, many + "\r\n"));
            DLIB_TEST_MSG(st == 400 || st == 431, st);
            const std::string longh = "GET /ok HTTP/1.1\r\n" + host + "X-Long: " + std::string(CPPHTTPLIB_HEADER_MAX_LENGTH + 10, 'b') + "\r\n\r\n";
            const int st2 = status_of(raw_exchange(p, longh));
            DLIB_TEST_MSG(st2 == 400 || st2 == 431, st2);
        }
        // Malformed request lines and versions.
        DLIB_TEST(status_of(raw_exchange(p, "GET\r\n" + host + "\r\n")) == 400);
        DLIB_TEST(status_of(raw_exchange(p, "GET /ok HTTP/9.9\r\n" + host + "\r\n")) >= 400);
        DLIB_TEST(status_of(raw_exchange(p, "G\x01T /ok HTTP/1.1\r\n" + host + "\r\n")) >= 400);
        // Header lines ended by a bare LF make the request invalid (see the README of
        // dlib/external/httplib).
        DLIB_TEST(status_of(raw_exchange(p, "GET /ok HTTP/1.1\r\nHost: localhost\nX-Smuggle: 1\r\nConnection: close\r\n\r\n")) == 400);
        // Obsolete line folding, a header without colon, a space before the colon.
        DLIB_TEST(status_of(raw_exchange(p, "GET /ok HTTP/1.1\r\n" + host + "X-A: 1\r\n folded\r\n\r\n")) == 400);
        DLIB_TEST(status_of(raw_exchange(p, "GET /ok HTTP/1.1\r\n" + host + "NoColon\r\n\r\n")) == 400);
        DLIB_TEST(status_of(raw_exchange(p, "GET /ok HTTP/1.1\r\n" + host + "X-A : 1\r\n\r\n")) == 400);
        // Request smuggling: conflicting framing.
        DLIB_TEST(status_of(raw_exchange(p, "POST /small HTTP/1.1\r\n" + host
            + "Content-Length: 3\r\nContent-Length: 4\r\n\r\nabcd")) == 400);
        DLIB_TEST(status_of(raw_exchange(p, "POST /small HTTP/1.1\r\n" + host
            + "Content-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")) == 400);
        DLIB_TEST(status_of(raw_exchange(p, "POST /small HTTP/1.1\r\n" + host
            + "Content-Length: -1\r\n\r\n")) == 400);
        DLIB_TEST(status_of(raw_exchange(p, "POST /small HTTP/1.1\r\n" + host
            + "Transfer-Encoding: chunked\r\n\r\nzz\r\nabc\r\n0\r\n\r\n")) == 400);
        // Transfer codings the library does not decode, and chunked applied twice (see
        // the README of dlib/external/httplib).
        DLIB_TEST(status_of(raw_exchange(p, "POST /small HTTP/1.1\r\n" + host
            + "Transfer-Encoding: gzip, chunked\r\n\r\n0\r\n\r\n")) == 501);
        DLIB_TEST(status_of(raw_exchange(p, "POST /small HTTP/1.1\r\n" + host
            + "Transfer-Encoding: deflate\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")) == 501);
        DLIB_TEST(status_of(raw_exchange(p, "POST /small HTTP/1.1\r\n" + host
            + "Transfer-Encoding: chunked, chunked\r\n\r\n0\r\n\r\n")) == 400);
        DLIB_TEST(status_of(raw_exchange(p, "POST /small HTTP/1.1\r\n" + host
            + "Transfer-Encoding: Chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n")) == 200);
        // A chunked body larger than the limit.
        {
            std::string req = "POST /small HTTP/1.1\r\n" + host + "Transfer-Encoding: chunked\r\n\r\n";
            for (int i = 0; i < 8; ++i) req += "200\r\n" + std::string(0x200, 'z') + "\r\n";
            req += "0\r\n\r\n";
            DLIB_TEST(status_of(raw_exchange(p, req)) == 413);
        }

        // The server is still healthy after all of the above.
        r = cli.Get("/ok");
        DLIB_TEST(r && r->status == 200);
    }

// ----------------------------------------------------------------------------------------

    void test_handlers_redirects_and_auth()
    {
        print_spinner();
        httplib::Server srv;
        srv.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            if (req.path == "/blocked")
            {
                res.status = 403;
                res.set_content("blocked", "text/plain");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        srv.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
            res.set_header("X-Post", "1");
        });
        // The error handler sees every response of status 400 and above, those of the
        // other handlers included: it only fills the ones left without a body.
        srv.set_error_handler([](const httplib::Request&, httplib::Response& res) {
            if (res.body.empty()) res.set_content("custom error " + std::to_string(res.status), "text/plain");
        });
        srv.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            std::string what = "unknown";
            try { std::rethrow_exception(ep); }
            catch (const std::exception& e) { what = e.what(); }
            catch (...) {}
            res.status = 500;
            res.set_content("caught: " + what, "text/plain");
        });
        srv.Get("/throw", [](const httplib::Request&, httplib::Response&) {
            throw std::runtime_error("handler failure");
        });
        srv.Get("/a", [](const httplib::Request&, httplib::Response& res) { res.set_redirect("/b"); });
        srv.Get("/b", [](const httplib::Request&, httplib::Response& res) { res.set_content("at b", "text/plain"); });
        srv.Get("/auth", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content(req.get_header_value("Authorization"), "text/plain");
        });

        running_server<httplib::Server> run(srv);
        httplib::Client cli("127.0.0.1", run.port());

        auto r = cli.Get("/blocked");
        DLIB_TEST(r && r->status == 403 && r->body == "blocked");
        r = cli.Get("/throw");
        DLIB_TEST(r && r->status == 500);
        DLIB_TEST_MSG(r->body == "caught: handler failure", r->body);
        r = cli.Get("/nowhere");
        DLIB_TEST(r && r->status == 404 && r->body == "custom error 404");
        DLIB_TEST(r->get_header_value("X-Post") == "1");

        r = cli.Get("/a");
        DLIB_TEST(r && r->status == 302);
        cli.set_follow_location(true);
        r = cli.Get("/a");
        DLIB_TEST(r && r->status == 200 && r->body == "at b");

        cli.set_bearer_token_auth("xxx");
        r = cli.Get("/auth");
        DLIB_TEST(r && r->body == "Bearer xxx");
        cli.set_bearer_token_auth("");
        cli.set_basic_auth("username", "xxx");
        r = cli.Get("/auth");
        DLIB_TEST(r && r->body == "Basic dXNlcm5hbWU6eHh4");
    }

// ----------------------------------------------------------------------------------------

    void test_keep_alive_concurrency_and_timeouts()
    {
        print_spinner();
        httplib::Server srv;
        std::atomic<int> served{0};
        srv.Get("/n", [&](const httplib::Request& req, httplib::Response& res) {
            ++served;
            res.set_content(req.get_param_value("i"), "text/plain");
        });
        srv.Get("/slow", [](const httplib::Request&, httplib::Response& res) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            res.set_content("late", "text/plain");
        });
        srv.set_read_timeout(1, 0);

        running_server<httplib::Server> run(srv);

        // One connection reused for many requests.
        {
            httplib::Client cli("127.0.0.1", run.port());
            cli.set_keep_alive(true);
            bool all = true;
            for (int i = 0; i < 50; ++i)
            {
                auto r = cli.Get("/n?i=" + std::to_string(i));
                all = all && r && r->body == std::to_string(i);
            }
            DLIB_TEST(all);
        }

        // Parallel clients.
        const int threads = 8, per_thread = 25;
        std::vector<std::thread> pool;
        std::atomic<int> ok{0};
        for (int t = 0; t < threads; ++t)
        {
            pool.emplace_back([&, t]() {
                httplib::Client cli("127.0.0.1", run.port());
                for (int i = 0; i < per_thread; ++i)
                {
                    const std::string v = std::to_string(t * 1000 + i);
                    auto r = cli.Get("/n?i=" + v);
                    if (r && r->body == v) ++ok;
                }
            });
        }
        for (auto& th : pool) th.join();
        DLIB_TEST(ok == threads * per_thread);

        // The client gives up on a slow answer.
        {
            httplib::Client cli("127.0.0.1", run.port());
            cli.set_read_timeout(0, 300000);
            auto r = cli.Get("/slow");
            DLIB_TEST(!r);
            DLIB_TEST(r.error() == httplib::Error::Read);
        }

        // The server drops a connection that sends an incomplete request and goes quiet.
        {
            const auto start = std::chrono::steady_clock::now();
            const std::string answer = raw_exchange(run.port(), "GET /n HTTP/1.1\r\nHost: loc", 5000);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            DLIB_TEST_MSG(ms < 4500, "the server kept an idle connection for " << ms << " ms");
            (void)answer;
        }
    }

// ----------------------------------------------------------------------------------------

#ifdef CPPHTTPLIB_MBEDTLS_SUPPORT

    /* Certificates made for the test: a CA, a server certificate it signed for localhost
       and 127.0.0.1, and a second self-signed certificate for another name. */
    struct test_pki
    {
        std::string ca_cert;
        std::string server_cert, server_key;
        std::string other_cert, other_key;
    };

    class pki_maker
    {
    public:
        pki_maker()
        {
            psa_crypto_init();
            mbedtls_entropy_init(&entropy_);
            mbedtls_ctr_drbg_init(&drbg_);
            const char* pers = "dlib httplib test";
            DLIB_CASSERT(mbedtls_ctr_drbg_seed(&drbg_, mbedtls_entropy_func, &entropy_,
                reinterpret_cast<const unsigned char*>(pers), std::strlen(pers)) == 0);
        }
        ~pki_maker()
        {
            mbedtls_ctr_drbg_free(&drbg_);
            mbedtls_entropy_free(&entropy_);
        }

        void new_key(mbedtls_pk_context& key)
        {
            mbedtls_pk_init(&key);
            DLIB_CASSERT(mbedtls_pk_setup(&key, mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY)) == 0);
            DLIB_CASSERT(mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, mbedtls_pk_ec(key),
                mbedtls_ctr_drbg_random, &drbg_) == 0);
        }

        std::string key_pem(mbedtls_pk_context& key)
        {
            std::vector<unsigned char> buf(4096);
            DLIB_CASSERT(mbedtls_pk_write_key_pem(&key, buf.data(), buf.size()) == 0);
            return std::string(reinterpret_cast<const char*>(buf.data()));
        }

        std::string certificate(mbedtls_pk_context& subject_key, mbedtls_pk_context& issuer_key,
            const char* subject, const char* issuer, bool is_ca, const std::vector<std::string>& dns,
            const std::vector<std::string>& ips, unsigned char serial)
        {
            mbedtls_x509write_cert crt;
            mbedtls_x509write_crt_init(&crt);
            mbedtls_x509write_crt_set_version(&crt, MBEDTLS_X509_CRT_VERSION_3);
            mbedtls_x509write_crt_set_md_alg(&crt, MBEDTLS_MD_SHA256);
            mbedtls_x509write_crt_set_subject_key(&crt, &subject_key);
            mbedtls_x509write_crt_set_issuer_key(&crt, &issuer_key);
            DLIB_CASSERT(mbedtls_x509write_crt_set_subject_name(&crt, subject) == 0);
            DLIB_CASSERT(mbedtls_x509write_crt_set_issuer_name(&crt, issuer) == 0);
            unsigned char serial_bytes[1] = { serial };
            DLIB_CASSERT(mbedtls_x509write_crt_set_serial_raw(&crt, serial_bytes, 1) == 0);
            DLIB_CASSERT(mbedtls_x509write_crt_set_validity(&crt, "20240101000000", "20450101000000") == 0);
            DLIB_CASSERT(mbedtls_x509write_crt_set_basic_constraints(&crt, is_ca ? 1 : 0, -1) == 0);
            DLIB_CASSERT(mbedtls_x509write_crt_set_key_usage(&crt, is_ca
                ? (MBEDTLS_X509_KU_KEY_CERT_SIGN | MBEDTLS_X509_KU_CRL_SIGN | MBEDTLS_X509_KU_DIGITAL_SIGNATURE)
                : (MBEDTLS_X509_KU_DIGITAL_SIGNATURE | MBEDTLS_X509_KU_KEY_AGREEMENT)) == 0);

            // Subject alternative names: DNS names and IPv4 addresses.
            std::vector<mbedtls_x509_san_list> sans(dns.size() + ips.size());
            std::vector<std::vector<unsigned char>> ip_bytes;
            size_t k = 0;
            for (const auto& d : dns)
            {
                sans[k].node.type = MBEDTLS_X509_SAN_DNS_NAME;
                sans[k].node.san.unstructured_name.p = reinterpret_cast<unsigned char*>(const_cast<char*>(d.data()));
                sans[k].node.san.unstructured_name.len = d.size();
                ++k;
            }
            for (const auto& ip : ips)
            {
                unsigned a, b, c, e;
                std::sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &e);
                ip_bytes.push_back({ static_cast<unsigned char>(a), static_cast<unsigned char>(b),
                                     static_cast<unsigned char>(c), static_cast<unsigned char>(e) });
            }
            for (auto& bytes : ip_bytes)
            {
                sans[k].node.type = MBEDTLS_X509_SAN_IP_ADDRESS;
                sans[k].node.san.unstructured_name.p = bytes.data();
                sans[k].node.san.unstructured_name.len = bytes.size();
                ++k;
            }
            for (size_t i = 0; i < sans.size(); ++i)
                sans[i].next = i + 1 < sans.size() ? &sans[i + 1] : nullptr;
            if (!sans.empty())
                DLIB_CASSERT(mbedtls_x509write_crt_set_subject_alternative_name(&crt, &sans[0]) == 0);

            std::vector<unsigned char> buf(8192);
            DLIB_CASSERT(mbedtls_x509write_crt_pem(&crt, buf.data(), buf.size(), mbedtls_ctr_drbg_random, &drbg_) == 0);
            mbedtls_x509write_crt_free(&crt);
            return std::string(reinterpret_cast<const char*>(buf.data()));
        }

        test_pki make()
        {
            test_pki p;
            mbedtls_pk_context ca_key, server_key, other_key;
            new_key(ca_key);
            new_key(server_key);
            new_key(other_key);
            p.ca_cert = certificate(ca_key, ca_key, "CN=dlib test CA,O=dlib", "CN=dlib test CA,O=dlib", true, {}, {}, 1);
            p.server_cert = certificate(server_key, ca_key, "CN=localhost,O=dlib", "CN=dlib test CA,O=dlib", false,
                { "localhost" }, { "127.0.0.1" }, 2);
            p.server_key = key_pem(server_key);
            p.other_cert = certificate(other_key, other_key, "CN=elsewhere.example,O=dlib", "CN=elsewhere.example,O=dlib",
                false, { "elsewhere.example" }, {}, 3);
            p.other_key = key_pem(other_key);
            mbedtls_pk_free(&ca_key);
            mbedtls_pk_free(&server_key);
            mbedtls_pk_free(&other_key);
            return p;
        }

    private:
        mbedtls_entropy_context entropy_;
        mbedtls_ctr_drbg_context drbg_;
    };

    void test_tls()
    {
        print_spinner();
        const test_pki pki = pki_maker().make();

        httplib::SSLServer::PemMemory pem = {
            pki.server_cert.c_str(), pki.server_cert.size(),
            pki.server_key.c_str(), pki.server_key.size(),
            nullptr, 0, nullptr
        };
        httplib::SSLServer srv(pem);
        DLIB_TEST(srv.is_valid());
        srv.Get("/secure", [](const httplib::Request& req, httplib::Response& res) {
            res.set_content("secure:" + req.get_param_value("v"), "text/plain");
        });
        srv.Get("/stream", [](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider("text/event-stream", [](size_t offset, httplib::DataSink& sink) {
                if (offset >= 3 * 10) { sink.done(); return true; }
                sink.write("data: 12\n\n", 10);
                return true;
            });
        });
        running_server<httplib::SSLServer> run(srv);

        // Trusted through the test CA, by name and by address.
        {
            httplib::SSLClient cli("localhost", run.port());
            cli.load_ca_cert_store(pki.ca_cert.c_str(), pki.ca_cert.size());
            cli.enable_server_certificate_verification(true);
            cli.enable_server_hostname_verification(true);
#ifdef CPPHTTPLIB_WINDOWS_AUTOMATIC_ROOT_CERTIFICATES_UPDATE
            cli.enable_windows_certificate_verification(false);
#endif
            auto r = cli.Get("/secure?v=1");
            DLIB_TEST_MSG(r && r->status == 200 && r->body == "secure:1",
                (r ? std::string("status ") + std::to_string(r->status) : httplib::to_string(r.error())));
            std::string events;
            r = cli.Get("/stream", [&](const char* d, size_t n) { events.append(d, n); return true; });
            DLIB_TEST(r && events == "data: 12\n\ndata: 12\n\ndata: 12\n\n");
        }
        {
            httplib::SSLClient cli("127.0.0.1", run.port());
            cli.load_ca_cert_store(pki.ca_cert.c_str(), pki.ca_cert.size());
#ifdef CPPHTTPLIB_WINDOWS_AUTOMATIC_ROOT_CERTIFICATES_UPDATE
            cli.enable_windows_certificate_verification(false);
#endif
            auto r = cli.Get("/secure?v=2");
            DLIB_TEST(r && r->body == "secure:2");
        }

        // Without the CA, the server certificate is refused.
        {
            httplib::SSLClient cli("localhost", run.port());
            cli.load_ca_cert_store(pki.other_cert.c_str(), pki.other_cert.size());
#ifdef CPPHTTPLIB_WINDOWS_AUTOMATIC_ROOT_CERTIFICATES_UPDATE
            cli.enable_windows_certificate_verification(false);
#endif
            auto r = cli.Get("/secure");
            DLIB_TEST(!r);
            DLIB_TEST_MSG(r.error() == httplib::Error::SSLServerVerification, httplib::to_string(r.error()));
        }

        // A server presenting a certificate for another name fails host name verification.
        {
            httplib::SSLServer::PemMemory other = {
                pki.other_cert.c_str(), pki.other_cert.size(),
                pki.other_key.c_str(), pki.other_key.size(),
                nullptr, 0, nullptr
            };
            httplib::SSLServer srv2(other);
            srv2.Get("/x", [](const httplib::Request&, httplib::Response& res) { res.set_content("x", "text/plain"); });
            running_server<httplib::SSLServer> run2(srv2);
            httplib::SSLClient cli("localhost", run2.port());
            cli.load_ca_cert_store(pki.other_cert.c_str(), pki.other_cert.size());
#ifdef CPPHTTPLIB_WINDOWS_AUTOMATIC_ROOT_CERTIFICATES_UPDATE
            cli.enable_windows_certificate_verification(false);
#endif
            auto r = cli.Get("/x");
            DLIB_TEST(!r);
            DLIB_TEST_MSG(r.error() == httplib::Error::SSLServerHostnameVerification, httplib::to_string(r.error()));
        }

        // Plain HTTP spoken to the TLS port gets no answer in clear.
        {
            const std::string answer = raw_exchange(run.port(), "GET /secure HTTP/1.1\r\nHost: localhost\r\n\r\n", 1500);
            DLIB_TEST(answer.find("secure:") == std::string::npos);
        }

        // Parallel TLS sessions, which exercise the thread safety of the TLS layer.
        {
            std::atomic<int> ok{0};
            std::vector<std::thread> pool;
            for (int t = 0; t < 6; ++t)
            {
                pool.emplace_back([&, t]() {
                    httplib::SSLClient cli("localhost", run.port());
                    cli.load_ca_cert_store(pki.ca_cert.c_str(), pki.ca_cert.size());
#ifdef CPPHTTPLIB_WINDOWS_AUTOMATIC_ROOT_CERTIFICATES_UPDATE
                    cli.enable_windows_certificate_verification(false);
#endif
                    for (int i = 0; i < 10; ++i)
                    {
                        const std::string v = std::to_string(t * 100 + i);
                        auto r = cli.Get("/secure?v=" + v);
                        if (r && r->body == "secure:" + v) ++ok;
                    }
                });
            }
            for (auto& th : pool) th.join();
            DLIB_TEST(ok == 60);
        }
    }
#endif

// ----------------------------------------------------------------------------------------

    class httplib_tester : public tester
    {
    public:
        httplib_tester (
        ) :
            tester ("test_httplib",
                    "Runs tests on the cpp-httplib copy in dlib/external/httplib.")
        {}

        void perform_test (
        )
        {
            dlog << dlib::LINFO << "cpp-httplib " << CPPHTTPLIB_VERSION;
            test_routes_and_methods();
            test_chunked_bodies_and_streaming();
            test_multipart_and_ranges();
            test_static_files();
            test_limits_and_malformed_requests();
            test_handlers_redirects_and_auth();
            test_keep_alive_concurrency_and_timeouts();
#ifdef CPPHTTPLIB_MBEDTLS_SUPPORT
            test_tls();
#endif
        }
    } a;

}
