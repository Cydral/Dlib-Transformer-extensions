// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#if !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#include <dlib/external/httplib/httplib.h>

#include "service_client.h"

#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>

#include "../common/log.h"
#include "../common/platform.h"

namespace albireo
{

    struct service_client::kept_connection
    {
        std::mutex m;
        std::unique_ptr<httplib::SSLClient> client;
        // The port and the authority the connection was opened for, and its last use.
        int port = 0;
        std::string ca;
        std::chrono::steady_clock::time_point used;
        unsigned long long opened = 0;
    };

    namespace
    {
        // The time a kept connection stays open without a request, below the 30 seconds after
        // which the service closes it.
        const std::chrono::seconds kept_idle_limit(20);

        std::string trim(const std::string& s)
        {
            const size_t b = s.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return std::string();
            return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
        }

        std::unique_ptr<httplib::SSLClient> make_client(int port, const std::string& ca, int timeout_seconds)
        {
            std::unique_ptr<httplib::SSLClient> c(new httplib::SSLClient("localhost", port));
            c->set_ca_cert_path(ca.c_str());
            c->enable_server_certificate_verification(true);
            c->set_connection_timeout(3);
            c->set_read_timeout(timeout_seconds);
            c->set_write_timeout(timeout_seconds);
            c->set_keep_alive(false);
            return c;
        }

        bool finish(const httplib::Result& r, json& out, std::string& error)
        {
            if (!r)
            {
                error = httplib::to_string(r.error());
                return false;
            }
            out = json::parse_or_null(r->body);
            // The windows show no address of the publication space, which a service before this version gave.
            if (errors_name_publication_space(out)) hide_publication_space_in_errors(out);
            if (r->status >= 400)
            {
                const json& e = out["error"];
                error = e.is_string() ? e.as_string() : e["message"].is_string() ? e["message"].as_string()
                    : "status " + std::to_string(r->status);
                return false;
            }
            return true;
        }
    }

    service_client::service_client() : kept_(std::make_shared<kept_connection>())
    {
        reload();
    }

    void service_client::own_connection()
    {
        kept_ = std::make_shared<kept_connection>();
    }

    unsigned long long service_client::poll_connections() const
    {
        std::lock_guard<std::mutex> lock(kept_->m);
        return kept_->opened;
    }

    bool service_client::poll(const std::string& path, json& out, std::string& error, int timeout_seconds) const
    {
        kept_connection& k = *kept_;
        std::lock_guard<std::mutex> lock(k.m);
        // A connection unused for too long, or opened for another port or authority, closes first.
        const auto now = std::chrono::steady_clock::now();
        if (k.client && (k.port != port_ || k.ca != ca_path_ || now - k.used > kept_idle_limit)) k.client.reset();
        httplib::Headers h = { { "X-Albireo-Admin", token_ } };
        httplib::Result r;
        for (int attempt = 0; attempt < 2; ++attempt)
        {
            if (!k.client)
            {
                k.client = make_client(port_, ca_path_, timeout_seconds);
                k.client->set_keep_alive(true);
                k.port = port_;
                k.ca = ca_path_;
            }
            k.client->set_read_timeout(timeout_seconds);
            k.client->set_write_timeout(timeout_seconds);
            const bool reused = k.client->is_socket_open() != 0;
            if (!reused) ++k.opened;
            r = k.client->Get(path, h);
            k.used = std::chrono::steady_clock::now();
            if (r) break;
            // A restart of the service, or a new authority, takes a new connection.
            k.client.reset();
            if (!reused) break;
        }
        return finish(r, out, error);
    }

    void service_client::reload()
    {
        data_dir_ = albireo::data_dir();
        ca_path_ = join_path(join_path(data_dir_, "certificates"), "albireo-authority.crt");
        std::string text;
        if (read_file(join_path(data_dir_, "admin.token"), text, 4096)) token_ = trim(text);
        if (read_file(join_path(data_dir_, "settings.json"), text, 1u << 20))
        {
            const json s = json::parse_or_null(text);
            const long long p = s["port"].as_int(8443);
            port_ = p > 0 && p < 65536 ? static_cast<int>(p) : 8443;
        }
    }

    std::string service_client::base_url() const
    {
        return "https://localhost:" + std::to_string(port_) + "/";
    }

    bool service_client::get(const std::string& path, json& out, std::string& error, int timeout_seconds) const
    {
        auto c = make_client(port_, ca_path_, timeout_seconds);
        httplib::Headers h = { { "X-Albireo-Admin", token_ } };
        return finish(c->Get(path, h), out, error);
    }

    bool service_client::send(const char* method, const std::string& path, const json& body, json& out,
        std::string& error, int timeout_seconds) const
    {
        auto c = make_client(port_, ca_path_, timeout_seconds);
        httplib::Headers h = { { "X-Albireo-Admin", token_ } };
        const std::string m = method;
        const std::string text = body.is_null() ? std::string() : body.dump();
        if (m == "POST") return finish(c->Post(path, h, text, "application/json"), out, error);
        if (m == "PUT") return finish(c->Put(path, h, text, "application/json"), out, error);
        if (m == "DELETE") return finish(c->Delete(path, h), out, error);
        error = "unknown method";
        return false;
    }

    bool service_client::upload(const std::string& path, const std::string& file,
        const std::function<bool(uint64_t, uint64_t)>& progress, json& out, std::string& error) const
    {
        FILE* f = _wfopen(widen(file).c_str(), L"rb");
        if (!f)
        {
            error = "cannot open " + file;
            return false;
        }
        const uint64_t total = file_size(file);
        auto c = make_client(port_, ca_path_, 600);
        httplib::Headers h = { { "X-Albireo-Admin", token_ } };
        std::vector<char> buf(1u << 20);
        uint64_t sent = 0;
        auto r = c->Post(path, h, static_cast<size_t>(total),
            [&](size_t offset, size_t length, httplib::DataSink& sink) {
                if (_fseeki64(f, static_cast<long long>(offset), SEEK_SET) != 0) return false;
                const size_t n = fread(buf.data(), 1, std::min(length, buf.size()), f);
                if (n == 0) return false;
                if (!sink.write(buf.data(), n)) return false;
                sent = offset + n;
                return !progress || progress(sent, total);
            },
            "application/octet-stream");
        fclose(f);
        return finish(r, out, error);
    }

}
