// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The routes of the local HTTPS server:

        /                       the web application (files of the web folder)
        /docs/                  the interactive documentation of the interface (Swagger UI)
        /openapi.json           its OpenAPI description
        /v1/...                 the interface, described by openapi.json

    The files go in the body of the requests as they are (application/octet-stream), and come
    back so, their name and their report in headers, or as JSON with format=json.  The
    operations that change the installation or reveal its records (settings, restart,
    updates, records, key, logs) require the administration token of the data folder in the
    X-Pseudalis-Admin header; the tray and the shell read it there.  Every request must name
    the local server in its Host header, and a request that comes from a page must come from
    the local server too, so that no web site can drive the service through the browser.
*/

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
#include <dlib/external/httplib/httplib.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>

#include "../common/log.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "crypto.h"
#include "drive.h"
#include "http_fetch.h"
#include "service.h"

namespace pseudalis
{
    namespace
    {
        void send_json(httplib::Response& res, int status, const json& j)
        {
            res.status = status;
            res.set_content(j.dump(), "application/json; charset=utf-8");
        }

        void send_error(httplib::Response& res, int status, const user_error& e)
        {
            json j = json::object();
            e.add_to(j);
            send_json(res, status, j);
        }

        void send_error(httplib::Response& res, int status, const std::string& message, const std::string& code)
        {
            send_error(res, status, user_error().set(message, code));
        }

        std::string lower(std::string s)
        {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        bool same_secret(const std::string& a, const std::string& b)
        {
            if (a.size() != b.size()) return false;
            unsigned char d = 0;
            for (size_t i = 0; i < a.size(); ++i) d |= static_cast<unsigned char>(a[i] ^ b[i]);
            return d == 0;
        }

        bool is_local_host(const std::string& host, int port)
        {
            const std::string h = lower(host);
            const std::string p = ":" + std::to_string(port);
            return h == "localhost" || h == "127.0.0.1" || h == "[::1]" || h == "localhost" + p ||
                   h == "127.0.0.1" + p || h == "[::1]" + p;
        }

        bool is_local_origin(const std::string& origin, int port)
        {
            const std::string o = lower(origin);
            const std::string p = ":" + std::to_string(port);
            return o == "https://localhost" + p || o == "https://127.0.0.1" + p || o == "https://[::1]" + p;
        }

        // Percent-encoding of UTF-8 for a header: the name of a file, a report.
        std::string percent_encode(const std::string& s)
        {
            static const char hex[] = "0123456789ABCDEF";
            std::string out;
            for (unsigned char c : s)
            {
                if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
                    out.push_back(static_cast<char>(c));
                else
                {
                    out.push_back('%');
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 15]);
                }
            }
            return out;
        }

        // A name of file sent by a client: its last component, without control characters.
        std::string clean_name(const std::string& name, const char* fallback)
        {
            std::string n = name;
            const size_t slash = n.find_last_of("/\\");
            if (slash != std::string::npos) n = n.substr(slash + 1);
            std::string out;
            for (char c : n)
                if (static_cast<unsigned char>(c) >= 32 && c != ':' && c != '*' && c != '?' && c != '"' && c != '<' &&
                    c != '>' && c != '|')
                    out.push_back(c);
            if (out.empty() || out == "." || out == "..") out = fallback;
            return out.size() > 255 ? out.substr(0, 255) : out;
        }

        // The counts of a report, small enough for a header.
        json report_summary(const json& report)
        {
            json s = json::object();
            s["values"] = report["values"];
            s["total"] = report["total"];
            s["dropped"] = static_cast<unsigned long long>(report["dropped"].size());
            s["tags_restored"] = report["tags_restored"];
            // A restoration from the tags: the pictures it removed, the day and the reason of a forgetting.
            if (report["pictures_removed"].as_int(0) > 0) s["pictures_removed"] = report["pictures_removed"];
            if (report["original_forgotten"].is_object()) s["original_forgotten"] = report["original_forgotten"];
            return s;
        }

        void send_file(const httplib::Request& req, httplib::Response& res, const std::string& bytes,
                       const std::string& name, const json& meta)
        {
            if (req.get_param_value("format") == "json")
            {
                json j = meta;
                j["name"] = name;
                j["data"] = base64_encode(bytes);
                send_json(res, 200, j);
                return;
            }
            res.status = 200;
            res.set_header("Content-Disposition", "attachment; filename*=UTF-8''" + percent_encode(name));
            res.set_header("X-Pseudalis-Name", percent_encode(name));
            for (const auto& kv : meta.items())
            {
                if (kv.first == "report")
                    res.set_header("X-Pseudalis-Report", percent_encode(report_summary(kv.second).dump()));
                else if (kv.second.is_string())
                    res.set_header("X-Pseudalis-" + kv.first, percent_encode(kv.second.str()));
                else res.set_header("X-Pseudalis-" + kv.first, kv.second.dump());
            }
            res.set_header("Access-Control-Expose-Headers",
                "Content-Disposition, X-Pseudalis-Name, X-Pseudalis-Report");
            res.set_content(bytes, "application/octet-stream");
        }

        // days: the days an original stays kept once its file left the drive.
        json record_json(const file_record& r, int days)
        {
            json j = json::object();
            j["id"] = r.id;
            j["created"] = iso_time(r.created);
            j["source"] = r.source;
            j["original_name"] = r.original_name;
            j["output_name"] = r.output_name;
            j["original_size"] = r.original_size;
            j["output_size"] = r.output_size;
            j["kind"] = r.kind;
            j["output_sha256"] = r.output_sha256;
            j["object_id"] = r.object_id;
            j["path"] = r.path;
            // When its file left the drive, and the day its original goes: none when the originals stay (0).
            j["removed"] = r.removed ? iso_time(r.removed) : std::string();
            j["forgotten"] = r.removed && days > 0 ? iso_time(r.removed + static_cast<int64_t>(days) * 86400)
                                                   : std::string();
            return j;
        }

        /*
            The file a request carries: its body as it comes, or the first file of a multipart
            form with its name.  The routes of the files read it through their content reader,
            so that httplib never parses it as a form (application/x-www-form-urlencoded, the
            type curl gives a body by default), which would limit it to 8 KB and mix its bytes
            with the parameters of the address.
        */
        bool read_file_body(const httplib::Request& req, const httplib::ContentReader& reader, std::string& body,
                            std::string& filename)
        {
            body.clear();
            filename.clear();
            if (!req.is_multipart_form_data())
                return reader([&](const char* data, size_t size) {
                    body.append(data, size);
                    return true;
                });
            bool is_file = false;
            return reader(
                [&](const httplib::FormData& f) {
                    is_file = !f.filename.empty() && filename.empty();
                    if (is_file) filename = f.filename;
                    return true;
                },
                [&](const char* data, size_t size) {
                    if (is_file) body.append(data, size);
                    return true;
                });
        }

        // false, with 503, when no slot of work was free in time.
        bool busy_refused(const work_slot& slot, httplib::Response& res)
        {
            if (slot.held()) return true;
            res.set_header("Retry-After", "15");
            send_error(res, 503, "the service is busy with other files: try again in a moment", "busy");
            return false;
        }

        bool parse_body(const httplib::Request& req, httplib::Response& res, json& body)
        {
            if (!json::parse(req.body, body) || !body.is_object())
            {
                send_error(res, 400, "the body must be a JSON object", "bad_request");
                return false;
            }
            return true;
        }
    }

    void install_routes(httplib::Server& srv, service& svc)
    {
        const int port = svc.port();
        service* s = &svc;

        srv.set_pre_routing_handler([port, s](const httplib::Request& req, httplib::Response& res) {
            auto refuse = [&](int status, const char* message) {
                res.status = status;
                res.set_header("Connection", "close");
                res.set_content(message, "text/plain");
                return httplib::Server::HandlerResponse::Handled;
            };
            if (!is_local_host(req.get_header_value("Host"), port)) return refuse(421, "misdirected request");
            const std::string origin = req.get_header_value("Origin");
            if (!origin.empty() && !is_local_origin(origin, port))
                return refuse(403, "cross-origin requests are refused");
            /*
                A body larger than its route takes is refused before it is read: the files up to
                the size of the settings, a JSON body up to 16 MB.
            */
            const unsigned long long length =
                std::strtoull(req.get_header_value("Content-Length").c_str(), nullptr, 10);
            const bool file_route = req.path == "/v1/anonymize" || req.path == "/v1/restore";
            const unsigned long long limit = file_route
                ? (static_cast<unsigned long long>(s->settings().get().max_file_mb) + 1) << 20 : 16ull << 20;
            if (length > limit)
            {
                user_error e;
                e.set("the request is larger than the service takes", "file_too_large")
                    .arg("mb", std::to_string(limit >> 20));
                res.set_header("Connection", "close");
                send_error(res, 413, e);
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        srv.set_post_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            res.set_header("X-Content-Type-Options", "nosniff");
            res.set_header("Referrer-Policy", "no-referrer");
            res.set_header("X-Frame-Options", "DENY");
            res.set_header("Cross-Origin-Opener-Policy", "same-origin");
            res.set_header("Cross-Origin-Resource-Policy", "same-origin");
            if (req.path.compare(0, 5, "/docs") == 0)
                res.set_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                    "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'");
            else
                res.set_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; "
                    "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
                    "base-uri 'none'; form-action 'none'");
            if (req.path.compare(0, 4, "/v1/") == 0) res.set_header("Cache-Control", "no-store");
        });
        srv.set_exception_handler([](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            std::string what = "internal error";
            try
            {
                std::rethrow_exception(ep);
            }
            catch (const std::exception& e)
            {
                what = e.what();
            }
            catch (...)
            {
            }
            // The detail goes to the log alone.
            log_error("http: " + req.method + " " + req.path + ": " + what);
            send_error(res, 500, "the service could not process the request", "server_error");
        });

        auto admin = [s](const httplib::Request& req, httplib::Response& res) {
            const std::string token = s->admin_token();
            if (!token.empty() && same_secret(req.get_header_value("X-Pseudalis-Admin"), token)) return true;
            res.set_header("Connection", "close");
            send_error(res, 401, "this operation requires the administration token", "authentication");
            return false;
        };

        srv.Get("/v1/health", [](const httplib::Request&, httplib::Response& res) {
            json j = json::object();
            j["status"] = "ok";
            j["version"] = PSEUDALIS_VERSION;
            send_json(res, 200, j);
        });
        srv.Get("/v1/info", [s](const httplib::Request&, httplib::Response& res) {
            json j = s->status();
            j.erase("updates");
            send_json(res, 200, j);
        });

        // Files.
        srv.Post("/v1/anonymize", [s](const httplib::Request& req, httplib::Response& res,
                                      const httplib::ContentReader& reader) {
            work_slot slot(s->work());
            if (!busy_refused(slot, res)) return;
            std::string body, filename;
            if (!read_file_body(req, reader, body, filename))
            {
                send_error(res, 400, "the file of the request cannot be read", "bad_request");
                return;
            }
            const std::string given = req.get_param_value("name");
            const std::string name = clean_name(given.empty() ? filename : given, "document");
            const std::string source = req.get_param_value("source");
            const bool record = req.get_param_value("record") != "false";
            protect_result r;
            user_error e;
            if (!s->protect(body, name, source == "web" || source == "shell" ? source : "api", record, r, e))
            {
                send_error(res, e.code == "file_too_large" ? 413 : 422, e);
                return;
            }
            json meta = json::object();
            meta["kind"] = r.kind;
            meta["record"] = r.record;
            meta["report"] = r.report;
            send_file(req, res, r.bytes, r.name, meta);
        });
        srv.Post("/v1/restore", [s](const httplib::Request& req, httplib::Response& res,
                                    const httplib::ContentReader& reader) {
            work_slot slot(s->work());
            if (!busy_refused(slot, res)) return;
            std::string body, filename;
            if (!read_file_body(req, reader, body, filename))
            {
                send_error(res, 400, "the file of the request cannot be read", "bad_request");
                return;
            }
            const std::string given = req.get_param_value("name");
            const std::string name = clean_name(given.empty() ? filename : given, "document");
            restore_result r;
            user_error e;
            if (!s->restore(body, name, lower(req.get_param_value("object_id")), r, e))
            {
                // The picture of a text file, forgotten, is gone for good: 410, not 404.
                send_error(res, e.code == "not_protected" || e.code == "picture_missing" ? 404 :
                                e.code == "picture_forgotten" ? 410 : 422, e);
                return;
            }
            json meta = json::object();
            meta["restored"] = r.how;
            meta["record"] = r.record;
            meta["report"] = r.report;
            send_file(req, res, r.bytes, r.name, meta);
        });
        srv.Get("/v1/lookup", [s](const httplib::Request& req, httplib::Response& res) {
            file_record r;
            json j = json::object();
            const bool known = s->db().find_by_output(lower(req.get_param_value("sha256")), r) ||
                               s->db().find_by_object(lower(req.get_param_value("object_id")), r);
            j["known"] = known;
            if (known)
            {
                j["record"] = r.id;
                j["output_name"] = r.output_name;
                j["exact"] = r.output_sha256 == lower(req.get_param_value("sha256"));
            }
            else
            {
                // A protected file whose original was forgotten: its restoration puts back its tags.
                int64_t when = 0;
                std::string reason;
                if (s->db().find_forgotten(lower(req.get_param_value("sha256")),
                                           lower(req.get_param_value("object_id")), when, reason))
                {
                    json f = json::object();
                    f["time"] = iso_time(when);
                    f["reason"] = reason;
                    j["forgotten"] = f;
                }
            }
            send_json(res, 200, j);
        });
        srv.Post(R"(/v1/records/(\d+)/location)", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json body;
            if (!parse_body(req, res, body)) return;
            const long long id = std::atoll(req.matches[1].str().c_str());
            file_record r;
            if (!s->db().get_file(id, r))
            {
                send_error(res, 404, "no such record", "not_found");
                return;
            }
            std::string object_id = lower(body["object_id"].str());
            if (object_id.size() != 32 || object_id.find_first_not_of("0123456789abcdef") != std::string::npos)
                object_id.clear();
            s->db().set_file_location(id, object_id, body["path"].str());
            send_json(res, 200, json::object());
        });

        // Texts.
        auto text_route = [s](const char* operation) {
            return [s, operation](const httplib::Request& req, httplib::Response& res) {
                work_slot slot(s->work());
                if (!busy_refused(slot, res)) return;
                json body, out;
                if (!parse_body(req, res, body)) return;
                user_error e;
                if (!s->text(operation, body["text"].str(), out, e))
                {
                    send_error(res, 422, e);
                    return;
                }
                send_json(res, 200, out);
            };
        };
        srv.Post("/v1/text/anonymize", text_route("anonymize_text"));
        srv.Post("/v1/text/restore", text_route("restore_text"));
        srv.Post("/v1/text/detect", text_route("detect"));

        /*
            Names: the name of a folder, or any name, its values written as substitutes that the
            database keeps for their restoration, without tags, which a name cannot hold.
        */
        auto name_route = [s](bool protect) {
            return [s, protect](const httplib::Request& req, httplib::Response& res) {
                work_slot slot(s->work());
                if (!busy_refused(slot, res)) return;
                json body;
                if (!parse_body(req, res, body)) return;
                const std::string name = body["name"].str();
                bool plain = !name.empty() && name.size() <= 1024 && name != "." && name != "..";
                for (const char c : name)
                    plain = plain && static_cast<unsigned char>(c) >= 32 && c != '/' && c != '\\';
                if (!plain)
                {
                    send_error(res, 400, "the name must be a name alone, of 1024 bytes at most", "bad_request");
                    return;
                }
                json j = json::object();
                if (!protect)
                {
                    j["name"] = s->restore_name(name);
                    send_json(res, 200, j);
                    return;
                }
                std::string out;
                user_error e;
                if (!s->protect_name(name, out, e))
                {
                    send_error(res, 422, e);
                    return;
                }
                j["name"] = out.empty() ? name : out;
                send_json(res, 200, j);
            };
        };
        srv.Post("/v1/names/anonymize", name_route(true));
        srv.Post("/v1/names/restore", name_route(false));

        // The drive, its events, Albireo.
        srv.Get("/v1/drive", [s](const httplib::Request&, httplib::Response& res) {
            send_json(res, 200, s->status()["drive"]);
        });
        srv.Get("/v1/events", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json j = json::object();
            j["events"] = s->db().events(std::atoll(req.get_param_value("since").c_str()));
            send_json(res, 200, j);
        });
        srv.Get("/v1/albireo", [s](const httplib::Request&, httplib::Response& res) {
            send_json(res, 200, s->albireo().status());
        });

        // Administration.
        srv.Get("/v1/records", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            const long long limit = std::max(1LL, std::min(1000LL, std::atoll(req.get_param_value("limit").c_str())));
            const long long offset = std::max(0LL, std::atoll(req.get_param_value("offset").c_str()));
            json list = json::array();
            const int days = s->settings().get().removed_days;
            for (const file_record& r : s->db().list_files(req.has_param("limit") ? limit : 100, offset,
                                                          req.get_param_value("source")))
                list.push_back(record_json(r, days));
            json j = json::object();
            j["records"] = list;
            j["total"] = s->db().count_files(req.get_param_value("source"));
            send_json(res, 200, j);
        });
        srv.Get(R"(/v1/records/(\d+)/original)", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            file_record r;
            restore_result out;
            user_error e;
            // The original as a restoration gives it: an edited protected file, its tags restored.
            if (!s->db().get_file(std::atoll(req.matches[1].str().c_str()), r) || !s->recorded_original(r, out, e))
            {
                send_error(res, 404, "no such record", "not_found");
                return;
            }
            json meta = json::object();
            meta["record"] = r.id;
            send_file(req, res, out.bytes, out.name, meta);
        });
        srv.Delete(R"(/v1/records/(\d+))", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            if (!s->forget(std::atoll(req.matches[1].str().c_str()), "forgotten"))
            {
                send_error(res, 404, "no such record", "not_found");
                return;
            }
            send_json(res, 200, json::object());
        });
        srv.Get("/v1/settings", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            send_json(res, 200, s->settings().get().to_json());
        });
        srv.Put("/v1/settings", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json parsed;
            if (!parse_body(req, res, parsed)) return;
            // Read without adding a member: a setting the request leaves out keeps its value.
            const json& body = parsed;
            const std::string letter = body["drive_letter"].str();
            if (body.contains("drive_letter") && !letter.empty() && !valid_drive_letter(letter))
            {
                send_error(res, 400, "the letter of the drive must be A, B or D to Z", "settings_drive");
                return;
            }
            // The place of the publication and the proxy, in the forms the service reads only.
            const std::string feed = body["feed_url"].str();
            if ((body.contains("feed_url") && (!body["feed_url"].is_string() || feed.compare(0, 8, "https://") != 0 ||
                 feed.size() > 1024 || feed.find_first_of(" \t\r\n\"<>") != std::string::npos)) ||
                (body.contains("proxy") && (!body["proxy"].is_string() || !valid_proxy_setting(body["proxy"].str()))))
            {
                send_error(res, 400, "the feed or the proxy is not in a form the service reads",
                           "settings_publication");
                return;
            }
            // The other settings, each in its range: a value out of it is refused, not replaced.
            struct range
            {
                const char* name;
                long long low, high;
            };
            const range ranges[] = { { "port", 1024, 65535 }, { "max_file_mb", 1, 2048 },
                                     { "removed_days", 0, host_settings::max_removed_days },
                                     { "history_mb", 1, host_settings::max_history_mb } };
            for (const range& r : ranges)
            {
                if (!body.contains(r.name)) continue;
                const json& v = body[r.name];
                const long long n = v.as_int(r.low - 1);
                if (!v.is_number() || static_cast<double>(n) != v.as_double() || n < r.low || n > r.high)
                {
                    user_error e;
                    e.set(std::string("the setting ") + r.name + " must be a whole number from " +
                          std::to_string(r.low) + " to " + std::to_string(r.high), "settings_value")
                        .arg("field", r.name).arg("min", std::to_string(r.low)).arg("max", std::to_string(r.high));
                    send_error(res, 400, e);
                    return;
                }
            }
            const char* const choices[][2] = { { "language", " auto en fr de it es " }, { "date_order", " dmy mdy " },
                                               { "albireo", " auto off " } };
            for (const auto& c : choices)
            {
                if (!body.contains(c[0])) continue;
                const std::string v = body[c[0]].str();
                if (!body[c[0]].is_string() || v.empty() || std::string(c[1]).find(" " + v + " ") == std::string::npos)
                {
                    user_error e;
                    e.set(std::string("the setting ") + c[0] + " has a value the service does not take",
                          "settings_value").arg("field", c[0]);
                    send_error(res, 400, e);
                    return;
                }
            }
            for (const char* flag : { "check_updates", "trust_authority" })
                if (body.contains(flag) && !body[flag].is_bool())
                {
                    user_error e;
                    e.set(std::string("the setting ") + flag + " must be true or false", "settings_value")
                        .arg("field", flag);
                    send_error(res, 400, e);
                    return;
                }
            // The space of the originals holds the largest file.
            {
                const host_settings current = s->settings().get();
                const long long space = body.contains("history_mb") ? body["history_mb"].as_int(0) : current.history_mb;
                const long long largest =
                    body.contains("max_file_mb") ? body["max_file_mb"].as_int(0) : current.max_file_mb;
                if (space < largest)
                {
                    user_error e;
                    e.set("the setting history_mb must be a whole number from " + std::to_string(largest) + " to " +
                          std::to_string(host_settings::max_history_mb) + ", the largest file at least",
                          "settings_value")
                        .arg("field", "history_mb").arg("min", std::to_string(largest))
                        .arg("max", std::to_string(host_settings::max_history_mb));
                    send_error(res, 400, e);
                    return;
                }
            }
            host_settings before, after;
            if (!s->settings().update([&](host_settings& st) {
                    before = st;
                    json merged = st.to_json();
                    for (const auto& kv : body.items()) merged[kv.first] = kv.second;
                    st = host_settings::from_json(merged);
                }, after))
            {
                send_error(res, 500, "the settings cannot be written", "settings");
                return;
            }
            const bool restart = after.port != before.port || after.drive_letter != before.drive_letter ||
                                 after.date_order != before.date_order || after.proxy != before.proxy ||
                                 after.feed_url != before.feed_url;
            s->albireo().set_enabled(after.albireo != "off");
            // Fewer days or less space: the originals beyond them go now, in the background.
            if (after.removed_days != before.removed_days || after.history_mb != before.history_mb)
                s->request_cleanup();
            json j = after.to_json();
            j["restart"] = restart;
            send_json(res, 200, j);
            if (restart)
            {
                // The listener closes now; the answer leaves before the service stops.
                log_info("settings: changed, the service restarts");
                s->request_restart();
            }
        });
        // The imposed terms, which the engine reads before anything else of a text.
        srv.Get("/v1/terms", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json j = json::object();
            j["terms"] = s->terms();
            j["max"] = static_cast<unsigned long long>(max_imposed_terms);
            j["max_length"] = static_cast<unsigned long long>(max_imposed_term_length);
            json types = json::array();
            for (const std::string& t : imposed_term_types()) types.push_back(t);
            j["types"] = types;
            send_json(res, 200, j);
        });
        srv.Put("/v1/terms", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json parsed;
            if (!parse_body(req, res, parsed)) return;
            const json& body = parsed;
            user_error e;
            if (!s->set_terms(body["terms"], e))
            {
                send_error(res, e.code == "engine" || e.code == "database" ? 500 : 400, e);
                return;
            }
            json j = json::object();
            j["terms"] = s->terms();
            send_json(res, 200, j);
        });
        srv.Post("/v1/restart", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            send_json(res, 200, json::object());
            s->request_restart();
        });
        srv.Get("/v1/updates", [s](const httplib::Request&, httplib::Response& res) {
            send_json(res, 200, s->updates().status());
        });
        srv.Post("/v1/updates/check", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            s->updates().check_async();
            send_json(res, 202, s->updates().status());
        });
        srv.Post("/v1/updates/apply", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            user_error e;
            if (!s->updates().can_apply(e))
            {
                send_error(res, 409, e);
                return;
            }
            send_json(res, 202, json::object());
            s->request_restart();
        });
        srv.Post("/v1/key/export", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json body;
            if (!parse_body(req, res, body)) return;
            std::string exported;
            user_error e;
            if (!s->export_key(body["passphrase"].str(), exported, e))
            {
                send_error(res, 422, e);
                return;
            }
            json j = json::object();
            j["key"] = exported;
            send_json(res, 200, j);
        });
        srv.Post("/v1/key/import", [s, admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json body;
            if (!parse_body(req, res, body)) return;
            user_error e;
            if (!s->import_key(body["key"].str(), body["passphrase"].str(), e))
            {
                send_error(res, 422, e);
                return;
            }
            send_json(res, 200, json::object());
        });
        srv.Get("/v1/logs", [admin](const httplib::Request& req, httplib::Response& res) {
            if (!admin(req, res)) return;
            json lines = json::array();
            for (const std::string& l : log_file::instance().recent()) lines.push_back(l);
            json j = json::object();
            j["lines"] = lines;
            send_json(res, 200, j);
        });

        // The web application and the documentation.
        const std::string web = join_path(svc.program_dir(), "web");
        if (!srv.set_mount_point("/", web)) log_warning("http: no web folder at " + web);
        srv.set_file_request_handler([](const httplib::Request&, httplib::Response& res) {
            res.set_header("Cache-Control", "no-cache");
        });
        srv.Get("/docs", [](const httplib::Request&, httplib::Response& res) { res.set_redirect("/docs/"); });
    }
}
