// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "public_link.h"

#include <chrono>
#include <map>
#include <mutex>

#include "../common/json.h"
#include "http_fetch.h"
#include "public_link_secret.h"

namespace albireo
{

    namespace
    {
        using clock_type = std::chrono::steady_clock;

        const unsigned char masked_password[] = ALBIREO_PUBLIC_LINK_SECRET;
        const size_t masked_size = ALBIREO_PUBLIC_LINK_SECRET_SIZE;

        // The password, unmasked by the generator the configuration masked it with.
        std::string link_password()
        {
            std::string p(masked_size, '\0');
            unsigned long long seed = 20261101;
            for (size_t i = 0; i < masked_size; ++i)
            {
                seed = (seed * 1103515245ull + 12345ull) % 2147483648ull;
                p[i] = static_cast<char>(masked_password[i] ^ static_cast<unsigned char>((seed >> 16) & 255));
            }
            return p;
        }

        std::string lower(std::string s)
        {
            for (char& c : s)
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            return s;
        }

        bool alphanumeric(char c)
        {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        }

        // A value of a query: every byte but the unreserved characters of RFC 3986 as %XX.
        std::string query_value(const std::string& s)
        {
            static const char digits[] = "0123456789ABCDEF";
            std::string out;
            for (unsigned char c : s)
            {
                if (alphanumeric(static_cast<char>(c)) || c == '-' || c == '_' || c == '.' || c == '~')
                    out += static_cast<char>(c);
                else
                {
                    out += '%';
                    out += digits[c >> 4];
                    out += digits[c & 15];
                }
            }
            return out;
        }

        // The path of an address of a content server, the characters an address cannot hold as %XX.
        std::string address_path(const std::string& s)
        {
            static const char digits[] = "0123456789ABCDEF";
            std::string out;
            for (unsigned char c : s)
            {
                if (c <= 0x20 || c >= 0x7f || c == '"' || c == '<' || c == '>' || c == '\\' || c == '^' || c == '`'
                    || c == '{' || c == '|' || c == '}' || c == '#')
                {
                    out += '%';
                    out += digits[c >> 4];
                    out += digits[c & 15];
                }
                else
                    out += static_cast<char>(c);
            }
            return out;
        }

        /*
            The origin of the API of a link, and its code.  The links of the region of Europe
            (e.pcloud.link, e.pcloud.com) answer on eapi.pcloud.com, the others on
            api.pcloud.com; a loopback address is its own API, for the tests.
        */
        bool parse_link(const std::string& link, std::string& api, std::string& code)
        {
            if (link.size() > 512) return false;
            std::string origin, rest;
            const size_t scheme = link.find("://");
            if (scheme == std::string::npos) return false;
            const size_t slash = link.find('/', scheme + 3);
            if (slash == std::string::npos) return false;
            origin = lower(link.substr(0, slash));
            rest = link.substr(slash);
            if (origin == "https://u.pcloud.link" || origin == "https://my.pcloud.com") api = "https://api.pcloud.com";
            else if (origin == "https://e.pcloud.link" || origin == "https://e.pcloud.com")
                api = "https://eapi.pcloud.com";
            else if (origin.compare(0, 17, "http://127.0.0.1:") == 0 && origin.size() > 17 && origin.size() <= 22
                && origin.find_first_not_of("0123456789", 17) == std::string::npos)
                api = origin;
            else
                return false;
            const std::string prefix = "/publink/show?code=";
            if (rest.compare(0, prefix.size(), prefix) != 0) return false;
            code = rest.substr(prefix.size());
            const size_t end = code.find('&');
            if (end != std::string::npos) code.resize(end);
            if (code.size() < 8 || code.size() > 64) return false;
            for (char c : code)
                if (!alphanumeric(c)) return false;
            return true;
        }

        struct listed_file
        {
            long long id = 0;
            uint64_t size = 0;
        };

        struct direct_address
        {
            std::string url;
            clock_type::time_point at;
        };

        const std::chrono::minutes listing_life(10);
        const std::chrono::minutes address_life(30);

        std::mutex m;
        std::string api_origin;    // empty: no link
        std::string link_code;
        unsigned long long generation = 0;
        bool listed = false;
        clock_type::time_point listed_at;
        std::map<std::string, listed_file> listing;
        std::map<std::string, direct_address> addresses;

        // The files of folder, under the paths that prefix begins, its subfolders walked.
        void walk(const json& folder, const std::string& prefix, std::map<std::string, listed_file>& out, int depth)
        {
            if (depth > 8) return;
            for (const auto& c : folder["contents"].elements())
            {
                const std::string name = c["name"].as_string();
                if (name.empty() || name == "." || name == ".." || name.find('/') != std::string::npos) continue;
                if (c["isfolder"].as_bool(false)) walk(c, prefix + name + "/", out, depth + 1);
                else
                {
                    listed_file f;
                    f.id = c["fileid"].as_int(0);
                    const long long size = c["size"].as_int(-1);
                    if (f.id <= 0 || size < 0) continue;
                    f.size = static_cast<uint64_t>(size);
                    out[prefix + name] = f;
                }
            }
        }

        // A method of the API with its parameters, the code of the link and its password.
        bool call(const std::string& api, const std::string& code, const std::string& method, const std::string& params,
            json& answer, std::string& error)
        {
            std::string url = api + "/" + method + "?code=" + query_value(code) + params;
            const std::string password = link_password();
            if (!password.empty()) url += "&linkpassword=" + query_value(password);
            std::string body, e;
            if (!http_get(url, body, e, 16u << 20, 20))
            {
                error = "the public link cannot be reached: " + without_queries(e);
                return false;
            }
            answer = json::parse_or_null(body);
            const json& a = answer;
            if (!a.is_object())
            {
                error = "the public link gives an answer that is not JSON";
                return false;
            }
            const long long result = a["result"].as_int(-1);
            if (result != 0)
            {
                error = "the public link answers \"" + a["error"].as_string() + "\" (" + std::to_string(result) + ")";
                return false;
            }
            return true;
        }

        // The entry of path in the listing of the link, read again when it is older than its life or fresh asks.
        bool find_listed(const std::string& path, bool fresh, listed_file& f, std::string& error)
        {
            std::string api, code;
            unsigned long long g = 0;
            {
                std::lock_guard<std::mutex> lock(m);
                if (api_origin.empty())
                {
                    error = "no public link is set";
                    return false;
                }
                if (!fresh && listed && clock_type::now() - listed_at < listing_life)
                {
                    const auto it = listing.find(path);
                    if (it == listing.end())
                    {
                        error = "the public link holds no " + path;
                        return false;
                    }
                    f = it->second;
                    return true;
                }
                api = api_origin;
                code = link_code;
                g = generation;
            }
            json answer;
            if (!call(api, code, "showpublink", "", answer, error)) return false;
            std::map<std::string, listed_file> files;
            const json& a = answer;
            walk(a["metadata"], "", files, 0);
            std::lock_guard<std::mutex> lock(m);
            if (g != generation)
            {
                error = "the public link changed during its reading";
                return false;
            }
            listing.swap(files);
            listed = true;
            listed_at = clock_type::now();
            const auto it = listing.find(path);
            if (it == listing.end())
            {
                error = "the public link holds no " + path;
                return false;
            }
            f = it->second;
            return true;
        }
    }

    bool valid_public_link(const std::string& link)
    {
        std::string api, code;
        return parse_link(link, api, code);
    }

    void set_public_link(const std::string& link)
    {
        std::string api, code;
        const bool valid = parse_link(link, api, code);
        std::lock_guard<std::mutex> lock(m);
        api_origin = valid ? api : std::string();
        link_code = valid ? code : std::string();
        ++generation;
        listed = false;
        listing.clear();
        addresses.clear();
    }

    bool public_link_enabled()
    {
        std::lock_guard<std::mutex> lock(m);
        return !api_origin.empty();
    }

    bool public_link_has_password()
    {
        return masked_size > 0;
    }

    bool public_link_file(const std::string& path, std::string& url, uint64_t& size, std::string& error, bool fresh)
    {
        listed_file f;
        if (!find_listed(path, fresh, f, error)) return false;
        std::string api, code;
        unsigned long long g = 0;
        {
            std::lock_guard<std::mutex> lock(m);
            const auto it = addresses.find(path);
            if (!fresh && it != addresses.end() && clock_type::now() - it->second.at < address_life)
            {
                url = it->second.url;
                size = f.size;
                return true;
            }
            api = api_origin;
            code = link_code;
            g = generation;
        }
        json answer;
        if (!call(api, code, "getpublinkdownload", "&fileid=" + std::to_string(f.id) + "&forcedownload=1", answer,
                error))
            return false;
        // The first content server, a name of letters, digits, dots and hyphens (a port for a local API).
        const json& a = answer;
        const std::string host = a["hosts"].elements().empty() ? std::string() : a["hosts"].elements()[0].as_string();
        const std::string where = a["path"].as_string();
        bool valid_host = !host.empty() && host.size() <= 255;
        for (char c : host)
            valid_host = valid_host && (alphanumeric(c) || c == '.' || c == '-' || c == ':');
        if (!valid_host || where.empty() || where[0] != '/' || where.size() > 4096)
        {
            error = "the public link gives no address for " + path;
            return false;
        }
        const std::string scheme = api.compare(0, 8, "https://") == 0 ? "https://" : "http://";
        url = scheme + host + address_path(where);
        size = f.size;
        std::lock_guard<std::mutex> lock(m);
        if (g == generation)
        {
            direct_address kept;
            kept.url = url;
            kept.at = clock_type::now();
            addresses[path] = kept;
        }
        return true;
    }

    bool public_link_get(const std::string& path, std::string& body, std::string& error, size_t max_bytes)
    {
        // A second attempt asks the listing and the address again: an address of the cache may have expired.
        for (int attempt = 0; attempt < 2; ++attempt)
        {
            std::string url;
            uint64_t size = 0;
            if (!public_link_file(path, url, size, error, attempt > 0)) return false;
            if (size > max_bytes)
            {
                error = "the copy of " + path + " in the public link is larger than expected";
                return false;
            }
            std::string e;
            if (http_get(url, body, e, max_bytes))
            {
                if (body.size() == size) return true;
                error = "the copy of " + path + " in the public link does not have its listed size";
                continue;
            }
            // The address of the content server stays out of the message, as the queries of the API.
            const size_t at = e.find(url);
            if (at != std::string::npos) e.replace(at, url.size(), "the copy of " + path + " in the public link");
            error = without_queries(e);
            if (http_cancelled()) return false;
        }
        return false;
    }

    std::string public_link_model_path(const std::string& id, const std::string& url)
    {
        const std::string address = url.substr(0, url.find_first_of("?#"));
        const size_t slash = address.rfind('/');
        const std::string name = slash == std::string::npos ? address : address.substr(slash + 1);
        if (id.empty() || name.empty() || id.find('/') != std::string::npos) return std::string();
        return "models/" + id + "/" + name;
    }

    std::string about_public_link(const std::string& message)
    {
        return message.find("public link") == std::string::npos ? "public link: " + message : message;
    }

    std::string without_queries(const std::string& message)
    {
        std::string out = message;
        size_t from = 0;
        for (;;)
        {
            const size_t scheme = out.find("://", from);
            if (scheme == std::string::npos) break;
            size_t end = scheme + 3;
            while (end < out.size() && out[end] != ' ' && out[end] != '\t' && out[end] != '\n' && out[end] != '"'
                && out[end] != '\'')
                ++end;
            // The punctuation that follows an address in a message, "<address>: status 404", is not part of it.
            while (end > scheme + 3 && (out[end - 1] == ':' || out[end - 1] == ',' || out[end - 1] == ';'
                || out[end - 1] == ')'))
                --end;
            const size_t query = out.find('?', scheme + 3);
            if (query != std::string::npos && query < end)
            {
                out.erase(query, end - query);
                end = query;
            }
            from = end;
        }
        return out;
    }

}
