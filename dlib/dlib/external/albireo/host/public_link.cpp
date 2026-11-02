// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "public_link.h"

#include <chrono>
#include <map>
#include <mutex>

#include "../common/json.h"
#include "../common/log.h"
#include "http_fetch.h"
#include "masked_secret.h"
#include "public_link_secret.h"

namespace albireo
{

    namespace
    {
        using clock_type = std::chrono::steady_clock;

        // The link of the build, which no setting, route, document or message gives.
        const char* const build_link = "https://u.pcloud.link/publink/show?code=kZeNhSJZpcMygkcCvGXmq1MTvkCfBuSsx2GV";

        const unsigned char masked_password[] = ALBIREO_PUBLIC_LINK_SECRET;
        const size_t masked_size = ALBIREO_PUBLIC_LINK_SECRET_SIZE;

        // The bounds of a request to the interface: each wait for data, and the whole request.
        const int api_read_seconds = 20;
        const int api_total_seconds = 60;

        // The password, unmasked by the generator the configuration masked it with.
        std::string link_password()
        {
            return unmask_secret(masked_password, masked_size, 20261101);
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
            The origin of the interface of a link, and its code.  The links of the region of
            Europe (e.pcloud.link, e.pcloud.com) answer on eapi.pcloud.com, the others on
            api.pcloud.com; a loopback address is its own interface, for the tests.
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

        // A file of the listing: its identifier and size, or valid false for an entry without them.
        struct listed_file
        {
            long long id = 0;
            uint64_t size = 0;
            bool valid = false;
        };

        struct direct_address
        {
            std::string url;
            clock_type::time_point at;
        };

        const std::chrono::minutes listing_life(10);
        const std::chrono::minutes address_life(30);

        std::mutex m;
        bool configured = false;   // the link of the build read, until the tests replace it
        std::string api_origin;    // empty: no space set
        std::string link_code;
        unsigned long long generation = 0;
        bool listed = false;
        clock_type::time_point listed_at;
        std::map<std::string, listed_file> listing;
        std::map<std::string, direct_address> addresses;

        // Uses link, given to hide_in_messages() with the password; m is held.
        void use_link(const std::string& link)
        {
            std::string api, code;
            const bool valid = parse_link(link, api, code);
            api_origin = valid ? api : std::string();
            link_code = valid ? code : std::string();
            configured = true;
            ++generation;
            listed = false;
            listing.clear();
            addresses.clear();
            if (valid) hide_in_messages(code);
            hide_in_messages(link_password());
        }

        // The link of the build, read at the first use; m is held.
        void configure()
        {
            if (!configured) use_link(build_link);
        }

        // The files of folder, under the paths that prefix begins, its subfolders walked.
        void walk(const json& folder, const std::string& prefix, std::map<std::string, listed_file>& out, int depth)
        {
            if (depth > 8) return;
            for (const auto& c : folder["contents"].elements())
            {
                const std::string name = c["name"].as_string();
                if (name.empty() || name == "." || name == ".." || name.find('/') != std::string::npos) continue;
                if (c["isfolder"].as_bool(false))
                {
                    walk(c, prefix + name + "/", out, depth + 1);
                    continue;
                }
                // An identifier and a size of another type, negative, or beyond the integers, make the entry invalid.
                listed_file f;
                const json& id = c["fileid"];
                const json& size = c["size"];
                f.valid = id.is_integer() && id.as_int() > 0 && size.is_integer() && size.as_int() >= 0;
                f.id = f.valid ? id.as_int() : 0;
                f.size = f.valid ? static_cast<uint64_t>(size.as_int()) : 0;
                out[prefix + name] = f;
            }
        }

        // A text of an answer quoted in a message: 80 characters at most, its control characters left out.
        std::string quoted(const std::string& text)
        {
            std::string out;
            for (char c : without_publication_space(text))
            {
                if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) continue;
                if (out.size() >= 80)
                {
                    out += "...";
                    break;
                }
                out += c;
            }
            return out;
        }

        /*
            A method of the interface with its parameters, the code of the link and its password,
            for the file path of the space, which the messages name.  The answer is a JSON object
            whose result is 0, else the request fails.
        */
        bool call(const std::string& api, const std::string& code, const std::string& method, const std::string& params,
            const std::string& path, json& answer, std::string& error)
        {
            std::string url = api + "/" + method + "?code=" + query_value(code) + params;
            const std::string password = link_password();
            if (!password.empty()) url += "&linkpassword=" + query_value(password);
            std::string body, e;
            if (!http_get(url, body, e, 16u << 20, api_read_seconds, api_total_seconds))
            {
                error = "the publication space of Albireo cannot be reached for " + path + ": " + e;
                return false;
            }
            answer = json::parse_or_null(body);
            const json& a = answer;
            if (!a.is_object() || !a["result"].is_integer())
            {
                error = "the publication space of Albireo gives an answer that is not the one of its interface for "
                    + path;
                return false;
            }
            const long long result = a["result"].as_int();
            if (result != 0)
            {
                error = "the publication space of Albireo answers \"" + quoted(a["error"].as_string()) + "\" ("
                    + std::to_string(result) + ") for " + path;
                return false;
            }
            return true;
        }

        void set_status(link_status* status, link_status value)
        {
            if (status) *status = value;
        }

        // The entry of path in the listing, read again when it is older than its life or fresh asks.
        bool find_listed(const std::string& path, bool fresh, listed_file& f, std::string& error, link_status* status)
        {
            std::string api, code;
            unsigned long long g = 0;
            {
                std::lock_guard<std::mutex> lock(m);
                configure();
                if (api_origin.empty())
                {
                    error = path + " cannot be downloaded: no publication space is set";
                    set_status(status, link_status::unset);
                    return false;
                }
                if (!fresh && listed && clock_type::now() - listed_at < listing_life)
                {
                    const auto it = listing.find(path);
                    if (it == listing.end())
                    {
                        error = not_published(path);
                        set_status(status, link_status::missing);
                        return false;
                    }
                    f = it->second;
                    if (!f.valid)
                    {
                        error = "the publication space of Albireo lists " + path + " without a valid identifier "
                            "or size";
                        set_status(status, link_status::failed);
                        return false;
                    }
                    return true;
                }
                api = api_origin;
                code = link_code;
                g = generation;
            }
            json answer;
            set_status(status, link_status::failed);
            if (!call(api, code, "showpublink", "", path, answer, error)) return false;
            // A listing whose folder is no object with its contents is no listing: it is not kept.
            const json& metadata = static_cast<const json&>(answer)["metadata"];
            if (!metadata.is_object() || !metadata["contents"].is_array())
            {
                error = "the publication space of Albireo gives a listing that is not the one of its interface for "
                    + path;
                return false;
            }
            std::map<std::string, listed_file> files;
            walk(metadata, "", files, 0);
            std::lock_guard<std::mutex> lock(m);
            if (g != generation)
            {
                error = "the publication space of Albireo changed during its reading for " + path;
                return false;
            }
            listing.swap(files);
            listed = true;
            listed_at = clock_type::now();
            const auto it = listing.find(path);
            if (it == listing.end())
            {
                error = not_published(path);
                set_status(status, link_status::missing);
                return false;
            }
            f = it->second;
            if (!f.valid)
            {
                error = "the publication space of Albireo lists " + path + " without a valid identifier or size";
                return false;
            }
            set_status(status, link_status::ok);
            return true;
        }
    }

    bool public_link_enabled()
    {
        std::lock_guard<std::mutex> lock(m);
        configure();
        return !api_origin.empty();
    }

    bool public_link_has_password()
    {
        return masked_size > 0;
    }

    void public_link_for_tests(const std::string& link)
    {
        std::lock_guard<std::mutex> lock(m);
        use_link(link);
    }

    bool public_link_file(const std::string& path, std::string& url, uint64_t& size, std::string& error, bool fresh,
        link_status* status)
    {
        set_status(status, link_status::ok);
        listed_file f;
        if (!find_listed(path, fresh, f, error, status)) return false;
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
        set_status(status, link_status::failed);
        if (!call(api, code, "getpublinkdownload", "&fileid=" + std::to_string(f.id) + "&forcedownload=1", path,
                answer, error))
            return false;
        // The first content server, a name of letters, digits, dots and hyphens (a port for a local interface).
        const json& a = answer;
        const json& first = a["hosts"][0];
        const std::string host = first.as_string();
        const std::string where = a["path"].as_string();
        bool valid_host = !host.empty() && host.size() <= 255;
        for (char c : host)
            valid_host = valid_host && (alphanumeric(c) || c == '.' || c == '-' || c == ':');
        if (!valid_host || where.empty() || where[0] != '/' || where.size() > 4096)
        {
            error = "the publication space of Albireo gives no address for " + path;
            return false;
        }
        const std::string scheme = api.compare(0, 8, "https://") == 0 ? "https://" : "http://";
        url = scheme + host + address_path(where);
        size = f.size;
        set_status(status, link_status::ok);
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

    bool public_link_get(const std::string& path, std::string& body, std::string& error, size_t max_bytes,
        link_status* status)
    {
        /*
            A second attempt asks the listing and the address again: an address of the cache may
            have expired, and a file the listing of the cache lacks may have been published since.
        */
        for (int attempt = 0; attempt < 2; ++attempt)
        {
            std::string url;
            uint64_t size = 0;
            link_status found = link_status::ok;
            if (!public_link_file(path, url, size, error, attempt > 0, &found))
            {
                set_status(status, found);
                if (found == link_status::missing && attempt == 0 && !http_cancelled()) continue;
                return false;
            }
            set_status(status, link_status::failed);
            if (size > max_bytes)
            {
                error = "the copy of " + path + " in the publication space of Albireo is larger than expected";
                return false;
            }
            std::string e;
            if (http_get(url, body, e, max_bytes))
            {
                if (body.size() == size)
                {
                    set_status(status, link_status::ok);
                    return true;
                }
                error = "the copy of " + path + " in the publication space of Albireo does not have its listed size";
                continue;
            }
            error = "the copy of " + path + " in the publication space of Albireo cannot be downloaded: " + e;
            if (http_cancelled()) return false;
        }
        return false;
    }

    std::string not_published(const std::string& path)
    {
        return "the publication space of Albireo does not hold " + path + " yet";
    }

    std::string public_link_model_path(const std::string& id, const std::string& url)
    {
        const std::string address = url.substr(0, url.find_first_of("?#"));
        const size_t slash = address.rfind('/');
        const std::string name = slash == std::string::npos ? address : address.substr(slash + 1);
        if (id.empty() || name.empty() || id.find('/') != std::string::npos) return std::string();
        return "models/" + id + "/" + name;
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
