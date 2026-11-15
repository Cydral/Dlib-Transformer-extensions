// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
#include <dlib/external/httplib/httplib.h>

#include "pseudonyms.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "../common/log.h"
#include "../common/platform.h"
#include "certificates.h"
#include "crypto.h"
#include "http_fetch.h"

namespace albireo
{

    namespace
    {
        std::mutex test_origin_m;
        std::string test_origin;

        // The reasons Pseudalis is not reached that the state names by a code, which the icon translates.
        const char* const pseudalis_not_installed = "Pseudalis is not installed";
        const char* const pseudalis_no_authority = "the local authority of Pseudalis is missing";
        const char* const pseudalis_unprotected = "the data folder of Pseudalis is not protected";

        std::string pseudalis_code(const std::string& reason)
        {
            if (reason == pseudalis_not_installed) return "not_installed";
            if (reason == pseudalis_no_authority) return "no_authority";
            if (reason == pseudalis_unprotected) return "unprotected";
            return "failed";
        }

        // Thrown from the socket of a client about to connect once the requests of the service are cancelled.
        struct request_cancelled
        {
        };

        // Whether the code point cp counts as a character of a word: a letter, a digit or '_'.
        bool word_code_point(uint32_t cp)
        {
            if (cp < 0x80) return (cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')
                || cp == '_';
            // The punctuation, the spaces and the symbols of Latin-1, of the general punctuation and of CJK.
            if (cp < 0xC0 || cp == 0xD7 || cp == 0xF7) return false;
            if (cp >= 0x2000 && cp <= 0x2BFF) return false;
            if (cp >= 0x3000 && cp <= 0x303F) return false;
            if (cp >= 0xFE30 && cp <= 0xFE4F) return false;
            if (cp >= 0xFF00 && cp <= 0xFF0F) return false;
            return true;
        }

        // The code point that begins at s[i], with its length in n; U+FFFD for an invalid sequence.
        uint32_t code_point_at(const std::string& s, size_t i, size_t& n)
        {
            const unsigned char c = static_cast<unsigned char>(s[i]);
            n = 1;
            if (c < 0x80) return c;
            size_t len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
            if (len == 0 || i + len > s.size()) return 0xFFFD;
            uint32_t cp = c & (0x7F >> len);
            for (size_t k = 1; k < len; ++k)
            {
                const unsigned char d = static_cast<unsigned char>(s[i + k]);
                if ((d & 0xC0) != 0x80) return 0xFFFD;
                cp = (cp << 6) | (d & 0x3F);
            }
            n = len;
            return cp;
        }

        // The last code point of s, 0 when s is empty.
        uint32_t last_code_point(const std::string& s)
        {
            if (s.empty()) return 0;
            size_t i = s.size() - 1;
            while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80 && s.size() - i < 4) --i;
            size_t n = 0;
            const uint32_t cp = code_point_at(s, i, n);
            return i + n == s.size() ? cp : 0xFFFD;
        }

        bool begins_with_word(const std::string& token)
        {
            size_t n = 0;
            return !token.empty() && word_code_point(code_point_at(token, 0, n));
        }

        bool ends_with_word(const std::string& token)
        {
            return word_code_point(last_code_point(token));
        }

        // The words of a text separated by spaces.
        std::vector<std::string> words_of(const std::string& text)
        {
            std::vector<std::string> words;
            size_t i = 0;
            while (i < text.size())
            {
                const size_t b = text.find_first_not_of(' ', i);
                if (b == std::string::npos) break;
                const size_t e = std::min(text.find(' ', b), text.size());
                words.push_back(text.substr(b, e - b));
                i = e;
            }
            return words;
        }

        // The text of a tag [[TYPE:body#CHECK]], empty when token is no tag.
        std::string tag_body(const std::string& token)
        {
            if (token.size() < 8 || token.compare(0, 2, "[[") != 0 || token.compare(token.size() - 2, 2, "]]") != 0)
                return std::string();
            const size_t colon = token.find(':');
            const size_t hash = token.rfind('#');
            if (colon == std::string::npos || hash == std::string::npos || hash <= colon + 1) return std::string();
            return token.substr(colon + 1, hash - colon - 1);
        }

        std::string error_of(const httplib::Result& r, const json& answer)
        {
            if (!r) return "Pseudalis does not answer";
            const std::string message = answer["error"].is_string() ? answer["error"].as_string()
                : answer["error"]["message"].as_string();
            return "Pseudalis refuses the text (HTTP " + std::to_string(r->status) + ")"
                + (message.empty() ? std::string() : ": " + message.substr(0, 300));
        }
    }

    // ----------------------------------------------------------------------------------------

    namespace
    {
        // Stops a client from a thread of its own, or in place when the system refuses the thread.
        void stop_client(const std::shared_ptr<httplib::Client>& c)
        {
            try
            {
                std::thread([c]() { c->stop(); }).detach();
            }
            catch (const std::exception&)
            {
                c->stop();
            }
        }
    }

    void cancellable::cancel()
    {
        cancelled_ = true;
        std::shared_ptr<httplib::Client> c;
        {
            std::lock_guard<std::mutex> lock(m_);
            c = client_;
        }
        if (c) stop_client(c);
    }

    void cancellable::hold(const std::shared_ptr<httplib::Client>& c)
    {
        {
            std::lock_guard<std::mutex> lock(m_);
            client_ = c;
        }
        // Either cancel() finds the client, or the client finds the flag.
        if (cancelled_) stop_client(c);
    }

    void cancellable::release()
    {
        std::lock_guard<std::mutex> lock(m_);
        client_.reset();
    }

    // ----------------------------------------------------------------------------------------

    std::string pseudalis_client::pseudalis_data_dir()
    {
#if defined(_WIN32)
        HKEY key;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"Software\\Pseudalis", 0, KEY_READ | KEY_WOW64_64KEY, &key)
            == ERROR_SUCCESS)
        {
            wchar_t buf[1024];
            DWORD size = sizeof(buf), type = 0;
            const LONG rc = RegQueryValueExW(key, L"DataDir", nullptr, &type, reinterpret_cast<BYTE*>(buf), &size);
            RegCloseKey(key);
            if (rc == ERROR_SUCCESS && type == REG_SZ && size >= sizeof(wchar_t))
            {
                const std::string dir = narrow(std::wstring(buf, size / sizeof(wchar_t) - 1));
                if (dir_exists(dir)) return dir;
            }
        }
        /*
            Only the folder that the installer of Pseudalis recorded, which only an administrator
            can record: any user can create a folder Pseudalis in %ProgramData%.
        */
        return std::string();
#else
        const char* home = std::getenv("PSEUDALIS_HOME");
        if (home && *home) return dir_exists(home) ? std::string(home) : std::string();
        const char* user = std::getenv("HOME");
        const std::string dir = user ? join_path(user, ".local/share/pseudalis") : std::string();
        return dir_exists(dir) ? dir : std::string();
#endif
    }

    void pseudalis_client::endpoint_for_tests(const std::string& origin)
    {
        std::lock_guard<std::mutex> lock(test_origin_m);
        test_origin = origin;
    }

    bool pseudalis_client::probe(endpoint& e, std::string& reason)
    {
        {
            std::lock_guard<std::mutex> lock(test_origin_m);
            if (!test_origin.empty())
            {
                e.origin = test_origin;
                e.ca.clear();
                return true;
            }
        }
        const std::string dir = pseudalis_data_dir();
        if (dir.empty())
        {
            reason = pseudalis_not_installed;
            return false;
        }
        const std::string settings_path = join_path(dir, "settings.json");
        e.ca = join_path(join_path(dir, "certificates"), "pseudalis-authority.crt");
        if (!file_exists(e.ca))
        {
            reason = pseudalis_no_authority;
            return false;
        }
        /*
            A folder, an authority or settings that another account than SYSTEM or the
            administrators could have written are not those of Pseudalis: the texts would reach
            whoever placed them.
        */
        if (!folder_admin_only(dir) || !folder_admin_only(join_path(dir, "certificates")) || !owned_by_system(e.ca)
            || (file_exists(settings_path) && !owned_by_system(settings_path)))
        {
            reason = pseudalis_unprotected;
            return false;
        }
        int port = 8444;
        std::string text;
        json settings;
        if (read_file(settings_path, text, 1u << 20) && json::parse(text, settings))
        {
            const long long p = settings["port"].as_int(8444);
            port = p >= 1 && p <= 65535 ? static_cast<int>(p) : 8444;
        }
        e.origin = "https://localhost:" + std::to_string(port);
        return true;
    }

    bool pseudalis_client::reach(endpoint& e, std::string& error)
    {
        /*
            The state is checked again once a minute, by one caller at a time and outside the
            lock, the others waiting for its result, which reads a few files.
        */
        {
            std::unique_lock<std::mutex> lock(m_);
            checked_cv_.wait(lock, [this]() { return !checking_; });
            if (checked_ && now_unix() - checked_ < 60)
            {
                e = endpoint_;
                error = reason_;
                return found_;
            }
            checking_ = true;
        }
        endpoint found_at;
        std::string why;
        bool found = false;
        // Whatever the search throws, the callers that wait for it get its result.
        try
        {
            found = probe(found_at, why);
        }
        catch (const std::exception& x)
        {
            why = x.what();
        }
        catch (...)
        {
            why = "the search of Pseudalis failed";
        }
        std::lock_guard<std::mutex> lock(m_);
        checking_ = false;
        checked_ = now_unix();
        found_ = found;
        endpoint_ = found_at;
        reason_ = why;
        e = found_at;
        error = why;
        checked_cv_.notify_all();
        return found;
    }

    json pseudalis_client::status()
    {
        endpoint e;
        std::string reason;
        const bool found = reach(e, reason);
        json j = json::object();
        j["found"] = found;
        const size_t colon = e.origin.rfind(':');
        j["port"] = found && colon != std::string::npos ? std::atoll(e.origin.c_str() + colon + 1) : 0LL;
        j["reason"] = found ? std::string() : reason;
        j["code"] = found ? std::string() : pseudalis_code(reason);
        return j;
    }

    bool pseudalis_client::post(const endpoint& e, const std::string& path, const json& body, cancellable* cancel,
        json& answer, std::string& error)
    {
        const std::string payload = body.dump();
        const auto start = std::chrono::steady_clock::now();
        for (;;)
        {
            const auto c = std::make_shared<httplib::Client>(e.origin);
            if (!e.ca.empty())
            {
                // The certificate names localhost, reached on 127.0.0.1 alone, where no other account listens.
                c->set_hostname_addr_map({ { "localhost", "127.0.0.1" } });
                c->enable_system_ca(false);
                c->set_ca_cert_path(e.ca);
                c->enable_server_certificate_verification(true);
            }
            c->set_connection_timeout(3);
            // The detection of a long text takes its time on the processor.
            c->set_read_timeout(300);
            c->set_write_timeout(60);
            c->set_keep_alive(false);
            /*
                Pseudalis answers on this computer, in the offline mode too, which cancels the
                downloads: the cancellation of its answer ends the request, and the stop of the
                service through its registration (http_cancel_all()).
            */
            c->set_socket_options([cancel](socket_t sock) {
                if (!(cancel && cancel->cancelled())) return;
                httplib::detail::close_socket(sock);
                throw request_cancelled();
            });
            http_registration registered(c);
            if (cancel) cancel->hold(c);
            httplib::Result r;
            bool thrown = false;
            if (!(cancel && cancel->cancelled()))
            {
                try
                {
                    r = c->Post(path, payload, "application/json");
                }
                catch (...)
                {
                    thrown = true;
                }
            }
            if (cancel) cancel->release();
            if (thrown || (cancel && cancel->cancelled()))
            {
                error = "the request to Pseudalis was cancelled";
                return false;
            }
            answer = r ? json::parse_or_null(r->body) : json();
            if (r && r->status == 200 && answer.is_object()) return true;
            /*
                Pseudalis busy with files answers 503 for a while: the text waits for it, two
                minutes at most, unless its request ends.
            */
            const bool busy = r && r->status == 503;
            if (!busy || std::chrono::steady_clock::now() - start > std::chrono::seconds(120))
            {
                error = busy ? std::string("Pseudalis is busy with other files: try again in a moment")
                    : error_of(r, answer);
                if (!r)
                {
                    // Found again at the next text: it may have stopped, or moved to another port.
                    std::lock_guard<std::mutex> lock(m_);
                    checked_ = 0;
                }
                return false;
            }
            for (int i = 0; i < 20 && !(cancel && cancel->cancelled()); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    bool pseudalis_client::protect(const std::string& text, bool tags, cancellable* cancel, std::string& out,
        std::vector<std::pair<std::string, std::string>>& values, std::string& error)
    {
        values.clear();
        endpoint e;
        if (!reach(e, error))
        {
            error = "the pseudonymization is unavailable: " + error;
            return false;
        }
        json body = json::object();
        body["text"] = text;
        body["tags"] = tags;
        json answer;
        if (!post(e, "/v1/text/anonymize", body, cancel, answer, error)) return false;
        out = answer["text"].as_string();
        // The tags or substitutes written, each once, in the order of the text.
        std::vector<std::string> tokens;
        for (const json& span : answer["tags"].elements())
        {
            const long long b = span["begin"].as_int(-1), en = span["end"].as_int(-1);
            if (b < 0 || en <= b || static_cast<size_t>(en) > out.size()) continue;
            const std::string token = out.substr(static_cast<size_t>(b), static_cast<size_t>(en - b));
            if (std::find(tokens.begin(), tokens.end(), token) == tokens.end()) tokens.push_back(token);
        }
        if (tokens.empty()) return true;
        /*
            Their values, which the restoration of the tokens one per line gives back at the places
            it names: the substitutes written without tags need their list, the tags carry their
            values.
        */
        std::string joined;
        for (const std::string& t : tokens) joined += t + "\n";
        json restore = json::object();
        restore["text"] = joined;
        restore["tags"] = tags;
        if (answer["substitutions"].is_array()) restore["substitutions"] = answer["substitutions"];
        json restored;
        if (!post(e, "/v1/text/restore", restore, cancel, restored, error)) return false;
        const std::string back = restored["text"].as_string();
        const json& places = restored["restored"];
        if (places.size() == tokens.size())
        {
            size_t last = 0;
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                const long long b = places[i]["begin"].as_int(-1), en = places[i]["end"].as_int(-1);
                if (b < static_cast<long long>(last) || en <= b || static_cast<size_t>(en) > back.size()) break;
                values.emplace_back(tokens[i], back.substr(static_cast<size_t>(b), static_cast<size_t>(en - b)));
                last = static_cast<size_t>(en);
            }
            if (values.size() == tokens.size()) return true;
            values.clear();
        }
        // Otherwise the lines of the restored text, when each token gave back one line.
        std::vector<std::string> lines;
        size_t i = 0;
        while (i < back.size())
        {
            const size_t eol = std::min(back.find('\n', i), back.size());
            lines.push_back(back.substr(i, eol - i));
            i = eol + 1;
        }
        bool one_line = true;
        for (const std::string& t : tokens) one_line = one_line && t.find('\n') == std::string::npos;
        if (one_line && lines.size() == tokens.size())
        {
            for (size_t k = 0; k < tokens.size(); ++k)
                if (lines[k] != tokens[k]) values.emplace_back(tokens[k], lines[k]);
            return true;
        }
        /*
            The protected text stays valid and leaves; the answer then shows the substitutes that
            the restoration did not match.
        */
        log_warning("relay: the values of " + std::to_string(tokens.size()) + " substitutes cannot be matched");
        return true;
    }

    // ----------------------------------------------------------------------------------------

    pseudonym_memory::pseudonym_memory() {}

    pseudonym_memory::pseudonym_memory(const limits& l) : limits_(l) {}

    pseudonym_memory::~pseudonym_memory()
    {
        clear();
    }

    int64_t pseudonym_memory::now() const
    {
        return time_for_tests_ ? time_for_tests_ : now_unix();
    }

    void pseudonym_memory::set_time_for_tests(int64_t t)
    {
        std::lock_guard<std::mutex> lock(m_);
        time_for_tests_ = t;
    }

    void pseudonym_memory::drop_text(std::map<std::string, text_entry>::iterator it)
    {
        text_bytes_ -= it->second.bytes;
        text_lru_.erase(it->second.lru);
        texts_.erase(it);
    }

    void pseudonym_memory::drop_value(std::map<std::string, value_entry>::iterator it)
    {
        value_bytes_ -= it->second.bytes;
        value_lru_.erase(it->second.lru);
        wipe_secret(it->second.value);
        values_.erase(it);
    }

    void pseudonym_memory::trim()
    {
        const int64_t t = now();
        while (!text_lru_.empty())
        {
            auto it = texts_.find(text_lru_.back());
            if (texts_.size() <= limits_.texts && text_bytes_ <= limits_.text_bytes
                && t - it->second.used < limits_.max_idle)
                break;
            drop_text(it);
        }
        while (!value_lru_.empty())
        {
            auto it = values_.find(value_lru_.back());
            if (values_.size() <= limits_.values && value_bytes_ <= limits_.value_bytes
                && t - it->second.used < limits_.max_idle)
                break;
            drop_value(it);
        }
    }

    bool pseudonym_memory::protect(const std::string& text, bool tags, pseudalis_client& client,
        cancellable* cancel, std::string& out, std::map<std::string, std::string>& values, std::string& error)
    {
        const std::string key = sha256_hex(std::string(tags ? "1" : "0") + text);
        {
            std::lock_guard<std::mutex> lock(m_);
            trim();
            auto it = texts_.find(key);
            if (it != texts_.end())
            {
                // A text whose values all remain: the values of the others were forgotten since.
                bool whole = true;
                for (const std::string& token : it->second.tokens) whole = whole && values_.count(token) > 0;
                if (whole)
                {
                    const int64_t t = now();
                    it->second.used = t;
                    text_lru_.splice(text_lru_.begin(), text_lru_, it->second.lru);
                    for (const std::string& token : it->second.tokens)
                    {
                        value_entry& v = values_[token];
                        v.used = t;
                        value_lru_.splice(value_lru_.begin(), value_lru_, v.lru);
                        values[token] = v.value;
                    }
                    out = it->second.out;
                    return true;
                }
                drop_text(it);
            }
        }
        std::vector<std::pair<std::string, std::string>> found;
        if (!client.protect(text, tags, cancel, out, found, error)) return false;
        std::lock_guard<std::mutex> lock(m_);
        const int64_t t = now();
        text_entry entry;
        entry.out = out;
        entry.used = t;
        entry.bytes = key.size() + out.size();
        for (auto& f : found)
        {
            values[f.first] = f.second;
            entry.tokens.push_back(f.first);
            entry.bytes += f.first.size();
            auto v = values_.find(f.first);
            if (v == values_.end())
            {
                value_entry ve;
                ve.value = f.second;
                ve.bytes = f.first.size() + f.second.size();
                value_lru_.push_front(f.first);
                ve.lru = value_lru_.begin();
                ve.used = t;
                value_bytes_ += ve.bytes;
                values_.emplace(f.first, std::move(ve));
            }
            else
            {
                if (v->second.value != f.second)
                {
                    value_bytes_ = value_bytes_ - v->second.bytes + f.first.size() + f.second.size();
                    wipe_secret(v->second.value);
                    v->second.value = f.second;
                    v->second.bytes = f.first.size() + f.second.size();
                }
                v->second.used = t;
                value_lru_.splice(value_lru_.begin(), value_lru_, v->second.lru);
            }
            wipe_secret(f.second);
        }
        // A text larger than the memory itself is protected each time it is sent.
        if (entry.bytes <= limits_.text_bytes)
        {
            auto old = texts_.find(key);
            if (old != texts_.end()) drop_text(old);
            text_lru_.push_front(key);
            entry.lru = text_lru_.begin();
            text_bytes_ += entry.bytes;
            texts_.emplace(key, std::move(entry));
        }
        trim();
        return true;
    }

    void pseudonym_memory::clear()
    {
        std::lock_guard<std::mutex> lock(m_);
        for (auto& v : values_) wipe_secret(v.second.value);
        values_.clear();
        value_lru_.clear();
        value_bytes_ = 0;
        texts_.clear();
        text_lru_.clear();
        text_bytes_ = 0;
    }

    json pseudonym_memory::stats()
    {
        std::lock_guard<std::mutex> lock(m_);
        trim();
        json j = json::object();
        j["texts"] = static_cast<long long>(texts_.size());
        j["values"] = static_cast<long long>(values_.size());
        j["bytes"] = static_cast<long long>(text_bytes_ + value_bytes_);
        return j;
    }

    // ----------------------------------------------------------------------------------------

    stream_restorer::stream_restorer(const std::map<std::string, std::string>& values) : by_byte_(256)
    {
        for (const auto& kv : values)
        {
            // A substitute of one byte would replace letters: its value stays out.
            if (kv.first.size() >= 2 && kv.first != kv.second) add(kv.first, kv.second, true);
            const std::string body = tag_body(kv.first);
            if (body.size() >= 4 && body != kv.second) add(body, kv.second, true);
        }
        // The words of the substitutes of several words, each with the word of its value.
        for (const auto& kv : values)
        {
            const std::string body = tag_body(kv.first);
            const std::vector<std::string> from = words_of(body.empty() ? kv.first : body);
            const std::vector<std::string> to = words_of(kv.second);
            if (from.size() < 2 || from.size() != to.size()) continue;
            for (size_t i = 0; i < from.size(); ++i)
                if (from[i].size() >= 4 && from[i] != to[i]) add(from[i], to[i], false);
        }
        for (size_t i = 0; i < tokens_.size(); ++i)
            if (!ambiguous_[i]) by_byte_[static_cast<unsigned char>(tokens_[i].first[0])].push_back(i);
        for (auto& list : by_byte_)
            std::stable_sort(list.begin(), list.end(),
                [this](size_t a, size_t b) { return tokens_[a].first.size() > tokens_[b].first.size(); });
    }

    stream_restorer::~stream_restorer()
    {
        for (auto& t : tokens_) wipe_secret(t.second);
        wipe_secret(pending_);
        wipe_secret(tail_);
    }

    void stream_restorer::add(const std::string& token, const std::string& value, bool whole)
    {
        auto it = index_.find(token);
        if (it == index_.end())
        {
            index_[token] = tokens_.size();
            tokens_.emplace_back(token, value);
            ambiguous_.push_back(false);
            whole_.push_back(whole);
            return;
        }
        // A whole substitute keeps its value; a word that stands for two values stands for none.
        const size_t i = it->second;
        if (whole_[i] || tokens_[i].second == value) return;
        ambiguous_[i] = true;
    }

    std::string stream_restorer::feed(const std::string& chunk)
    {
        pending_ += chunk;
        return run(false);
    }

    std::string stream_restorer::finish()
    {
        std::string out = run(true);
        pending_.clear();
        return out;
    }

    std::string stream_restorer::run(bool end)
    {
        std::string out;
        size_t i = 0;
        const size_t n = pending_.size();
        while (i < n)
        {
            const auto& candidates = by_byte_[static_cast<unsigned char>(pending_[i])];
            bool matched = false, wait = false;
            if (!candidates.empty())
            {
                const bool word_before = word_code_point(last_code_point(tail_));
                for (size_t c : candidates)
                {
                    const std::string& token = tokens_[c].first;
                    if (word_before && begins_with_word(token)) continue;
                    const size_t left = n - i;
                    if (left < token.size())
                    {
                        // The start of a longer substitute, which the next chunk may complete.
                        if (!end && token.compare(0, left, pending_, i, left) == 0)
                        {
                            wait = true;
                            break;
                        }
                        continue;
                    }
                    if (pending_.compare(i, token.size(), token) != 0) continue;
                    if (ends_with_word(token))
                    {
                        if (i + token.size() == n && !end)
                        {
                            // The character that follows tells whether the word ends there.
                            wait = true;
                            break;
                        }
                        size_t len = 0;
                        if (i + token.size() < n && word_code_point(code_point_at(pending_, i + token.size(), len)))
                            continue;
                    }
                    out += tokens_[c].second;
                    tail_ = token.substr(token.size() >= 4 ? token.size() - 4 : 0);
                    i += token.size();
                    matched = true;
                    break;
                }
            }
            if (wait) break;
            if (matched) continue;
            out += pending_[i];
            tail_ += pending_[i];
            if (tail_.size() > 8) tail_.erase(0, tail_.size() - 4);
            ++i;
        }
        pending_.erase(0, i);
        return out;
    }

}
