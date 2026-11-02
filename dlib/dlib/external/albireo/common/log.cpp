// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "log.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <vector>

#include "platform.h"

namespace albireo
{

    namespace
    {
        const char* const publication_space = "the publication space of Albireo";

        // The texts hide_in_messages() was given, under hidden_m.
        std::mutex hidden_m;
        std::vector<std::string>& hidden_texts()
        {
            static std::vector<std::string> texts;
            return texts;
        }

        bool word_end(char c)
        {
            switch (c)
            {
            case ' ': case '\t': case '\r': case '\n': case '\v': case '\f': case '"': case '\'': case '`':
            case '<': case '>': case '(': case ')': case '[': case ']': case '{': case '}': case ',': case ';':
            case '|': case '\\':
                return true;
            default:
                return false;
            }
        }

        // The punctuation that ends a sentence or a clause after a word, which stays after its replacement.
        bool final_punctuation(char c)
        {
            return c == '.' || c == ':' || c == '!' || c == '?';
        }

        // Whether text holds word, its ASCII letters compared whatever their case (word in lowercase).
        bool holds_folded(const std::string& text, size_t from, size_t to, const char* word)
        {
            const size_t n = std::char_traits<char>::length(word);
            for (size_t i = from; i + n <= to; ++i)
            {
                size_t k = 0;
                while (k < n)
                {
                    char c = text[i + k];
                    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
                    if (c != word[k]) break;
                    ++k;
                }
                if (k == n) return true;
            }
            return false;
        }

        // Whether the bytes [from, to) of text name the publication space; hidden_m is held.
        bool names_space(const std::string& text, size_t from, size_t to)
        {
            if (holds_folded(text, from, to, "pcloud") || holds_folded(text, from, to, "publink")) return true;
            for (const auto& t : hidden_texts())
            {
                const size_t at = text.find(t, from);
                if (at != std::string::npos && at + t.size() <= to) return true;
            }
            return false;
        }
    }

    std::string without_publication_space(const std::string& text)
    {
        std::lock_guard<std::mutex> lock(hidden_m);
        if (!names_space(text, 0, text.size())) return text;
        std::string out;
        size_t i = 0;
        while (i < text.size())
        {
            if (word_end(text[i]))
            {
                out += text[i++];
                continue;
            }
            size_t end = i;
            while (end < text.size() && !word_end(text[end])) ++end;
            size_t last = end;
            while (last > i && final_punctuation(text[last - 1])) --last;
            if (last > i && names_space(text, i, last)) out += publication_space;
            else out.append(text, i, last - i);
            out.append(text, last, end - last);
            i = end;
        }
        return out;
    }

    bool names_publication_space(const std::string& text)
    {
        std::lock_guard<std::mutex> lock(hidden_m);
        return names_space(text, 0, text.size());
    }

    void hide_in_messages(const std::string& secret)
    {
        if (secret.size() < 6) return;
        std::lock_guard<std::mutex> lock(hidden_m);
        std::vector<std::string>& texts = hidden_texts();
        for (const auto& t : texts)
            if (t == secret) return;
        texts.push_back(secret);
    }

    void hide_publication_space_in_file(const std::string& path)
    {
        std::string text;
        if (!read_file(path, text, 64u << 20) || !names_publication_space(text)) return;
        write_file_atomic(path, without_publication_space(text));
    }

    namespace
    {
        // The members whose values tell an error.
        bool error_member(const std::string& name)
        {
            return name == "error" || name == "errors" || name == "args";
        }

        bool errors_name_space(const json& j, bool inside)
        {
            if (j.is_string()) return inside && names_publication_space(j.as_string());
            for (const auto& kv : j.items())
                if (errors_name_space(kv.second, inside || error_member(kv.first))) return true;
            for (const auto& e : j.elements())
                if (errors_name_space(e, inside)) return true;
            return false;
        }

        void hide_space_in_errors(json& j, bool inside)
        {
            if (j.is_string() && inside) j = without_publication_space(j.as_string());
            else if (j.is_object())
                for (auto& kv : j.items()) hide_space_in_errors(kv.second, inside || error_member(kv.first));
            else if (j.is_array())
                for (auto& e : j.elements()) hide_space_in_errors(e, inside);
        }
    }

    bool errors_name_publication_space(const json& document)
    {
        return errors_name_space(document, false);
    }

    void hide_publication_space_in_errors(json& document)
    {
        hide_space_in_errors(document, false);
    }

    log_file& log_file::instance()
    {
        static log_file f;
        return f;
    }

    void log_file::open(const std::string& path, uint64_t max_bytes)
    {
        std::lock_guard<std::mutex> lock(m_);
        path_ = path;
        max_bytes_ = max_bytes;
        make_dirs(parent_path(path));
        // The lines a version before this one wrote may name the publication space.
        hide_publication_space_in_file(path);
        hide_publication_space_in_file(path + ".1");
    }

    void log_file::write(log_level level, const std::string& given)
    {
        if (static_cast<int>(level) < static_cast<int>(level_)) return;
        const std::string message = without_publication_space(given);
        static const char* names[] = { "debug", "info", "warning", "error" };
        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        const long ms = static_cast<long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
        std::tm tm;
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char stamp[64];
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm);
        char ms_part[8];
        std::snprintf(ms_part, sizeof(ms_part), ".%03ld", ms);
        const std::string line = std::string(stamp) + ms_part + " [" + names[static_cast<int>(level)] + "] " + message;

        std::lock_guard<std::mutex> lock(m_);
        recent_.push_back(line);
        while (recent_.size() > 500) recent_.pop_front();
        if (console_)
        {
            std::fputs(line.c_str(), stderr);
            std::fputc('\n', stderr);
        }
        if (path_.empty()) return;
        if (file_size(path_) > max_bytes_) rename_file(path_, path_ + ".1");
#if defined(_WIN32)
        std::ofstream out(widen(path_).c_str(), std::ios::app | std::ios::binary);
#else
        std::ofstream out(path_, std::ios::app | std::ios::binary);
#endif
        if (out)
        {
            out << line;
#if defined(_WIN32)
            out << "\r\n";
#else
            out << "\n";
#endif
        }
    }

    std::deque<std::string> log_file::recent() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return recent_;
    }

}
