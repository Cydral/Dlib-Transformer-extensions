// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_XML_SCANNER_H_
#define DLIB_XML_SCANNER_H_

#include "xml_scanner_abstract.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "../error.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class xml_error : public error
    {
    public:
        xml_error(const std::string& message) : error(message) {}
    };

    struct xml_attribute
    {
        std::string name;
        std::size_t value_begin = 0;
        std::size_t value_end = 0;
    };

    struct xml_token
    {
        enum token_kind { start_tag, end_tag, empty_tag, text, cdata, comment, processing_instruction, doctype };
        token_kind kind = text;
        std::size_t begin = 0;
        std::size_t end = 0;
        std::size_t content_begin = 0;
        std::size_t content_end = 0;
        std::string name;
        std::vector<xml_attribute> attributes;
    };

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline bool xml_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
    }

    inline std::vector<xml_token> xml_tokenize(const std::string& s)
    {
        using impl::xml_space;
        std::vector<xml_token> out;
        const std::size_t n = s.size();
        std::size_t i = 0;
        while (i < n)
        {
            xml_token t;
            t.begin = i;
            if (s[i] != '<')
            {
                const std::size_t j = s.find('<', i);
                t.kind = xml_token::text;
                t.end = (j == std::string::npos) ? n : j;
                t.content_begin = i;
                t.content_end = t.end;
                out.push_back(t);
                i = t.end;
                continue;
            }
            if (s.compare(i, 4, "<!--") == 0)
            {
                const std::size_t j = s.find("-->", i + 4);
                if (j == std::string::npos) throw xml_error("xml_tokenize(): unterminated comment");
                t.kind = xml_token::comment;
                t.end = j + 3;
            }
            else if (s.compare(i, 9, "<![CDATA[") == 0)
            {
                const std::size_t j = s.find("]]>", i + 9);
                if (j == std::string::npos) throw xml_error("xml_tokenize(): unterminated CDATA section");
                t.kind = xml_token::cdata;
                t.end = j + 3;
                t.content_begin = i + 9;
                t.content_end = j;
                out.push_back(t);
                i = t.end;
                continue;
            }
            else if (s.compare(i, 2, "<?") == 0)
            {
                const std::size_t j = s.find("?>", i + 2);
                if (j == std::string::npos) throw xml_error("xml_tokenize(): unterminated processing instruction");
                t.kind = xml_token::processing_instruction;
                t.end = j + 2;
            }
            else if (s.compare(i, 2, "<!") == 0)
            {
                // DOCTYPE and other declarations, with an optional internal subset.
                std::size_t j = i + 2;
                int depth = 0;
                char quote = 0;
                for (; j < n; ++j)
                {
                    const char c = s[j];
                    if (quote) { if (c == quote) quote = 0; continue; }
                    if (c == '"' || c == '\'') quote = c;
                    else if (c == '[') ++depth;
                    else if (c == ']') --depth;
                    else if (c == '>' && depth <= 0) break;
                }
                if (j >= n) throw xml_error("xml_tokenize(): unterminated declaration");
                t.kind = xml_token::doctype;
                t.end = j + 1;
            }
            else
            {
                std::size_t j = i + 1;
                const bool closing = j < n && s[j] == '/';
                if (closing) ++j;
                const std::size_t name_begin = j;
                while (j < n && !xml_space(s[j]) && s[j] != '>' && s[j] != '/') ++j;
                t.name = s.substr(name_begin, j - name_begin);
                if (t.name.empty()) throw xml_error("xml_tokenize(): tag without a name");
                bool self_closing = false;
                for (;;)
                {
                    while (j < n && xml_space(s[j])) ++j;
                    if (j >= n) throw xml_error("xml_tokenize(): unterminated tag " + t.name);
                    if (s[j] == '>') { ++j; break; }
                    if (s[j] == '/' && j + 1 < n && s[j + 1] == '>') { self_closing = true; j += 2; break; }
                    if (closing) throw xml_error("xml_tokenize(): attributes on an end tag");
                    const std::size_t an = j;
                    while (j < n && !xml_space(s[j]) && s[j] != '=' && s[j] != '>' && s[j] != '/') ++j;
                    xml_attribute a;
                    a.name = s.substr(an, j - an);
                    while (j < n && xml_space(s[j])) ++j;
                    if (j >= n || s[j] != '=')
                        throw xml_error("xml_tokenize(): attribute without a value in " + t.name);
                    ++j;
                    while (j < n && xml_space(s[j])) ++j;
                    if (j >= n || (s[j] != '"' && s[j] != '\''))
                        throw xml_error("xml_tokenize(): unquoted attribute in " + t.name);
                    const char q = s[j];
                    const std::size_t vb = j + 1;
                    const std::size_t ve = s.find(q, vb);
                    if (ve == std::string::npos) throw xml_error("xml_tokenize(): unterminated attribute in " + t.name);
                    a.value_begin = vb;
                    a.value_end = ve;
                    t.attributes.push_back(a);
                    j = ve + 1;
                }
                t.kind = closing ? xml_token::end_tag : (self_closing ? xml_token::empty_tag : xml_token::start_tag);
                t.end = j;
            }
            t.content_begin = t.begin;
            t.content_end = t.end;
            out.push_back(t);
            i = t.end;
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline bool html_name_char(char c)
        {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
                   c == '_' || c == ':' || c == '.' || static_cast<unsigned char>(c) >= 0x80;
        }

        inline std::string html_lower(std::string s)
        {
            for (char& c : s) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            return s;
        }

        inline bool html_void_element(const std::string& lower_name)
        {
            static const char* const names[] = { "area", "base", "br", "col", "embed", "hr", "img", "input",
                                                 "link", "meta", "param", "source", "track", "wbr" };
            for (const char* n : names)
                if (lower_name == n) return true;
            return false;
        }
    }

    inline std::vector<xml_token> html_tokenize(const std::string& s)
    {
        using impl::xml_space;
        std::vector<xml_token> out;
        const std::size_t n = s.size();
        std::size_t i = 0;
        auto text_until = [&](std::size_t from, std::size_t to) {
            if (to <= from) return;
            if (!out.empty() && out.back().kind == xml_token::text && out.back().end == from)
            {
                out.back().end = to;
                out.back().content_end = to;
                return;
            }
            xml_token t;
            t.kind = xml_token::text;
            t.begin = t.content_begin = from;
            t.end = t.content_end = to;
            out.push_back(t);
        };
        while (i < n)
        {
            if (s[i] != '<')
            {
                const std::size_t j = s.find('<', i);
                text_until(i, j == std::string::npos ? n : j);
                i = j == std::string::npos ? n : j;
                continue;
            }
            xml_token t;
            t.begin = i;
            if (s.compare(i, 4, "<!--") == 0)
            {
                const std::size_t j = s.find("-->", i + 4);
                t.kind = xml_token::comment;
                t.end = j == std::string::npos ? n : j + 3;
            }
            else if (s.compare(i, 9, "<![CDATA[") == 0)
            {
                const std::size_t j = s.find("]]>", i + 9);
                t.kind = xml_token::cdata;
                t.end = j == std::string::npos ? n : j + 3;
                t.content_begin = i + 9;
                t.content_end = j == std::string::npos ? n : j;
                out.push_back(t);
                i = t.end;
                continue;
            }
            else if (i + 1 < n && (s[i + 1] == '!' || s[i + 1] == '?'))
            {
                const std::size_t j = s.find('>', i + 2);
                t.kind = s[i + 1] == '?' ? xml_token::processing_instruction : xml_token::doctype;
                t.end = j == std::string::npos ? n : j + 1;
            }
            else
            {
                std::size_t j = i + 1;
                const bool closing = j < n && s[j] == '/';
                if (closing) ++j;
                const std::size_t name_begin = j;
                if (j >= n || !((s[j] >= 'a' && s[j] <= 'z') || (s[j] >= 'A' && s[j] <= 'Z')))
                {
                    // A '<' that starts no tag, as in "a < b", is text.
                    text_until(i, i + 1);
                    ++i;
                    continue;
                }
                while (j < n && impl::html_name_char(s[j])) ++j;
                t.name = s.substr(name_begin, j - name_begin);
                bool self_closing = false;
                for (;;)
                {
                    while (j < n && xml_space(s[j])) ++j;
                    if (j >= n) break;
                    if (s[j] == '>') { ++j; break; }
                    if (s[j] == '/' && j + 1 < n && s[j + 1] == '>') { self_closing = true; j += 2; break; }
                    if (s[j] == '/') { ++j; continue; }
                    const std::size_t an = j;
                    while (j < n && !xml_space(s[j]) && s[j] != '=' && s[j] != '>' &&
                           !(s[j] == '/' && j + 1 < n && s[j + 1] == '>'))
                        ++j;
                    xml_attribute a;
                    a.name = s.substr(an, j - an);
                    std::size_t k = j;
                    while (k < n && xml_space(s[k])) ++k;
                    if (k < n && s[k] == '=')
                    {
                        ++k;
                        while (k < n && xml_space(s[k])) ++k;
                        if (k < n && (s[k] == '"' || s[k] == '\''))
                        {
                            const std::size_t ve = s.find(s[k], k + 1);
                            a.value_begin = k + 1;
                            a.value_end = ve == std::string::npos ? n : ve;
                            j = ve == std::string::npos ? n : ve + 1;
                        }
                        else
                        {
                            const std::size_t vb = k;
                            while (k < n && !xml_space(s[k]) && s[k] != '>') ++k;
                            a.value_begin = vb;
                            a.value_end = k;
                            j = k;
                        }
                    }
                    else
                    {
                        // An attribute without value, such as "checked".
                        a.value_begin = a.value_end = j;
                    }
                    if (!closing && !a.name.empty()) t.attributes.push_back(a);
                }
                const std::string lower = impl::html_lower(t.name);
                t.kind = closing ? xml_token::end_tag
                                 : ((self_closing || impl::html_void_element(lower)) ? xml_token::empty_tag
                                                                                      : xml_token::start_tag);
                t.end = j;
                t.content_begin = t.begin;
                t.content_end = t.end;
                out.push_back(t);
                i = t.end;
                // Script and style hold raw text up to their end tag.
                if (t.kind == xml_token::start_tag && (lower == "script" || lower == "style"))
                {
                    std::size_t e = i;
                    for (;;)
                    {
                        e = s.find("</", e);
                        if (e == std::string::npos || impl::html_lower(s.substr(e + 2, lower.size())) == lower) break;
                        e += 2;
                    }
                    const std::size_t stop = e == std::string::npos ? n : e;
                    text_until(i, stop);
                    i = stop;
                }
                continue;
            }
            t.content_begin = t.begin;
            t.content_end = t.end;
            out.push_back(t);
            i = t.end;
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        /* The code point of a numeric character reference body ("#65", "#x41", leading
           zeros allowed), or 0 when it is malformed, out of range or a surrogate. */
        inline unsigned long xml_reference(const std::string& ent)
        {
            if (ent.size() < 2 || ent[0] != '#') return 0;
            const bool hex = ent[1] == 'x' || ent[1] == 'X';
            std::size_t k = hex ? 2 : 1;
            if (k >= ent.size()) return 0;
            while (k + 1 < ent.size() && ent[k] == '0') ++k;
            if (ent.size() - k > 8) return 0;
            unsigned long cp = 0;
            for (; k < ent.size(); ++k)
            {
                const char c = ent[k];
                int v;
                if (c >= '0' && c <= '9') v = c - '0';
                else if (hex && c >= 'a' && c <= 'f') v = c - 'a' + 10;
                else if (hex && c >= 'A' && c <= 'F') v = c - 'A' + 10;
                else return 0;
                cp = cp * (hex ? 16 : 10) + static_cast<unsigned long>(v);
            }
            if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
            return cp;
        }
    }

    inline std::string xml_unescape(const std::string& raw)
    {
        if (raw.find('&') == std::string::npos)
            return raw;
        std::string out;
        out.reserve(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i)
        {
            if (raw[i] != '&')
            {
                out.push_back(raw[i]);
                continue;
            }
            // References are short; the search for ';' is bounded so that a run of '&' stays linear.
            std::size_t semi = std::string::npos;
            for (std::size_t k = i + 1; k < raw.size() && k <= i + 64; ++k)
            {
                if (raw[k] == ';') { semi = k; break; }
                if (raw[k] == '&') break;
            }
            if (semi == std::string::npos)
            {
                out.push_back('&');
                continue;
            }
            const std::string ent = raw.substr(i + 1, semi - i - 1);
            std::string rep;
            if (ent == "amp") rep = "&";
            else if (ent == "lt") rep = "<";
            else if (ent == "gt") rep = ">";
            else if (ent == "quot") rep = "\"";
            else if (ent == "apos") rep = "'";
            else if (!ent.empty() && ent[0] == '#')
            {
                const unsigned long cp = impl::xml_reference(ent);
                if (cp > 0)
                {
                    if (cp < 0x80) rep.push_back(static_cast<char>(cp));
                    else if (cp < 0x800)
                    {
                        rep.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                        rep.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    else if (cp < 0x10000)
                    {
                        rep.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                        rep.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        rep.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    else
                    {
                        rep.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                        rep.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                        rep.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        rep.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                }
            }
            if (rep.empty())
            {
                out.push_back('&');
                continue;
            }
            out += rep;
            i = semi;
        }
        return out;
    }

    inline std::string xml_escape_text(const std::string& text)
    {
        std::string out;
        out.reserve(text.size() + 8);
        for (char c : text)
        {
            switch (c)
            {
                case '&': out += "&amp;"; break;
                case '<': out += "&lt;"; break;
                case '>': out += "&gt;"; break;
                default: out.push_back(c);
            }
        }
        return out;
    }

    inline std::string xml_escape_attribute(const std::string& text)
    {
        std::string out;
        out.reserve(text.size() + 8);
        for (char c : text)
        {
            switch (c)
            {
                case '&': out += "&amp;"; break;
                case '<': out += "&lt;"; break;
                case '>': out += "&gt;"; break;
                case '"': out += "&quot;"; break;
                case '\'': out += "&apos;"; break;
                case '\t': out += "&#9;"; break;
                case '\n': out += "&#10;"; break;
                case '\r': out += "&#13;"; break;
                default: out.push_back(c);
            }
        }
        return out;
    }

    inline std::string xml_local_name(const std::string& qname)
    {
        const std::size_t c = qname.find(':');
        return c == std::string::npos ? qname : qname.substr(c + 1);
    }

    inline std::string xml_prefix(const std::string& qname)
    {
        const std::size_t c = qname.find(':');
        return c == std::string::npos ? std::string() : qname.substr(0, c);
    }

    inline std::string xml_to_utf8(const std::string& xml)
    {
        const std::size_t n = xml.size();
        auto byte = [&](std::size_t i) { return static_cast<unsigned char>(xml[i]); };
        bool little = false, big = false;
        std::size_t i = 0;
        if (n >= 2 && byte(0) == 0xFF && byte(1) == 0xFE) { little = true; i = 2; }
        else if (n >= 2 && byte(0) == 0xFE && byte(1) == 0xFF) { big = true; i = 2; }
        else if (n >= 2 && xml[0] == '<' && xml[1] == '\0') little = true;
        else if (n >= 2 && xml[0] == '\0' && xml[1] == '<') big = true;
        if (!little && !big)
            return xml;
        auto unit = [&](std::size_t k) {
            return little ? static_cast<std::uint32_t>(byte(k) | (byte(k + 1) << 8))
                          : static_cast<std::uint32_t>((byte(k) << 8) | byte(k + 1));
        };
        std::string out;
        out.reserve(n / 2 + 16);
        for (; i + 1 < n; i += 2)
        {
            std::uint32_t cp = unit(i);
            if (cp >= 0xD800 && cp <= 0xDBFF && i + 3 < n && unit(i + 2) >= 0xDC00 && unit(i + 2) <= 0xDFFF)
            {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (unit(i + 2) - 0xDC00);
                i += 2;
            }
            else if (cp >= 0xD800 && cp <= 0xDFFF)
            {
                cp = 0xFFFD;
            }
            if (cp < 0x80) out.push_back(static_cast<char>(cp));
            else if (cp < 0x800)
            {
                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            else if (cp < 0x10000)
            {
                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            else
            {
                out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
        }
        // The declaration now names UTF-8.
        if (out.compare(0, 5, "<?xml") == 0)
        {
            const std::size_t close = out.find("?>");
            const std::size_t enc = out.find("encoding", 5);
            if (close != std::string::npos && enc != std::string::npos && enc < close)
            {
                const std::size_t q = out.find_first_of("\"'", enc);
                const std::size_t e = q == std::string::npos ? q : out.find(out[q], q + 1);
                if (q != std::string::npos && e != std::string::npos && e < close)
                    out.replace(q + 1, e - q - 1, "UTF-8");
            }
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

    class xml_namespace_tracker
    {
    public:
        void enter(const std::string& source, const xml_token& tag)
        {
            std::vector<std::string> frame;
            for (const xml_attribute& a : tag.attributes)
            {
                std::string prefix;
                if (a.name == "xmlns") prefix.clear();
                else if (a.name.compare(0, 6, "xmlns:") == 0) prefix = a.name.substr(6);
                else continue;
                bindings_[prefix].push_back(xml_unescape(source.substr(a.value_begin, a.value_end - a.value_begin)));
                frame.push_back(prefix);
            }
            frames_.push_back(frame);
        }

        void leave()
        {
            if (frames_.empty())
                return;
            for (const std::string& prefix : frames_.back())
            {
                const auto it = bindings_.find(prefix);
                if (it == bindings_.end()) continue;
                it->second.pop_back();
                if (it->second.empty()) bindings_.erase(it);
            }
            frames_.pop_back();
        }

        std::string resolve(const std::string& qname, bool is_attribute = false) const
        {
            const std::size_t c = qname.find(':');
            if (c == std::string::npos && is_attribute)
                return std::string();
            const std::string prefix = c == std::string::npos ? std::string() : qname.substr(0, c);
            if (prefix == "xml")
                return "http://www.w3.org/XML/1998/namespace";
            const auto it = bindings_.find(prefix);
            return it == bindings_.end() ? std::string() : it->second.back();
        }

        std::size_t depth() const { return frames_.size(); }

    private:
        std::vector<std::vector<std::string>> frames_;              // prefixes bound by each open element
        std::map<std::string, std::vector<std::string>> bindings_;  // URIs in scope, innermost last
    };

// ----------------------------------------------------------------------------------------

    class xml_edit_list
    {
    public:
        void replace(std::size_t begin, std::size_t end, const std::string& text)
        {
            edits_.push_back(edit{ begin, end, text, edits_.size() });
        }

        void insert(std::size_t at, const std::string& text)
        {
            edits_.push_back(edit{ at, at, text, edits_.size() });
        }

        bool empty() const { return edits_.empty(); }

        bool covers(std::size_t begin, std::size_t end) const
        {
            for (const edit& e : edits_)
                if (e.end > e.begin && e.begin <= begin && end <= e.end)
                    return true;
            return false;
        }

        std::string apply(const std::string& source) const
        {
            // Insertions first at a position, then the widest replacement, which drops the
            // edits nested inside it.
            std::vector<edit> sorted(edits_);
            std::sort(sorted.begin(), sorted.end(), [](const edit& a, const edit& b) {
                if (a.begin != b.begin) return a.begin < b.begin;
                if ((a.end == a.begin) != (b.end == b.begin)) return a.end == a.begin;
                if (a.end != b.end) return a.end > b.end;
                return a.order < b.order;
            });
            std::string out;
            out.reserve(source.size());
            std::size_t pos = 0;
            for (const edit& e : sorted)
            {
                if (e.begin < pos || e.end < e.begin || e.end > source.size())
                    continue;
                out.append(source, pos, e.begin - pos);
                out += e.text;
                pos = e.end;
            }
            out.append(source, pos, std::string::npos);
            return out;
        }

    private:
        struct edit
        {
            std::size_t begin;
            std::size_t end;
            std::string text;
            std::size_t order;
        };
        std::vector<edit> edits_;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_XML_SCANNER_H_
