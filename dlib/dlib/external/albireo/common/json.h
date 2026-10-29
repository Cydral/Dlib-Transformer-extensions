// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_COMMON_JSON_H_
#define ALBIREO_COMMON_JSON_H_

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace albireo
{

    class json
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A JSON value: null, boolean, number, string, array or object.  It is the
                document type exchanged between the host, the inference engines, the web
                interface and the GitHub feed.

                Objects keep their members in insertion order, so a document written by
                dump() lists its members in the order the program set them.  Numbers
                written without fraction or exponent and that fit in 64 bits are kept as
                integers, so identifiers and sizes survive a round trip exactly.

                Reading a member or an element that does not exist from a const value
                yields a shared null value rather than throwing, which keeps the code
                that walks requests from untrusted clients free of existence checks.

            PARSING
                parse() accepts the strict grammar of RFC 8259 and refuses documents
                nested deeper than a given limit, so that a hostile body cannot exhaust
                the stack.

            SERIALIZATION
                dump() produces compact UTF-8 text.  Invalid UTF-8 sequences found in
                strings are replaced by U+FFFD, so the output is always valid JSON.
        !*/

    public:
        enum class kind { null_value, boolean_value, number_value, string_value, array_value, object_value };
        typedef std::vector<json> array_type;
        typedef std::vector<std::pair<std::string, json>> object_type;

        json() {}
        json(std::nullptr_t) {}
        json(bool v) : kind_(kind::boolean_value), bool_(v) {}
        json(int v) { set_integer(v); }
        json(long v) { set_integer(v); }
        json(long long v) { set_integer(v); }
        json(unsigned int v) { set_integer(static_cast<long long>(v)); }
        json(unsigned long v) { set_unsigned(static_cast<unsigned long long>(v)); }
        json(unsigned long long v) { set_unsigned(v); }
        json(float v) : kind_(kind::number_value), number_(v) {}
        json(double v) : kind_(kind::number_value), number_(v) {}
        json(const char* v) : kind_(kind::string_value), string_(v ? v : "") {}
        json(const std::string& v) : kind_(kind::string_value), string_(v) {}
        json(std::string&& v) : kind_(kind::string_value), string_(std::move(v)) {}

        static json array() { json j; j.kind_ = kind::array_value; return j; }
        static json object() { json j; j.kind_ = kind::object_value; return j; }

        kind type() const { return kind_; }
        bool is_null() const { return kind_ == kind::null_value; }
        bool is_bool() const { return kind_ == kind::boolean_value; }
        bool is_number() const { return kind_ == kind::number_value; }
        bool is_integer() const { return kind_ == kind::number_value && is_integer_; }
        bool is_string() const { return kind_ == kind::string_value; }
        bool is_array() const { return kind_ == kind::array_value; }
        bool is_object() const { return kind_ == kind::object_value; }

        bool as_bool(bool def = false) const { return kind_ == kind::boolean_value ? bool_ : def; }
        double as_double(double def = 0.0) const
        {
            if (kind_ != kind::number_value) return def;
            return is_integer_ ? static_cast<double>(integer_) : number_;
        }
        long long as_int(long long def = 0) const
        {
            if (kind_ != kind::number_value) return def;
            if (is_integer_) return integer_;
            if (!(number_ >= -9.2e18 && number_ <= 9.2e18)) return def;
            return static_cast<long long>(number_);
        }
        const std::string& as_string() const
        {
            return kind_ == kind::string_value ? string_ : empty_string();
        }
        std::string str(const std::string& def = std::string()) const
        {
            return kind_ == kind::string_value ? string_ : def;
        }

        // Objects

        bool contains(const std::string& key) const
        {
            if (kind_ != kind::object_value) return false;
            for (const auto& kv : object_) if (kv.first == key) return true;
            return false;
        }

        const json& operator[](const std::string& key) const
        {
            if (kind_ == kind::object_value)
                for (const auto& kv : object_) if (kv.first == key) return kv.second;
            return null_json();
        }
        const json& operator[](const char* key) const { return (*this)[std::string(key)]; }

        json& operator[](const std::string& key)
        {
            if (kind_ != kind::object_value) { reset(); kind_ = kind::object_value; }
            for (auto& kv : object_) if (kv.first == key) return kv.second;
            object_.emplace_back(key, json());
            return object_.back().second;
        }
        json& operator[](const char* key) { return (*this)[std::string(key)]; }

        bool erase(const std::string& key)
        {
            if (kind_ != kind::object_value) return false;
            for (auto it = object_.begin(); it != object_.end(); ++it)
                if (it->first == key) { object_.erase(it); return true; }
            return false;
        }

        const object_type& items() const { return kind_ == kind::object_value ? object_ : empty_object(); }
        object_type& items()
        {
            if (kind_ != kind::object_value)
            {
                reset();
                kind_ = kind::object_value;
            }
            return object_;
        }

        // Arrays

        size_t size() const
        {
            if (kind_ == kind::array_value) return array_.size();
            if (kind_ == kind::object_value) return object_.size();
            return 0;
        }
        bool empty() const { return size() == 0; }

        const json& operator[](size_t i) const
        {
            if (kind_ == kind::array_value && i < array_.size()) return array_[i];
            return null_json();
        }
        const json& operator[](int i) const { return i < 0 ? null_json() : (*this)[static_cast<size_t>(i)]; }
        json& at(size_t i) { return array_.at(i); }

        json& push_back(json v)
        {
            if (kind_ != kind::array_value) { reset(); kind_ = kind::array_value; }
            array_.push_back(std::move(v));
            return array_.back();
        }

        const array_type& elements() const { return kind_ == kind::array_value ? array_ : empty_array(); }
        array_type& elements()
        {
            if (kind_ != kind::array_value)
            {
                reset();
                kind_ = kind::array_value;
            }
            return array_;
        }

        // Serialization

        std::string dump() const
        {
            std::string out;
            dump_to(out);
            return out;
        }

        void dump_to(std::string& out) const
        {
            switch (kind_)
            {
            case kind::null_value: out += "null"; break;
            case kind::boolean_value: out += bool_ ? "true" : "false"; break;
            case kind::number_value: append_number(out); break;
            case kind::string_value: append_quoted(out, string_); break;
            case kind::array_value:
                out += '[';
                for (size_t i = 0; i < array_.size(); ++i)
                {
                    if (i) out += ',';
                    array_[i].dump_to(out);
                }
                out += ']';
                break;
            case kind::object_value:
                out += '{';
                for (size_t i = 0; i < object_.size(); ++i)
                {
                    if (i) out += ',';
                    append_quoted(out, object_[i].first);
                    out += ':';
                    object_[i].second.dump_to(out);
                }
                out += '}';
                break;
            }
        }

        /*!
            ensures
                - Returns the document in the layout of the json module of Python, which is
                  the layout the chat templates of the model hub give to the tool
                  definitions they insert in prompts: ", " and ": " between the members of
                  a single line when indent < 0, otherwise one member per line indented by
                  indent spaces a level, "," at the ends of lines.  A model generates best
                  when its prompt is laid out exactly as during its training.
        !*/
        std::string dump_formatted(int indent = -1) const
        {
            std::string out;
            dump_formatted_to(out, indent, 0);
            return out;
        }

        // A JSON string literal, quotes included, with invalid UTF-8 replaced by U+FFFD.
        static std::string quote(const std::string& s)
        {
            std::string out;
            append_quoted(out, s);
            return out;
        }

        static void append_quoted(std::string& out, const std::string& s)
        {
            static const char* hex = "0123456789abcdef";
            out.reserve(out.size() + s.size() + 2);
            out += '"';
            const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
            const size_t n = s.size();
            size_t i = 0;
            while (i < n)
            {
                const unsigned char c = p[i];
                if (c < 0x80)
                {
                    switch (c)
                    {
                    case '"': out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\b': out += "\\b"; break;
                    case '\f': out += "\\f"; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    case '\t': out += "\\t"; break;
                    default:
                        if (c < 0x20 || c == 0x7f)
                        {
                            out += "\\u00";
                            out += hex[c >> 4];
                            out += hex[c & 0xF];
                        }
                        else out += static_cast<char>(c);
                    }
                    ++i;
                    continue;
                }
                const size_t len = utf8_sequence_length(p + i, n - i);
                if (len == 0)
                {
                    out += "\xEF\xBF\xBD";
                    ++i;
                }
                else
                {
                    out.append(s, i, len);
                    i += len;
                }
            }
            out += '"';
        }

        // Length of the valid UTF-8 sequence starting at p (at most n bytes available), or 0
        // when the bytes do not form one.  Overlong forms and surrogates are refused.
        static size_t utf8_sequence_length(const unsigned char* p, size_t n)
        {
            const unsigned char c = p[0];
            if (c < 0x80) return 1;
            size_t len;
            unsigned long cp;
            if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
            else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
            else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
            else return 0;
            if (len > n) return 0;
            for (size_t k = 1; k < len; ++k)
            {
                if ((p[k] & 0xC0) != 0x80) return 0;
                cp = (cp << 6) | (p[k] & 0x3F);
            }
            if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000))
                return 0;
            if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
            return len;
        }

        // Parsing

        static bool parse(const std::string& text, json& out, std::string* error = nullptr,
            size_t max_depth = 128)
        {
            parser p(text, max_depth);
            json v;
            if (!p.value(v, 0) || (p.skip_ws(), p.pos != text.size()))
            {
                if (error)
                {
                    *error = p.error.empty() ? "unexpected characters after the document" : p.error;
                    *error += " at offset " + std::to_string(p.pos);
                }
                return false;
            }
            out = std::move(v);
            return true;
        }

        static json parse_or_null(const std::string& text)
        {
            json v;
            if (!parse(text, v)) return json();
            return v;
        }

        friend bool operator==(const json& a, const json& b)
        {
            if (a.kind_ != b.kind_) return false;
            switch (a.kind_)
            {
            case kind::null_value: return true;
            case kind::boolean_value: return a.bool_ == b.bool_;
            case kind::number_value: return a.as_double() == b.as_double();
            case kind::string_value: return a.string_ == b.string_;
            case kind::array_value: return a.array_ == b.array_;
            case kind::object_value:
                if (a.object_.size() != b.object_.size()) return false;
                for (const auto& kv : a.object_)
                    if (!b.contains(kv.first) || !(b[kv.first] == kv.second)) return false;
                return true;
            }
            return false;
        }
        friend bool operator!=(const json& a, const json& b) { return !(a == b); }

    private:

        void dump_formatted_to(std::string& out, int indent, int level) const
        {
            auto newline = [&](int l) {
                out += '\n';
                out.append(static_cast<size_t>(indent) * static_cast<size_t>(l), ' ');
            };
            if (kind_ == kind::array_value && !array_.empty())
            {
                out += '[';
                for (size_t i = 0; i < array_.size(); ++i)
                {
                    if (i) out += indent < 0 ? ", " : ",";
                    if (indent >= 0) newline(level + 1);
                    array_[i].dump_formatted_to(out, indent, level + 1);
                }
                if (indent >= 0) newline(level);
                out += ']';
            }
            else if (kind_ == kind::object_value && !object_.empty())
            {
                out += '{';
                for (size_t i = 0; i < object_.size(); ++i)
                {
                    if (i) out += indent < 0 ? ", " : ",";
                    if (indent >= 0) newline(level + 1);
                    append_quoted(out, object_[i].first);
                    out += ": ";
                    object_[i].second.dump_formatted_to(out, indent, level + 1);
                }
                if (indent >= 0) newline(level);
                out += '}';
            }
            else
            {
                dump_to(out);
            }
        }

        struct parser
        {
            parser(const std::string& t, size_t d) : s(t), max_depth(d) {}
            const std::string& s;
            size_t pos = 0;
            size_t max_depth;
            std::string error;

            void skip_ws()
            {
                while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
                    ++pos;
            }

            bool fail(const char* what) { if (error.empty()) error = what; return false; }

            bool literal(const char* word)
            {
                const size_t n = std::strlen(word);
                if (s.compare(pos, n, word) != 0) return fail("invalid literal");
                pos += n;
                return true;
            }

            bool value(json& v, size_t depth)
            {
                if (depth > max_depth) return fail("document nested too deeply");
                skip_ws();
                if (pos >= s.size()) return fail("unexpected end of document");
                const char c = s[pos];
                if (c == '{') return object(v, depth);
                if (c == '[') return array(v, depth);
                if (c == '"') { v.reset(); v.kind_ = kind::string_value; return string(v.string_); }
                if (c == 't') { if (!literal("true")) return false; v = json(true); return true; }
                if (c == 'f') { if (!literal("false")) return false; v = json(false); return true; }
                if (c == 'n') { if (!literal("null")) return false; v = json(); return true; }
                if (c == '-' || (c >= '0' && c <= '9')) return number(v);
                return fail("unexpected character");
            }

            bool object(json& v, size_t depth)
            {
                ++pos;
                v.reset();
                v.kind_ = kind::object_value;
                skip_ws();
                if (pos < s.size() && s[pos] == '}') { ++pos; return true; }
                for (;;)
                {
                    skip_ws();
                    if (pos >= s.size() || s[pos] != '"') return fail("member name expected");
                    std::string key;
                    if (!string(key)) return false;
                    skip_ws();
                    if (pos >= s.size() || s[pos] != ':') return fail("':' expected");
                    ++pos;
                    json member;
                    if (!value(member, depth + 1)) return false;
                    // A repeated member replaces the earlier one, as most parsers do.
                    bool replaced = false;
                    for (auto& kv : v.object_)
                        if (kv.first == key) { kv.second = std::move(member); replaced = true; break; }
                    if (!replaced) v.object_.emplace_back(std::move(key), std::move(member));
                    skip_ws();
                    if (pos >= s.size()) return fail("unexpected end of document");
                    if (s[pos] == ',') { ++pos; continue; }
                    if (s[pos] == '}') { ++pos; return true; }
                    return fail("',' or '}' expected");
                }
            }

            bool array(json& v, size_t depth)
            {
                ++pos;
                v.reset();
                v.kind_ = kind::array_value;
                skip_ws();
                if (pos < s.size() && s[pos] == ']') { ++pos; return true; }
                for (;;)
                {
                    json element;
                    if (!value(element, depth + 1)) return false;
                    v.array_.push_back(std::move(element));
                    skip_ws();
                    if (pos >= s.size()) return fail("unexpected end of document");
                    if (s[pos] == ',') { ++pos; continue; }
                    if (s[pos] == ']') { ++pos; return true; }
                    return fail("',' or ']' expected");
                }
            }

            static int hex_digit(char c)
            {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            }

            bool hex4(unsigned long& cp)
            {
                if (pos + 4 > s.size()) return fail("truncated \\u escape");
                cp = 0;
                for (int k = 0; k < 4; ++k)
                {
                    const int d = hex_digit(s[pos + k]);
                    if (d < 0) return fail("invalid \\u escape");
                    cp = (cp << 4) | static_cast<unsigned long>(d);
                }
                pos += 4;
                return true;
            }

            static void append_utf8(std::string& out, unsigned long cp)
            {
                if (cp < 0x80) out += static_cast<char>(cp);
                else if (cp < 0x800)
                {
                    out += static_cast<char>(0xC0 | (cp >> 6));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                }
                else if (cp < 0x10000)
                {
                    out += static_cast<char>(0xE0 | (cp >> 12));
                    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                }
                else
                {
                    out += static_cast<char>(0xF0 | (cp >> 18));
                    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                }
            }

            bool string(std::string& out)
            {
                ++pos;
                out.clear();
                for (;;)
                {
                    if (pos >= s.size()) return fail("unterminated string");
                    const unsigned char c = static_cast<unsigned char>(s[pos]);
                    if (c == '"') { ++pos; return true; }
                    if (c < 0x20) return fail("control character in string");
                    if (c != '\\')
                    {
                        // Copy a run of plain characters at once.
                        size_t end = pos + 1;
                        while (end < s.size())
                        {
                            const unsigned char d = static_cast<unsigned char>(s[end]);
                            if (d == '"' || d == '\\' || d < 0x20) break;
                            ++end;
                        }
                        out.append(s, pos, end - pos);
                        pos = end;
                        continue;
                    }
                    ++pos;
                    if (pos >= s.size()) return fail("unterminated escape");
                    const char e = s[pos++];
                    switch (e)
                    {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u':
                    {
                        unsigned long cp;
                        if (!hex4(cp)) return false;
                        if (cp >= 0xD800 && cp <= 0xDBFF)
                        {
                            unsigned long lo = 0;
                            if (pos + 1 < s.size() && s[pos] == '\\' && s[pos + 1] == 'u')
                            {
                                pos += 2;
                                if (!hex4(lo)) return false;
                            }
                            if (lo >= 0xDC00 && lo <= 0xDFFF)
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            else
                                cp = 0xFFFD;   // lone high surrogate
                        }
                        else if (cp >= 0xDC00 && cp <= 0xDFFF)
                            cp = 0xFFFD;       // lone low surrogate
                        append_utf8(out, cp);
                        break;
                    }
                    default:
                        return fail("invalid escape");
                    }
                }
            }

            bool number(json& v)
            {
                const size_t start = pos;
                bool integral = true;
                if (s[pos] == '-') ++pos;
                if (pos >= s.size()) return fail("invalid number");
                if (s[pos] == '0') ++pos;
                else if (s[pos] >= '1' && s[pos] <= '9')
                    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
                else return fail("invalid number");
                if (pos < s.size() && s[pos] == '.')
                {
                    integral = false;
                    ++pos;
                    if (pos >= s.size() || !(s[pos] >= '0' && s[pos] <= '9')) return fail("invalid number");
                    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
                }
                if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E'))
                {
                    integral = false;
                    ++pos;
                    if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
                    if (pos >= s.size() || !(s[pos] >= '0' && s[pos] <= '9')) return fail("invalid number");
                    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
                }
                const std::string text = s.substr(start, pos - start);
                v.reset();
                v.kind_ = kind::number_value;
                if (integral && text.size() <= 19)
                {
                    errno = 0;
                    char* end = nullptr;
                    const long long iv = std::strtoll(text.c_str(), &end, 10);
                    if (errno == 0 && end && *end == '\0')
                    {
                        v.is_integer_ = true;
                        v.integer_ = iv;
                        v.number_ = static_cast<double>(iv);
                        return true;
                    }
                }
                v.number_ = std::strtod(text.c_str(), nullptr);
                return true;
            }
        };

        void reset()
        {
            kind_ = kind::null_value;
            bool_ = false;
            number_ = 0;
            is_integer_ = false;
            integer_ = 0;
            string_.clear();
            array_.clear();
            object_.clear();
        }

        void set_integer(long long v)
        {
            kind_ = kind::number_value;
            is_integer_ = true;
            integer_ = v;
            number_ = static_cast<double>(v);
        }

        void set_unsigned(unsigned long long v)
        {
            if (v <= static_cast<unsigned long long>(std::numeric_limits<long long>::max()))
                set_integer(static_cast<long long>(v));
            else
            {
                kind_ = kind::number_value;
                number_ = static_cast<double>(v);
            }
        }

        // The conversions assume the "C" numeric locale, which a program keeps as long as it
        // never calls setlocale() for LC_NUMERIC; the Albireo programs never do.
        void append_number(std::string& out) const
        {
            if (is_integer_)
            {
                out += std::to_string(integer_);
                return;
            }
            if (!std::isfinite(number_))
            {
                out += "null";   // JSON has no representation for infinities and NaN
                return;
            }
            // Shortest of the precisions that read back to the same double.
            char buf[40];
            for (int precision = 15; precision <= 17; ++precision)
            {
                std::snprintf(buf, sizeof(buf), "%.*g", precision, number_);
                if (std::strtod(buf, nullptr) == number_) break;
            }
            out += buf;
        }

        static const json& null_json() { static const json v; return v; }
        static const std::string& empty_string() { static const std::string v; return v; }
        static const array_type& empty_array() { static const array_type v; return v; }
        static const object_type& empty_object() { static const object_type v; return v; }

        kind kind_ = kind::null_value;
        bool bool_ = false;
        double number_ = 0;
        bool is_integer_ = false;
        long long integer_ = 0;
        std::string string_;
        array_type array_;
        object_type object_;
    };

}

#endif // ALBIREO_COMMON_JSON_H_
