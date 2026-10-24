// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PDF_TEXT_H_
#define DLIB_PDF_TEXT_H_

#include "pdf_text_abstract.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "../error.h"
#include "../numeric_constants.h"
#include "../zip/deflate.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class pdf_error : public error
    {
    public:
        pdf_error(const std::string& message) : error(message) {}
    };

    struct pdf_text_span
    {
        std::string text;
        double x = 0, y = 0;
        double width = 0;
        double font_size = 0;
        double angle = 0;
    };

    struct pdf_page_text
    {
        double x0 = 0, y0 = 0, x1 = 612, y1 = 792;
        int rotate = 0;
        std::vector<pdf_text_span> spans;
        std::size_t images = 0;
        std::size_t annotations = 0;
    };

    struct pdf_text_document
    {
        std::vector<pdf_page_text> pages;
        std::size_t undecodable_glyphs = 0;
        bool produced_by_dlib = false;
        std::vector<std::string> notes;
        std::vector<std::string> features;
    };

    struct pdf_text_segment
    {
        std::string text;
        double x = 0, y = 0;
        double width = 0;
        double font_size = 0;
        double angle = 0;
        std::size_t line = 0;
    };

// ----------------------------------------------------------------------------------------

    namespace pdf_impl
    {
        // Bounds on decoded data: per stream, and for the whole document.
        const std::size_t max_stream_size = std::size_t(64) << 20;
        const std::size_t max_decoded_size = std::size_t(1024) << 20;
        // Bounds on parsing work: elements of one array or dictionary, objects parsed
        // for the whole document, content operators run, and bytes of one shown string.
        const std::size_t max_container_size = std::size_t(1) << 20;
        const std::size_t max_objects = std::size_t(1) << 23;
        const std::size_t max_operations = std::size_t(1) << 27;
        const std::size_t max_shown_string = std::size_t(1) << 20;

        /* Conversions of numbers read from the file. Out of range values are clamped,
           since a direct cast of such a double is undefined. */
        inline int to_int(double v)
        {
            if (!(v == v)) return 0;
            if (v > 1e9) return 1000000000;
            if (v < -1e9) return -1000000000;
            return static_cast<int>(v);
        }

        inline std::size_t to_size(double v)
        {
            const std::size_t top = std::numeric_limits<std::size_t>::max() / 2;
            if (!(v > 0)) return 0;
            if (v >= static_cast<double>(top)) return top;
            return static_cast<std::size_t>(v);
        }

        inline std::uint32_t to_u32(double v)
        {
            if (!(v > 0)) return 0;
            if (v >= 4294967295.0) return 0xFFFFFFFFu;
            return static_cast<std::uint32_t>(v);
        }

        /* A PDF number: an optional sign, digits and at most one period, without exponent.
           It is read without the C library, whose result depends on the locale. */
        inline double parse_number(const char* b, const char* e)
        {
            bool negative = false;
            while (b < e && (*b == '+' || *b == '-'))
            {
                if (*b == '-') negative = !negative;
                ++b;
            }
            static const double powers[] = { 1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
                                             1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                                             1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22 };
            std::uint64_t mantissa = 0;
            int digits = 0, scale = 0, extra = 0;
            bool fraction = false;
            for (; b < e; ++b)
            {
                if (*b == '.')
                {
                    if (fraction) break;
                    fraction = true;
                    continue;
                }
                if (*b < '0' || *b > '9') break;
                if (digits >= 18 || (fraction && scale >= 22))
                {
                    // Digits beyond the precision of a double: only the magnitude counts.
                    if (!fraction) ++extra;
                    continue;
                }
                mantissa = mantissa * 10 + static_cast<std::uint64_t>(*b - '0');
                if (mantissa != 0) ++digits;
                if (fraction) ++scale;
            }
            double v = static_cast<double>(mantissa);
            if (scale > 0) v /= powers[scale];
            for (; extra > 0; extra -= 22) v *= powers[extra > 22 ? 22 : extra];
            if (!(v < 1e300)) v = 0;
            return negative ? -v : v;
        }

        inline double parse_number(const std::string& s)
        {
            return parse_number(s.data(), s.data() + s.size());
        }

        struct object
        {
            enum kind_t { null_k, bool_k, num_k, str_k, name_k, array_k, dict_k, ref_k, stream_k, op_k };
            kind_t kind = null_k;
            bool b = false;
            double num = 0;
            std::string str;
            std::vector<object> arr;
            std::map<std::string, object> dict;
            int ref = 0;
            int gen = 0;
            std::size_t data_begin = 0, data_end = 0;
            const std::string* buffer = nullptr;

            bool is(kind_t k) const { return kind == k; }
            const object* get(const std::string& key) const
            {
                const auto it = dict.find(key);
                return it == dict.end() ? nullptr : &it->second;
            }
        };

        inline bool is_ws(char c) { return c == 0 || c == 9 || c == 10 || c == 12 || c == 13 || c == 32; }
        inline bool is_delim(char c)
        {
            return c == '(' || c == ')' || c == '<' || c == '>' || c == '[' || c == ']' || c == '{' || c == '}' ||
                c == '/' || c == '%';
        }

        inline void utf8_put(std::string& out, std::uint32_t cp)
        {
            if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
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

        inline std::uint32_t utf8_get(const std::string& s, std::size_t& i)
        {
            const unsigned char c = static_cast<unsigned char>(s[i]);
            std::uint32_t cp = c;
            int len = 1;
            if (c >= 0xF0) { cp = c & 0x07; len = 4; }
            else if (c >= 0xE0) { cp = c & 0x0F; len = 3; }
            else if (c >= 0xC0) { cp = c & 0x1F; len = 2; }
            else if (c >= 0x80) { ++i; return 0xFFFD; }
            if (i + static_cast<std::size_t>(len) > s.size()) { i = s.size(); return 0xFFFD; }
            for (int k = 1; k < len; ++k)
            {
                const unsigned char d = static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]);
                if ((d & 0xC0) != 0x80) { ++i; return 0xFFFD; }
                cp = (cp << 6) | (d & 0x3F);
            }
            i += static_cast<std::size_t>(len);
            return cp;
        }

        inline std::string utf16be_to_utf8(const std::string& b)
        {
            std::string out;
            for (std::size_t i = 0; i + 1 < b.size(); i += 2)
            {
                std::uint32_t u = (static_cast<std::uint32_t>(static_cast<unsigned char>(b[i])) << 8) |
                                  static_cast<unsigned char>(b[i + 1]);
                if (u >= 0xD800 && u <= 0xDBFF && i + 3 < b.size())
                {
                    const std::uint32_t lo = (static_cast<std::uint32_t>(static_cast<unsigned char>(b[i + 2])) << 8) |
                                             static_cast<unsigned char>(b[i + 3]);
                    if (lo >= 0xDC00 && lo <= 0xDFFF)
                    {
                        u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                        i += 2;
                    }
                }
                utf8_put(out, u);
            }
            return out;
        }

        inline std::uint32_t pdfdoc_to_unicode(unsigned char c)
        {
            static const std::uint16_t high[33] = {
                0x2022, 0x2020, 0x2021, 0x2026, 0x2014, 0x2013, 0x0192, 0x2044, 0x2039, 0x203A, 0x2212,
                0x2030, 0x201E, 0x201C, 0x201D, 0x2018, 0x2019, 0x201A, 0x2122, 0xFB01, 0xFB02, 0x0141,
                0x0152, 0x0160, 0x0178, 0x017D, 0x0131, 0x0142, 0x0153, 0x0161, 0x017E, 0xFFFD, 0x20AC };
            static const std::uint16_t accents[8] = { 0x02D8, 0x02C7, 0x02C6, 0x02D9, 0x02DD, 0x02DB, 0x02DA, 0x02DC };
            if (c >= 0x80 && c <= 0xA0) return high[c - 0x80];
            if (c >= 0x18 && c <= 0x1F) return accents[c - 0x18];
            return c;
        }

        // A PDF text string: UTF-16BE with BOM, UTF-8 with BOM (PDF 2.0) or PDFDocEncoding.
        inline std::string text_string(const std::string& s)
        {
            if (s.size() >= 2 && static_cast<unsigned char>(s[0]) == 0xFE && static_cast<unsigned char>(s[1]) == 0xFF)
                return utf16be_to_utf8(s.substr(2));
            if (s.size() >= 3 && s.compare(0, 3, "\xEF\xBB\xBF") == 0)
                return s.substr(3);
            std::string out;
            for (unsigned char c : s) utf8_put(out, pdfdoc_to_unicode(c));
            return out;
        }

    // ------------------------------------------------------------------------------------

        class lexer
        {
        public:
            /* objects, when not null, counts the objects read and is checked against
               max_objects. */
            lexer(const std::string& d, std::size_t pos = 0, std::size_t* objects = nullptr)
                : d_(d), p_(pos), objects_(objects)
            {
            }

            void skip_ws()
            {
                while (p_ < d_.size())
                {
                    const char c = d_[p_];
                    if (is_ws(c)) ++p_;
                    else if (c == '%') { while (p_ < d_.size() && d_[p_] != '\n' && d_[p_] != '\r') ++p_; }
                    else break;
                }
            }

            bool at_end() { skip_ws(); return p_ >= d_.size(); }
            std::size_t pos() const { return p_; }
            void seek(std::size_t p) { p_ = std::min(p, d_.size()); }

            object read(bool allow_refs = true, int depth = 0)
            {
                if (depth > 64) throw pdf_error("PDF: nesting too deep");
                skip_ws();
                if (p_ >= d_.size()) throw pdf_error("PDF: unexpected end of data");
                if (objects_ && ++*objects_ > max_objects) throw pdf_error("PDF: the document holds too many objects");
                object o;
                const char c = d_[p_];
                if (c == '/')
                {
                    ++p_;
                    o.kind = object::name_k;
                    o.str = read_name();
                    return o;
                }
                if (c == '(')
                {
                    o.kind = object::str_k;
                    o.str = read_literal();
                    return o;
                }
                if (c == '<')
                {
                    if (p_ + 1 < d_.size() && d_[p_ + 1] == '<')
                    {
                        p_ += 2;
                        o.kind = object::dict_k;
                        for (;;)
                        {
                            skip_ws();
                            if (p_ >= d_.size()) throw pdf_error("PDF: unterminated dictionary");
                            if (d_[p_] == '>' && p_ + 1 < d_.size() && d_[p_ + 1] == '>') { p_ += 2; break; }
                            object key = read(allow_refs, depth + 1);
                            if (!key.is(object::name_k))
                                continue;   // stray token, skipped
                            skip_ws();
                            if (p_ + 1 < d_.size() && d_[p_] == '>' && d_[p_ + 1] == '>')
                            {
                                o.dict[key.str] = object();
                                continue;
                            }
                            o.dict[key.str] = read(allow_refs, depth + 1);
                            if (o.dict.size() > max_container_size) throw pdf_error("PDF: dictionary too large");
                        }
                        return o;
                    }
                    o.kind = object::str_k;
                    o.str = read_hex();
                    return o;
                }
                if (c == '[')
                {
                    ++p_;
                    o.kind = object::array_k;
                    for (;;)
                    {
                        skip_ws();
                        if (p_ >= d_.size()) throw pdf_error("PDF: unterminated array");
                        if (d_[p_] == ']') { ++p_; break; }
                        o.arr.push_back(read(allow_refs, depth + 1));
                        if (o.arr.size() > max_container_size) throw pdf_error("PDF: array too large");
                    }
                    return o;
                }
                if ((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.')
                {
                    const std::size_t start = p_;
                    ++p_;
                    bool integer = c != '.';
                    while (p_ < d_.size() && ((d_[p_] >= '0' && d_[p_] <= '9') || d_[p_] == '.'))
                    {
                        if (d_[p_] == '.') integer = false;
                        ++p_;
                    }
                    o.kind = object::num_k;
                    o.num = parse_number(d_.data() + start, d_.data() + p_);
                    if (allow_refs && integer && o.num >= 0 && o.num < 1e9)
                    {
                        std::size_t q = p_;
                        while (q < d_.size() && is_ws(d_[q])) ++q;
                        const std::size_t g0 = q;
                        while (q < d_.size() && d_[q] >= '0' && d_[q] <= '9') ++q;
                        if (q > g0 && q - g0 < 6)
                        {
                            const int g = std::atoi(d_.substr(g0, q - g0).c_str());
                            while (q < d_.size() && is_ws(d_[q])) ++q;
                            if (q < d_.size() && d_[q] == 'R' &&
                                (q + 1 >= d_.size() || is_ws(d_[q + 1]) || is_delim(d_[q + 1])))
                            {
                                o.kind = object::ref_k;
                                o.ref = static_cast<int>(o.num);
                                o.gen = g;
                                p_ = q + 1;
                            }
                        }
                    }
                    return o;
                }
                if (c == ')' || c == '>' || c == ']' || c == '{' || c == '}')
                {
                    ++p_;
                    o.kind = object::op_k;
                    o.str = std::string(1, c);
                    return o;
                }
                const std::size_t start = p_;
                while (p_ < d_.size() && !is_ws(d_[p_]) && !is_delim(d_[p_])) ++p_;
                if (p_ == start) ++p_;
                const std::string word = d_.substr(start, p_ - start);
                if (word == "true" || word == "false") { o.kind = object::bool_k; o.b = word == "true"; }
                else if (word == "null") o.kind = object::null_k;
                else { o.kind = object::op_k; o.str = word; }
                return o;
            }

        private:
            static int hex_value(char c)
            {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            }

            std::string read_name()
            {
                std::string n;
                while (p_ < d_.size() && !is_ws(d_[p_]) && !is_delim(d_[p_]))
                {
                    if (d_[p_] == '#' && p_ + 2 < d_.size() && hex_value(d_[p_ + 1]) >= 0 && hex_value(d_[p_ + 2]) >= 0)
                    {
                        n.push_back(static_cast<char>(hex_value(d_[p_ + 1]) * 16 + hex_value(d_[p_ + 2])));
                        p_ += 3;
                    }
                    else
                    {
                        n.push_back(d_[p_++]);
                    }
                }
                return n;
            }

            std::string read_literal()
            {
                std::string s;
                ++p_;
                int depth = 1;
                while (p_ < d_.size())
                {
                    char c = d_[p_++];
                    if (c == '\\')
                    {
                        if (p_ >= d_.size()) break;
                        c = d_[p_++];
                        switch (c)
                        {
                            case 'n': s.push_back('\n'); break;
                            case 'r': s.push_back('\r'); break;
                            case 't': s.push_back('\t'); break;
                            case 'b': s.push_back('\b'); break;
                            case 'f': s.push_back('\f'); break;
                            case '\r': if (p_ < d_.size() && d_[p_] == '\n') ++p_; break;
                            case '\n': break;
                            default:
                                if (c >= '0' && c <= '7')
                                {
                                    int v = c - '0';
                                    for (int k = 0; k < 2 && p_ < d_.size() && d_[p_] >= '0' && d_[p_] <= '7'; ++k)
                                        v = v * 8 + (d_[p_++] - '0');
                                    s.push_back(static_cast<char>(v & 0xFF));
                                }
                                else
                                {
                                    s.push_back(c);
                                }
                        }
                        continue;
                    }
                    if (c == '(') ++depth;
                    else if (c == ')' && --depth == 0) break;
                    s.push_back(c);
                }
                return s;
            }

            std::string read_hex()
            {
                ++p_;
                std::string s;
                int pending = -1;
                while (p_ < d_.size() && d_[p_] != '>')
                {
                    const int v = hex_value(d_[p_++]);
                    if (v < 0) continue;
                    if (pending < 0) pending = v;
                    else { s.push_back(static_cast<char>((pending << 4) | v)); pending = -1; }
                }
                if (pending >= 0) s.push_back(static_cast<char>(pending << 4));
                if (p_ < d_.size()) ++p_;
                return s;
            }

            const std::string& d_;
            std::size_t p_;
            std::size_t* objects_;
        };

    // ------------------------------------------------------------------------------------
    // Stream filters.

        inline std::string lzw_decode(const std::string& in, int early_change)
        {
            std::string out;
            std::vector<std::string> table;
            auto reset = [&]() {
                table.clear();
                for (int i = 0; i < 256; ++i) table.push_back(std::string(1, static_cast<char>(i)));
                table.push_back(std::string());
                table.push_back(std::string());
            };
            reset();
            int width = 9;
            std::uint32_t buffer = 0;
            int bits = 0;
            std::string prev;
            for (std::size_t i = 0; i < in.size(); ++i)
            {
                buffer = (buffer << 8) | static_cast<unsigned char>(in[i]);
                bits += 8;
                while (bits >= width)
                {
                    const int code = static_cast<int>((buffer >> (bits - width)) & ((1u << width) - 1));
                    bits -= width;
                    if (code == 256) { reset(); width = 9; prev.clear(); continue; }
                    if (code == 257) return out;
                    std::string entry;
                    if (code < static_cast<int>(table.size()) && (code < 256 || code > 257))
                        entry = table[static_cast<std::size_t>(code)];
                    else if (code == static_cast<int>(table.size()) && !prev.empty()) entry = prev + prev[0];
                    else return out;
                    out += entry;
                    if (out.size() > max_stream_size) throw pdf_error("PDF: LZW stream too large");
                    if (!prev.empty() && table.size() < 4096) table.push_back(prev + entry[0]);
                    prev = entry;
                    const std::size_t limit = (std::size_t(1) << width) - static_cast<std::size_t>(early_change);
                    if (table.size() >= limit && width < 12) ++width;
                }
            }
            return out;
        }

        inline std::string ascii85_decode(const std::string& in)
        {
            std::string out;
            std::uint32_t tuple = 0;
            int count = 0;
            std::size_t i = 0;
            if (in.compare(0, 2, "<~") == 0) i = 2;
            for (; i < in.size(); ++i)
            {
                const char c = in[i];
                if (c == '~') break;
                if (is_ws(c)) continue;
                if (c == 'z' && count == 0) { out.append(4, '\0'); continue; }
                if (c < '!' || c > 'u') continue;
                tuple = tuple * 85 + static_cast<std::uint32_t>(c - '!');
                if (++count == 5)
                {
                    for (int k = 3; k >= 0; --k) out.push_back(static_cast<char>((tuple >> (8 * k)) & 0xFF));
                    tuple = 0;
                    count = 0;
                }
            }
            if (count > 1)
            {
                for (int k = count; k < 5; ++k) tuple = tuple * 85 + 84;
                for (int k = 0; k < count - 1; ++k) out.push_back(static_cast<char>((tuple >> (8 * (3 - k))) & 0xFF));
            }
            return out;
        }

        inline std::string ascii_hex_decode(const std::string& in)
        {
            std::string out;
            int pending = -1;
            for (char c : in)
            {
                if (c == '>') break;
                int v;
                if (c >= '0' && c <= '9') v = c - '0';
                else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
                else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
                else continue;
                if (pending < 0) pending = v;
                else { out.push_back(static_cast<char>((pending << 4) | v)); pending = -1; }
            }
            if (pending >= 0) out.push_back(static_cast<char>(pending << 4));
            return out;
        }

        inline std::string run_length_decode(const std::string& in)
        {
            std::string out;
            std::size_t i = 0;
            while (i < in.size())
            {
                const int len = static_cast<unsigned char>(in[i++]);
                if (len == 128) break;
                if (len < 128)
                {
                    const std::size_t n = static_cast<std::size_t>(len) + 1;
                    out.append(in, i, std::min(n, in.size() - i));
                    i += n;
                }
                else if (i < in.size())
                {
                    out.append(static_cast<std::size_t>(257 - len), in[i++]);
                }
                if (out.size() > max_stream_size) throw pdf_error("PDF: run length stream too large");
            }
            return out;
        }

        inline std::string apply_predictor(const std::string& data, const object* parms)
        {
            if (!parms || !parms->is(object::dict_k)) return data;
            auto num = [&](const char* k, double def) {
                const object* o = parms->get(k);
                return (o && o->is(object::num_k)) ? o->num : def;
            };
            const int predictor = to_int(num("Predictor", 1));
            if (predictor < 2) return data;
            const int colors = std::min(32, std::max(1, to_int(num("Colors", 1))));
            const int bpc = std::min(16, std::max(1, to_int(num("BitsPerComponent", 8))));
            const int columns = std::min(1 << 20, std::max(1, to_int(num("Columns", 1))));
            const std::size_t bpp = static_cast<std::size_t>(std::max(1, colors * bpc / 8));
            const std::size_t row = static_cast<std::size_t>((static_cast<long long>(columns) * colors * bpc + 7) / 8);
            if (predictor == 2)
            {
                std::string out = data;
                if (bpc == 8)
                    for (std::size_t r = 0; r + row <= out.size(); r += row)
                        for (std::size_t k = bpp; k < row; ++k)
                            out[r + k] = static_cast<char>(out[r + k] + out[r + k - bpp]);
                return out;
            }
            std::string out;
            std::string prior(row, '\0');
            std::size_t i = 0;
            while (i < data.size())
            {
                const int type = static_cast<unsigned char>(data[i++]);
                std::string cur = data.substr(i, std::min(row, data.size() - i));
                cur.resize(row, '\0');
                i += row;
                for (std::size_t k = 0; k < row; ++k)
                {
                    const int a = k >= bpp ? static_cast<unsigned char>(cur[k - bpp]) : 0;
                    const int b = static_cast<unsigned char>(prior[k]);
                    const int c = k >= bpp ? static_cast<unsigned char>(prior[k - bpp]) : 0;
                    int x = static_cast<unsigned char>(cur[k]);
                    switch (type)
                    {
                        case 1: x += a; break;
                        case 2: x += b; break;
                        case 3: x += (a + b) / 2; break;
                        case 4:
                        {
                            const int pp = a + b - c;
                            const int pa = std::abs(pp - a), pb = std::abs(pp - b), pc = std::abs(pp - c);
                            x += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
                            break;
                        }
                        default: break;
                    }
                    cur[k] = static_cast<char>(x & 0xFF);
                }
                out += cur;
                prior.swap(cur);
                if (out.size() > max_stream_size) throw pdf_error("PDF: stream too large");
            }
            return out;
        }

    // ------------------------------------------------------------------------------------

        class document
        {
        public:
            explicit document(const std::string& data) : data_(data)
            {
                const std::size_t header = data_.find("%PDF-");
                if (header == std::string::npos || header > 1024)
                    throw pdf_error("PDF: missing %PDF header");
                load_xref();
                if (trailer.get("Encrypt"))
                    throw pdf_error(
                        "PDF: the document is encrypted; decrypt it first, for instance with qpdf --decrypt");
            }

            object trailer;

            // True once a document-wide bound is exceeded; the document is then refused.
            bool over_budget() const
            {
                return decoded_ > max_decoded_size || objects_ > max_objects || operations_ > max_operations;
            }

            // Counts one content operator against max_operations.
            void count_operation()
            {
                if (++operations_ > max_operations)
                    throw pdf_error("PDF: the content streams exceed the work limit");
            }

            // Streams decoded up to an error, for the reading notes.
            std::size_t damaged_streams() const { return damaged_; }

            const object& resolve(const object& o, int depth = 0)
            {
                if (!o.is(object::ref_k)) return o;
                if (depth > 32) return null_;
                return resolve(get_object(o.ref), depth + 1);
            }

            const object* resolve_key(const object& dict, const std::string& key)
            {
                const object* o = dict.get(key);
                if (!o) return nullptr;
                const object& r = resolve(*o);
                return r.is(object::null_k) ? nullptr : &r;
            }

            const object& get_object(int num)
            {
                const auto c = cache_.find(num);
                if (c != cache_.end()) return c->second;
                const auto e = xref_.find(num);
                // Chains of indirect lengths and object streams are cut at a fixed depth.
                if (e == xref_.end() || e->second.type == 0 || loading_.count(num) || loading_.size() > 32)
                    return null_;
                if (e->second.type == 2)
                {
                    loading_.insert(num);
                    try { load_object_stream(to_int(static_cast<double>(e->second.a))); }
                    catch (...) { loading_.erase(num); throw; }
                    loading_.erase(num);
                    const auto c2 = cache_.find(num);
                    return c2 != cache_.end() ? c2->second : null_;
                }
                loading_.insert(num);
                object o;
                try
                {
                    o = parse_indirect_at(e->second.a, num);
                }
                catch (pdf_error&)
                {
                    if (over_budget()) { loading_.erase(num); throw; }
                    // A damaged table may point elsewhere: the object is looked up by scanning.
                    o = object();
                    const std::size_t found = scanned_offset(num);
                    if (found != std::string::npos && found != e->second.a)
                    {
                        try { o = parse_indirect_at(found, num); }
                        catch (pdf_error&) { if (over_budget()) { loading_.erase(num); throw; } o = object(); }
                    }
                }
                loading_.erase(num);
                return cache_[num] = o;
            }

            std::string stream_data(const object& s)
            {
                if (!s.is(object::stream_k) || !s.buffer) return std::string();
                std::string data = s.buffer->substr(s.data_begin, s.data_end - s.data_begin);
                const object* f = resolve_key(s, "Filter");
                if (!f)
                {
                    charge(data.size());
                    return data;
                }
                const object* parms = resolve_key(s, "DecodeParms");
                if (!parms) parms = resolve_key(s, "DP");
                std::vector<const object*> filters, params;
                if (f->is(object::name_k)) { filters.push_back(f); params.push_back(parms); }
                else if (f->is(object::array_k))
                {
                    for (std::size_t i = 0; i < f->arr.size(); ++i)
                    {
                        filters.push_back(&resolve(f->arr[i]));
                        const object* parm = nullptr;
                        if (parms && parms->is(object::array_k) && i < parms->arr.size())
                            parm = &resolve(parms->arr[i]);
                        else if (parms && parms->is(object::dict_k) && f->arr.size() == 1) parm = parms;
                        params.push_back(parm);
                    }
                }
                for (std::size_t i = 0; i < filters.size(); ++i)
                {
                    if (!filters[i]->is(object::name_k)) return std::string();
                    const std::string& n = filters[i]->str;
                    if (n == "FlateDecode" || n == "Fl")
                    {
                        /* One inflate pass, past the zlib header when there is one; the
                           Adler-32 trailer is not required, since producers often get it
                           wrong. */
                        const bool zlib_header = data.size() >= 2 && (static_cast<unsigned char>(data[0]) & 0x0F) == 8 &&
                            ((static_cast<unsigned char>(data[0]) << 8) | static_cast<unsigned char>(data[1])) % 31 == 0;
                        // A damaged stream keeps what was decoded before the damage.
                        bool complete = true;
                        const std::string out =
                            inflate_raw_prefix(zlib_header ? data.substr(2) : data, max_stream_size, complete);
                        if (!complete) ++damaged_;
                        data = apply_predictor(out, params[i]);
                    }
                    else if (n == "LZWDecode" || n == "LZW")
                    {
                        int early = 1;
                        if (params[i] && params[i]->is(object::dict_k))
                            if (const object* ec = params[i]->get("EarlyChange"))
                                if (ec->is(object::num_k)) early = ec->num == 0 ? 0 : 1;
                        data = apply_predictor(lzw_decode(data, early), params[i]);
                    }
                    else if (n == "ASCIIHexDecode" || n == "AHx") data = ascii_hex_decode(data);
                    else if (n == "ASCII85Decode" || n == "A85") data = ascii85_decode(data);
                    else if (n == "RunLengthDecode" || n == "RL") data = run_length_decode(data);
                    else return std::string();   // image codecs and encryption filters
                    charge(data.size());
                }
                return data;
            }

        private:
            struct entry { int type; std::size_t a; int b; };

            void charge(std::size_t bytes)
            {
                decoded_ += bytes;
                if (decoded_ > max_decoded_size)
                    throw pdf_error("PDF: the decoded streams exceed the size limit");
            }

            /* The object at off, whose header must carry the number expected when one is
               given. */
            object parse_indirect_at(std::size_t off, int expected = -1)
            {
                lexer lx(data_, off, &objects_);
                const object num = lx.read(false);
                const object gen = lx.read(false);
                const object kw = lx.read(false);
                if (!num.is(object::num_k) || !gen.is(object::num_k) || !kw.is(object::op_k) || kw.str != "obj")
                    throw pdf_error("PDF: bad object header");
                if (expected >= 0 && to_int(num.num) != expected)
                    throw pdf_error("PDF: the cross-reference data points to another object");
                object o = lx.read(true);
                lx.skip_ws();
                const std::size_t p = lx.pos();
                if (o.is(object::dict_k) && data_.compare(p, 6, "stream") == 0)
                {
                    std::size_t start = p + 6;
                    if (start < data_.size() && data_[start] == '\r') ++start;
                    if (start < data_.size() && data_[start] == '\n') ++start;
                    std::size_t len = std::string::npos;
                    if (const object* l = o.get("Length"))
                    {
                        if (l->is(object::num_k) && l->num >= 0) len = to_size(l->num);
                        else if (l->is(object::ref_k))
                        {
                            const object& r = get_object(l->ref);
                            if (r.is(object::num_k) && r.num >= 0) len = to_size(r.num);
                        }
                    }
                    bool ok = start <= data_.size() && len != std::string::npos && len <= data_.size() - start;
                    if (ok)
                    {
                        std::size_t q = start + len;
                        while (q < data_.size() && is_ws(data_[q])) ++q;
                        ok = data_.compare(q, 9, "endstream") == 0;
                    }
                    if (!ok)
                    {
                        const std::size_t es = next_endstream(start);
                        if (es == std::string::npos) throw pdf_error("PDF: unterminated stream");
                        len = es - start;
                        if (len > 0 && data_[start + len - 1] == '\n') --len;
                        if (len > 0 && data_[start + len - 1] == '\r') --len;
                    }
                    o.kind = object::stream_k;
                    o.data_begin = start;
                    o.data_end = start + len;
                    o.buffer = &data_;
                }
                return o;
            }

            void load_object_stream(int num)
            {
                if (loaded_streams_.count(num)) return;
                loaded_streams_.insert(num);
                const object& s = get_object(num);
                if (!s.is(object::stream_k)) return;
                const object* n = s.get("N");
                const object* first = s.get("First");
                if (!n || !first || !n->is(object::num_k) || !first->is(object::num_k)) return;
                const std::string buf = stream_data(s);
                lexer header(buf, 0, &objects_);
                std::vector<std::pair<int, std::size_t>> items;
                try
                {
                    for (int i = 0; i < to_int(n->num) && i < 1000000; ++i)
                    {
                        const object a = header.read(false);
                        const object b = header.read(false);
                        if (!a.is(object::num_k) || !b.is(object::num_k)) break;
                        items.push_back(std::make_pair(to_int(a.num), to_size(first->num + b.num)));
                    }
                }
                catch (pdf_error&) { if (over_budget()) throw; }
                for (const auto& it : items)
                {
                    if (cache_.count(it.first)) continue;
                    // Only the version the cross-reference data points to is taken.
                    const auto e = xref_.find(it.first);
                    if (e != xref_.end() && !(e->second.type == 2 && e->second.a == static_cast<std::size_t>(num)))
                        continue;
                    try
                    {
                        lexer lx(buf, it.second, &objects_);
                        cache_[it.first] = lx.read(true);
                    }
                    catch (pdf_error&) { if (over_budget()) throw; }
                }
            }

            void add_entry(int num, const entry& e)
            {
                if (num < 0 || xref_.count(num)) return;
                xref_[num] = e;
            }

            bool parse_xref_section(std::size_t off, object& tr, std::vector<std::pair<int, entry>>& entries)
            {
                if (off >= data_.size()) return false;
                lexer lx(data_, off, &objects_);
                lx.skip_ws();
                if (data_.compare(lx.pos(), 4, "xref") == 0)
                {
                    lx.seek(lx.pos() + 4);
                    for (;;)
                    {
                        lx.skip_ws();
                        if (data_.compare(lx.pos(), 7, "trailer") == 0)
                        {
                            lx.seek(lx.pos() + 7);
                            tr = lx.read(true);
                            return tr.is(object::dict_k);
                        }
                        const object start = lx.read(false);
                        const object count = lx.read(false);
                        if (!start.is(object::num_k) || !count.is(object::num_k) || count.num < 0 ||
                            count.num > 10000000)
                            return false;
                        for (int i = 0; i < to_int(count.num); ++i)
                        {
                            const object o = lx.read(false);
                            const object g = lx.read(false);
                            const object t = lx.read(false);
                            if (!o.is(object::num_k) || !t.is(object::op_k)) return false;
                            entries.push_back(std::make_pair(to_int(start.num) + i,
                                                             entry{ t.str == "n" ? 1 : 0, to_size(o.num), to_int(g.num) }));
                        }
                    }
                }
                const object xs = parse_indirect_at(lx.pos());
                const object* type = xs.get("Type");
                if (!xs.is(object::stream_k) || !type || type->str != "XRef") return false;
                const std::string d = stream_data(xs);
                const object* w = xs.get("W");
                if (!w || !w->is(object::array_k) || w->arr.size() < 3) return false;
                int wv[3];
                for (int k = 0; k < 3; ++k)
                {
                    wv[k] = to_int(w->arr[static_cast<std::size_t>(k)].num);
                    if (wv[k] < 0 || wv[k] > 8) return false;
                }
                std::vector<std::pair<int, int>> ranges;
                const object* idx = xs.get("Index");
                const object* size = xs.get("Size");
                if (idx && idx->is(object::array_k))
                    for (std::size_t k = 0; k + 1 < idx->arr.size(); k += 2)
                        ranges.push_back(std::make_pair(to_int(idx->arr[k].num), to_int(idx->arr[k + 1].num)));
                else if (size)
                    ranges.push_back(std::make_pair(0, to_int(size->num)));
                const std::size_t rec = static_cast<std::size_t>(wv[0] + wv[1] + wv[2]);
                if (rec == 0) return false;
                std::size_t pos = 0;
                auto field = [&](int width, std::uint64_t def) {
                    if (width == 0) return def;
                    std::uint64_t v = 0;
                    for (int k = 0; k < width && pos < d.size(); ++k)
                        v = (v << 8) | static_cast<unsigned char>(d[pos++]);
                    return v;
                };
                for (const auto& r : ranges)
                {
                    for (int i = 0; i < r.second && pos + rec <= d.size(); ++i)
                    {
                        const int t = static_cast<int>(field(wv[0], 1) & 0xFF);
                        const std::uint64_t a = field(wv[1], 0);
                        const std::uint64_t b = field(wv[2], 0);
                        entries.push_back(std::make_pair(
                            r.first + i, entry{ t, to_size(static_cast<double>(a)), static_cast<int>(b & 0xFFFF) }));
                    }
                }
                tr = xs;
                tr.kind = object::dict_k;
                return true;
            }

            bool parse_xref_chain(std::size_t off)
            {
                std::set<std::size_t> seen;
                bool first = true;
                while (seen.insert(off).second && seen.size() < 4096)
                {
                    object tr;
                    std::vector<std::pair<int, entry>> entries;
                    bool ok = false;
                    try { ok = parse_xref_section(off, tr, entries); }
                    catch (pdf_error&) { if (over_budget()) throw; ok = false; }
                    if (!ok) return !first;
                    if (first) { trailer = tr; first = false; }
                    else
                    {
                        for (const char* k : { "Root", "Info", "Encrypt", "ID" })
                            if (!trailer.get(k) && tr.get(k)) trailer.dict[k] = *tr.get(k);
                    }
                    // In a hybrid file the stream entries take precedence over the table.
                    if (const object* xs = tr.get("XRefStm"))
                    {
                        if (xs->is(object::num_k) && xs->num >= 0)
                        {
                            object dummy;
                            std::vector<std::pair<int, entry>> hidden;
                            try
                            {
                                if (parse_xref_section(to_size(xs->num), dummy, hidden))
                                    for (const auto& e : hidden) add_entry(e.first, e.second);
                            }
                            catch (pdf_error&) { if (over_budget()) throw; }
                        }
                    }
                    for (const auto& e : entries) add_entry(e.first, e.second);
                    const object* prev = tr.get("Prev");
                    if (!prev || !prev->is(object::num_k) || prev->num < 0) break;
                    off = to_size(prev->num);
                }
                return !first;
            }

            void load_xref()
            {
                bool ok = false;
                const std::size_t sx = data_.rfind("startxref");
                if (sx != std::string::npos)
                {
                    try
                    {
                        lexer lx(data_, sx + 9);
                        const object o = lx.read(false);
                        if (o.is(object::num_k) && o.num >= 0) ok = parse_xref_chain(to_size(o.num));
                    }
                    catch (pdf_error&) { if (over_budget()) throw; ok = false; }
                }
                if (ok)
                {
                    // A damaged table often has offsets that miss: check that the catalog resolves.
                    const object* root = trailer.get("Root");
                    if (root && root->is(object::ref_k) && get_object(root->ref).is(object::dict_k))
                        return;
                }
                reconstruct();
            }

            // Offsets of the "N G obj" headers found by scanning the file, the last one of each N.
            const std::map<int, std::size_t>& scanned_objects()
            {
                if (scanned_done_) return scanned_;
                scanned_done_ = true;
                std::size_t p = 0;
                while ((p = data_.find("obj", p)) != std::string::npos)
                {
                    const std::size_t after = p + 3;
                    if (p > 0 && is_ws(data_[p - 1]) &&
                        (after >= data_.size() || is_ws(data_[after]) || is_delim(data_[after])))
                    {
                        std::size_t q = p - 1;
                        while (q > 0 && is_ws(data_[q])) --q;
                        const std::size_t g_end = q + 1;
                        while (q > 0 && data_[q] >= '0' && data_[q] <= '9') --q;
                        const std::size_t g_begin = q + 1;
                        if (g_begin < g_end && q > 0 && is_ws(data_[q]))
                        {
                            while (q > 0 && is_ws(data_[q])) --q;
                            const std::size_t n_end = q + 1;
                            while (q > 0 && data_[q] >= '0' && data_[q] <= '9') --q;
                            const std::size_t n_begin = (data_[q] >= '0' && data_[q] <= '9') ? q : q + 1;
                            if (n_begin < n_end && n_end - n_begin < 9)
                                scanned_[std::atoi(data_.substr(n_begin, n_end - n_begin).c_str())] = n_begin;
                        }
                    }
                    p = after;
                }
                return scanned_;
            }

            std::size_t scanned_offset(int num)
            {
                const std::map<int, std::size_t>& m = scanned_objects();
                const auto it = m.find(num);
                return it == m.end() ? std::string::npos : it->second;
            }

            // The first "endstream" at or after from, from a table built once.
            std::size_t next_endstream(std::size_t from)
            {
                if (!endstreams_done_)
                {
                    endstreams_done_ = true;
                    for (std::size_t p = data_.find("endstream"); p != std::string::npos; p = data_.find("endstream", p + 9))
                        endstreams_.push_back(p);
                }
                const auto it = std::lower_bound(endstreams_.begin(), endstreams_.end(), from);
                return it == endstreams_.end() ? std::string::npos : *it;
            }

            void reconstruct()
            {
                xref_.clear();
                cache_.clear();
                loaded_streams_.clear();
                const std::map<int, std::size_t>& found = scanned_objects();
                for (const auto& f : found)
                    xref_[f.first] = entry{ 1, f.second, 0 };

                // Objects held in object streams.
                std::vector<int> xref_streams;
                for (const auto& f : found)
                {
                    const object& o = get_object(f.first);
                    if (!o.is(object::stream_k)) continue;
                    const object* t = o.get("Type");
                    if (t && t->is(object::name_k) && t->str == "ObjStm") load_object_stream(f.first);
                    if (t && t->is(object::name_k) && t->str == "XRef") xref_streams.push_back(f.first);
                }

                trailer = object();
                trailer.kind = object::dict_k;
                const std::size_t tp = data_.rfind("trailer");
                if (tp != std::string::npos)
                {
                    try
                    {
                        lexer lx(data_, tp + 7, &objects_);
                        const object t = lx.read(true);
                        if (t.is(object::dict_k)) trailer = t;
                    }
                    catch (pdf_error&) { if (over_budget()) throw; }
                }
                if (!trailer.get("Root"))
                {
                    for (auto it = xref_streams.rbegin(); it != xref_streams.rend(); ++it)
                    {
                        const object& o = get_object(*it);
                        if (o.get("Root"))
                        {
                            trailer = o;
                            trailer.kind = object::dict_k;
                            break;
                        }
                    }
                }
                const object* root = trailer.get("Root");
                if (root && root->is(object::ref_k) && get_object(root->ref).is(object::dict_k))
                    return;
                for (auto& c : cache_)
                {
                    const object* t = c.second.get("Type");
                    if (c.second.is(object::dict_k) && t && t->is(object::name_k) && t->str == "Catalog")
                    {
                        object r;
                        r.kind = object::ref_k;
                        r.ref = c.first;
                        trailer.dict["Root"] = r;
                        return;
                    }
                }
                throw pdf_error("PDF: no document catalog found");
            }

            const std::string& data_;
            std::map<int, entry> xref_;
            std::map<int, object> cache_;
            std::set<int> loading_;
            std::set<int> loaded_streams_;
            std::map<int, std::size_t> scanned_;
            bool scanned_done_ = false;
            std::vector<std::size_t> endstreams_;
            bool endstreams_done_ = false;
            std::size_t decoded_ = 0;
            std::size_t objects_ = 0;
            std::size_t operations_ = 0;
            std::size_t damaged_ = 0;
            object null_;
        };

    // ------------------------------------------------------------------------------------
    // Encodings and glyph names.

        inline std::uint32_t winansi_to_unicode(unsigned char c)
        {
            static const std::uint16_t high[32] = { 0x20AC, 0,      0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
                                                    0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0,      0x017D, 0,
                                                    0,      0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
                                                    0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0,      0x017E, 0x0178 };
            if (c >= 0x80 && c < 0xA0) return high[c - 0x80];
            if (c < 0x20) return 0;
            return c;
        }

        inline std::uint32_t macroman_to_unicode(unsigned char c)
        {
            static const std::uint16_t high[128] = {
                0x00C4, 0x00C5, 0x00C7, 0x00C9, 0x00D1, 0x00D6, 0x00DC, 0x00E1, 0x00E0, 0x00E2, 0x00E4, 0x00E3, 0x00E5,
                0x00E7, 0x00E9, 0x00E8, 0x00EA, 0x00EB, 0x00ED, 0x00EC, 0x00EE, 0x00EF, 0x00F1, 0x00F3, 0x00F2, 0x00F4,
                0x00F6, 0x00F5, 0x00FA, 0x00F9, 0x00FB, 0x00FC, 0x2020, 0x00B0, 0x00A2, 0x00A3, 0x00A7, 0x2022, 0x00B6,
                0x00DF, 0x00AE, 0x00A9, 0x2122, 0x00B4, 0x00A8, 0x2260, 0x00C6, 0x00D8, 0x221E, 0x00B1, 0x2264, 0x2265,
                0x00A5, 0x00B5, 0x2202, 0x2211, 0x220F, 0x03C0, 0x222B, 0x00AA, 0x00BA, 0x03A9, 0x00E6, 0x00F8, 0x00BF,
                0x00A1, 0x00AC, 0x221A, 0x0192, 0x2248, 0x2206, 0x00AB, 0x00BB, 0x2026, 0x00A0, 0x00C0, 0x00C3, 0x00D5,
                0x0152, 0x0153, 0x2013, 0x2014, 0x201C, 0x201D, 0x2018, 0x2019, 0x00F7, 0x25CA, 0x00FF, 0x0178, 0x2044,
                0x20AC, 0x2039, 0x203A, 0xFB01, 0xFB02, 0x2021, 0x00B7, 0x201A, 0x201E, 0x2030, 0x00C2, 0x00CA, 0x00C1,
                0x00CB, 0x00C8, 0x00CD, 0x00CE, 0x00CF, 0x00CC, 0x00D3, 0x00D4, 0xF8FF, 0x00D2, 0x00DA, 0x00DB, 0x00D9,
                0x0131, 0x02C6, 0x02DC, 0x00AF, 0x02D8, 0x02D9, 0x02DA, 0x00B8, 0x02DD, 0x02DB, 0x02C7
            };
            if (c >= 0x80) return high[c - 0x80];
            if (c < 0x20) return 0;
            return c;
        }

        inline std::uint32_t standard_to_unicode(unsigned char c)
        {
            if (c == 0x27) return 0x2019;
            if (c == 0x60) return 0x2018;
            if (c >= 0x20 && c < 0x7F) return c;
            switch (c)
            {
                case 0xA1: return 0x00A1; case 0xA2: return 0x00A2; case 0xA3: return 0x00A3; case 0xA4: return 0x2044;
                case 0xA5: return 0x00A5; case 0xA6: return 0x0192; case 0xA7: return 0x00A7; case 0xA8: return 0x00A4;
                case 0xA9: return 0x0027; case 0xAA: return 0x201C; case 0xAB: return 0x00AB; case 0xAC: return 0x2039;
                case 0xAD: return 0x203A; case 0xAE: return 0xFB01; case 0xAF: return 0xFB02; case 0xB1: return 0x2013;
                case 0xB2: return 0x2020; case 0xB3: return 0x2021; case 0xB4: return 0x00B7; case 0xB6: return 0x00B6;
                case 0xB7: return 0x2022; case 0xB8: return 0x201A; case 0xB9: return 0x201E; case 0xBA: return 0x201D;
                case 0xBB: return 0x00BB; case 0xBC: return 0x2026; case 0xBD: return 0x2030; case 0xBF: return 0x00BF;
                case 0xC1: return 0x0060; case 0xC2: return 0x00B4; case 0xC3: return 0x02C6; case 0xC4: return 0x02DC;
                case 0xC5: return 0x00AF; case 0xC6: return 0x02D8; case 0xC7: return 0x02D9; case 0xC8: return 0x00A8;
                case 0xCA: return 0x02DA; case 0xCB: return 0x00B8; case 0xCD: return 0x02DD; case 0xCE: return 0x02DB;
                case 0xCF: return 0x02C7; case 0xD0: return 0x2014; case 0xE1: return 0x00C6; case 0xE3: return 0x00AA;
                case 0xE8: return 0x0141; case 0xE9: return 0x00D8; case 0xEA: return 0x0152; case 0xEB: return 0x00BA;
                case 0xF1: return 0x00E6; case 0xF5: return 0x0131; case 0xF8: return 0x0142; case 0xF9: return 0x00F8;
                case 0xFA: return 0x0153; case 0xFB: return 0x00DF;
                default: return 0;
            }
        }

        /* Adobe Glyph List names of the Latin Extended-A block, U+0100 to U+017F, and of
           the Romanian comma below letters; latin_extended_a_base() gives the ASCII letter
           whose Helvetica width each one takes. */
        inline const char* latin_extended_a_name(std::uint32_t cp)
        {
            static const char* const names[128] = { "Amacron",       "amacron",       "Abreve",       "abreve",
                                                    "Aogonek",       "aogonek",       "Cacute",       "cacute",
                                                    "Ccircumflex",   "ccircumflex",   "Cdotaccent",   "cdotaccent",
                                                    "Ccaron",        "ccaron",        "Dcaron",       "dcaron",
                                                    "Dcroat",        "dcroat",        "Emacron",      "emacron",
                                                    "Ebreve",        "ebreve",        "Edotaccent",   "edotaccent",
                                                    "Eogonek",       "eogonek",       "Ecaron",       "ecaron",
                                                    "Gcircumflex",   "gcircumflex",   "Gbreve",       "gbreve",
                                                    "Gdotaccent",    "gdotaccent",    "Gcommaaccent", "gcommaaccent",
                                                    "Hcircumflex",   "hcircumflex",   "Hbar",         "hbar",
                                                    "Itilde",        "itilde",        "Imacron",      "imacron",
                                                    "Ibreve",        "ibreve",        "Iogonek",      "iogonek",
                                                    "Idotaccent",    "dotlessi",      "IJ",           "ij",
                                                    "Jcircumflex",   "jcircumflex",   "Kcommaaccent", "kcommaaccent",
                                                    "kgreenlandic",  "Lacute",        "lacute",       "Lcommaaccent",
                                                    "lcommaaccent",  "Lcaron",        "lcaron",       "Ldot",
                                                    "ldot",          "Lslash",        "lslash",       "Nacute",
                                                    "nacute",        "Ncommaaccent",  "ncommaaccent", "Ncaron",
                                                    "ncaron",        "napostrophe",   "Eng",          "eng",
                                                    "Omacron",       "omacron",       "Obreve",       "obreve",
                                                    "Ohungarumlaut", "ohungarumlaut", "OE",           "oe",
                                                    "Racute",        "racute",        "Rcommaaccent", "rcommaaccent",
                                                    "Rcaron",        "rcaron",        "Sacute",       "sacute",
                                                    "Scircumflex",   "scircumflex",   "Scedilla",     "scedilla",
                                                    "Scaron",        "scaron",        "Tcommaaccent", "tcommaaccent",
                                                    "Tcaron",        "tcaron",        "Tbar",         "tbar",
                                                    "Utilde",        "utilde",        "Umacron",      "umacron",
                                                    "Ubreve",        "ubreve",        "Uring",        "uring",
                                                    "Uhungarumlaut", "uhungarumlaut", "Uogonek",      "uogonek",
                                                    "Wcircumflex",   "wcircumflex",   "Ycircumflex",  "ycircumflex",
                                                    "Ydieresis",     "Zacute",        "zacute",       "Zdotaccent",
                                                    "zdotaccent",    "Zcaron",        "zcaron",       "longs" };
            if (cp >= 0x100 && cp <= 0x17F) return names[cp - 0x100];
            if (cp == 0x218) return "Scommaaccent";
            if (cp == 0x219) return "scommaaccent";
            if (cp == 0x21A) return "uni021A";
            if (cp == 0x21B) return "uni021B";
            return nullptr;
        }

        inline char latin_extended_a_base(std::uint32_t cp)
        {
            static const char base[] =
                "AaAaAaCcCcCcCcDdDdEeEeEeEeEeGgGgGgGgHhHhIiIiIiIiIiJjJjKkkLlLlLlLlLlNnNnNnnNnOoOoOoOoRrRr"
                "RrSsSsSsSsTtTtTtUuUuUuUuUuUuWwYyYZzZzZzf";
            if (cp >= 0x100 && cp <= 0x17F) return base[cp - 0x100];
            if (cp == 0x218 || cp == 0x219) return cp == 0x218 ? 'S' : 's';
            if (cp == 0x21A || cp == 0x21B) return cp == 0x21A ? 'T' : 't';
            return 0;
        }

        inline std::uint32_t glyph_name_to_unicode(const std::string& n, int depth = 0)
        {
            static const std::map<std::string, std::uint32_t> names = []() {
                std::map<std::string, std::uint32_t> m;
                const char* ascii[] = { "space",     "exclam",      "quotedbl",  "numbersign", "dollar",   "percent",
                                        "ampersand", "quotesingle", "parenleft", "parenright", "asterisk", "plus",
                                        "comma",     "hyphen",      "period",    "slash",      "zero",     "one",
                                        "two",       "three",       "four",      "five",       "six",      "seven",
                                        "eight",     "nine",        "colon",     "semicolon",  "less",     "equal",
                                        "greater",   "question",    "at" };
                for (int i = 0; i < 33; ++i) m[ascii[i]] = static_cast<std::uint32_t>(0x20 + i);
                for (int i = 0; i < 26; ++i)
                {
                    m[std::string(1, static_cast<char>('A' + i))] = static_cast<std::uint32_t>('A' + i);
                    m[std::string(1, static_cast<char>('a' + i))] = static_cast<std::uint32_t>('a' + i);
                }
                for (std::uint32_t cp = 0x100; cp <= 0x17F; ++cp) m[latin_extended_a_name(cp)] = cp;
                m["Scommaaccent"] = 0x218;
                m["scommaaccent"] = 0x219;
                const std::pair<const char*, std::uint32_t> extra[] = { { "bracketleft", 0x5B },
                                                                        { "backslash", 0x5C },
                                                                        { "bracketright", 0x5D },
                                                                        { "asciicircum", 0x5E },
                                                                        { "underscore", 0x5F },
                                                                        { "grave", 0x60 },
                                                                        { "braceleft", 0x7B },
                                                                        { "bar", 0x7C },
                                                                        { "braceright", 0x7D },
                                                                        { "asciitilde", 0x7E },
                                                                        { "quoteleft", 0x2018 },
                                                                        { "quoteright", 0x2019 },
                                                                        { "quotedblleft", 0x201C },
                                                                        { "quotedblright", 0x201D },
                                                                        { "quotesinglbase", 0x201A },
                                                                        { "quotedblbase", 0x201E },
                                                                        { "endash", 0x2013 },
                                                                        { "emdash", 0x2014 },
                                                                        { "bullet", 0x2022 },
                                                                        { "ellipsis", 0x2026 },
                                                                        { "dagger", 0x2020 },
                                                                        { "daggerdbl", 0x2021 },
                                                                        { "perthousand", 0x2030 },
                                                                        { "trademark", 0x2122 },
                                                                        { "Euro", 0x20AC },
                                                                        { "fi", 0xFB01 },
                                                                        { "fl", 0xFB02 },
                                                                        { "ff", 0xFB00 },
                                                                        { "ffi", 0xFB03 },
                                                                        { "ffl", 0xFB04 },
                                                                        { "florin", 0x192 },
                                                                        { "circumflex", 0x2C6 },
                                                                        { "tilde", 0x2DC },
                                                                        { "caron", 0x2C7 },
                                                                        { "breve", 0x2D8 },
                                                                        { "dotaccent", 0x2D9 },
                                                                        { "ring", 0x2DA },
                                                                        { "ogonek", 0x2DB },
                                                                        { "hungarumlaut", 0x2DD },
                                                                        { "minus", 0x2212 },
                                                                        { "nbspace", 0xA0 },
                                                                        { "sfthyphen", 0xAD },
                                                                        { "guilsinglleft", 0x2039 },
                                                                        { "guilsinglright", 0x203A },
                                                                        { "fraction", 0x2044 },
                                                                        { "exclamdown", 0xA1 },
                                                                        { "cent", 0xA2 },
                                                                        { "sterling", 0xA3 },
                                                                        { "currency", 0xA4 },
                                                                        { "yen", 0xA5 },
                                                                        { "brokenbar", 0xA6 },
                                                                        { "section", 0xA7 },
                                                                        { "dieresis", 0xA8 },
                                                                        { "copyright", 0xA9 },
                                                                        { "ordfeminine", 0xAA },
                                                                        { "guillemotleft", 0xAB },
                                                                        { "logicalnot", 0xAC },
                                                                        { "registered", 0xAE },
                                                                        { "macron", 0xAF },
                                                                        { "degree", 0xB0 },
                                                                        { "plusminus", 0xB1 },
                                                                        { "twosuperior", 0xB2 },
                                                                        { "threesuperior", 0xB3 },
                                                                        { "acute", 0xB4 },
                                                                        { "mu", 0xB5 },
                                                                        { "paragraph", 0xB6 },
                                                                        { "periodcentered", 0xB7 },
                                                                        { "cedilla", 0xB8 },
                                                                        { "onesuperior", 0xB9 },
                                                                        { "ordmasculine", 0xBA },
                                                                        { "guillemotright", 0xBB },
                                                                        { "onequarter", 0xBC },
                                                                        { "onehalf", 0xBD },
                                                                        { "threequarters", 0xBE },
                                                                        { "questiondown", 0xBF },
                                                                        { "multiply", 0xD7 },
                                                                        { "divide", 0xF7 },
                                                                        { "Agrave", 0xC0 },
                                                                        { "Aacute", 0xC1 },
                                                                        { "Acircumflex", 0xC2 },
                                                                        { "Atilde", 0xC3 },
                                                                        { "Adieresis", 0xC4 },
                                                                        { "Aring", 0xC5 },
                                                                        { "AE", 0xC6 },
                                                                        { "Ccedilla", 0xC7 },
                                                                        { "Egrave", 0xC8 },
                                                                        { "Eacute", 0xC9 },
                                                                        { "Ecircumflex", 0xCA },
                                                                        { "Edieresis", 0xCB },
                                                                        { "Igrave", 0xCC },
                                                                        { "Iacute", 0xCD },
                                                                        { "Icircumflex", 0xCE },
                                                                        { "Idieresis", 0xCF },
                                                                        { "Eth", 0xD0 },
                                                                        { "Ntilde", 0xD1 },
                                                                        { "Ograve", 0xD2 },
                                                                        { "Oacute", 0xD3 },
                                                                        { "Ocircumflex", 0xD4 },
                                                                        { "Otilde", 0xD5 },
                                                                        { "Odieresis", 0xD6 },
                                                                        { "Oslash", 0xD8 },
                                                                        { "Ugrave", 0xD9 },
                                                                        { "Uacute", 0xDA },
                                                                        { "Ucircumflex", 0xDB },
                                                                        { "Udieresis", 0xDC },
                                                                        { "Yacute", 0xDD },
                                                                        { "Thorn", 0xDE },
                                                                        { "germandbls", 0xDF },
                                                                        { "agrave", 0xE0 },
                                                                        { "aacute", 0xE1 },
                                                                        { "acircumflex", 0xE2 },
                                                                        { "atilde", 0xE3 },
                                                                        { "adieresis", 0xE4 },
                                                                        { "aring", 0xE5 },
                                                                        { "ae", 0xE6 },
                                                                        { "ccedilla", 0xE7 },
                                                                        { "egrave", 0xE8 },
                                                                        { "eacute", 0xE9 },
                                                                        { "ecircumflex", 0xEA },
                                                                        { "edieresis", 0xEB },
                                                                        { "igrave", 0xEC },
                                                                        { "iacute", 0xED },
                                                                        { "icircumflex", 0xEE },
                                                                        { "idieresis", 0xEF },
                                                                        { "eth", 0xF0 },
                                                                        { "ntilde", 0xF1 },
                                                                        { "ograve", 0xF2 },
                                                                        { "oacute", 0xF3 },
                                                                        { "ocircumflex", 0xF4 },
                                                                        { "otilde", 0xF5 },
                                                                        { "odieresis", 0xF6 },
                                                                        { "oslash", 0xF8 },
                                                                        { "ugrave", 0xF9 },
                                                                        { "uacute", 0xFA },
                                                                        { "ucircumflex", 0xFB },
                                                                        { "udieresis", 0xFC },
                                                                        { "yacute", 0xFD },
                                                                        { "thorn", 0xFE },
                                                                        { "ydieresis", 0xFF } };
                for (const auto& e : extra) m[e.first] = e.second;
                return m;
            }();
            const auto it = names.find(n);
            if (it != names.end()) return it->second;
            auto hex_ok = [&](std::size_t from) {
                for (std::size_t k = from; k < n.size(); ++k)
                    if (!std::isxdigit(static_cast<unsigned char>(n[k]))) return false;
                return true;
            };
            if (n.size() == 7 && n.compare(0, 3, "uni") == 0 && hex_ok(3))
                return static_cast<std::uint32_t>(std::strtoul(n.c_str() + 3, nullptr, 16));
            if (n.size() >= 5 && n.size() <= 7 && n[0] == 'u' && hex_ok(1))
                return static_cast<std::uint32_t>(std::strtoul(n.c_str() + 1, nullptr, 16));
            const std::size_t dot = n.find('.');
            if (dot != std::string::npos && dot > 0 && depth == 0) return glyph_name_to_unicode(n.substr(0, dot), 1);
            return 0;
        }

        // Advance widths of the standard Helvetica font, in thousandths of an em.
        inline double helvetica_width(std::uint32_t cp)
        {
            static const short ascii[95] = {
                278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
                556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
                1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
                667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
                333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
                556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584 };
            static const short latin1[96] = {
                278, 333, 556, 556, 556, 556, 260, 556, 333, 737, 370, 556, 584, 333, 737, 333,
                400, 584, 333, 333, 333, 556, 537, 278, 333, 333, 365, 556, 834, 834, 834, 611,
                667, 667, 667, 667, 667, 667, 1000, 722, 667, 667, 667, 667, 278, 278, 278, 278,
                722, 722, 778, 778, 778, 778, 778, 584, 778, 722, 722, 722, 722, 667, 667, 611,
                556, 556, 556, 556, 556, 556, 889, 500, 556, 556, 556, 556, 278, 278, 278, 278,
                556, 556, 556, 556, 556, 556, 556, 584, 611, 556, 556, 556, 556, 500, 556, 500 };
            if (cp >= 32 && cp < 127) return ascii[cp - 32];
            if (cp >= 0xA0 && cp <= 0xFF) return latin1[cp - 0xA0];
            switch (cp)
            {
                case 0x20AC: return 556; case 0x201A: return 222; case 0x0192: return 556; case 0x201E: return 333;
                case 0x2026: return 1000; case 0x2020: case 0x2021: return 556; case 0x2030: return 1000;
                case 0x02C6: case 0x02DC: return 333; case 0x0160: return 667; case 0x2039: case 0x203A: return 333;
                case 0x0152: return 1000; case 0x017D: return 611; case 0x2018: case 0x2019: return 222;
                case 0x201C: case 0x201D: return 333; case 0x2022: return 350; case 0x2013: return 556;
                case 0x2014: return 1000; case 0x2122: return 1000; case 0x0161: return 500; case 0x0153: return 944;
                case 0x017E: return 500; case 0x0178: return 667; case 0x0132: return 778; case 0x0133: return 444;
                default: break;
            }
            const char base = latin_extended_a_base(cp);
            if (base) return ascii[base - 32];
            return 556;
        }

    // ------------------------------------------------------------------------------------
    // Fonts.

        // Codes lo to hi mapped to consecutive values: Unicode text, CIDs or widths.
        struct code_range
        {
            std::uint32_t hi = 0;
            std::string first;              // ToUnicode: UTF-16BE text of lo
            std::uint32_t cid = 0;          // CMap: CID of lo
            double width = 0;               // widths: constant width of the range
            std::vector<double> widths;     // widths: one width per code, when listed
        };

        typedef std::map<std::uint32_t, code_range> range_map;

        // The range holding code, or nullptr; ranges are keyed by their first code.
        inline const code_range* find_range(const range_map& m, std::uint32_t code, std::uint32_t* lo = nullptr)
        {
            auto it = m.upper_bound(code);
            if (it == m.begin()) return nullptr;
            --it;
            if (code > it->second.hi) return nullptr;
            if (lo) *lo = it->first;
            return &it->second;
        }

        struct font
        {
            bool composite = false;
            bool two_byte = false;          // Identity and UCS-2 encodings: fixed 2 byte codes
            bool ucs2 = false;              // codes are UTF-16 code units
            std::vector<std::pair<std::string, std::string>> codespaces;
            std::map<std::uint32_t, std::string> to_unicode;
            range_map unicode_ranges;       // bfrange entries with a string destination
            range_map cids;                 // code to CID, from an embedded CMap
            range_map cid_widths;
            std::uint32_t simple[256] = {};
            bool differs[256] = {};         // set by /Differences
            double widths[256] = {};
            bool has_width[256] = {};
            double missing = 0;
            bool fixed_pitch = false;
            double default_width = 1000;
            double scale = 0.001;

            // Unicode text of a code from the ToUnicode map; false if it has none.
            bool unicode_of(std::uint32_t code, std::string& text) const
            {
                const auto it = to_unicode.find(code);
                if (it != to_unicode.end())
                {
                    text = it->second;
                    return true;
                }
                std::uint32_t lo = 0;
                const code_range* r = find_range(unicode_ranges, code, &lo);
                if (!r) return false;
                // The destination of lo plus the offset, as a big-endian integer.
                std::string dst = r->first;
                std::uint32_t carry = code - lo;
                for (std::size_t k = dst.size(); k-- > 0 && carry != 0;)
                {
                    const std::uint32_t v = static_cast<unsigned char>(dst[k]) + (carry & 0xFF);
                    dst[k] = static_cast<char>(v & 0xFF);
                    carry = (carry >> 8) + (v >> 8);
                }
                text = utf16be_to_utf8(dst);
                return true;
            }

            std::uint32_t cid_of(std::uint32_t code) const
            {
                std::uint32_t lo = 0;
                const code_range* r = find_range(cids, code, &lo);
                return r ? r->cid + (code - lo) : code;
            }

            double cid_width(std::uint32_t cid) const
            {
                std::uint32_t lo = 0;
                const code_range* r = find_range(cid_widths, cid, &lo);
                if (!r) return default_width;
                if (!r->widths.empty()) return cid - lo < r->widths.size() ? r->widths[cid - lo] : default_width;
                return r->width;
            }
        };

        struct glyph
        {
            std::uint32_t code = 0;
            int bytes = 1;
            std::string text;
            double width = 0;           // text space units for a 1 point font
            bool known = true;
        };

        inline void parse_cmap(const std::string& data, font& f)
        {
            lexer lx(data);
            std::vector<object> stack;
            auto code_of = [](const std::string& s) {
                std::uint32_t v = 0;
                for (std::size_t k = 0; k < s.size() && k < 4; ++k) v = (v << 8) | static_cast<unsigned char>(s[k]);
                return v;
            };
            try
            {
                while (!lx.at_end())
                {
                    object o = lx.read(false);
                    if (!o.is(object::op_k))
                    {
                        stack.push_back(o);
                        if (stack.size() > 300000) stack.clear();
                        continue;
                    }
                    if (o.str == "endcodespacerange")
                    {
                        for (std::size_t i = 0; i + 1 < stack.size(); i += 2)
                            if (stack[i].is(object::str_k) && stack[i + 1].is(object::str_k) &&
                                stack[i].str.size() == stack[i + 1].str.size() && !stack[i].str.empty() &&
                                stack[i].str.size() <= 4)
                                f.codespaces.push_back(std::make_pair(stack[i].str, stack[i + 1].str));
                    }
                    else if (o.str == "endbfchar")
                    {
                        for (std::size_t i = 0; i + 1 < stack.size(); i += 2)
                        {
                            if (!stack[i].is(object::str_k)) continue;
                            if (stack[i + 1].is(object::str_k))
                                f.to_unicode[code_of(stack[i].str)] = utf16be_to_utf8(stack[i + 1].str);
                            else if (stack[i + 1].is(object::name_k))
                            {
                                std::string t;
                                const std::uint32_t cp = glyph_name_to_unicode(stack[i + 1].str);
                                if (cp) { utf8_put(t, cp); f.to_unicode[code_of(stack[i].str)] = t; }
                            }
                        }
                    }
                    else if (o.str == "endbfrange")
                    {
                        for (std::size_t i = 0; i + 2 < stack.size(); i += 3)
                        {
                            if (!stack[i].is(object::str_k) || !stack[i + 1].is(object::str_k)) continue;
                            const std::uint32_t lo = code_of(stack[i].str), hi = code_of(stack[i + 1].str);
                            if (hi < lo) continue;
                            if (stack[i + 2].is(object::str_k))
                            {
                                code_range r;
                                r.hi = hi;
                                r.first = stack[i + 2].str;
                                f.unicode_ranges[lo] = r;
                            }
                            else if (stack[i + 2].is(object::array_k))
                            {
                                const std::vector<object>& a = stack[i + 2].arr;
                                for (std::size_t k = 0; k < a.size() && k <= hi - lo; ++k)
                                    if (a[k].is(object::str_k))
                                        f.to_unicode[lo + static_cast<std::uint32_t>(k)] = utf16be_to_utf8(a[k].str);
                            }
                        }
                    }
                    else if (o.str == "endcidrange")
                    {
                        for (std::size_t i = 0; i + 2 < stack.size(); i += 3)
                        {
                            if (!stack[i].is(object::str_k) || !stack[i + 1].is(object::str_k) ||
                                !stack[i + 2].is(object::num_k))
                                continue;
                            const std::uint32_t lo = code_of(stack[i].str), hi = code_of(stack[i + 1].str);
                            if (hi < lo) continue;
                            code_range r;
                            r.hi = hi;
                            r.cid = to_u32(stack[i + 2].num);
                            f.cids[lo] = r;
                        }
                    }
                    else if (o.str == "endcidchar")
                    {
                        for (std::size_t i = 0; i + 1 < stack.size(); i += 2)
                        {
                            if (!stack[i].is(object::str_k) || !stack[i + 1].is(object::num_k)) continue;
                            const std::uint32_t c = code_of(stack[i].str);
                            code_range r;
                            r.hi = c;
                            r.cid = to_u32(stack[i + 1].num);
                            f.cids[c] = r;
                        }
                    }
                    stack.clear();
                }
            }
            catch (pdf_error&) {}
        }

        inline font build_font(document& doc, const object& fd)
        {
            font f;
            const object* subtype = doc.resolve_key(fd, "Subtype");
            const std::string st = subtype && subtype->is(object::name_k) ? subtype->str : "";
            if (const object* tu = doc.resolve_key(fd, "ToUnicode"))
                if (tu->is(object::stream_k)) parse_cmap(doc.stream_data(*tu), f);

            if (st == "Type0")
            {
                f.composite = true;
                const object* enc = doc.resolve_key(fd, "Encoding");
                if (enc && enc->is(object::name_k))
                {
                    const std::string& e = enc->str;
                    f.ucs2 = e.find("UCS2") != std::string::npos || e.find("UTF16") != std::string::npos;
                    f.two_byte = e.compare(0, 8, "Identity") == 0 || f.ucs2;
                }
                else if (enc && enc->is(object::stream_k))
                {
                    font tmp;
                    parse_cmap(doc.stream_data(*enc), tmp);
                    if (!tmp.codespaces.empty()) f.codespaces = tmp.codespaces;
                    f.cids.swap(tmp.cids);
                }
                else
                {
                    f.two_byte = true;
                }
                if (const object* df = doc.resolve_key(fd, "DescendantFonts"))
                {
                    if (df->is(object::array_k) && !df->arr.empty())
                    {
                        const object& d = doc.resolve(df->arr[0]);
                        if (const object* dw = doc.resolve_key(d, "DW"))
                            if (dw->is(object::num_k)) f.default_width = dw->num;
                        const object* w = doc.resolve_key(d, "W");
                        if (w && w->is(object::array_k))
                        {
                            // "c [w1 w2 ...]" lists widths from c on; "c_first c_last w" gives one width.
                            const std::vector<object>& a = w->arr;
                            for (std::size_t i = 0; i < a.size();)
                            {
                                const object& c0 = doc.resolve(a[i]);
                                if (!c0.is(object::num_k) || i + 1 >= a.size()) break;
                                const object& n1 = doc.resolve(a[i + 1]);
                                const std::uint32_t lo = to_u32(c0.num);
                                code_range r;
                                if (n1.is(object::array_k))
                                {
                                    if (n1.arr.empty()) { i += 2; continue; }
                                    for (const object& x : n1.arr) r.widths.push_back(doc.resolve(x).num);
                                    r.hi = lo + static_cast<std::uint32_t>(std::min<std::size_t>(r.widths.size() - 1,
                                                                                                 0xFFFFFFFFu - lo));
                                    i += 2;
                                }
                                else if (i + 2 < a.size())
                                {
                                    r.hi = to_u32(n1.num);
                                    r.width = doc.resolve(a[i + 2]).num;
                                    i += 3;
                                    if (r.hi < lo) continue;
                                }
                                else break;
                                f.cid_widths[lo] = r;
                            }
                        }
                    }
                }
                return f;
            }

            // Simple fonts.
            if (st == "Type3")
            {
                if (const object* fm = doc.resolve_key(fd, "FontMatrix"))
                    if (fm->is(object::array_k) && !fm->arr.empty()) f.scale = doc.resolve(fm->arr[0]).num;
            }
            std::string base;
            if (const object* bf = doc.resolve_key(fd, "BaseFont")) if (bf->is(object::name_k)) base = bf->str;
            f.fixed_pitch = base.find("Courier") != std::string::npos;
            const object* widths = doc.resolve_key(fd, "Widths");
            if (widths && widths->is(object::array_k))
            {
                int first = 0;
                if (const object* fc = doc.resolve_key(fd, "FirstChar")) first = to_int(fc->num);
                for (std::size_t i = 0; i < widths->arr.size() && i < 256; ++i)
                {
                    const int code = first + static_cast<int>(i);
                    if (code < 0 || code > 255) continue;
                    f.widths[code] = doc.resolve(widths->arr[i]).num;
                    f.has_width[code] = true;
                }
            }
            if (const object* desc = doc.resolve_key(fd, "FontDescriptor"))
                if (const object* mw = doc.resolve_key(*desc, "MissingWidth")) f.missing = mw->num;

            std::string base_encoding = (st == "TrueType") ? "WinAnsiEncoding" : "StandardEncoding";
            const object* enc = doc.resolve_key(fd, "Encoding");
            const object* differences = nullptr;
            if (enc && enc->is(object::name_k)) base_encoding = enc->str;
            else if (enc && enc->is(object::dict_k))
            {
                if (const object* be = doc.resolve_key(*enc, "BaseEncoding"))
                    if (be->is(object::name_k)) base_encoding = be->str;
                differences = doc.resolve_key(*enc, "Differences");
            }
            for (int c = 0; c < 256; ++c)
            {
                const unsigned char uc = static_cast<unsigned char>(c);
                if (base_encoding == "WinAnsiEncoding") f.simple[c] = winansi_to_unicode(uc);
                else if (base_encoding == "MacRomanEncoding") f.simple[c] = macroman_to_unicode(uc);
                else if (base_encoding == "PDFDocEncoding") f.simple[c] = c >= 0x20 ? pdfdoc_to_unicode(uc) : 0;
                else f.simple[c] = standard_to_unicode(uc);
            }
            if (differences && differences->is(object::array_k))
            {
                int code = 0;
                for (const object& d : differences->arr)
                {
                    const object& r = doc.resolve(d);
                    if (r.is(object::num_k)) code = to_int(r.num);
                    else if (r.is(object::name_k))
                    {
                        if (code >= 0 && code < 256)
                        {
                            f.simple[code] = glyph_name_to_unicode(r.str);
                            f.differs[code] = true;
                        }
                        ++code;
                    }
                }
            }
            return f;
        }

        inline std::vector<glyph> decode_string(const font& f, const std::string& s, std::size_t& unknown)
        {
            std::vector<glyph> out;
            if (f.composite)
            {
                std::size_t i = 0;
                while (i < s.size())
                {
                    int len = 0;
                    if (f.two_byte) len = 2;
                    else if (!f.codespaces.empty())
                    {
                        for (int n = 1; n <= 4 && len == 0; ++n)
                        {
                            if (i + static_cast<std::size_t>(n) > s.size()) break;
                            for (const auto& cs : f.codespaces)
                            {
                                if (cs.first.size() != static_cast<std::size_t>(n)) continue;
                                bool in = true;
                                for (int k = 0; k < n && in; ++k)
                                {
                                    const unsigned char c =
                                        static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]);
                                    in = c >= static_cast<unsigned char>(cs.first[static_cast<std::size_t>(k)]) &&
                                         c <= static_cast<unsigned char>(cs.second[static_cast<std::size_t>(k)]);
                                }
                                if (in) { len = n; break; }
                            }
                        }
                    }
                    if (len == 0) len = 2;
                    if (i + static_cast<std::size_t>(len) > s.size()) len = static_cast<int>(s.size() - i);
                    // UTF-16 encodings: a surrogate pair is one character.
                    if (f.ucs2 && len == 2 && i + 4 <= s.size())
                    {
                        const unsigned char h = static_cast<unsigned char>(s[i]), l = static_cast<unsigned char>(s[i + 2]);
                        if (h >= 0xD8 && h <= 0xDB && l >= 0xDC && l <= 0xDF) len = 4;
                    }
                    glyph g;
                    for (int k = 0; k < len; ++k)
                        g.code = (g.code << 8) | static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]);
                    g.bytes = len;
                    if (!f.unicode_of(g.code, g.text))
                    {
                        if (f.ucs2) g.text = utf16be_to_utf8(s.substr(i, static_cast<std::size_t>(len)));
                        else { g.known = false; ++unknown; }
                    }
                    g.width = f.cid_width(f.cid_of(g.code)) * 0.001;
                    out.push_back(g);
                    i += static_cast<std::size_t>(len);
                }
                return out;
            }
            for (unsigned char c : s)
            {
                glyph g;
                g.code = c;
                const std::uint32_t cp = f.simple[c];
                if (!f.unicode_of(c, g.text))
                {
                    if (cp != 0) utf8_put(g.text, cp);
                    // A code renamed by /Differences to an unknown glyph name is not read as ASCII.
                    else if (c >= 0x20 && c < 0x7F && !f.differs[c]) g.text = std::string(1, static_cast<char>(c));
                    else { g.known = false; ++unknown; }
                }
                double w;
                if (f.has_width[c]) w = f.widths[c];
                else if (f.missing > 0) w = f.missing;
                else if (f.fixed_pitch) w = 600;
                else w = helvetica_width(cp ? cp : c);
                g.width = w * f.scale;
                out.push_back(g);
            }
            return out;
        }

    // ------------------------------------------------------------------------------------
    // Content stream interpretation.

        struct pdf_matrix
        {
            double a = 1, b = 0, c = 0, d = 1, e = 0, f = 0;
            pdf_matrix() {}
            pdf_matrix(double a_, double b_, double c_, double d_, double e_, double f_)
                : a(a_), b(b_), c(c_), d(d_), e(e_), f(f_)
            {
            }
            pdf_matrix operator*(const pdf_matrix& m) const
            {
                return pdf_matrix(a * m.a + b * m.c, a * m.b + b * m.d, c * m.a + d * m.c, c * m.b + d * m.d,
                              e * m.a + f * m.c + m.e, e * m.b + f * m.d + m.f);
            }
        };

        // Fonts built once per document, by the address of their dictionary in the object cache.
        typedef std::map<const object*, font> font_cache;

        class interpreter
        {
        public:
            interpreter(document& doc, font_cache& fonts, pdf_page_text& page, std::size_t& unknown)
                : doc_(doc), fonts_(fonts), page_(page), unknown_(unknown)
            {
            }

            // Text runs shown with a font missing from the resources, and shown strings cut short.
            std::size_t missing_font_runs() const { return missing_font_runs_; }
            std::size_t truncated_strings() const { return truncated_strings_; }

            void run(const std::string& content, const object* resources, const pdf_matrix& ctm, int depth)
            {
                if (depth > 12) return;
                state s;
                s.ctm = ctm;
                std::vector<state> saved;
                std::vector<object> operands;
                pdf_matrix tm, tlm;
                lexer lx(content);
                std::size_t guard = 0;
                while (!lx.at_end() && ++guard < 50000000)
                {
                    object o;
                    try { o = lx.read(false); }
                    catch (pdf_error&) { break; }
                    if (!o.is(object::op_k))
                    {
                        operands.push_back(o);
                        if (operands.size() > 4096) operands.clear();
                        continue;
                    }
                    doc_.count_operation();
                    const std::string& op = o.str;
                    auto num = [&](std::size_t k) {
                        return (k < operands.size() && operands[k].is(object::num_k)) ? operands[k].num : 0.0;
                    };
                    if (op == "q") { if (saved.size() < 1000) saved.push_back(s); }
                    else if (op == "Q") { if (!saved.empty()) { s = saved.back(); saved.pop_back(); } }
                    else if (op == "cm" && operands.size() >= 6)
                        s.ctm = pdf_matrix(num(0), num(1), num(2), num(3), num(4), num(5)) * s.ctm;
                    else if (op == "BT") { tm = pdf_matrix(); tlm = pdf_matrix(); }
                    else if (op == "Tf" && operands.size() >= 2)
                    {
                        s.size = num(1);
                        s.current_font =
                            operands[0].is(object::name_k) ? lookup_font(resources, operands[0].str) : nullptr;
                    }
                    else if (op == "Tc") s.tc = num(0);
                    else if (op == "Tw") s.tw = num(0);
                    else if (op == "Tz") s.th = num(0) / 100.0;
                    else if (op == "TL") s.tl = num(0);
                    else if (op == "Ts") s.ts = num(0);
                    else if (op == "Td" && operands.size() >= 2)
                    {
                        tlm = pdf_matrix(1, 0, 0, 1, num(0), num(1)) * tlm;
                        tm = tlm;
                    }
                    else if (op == "TD" && operands.size() >= 2)
                    {
                        s.tl = -num(1);
                        tlm = pdf_matrix(1, 0, 0, 1, num(0), num(1)) * tlm;
                        tm = tlm;
                    }
                    else if (op == "Tm" && operands.size() >= 6)
                    {
                        tlm = pdf_matrix(num(0), num(1), num(2), num(3), num(4), num(5));
                        tm = tlm;
                    }
                    else if (op == "T*") { tlm = pdf_matrix(1, 0, 0, 1, 0, -s.tl) * tlm; tm = tlm; }
                    else if (op == "Tj" && !operands.empty() && operands[0].is(object::str_k))
                        show(s, tm, operands[0].str);
                    else if (op == "'" && !operands.empty())
                    {
                        tlm = pdf_matrix(1, 0, 0, 1, 0, -s.tl) * tlm;
                        tm = tlm;
                        if (operands.back().is(object::str_k)) show(s, tm, operands.back().str);
                    }
                    else if (op == "\"" && operands.size() >= 3)
                    {
                        s.tw = num(0);
                        s.tc = num(1);
                        tlm = pdf_matrix(1, 0, 0, 1, 0, -s.tl) * tlm;
                        tm = tlm;
                        if (operands[2].is(object::str_k)) show(s, tm, operands[2].str);
                    }
                    else if (op == "TJ" && !operands.empty() && operands[0].is(object::array_k))
                    {
                        for (const object& e : operands[0].arr)
                        {
                            if (e.is(object::str_k)) show(s, tm, e.str);
                            else if (e.is(object::num_k))
                                tm = pdf_matrix(1, 0, 0, 1, -e.num / 1000.0 * s.size * s.th, 0) * tm;
                        }
                    }
                    else if (op == "Do" && !operands.empty() && operands[0].is(object::name_k))
                        do_xobject(resources, operands[0].str, s.ctm, depth);
                    else if (op == "BI")
                        skip_inline_image(lx, content);
                    operands.clear();
                }
            }

            void run_form(const object& form, const object* fallback_resources, const pdf_matrix& ctm, int depth)
            {
                const object* res = doc_.resolve_key(form, "Resources");
                run(doc_.stream_data(form), res ? res : fallback_resources, form_matrix(form) * ctm, depth + 1);
            }

            pdf_matrix form_matrix(const object& form)
            {
                pdf_matrix m;
                if (const object* mo = doc_.resolve_key(form, "Matrix"))
                    if (mo->is(object::array_k) && mo->arr.size() >= 6)
                        m = pdf_matrix(doc_.resolve(mo->arr[0]).num, doc_.resolve(mo->arr[1]).num,
                                   doc_.resolve(mo->arr[2]).num, doc_.resolve(mo->arr[3]).num,
                                   doc_.resolve(mo->arr[4]).num, doc_.resolve(mo->arr[5]).num);
                return m;
            }

        private:
            struct state
            {
                pdf_matrix ctm;
                const font* current_font = nullptr;
                double size = 0, tc = 0, tw = 0, th = 1, tl = 0, ts = 0;
            };

            const font* lookup_font(const object* resources, const std::string& name)
            {
                if (!resources) return nullptr;
                const object* fonts = doc_.resolve_key(*resources, "Font");
                if (!fonts || !fonts->is(object::dict_k)) return nullptr;
                const object* entry = fonts->get(name);
                if (!entry) return nullptr;
                // Resolved objects live in the document cache, so their address identifies them.
                const object& fd = doc_.resolve(*entry);
                if (!fd.is(object::dict_k)) return nullptr;
                const auto it = fonts_.find(&fd);
                if (it != fonts_.end()) return &it->second;
                return &(fonts_[&fd] = build_font(doc_, fd));
            }

            void show(const state& s, pdf_matrix& tm, const std::string& shown)
            {
                if (shown.empty()) return;
                const bool cut = shown.size() > max_shown_string;
                if (cut) ++truncated_strings_;
                std::string part;
                if (cut) part = shown.substr(0, max_shown_string);
                const std::string& bytes = cut ? part : shown;
                std::vector<glyph> glyphs;
                if (s.current_font)
                {
                    glyphs = decode_string(*s.current_font, bytes, unknown_);
                }
                else
                {
                    // Without its font the text cannot be decoded; each byte is an unknown glyph.
                    ++missing_font_runs_;
                    unknown_ += bytes.size();
                    glyphs.resize(bytes.size());
                    for (glyph& g : glyphs)
                    {
                        g.known = false;
                        g.width = 0.5;
                    }
                }
                const pdf_matrix trm = pdf_matrix(s.size * s.th, 0, 0, s.size, 0, s.ts) * tm * s.ctm;
                const pdf_matrix tc = tm * s.ctm;
                pdf_text_span span;
                span.x = trm.e;
                span.y = trm.f;
                span.font_size = std::abs(s.size) * std::sqrt(tc.c * tc.c + tc.d * tc.d);
                span.angle = std::atan2(tc.b, tc.a) * 180.0 / pi;
                if (std::abs(span.angle) < 0.01) span.angle = 0;
                double advance = 0;
                for (const glyph& g : glyphs)
                {
                    span.text += g.known ? g.text : std::string("\xEF\xBF\xBD");
                    const double tx = (g.width * s.size + s.tc + ((g.bytes == 1 && g.code == 32) ? s.tw : 0)) * s.th;
                    advance += tx;
                    tm = pdf_matrix(1, 0, 0, 1, tx, 0) * tm;
                }
                span.width = advance * std::sqrt(tc.a * tc.a + tc.b * tc.b);
                if (!span.text.empty() && std::isfinite(span.x) && std::isfinite(span.y) &&
                    page_.spans.size() < 5000000)
                    page_.spans.push_back(span);
            }

            void do_xobject(const object* resources, const std::string& name, const pdf_matrix& ctm, int depth)
            {
                if (!resources) return;
                const object* xobjects = doc_.resolve_key(*resources, "XObject");
                if (!xobjects || !xobjects->is(object::dict_k)) return;
                const object* entry = xobjects->get(name);
                if (!entry) return;
                const object& x = doc_.resolve(*entry);
                if (!x.is(object::stream_k)) return;
                const object* subtype = x.get("Subtype");
                if (subtype && subtype->str == "Image") { ++page_.images; return; }
                if (!subtype || subtype->str != "Form") return;
                if (active_forms_.count(&x) || active_forms_.size() > 64) return;
                active_forms_.insert(&x);
                run_form(x, resources, ctm, depth);
                active_forms_.erase(&x);
            }

            void skip_inline_image(lexer& lx, const std::string& content)
            {
                // Dictionary entries up to ID, then binary data up to an EI between white space.
                while (!lx.at_end())
                {
                    object o;
                    try { o = lx.read(false); }
                    catch (pdf_error&) { return; }
                    if (o.is(object::op_k) && o.str == "ID") break;
                }
                ++page_.images;
                std::size_t p = lx.pos() + 1;
                while (p + 2 <= content.size())
                {
                    if (content[p] == 'E' && content[p + 1] == 'I' && is_ws(content[p - 1]) &&
                        (p + 2 == content.size() || is_ws(content[p + 2])))
                    {
                        lx.seek(p + 2);
                        return;
                    }
                    ++p;
                }
                lx.seek(content.size());
            }

            document& doc_;
            font_cache& fonts_;
            pdf_page_text& page_;
            std::size_t& unknown_;
            std::set<const object*> active_forms_;
            std::size_t missing_font_runs_ = 0;
            std::size_t truncated_strings_ = 0;
        };

        /* Annotation appearances drawn as page text: the form XObject's bounding box,
           transformed by its matrix, is mapped onto the annotation rectangle. */
        inline void read_annotations(document& doc, interpreter& it, const object& page_dict, pdf_page_text& page,
                                     const object* default_resources)
        {
            const object* annots = doc.resolve_key(page_dict, "Annots");
            if (!annots || !annots->is(object::array_k)) return;
            for (const object& a : annots->arr)
            {
                const object& ad = doc.resolve(a);
                if (!ad.is(object::dict_k)) continue;
                const object* subtype = doc.resolve_key(ad, "Subtype");
                const std::string st = subtype && subtype->is(object::name_k) ? subtype->str : "";
                if (st == "Popup") continue;
                const object* flags = doc.resolve_key(ad, "F");
                const int fl = flags && flags->is(object::num_k) ? to_int(flags->num) : 0;
                const bool drawn = st == "Widget" || st == "FreeText" || st == "Stamp";
                if (!drawn || (fl & 2) || (fl & 32))
                {
                    ++page.annotations;
                    continue;
                }
                double r[4] = { 0, 0, 0, 0 };
                const object* rect = doc.resolve_key(ad, "Rect");
                if (!rect || !rect->is(object::array_k) || rect->arr.size() < 4) { ++page.annotations; continue; }
                for (int k = 0; k < 4; ++k) r[k] = doc.resolve(rect->arr[static_cast<std::size_t>(k)]).num;
                if (r[0] > r[2]) std::swap(r[0], r[2]);
                if (r[1] > r[3]) std::swap(r[1], r[3]);

                const object* ap = doc.resolve_key(ad, "AP");
                const object* n = ap ? doc.resolve_key(*ap, "N") : nullptr;
                if (n && n->is(object::dict_k))
                {
                    const object* as = doc.resolve_key(ad, "AS");
                    n = (as && as->is(object::name_k)) ? doc.resolve_key(*n, as->str) : nullptr;
                }
                if (n && n->is(object::stream_k))
                {
                    double b[4] = { 0, 0, 1, 1 };
                    if (const object* bbox = doc.resolve_key(*n, "BBox"))
                        if (bbox->is(object::array_k) && bbox->arr.size() >= 4)
                            for (int k = 0; k < 4; ++k) b[k] = doc.resolve(bbox->arr[static_cast<std::size_t>(k)]).num;
                    const pdf_matrix fm = it.form_matrix(*n);
                    double xs[4], ys[4];
                    const double cx[4] = { b[0], b[2], b[0], b[2] }, cy[4] = { b[1], b[1], b[3], b[3] };
                    for (int k = 0; k < 4; ++k)
                    {
                        xs[k] = fm.a * cx[k] + fm.c * cy[k] + fm.e;
                        ys[k] = fm.b * cx[k] + fm.d * cy[k] + fm.f;
                    }
                    const double bx0 = *std::min_element(xs, xs + 4), bx1 = *std::max_element(xs, xs + 4);
                    const double by0 = *std::min_element(ys, ys + 4), by1 = *std::max_element(ys, ys + 4);
                    const double sx = bx1 - bx0 > 1e-6 ? (r[2] - r[0]) / (bx1 - bx0) : 1;
                    const double sy = by1 - by0 > 1e-6 ? (r[3] - r[1]) / (by1 - by0) : 1;
                    const pdf_matrix to_rect(sx, 0, 0, sy, r[0] - sx * bx0, r[1] - sy * by0);
                    it.run_form(*n, default_resources, to_rect, 0);
                    continue;
                }
                // A text field without appearance: its value, on one line inside the rectangle.
                const object* v = doc.resolve_key(ad, "V");
                const object* parent = doc.resolve_key(ad, "Parent");
                if (!v && parent && parent->is(object::dict_k)) v = doc.resolve_key(*parent, "V");
                if (v && v->is(object::str_k) && !v->str.empty())
                {
                    pdf_text_span span;
                    span.text = text_string(v->str);
                    for (char& c : span.text) if (c == '\r' || c == '\n') c = ' ';
                    double fs = 10;
                    const object* da = doc.resolve_key(ad, "DA");
                    if (!da && parent && parent->is(object::dict_k)) da = doc.resolve_key(*parent, "DA");
                    if (da && da->is(object::str_k))
                    {
                        const std::size_t tf = da->str.find("Tf");
                        if (tf != std::string::npos)
                        {
                            std::size_t e = tf;
                            while (e > 0 && is_ws(da->str[e - 1])) --e;
                            std::size_t b0 = e;
                            while (
                                b0 > 0 &&
                                (std::isdigit(static_cast<unsigned char>(da->str[b0 - 1])) || da->str[b0 - 1] == '.'))
                                --b0;
                            const double size = parse_number(da->str.substr(b0, e - b0));
                            if (size > 0) fs = size;
                        }
                    }
                    fs = std::min(fs, std::max(4.0, (r[3] - r[1]) * 0.8));
                    span.font_size = fs;
                    span.x = r[0] + 2;
                    span.y = r[1] + std::max(0.0, (r[3] - r[1] - fs) / 2) + 0.2 * fs;
                    double w = 0;
                    for (std::size_t i = 0; i < span.text.size();) w += helvetica_width(utf8_get(span.text, i));
                    span.width = w * fs / 1000.0;
                    page.spans.push_back(span);
                    continue;
                }
                if (st != "Widget") ++page.annotations;
            }
        }

        inline void list_features(document& doc, const object& root, pdf_text_document& out)
        {
            if (const object* info = doc.resolve_key(doc.trailer, "Info"))
            {
                if (info->is(object::dict_k) && !info->dict.empty())
                {
                    std::string keys;
                    for (const auto& kv : info->dict)
                        keys += (keys.empty() ? "" : ", ") + kv.first;
                    out.features.push_back("document information (" + keys + ")");
                }
            }
            if (doc.resolve_key(root, "Metadata")) out.features.push_back("XMP metadata");
            if (const object* ol = doc.resolve_key(root, "Outlines"))
                if (ol->is(object::dict_k) && ol->get("First")) out.features.push_back("bookmarks");
            if (const object* names = doc.resolve_key(root, "Names"))
            {
                if (names->is(object::dict_k))
                {
                    if (names->get("EmbeddedFiles")) out.features.push_back("embedded files");
                    if (names->get("JavaScript")) out.features.push_back("JavaScript");
                }
            }
            if (const object* oa = doc.resolve_key(root, "OpenAction"))
                if (oa->is(object::dict_k) && oa->get("JS")) out.features.push_back("JavaScript");
            if (const object* af = doc.resolve_key(root, "AcroForm"))
            {
                if (af->is(object::dict_k))
                {
                    const object* fields = doc.resolve_key(*af, "Fields");
                    if (fields && fields->is(object::array_k) && !fields->arr.empty())
                        out.features.push_back("form fields");
                    if (af->get("XFA")) out.features.push_back("XFA form");
                }
            }
            if (doc.resolve_key(root, "StructTreeRoot")) out.features.push_back("structure tree");
            if (doc.resolve_key(root, "AA")) out.features.push_back("document actions");
        }
    }

// ----------------------------------------------------------------------------------------

    inline pdf_text_document read_pdf_text(const std::string& bytes)
    {
        using namespace pdf_impl;
        document doc(bytes);
        pdf_text_document out;
        const object* root = doc.resolve_key(doc.trailer, "Root");
        if (!root || !root->is(object::dict_k))
            throw pdf_error("PDF: missing document catalog");
        if (const object* mark = root->get("DlibPrivacy"))
            out.produced_by_dlib = mark->is(object::bool_k) && mark->b;
        list_features(doc, *root, out);
        const object* pages = doc.resolve_key(*root, "Pages");
        if (!pages || !pages->is(object::dict_k)) throw pdf_error("PDF: missing page tree");
        const object* default_resources = nullptr;
        if (const object* af = doc.resolve_key(*root, "AcroForm"))
            if (af->is(object::dict_k)) default_resources = doc.resolve_key(*af, "DR");

        struct inherited { const object* resources = nullptr; const object* mediabox = nullptr; int rotate = 0; };
        std::set<const object*> visited;
        std::vector<std::pair<const object*, inherited>> stack;
        stack.push_back(std::make_pair(pages, inherited()));
        std::vector<std::pair<const object*, inherited>> leaves;
        while (!stack.empty())
        {
            const auto node = stack.back();
            stack.pop_back();
            inherited inh = node.second;
            const object& n = *node.first;
            if (const object* r = doc.resolve_key(n, "Resources")) if (r->is(object::dict_k)) inh.resources = r;
            if (const object* m = doc.resolve_key(n, "MediaBox")) if (m->is(object::array_k)) inh.mediabox = m;
            if (const object* rot = doc.resolve_key(n, "Rotate"))
                if (rot->is(object::num_k)) inh.rotate = to_int(rot->num);
            const object* kids = doc.resolve_key(n, "Kids");
            const object* type = doc.resolve_key(n, "Type");
            if (kids && kids->is(object::array_k) && !(type && type->is(object::name_k) && type->str == "Page"))
            {
                for (std::size_t k = kids->arr.size(); k-- > 0;)
                {
                    const object& kr = doc.resolve(kids->arr[k]);
                    if (!kr.is(object::dict_k) || visited.count(&kr) || visited.size() > 200000) continue;
                    visited.insert(&kr);
                    stack.push_back(std::make_pair(&kr, inh));
                }
            }
            else
            {
                leaves.push_back(std::make_pair(&n, inh));
            }
        }

        font_cache fonts;
        std::size_t damaged = doc.damaged_streams();
        if (damaged > 0)
            out.notes.push_back(std::to_string(damaged) + " damaged stream(s) read up to the damage");
        for (const auto& leaf : leaves)
        {
            pdf_page_text page;
            const inherited& inh = leaf.second;
            if (inh.mediabox && inh.mediabox->arr.size() >= 4)
            {
                double v[4];
                for (int k = 0; k < 4; ++k) v[k] = doc.resolve(inh.mediabox->arr[static_cast<std::size_t>(k)]).num;
                page.x0 = std::min(v[0], v[2]);
                page.y0 = std::min(v[1], v[3]);
                page.x1 = std::max(v[0], v[2]);
                page.y1 = std::max(v[1], v[3]);
                if (page.x1 - page.x0 < 1 || page.y1 - page.y0 < 1)
                {
                    page.x0 = 0;
                    page.y0 = 0;
                    page.x1 = 612;
                    page.y1 = 792;
                }
            }
            page.rotate = ((inh.rotate % 360) + 360) % 360;
            page.rotate -= page.rotate % 90;
            const std::size_t number = out.pages.size() + 1;
            try
            {
                std::string content;
                if (const object* c = doc.resolve_key(*leaf.first, "Contents"))
                {
                    if (c->is(object::stream_k)) content = doc.stream_data(*c);
                    else if (c->is(object::array_k))
                        for (const object& part : c->arr)
                        {
                            content += doc.stream_data(doc.resolve(part));
                            content += "\n";
                        }
                }
                interpreter it(doc, fonts, page, out.undecodable_glyphs);
                it.run(content, inh.resources, pdf_matrix(), 0);
                read_annotations(doc, it, *leaf.first, page, default_resources);
                if (it.missing_font_runs() > 0)
                    out.notes.push_back("page " + std::to_string(number) + ": " + std::to_string(it.missing_font_runs()) +
                                        " text run(s) use a font missing from the resources");
                if (it.truncated_strings() > 0)
                    out.notes.push_back("page " + std::to_string(number) + ": " + std::to_string(it.truncated_strings()) +
                                        " shown string(s) longer than 1 MiB read in part");
            }
            catch (pdf_error& e)
            {
                // A file whose streams decode to more than the budget is refused as a whole.
                if (doc.over_budget()) throw;
                out.notes.push_back("page " + std::to_string(number) + ": " + e.info);
            }
            catch (std::exception& e)
            {
                out.notes.push_back("page " + std::to_string(number) + ": " + e.what());
            }
            if (doc.damaged_streams() > damaged)
            {
                out.notes.push_back("page " + std::to_string(number) + ": " +
                                    std::to_string(doc.damaged_streams() - damaged) +
                                    " damaged stream(s) read up to the damage");
                damaged = doc.damaged_streams();
            }
            if (doc.resolve_key(*leaf.first, "Thumb"))
                out.features.push_back("page " + std::to_string(number) + " thumbnail");
            out.pages.push_back(page);
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

    namespace pdf_impl
    {
        struct layout_item
        {
            const pdf_text_span* s;
            double u, v;
        };

        // Joins the spans of one baseline into segments, splitting at wide gaps.
        inline void join_line(std::vector<layout_item>& line, std::size_t line_id, std::vector<pdf_text_segment>& segs)
        {
            std::stable_sort(line.begin(), line.end(),
                             [](const layout_item& a, const layout_item& b) { return a.u < b.u; });
            pdf_text_segment cur;
            bool open = false;
            double u0 = 0, end_u = 0;
            const layout_item* last = nullptr;
            for (const layout_item& it : line)
            {
                const pdf_text_span& s = *it.s;
                const double fs = std::max(s.font_size, 1.0);
                // Text drawn twice at almost the same place: fake bold, shadows.
                const double dup_tol = std::min(0.3 * fs, 0.3 * std::max(s.width, 0.1));
                if (last && last->s->text == s.text && std::abs(last->u - it.u) < dup_tol &&
                    std::abs(last->v - it.v) < dup_tol)
                    continue;
                last = &it;
                const double gap = it.u - end_u;
                if (open && gap < 2.0 * fs && gap > -0.6 * fs)
                {
                    const bool spaced = (!cur.text.empty() && cur.text.back() == ' ') || s.text[0] == ' ';
                    if (gap > 0.1 * fs && !spaced) cur.text += ' ';
                    cur.text += s.text;
                    end_u = std::max(end_u, it.u + s.width);
                    cur.width = end_u - u0;
                    cur.font_size = std::max(cur.font_size, s.font_size);
                    continue;
                }
                if (open) segs.push_back(cur);
                cur = pdf_text_segment();
                cur.text = s.text;
                cur.x = s.x;
                cur.y = s.y;
                cur.width = s.width;
                cur.font_size = s.font_size;
                cur.angle = s.angle;
                cur.line = line_id;
                u0 = it.u;
                end_u = it.u + s.width;
                open = true;
            }
            if (open) segs.push_back(cur);
        }
    }

    inline std::vector<std::vector<pdf_text_segment>> layout_pdf_text(const pdf_text_document& doc)
    {
        using namespace pdf_impl;
        std::vector<std::vector<pdf_text_segment>> pages;
        for (const pdf_page_text& page : doc.pages)
        {
            std::vector<pdf_text_segment> segs;
            if (doc.produced_by_dlib)
            {
                // The runs written for one segment follow each other on the same baseline.
                for (const pdf_text_span& s : page.spans)
                {
                    if (!segs.empty())
                    {
                        pdf_text_segment& g = segs.back();
                        const double rad = g.angle * pi / 180.0;
                        const double ex = g.x + g.width * std::cos(rad), ey = g.y + g.width * std::sin(rad);
                        const double tol = 0.05 + 0.01 * g.font_size;
                        if (std::abs(s.angle - g.angle) < 0.01 && std::abs(s.font_size - g.font_size) < 0.01 &&
                            std::abs(s.x - ex) < tol && std::abs(s.y - ey) < tol)
                        {
                            g.text += s.text;
                            g.width += s.width;
                            continue;
                        }
                    }
                    pdf_text_segment g;
                    g.text = s.text;
                    g.x = s.x;
                    g.y = s.y;
                    g.width = s.width;
                    g.font_size = s.font_size;
                    g.angle = s.angle;
                    segs.push_back(g);
                }
                // Segments written for one line share its baseline.
                std::size_t line = 0;
                for (std::size_t i = 0; i < segs.size(); ++i)
                {
                    if (i > 0)
                    {
                        const pdf_text_segment& p = segs[i - 1];
                        const double rad = p.angle * pi / 180.0;
                        const double dv = -(segs[i].x - p.x) * std::sin(rad) + (segs[i].y - p.y) * std::cos(rad);
                        const double du = (segs[i].x - p.x) * std::cos(rad) + (segs[i].y - p.y) * std::sin(rad);
                        if (!(std::abs(segs[i].angle - p.angle) < 0.01 &&
                              std::abs(dv) <= 0.35 * std::max(1.0, p.font_size) && du > 0))
                            ++line;
                    }
                    segs[i].line = line;
                }
                pages.push_back(segs);
                continue;
            }

            // Spans grouped by baseline direction, to the degree; upright text first.
            std::map<int, std::vector<const pdf_text_span*>> groups;
            for (const pdf_text_span& s : page.spans)
            {
                if (s.text.empty()) continue;
                int key = static_cast<int>(std::lround(s.angle));
                key = ((key % 360) + 540) % 360 - 180;
                groups[key].push_back(&s);
            }
            std::vector<int> keys;
            if (groups.count(0)) keys.push_back(0);
            for (const auto& g : groups) if (g.first != 0) keys.push_back(g.first);

            std::size_t line_id = 0;
            for (int key : keys)
            {
                const double rad = key * pi / 180.0;
                const double c = std::cos(rad), sn = std::sin(rad);
                std::vector<layout_item> items;
                for (const pdf_text_span* s : groups[key])
                    items.push_back(layout_item{ s, s->x * c + s->y * sn, -s->x * sn + s->y * c });
                std::stable_sort(items.begin(), items.end(),
                                 [](const layout_item& a, const layout_item& b) { return a.v > b.v; });
                std::vector<layout_item> line;
                double line_v = 0, line_fs = 0;
                for (const layout_item& it : items)
                {
                    const double fs = std::max(it.s->font_size, 1.0);
                    if (!line.empty() && std::abs(line_v - it.v) <= std::max(1.0, 0.5 * std::max(fs, line_fs)))
                    {
                        line.push_back(it);
                        line_fs = std::max(line_fs, fs);
                        continue;
                    }
                    if (!line.empty()) join_line(line, line_id++, segs);
                    line.assign(1, it);
                    line_v = it.v;
                    line_fs = fs;
                }
                if (!line.empty()) join_line(line, line_id++, segs);
            }
            pages.push_back(segs);
        }
        return pages;
    }

// ----------------------------------------------------------------------------------------

    namespace pdf_impl
    {
        inline int unicode_to_winansi(std::uint32_t cp)
        {
            if ((cp >= 0x20 && cp < 0x7F) || (cp >= 0xA0 && cp <= 0xFF)) return static_cast<int>(cp);
            static const std::uint16_t high[32] = { 0x20AC, 0,      0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
                                                    0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0,      0x017D, 0,
                                                    0,      0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
                                                    0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0,      0x017E, 0x0178 };
            for (int i = 0; i < 32; ++i)
                if (high[i] != 0 && high[i] == cp) return 0x80 + i;
            return -1;
        }

        /* ASCII stand-ins for frequent characters outside the fonts' repertoire. An empty
           string drops the character (zero width marks). */
        inline const char* writer_substitute(std::uint32_t cp)
        {
            switch (cp)
            {
                case '\t': case 0x2000: case 0x2001: case 0x2002: case 0x2003: case 0x2004: case 0x2005: case 0x2006:
                case 0x2007: case 0x2008: case 0x2009: case 0x200A: case 0x202F: case 0x205F: case 0x3000: return " ";
                case 0x200B: case 0x200C: case 0x200D: case 0x2060: case 0xFEFF: case 0x200E: case 0x200F: return "";
                case 0x2010: case 0x2011: case 0x2012: case 0x2212: case 0x2043: return "-";
                case 0x2015: return "\x97";
                case 0xFB00: return "ff"; case 0xFB01: return "fi"; case 0xFB02: return "fl";
                case 0xFB03: return "ffi"; case 0xFB04: return "ffl"; case 0xFB05: case 0xFB06: return "st";
                case 0x2190: return "<-"; case 0x2192: return "->"; case 0x2194: return "<->"; case 0x21D2: return "=>";
                case 0x21D0: return "<="; case 0x2264: return "<="; case 0x2265: return ">="; case 0x2260: return "!=";
                case 0x2248: return "~"; case 0x2032: return "'"; case 0x2033: return "\"";
                case 0x2044: case 0x2215: return "/"; case 0x2219: case 0x22C5: case 0x2027: return "\xB7";
                case 0x2116: return "No"; case 0x2610: return "[ ]"; case 0x2611: case 0x2612: return "[x]";
                case 0x2713: case 0x2714: case 0x2705: return "v"; case 0x2717: case 0x2718: return "x";
                case 0x25CF: case 0x25AA: case 0x25A0: case 0x25E6: case 0x2023: case 0x25CB: case 0x25A1:
                case 0xF0B7: case 0xF0A7: case 0xF0D8: case 0xF076: case 0xF0FC: case 0xF0A8: return "\x95";
                default: return nullptr;
            }
        }

        // Three decimals at most, written without the C library, whose output depends on the locale.
        inline std::string format_number(double v)
        {
            if (!std::isfinite(v)) v = 0;
            v = std::max(-1e12, std::min(1e12, v));
            long long m = std::llround(v * 1000);
            std::string s;
            if (m < 0)
            {
                s.push_back('-');
                m = -m;
            }
            s += std::to_string(m / 1000);
            const int f = static_cast<int>(m % 1000);
            if (f != 0)
            {
                std::string frac;
                frac.push_back(static_cast<char>('0' + f / 100));
                frac.push_back(static_cast<char>('0' + f / 10 % 10));
                frac.push_back(static_cast<char>('0' + f % 10));
                while (frac.back() == '0') frac.pop_back();
                s += "." + frac;
            }
            if (s == "-0") s = "0";
            return s;
        }

        inline std::string pdf_literal(const std::string& bytes)
        {
            std::string escaped;
            for (unsigned char c : bytes)
            {
                if (c == '(' || c == ')' || c == '\\')
                {
                    escaped.push_back('\\');
                    escaped.push_back(static_cast<char>(c));
                }
                else if (c < 0x20 || c >= 0x7F)
                {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\%03o", c);
                    escaped += buf;
                }
                else escaped.push_back(static_cast<char>(c));
            }
            return escaped;
        }

        struct text_run
        {
            int font;               // 1 WinAnsi, 2 Latin Extended-A
            std::string bytes;
        };
    }

    inline std::string write_pdf_text(
        const pdf_text_document& layout_source,
        const std::vector<std::vector<pdf_text_segment>>& pages,
        std::size_t* unencodable = nullptr
    )
    {
        using namespace pdf_impl;
        if (pages.size() != layout_source.pages.size())
            throw pdf_error("write_pdf_text(): page count mismatch");

        // Characters of the second font, in order of first use.
        std::map<std::uint32_t, int> extra_codes;
        std::vector<std::uint32_t> extra_chars;
        for (const auto& segs : pages)
        {
            for (const pdf_text_segment& g : segs)
            {
                for (std::size_t i = 0; i < g.text.size();)
                {
                    const std::uint32_t cp = utf8_get(g.text, i);
                    if (unicode_to_winansi(cp) < 0 && !writer_substitute(cp) && latin_extended_a_name(cp) &&
                        !extra_codes.count(cp) && extra_chars.size() < 223)
                    {
                        extra_codes[cp] = static_cast<int>(33 + extra_chars.size());
                        extra_chars.push_back(cp);
                    }
                }
            }
        }

        std::size_t bad = 0;
        std::vector<std::string> objects;   // object i + 1
        objects.push_back(std::string());   // 1 catalog
        objects.push_back(std::string());   // 2 page tree
        objects.push_back("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
        if (!extra_chars.empty())
        {
            std::string diff = "33";
            for (std::uint32_t cp : extra_chars) diff += std::string(" /") + latin_extended_a_name(cp);
            std::string cmap =
                "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
                "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
                "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
                "1 begincodespacerange\n<00> <FF>\nendcodespacerange\n";
            for (std::size_t i = 0; i < extra_chars.size(); i += 100)
            {
                const std::size_t n = std::min<std::size_t>(100, extra_chars.size() - i);
                cmap += std::to_string(n) + " beginbfchar\n";
                for (std::size_t k = i; k < i + n; ++k)
                {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "<%02X> <%04X>\n", static_cast<unsigned>(33 + k),
                                  static_cast<unsigned>(extra_chars[k]));
                    cmap += buf;
                }
                cmap += "endbfchar\n";
            }
            cmap += "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n";
            objects.push_back(
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding << /Type /Encoding /Differences [" +
                diff + "] >> /ToUnicode 5 0 R >>");
            objects.push_back("<< /Length " + std::to_string(cmap.size()) + " >>\nstream\n" + cmap + "\nendstream");
        }
        const std::string font_resources = extra_chars.empty() ? "/F1 3 0 R" : "/F1 3 0 R /F2 4 0 R";

        std::vector<int> page_ids;
        for (std::size_t p = 0; p < pages.size(); ++p)
        {
            const pdf_page_text& src = layout_source.pages[p];
            const std::vector<pdf_text_segment>& segs = pages[p];
            std::string content;
            for (const pdf_text_segment& g : segs)
            {
                std::vector<text_run> runs;
                double width = 0;
                auto put = [&](int f, unsigned char b, double w) {
                    if (runs.empty() || runs.back().font != f) runs.push_back(text_run{ f, std::string() });
                    runs.back().bytes.push_back(static_cast<char>(b));
                    width += w;
                };
                for (std::size_t i = 0; i < g.text.size();)
                {
                    const std::uint32_t cp = utf8_get(g.text, i);
                    if (cp == '\r' || cp == '\n') { put(1, ' ', 278); continue; }
                    const int b = unicode_to_winansi(cp);
                    if (b >= 0) { put(1, static_cast<unsigned char>(b), helvetica_width(cp)); continue; }
                    if (const char* sub = writer_substitute(cp))
                    {
                        for (const char* q = sub; *q; ++q)
                        {
                            const unsigned char sb = static_cast<unsigned char>(*q);
                            put(1, sb, helvetica_width(winansi_to_unicode(sb)));
                        }
                        continue;
                    }
                    const auto e = extra_codes.find(cp);
                    if (e != extra_codes.end())
                    {
                        put(2, static_cast<unsigned char>(e->second), helvetica_width(cp));
                        continue;
                    }
                    put(1, '?', helvetica_width('?'));
                    ++bad;
                }
                if (runs.empty()) continue;
                const double fs = g.font_size > 0.5 ? std::min(g.font_size, 500.0) : 10;
                width = width * fs / 1000.0;

                /* Room along the baseline up to the next segment of the line, or to the
                   page edge for axis-aligned text. */
                const double rad = g.angle * pi / 180.0;
                const double c = std::cos(rad), sn = std::sin(rad);
                double room = 1e9;
                const int quadrant = static_cast<int>(std::lround(g.angle / 90.0));
                if (std::abs(g.angle - 90.0 * quadrant) < 0.5)
                {
                    switch (((quadrant % 4) + 4) % 4)
                    {
                        case 0: room = src.x1 - g.x - 2; break;
                        case 1: room = src.y1 - g.y - 2; break;
                        case 2: room = g.x - src.x0 - 2; break;
                        default: room = g.y - src.y0 - 2; break;
                    }
                }
                for (const pdf_text_segment& h : segs)
                {
                    if (&h == &g || h.line != g.line) continue;
                    const double du = (h.x - g.x) * c + (h.y - g.y) * sn;
                    if (du > 0.5) room = std::min(room, du - 0.25 * fs);
                }
                double scale = 100;
                if (width > room && room > 0)
                    scale = std::max(35.0, 100.0 * room / width);

                content += "BT " + format_number(scale) + " Tz " + format_number(c) + " " + format_number(sn) + " " +
                    format_number(-sn) + " " + format_number(c) + " " + format_number(g.x) + " " + format_number(g.y) +
                    " Tm";
                for (const text_run& r : runs)
                    content += " /F" + std::to_string(r.font) + " " + format_number(fs) + " Tf (" +
                        pdf_literal(r.bytes) + ") Tj";
                content += " ET\n";
            }
            const std::string packed = zlib_compress(content);
            objects.push_back("<< /Length " + std::to_string(packed.size()) + " /Filter /FlateDecode >>\nstream\n" +
                              packed + "\nendstream");
            const int content_id = static_cast<int>(objects.size());
            objects.push_back("<< /Type /Page /Parent 2 0 R /MediaBox [" + format_number(src.x0) + " " +
                              format_number(src.y0) + " " + format_number(src.x1) + " " + format_number(src.y1) + "]" +
                              (src.rotate ? " /Rotate " + std::to_string(src.rotate) : std::string()) +
                              " /Resources << /Font << " + font_resources + " >> >> /Contents " +
                              std::to_string(content_id) + " 0 R >>");
            page_ids.push_back(static_cast<int>(objects.size()));
        }
        objects[0] = "<< /Type /Catalog /Pages 2 0 R /DlibPrivacy true >>";
        std::string kids;
        for (int id : page_ids) kids += std::to_string(id) + " 0 R ";
        objects[1] = "<< /Type /Pages /Kids [" + kids + "] /Count " + std::to_string(page_ids.size()) + " >>";

        std::string out = "%PDF-1.7\n%\xE2\xE3\xCF\xD3\n";
        std::vector<std::size_t> offsets;
        for (std::size_t i = 0; i < objects.size(); ++i)
        {
            offsets.push_back(out.size());
            out += std::to_string(i + 1) + " 0 obj\n" + objects[i] + "\nendobj\n";
        }
        const std::size_t xref = out.size();
        out += "xref\n0 " + std::to_string(objects.size() + 1) + "\n0000000000 65535 f \n";
        for (std::size_t off : offsets)
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%010lu 00000 n \n", static_cast<unsigned long>(off));
            out += buf;
        }
        out += "trailer\n<< /Size " + std::to_string(objects.size() + 1) + " /Root 1 0 R >>\nstartxref\n" +
               std::to_string(xref) + "\n%%EOF\n";
        if (unencodable) *unencodable = bad;
        return out;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PDF_TEXT_H_
