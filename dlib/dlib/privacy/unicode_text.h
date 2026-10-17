// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PRIVACY_UNICODE_TEXT_H_
#define DLIB_PRIVACY_UNICODE_TEXT_H_

/*
    Code point helpers shared by the detection and pseudonymization components.

    The detector reasons on letters, case and word boundaries of European scripts, so
    this header carries exactly that knowledge: UTF-8 decoding with byte offsets,
    letter and case tests, simple one-to-one case mapping, and accent stripping for
    lexicon lookups. Case mapping covers ASCII, Latin-1, Latin Extended-A and
    Additional, Greek and Cyrillic; other scripts are treated as letters without case,
    which is what the pseudonymizer needs to fall back to exact, case-sensitive coding.
*/

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dlib
{
    namespace privacy_impl
    {

    // ------------------------------------------------------------------------------------

        inline char32_t utf8_next(const std::string& s, std::size_t& i)
        {
            const unsigned char c0 = static_cast<unsigned char>(s[i]);
            if (c0 < 0x80)
            {
                ++i;
                return c0;
            }
            int len = 0;
            char32_t cp = 0;
            if ((c0 & 0xE0) == 0xC0) { len = 2; cp = c0 & 0x1F; }
            else if ((c0 & 0xF0) == 0xE0) { len = 3; cp = c0 & 0x0F; }
            else if ((c0 & 0xF8) == 0xF0) { len = 4; cp = c0 & 0x07; }
            else { ++i; return 0xFFFD; }
            if (i + static_cast<std::size_t>(len) > s.size()) { ++i; return 0xFFFD; }
            for (int k = 1; k < len; ++k)
            {
                const unsigned char ck = static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]);
                if ((ck & 0xC0) != 0x80) { ++i; return 0xFFFD; }
                cp = (cp << 6) | (ck & 0x3F);
            }
            const char32_t min_cp[5] = { 0, 0, 0x80, 0x800, 0x10000 };
            if (cp < min_cp[len] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) { ++i; return 0xFFFD; }
            i += static_cast<std::size_t>(len);
            return cp;
        }

        inline void utf8_append(std::string& out, char32_t cp)
        {
            if (cp < 0x80)
            {
                out.push_back(static_cast<char>(cp));
            }
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

        inline bool utf8_valid(const std::string& s)
        {
            std::size_t i = 0;
            while (i < s.size())
            {
                const std::size_t before = i;
                const char32_t cp = utf8_next(s, i);
                if (cp == 0xFFFD && !(i - before == 3 &&
                    static_cast<unsigned char>(s[before]) == 0xEF &&
                    static_cast<unsigned char>(s[before + 1]) == 0xBF &&
                    static_cast<unsigned char>(s[before + 2]) == 0xBD))
                    return false;
            }
            return true;
        }

        // Zero width space and joiners, word joiner, byte order mark and soft hyphen.
        inline bool is_invisible_mark(char32_t c)
        {
            return c == 0x200B || c == 0x200C || c == 0x200D || c == 0x2060 || c == 0xFEFF || c == 0xAD;
        }

        struct utf8_text
        {
            /*
                The code points of a UTF-8 string together with the byte offset of each,
                plus one final offset equal to the byte length. Detectors work on code
                points and report byte spans through this table.
            */
            std::vector<char32_t> cps;
            std::vector<std::size_t> offsets;

            explicit utf8_text(const std::string& s, bool for_detection = false)
            {
                /*
                    With for_detection, the typographic spaces (no-break, narrow no-break,
                    thin, figure...) read as a space, the Unicode hyphens as '-', and the
                    invisible marks (zero width space and joiners, soft hyphen, BOM) are
                    skipped, so that "06 12 34 56 78" written with no-break spaces or a name
                    with a hidden soft hyphen are still recognized. Offsets keep pointing
                    into s, so reported spans cover the original bytes.
                */
                cps.reserve(s.size());
                offsets.reserve(s.size() + 1);
                std::size_t i = 0;
                while (i < s.size())
                {
                    const std::size_t at = i;
                    char32_t c = utf8_next(s, i);
                    if (for_detection)
                    {
                        if (is_invisible_mark(c))
                            continue;
                        if (c == 0xA0 || (c >= 0x2000 && c <= 0x200A) || c == 0x202F || c == 0x205F || c == 0x3000)
                            c = ' ';
                        else if (c == 0x2010 || c == 0x2011 || c == 0x2012)
                            c = '-';
                    }
                    offsets.push_back(at);
                    cps.push_back(c);
                }
                offsets.push_back(s.size());
            }

            std::size_t size() const { return cps.size(); }
            char32_t at(std::size_t i) const { return i < cps.size() ? cps[i] : 0; }
            char32_t before(std::size_t i) const { return i > 0 && i <= cps.size() ? cps[i - 1] : 0; }
        };

    // ------------------------------------------------------------------------------------

        inline bool is_ascii_digit(char32_t c) { return c >= '0' && c <= '9'; }

        inline bool is_space(char32_t c)
        {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' ||
                   c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
                   c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
        }

        inline bool is_line_break(char32_t c)
        {
            return c == '\n' || c == '\r' || c == 0x2028 || c == 0x2029 || c == '\f' || c == '\v';
        }

        inline bool is_apostrophe(char32_t c)
        {
            return c == '\'' || c == 0x2019 || c == 0x2BC;
        }

        inline bool is_hyphen(char32_t c)
        {
            return c == '-' || c == 0x2010 || c == 0x2011;
        }

        inline bool is_letter(char32_t c)
        {
            if (c < 0x80)
                return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
            if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;
            if (c >= 0xC0 && c <= 0x24F) return c != 0xD7 && c != 0xF7;
            if (c >= 0x250 && c <= 0x2AF) return true;
            if (c >= 0x300 && c <= 0x36F) return true;
            if (c >= 0x370 && c <= 0x3FF) return c != 0x37E && c != 0x387;
            if (c >= 0x400 && c <= 0x52F) return true;
            if (c >= 0x531 && c <= 0x58F) return true;
            if (c >= 0x5D0 && c <= 0x5EA) return true;
            if (c >= 0x620 && c <= 0x64A) return true;
            if (c >= 0x671 && c <= 0x6D3) return true;
            if (c >= 0x900 && c <= 0x97F) return true;
            if (c >= 0xE00 && c <= 0xE7F) return true;
            if (c >= 0x1100 && c <= 0x11FF) return true;
            if (c >= 0x1E00 && c <= 0x1EFF) return true;
            if (c >= 0x3040 && c <= 0x30FF) return true;
            if (c >= 0x3400 && c <= 0x4DBF) return true;
            if (c >= 0x4E00 && c <= 0x9FFF) return true;
            if (c >= 0xAC00 && c <= 0xD7AF) return true;
            if (c >= 0xF900 && c <= 0xFAFF) return true;
            if ((c >= 0xFF21 && c <= 0xFF3A) || (c >= 0xFF41 && c <= 0xFF5A)) return true;
            return false;
        }

        inline bool is_word_char(char32_t c)
        {
            return is_letter(c) || is_ascii_digit(c);
        }

        inline char32_t to_lower(char32_t c)
        {
            if (c < 0x80)
                return (c >= 'A' && c <= 'Z') ? c + 32 : c;
            if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 0x20;
            if (c >= 0x100 && c <= 0x17F)
            {
                if (c == 0x130) return 'i';
                if (c == 0x178) return 0xFF;
                if ((c >= 0x100 && c <= 0x12F) || (c >= 0x132 && c <= 0x137) || (c >= 0x14A && c <= 0x177))
                    return (c % 2 == 0) ? c + 1 : c;
                if ((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E))
                    return (c % 2 == 1) ? c + 1 : c;
                return c;
            }
            if (c >= 0x391 && c <= 0x3A9 && c != 0x3A2) return c + 0x20;
            switch (c)
            {
                case 0x386: return 0x3AC;
                case 0x388: return 0x3AD;
                case 0x389: return 0x3AE;
                case 0x38A: return 0x3AF;
                case 0x38C: return 0x3CC;
                case 0x38E: return 0x3CD;
                case 0x38F: return 0x3CE;
                default: break;
            }
            if (c >= 0x410 && c <= 0x42F) return c + 0x20;
            if (c >= 0x400 && c <= 0x40F) return c + 0x50;
            if ((c >= 0x460 && c <= 0x481) || (c >= 0x48A && c <= 0x4BF) || (c >= 0x4D0 && c <= 0x4FF))
                return (c % 2 == 0) ? c + 1 : c;
            if (c >= 0x4C1 && c <= 0x4CE)
                return (c % 2 == 1) ? c + 1 : c;
            if ((c >= 0x1E00 && c <= 0x1E95) || (c >= 0x1EA0 && c <= 0x1EFF))
                return (c % 2 == 0) ? c + 1 : c;
            return c;
        }

        inline char32_t to_upper(char32_t c)
        {
            if (c < 0x80)
                return (c >= 'a' && c <= 'z') ? c - 32 : c;
            if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 0x20;
            if (c == 0xFF) return 0x178;
            if (c >= 0x100 && c <= 0x17F)
            {
                if (c == 0x131) return 'I';
                if ((c >= 0x100 && c <= 0x12F) || (c >= 0x132 && c <= 0x137) || (c >= 0x14A && c <= 0x177))
                    return (c % 2 == 1) ? c - 1 : c;
                if ((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E))
                    return (c % 2 == 0) ? c - 1 : c;
                return c;
            }
            if (c >= 0x3B1 && c <= 0x3C9 && c != 0x3C2) return c - 0x20;
            switch (c)
            {
                case 0x3AC: return 0x386;
                case 0x3AD: return 0x388;
                case 0x3AE: return 0x389;
                case 0x3AF: return 0x38A;
                case 0x3CC: return 0x38C;
                case 0x3CD: return 0x38E;
                case 0x3CE: return 0x38F;
                default: break;
            }
            if (c >= 0x430 && c <= 0x44F) return c - 0x20;
            if (c >= 0x450 && c <= 0x45F) return c - 0x50;
            if ((c >= 0x460 && c <= 0x481) || (c >= 0x48A && c <= 0x4BF) || (c >= 0x4D0 && c <= 0x4FF))
                return (c % 2 == 1) ? c - 1 : c;
            if (c >= 0x4C1 && c <= 0x4CE)
                return (c % 2 == 0) ? c - 1 : c;
            if ((c >= 0x1E00 && c <= 0x1E95) || (c >= 0x1EA0 && c <= 0x1EFF))
                return (c % 2 == 1) ? c - 1 : c;
            return c;
        }

        inline bool is_upper(char32_t c) { return to_lower(c) != c; }

        inline bool is_lower(char32_t c)
        {
            return to_upper(c) != c || c == 0xDF || c == 0x138 || c == 0x149 || c == 0x17F || c == 0x3C2;
        }

        inline char32_t strip_accent(char32_t c)
        {
            static const char latin1[] =
                "AAAAAAACEEEEIIII" "DNOOOOO*OUUUUYTs" "aaaaaaaceeeeiiii" "dnooooo*ouuuuyty";
            static const char latin_ext_a[] =
                "AaAaAaCc" "CcCcCcDd" "DdEeEeEe" "EeEeGgGg" "GgGgHhHh" "IiIiIiIi" "IiIiJjKk" "kLlLlLlL"
                "lLlNnNnN" "nnNnOoOo" "OoOoRrRr" "RrSsSsSs" "SsTtTtTt" "UuUuUuUu" "UuUuWwYy" "YZzZzZzs";
            if (c >= 0xC0 && c <= 0xFF)
            {
                const char b = latin1[c - 0xC0];
                return b == '*' ? c : static_cast<char32_t>(b);
            }
            if (c >= 0x100 && c <= 0x17F)
                return static_cast<char32_t>(latin_ext_a[c - 0x100]);
            return c;
        }

        inline std::string lowercase(const std::string& s)
        {
            std::string out;
            out.reserve(s.size());
            std::size_t i = 0;
            while (i < s.size())
                utf8_append(out, to_lower(utf8_next(s, i)));
            return out;
        }

        inline std::string uppercase(const std::string& s)
        {
            std::string out;
            out.reserve(s.size());
            std::size_t i = 0;
            while (i < s.size())
                utf8_append(out, to_upper(utf8_next(s, i)));
            return out;
        }

        inline std::string titlecase(const std::string& s)
        {
            std::string out;
            out.reserve(s.size());
            std::size_t i = 0;
            bool first = true;
            while (i < s.size())
            {
                const char32_t c = utf8_next(s, i);
                if (first && is_letter(c))
                {
                    utf8_append(out, to_upper(c));
                    first = false;
                }
                else
                {
                    utf8_append(out, to_lower(c));
                }
            }
            return out;
        }

        inline std::string fold_key(const std::string& s)
        {
            /*
                Lookup key for lexicons: lower case, Latin accents removed, typographic
                apostrophes unified, invisible marks dropped and runs of spaces reduced to
                one.
            */
            std::string out;
            out.reserve(s.size());
            std::size_t i = 0;
            bool last_space = false;
            while (i < s.size())
            {
                char32_t c = utf8_next(s, i);
                if (is_invisible_mark(c)) continue;
                if (is_space(c))
                {
                    if (!last_space && !out.empty()) out.push_back(' ');
                    last_space = true;
                    continue;
                }
                last_space = false;
                if (is_apostrophe(c)) c = '\'';
                if (is_hyphen(c)) c = '-';
                if (c == 0x152 || c == 0x153) { out += "oe"; continue; }
                if (c == 0xC6 || c == 0xE6) { out += "ae"; continue; }
                utf8_append(out, strip_accent(to_lower(c)));
            }
            while (!out.empty() && out.back() == ' ')
                out.pop_back();
            return out;
        }

    // ------------------------------------------------------------------------------------

    }
}

#endif // DLIB_PRIVACY_UNICODE_TEXT_H_
