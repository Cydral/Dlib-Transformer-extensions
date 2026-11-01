// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// The voices and the text of the speech synthesis, shared by the service and albireo-speech.

#include "speech.h"

#include <algorithm>

namespace albireo
{

    namespace
    {
        std::string lower(std::string s)
        {
            for (char& c : s)
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            return s;
        }

        // The next code point of UTF-8 text at i, advancing i; -1 for an invalid sequence.
        long next_code_point(const std::string& s, size_t& i)
        {
            const unsigned char c = static_cast<unsigned char>(s[i]);
            size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
            if (n == 0 || i + n > s.size())
            {
                ++i;
                return -1;
            }
            long cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
            for (size_t k = 1; k < n; ++k)
            {
                const unsigned char d = static_cast<unsigned char>(s[i + k]);
                if ((d >> 6) != 0x2)
                {
                    ++i;
                    return -1;
                }
                cp = (cp << 6) | (d & 0x3F);
            }
            // Overlong forms and surrogates are refused as the decoders of the browsers refuse them.
            const long least[] = { 0, 0, 0x80, 0x800, 0x10000 };
            if (cp < least[n] || (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF)
            {
                i += n;
                return -1;
            }
            i += n;
            return cp;
        }

        void append_utf8(std::string& out, long cp)
        {
            if (cp < 0x80) out += static_cast<char>(cp);
            else if (cp < 0x800)
            {
                out += static_cast<char>(0xC0 | (cp >> 6));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            }
            else
            {
                out += static_cast<char>(0xE0 | (cp >> 12));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }

        // A letter of the Latin scripts the voices read: Latin-1 and the Latin Extended blocks.
        bool latin_letter(long cp)
        {
            if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
            if (cp >= 0xC0 && cp <= 0x24F) return cp != 0xD7 && cp != 0xF7;
            return false;
        }

        // What a character becomes in the text a voice reads: itself, a replacement, or a space.
        std::string spoken(long cp)
        {
            if (latin_letter(cp) || (cp >= '0' && cp <= '9')) return std::string();
            switch (cp)
            {
            case '.': case ',': case ';': case ':': case '!': case '?': case '\'': case '"': case '-': case '(':
            case ')': case '/': case '%': case '+': case '=': case '$': case '\n': case 0xA3: case 0xB0: case 0x20AC:
            case 0xBF: case 0xA1:
                return std::string();
            case 0x2018: case 0x2019: case 0x02BC: return "'";
            case 0x201C: case 0x201D: case 0xAB: case 0xBB: return "\"";
            case 0x2013: case 0x2014: return ", ";
            case 0x2026: return "...";
            default: return " ";
            }
        }
    }

    const std::vector<speech_voice>& speech_voices()
    {
        static const std::vector<speech_voice> voices = {
            { "en-US", "en", "en-US_ta.bin", "en-US_lh0_sg.bin" },
            { "en-GB", "en", "en-GB_ta.bin", "en-GB_kh0_sg.bin" },
            { "fr-FR", "fr", "fr-FR_ta.bin", "fr-FR_nk0_sg.bin" },
            { "de-DE", "de", "de-DE_ta.bin", "de-DE_gl0_sg.bin" },
            { "it-IT", "it", "it-IT_ta.bin", "it-IT_cm0_sg.bin" },
            { "es-ES", "es", "es-ES_ta.bin", "es-ES_zl0_sg.bin" },
        };
        return voices;
    }

    const speech_voice* find_voice(const std::string& name)
    {
        const std::string n = lower(name);
        for (const auto& v : speech_voices())
            if (lower(v.id) == n) return &v;
        for (const auto& v : speech_voices())
            if (v.language == n) return &v;
        return nullptr;
    }

    std::string speech_text(const std::string& text, size_t max_bytes)
    {
        std::string out;
        out.reserve(std::min(text.size(), max_bytes) + 4);
        size_t word = 0;
        bool space = true;
        for (size_t i = 0; i < text.size();)
        {
            const long cp = next_code_point(text, i);
            std::string piece;
            if (cp < 0) piece = " ";
            else
            {
                piece = spoken(cp);
                if (piece.empty()) append_utf8(piece, cp);
            }
            for (size_t k = 0; k < piece.size();)
            {
                const long c = next_code_point(piece, k);
                if (c == ' ' || c == '\n')
                {
                    // A line break stays, the spaces before it go; runs of spaces become one.
                    if (c == '\n')
                    {
                        while (!out.empty() && out.back() == ' ') out.pop_back();
                        if (!out.empty() && out.back() != '\n') out += '\n';
                    }
                    else if (!space) out += ' ';
                    space = true;
                    word = 0;
                    continue;
                }
                const bool letter = latin_letter(c) || (c >= '0' && c <= '9');
                if (letter && ++word > 64)
                {
                    out += ' ';
                    word = 1;
                }
                if (!letter) word = 0;
                std::string bytes;
                append_utf8(bytes, c);
                if (out.size() + bytes.size() > max_bytes)
                {
                    while (!out.empty() && (out.back() == ' ' || out.back() == '\n')) out.pop_back();
                    return out;
                }
                out += bytes;
                space = false;
            }
        }
        while (!out.empty() && (out.back() == ' ' || out.back() == '\n')) out.pop_back();
        return out;
    }

}
