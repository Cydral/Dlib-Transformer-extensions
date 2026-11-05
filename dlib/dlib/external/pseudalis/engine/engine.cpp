// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "engine.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <dlib/base64.h>
#include <dlib/privacy.h>
#include <dlib/privacy/pii_recognizers.h>

#include "../common/file_kinds.h"
#include "archives.h"

/*
    The models compiled into the engine: the header the embed command of the example
    slm_document_anonymizer_ex writes (examples/slm_document_anonymizer_ex.h), whose path
    the build gives.
*/
#ifndef PSEUDALIS_MODELS_HEADER
#define PSEUDALIS_MODELS_HEADER "slm_document_anonymizer_ex.h"
#endif
#include PSEUDALIS_MODELS_HEADER

namespace pseudalis
{

    namespace
    {
        using namespace dlib;

        std::string lower(std::string s)
        {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        std::string base_name(const std::string& path)
        {
            const size_t slash = path.find_last_of("/\\");
            return slash == std::string::npos ? path : path.substr(slash + 1);
        }

        std::string sha256_hex(const std::string& data)
        {
            const std::array<unsigned char, 32> d = sha256_digest(data);
            return to_hex(d.data(), d.size());
        }

        std::string decode_model(const std::string& text)
        {
            base64 coder;
            std::istringstream in(text);
            std::ostringstream packed;
            coder.decode(in, packed);
            return zlib_decompress(packed.str());
        }

        struct model_set
        {
            std::shared_ptr<const pii_sequence_tagger> tagger;
            std::shared_ptr<const pii_span_recognizer> span;
        };

        // The models compiled into the engine, read once for every engine of the process.
        const model_set& shared_models()
        {
            static const model_set m = []() {
                model_set r;
                const std::string tagger_text = anonymizer_models::tagger();
                if (!tagger_text.empty())
                {
                    auto t = std::make_shared<pii_sequence_tagger>();
                    std::istringstream in(decode_model(tagger_text));
                    deserialize(*t, in);
                    r.tagger = t;
                }
                const std::string span_text = anonymizer_models::span_model();
                if (!span_text.empty())
                {
                    std::istringstream in(decode_model(span_text));
                    r.span = std::make_shared<pii_span_recognizer>(in);
                }
                return r;
            }();
            return m;
        }

        /*
            A name and its extension, ".docx" or ".tar.gz", which keeps its letters: the extension
            of a kind Pseudalis processes, since the end of a folder or of an e-mail address
            ("jean.dupont@example.fr") is no extension.
        */
        void split_extension(const std::string& base, std::string& stem, std::string& ext)
        {
            ext = pseudalis::kind_of_name(base) == name_kind::unsupported ? std::string() : name_extension(base);
            if (ext.size() >= base.size()) ext.clear();
            ext = base.substr(base.size() - ext.size());
            stem = base.substr(0, base.size() - ext.size());
        }

        // A substitute with the characters Windows refuses in a name replaced by '_'.
        std::string windows_safe(std::string s)
        {
            for (char& c : s)
                if (static_cast<unsigned char>(c) < 0x20 || std::strchr("<>:\"/\\|?*", c) != nullptr) c = '_';
            return s;
        }

        // The number of UTF-16 units of a UTF-8 text, which bounds a name of Windows.
        std::size_t utf16_units(const std::string& s)
        {
            std::size_t n = 0;
            for (std::size_t i = 0; i < s.size(); ++i)
            {
                const unsigned char c = static_cast<unsigned char>(s[i]);
                if ((c & 0xC0) != 0x80) n += c >= 0xF0 ? 2 : 1;
            }
            return n;
        }

        /*
            A protected name that Windows takes: a stem that names a device (CON, NUL, COM1...)
            gets '_', and a name past 255 UTF-16 units loses the end of its stem, its extension
            kept.  The original name of a file, which the caller records, gives it back whole.
        */
        std::string windows_name(std::string stem, const std::string& ext)
        {
            std::string head = lower(stem.substr(0, stem.find('.')));
            while (!head.empty() && head.back() == ' ') head.pop_back();
            const bool device = head == "con" || head == "prn" || head == "aux" || head == "nul" ||
                (head.size() == 4 && (head.compare(0, 3, "com") == 0 || head.compare(0, 3, "lpt") == 0) &&
                 head[3] >= '1' && head[3] <= '9');
            if (device) stem.insert(stem.find('.') == std::string::npos ? stem.size() : stem.find('.'), "_");
            while (!stem.empty() && utf16_units(stem + ext) > 255)
            {
                stem.pop_back();
                while (!stem.empty() && (static_cast<unsigned char>(stem.back()) & 0xC0) == 0x80) stem.pop_back();
                if (!stem.empty() && static_cast<unsigned char>(stem.back()) >= 0xC0) stem.pop_back();
            }
            while (!stem.empty() && (stem.back() == ' ' || stem.back() == '.')) stem.pop_back();
            return stem.empty() ? "_" + ext : stem + ext;
        }

        json report_json(const document_report& r)
        {
            json j = json::object();
            json values = json::object();
            for (const auto& e : r.entities) values[e.first] = static_cast<unsigned long long>(e.second);
            j["values"] = values;
            j["total"] = static_cast<unsigned long long>(r.total_entities());
            json removed = json::array();
            for (const std::string& s : r.removed) removed.push_back(s);
            j["removed"] = removed;
            json notes = json::array();
            for (const std::string& s : r.notes) notes.push_back(s);
            j["notes"] = notes;
            j["tags_restored"] = static_cast<unsigned long long>(r.tags_restored);
            j["tags_rejected"] = static_cast<unsigned long long>(r.tags_rejected);
            j["dates_shifted"] = static_cast<unsigned long long>(r.dates_shifted);
            j["substitutes"] = static_cast<unsigned long long>(r.substitutions.size());
            j["substitutes_restored"] = static_cast<unsigned long long>(r.substitutes_restored);
            j["dropped"] = json::array();
            return j;
        }

        // Adds the counts and the lists of part, an entry of an archive, to whole.
        void merge_report(json& whole, const json& part, const std::string& entry)
        {
            for (const auto& kv : part["values"].items())
                whole["values"][kv.first] =
                    static_cast<unsigned long long>(whole["values"][kv.first].as_int(0) + kv.second.as_int(0));
            for (const char* key : { "total", "tags_restored", "tags_rejected", "dates_shifted", "substitutes",
                                     "substitutes_restored" })
                whole[key] = static_cast<unsigned long long>(whole[key].as_int(0) + part[key].as_int(0));
            for (const char* key : { "removed", "notes", "dropped" })
                for (const json& v : part[key].elements())
                    whole[key].push_back(entry.empty() ? v.str() : entry + ": " + v.str());
        }

        /*
            The name of an entry not taken yet among names: "a.png.txt", then "a.png (2).txt"...
            next remembers, for each name, the number to try first, so that an archive that
            repeats a name thousands of times is not walked again for each of them.
        */
        std::string free_name(const std::string& name, std::set<std::string>& names, std::map<std::string, int>& next)
        {
            std::string candidate = name;
            const size_t dot = name.rfind('.');
            int& k = next[lower(name)];
            if (k < 2) k = 2;
            for (; names.count(lower(candidate)); ++k)
                candidate = dot == std::string::npos
                    ? name + " (" + std::to_string(k) + ")"
                    : name.substr(0, dot) + " (" + std::to_string(k) + ")" + name.substr(dot);
            names.insert(lower(candidate));
            return candidate;
        }

        /*
            The bytes that the archives and the Office documents of one file may decompress, all
            levels together: 64 times the file, from 16 MB to 2 GB.  The memory and the time of
            the processing grow with them: a small file that holds hundreds of megabytes of text
            is refused rather than read for hours.
        */
        uint64_t decompression_budget(size_t file_size)
        {
            const uint64_t wanted = static_cast<uint64_t>(file_size) * 64;
            return std::min<uint64_t>(2048ull << 20, std::max<uint64_t>(16ull << 20, wanted));
        }

        // The family of a document from its content: a text, an Office document, a PDF, or none.
        document_family family_of(document_format f)
        {
            switch (f)
            {
                case document_format::docx:
                case document_format::xlsx:
                case document_format::pptx: return document_family::office;
                case document_format::pdf: return document_family::pdf;
                case document_format::text:
                case document_format::delimited:
                case document_format::markup: return document_family::text;
                default: return document_family::none;
            }
        }

        // The classes of the bytes for the encodings, which no locale changes.
        enum : unsigned char { base64_class = 1, base64url_class = 2, hex_class = 4, alnum_class = 8 };

        const unsigned char* char_classes()
        {
            struct table
            {
                unsigned char of[256];
                table()
                {
                    for (int c = 0; c < 256; ++c)
                    {
                        const bool digit = c >= '0' && c <= '9';
                        const bool alpha = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
                        const bool hex = digit || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
                        of[c] = static_cast<unsigned char>((digit || alpha || c == '+' || c == '/' || c == '=' ?
                                                            base64_class : 0) |
                                                           (c == '-' || c == '_' ? base64url_class : 0) |
                                                           (hex ? hex_class : 0) | (digit || alpha ? alnum_class : 0));
                    }
                }
            };
            static const table t;
            return t.of;
        }

        // A character of base64, which holds the hexadecimal digits too, or of base64url with url.
        bool base64_char(unsigned char c, bool url = true)
        {
            return (char_classes()[c] & (url ? base64_class | base64url_class : base64_class)) != 0;
        }

        bool alnum_char(unsigned char c)
        {
            return (char_classes()[c] & alnum_class) != 0;
        }

        unsigned char ascii_lower(unsigned char c)
        {
            return c >= 'A' && c <= 'Z' ? static_cast<unsigned char>(c + 32) : c;
        }

        // Whether text holds needle, written in lower case, at i, its case aside.
        bool at_without_case(const std::string& text, size_t i, const char* needle)
        {
            for (; *needle; ++needle, ++i)
                if (i >= text.size() || ascii_lower(static_cast<unsigned char>(text[i])) != *needle) return false;
            return true;
        }

        // The end of the line of text that starts at b: its first '\r' or '\n', or the end of the text.
        size_t line_end(const std::string& text, size_t b)
        {
            const char* const s = text.data();
            const size_t n = text.size();
            while (b < n && s[b] != '\n' && s[b] != '\r') ++b;
            return b;
        }

        // The start of the line after the end of a line at e, "\r\n", '\n' or '\r', or the end of the text.
        size_t after_end(const std::string& text, size_t e)
        {
            if (e >= text.size()) return text.size();
            return text[e] == '\r' && e + 1 < text.size() && text[e + 1] == '\n' ? e + 2 : e + 1;
        }

        size_t next_line(const std::string& text, size_t b)
        {
            return after_end(text, line_end(text, b));
        }

        /*
            A variety of characters within one run: a block of base64 holds tens of them, a line of
            dashes or of equal signs one.  begin() opens a run, add() counts a character.
        */
        class variety
        {
        public:
            void begin()
            {
                ++run_;
                count_ = 0;
            }
            size_t add(unsigned char c)
            {
                if (seen_[c] != run_)
                {
                    seen_[c] = run_;
                    ++count_;
                }
                return count_;
            }

        private:
            size_t seen_[256] = {};
            size_t run_ = 1;
            size_t count_ = 0;
        };

        /*
            Whether the line [b, e) gives a value of LDIF in base64: "cn:: SMOpbMOobmUgRHVwcsOp", the
            name of the attribute, two colons, spaces or none, then base64 to the end of the line,
            where a long value folds.
        */
        bool ldif_base64(const std::string& text, size_t b, size_t e)
        {
            size_t i = b;
            if (i >= e || !alnum_char(static_cast<unsigned char>(text[i])) || (text[i] >= '0' && text[i] <= '9'))
                return false;
            while (i < e && (alnum_char(static_cast<unsigned char>(text[i])) || text[i] == '-' || text[i] == ';')) ++i;
            if (e - i < 2 || text.compare(i, 2, "::") != 0) return false;
            i += 2;
            while (i < e && text[i] == ' ') ++i;
            const size_t value = i;
            while (i < e && base64_char(static_cast<unsigned char>(text[i]), false)) ++i;
            return i == e && e - value >= 4;
        }

        // Whether the line that starts at b is a line of uuencode: its length character, then its groups.
        bool uuencode_body(const std::string& text, size_t b)
        {
            const size_t e = line_end(text, b);
            if (b >= e) return false;
            const int count = static_cast<unsigned char>(text[b]) - 32;
            return count >= 1 && count <= 45 && e - b >= 1 + static_cast<size_t>((count + 2) / 3 * 4);
        }

        // The start of the content of the line [b, e), after its byte order marks, blanks, form feeds and '>'.
        size_t line_content(const std::string& text, size_t b, size_t e)
        {
            for (;;)
            {
                if (e - b >= 3 && text.compare(b, 3, "\xEF\xBB\xBF") == 0) b += 3;
                else if (b < e && (text[b] == ' ' || text[b] == '\t' || text[b] == '\f' || text[b] == '>')) ++b;
                else return b;
            }
        }

        /*
            Whether a run of 24 characters of base64 or more, mixing capitals, small letters and
            digits, opens within the 512 bytes of text from i: the body of a block armored, which
            follows its header.
        */
        bool base64_follows(const std::string& text, size_t i)
        {
            const unsigned char* const cls = char_classes();
            const size_t end = std::min(text.size(), i + 512);
            size_t run = 0;
            bool capital = false, small = false, digit = false;
            for (; i < end; ++i)
            {
                const unsigned char c = static_cast<unsigned char>(text[i]);
                if (!(cls[c] & base64_class))
                {
                    run = 0;
                    capital = small = digit = false;
                    continue;
                }
                capital = capital || (c >= 'A' && c <= 'Z');
                small = small || (c >= 'a' && c <= 'z');
                digit = digit || (c >= '0' && c <= '9');
                if (++run >= 24 && capital && small && digit) return true;
            }
            return false;
        }

        // Whether the line that starts at b, its content at c, opens a block of uuencode that a line of it follows.
        bool uuencode_begin(const std::string& text, size_t b, size_t c, size_t e)
        {
            if (e - c < 10 || text.compare(c, 6, "begin ") != 0) return false;
            size_t i = c + 6;
            while (i < e && text[i] >= '0' && text[i] <= '7') ++i;
            return (i - c == 9 || i - c == 10) && i + 1 < e && text[i] == ' ' &&
                   uuencode_body(text, next_line(text, b));
        }

        /*
            Whether text holds at i, in a line whose content starts at content and that ends at e,
            an opening that hides what follows, in a string of a program too: the header of a block
            armored by PEM or OpenPGP ("-----BEGIN CERTIFICATE-----", "-----BEGIN PGP MESSAGE, PART
            1/2-----"), which inside a line its body must follow, an RTF document ("{\rtf", or
            "{\\rtf" escaped), or a block of base64 for uudecode ("begin-base64 644 ").  A program
            that only names the header of a block, as a constant, is no block.
        */
        bool encoded_opening(const std::string& text, size_t i, size_t content, size_t e)
        {
            const char c = text[i];
            if (c == '{') return text.compare(i, 5, "{\\rtf") == 0 || text.compare(i, 6, "{\\\\rtf") == 0;
            if (c == '-')
            {
                if (e - i < 17 || text.compare(i, 11, "-----BEGIN ") != 0) return false;
                // The label: printable characters up to "-----".
                size_t j = i + 11;
                while (j < e && j - i < 100 && text.compare(j, 5, "-----") != 0 &&
                       static_cast<unsigned char>(text[j]) >= 0x20 && static_cast<unsigned char>(text[j]) < 0x7F)
                    ++j;
                if (j == i + 11 || e - j < 5 || text.compare(j, 5, "-----") != 0) return false;
                return i == content || base64_follows(text, j + 5);
            }
            if (c == 'b')
            {
                if (e - i < 17 || text.compare(i, 13, "begin-base64 ") != 0) return false;
                size_t j = i + 13;
                while (j < e && text[j] >= '0' && text[j] <= '7') ++j;
                return j - i >= 16 && j - i <= 17 && j < e && text[j] == ' ';
            }
            return false;
        }

        /*
            Whether the name of the header Content-Transfer-Encoding, which ends at i, is followed by
            a colon and a value that hides the text: base64, quoted-printable or uuencode.  The value
            may stand on the next line, the header folded.
        */
        bool transfer_encoding(const std::string& text, size_t i)
        {
            while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
            if (i >= text.size() || text[i] != ':') return false;
            ++i;
            while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\r' || text[i] == '\n')) ++i;
            return at_without_case(text, i, "base64") || at_without_case(text, i, "quoted-printable") ||
                   at_without_case(text, i, "x-uuencode") || at_without_case(text, i, "uuencode") ||
                   at_without_case(text, i, "x-uue");
        }

        /*
            Whether ';' at i, in a line that ends at e, gives the encoding of a value of a vCard or of
            an iCalendar: ";ENCODING=b:", ";ENCODING=QUOTED-PRINTABLE;", or the bare parameter of
            vCard 2.1, ";QUOTED-PRINTABLE:" or ";BASE64:".
        */
        bool value_encoding(const std::string& text, size_t i, size_t e)
        {
            const bool named = at_without_case(text, i + 1, "encoding=");
            const size_t v = named ? i + 10 : i + 1;
            for (const char* x : { "quoted-printable", "base64", "b" })
            {
                const size_t end = v + std::strlen(x);
                if (end >= e || !at_without_case(text, v, x)) continue;
                if (named ? text[end] == ':' || text[end] == ';' : text[end] == ':' && x[1] != '\0') return true;
            }
            return false;
        }

        // The end of the opening of an encoded word at i, "=?UTF-8?B?" or "=?utf-8*fr?q?", or npos.
        size_t encoded_word_opening(const std::string& text, size_t i, size_t e)
        {
            size_t j = i + 2;
            while (j < e && j - i - 2 <= 60 &&
                   (alnum_char(static_cast<unsigned char>(text[j])) || text[j] == '-' || text[j] == '_' ||
                    text[j] == '*'))
                ++j;
            if (j == i + 2 || e - j < 3 || text[j] != '?' || text[j + 2] != '?') return std::string::npos;
            const unsigned char x = ascii_lower(static_cast<unsigned char>(text[j + 1]));
            return x == 'b' || x == 'q' ? j + 3 : std::string::npos;
        }

        /*
            Whether a text is, or holds, a format that writes its values encoded: an RTF document,
            a part of a message in base64, quoted-printable or uuencode, an encoded word of a
            header ("=?UTF-8?B?...?="), a vCard or an iCalendar, whose lines fold and whose values
            may be encoded, the encoding of a value of a vCard elsewhere, LDIF with values in base64,
            a block of uuencode or of ASCII85, a block armored by PEM or OpenPGP ("-----BEGIN"), in a
            string of a program too, a program of PostScript or an export of the registry of
            Windows.  Its values would stay readable once decoded.  Each line is read once, its byte
            order marks, spaces and marks of quotation aside.
        */
        bool encoded_format(const std::string& text)
        {
            size_t start = 0;
            while (text.compare(start, 3, "\xEF\xBB\xBF") == 0) start += 3;
            while (start < text.size() && (text[start] == ' ' || (text[start] >= '\t' && text[start] <= '\r')))
                ++start;
            static const char* const openings[] = { "%!ps", "windows registry editor version", "regedit4" };
            for (const char* o : openings)
                if (at_without_case(text, start, o)) return true;
            bool ldif_dn = false, ldif_value = false;
            long ascii85 = -1;          // the characters of the block of ASCII85 that "<~" opened
            for (size_t b = 0, e = 0; b < text.size(); b = after_end(text, e))
            {
                e = line_end(text, b);
                const size_t c = line_content(text, b, e);
                // A vCard or an iCalendar, in a message too.
                if (at_without_case(text, c, "begin:vcard") || at_without_case(text, c, "begin:vcalendar")) return true;
                if (uuencode_begin(text, b, c, e)) return true;
                // LDIF: a value in base64 in a file of entries, which open with "dn:".
                ldif_dn = ldif_dn || at_without_case(text, c, "dn:");
                ldif_value = ldif_value || ldif_base64(text, c, e);
                if (ldif_dn && ldif_value) return true;
                size_t opened = std::string::npos;      // the end of the opening of an encoded word on this line
                bool base64_word = false;               // its encoding: base64, else quoted-printable
                for (size_t i = b; i < e; ++i)
                {
                    const char x = text[i];
                    if ((x == '-' || x == '{' || x == 'b') && encoded_opening(text, i, c, e)) return true;
                    // The header of a transfer encoding, in a quoted line too, its value maybe on a folded line.
                    if ((x == 'c' || x == 'C') && e - i >= 25 &&
                        at_without_case(text, i, "content-transfer-encoding") && transfer_encoding(text, i + 25))
                        return true;
                    if (x == ';' && value_encoding(text, i, e)) return true;
                    if (ascii85 >= 0)
                    {
                        if ((x >= '!' && x <= 'u') || x == 'z')
                        {
                            if (++ascii85 >= 64) return true;
                            continue;
                        }
                        ascii85 = -1;
                    }
                    if (x == '<' && i + 1 < e && text[i + 1] == '~')
                    {
                        ascii85 = 0;
                        ++i;
                    }
                    else if (x == '=' && i + 1 < e && text[i + 1] == '?' && opened == std::string::npos)
                    {
                        const size_t end = encoded_word_opening(text, i, e);
                        if (end != std::string::npos)
                        {
                            opened = end;
                            base64_word = ascii_lower(static_cast<unsigned char>(text[end - 2])) == 'b';
                            i = end - 1;
                        }
                    }
                    else if (opened != std::string::npos)
                    {
                        // The word ends with "?=" after two characters at least of its encoding.
                        if (x == '?' && i + 1 < e && text[i + 1] == '=' && i - opened >= 2) return true;
                        const bool valid = base64_word ? base64_char(static_cast<unsigned char>(x), false)
                                                       : x != ' ' && x != '\t' && x != '?';
                        if (!valid || i - opened > 1000) opened = std::string::npos;
                    }
                }
            }
            return false;
        }

        /*
            Whether a text holds bytes written in hexadecimal, a dump or an array: 64 in a row with 8
            letters or marks among them, or 4 lines in a row of 8 bytes or more, in 4 tokens at
            least each, with 4 letters or marks among them ("4d 5a 90 00", "0x41, 0x42",
            "43,00,3a,00", "fffe 4a00 6500" of xxd, "\x4a\x65" in a string).  A byte is two digits;
            in a line of a dump, which opens with its offset, a group of 4 or 8 between spaces
            counts its bytes; a mark, "0x" or "\x", or the offset of xxd, counts as a letter.  The
            fingerprint of a key, one line, and the lists of code points of Unicode pass; bytes
            written with digits alone, no letter or mark among them, pass as a table of numbers.
        */
        bool hex_bytes(const std::string& text)
        {
            const unsigned char* const cls = char_classes();
            const unsigned char* const d = reinterpret_cast<const unsigned char*>(text.data());
            auto hex = [&](size_t k) { return (cls[d[k]] & hex_class) != 0; };
            auto separator = [](unsigned char c) { return c == ' ' || c == ',' || c == ':' || c == '-' || c == '\t'; };
            size_t rows = 0, rows_marked = 0;       // the lines in a row of 8 bytes or more, their marked bytes
            for (size_t b = 0, e = 0; b < text.size(); b = after_end(text, e))
            {
                e = line_end(text, b);
                // Groups of 4 or 8 digits count in the lines of a dump alone, which open with an offset of 6 to 8.
                size_t k = b;
                while (k < e && (d[k] == ' ' || d[k] == '\t')) ++k;
                const size_t offset = k;
                while (k < e && hex(k)) ++k;
                const bool dump = k - offset >= 6 && k - offset <= 8 && k < e && (d[k] == ':' || d[k] == ' ');
                // The offset of xxd, "00000010:", marks its line as a letter would.
                const bool colon = dump && d[k] == ':';
                size_t best = 0, best_tokens = 0, best_marked = 0, bytes = 0, tokens = 0, marked = 0, i = b;
                // 8 bytes and their separators take 16 characters at least.
                while (e - b >= 16 && i < e)
                {
                    size_t from = i, to = i;
                    bool mark = false;
                    if (d[i] == '\\' && e - i >= 4 && (d[i + 1] == 'x' || d[i + 1] == 'X') && hex(i + 2) && hex(i + 3))
                    {
                        // An escaped byte, which needs no separator.
                        from = i + 2;
                        to = i + 4;
                        mark = true;
                    }
                    else if (hex(i) && (i == b || !(cls[d[i - 1]] & alnum_class)))
                    {
                        if (d[i] == '0' && e - i > 2 && (d[i + 1] == 'x' || d[i + 1] == 'X') && hex(i + 2))
                        {
                            from = i + 2;
                            mark = true;
                        }
                        to = from;
                        while (to < e && to - from < 9 && hex(to)) ++to;
                        const unsigned char after = to < e ? d[to] : ' ';
                        const bool spaced = after == ' ' || after == '\t';
                        if (to - from == 2 ? !separator(after)
                                           : mark || !dump || !spaced || (to - from != 4 && to - from != 8))
                            to = from;
                    }
                    if (to == from)
                    {
                        bytes = tokens = marked = 0;
                        ++i;
                        continue;
                    }
                    bool letter = mark;
                    for (size_t j = from; j < to; ++j) letter = letter || d[j] > '9';
                    bytes += (to - from) / 2;
                    ++tokens;
                    if (letter) ++marked;
                    if (bytes >= 64 && marked >= 8) return true;
                    if (bytes > best)
                    {
                        best = bytes;
                        best_tokens = tokens;
                        best_marked = marked;
                    }
                    i = to;
                    while (i < e && separator(d[i])) ++i;
                }
                if (best >= 8 && best_tokens >= 4)
                {
                    ++rows;
                    rows_marked += best_marked + (colon ? 1 : 0);
                }
                else
                    rows = rows_marked = 0;
                if (rows >= 4 && rows_marked >= 4) return true;
            }
            return false;
        }

        // The characters of [from, to) of d, the backslashes of escapes aside, as many different ones as kinds counts.
        size_t run_variety(const unsigned char* d, size_t from, size_t to, variety& kinds)
        {
            kinds.begin();
            size_t count = 0;
            for (size_t k = from; k < to; ++k)
                if (d[k] != '\\') count = kinds.add(d[k]);
            return count;
        }

        /*
            Whether a run of base64 of length characters at [from, to) of d may be a line of an
            encoded block: 60 characters or more, mixing capitals, small letters and digits, of 16
            different ones at least, or hexadecimal digits alone, of 10 at least.  A line of dashes,
            a word, a path without capitals or digits is none.
        */
        bool block_run(const unsigned char* d, size_t from, size_t to, size_t length, variety& kinds)
        {
            if (length < 60) return false;
            const unsigned char* const cls = char_classes();
            bool capital = false, small = false, digit = false, hex = true;
            for (size_t k = from; k < to; ++k)
            {
                const unsigned char c = d[k];
                if (c == '\\') continue;
                capital = capital || (c >= 'A' && c <= 'Z');
                small = small || (c >= 'a' && c <= 'z');
                digit = digit || (c >= '0' && c <= '9');
                hex = hex && (cls[c] & hex_class);
            }
            const size_t count = run_variety(d, from, to, kinds);
            return (capital && small && digit && count >= 16) || (hex && count >= 10);
        }

        /*
            Whether a text holds an encoded block: a run of base64 or base64url of 1024 characters
            of 16 different ones at least, or 16 runs in a row that may each be a line of a block
            (block_run()), whatever separates them, an end of line, a quote, or "\n" escaped in a
            string of JSON or of a program, where "\/" stands for a slash; data in base64 after
            ";base64,"; JSON in base64 ("eyJ...", the tokens of OpenID Connect among them); or bytes
            in hexadecimal (hex_bytes()).
        */
        bool encoded_block(const std::string& text)
        {
            const unsigned char* const cls = char_classes();
            const unsigned char* const d = reinterpret_cast<const unsigned char*>(text.data());
            const size_t n = text.size();
            variety kinds;
            size_t lines = 0;               // the runs in a row that may be lines of a block
            size_t start = 0, run = 0;      // the run read: its first byte, its characters
            for (size_t i = 0; i < n; ++i)
            {
                const unsigned char c = d[i];
                const bool slash = c == '\\' && i + 1 < n && d[i + 1] == '/';
                if (slash || (cls[c] & (base64_class | base64url_class)))
                {
                    if (run++ == 0) start = i;
                    if (slash) ++i;
                    if (run >= 1024 && (run == 1024 ? run_variety(d, start, i + 1, kinds) : kinds.add(d[i])) >= 16)
                        return true;
                    // JSON in base64 opens with "eyJ", the encoding of '{"', after "=" or a space for instance.
                    if (c == 'e' && n - i >= 16 && (i == 0 || !(cls[d[i - 1]] & alnum_class)) && d[i + 1] == 'y' &&
                        d[i + 2] == 'J')
                    {
                        size_t k = i + 3;
                        while (k < n && k - i < 16 && (cls[d[k]] & (alnum_class | base64url_class))) ++k;
                        if (k - i == 16) return true;
                    }
                    continue;
                }
                if (run > 0)
                {
                    lines = block_run(d, start, i, run, kinds) ? lines + 1 : 0;
                    if (lines >= 16) return true;
                    run = 0;
                }
                // "\n", "\r" or "\t" escaped in a string separates two runs as the character itself does.
                if (c == '\\' && i + 1 < n && (d[i + 1] == 'n' || d[i + 1] == 'r' || d[i + 1] == 't'))
                    ++i;
                else if (c == ',' && i >= 7 && at_without_case(text, i - 7, ";base64,"))
                {
                    size_t k = i + 1;
                    while (k < n && k - i <= 64 && (cls[d[k]] & base64_class)) ++k;
                    if (k - i > 64) return true;
                }
            }
            if (run > 0 && block_run(d, start, n, run, kinds) && lines + 1 >= 16) return true;
            return hex_bytes(text);
        }

        // The text of bytes in UTF-8: bytes themselves when they are UTF-8, else their decoding, in decoded.
        const std::string& utf8_of(const std::string& bytes, std::string& decoded)
        {
            const bool decode = bytes.compare(0, 2, "\xFF\xFE") == 0 || bytes.compare(0, 2, "\xFE\xFF") == 0 ||
                                privacy_impl::utf16_without_bom(bytes) != 0 || !privacy_impl::utf8_valid(bytes);
            if (!decode) return bytes;
            privacy_impl::text_encoding encoding;
            decoded = privacy_impl::decode_text(bytes, encoding);
            return decoded;
        }

        // A byte of a text: no control character but the tabulation, the ends of lines and the form feed.
        bool text_byte(unsigned char c)
        {
            return c >= 0x20 ? c != 0x7F : (c == '\t' || c == '\n' || c == '\r' || c == '\f');
        }

        enum class text_verdict { text, encoded, other };

        /*
            Whether bytes are a text through and through, which a name of no kind requires: no
            binary signature that dlib knows (dlib/data_io), a text by the measures of dlib, its
            printable characters and its entropy below 6.5, over its start as over its whole
            content, every character decoded (UTF-8, UTF-16, or Windows-1252 for other bytes) with
            no control character but the tabulation, the ends of lines and the form feed, no
            noncharacter U+FFFE or U+FFFF, no U+FFFD where the decoding of UTF-16 met a broken
            character, and no encoded part (encoded_format(), encoded_block()), else encoded.
            Valid UTF-8 is read in place, without a copy, U+FFFD written in it included.
        */
        text_verdict through_and_through(const std::string& bytes, const std::string& base)
        {
            if (bytes.empty()) return text_verdict::text;
            for (const auto& sig : dlib::impl::signatures)
            {
                // "BM" and "MZ" also open texts: a bitmap or a program fails the reading below.
                if (sig.length < 3 && sig.bytes[0] >= 0x20 && sig.bytes[0] < 0x7F) continue;
                if (bytes.size() >= sig.offset + sig.length &&
                    std::memcmp(bytes.data() + sig.offset, sig.bytes, sig.length) == 0)
                    return text_verdict::other;
            }
            if (family_of(detect_document_format(bytes, base)) != document_family::text) return text_verdict::other;
            std::string decoded;
            const std::string& text = utf8_of(bytes, decoded);
            // A byte order mark of UTF-8 followed by other bytes leaves them undecoded.
            if (&text == &decoded && !privacy_impl::utf8_valid(text)) return text_verdict::other;
            const unsigned char* d = reinterpret_cast<const unsigned char*>(text.data());
            const size_t n = text.size();
            if (n == 0) return text_verdict::text;
            if (!dlib::impl::is_text_content(d, n) || dlib::impl::calculate_entropy(d, n) >= 6.5)
                return text_verdict::other;
            // U+FFFE and U+FFFF, and U+FFFD that the decoding wrote for a character it could not read.
            const unsigned char last = &text == &decoded ? 0xBD : 0xBE;
            for (size_t i = 0; i < n; ++i)
            {
                if (!text_byte(d[i])) return text_verdict::other;
                // A control of the range U+0080 to U+009F.
                if (d[i] == 0xC2 && i + 1 < n && d[i + 1] < 0xA0) return text_verdict::other;
                if (d[i] == 0xEF && i + 2 < n && d[i + 1] == 0xBF && d[i + 2] >= last) return text_verdict::other;
            }
            return encoded_format(text) || encoded_block(text) ? text_verdict::encoded : text_verdict::text;
        }

        /*
            The kind of a file (engine::kind_of()) of base name base.  verdict receives the verdict
            on its text when it was read for one, for a name of no kind or without extension, else
            text_verdict::other.
        */
        file_kind kind_of_content(const std::string& bytes, const std::string& base, text_verdict& verdict)
        {
            verdict = text_verdict::other;
            if (name_extension(base).empty())
            {
                // Without extension: a picture, an Office document, an archive or a PDF by its signature.
                if (!image_media_type(bytes).empty()) return file_kind::picture;
                const document_family family = family_of(detect_document_format(bytes, base));
                if (family == document_family::office) return file_kind::document;
                if (archive_kind_of(bytes) != archive_kind::none) return file_kind::archive;
                if (family == document_family::pdf) return file_kind::document;
            }
            else
            {
                switch (kind_of_name(base))
                {
                    case name_kind::picture:
                        return image_media_type(bytes).empty() ? file_kind::unsupported : file_kind::picture;
                    case name_kind::archive:
                        // An archive holds what it holds: an Office document renamed .zip is read as an archive.
                        return archive_kind_of(bytes) == archive_kind::none ? file_kind::unsupported
                                                                            : file_kind::archive;
                    case name_kind::document:
                        return family_of(detect_document_format(bytes, base)) == document_family_of_name(base)
                            ? file_kind::document : file_kind::unsupported;
                    default:
                        break;
                }
            }
            // A name of no kind, or without extension, is a text when its content is text through and through.
            verdict = through_and_through(bytes, base);
            return verdict == text_verdict::text ? file_kind::document : file_kind::unsupported;
        }

        /*
            The values of a name, without its known extension, written as substitutes: the name is
            read whole, since the end of an e-mail address or of a telephone number looks like an
            extension, and, when it ends with what looks like one, also without it, which can hide a
            value from the detector ("jean.dupont@example.fr.log2"); the reading that finds more
            values wins, the whole name on a tie.  r receives the substitutions of that reading.
        */
        std::string name_values(document_anonymizer& a, document_report& r, const std::string& name)
        {
            const std::string ext = name_extension(name);
            if (ext.empty() || ext.size() >= name.size()) return a.anonymize_name(name, r);
            const std::string stem = name.substr(0, name.size() - ext.size());
            document_report cut = r;
            const std::string without = a.anonymize_name(stem, cut) + name.substr(stem.size());
            const std::string whole = a.anonymize_name(name, r);
            if (cut.substitutions.size() <= r.substitutions.size()) return whole;
            r = std::move(cut);
            return without;
        }

        // Whether bytes read as a text, which may hold tags whatever the name of its file.
        bool text_content(const std::string& bytes, const std::string& base)
        {
            return archive_kind_of(bytes) == archive_kind::none &&
                   family_of(detect_document_format(bytes, base)) == document_family::text;
        }
    }

// ----------------------------------------------------------------------------------------

    struct engine::impl
    {
        std::unique_ptr<pseudonymizer> p;
        pii_detector_options dopt;
        model_set m;
        // The key of the identifiers of the pictures, derived from the key of the engine.
        std::string picture_key;
        // The imposed terms, which set_terms() replaces while files are processed.
        mutable std::mutex terms_m;
        std::vector<pii_imposed_term> terms;

        // The options of a detector: those of the engine, with the imposed terms of the moment.
        pii_detector_options options() const
        {
            pii_detector_options o = dopt;
            std::lock_guard<std::mutex> lock(terms_m);
            o.imposed_terms = terms;
            return o;
        }

        ~impl()
        {
            if (!picture_key.empty()) secure_wipe(&picture_key[0], picture_key.size());
        }

        void attach(pii_detector& d) const
        {
            attach_pii_recognizers(d, m.tagger, m.span, 0.5, 0.9);
        }

        /*
            The protected form of a description, which may name what the picture shows: one line,
            without brackets, read by the detector as the text of a document, its values replaced
            by tags before it reaches the file, then cut at 1000 bytes outside its tags.  known,
            the detector of the document that holds the picture, adds what the document taught it:
            a name, a place or an organisation of its text that the description repeats bare.
            found counts the values.
        */
        std::string protect_description(const std::string& description, const pii_detector* known,
                                        document_report& found) const
        {
            std::string line;
            bool bounded = false;
            for (char c : description)
            {
                if (c == '\r' || c == '\n' || c == '\t') c = ' ';
                if (c == '[') c = '(';
                if (c == ']') c = ')';
                if (c == ' ' && (line.empty() || line.back() == ' ')) continue;
                line.push_back(c);
                // The answers of Albireo stay far below; past 1000 bytes, nothing is written.
                if (line.size() >= 16384)
                {
                    bounded = true;
                    break;
                }
            }
            // The last character, which the bound may have cut, goes whole.
            if (bounded)
            {
                while (!line.empty() && (static_cast<unsigned char>(line.back()) & 0xC0) == 0x80) line.pop_back();
                if (!line.empty() && (static_cast<unsigned char>(line.back()) & 0x80)) line.pop_back();
            }
            while (!line.empty() && line.back() == ' ') line.pop_back();
            if (line.empty()) return line;
            pii_detector d(options());
            attach(d);
            d.begin_document();
            d.learn(line);
            std::vector<pii_entity> values;
            for (const pii_entity& e : d.detect(line))
                if (!p->keeps_in_clear(e.type)) values.push_back(e);
            if (known)
                for (const pii_entity& e : known->detect(line))
                    if (!p->keeps_in_clear(e.type)) values.push_back(e);
            // The values that overlap become one, whose tag covers them all.
            std::sort(values.begin(), values.end(), [](const pii_entity& a, const pii_entity& b) {
                return a.begin < b.begin || (a.begin == b.begin && a.end > b.end);
            });
            std::vector<pii_entity> joined;
            for (const pii_entity& e : values)
            {
                if (e.end <= e.begin || e.end > line.size()) continue;
                if (!joined.empty() && e.begin < joined.back().end)
                {
                    if (e.end > joined.back().end) joined.back().end = e.end;
                    continue;
                }
                joined.push_back(e);
            }
            for (const pii_entity& e : joined) found.count(e.type);
            std::string out = p->apply(line, joined);
            if (out.size() > 1000)
            {
                size_t cut = 1000;
                // Never inside a tag, which a cut would break.
                const size_t open = out.rfind("[[", cut);
                if (open != std::string::npos)
                {
                    const size_t close = out.find("]]", open);
                    if (close == std::string::npos || close + 2 > cut) cut = open;
                }
                while (cut > 0 && (static_cast<unsigned char>(out[cut]) & 0xC0) == 0x80) --cut;
                out = out.substr(0, cut) + "...";
            }
            return out;
        }

        /*
            The marker of a picture, its description protected (protect_description(), known and
            found), which written receives by the SHA-256 of the picture when it is not null.  Its
            identifier is a digest of the picture keyed by the key of the engine: it tells nothing
            of the picture to whoever does not hold the key, and links no picture from one
            installation to another.
        */
        std::string marker_of(const std::string& bytes, const std::string& media_type, const std::string& location,
                              const picture_handler& handler, std::vector<std::string>& ids,
                              const pii_detector* known, document_report& found,
                              std::map<std::string, std::string>* written = nullptr) const
        {
            picture_found f;
            f.sha256 = sha256_hex(bytes);
            const std::array<unsigned char, 32> keyed = hmac_sha256_digest(picture_key, f.sha256);
            f.id = to_hex(keyed.data(), keyed.size()).substr(0, 16);
            f.media_type = media_type;
            f.location = location;
            f.bytes = &bytes;
            if (std::find(ids.begin(), ids.end(), f.id) == ids.end()) ids.push_back(f.id);
            const std::string description = protect_description(handler(f), known, found);
            if (written) (*written)[f.sha256] = description;
            return engine::picture_marker(f.id, description);
        }

        /*
            The base name of a file, its values replaced by substitutes that Windows takes, its
            extension kept: a, with the detector of the document it names, reads it.
        */
        std::string protect_base(document_anonymizer& a, document_report& r, const std::string& base,
                                 std::vector<substitution>& made) const
        {
            std::string stem, ext;
            split_extension(base, stem, ext);
            const std::size_t first = r.substitutions.size();
            std::string out = name_values(a, r, stem);
            for (std::size_t k = first; k < r.substitutions.size(); ++k)
            {
                document_substitution& s = r.substitutions[k];
                const std::string safe = windows_safe(s.substitute);
                if (safe != s.substitute)
                {
                    const std::size_t at = out.find(s.substitute);
                    if (at != std::string::npos) out.replace(at, s.substitute.size(), safe);
                    s.substitute = safe;
                }
                made.push_back(substitution{ s.substitute, s.tag });
            }
            if (out == stem) return base;
            // An extension of no kind that the reading left as it was keeps its place in a name cut short.
            const size_t kept = ext.empty() ? name_extension(stem).size() : 0;
            if (kept > 0 && kept < out.size() &&
                out.compare(out.size() - kept, kept, stem, stem.size() - kept, kept) == 0)
                return windows_name(out.substr(0, out.size() - kept), out.substr(out.size() - kept));
            return windows_name(out, ext);
        }

        std::string restore_base(document_anonymizer& a, document_report& r, const std::string& base) const
        {
            std::string stem, ext;
            split_extension(base, stem, ext);
            return a.restore_name(stem, r) + ext;
        }

        // The resolver of the document anonymizer, from the lookup of the caller.
        static pseudonymizer::substitute_resolver resolver_of(const substitute_lookup& lookup)
        {
            if (!lookup) return pseudonymizer::substitute_resolver();
            return [lookup](const std::string& candidate) { return lookup(candidate); };
        }

        // The protected name of a file read alone: a picture, an archive, a folder.
        std::string protect_alone(const std::string& base, const substitute_lookup& lookup,
                                  std::vector<substitution>& made, document_report& r) const
        {
            pii_detector d(options());
            attach(d);
            document_anonymizer a(*p, d);
            a.set_substitute_resolver(resolver_of(lookup));
            return protect_base(a, r, base, made);
        }

        std::string restore_alone(const std::string& base, const substitute_lookup& lookup, document_report& r) const
        {
            pii_detector d(options());
            document_anonymizer a(*p, d);
            a.set_substitute_resolver(resolver_of(lookup));
            return restore_base(a, r, base);
        }

        static std::string folder_of(const std::string& name)
        {
            return name.substr(0, name.size() - base_name(name).size());
        }

        /*
            A document, protected or restored.  An Office document asks for the descriptions of its
            pictures while it is read, before its text taught the detector anything, a PDF once it
            did: the description given for each picture is kept, Albireo asked once, and read again
            with what the whole text taught; when that reveals a value that the first reading of a
            description missed, a name of the text that it repeats bare, the document is protected
            again, its pictures taking their descriptions read with the detector of the first
            protection.
        */
        engine_result document(const std::string& bytes, const std::string& name, document_format f,
                               const picture_handler& handler, bool restoring, const substitute_lookup& lookup,
                               uint64_t& budget) const
        {
            pii_detector first(options());
            if (!handler || restoring) return read_document(bytes, name, f, handler, restoring, lookup, budget, first);
            std::map<std::string, std::string> given;
            const picture_handler once = [&](const picture_found& found) {
                const auto it = given.find(found.sha256);
                if (it != given.end()) return it->second;
                const std::string description = handler(found);
                given[found.sha256] = description;
                return description;
            };
            std::map<std::string, std::string> written;
            engine_result out = read_document(bytes, name, f, once, false, lookup, budget, first, nullptr, &written);
            bool again = false;
            document_report unused;
            for (const auto& w : written)
            {
                const auto g = given.find(w.first);
                if (g != given.end() && protect_description(g->second, &first, unused) != w.second)
                {
                    again = true;
                    break;
                }
            }
            if (!again) return out;
            out = engine_result();
            pii_detector second(options());
            return read_document(bytes, name, f, once, false, lookup, budget, second, &first);
        }

        /*
            One reading of a document by the detector d, which learns its text.  known, when not
            null, reads the descriptions of the pictures in its place (document()); written
            receives the descriptions written, by the SHA-256 of their pictures.
        */
        engine_result read_document(const std::string& bytes, const std::string& name, document_format f,
                                    const picture_handler& handler, bool restoring, const substitute_lookup& lookup,
                                    const uint64_t& budget, pii_detector& d, const pii_detector* known = nullptr,
                                    std::map<std::string, std::string>* written = nullptr) const
        {
            engine_result out;
            if (!restoring) attach(d);
            // A PDF keeps its pages: its text is replaced in place (the default of the anonymizer).
            document_anonymizer a(*p, d);
            // The properties of the document take substitutes, which a window of properties shows as they are.
            a.set_untagged_properties(true);
            a.set_substitute_resolver(resolver_of(lookup));
            /*
                A part of an Office document takes about 26 bytes of memory for each of its bytes
                while it is read: each part stays within 32 times the document, from 32 MB.
            */
            zip_limits limits;
            limits.max_total_size = std::min<uint64_t>(limits.max_total_size, budget);
            limits.max_entry_size = std::min<uint64_t>(std::min<uint64_t>(limits.max_entry_size, budget),
                std::max<uint64_t>(32ull << 20, static_cast<uint64_t>(bytes.size()) * 32));
            a.set_package_limits(limits);
            std::vector<std::string> ids;
            document_report described;
            if (handler && !restoring)
                a.set_image_describer([&](const document_image& img) {
                    return marker_of(img.bytes, img.media_type, base_name(name) + ": " + img.location, handler, ids,
                                     known ? known : &d, described, written);
                });
            document_report report;
            out.bytes = restoring ? a.restore(bytes, f, report, base_name(name))
                                  : a.anonymize(bytes, f, report, base_name(name));
            // The values of the descriptions count with those of the document.
            for (const auto& e : described.entities) report.entities[e.first] += e.second;
            report.dates_shifted += described.dates_shifted;
            // The name is read with what the document taught the detector.
            std::vector<substitution> named;
            out.name = folder_of(name) + (restoring ? restore_base(a, report, base_name(name))
                                                    : protect_base(a, report, base_name(name), named));
            out.kind = document_format_name(f);
            out.report = report_json(report);
            out.pictures = ids;
            // The substitutes of the properties and of the name, as written.
            for (const document_substitution& s : report.substitutions)
                out.substitutions.push_back(substitution{ s.substitute, s.tag });
            return out;
        }

        engine_result picture(const std::string& bytes, const std::string& name, const picture_handler& handler,
                              const substitute_lookup& lookup) const
        {
            engine_result out;
            std::vector<std::string> ids;
            const picture_handler none = [](const picture_found&) { return std::string(); };
            document_report report;
            out.bytes = marker_of(bytes, image_media_type(bytes), base_name(name), handler ? handler : none, ids,
                                  nullptr, report) + "\r\n";
            out.name = folder_of(name) + protect_alone(base_name(name), lookup, out.substitutions, report) + ".txt";
            out.kind = "image";
            out.report = report_json(report);
            out.pictures = ids;
            return out;
        }

        // known: the kind of the file when the caller found it already.
        engine_result any(const std::string& bytes, const std::string& name, const picture_handler& handler,
                          bool restoring, const substitute_lookup& lookup, int depth, uint64_t& budget,
                          const file_kind* known = nullptr) const
        {
            const std::string base = base_name(name);
            text_verdict verdict = text_verdict::other;
            file_kind k = known ? *known : kind_of_content(bytes, base, verdict);
            // A text under another name may hold tags, which its restoration puts back.
            if (k == file_kind::unsupported && restoring && text_content(bytes, base)) k = file_kind::document;
            if (k == file_kind::picture)
            {
                if (!restoring) return picture(bytes, name, handler, lookup);
                engine_result out;
                document_report report;
                out.bytes = bytes;
                out.name = folder_of(name) + restore_alone(base_name(name), lookup, report);
                out.kind = "image";
                out.report = report_json(report);
                return out;
            }
            if (k == file_kind::archive) return archive(bytes, name, handler, restoring, lookup, depth, budget);
            if (k == file_kind::document)
                return document(bytes, name, detect_document_format(bytes, base_name(name)), handler, restoring,
                                lookup, budget);
            if (supported_name(base))
                throw std::runtime_error(base + ": the content of this file does not match its name");
            if (known) verdict = through_and_through(bytes, base);
            if (verdict == text_verdict::encoded)
                throw std::runtime_error(base + ": this file holds encoded parts that the detector cannot read");
            throw std::runtime_error(base + ": this kind of file is not supported");
        }

        /*
            The folders of the path of an entry, each one protected, or restored, once for the
            whole archive: "Clients/Jean Dupont/" gives the same folder to all its entries.
        */
        std::string entry_folders(const std::string& folders, bool restoring, const substitute_lookup& lookup,
                                  std::map<std::string, std::string>& done, engine_result& out) const
        {
            std::string result;
            std::size_t start = 0;
            while (start < folders.size())
            {
                std::size_t slash = folders.find('/', start);
                if (slash == std::string::npos) slash = folders.size();
                const std::string part = folders.substr(start, slash - start);
                if (!part.empty() && part != "." && part != "..")
                {
                    auto it = done.find(part);
                    if (it == done.end())
                    {
                        document_report r;
                        std::string changed;
                        if (restoring)
                        {
                            pii_detector d(options());
                            document_anonymizer a(*p, d);
                            a.set_substitute_resolver(resolver_of(lookup));
                            changed = a.restore_name(part, r);
                        }
                        else
                        {
                            pii_detector d(options());
                            attach(d);
                            document_anonymizer a(*p, d);
                            a.set_substitute_resolver(resolver_of(lookup));
                            // A folder has no extension: its whole name is read.
                            const std::size_t first = r.substitutions.size();
                            changed = name_values(a, r, part);
                            for (std::size_t k = first; k < r.substitutions.size(); ++k)
                            {
                                document_substitution& s = r.substitutions[k];
                                const std::string safe = windows_safe(s.substitute);
                                if (safe != s.substitute)
                                {
                                    const std::size_t at = changed.find(s.substitute);
                                    if (at != std::string::npos) changed.replace(at, s.substitute.size(), safe);
                                    s.substitute = safe;
                                }
                                out.substitutions.push_back(substitution{ s.substitute, s.tag });
                            }
                            if (changed != part) changed = windows_name(changed, std::string());
                        }
                        merge_report(out.report, report_json(r), std::string());
                        it = done.insert(std::make_pair(part, changed)).first;
                    }
                    result += it->second;
                }
                else
                {
                    result += part;
                }
                if (slash < folders.size()) result += '/';
                start = slash + 1;
            }
            return result;
        }

        engine_result archive(const std::string& bytes, const std::string& name, const picture_handler& handler,
                              bool restoring, const substitute_lookup& lookup, int depth, uint64_t& budget) const
        {
            if (depth > 3) throw std::runtime_error(base_name(name) + ": archives nested too deep");
            archive_kind ak = archive_kind_of(bytes);
            if (ak == archive_kind::none) ak = archive_kind_of_name(name);
            // The entries come out of what is left of the budget of the file.
            archive_limits limits;
            limits.max_total_size = std::min<uint64_t>(limits.max_total_size, budget);
            limits.max_entry_size = std::min<uint64_t>(limits.max_entry_size, budget);
            std::vector<archive_entry> entries = read_archive(ak, bytes, limits);
            uint64_t read = 0;
            for (const archive_entry& e : entries) read += e.data.size();
            budget -= std::min(budget, read);
            engine_result out;
            out.kind = archive_kind_name(ak);
            out.report = report_json(document_report());
            {
                document_report r;
                out.name = folder_of(name) + (restoring ? restore_alone(base_name(name), lookup, r)
                                                        : protect_alone(base_name(name), lookup, out.substitutions, r));
                merge_report(out.report, report_json(r), std::string());
            }
            std::set<std::string> names;
            std::map<std::string, int> next_number;
            std::set<std::string> pictures;
            std::map<std::string, std::string> folders;
            for (const archive_entry& e : entries) names.insert(lower(e.name));
            std::vector<archive_entry> kept;
            for (archive_entry& e : entries)
            {
                if (e.directory)
                {
                    const std::string renamed = entry_folders(e.name, restoring, lookup, folders, out);
                    if (renamed != e.name)
                    {
                        names.erase(lower(e.name));
                        e.name = renamed;
                        names.insert(lower(e.name));
                    }
                    kept.push_back(std::move(e));
                    continue;
                }
                // An entry left out gives its memory back at once.
                const file_kind kind =
                    e.regular && !restoring ? engine::kind_of(e.data, e.name) : file_kind::unsupported;
                if (!e.regular || (!restoring && kind == file_kind::unsupported))
                {
                    out.report["dropped"].push_back(e.name);
                    std::string().swap(e.data);
                    continue;
                }
                try
                {
                    engine_result r = any(e.data, e.name, handler, restoring, lookup, depth + 1, budget,
                                          restoring ? nullptr : &kind);
                    const std::string read_as = e.name;
                    const std::string renamed =
                        entry_folders(folder_of(e.name), restoring, lookup, folders, out) + base_name(r.name);
                    if (renamed != e.name)
                    {
                        names.erase(lower(e.name));
                        e.name = free_name(renamed, names, next_number);
                    }
                    // The entry written, by which a copy extracted from the archive is known again.
                    if (!restoring)
                    {
                        entry_written w;
                        w.original = read_as;
                        w.name = e.name;
                        w.sha256 = sha256_hex(r.bytes);
                        w.kind = r.kind;
                        out.entries.push_back(std::move(w));
                        for (entry_written& inner : r.entries)
                        {
                            inner.original = read_as + "!" + inner.original;
                            inner.name = e.name + "!" + inner.name;
                            out.entries.push_back(std::move(inner));
                        }
                    }
                    e.data = std::move(r.bytes);
                    merge_report(out.report, r.report, e.name);
                    for (const std::string& id : r.pictures)
                        if (pictures.insert(id).second) out.pictures.push_back(id);
                    for (substitution& s : r.substitutions) out.substitutions.push_back(std::move(s));
                    kept.push_back(std::move(e));
                }
                catch (std::exception& x)
                {
                    /*
                        A restoration keeps an entry it cannot read, which holds no tag, as it is, in
                        its folder restored with the others and under its name restored.  A protection
                        leaves it out: it could hold personal data.
                    */
                    if (restoring)
                    {
                        document_report r;
                        const std::string renamed = entry_folders(folder_of(e.name), true, lookup, folders, out) +
                                                    restore_alone(base_name(e.name), lookup, r);
                        merge_report(out.report, report_json(r), std::string());
                        if (renamed != e.name)
                        {
                            names.erase(lower(e.name));
                            e.name = free_name(renamed, names, next_number);
                        }
                        kept.push_back(std::move(e));
                        continue;
                    }
                    out.report["dropped"].push_back(e.name);
                    out.report["notes"].push_back(e.name + ": " + x.what());
                    std::string().swap(e.data);
                }
            }
            out.bytes = write_archive(ak, kept);
            return out;
        }

        /*
            The entry at path of an archive (engine::extract()).  An archive inside it is entered by
            its name and '!': the longest name of an entry that starts the path, so that a name that
            holds '!' is read whole.
        */
        std::string extract(const std::string& bytes, const std::string& name, const std::string& path) const
        {
            std::string data = bytes, label = name, rest = path;
            for (int depth = 0; depth <= 3; ++depth)
            {
                archive_kind ak = archive_kind_of(data);
                if (ak == archive_kind::none) ak = archive_kind_of_name(label);
                if (ak == archive_kind::none) break;
                std::vector<archive_entry> entries = read_archive(ak, data, archive_limits());
                std::size_t inner = entries.size();
                for (std::size_t k = 0; k < entries.size(); ++k)
                {
                    const archive_entry& e = entries[k];
                    if (e.directory || !e.regular) continue;
                    if (e.name == rest) return std::move(entries[k].data);
                    const bool starts = rest.size() > e.name.size() + 1 && rest[e.name.size()] == '!' &&
                                        rest.compare(0, e.name.size(), e.name) == 0;
                    if (starts && (inner == entries.size() || e.name.size() > entries[inner].name.size())) inner = k;
                }
                if (inner == entries.size()) break;
                rest = rest.substr(entries[inner].name.size() + 1);
                label = entries[inner].name;
                data = std::move(entries[inner].data);
            }
            throw std::runtime_error(base_name(name) + ": the archive holds no such entry");
        }
    };

// ----------------------------------------------------------------------------------------

    engine::engine(const std::string& key, const engine_options& options) : p_(std::make_shared<impl>())
    {
        if (key.size() != 32) throw std::runtime_error("the key of the engine must have 32 bytes");
        pseudonymizer_options popt;
        popt.context = options.context.empty() ? "default" : options.context;
        popt.dates = date_policy::shift;
        popt.date_order = options.date_order == "mdy" ? numeric_date_order::month_day_year
                                                      : numeric_date_order::day_month_year;
        p_->p.reset(new pseudonymizer(pseudonymization_key::from_bytes(key), popt));
        p_->dopt.date_order = popt.date_order;
        p_->picture_key = hkdf_sha256(key, "pseudalis.pictures", "identifier", 32);
        try
        {
            p_->m = shared_models();
        }
        catch (std::exception& e)
        {
            throw std::runtime_error(std::string("the models of the engine cannot be read: ") + e.what());
        }
    }

    file_kind engine::kind_of_name(const std::string& name)
    {
        switch (pseudalis::kind_of_name(name))
        {
            case name_kind::document: return file_kind::document;
            case name_kind::picture: return file_kind::picture;
            case name_kind::archive: return file_kind::archive;
            default: return file_kind::unsupported;
        }
    }

    file_kind engine::kind_of(const std::string& bytes, const std::string& name)
    {
        text_verdict verdict;
        return kind_of_content(bytes, base_name(name), verdict);
    }

    std::string engine::extract(const std::string& bytes, const std::string& name, const std::string& path) const
    {
        return p_->extract(bytes, name, path);
    }

    std::string engine::picture_marker(const std::string& id, const std::string& description)
    {
        return description.empty() ? "[Image@" + id + "]" : "[Image@" + id + " : " + description + "]";
    }

    engine_result engine::anonymize(const std::string& bytes, const std::string& name,
                                    const picture_handler& pictures, const substitute_lookup& lookup) const
    {
        uint64_t budget = decompression_budget(bytes.size());
        return p_->any(bytes, name, pictures, false, lookup, 0, budget);
    }

    engine_result engine::restore(const std::string& bytes, const std::string& name,
                                  const substitute_lookup& lookup) const
    {
        uint64_t budget = decompression_budget(bytes.size());
        return p_->any(bytes, name, picture_handler(), true, lookup, 0, budget);
    }

    std::string engine::anonymize_name(const std::string& name, const substitute_lookup& lookup,
                                       std::vector<substitution>& made, json& report) const
    {
        document_report r;
        const std::string out = impl::folder_of(name) + p_->protect_alone(base_name(name), lookup, made, r);
        report = report_json(r);
        return out;
    }

    std::string engine::restore_name(const std::string& name, const substitute_lookup& lookup, json& report) const
    {
        document_report r;
        const std::string out = impl::folder_of(name) + p_->restore_alone(base_name(name), lookup, r);
        report = report_json(r);
        return out;
    }

    std::string engine::anonymize_text(const std::string& text, json& report) const
    {
        pii_detector d(p_->options());
        p_->attach(d);
        document_anonymizer a(*p_->p, d);
        document_report r;
        const std::string out = a.anonymize_text(text, r);
        report = report_json(r);
        return out;
    }

    std::string engine::restore_text(const std::string& text, json& report) const
    {
        pii_detector d(p_->options());
        document_anonymizer a(*p_->p, d);
        document_report r;
        const std::string out = a.restore(text, document_format::text, r);
        report = report_json(r);
        return out;
    }

    void engine::set_terms(const std::vector<imposed_term>& terms)
    {
        std::vector<pii_imposed_term> converted;
        for (const imposed_term& t : terms)
        {
            pii_imposed_term c;
            c.text = t.term;
            c.type = pii_type_from_keyword(t.type);
            c.case_sensitive = t.case_sensitive;
            if (c.type != pii_type::none && !c.text.empty()) converted.push_back(c);
        }
        std::lock_guard<std::mutex> lock(p_->terms_m);
        p_->terms.swap(converted);
    }

    std::size_t engine::term_count() const
    {
        std::lock_guard<std::mutex> lock(p_->terms_m);
        return p_->terms.size();
    }

    json engine::detect(const std::string& text) const
    {
        pii_detector d(p_->options());
        p_->attach(d);
        d.begin_document();
        d.learn(text);
        json spans = json::array();
        for (const pii_entity& e : d.detect(text))
        {
            json s = json::object();
            s["begin"] = static_cast<unsigned long long>(e.begin);
            s["end"] = static_cast<unsigned long long>(e.end);
            s["type"] = pii_type_keyword(e.type);
            s["confidence"] = e.confidence;
            s["text"] = text.substr(e.begin, e.end - e.begin);
            spans.push_back(s);
        }
        return spans;
    }

    std::string engine::key_fingerprint() const
    {
        return p_->p->key_fingerprint();
    }

    json engine::models()
    {
        json j = json::object();
        j["tagger"] = anonymizer_models::tagger_source();
        j["span_model"] = anonymizer_models::span_model_source();
        j["size"] = anonymizer_models::span_model_size();
        return j;
    }

}
