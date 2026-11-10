// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DOCUMENT_ANONYMIZER_H_
#define DLIB_DOCUMENT_ANONYMIZER_H_

#include "document_anonymizer_abstract.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "../data_io/language_model_data.h"
#include "../zip/zip_archive.h"
#include "../documents/xml_scanner.h"
#include "unicode_text.h"
#include "pii_detector.h"
#include "pseudonymizer.h"
#include "document_report.h"
#include "ooxml_anonymizer.h"
#include "pdf_anonymizer.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class document_format
    {
        unknown,
        text,
        delimited,
        markup,
        docx,
        xlsx,
        pptx,
        pdf,
        legacy_office
    };

    inline const char* document_format_name(document_format f)
    {
        switch (f)
        {
            case document_format::text: return "txt";
            case document_format::delimited: return "csv";
            case document_format::markup: return "xml";
            case document_format::docx: return "docx";
            case document_format::xlsx: return "xlsx";
            case document_format::pptx: return "pptx";
            case document_format::pdf: return "pdf";
            case document_format::legacy_office: return "legacy-office";
            default: return "unknown";
        }
    }

    struct pii_imposed_column
    {
        std::string name;
        pii_type type = pii_type::none;
    };

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        enum class text_encoding { utf8, utf8_bom, utf16le, utf16be, utf16le_plain, utf16be_plain, cp1252 };

        inline std::uint32_t cp1252_to_unicode(unsigned char c)
        {
            static const std::uint16_t high[32] = { 0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
                                                    0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F,
                                                    0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
                                                    0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178 };
            if (c >= 0x80 && c < 0xA0) return high[c - 0x80];
            return c;
        }

        inline int unicode_to_cp1252(std::uint32_t cp)
        {
            if (cp < 0x80 || (cp >= 0xA0 && cp <= 0xFF)) return static_cast<int>(cp);
            for (int c = 0x80; c < 0xA0; ++c)
                if (cp1252_to_unicode(static_cast<unsigned char>(c)) == cp) return c;
            return -1;
        }

        /* UTF-16 without byte order mark: in the first bytes, one position of each pair
           is mostly zero and the other almost never. Returns 1 for little endian, 2 for
           big endian, 0 otherwise. */
        inline int utf16_without_bom(const std::string& bytes)
        {
            const std::size_t n = std::min<std::size_t>(bytes.size(), 4096) & ~std::size_t(1);
            if (n < 8) return 0;
            std::size_t zero_even = 0, zero_odd = 0;
            for (std::size_t i = 0; i < n; i += 2)
            {
                if (bytes[i] == 0) ++zero_even;
                if (bytes[i + 1] == 0) ++zero_odd;
            }
            const std::size_t pairs = n / 2;
            if (10 * zero_odd >= 4 * pairs && 20 * zero_even < pairs) return 1;
            if (10 * zero_even >= 4 * pairs && 20 * zero_odd < pairs) return 2;
            return 0;
        }

        /* Decodes a text file to UTF-8 and reports its encoding so that it can be written
           back the same way. Without a byte order mark, UTF-16 is recognized by its zero
           bytes, valid UTF-8 is taken as UTF-8 and anything else as Windows-1252, which
           decodes every byte. An unpaired surrogate or a final odd byte becomes U+FFFD. */
        inline std::string decode_text(const std::string& bytes, text_encoding& enc)
        {
            if (bytes.size() >= 3 && bytes.compare(0, 3, "\xEF\xBB\xBF") == 0)
            {
                enc = text_encoding::utf8_bom;
                return bytes.substr(3);
            }
            const bool bom = bytes.size() >= 2 &&
                             (bytes.compare(0, 2, "\xFF\xFE") == 0 || bytes.compare(0, 2, "\xFE\xFF") == 0);
            const int plain = bom ? 0 : utf16_without_bom(bytes);
            if (bom || plain)
            {
                const bool le = bom ? static_cast<unsigned char>(bytes[0]) == 0xFF : plain == 1;
                enc = bom ? (le ? text_encoding::utf16le : text_encoding::utf16be)
                          : (le ? text_encoding::utf16le_plain : text_encoding::utf16be_plain);
                std::string out;
                std::size_t i = bom ? 2 : 0;
                for (; i + 1 < bytes.size(); i += 2)
                {
                    auto unit = [&](std::size_t k)
                    {
                        const std::uint32_t a = static_cast<unsigned char>(bytes[k]),
                                            b = static_cast<unsigned char>(bytes[k + 1]);
                        return le ? (a | (b << 8)) : ((a << 8) | b);
                    };
                    std::uint32_t u = unit(i);
                    if (u >= 0xD800 && u <= 0xDBFF && i + 3 < bytes.size())
                    {
                        const std::uint32_t lo = unit(i + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF)
                        {
                            u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                            i += 2;
                        }
                    }
                    if (u >= 0xD800 && u <= 0xDFFF) u = 0xFFFD;
                    utf8_append(out, static_cast<char32_t>(u));
                }
                if (i < bytes.size()) utf8_append(out, static_cast<char32_t>(0xFFFD));
                return out;
            }
            if (utf8_valid(bytes))
            {
                enc = text_encoding::utf8;
                return bytes;
            }
            enc = text_encoding::cp1252;
            std::string out;
            out.reserve(bytes.size() + bytes.size() / 8);
            for (unsigned char c : bytes)
                utf8_append(out, static_cast<char32_t>(cp1252_to_unicode(c)));
            return out;
        }

        inline const char* text_encoding_name(text_encoding enc)
        {
            switch (enc)
            {
                case text_encoding::utf8: return "utf-8";
                case text_encoding::utf8_bom: return "utf-8-bom";
                case text_encoding::utf16le: return "utf-16le";
                case text_encoding::utf16be: return "utf-16be";
                case text_encoding::utf16le_plain: return "utf-16le-nobom";
                case text_encoding::utf16be_plain: return "utf-16be-nobom";
                case text_encoding::cp1252: return "windows-1252";
            }
            return "";
        }

        // Whether Windows-1252 writes every character of a text, which encode_text() would replace by '?'.
        inline bool cp1252_writes(const std::string& utf8)
        {
            std::size_t i = 0;
            while (i < utf8.size())
            {
                if ((utf8[i] & 0x80) == 0) { ++i; continue; }
                if (unicode_to_cp1252(static_cast<std::uint32_t>(utf8_next(utf8, i))) < 0) return false;
            }
            return true;
        }

        inline std::string encode_text(const std::string& utf8, text_encoding enc)
        {
            switch (enc)
            {
                case text_encoding::utf8: return utf8;
                case text_encoding::utf8_bom: return "\xEF\xBB\xBF" + utf8;
                default: break;
            }
            std::string out;
            if (enc == text_encoding::utf16le) out = "\xFF\xFE";
            if (enc == text_encoding::utf16be) out = "\xFE\xFF";
            const bool little = enc == text_encoding::utf16le || enc == text_encoding::utf16le_plain;
            std::size_t i = 0;
            while (i < utf8.size())
            {
                const char32_t cp = utf8_next(utf8, i);
                if (enc == text_encoding::cp1252)
                {
                    const int b = unicode_to_cp1252(static_cast<std::uint32_t>(cp));
                    out.push_back(static_cast<char>(b < 0 ? '?' : b));
                    continue;
                }
                auto put = [&](std::uint32_t u) {
                    const char hi = static_cast<char>((u >> 8) & 0xFF), lo = static_cast<char>(u & 0xFF);
                    if (little) { out.push_back(lo); out.push_back(hi); }
                    else { out.push_back(hi); out.push_back(lo); }
                };
                if (cp >= 0x10000)
                {
                    const std::uint32_t v = static_cast<std::uint32_t>(cp) - 0x10000;
                    put(0xD800 + (v >> 10));
                    put(0xDC00 + (v & 0x3FF));
                }
                else
                {
                    put(static_cast<std::uint32_t>(cp));
                }
            }
            return out;
        }

        inline bool ends_with_ci(const std::string& name, const char* ext)
        {
            return dlib::impl::has_extension(name, ext);
        }

        inline document_format zip_document_format(const std::string& bytes)
        {
            try
            {
                const zip_archive_reader z(bytes);
                if (!z.contains("[Content_Types].xml"))
                    return document_format::unknown;
                if (z.contains("word/document.xml")) return document_format::docx;
                if (z.contains("xl/workbook.xml")) return document_format::xlsx;
                if (z.contains("ppt/presentation.xml")) return document_format::pptx;
                const std::string ct = z.read("[Content_Types].xml");
                if (ct.find("wordprocessingml") != std::string::npos) return document_format::docx;
                if (ct.find("spreadsheetml") != std::string::npos) return document_format::xlsx;
                if (ct.find("presentationml") != std::string::npos) return document_format::pptx;
            }
            catch (std::exception&)
            {
            }
            return document_format::unknown;
        }

        // Counts the occurrences of delim outside quotes in each of the first records from start.
        inline std::vector<std::size_t> delimiter_counts(const std::string& text, char delim, std::size_t max_records,
                                                         std::size_t start = 0)
        {
            std::vector<std::size_t> counts;
            std::size_t count = 0;
            bool quoted = false, any = false;
            for (std::size_t i = start; i < text.size() && counts.size() < max_records; ++i)
            {
                const char c = text[i];
                if (c == '"') quoted = !quoted;
                else if (!quoted && c == delim) ++count;
                else if (!quoted && (c == '\n' || c == '\r'))
                {
                    if (c == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
                    if (any) counts.push_back(count);
                    count = 0;
                    any = false;
                    continue;
                }
                any = true;
            }
            if (any && counts.size() < max_records) counts.push_back(count);
            return counts;
        }

        // The delimiter of a delimiter-separated file, or 0 when the text is not one.
        inline char sniff_delimiter(const std::string& text, const std::string& name_hint)
        {
            const bool tsv = ends_with_ci(name_hint, ".tsv") || ends_with_ci(name_hint, ".tab");
            const bool csv = ends_with_ci(name_hint, ".csv");
            if (tsv) return '\t';
            char best = 0;
            std::size_t best_score = 0;
            for (char d : { ';', ',', '\t', '|' })
            {
                const std::vector<std::size_t> counts = delimiter_counts(text, d, 30);
                if (counts.empty() || counts[0] == 0) continue;
                bool consistent = true;
                for (std::size_t c : counts)
                    if (c != counts[0]) consistent = false;
                const std::size_t needed = csv ? 1 : 3;
                if (consistent && counts.size() >= needed && counts[0] * counts.size() > best_score)
                {
                    best = d;
                    best_score = counts[0] * counts.size();
                }
            }
            if (!best && csv)
            {
                // A ragged CSV file: the delimiter most present on the header line.
                std::size_t most = 0;
                for (char d : { ';', ',', '\t', '|' })
                {
                    const std::vector<std::size_t> counts = delimiter_counts(text, d, 1);
                    if (!counts.empty() && counts[0] > most) { most = counts[0]; best = d; }
                }
            }
            return best;
        }

        // The cells of one record of a delimited file, each without its quotes, a quote opening a cell alone.
        inline std::vector<std::string> split_record(const std::string& line, char delim)
        {
            std::vector<std::string> cells;
            std::size_t i = 0;
            for (;;)
            {
                std::string cell;
                if (i < line.size() && line[i] == '"')
                {
                    for (++i; i < line.size(); ++i)
                    {
                        if (line[i] != '"') { cell.push_back(line[i]); continue; }
                        if (i + 1 < line.size() && line[i + 1] == '"') { cell.push_back('"'); ++i; continue; }
                        ++i;
                        break;
                    }
                }
                while (i < line.size() && line[i] != delim) cell.push_back(line[i++]);
                cells.push_back(cell);
                if (i >= line.size()) break;
                ++i;
            }
            return cells;
        }

        /*
            The address of a value of an IP column, with what may surround it: a prefix length
            ("10.0.0.0/8", "2001:db8::/32"), a port ("192.0.2.1:443", "[2001:db8::1]:443") or a
            zone ("fe80::1%eth0").  Returns ipv4 or ipv6 with the bounds [b, e) of the address in
            v, or none for a value of another form.
        */
        inline pii_type ip_address_in(const std::string& v, std::size_t& b, std::size_t& e)
        {
            // A number of 1 to digits digits, from i to the end of v, of at most limit.
            const auto number_to_end = [&v](std::size_t i, std::size_t digits, unsigned long limit) {
                if (i >= v.size() || v.size() - i > digits) return false;
                unsigned long n = 0;
                for (std::size_t k = i; k < v.size(); ++k)
                {
                    if (!is_ascii_digit_char(v[k])) return false;
                    n = n * 10 + static_cast<unsigned long>(v[k] - '0');
                }
                return n <= limit;
            };
            std::uint32_t a4 = 0;
            std::array<unsigned char, 16> a6;
            if (!v.empty() && v[0] == '[')
            {
                const std::size_t close = v.find(']');
                if (close == std::string::npos || !parse_ipv6(v.substr(1, close - 1), a6)) return pii_type::none;
                if (close + 1 != v.size() && (v[close + 1] != ':' || !number_to_end(close + 2, 5, 65535)))
                    return pii_type::none;
                b = 1;
                e = close;
                return pii_type::ipv6;
            }
            const std::size_t slash = v.find('/');
            const std::string head = v.substr(0, slash);
            if (parse_ipv4(head, a4))
            {
                if (slash != std::string::npos && !number_to_end(slash + 1, 2, 32)) return pii_type::none;
                b = 0;
                e = head.size();
                return pii_type::ipv4;
            }
            const std::size_t colon = head.find(':');
            if (slash == std::string::npos && colon != std::string::npos &&
                head.find(':', colon + 1) == std::string::npos && parse_ipv4(head.substr(0, colon), a4))
            {
                if (!number_to_end(colon + 1, 5, 65535)) return pii_type::none;
                b = 0;
                e = colon;
                return pii_type::ipv4;
            }
            const std::size_t zone = head.find('%');
            const std::string address = head.substr(0, zone);
            if (!parse_ipv6(address, a6)) return pii_type::none;
            if (zone != std::string::npos)
            {
                if (slash != std::string::npos || zone + 1 >= head.size()) return pii_type::none;
                for (std::size_t k = zone + 1; k < head.size(); ++k)
                {
                    const char c = head[k];
                    if (!is_ascii_digit_char(c) && !((c | 0x20) >= 'a' && (c | 0x20) <= 'z') && c != '.' && c != '_' &&
                        c != '-')
                        return pii_type::none;
                }
            }
            if (slash != std::string::npos && !number_to_end(slash + 1, 3, 128)) return pii_type::none;
            b = 0;
            e = address.size();
            return pii_type::ipv6;
        }

        /*
            The delimiter of a text whose header names a column of keys (fold_key() of the names
            of the imposed columns), or 0.  The header is the first of the first 16 lines that are
            not blank to hold such a name in a cell, a first line "sep=;" of Excel giving the
            delimiter it names first; the delimiter that makes the most cells named, then the
            most cells, wins.  A file named .csv, .tsv or .tab may hold a column alone, its name
            alone on its line; it is then read with a delimiter that the text does not hold, the
            unit separator of ASCII first.  In another text, the header holds the delimiter, and
            the two records that follow it hold it as many times.
        */
        inline char imposed_delimiter(const std::string& text, const std::vector<std::string>& keys,
                                      const std::string& name_hint)
        {
            if (keys.empty()) return 0;
            const bool tsv = ends_with_ci(name_hint, ".tsv") || ends_with_ci(name_hint, ".tab");
            const bool named = tsv || ends_with_ci(name_hint, ".csv");
            char hinted = 0;
            std::size_t pos = 0, lines = 0;
            while (pos < text.size() && lines < 16)
            {
                std::size_t end = text.find('\n', pos);
                if (end == std::string::npos) end = text.size();
                const std::size_t next = end + 1;
                if (end > pos && text[end - 1] == '\r') --end;
                // A header is short: a longer line is no header, and is not split.
                if (end - pos > 65536)
                {
                    ++lines;
                    pos = next;
                    continue;
                }
                const std::string line = text.substr(pos, end - pos);
                pos = next;
                if (line.find_first_not_of(" \t") == std::string::npos) continue;
                ++lines;
                if (lines == 1 && line.size() == 5 && fold_key(line.substr(0, 4)) == "sep=")
                {
                    hinted = line[4];
                    continue;
                }
                char best = 0;
                std::size_t best_named = 0, best_cells = 0;
                for (char d : { hinted, '\t', ';', ',', '|' })
                {
                    if (!d) continue;
                    const std::vector<std::string> cells = split_record(line, d);
                    std::size_t found = 0;
                    for (const std::string& c : cells)
                        if (std::find(keys.begin(), keys.end(), fold_key(c)) != keys.end()) ++found;
                    if (found > best_named || (found == best_named && found > 0 && cells.size() > best_cells))
                    {
                        best = d;
                        best_named = found;
                        best_cells = cells.size();
                    }
                }
                if (best_named == 0) continue;
                if (best_cells >= 2)
                {
                    if (named) return best;
                    const std::vector<std::size_t> after = delimiter_counts(text, best, 2, pos);
                    return after.size() == 2 && after[0] == best_cells - 1 && after[1] == best_cells - 1 ? best : 0;
                }
                /* A column alone: a delimiter that no value holds, the protected text and its
                   restoration included, so that each line stays one cell. */
                if (!named) return 0;
                if (tsv) return '\t';
                for (char d : { '\x1f', ',', ';', '\t', '|' })
                    if (text.find(d) == std::string::npos) return d;
                return ',';
            }
            return 0;
        }

        inline bool looks_like_markup(const std::string& text)
        {
            std::size_t i = 0;
            while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) ++i;
            const std::string head = text.substr(i, 512);
            return dlib::impl::is_xml_content(reinterpret_cast<const unsigned char*>(head.data()), head.size()) ||
                (head.compare(0, 1, "<") == 0 && head.find('>') != std::string::npos &&
                 (head.compare(0, 4, "<svg") == 0 || head.compare(0, 5, "<html") == 0 ||
                  head.compare(0, 5, "<HTML") == 0));
        }

        inline bool is_text_head(const std::string& bytes)
        {
            const std::size_t n = std::min<std::size_t>(bytes.size(), 8192);
            const unsigned char* d = reinterpret_cast<const unsigned char*>(bytes.data());
            return dlib::impl::is_text_content(d, n) && dlib::impl::calculate_entropy(d, n) < 6.5;
        }

        inline document_format classify_text(const std::string& bytes, const std::string& name_hint)
        {
            text_encoding enc;
            const std::string text = decode_text(bytes.substr(0, 65536), enc);
            if (looks_like_markup(text)) return document_format::markup;
            if (sniff_delimiter(text, name_hint)) return document_format::delimited;
            return document_format::text;
        }
    }

// ----------------------------------------------------------------------------------------

    inline document_format detect_document_format(const std::string& bytes, const std::string& name_hint = "")
    {
        using namespace privacy_impl;
        if (bytes.empty()) return document_format::text;
        if (bytes.size() >= 8 && bytes.compare(0, 8, "\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1") == 0)
            return document_format::legacy_office;
        if (bytes.compare(0, 4, "PK\x03\x04") == 0)
            return zip_document_format(bytes);
        const std::size_t pdf = bytes.find("%PDF-");
        if (pdf == 0 ||
            (pdf != std::string::npos && pdf < 1024 && bytes.find("%%EOF") != std::string::npos &&
             !is_text_head(bytes.substr(0, pdf))))
            return document_format::pdf;
        const bool bom = bytes.compare(0, 3, "\xEF\xBB\xBF") == 0 || bytes.compare(0, 2, "\xFF\xFE") == 0 ||
                         bytes.compare(0, 2, "\xFE\xFF") == 0;
        if (bom || utf16_without_bom(bytes) || is_text_head(bytes))
            return classify_text(bytes, name_hint);
        return document_format::unknown;
    }

    inline document_format detect_document_file_format(const std::string& filename)
    {
        using namespace privacy_impl;
        file_content_type type;
        const bool text = detect_file_type(filename, type);
        if (type == file_content_type::PDF)
            return document_format::pdf;

        std::ifstream in(filename, std::ios::binary);
        if (!in) return document_format::unknown;
        std::string head(65536, '\0');
        in.read(&head[0], static_cast<std::streamsize>(head.size()));
        head.resize(static_cast<std::size_t>(in.gcount()));
        if (head.empty()) return document_format::text;

        if (type == file_content_type::OFFICE || type == file_content_type::COMPRESSED)
        {
            if (head.compare(0, 4, "PK\x03\x04") != 0) return document_format::unknown;
            in.clear();
            in.seekg(0, std::ios::end);
            const std::streamoff size = in.tellg();
            if (size <= 0 || size > (std::streamoff(2048) << 20)) return document_format::unknown;
            std::string all(static_cast<std::size_t>(size), '\0');
            in.seekg(0, std::ios::beg);
            in.read(&all[0], size);
            return zip_document_format(all);
        }
        if (text && type == file_content_type::TEXT_XML)
            return document_format::markup;
        if (text)
            return classify_text(head, filename);
        /* Short binary signatures ("BM", "MZ"...) can match the start of a text file, and
           UTF-16 text is not seen as text by the byte heuristics: look again. */
        if (head.size() >= 8 && head.compare(0, 8, "\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1") == 0)
            return document_format::legacy_office;
        const bool bom = head.compare(0, 2, "\xFF\xFE") == 0 || head.compare(0, 2, "\xFE\xFF") == 0 ||
                         utf16_without_bom(head) != 0;
        if (bom || ((type == file_content_type::UNKNOWN || type == file_content_type::IMAGE ||
                     type == file_content_type::EXECUTABLE || type == file_content_type::AUDIO) && is_text_head(head)))
            return classify_text(head, filename);
        return document_format::unknown;
    }

// ----------------------------------------------------------------------------------------

    class document_anonymizer
    {
    public:
        document_anonymizer(
            const pseudonymizer& p,
            pii_detector& detector
        ) : p_(p), detector_(detector) {}

        std::string anonymize(const std::string& bytes, document_format format, document_report& report,
                              const std::string& name_hint = "")
        {
            name_hint_ = name_hint;
            return run(bytes, format, report, false);
        }

        std::string restore(const std::string& bytes, document_format format, document_report& report,
                            const std::string& name_hint = "")
        {
            name_hint_ = name_hint;
            return run(bytes, format, report, true);
        }

        std::string anonymize_text(const std::string& utf8_text, document_report& report)
        {
            report.operation = "anonymize";
            report.format = "txt";
            report.key_fingerprint = p_.key_fingerprint();
            return plain_text(utf8_text, report, false);
        }

        void set_pdf_layout(bool keep) { pdf_layout_ = keep; }
        bool pdf_layout() const { return pdf_layout_; }

        void set_image_describer(const document_image_describer& describer) { describer_ = describer; }

        void set_package_limits(const zip_limits& limits) { package_limits_ = limits; }
        const zip_limits& package_limits() const { return package_limits_; }

        void set_untagged_properties(bool untagged) { untagged_properties_ = untagged; }
        bool untagged_properties() const { return untagged_properties_; }

        void set_substitute_resolver(const pseudonymizer::substitute_resolver& resolve) { resolve_ = resolve; }

        void set_untagged_text(bool untagged) { untagged_text_ = untagged; }
        bool untagged_text() const { return untagged_text_; }

        void set_text_substitute_resolver(const pseudonymizer::substitute_resolver& resolve)
        {
            text_resolve_ = resolve;
        }

        void set_ascii_windows_1252(bool yes) { ascii_1252_ = yes; }

        bool ascii_windows_1252() const { return ascii_1252_; }

        void set_imposed_columns(const std::vector<pii_imposed_column>& columns)
        {
            columns_.clear();
            column_keys_.clear();
            for (const pii_imposed_column& c : columns)
            {
                const std::string key = privacy_impl::fold_key(c.name);
                if (c.type == pii_type::none || key.empty()) continue;
                if (std::find(column_keys_.begin(), column_keys_.end(), key) != column_keys_.end()) continue;
                columns_.push_back(c);
                column_keys_.push_back(key);
            }
        }

        const std::vector<pii_imposed_column>& imposed_columns() const { return columns_; }

        void set_picture_text_finder(const document_picture_text_finder& find) { pictures_ = find; }

        std::string anonymize_name(const std::string& name, document_report& report)
        {
            if (name.empty()) return name;
            /* Words of a file name are often joined by underscores: they read as spaces, the
               underscores staying in the name and in the values replaced. */
            std::string seen = name;
            std::replace(seen.begin(), seen.end(), '_', ' ');
            // A substitute written before, a name protected and then edited, is not replaced again.
            const std::vector<pseudonymizer::substitute_occurrence> known = p_.find_substitutes(name, resolve_);
            std::vector<pii_entity> found;
            for (const pii_entity& e : detector_.detect(seen))
            {
                if (p_.keeps_in_clear(e.type)) continue;
                bool written = false;
                for (const auto& o : known)
                    if (e.begin < o.end && o.begin < e.end) written = true;
                if (written) continue;
                std::size_t b = e.begin, en = e.end;
                while (b < en && seen[b] == ' ') ++b;
                while (en > b && seen[en - 1] == ' ') --en;
                if (e.type == pii_type::person || e.type == pii_type::organization) b = past_opening_words(seen, b, en);
                if (en <= b) continue;
                pii_entity v = e;
                v.begin = b;
                v.end = en;
                found.push_back(v);
                report.count(e.type);
            }
            std::vector<pseudonymizer::substitution> made;
            const std::string out = p_.apply_substitutes(name, found, &made);
            for (const pseudonymizer::substitution& s : made)
                report.substitutions.push_back(document_substitution{ s.substitute, s.tag });
            return out;
        }

        std::string restore_name(const std::string& name, document_report& report) const
        {
            pseudonymizer::restore_report rr;
            const std::string out = p_.restore_substitutes(name, resolve_, &rr);
            report.substitutes_restored += rr.restored;
            return out;
        }

    private:
        struct edit
        {
            std::size_t b, e;
            std::string rep;
        };

        /* The protected form of a value of the text: its tag, or its substitute when the text is
           untagged, which report lists with the tag it stands for. */
        std::string text_value(pii_type type, const std::string& value, document_report& report) const
        {
            if (!untagged_text_) return p_.tag(type, value);
            document_substitution s;
            s.substitute = p_.substitute(type, value, &s.tag);
            s.text = true;
            report.substitutions.push_back(s);
            return s.substitute;
        }

        // A text with its entities replaced as text_value() replaces each one.
        std::string text_values(const std::string& text, const std::vector<pii_entity>& entities,
                                document_report& report) const
        {
            if (!untagged_text_) return p_.apply(text, entities);
            std::vector<pseudonymizer::substitution> made;
            const std::string out = p_.apply_substitutes(text, entities, &made);
            for (const pseudonymizer::substitution& s : made)
                report.substitutions.push_back(document_substitution{ s.substitute, s.tag, true });
            return out;
        }

        /* The entities of a text but those that cover a substitute the text resolver knows: an
           untagged text protected again keeps as they are the values it holds protected already. */
        std::vector<pii_entity> without_known(const std::string& text, std::vector<pii_entity> entities) const
        {
            if (!text_resolve_ || entities.empty()) return entities;
            const std::vector<pseudonymizer::substitute_occurrence> known = p_.find_substitutes(text, text_resolve_);
            if (known.empty()) return entities;
            std::vector<pii_entity> out;
            for (const pii_entity& e : entities)
            {
                bool written = false;
                for (const auto& o : known)
                    if (e.begin < o.end && o.begin < e.end) written = true;
                if (!written) out.push_back(e);
            }
            return out;
        }

        /* The entities of a text to protect: those found but those that cover a known substitute, and,
           without tags, the tags of a text protected before with them, whose values take substitutes. */
        std::vector<pii_entity> to_write(const std::string& text, std::vector<pii_entity> entities) const
        {
            entities = without_known(text, std::move(entities));
            return untagged_text_ ? p_.with_tag_entities(text, std::move(entities)) : entities;
        }

        // A restored text with the substitutes that the text resolver knows given back too.
        std::string with_text_substitutes(const std::string& text, document_report& report) const
        {
            if (!text_resolve_) return text;
            pseudonymizer::restore_report sr;
            const std::string out = p_.restore_substitutes(text, text_resolve_, &sr);
            report.substitutes_restored += sr.restored;
            return out;
        }

        /*
            The start of a value found in a name, past the common words that open it: "Contrat"
            in "Contrat Jean Dupont", which a name read without a sentence around it takes in.
            The last word of the value stays.
        */
        static std::size_t past_opening_words(const std::string& s, std::size_t b, std::size_t e)
        {
            while (b < e)
            {
                std::size_t w = b;
                while (w < e && s[w] != ' ') ++w;
                if (w >= e) break;
                const std::string key = privacy_impl::fold_key(s.substr(b, w - b));
                if (!privacy_impl::common_words().count(key) || privacy_impl::first_names().count(key)) break;
                b = w;
                while (b < e && s[b] == ' ') ++b;
            }
            return b;
        }

        std::string run(const std::string& bytes, document_format format, document_report& report, bool restoring)
        {
            using namespace privacy_impl;
            report.operation = restoring ? "restore" : "anonymize";
            report.key_fingerprint = p_.key_fingerprint();
            report.format = document_format_name(format);
            switch (format)
            {
                case document_format::docx:
                case document_format::xlsx:
                case document_format::pptx:
                {
                    ooxml_anonymizer a(p_, &detector_, report);
                    a.set_image_describer(describer_);
                    a.set_package_limits(package_limits_);
                    a.set_untagged_properties(untagged_properties_);
                    a.set_substitute_resolver(resolve_);
                    a.set_untagged_text(untagged_text_);
                    a.set_text_substitute_resolver(text_resolve_);
                    a.set_picture_text_finder(pictures_);
                    return restoring ? a.restore(bytes) : a.anonymize(bytes);
                }
                case document_format::pdf:
                {
                    pdf_anonymizer a(p_, &detector_, report, pdf_layout_);
                    a.set_image_describer(describer_);
                    a.set_untagged_properties(untagged_properties_);
                    a.set_substitute_resolver(resolve_);
                    a.set_untagged_text(untagged_text_);
                    a.set_text_substitute_resolver(text_resolve_);
                    a.set_picture_text_finder(pictures_);
                    return restoring ? a.restore(bytes) : a.anonymize(bytes);
                }
                case document_format::text:
                case document_format::delimited:
                case document_format::markup:
                {
                    text_encoding enc;
                    const std::string text = decode_text(bytes, enc);
                    if (enc == text_encoding::cp1252)
                        report.note("text decoded as Windows-1252 and written back in that encoding");
                    std::string out;
                    // A text whose header names an imposed column is a delimited file, a title above it.
                    if (format == document_format::text && imposed_delimiter(text, column_keys_, name_hint_))
                    {
                        format = document_format::delimited;
                        report.format = document_format_name(format);
                    }
                    if (format == document_format::delimited) out = delimited(text, report, restoring);
                    else if (format == document_format::markup) out = markup(text, report, restoring);
                    else out = plain_text(text, report, restoring);
                    const bool ascii = std::all_of(out.begin(), out.end(), [](char c) { return (c & 0x80) == 0; });
                    if (!restoring && enc == text_encoding::cp1252 && ascii)
                        report.note("the protected text is plain ASCII: its restoration will be written in UTF-8, "
                                    "unless the caller knows its original was written in Windows-1252");
                    if (restoring && enc == text_encoding::utf8 && !ascii &&
                        std::all_of(text.begin(), text.end(), [](char c) { return (c & 0x80) == 0; }))
                    {
                        /* A text of plain ASCII says nothing of the encoding of its original: the caller
                           that knows it was Windows-1252 has it written so, when that encoding writes
                           every character restored. */
                        if (ascii_1252_ && cp1252_writes(out))
                        {
                            enc = text_encoding::cp1252;
                            report.note("the restored text is written in Windows-1252, the encoding of its original");
                        }
                        else
                            report.note("the restored text is written in UTF-8, the encoding of ASCII input is "
                                        "unknown");
                    }
                    report.encoding = text_encoding_name(enc);
                    return encode_text(out, enc);
                }
                case document_format::legacy_office:
                    throw error("binary or password protected Office file: save it as DOCX, XLSX or PPTX without "
                                "password first");
                default:
                    throw error("unsupported document format");
            }
        }

    // ------------------------------------------------------------------------------------
    // Plain text: paragraphs separated by blank lines, cut at a line break when long.

        static std::vector<std::pair<std::size_t, std::size_t>> paragraphs(const std::string& s)
        {
            const std::size_t max_unit = 64 * 1024;
            std::vector<std::pair<std::size_t, std::size_t>> out;
            std::size_t start = 0, i = 0;
            std::size_t last_break = std::string::npos;
            while (i < s.size())
            {
                if (s[i] == '\n')
                {
                    // A blank line: only spaces until the next line break.
                    std::size_t j = i + 1;
                    while (j < s.size() && (s[j] == ' ' || s[j] == '\t' || s[j] == '\r')) ++j;
                    if (j < s.size() && s[j] == '\n')
                    {
                        if (i > start) out.push_back(std::make_pair(start, i));
                        while (j < s.size() && (s[j] == '\n' || s[j] == ' ' || s[j] == '\t' || s[j] == '\r')) ++j;
                        start = i = j;
                        last_break = std::string::npos;
                        continue;
                    }
                    last_break = i;
                }
                if (i - start >= max_unit && last_break != std::string::npos && last_break > start)
                {
                    out.push_back(std::make_pair(start, last_break));
                    start = last_break;
                    last_break = std::string::npos;
                }
                ++i;
            }
            if (start < s.size()) out.push_back(std::make_pair(start, s.size()));
            return out;
        }

        // The text without the picture texts that the finder gives, which leave with the tags they hold.
        std::string without_pictures(const std::string& text, document_report& report) const
        {
            if (!pictures_) return text;
            const std::vector<std::pair<std::size_t, std::size_t>> found = pictures_(text);
            if (found.empty()) return text;
            std::string out;
            std::size_t pos = 0;
            for (const auto& r : found)
            {
                if (r.first < pos || r.second > text.size() || r.second <= r.first) continue;
                out.append(text, pos, r.first - pos);
                pos = r.second;
                ++report.pictures_removed;
            }
            out.append(text, pos, std::string::npos);
            return out;
        }

        std::string plain_text(const std::string& text, document_report& report, bool restoring)
        {
            if (restoring)
            {
                pseudonymizer::restore_report rr;
                const std::string out = with_text_substitutes(p_.restore(without_pictures(text, report), &rr), report);
                report.tags_restored += rr.restored;
                report.tags_rejected += rr.rejected;
                report.text_units += 1;
                return out;
            }
            const std::vector<std::pair<std::size_t, std::size_t>> units = paragraphs(text);
            detector_.begin_document();
            for (const auto& u : units)
                detector_.learn(text.substr(u.first, u.second - u.first));
            std::string out;
            std::size_t pos = 0;
            for (const auto& u : units)
            {
                const std::string unit = text.substr(u.first, u.second - u.first);
                const std::vector<pii_entity> entities = to_write(unit, detector_.detect(unit));
                for (const pii_entity& e : entities)
                    if (!p_.keeps_in_clear(e.type)) report.count(e.type);
                out.append(text, pos, u.first - pos);
                out += text_values(unit, entities, report);
                pos = u.second;
            }
            out.append(text, pos, std::string::npos);
            report.text_units += units.size();
            return out;
        }

    // ------------------------------------------------------------------------------------
    // Delimiter-separated values.

        struct cell
        {
            std::size_t raw_begin, raw_end;     // content, quotes excluded
            bool quoted;
            std::size_t row, col;
            std::string value;
        };

        static bool parse_delimited(const std::string& text, char delim, std::vector<cell>& cells)
        {
            std::size_t i = 0, row = 0, col = 0;
            const std::size_t n = text.size();
            while (i < n)
            {
                cell c;
                c.row = row;
                c.col = col;
                if (text[i] == '"')
                {
                    c.quoted = true;
                    std::size_t j = i + 1;
                    for (;;)
                    {
                        if (j >= n) return false;
                        if (text[j] == '"')
                        {
                            if (j + 1 < n && text[j + 1] == '"') { c.value.push_back('"'); j += 2; continue; }
                            break;
                        }
                        c.value.push_back(text[j++]);
                    }
                    c.raw_begin = i + 1;
                    c.raw_end = j;
                    i = j + 1;
                    if (i < n && text[i] != delim && text[i] != '\n' && text[i] != '\r')
                        return false;
                }
                else
                {
                    std::size_t j = i;
                    while (j < n && text[j] != delim && text[j] != '\n' && text[j] != '\r') ++j;
                    c.quoted = false;
                    c.raw_begin = i;
                    c.raw_end = j;
                    c.value = text.substr(i, j - i);
                    i = j;
                }
                cells.push_back(c);
                if (i >= n) break;
                if (text[i] == delim)
                {
                    ++i;
                    ++col;
                    if (i == n || text[i] == '\n' || text[i] == '\r')
                    {
                        cell empty;
                        empty.raw_begin = empty.raw_end = i;
                        empty.quoted = false;
                        empty.row = row;
                        empty.col = col;
                        cells.push_back(empty);
                    }
                    else
                    {
                        continue;
                    }
                }
                if (i < n && text[i] == '\r') ++i;
                if (i < n && text[i] == '\n') ++i;
                ++row;
                col = 0;
            }
            return true;
        }

        // The type of the imposed column that a cell of a header names, or none.
        pii_type column_named(const std::string& value) const
        {
            if (columns_.empty()) return pii_type::none;
            const std::string key = privacy_impl::fold_key(value);
            for (std::size_t i = 0; i < column_keys_.size(); ++i)
                if (column_keys_[i] == key) return columns_[i].type;
            return pii_type::none;
        }

        // The bounds of a value of a cell without the spaces and the line breaks at its ends.
        static std::pair<std::size_t, std::size_t> value_bounds(const std::string& v)
        {
            const std::size_t b = v.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return std::make_pair(v.size(), v.size());
            return std::make_pair(b, v.find_last_not_of(" \t\r\n") + 1);
        }

        // Whether a value of type known stands in a column of type: an IP column takes both forms.
        static bool column_family(pii_type known, pii_type type)
        {
            const auto ip = [](pii_type t) { return t == pii_type::ipv4 || t == pii_type::ipv6; };
            return known == type || (ip(known) && ip(type));
        }

        // Whether s is an address of the form given, ipv4 or ipv6.
        static bool address_of_form(const std::string& s, pii_type form)
        {
            std::uint32_t a4 = 0;
            std::array<unsigned char, 16> a6;
            return form == pii_type::ipv4 ? privacy_impl::parse_ipv4(s, a4) : privacy_impl::parse_ipv6(s, a6);
        }

        // Whether the text resolver knows candidate as a substitute of the family of type, and its value.
        bool known_substitute(const std::string& candidate, pii_type type, std::string* original) const
        {
            if (!text_resolve_ || candidate.empty()) return false;
            for (const std::string& tag : text_resolve_(candidate))
            {
                pii_type known = pii_type::none;
                std::string value;
                if (!p_.decode(tag, known, value) || !column_family(known, type)) continue;
                if (original) *original = value;
                return true;
            }
            return false;
        }

        /*
            The protected form of the value of a cell of an imposed column: its substitute, with or
            without untagged text, the spaces at its ends kept; listed in report.substitutions once
            for each value, made is the substitute of each value already written.  A substitute
            that the text resolver knows for the type, written by a protection before, stays, and
            a tag stands for its value.  An IP column takes the IPv4 and the IPv6 addresses alike,
            each by its form, and keeps what surrounds the address: a prefix length, a zone, the
            brackets and the port.  An address whose substitute is no address, written with
            capitals or leading zeros, takes an opaque code with what surrounds it, so that the
            restoration finds that code whole.
        */
        std::string column_value(const std::string& value, pii_type type, document_report& report,
                                 std::unordered_map<std::string, std::string>& made) const
        {
            const std::pair<std::size_t, std::size_t> r = value_bounds(value);
            if (r.first >= r.second || p_.keeps_in_clear(type)) return value;
            std::string core = value.substr(r.first, r.second - r.first);
            if (known_substitute(core, type, nullptr)) return value;
            pii_type tagged = pii_type::none;
            std::string original;
            if (p_.decode(core, tagged, original)) core = original;
            std::size_t ab = 0, ae = core.size();
            pii_type form = type;
            if (type == pii_type::ipv4 || type == pii_type::ipv6)
            {
                std::size_t b = 0, e = 0;
                const pii_type found = privacy_impl::ip_address_in(core, b, e);
                if (found != pii_type::none)
                {
                    const std::string part = core.substr(b, e - b);
                    if (known_substitute(part, type, nullptr)) return value;
                    form = found;
                    if (address_of_form(p_.substitute(found, part), found))
                    {
                        ab = b;
                        ae = e;
                    }
                }
            }
            const std::string address = core.substr(ab, ae - ab);
            const std::string before = value.substr(0, r.first) + core.substr(0, ab);
            const std::string after = core.substr(ae) + value.substr(r.second);
            report.count(form);
            // A civility, a sex or a nationality takes another substitute at each place, as its tags do.
            const bool varying = form == pii_type::title || form == pii_type::gender || form == pii_type::nationality;
            const std::string key = std::string(pii_type_keyword(form)) + '\x1f' + address;
            if (!varying)
            {
                const auto it = made.find(key);
                if (it != made.end()) return before + it->second + after;
            }
            document_substitution s;
            s.substitute = p_.substitute(form, address, &s.tag);
            s.text = true;
            report.substitutions.push_back(s);
            if (!varying && made.size() < 262144) made[key] = s.substitute;
            return before + s.substitute + after;
        }

        /*
            The value of a cell of an imposed column that the text resolver knows as a substitute of
            its type, whole or, in an IP column, as the address that a prefix length, a zone or a port
            surrounds.
        */
        bool column_original(const std::string& value, pii_type type, std::string& out, document_report& report) const
        {
            const std::pair<std::size_t, std::size_t> r = value_bounds(value);
            if (r.first >= r.second || !text_resolve_) return false;
            const std::string core = value.substr(r.first, r.second - r.first);
            std::size_t ab = 0, ae = core.size();
            std::string original;
            bool found = known_substitute(core, type, &original);
            if (!found && (type == pii_type::ipv4 || type == pii_type::ipv6))
            {
                std::size_t b = 0, e = 0;
                if (privacy_impl::ip_address_in(core, b, e) != pii_type::none &&
                    known_substitute(core.substr(b, e - b), type, &original))
                {
                    found = true;
                    ab = b;
                    ae = e;
                }
            }
            if (!found) return false;
            out = value.substr(0, r.first) + core.substr(0, ab) + original + core.substr(ae) + value.substr(r.second);
            ++report.substitutes_restored;
            return true;
        }

        std::string delimited(const std::string& text, document_report& report, bool restoring)
        {
            const std::string hint = name_hint_.empty() ? ".csv" : name_hint_;
            char delim = privacy_impl::imposed_delimiter(text, column_keys_, hint);
            if (!delim) delim = privacy_impl::sniff_delimiter(text, hint);
            std::vector<cell> cells;
            if (!delim || !parse_delimited(text, delim, cells))
            {
                report.note("the delimited structure could not be parsed; the file was processed as plain text");
                report.format = "txt";
                return plain_text(text, report, restoring);
            }

            /* The imposed columns, named by a header among the first 16 rows: every value below
               their names is personal data of their types, replaced whole, without detection. */
            std::vector<pii_type> imposed(cells.size(), pii_type::none);
            std::vector<bool> names(cells.size(), false);
            std::size_t first = 0;
            bool named = false;
            {
                std::map<std::size_t, pii_type> typed;
                for (std::size_t k = 0; k < cells.size() && cells[k].row < 16; ++k)
                {
                    if (named && cells[k].row != first) break;
                    const pii_type t = column_named(cells[k].value);
                    if (t == pii_type::none) continue;
                    if (!named) first = cells[k].row;
                    named = true;
                    names[k] = true;
                    typed[cells[k].col] = t;
                }
                for (std::size_t k = 0; named && k < cells.size(); ++k)
                {
                    const auto it = typed.find(cells[k].col);
                    if (cells[k].row > first && it != typed.end()) imposed[k] = it->second;
                }
            }

            std::vector<std::string> values(cells.size());
            if (restoring)
            {
                pseudonymizer::restore_report rr;
                for (std::size_t k = 0; k < cells.size(); ++k)
                {
                    if (imposed[k] != pii_type::none && column_original(cells[k].value, imposed[k], values[k], report))
                        continue;
                    values[k] =
                        with_text_substitutes(p_.restore(without_pictures(cells[k].value, report), &rr), report);
                }
                report.tags_restored += rr.restored;
                report.tags_rejected += rr.rejected;
            }
            else
            {
                /* Column types from a header row, made of labels or naming an imposed column; a
                   label on the left of a cell types it too, as in key and value files
                   ("Nom;Dupont"). */
                std::vector<pii_type> labels(cells.size(), pii_type::none);
                for (std::size_t k = 0; k < cells.size(); ++k)
                    if (imposed[k] == pii_type::none) labels[k] = detector_.label_type(cells[k].value);
                std::map<std::size_t, pii_type> columns;
                std::size_t header_cells = 0, header_labels = 0;
                for (std::size_t k = 0; k < cells.size() && cells[k].row <= first; ++k)
                {
                    if (cells[k].row != first || cells[k].value.empty()) continue;
                    ++header_cells;
                    if (labels[k] != pii_type::none) { columns[cells[k].col] = labels[k]; ++header_labels; }
                }
                const bool header = named || header_labels >= 2 || (header_labels == 1 && header_cells == 1);
                // A name of a header is no value of the label on its left.
                auto context_of = [&](std::size_t k) {
                    detection_context ctx;
                    const cell& c = cells[k];
                    if (k > 0 && cells[k - 1].row == c.row && labels[k - 1] != pii_type::none &&
                        labels[k] == pii_type::none && !(header && c.row == first))
                        ctx.label = labels[k - 1];
                    else if (header && c.row > first)
                    {
                        const auto it = columns.find(c.col);
                        if (it != columns.end()) ctx.label = it->second;
                    }
                    return ctx;
                };
                /* Learning reads whole rows, where names have their context; detection
                   runs cell by cell, so that a value never spills over a column.  The values of
                   the imposed columns are not read: each is recorded once as a value of its
                   type (pii_detector::learn_value()), so that the other cells give it up too. */
                detector_.begin_document();
                std::string row_text;
                std::size_t row = 0;
                std::unordered_set<std::string> learned;
                for (std::size_t k = 0; k < cells.size(); ++k)
                {
                    const cell& c = cells[k];
                    if (c.row != row) { detector_.learn(row_text); row_text.clear(); row = c.row; }
                    if (imposed[k] == pii_type::none)
                    {
                        row_text += (row_text.empty() ? "" : "\t") + c.value;
                        detector_.learn(c.value, context_of(k));
                        continue;
                    }
                    const std::pair<std::size_t, std::size_t> r = value_bounds(c.value);
                    if (r.first >= r.second) continue;
                    const std::string v = c.value.substr(r.first, r.second - r.first);
                    if (learned.size() >= 262144 || learned.insert(pii_type_keyword(imposed[k]) + ('\x1f' + v)).second)
                        detector_.learn_value(v, imposed[k]);
                }
                detector_.learn(row_text);
                std::unordered_map<std::string, std::string> made;
                for (std::size_t k = 0; k < cells.size(); ++k)
                {
                    const cell& c = cells[k];
                    // The names of the imposed columns stay as they are, the header that tells them.
                    if (names[k])
                    {
                        values[k] = c.value;
                        continue;
                    }
                    if (imposed[k] != pii_type::none)
                    {
                        values[k] = column_value(c.value, imposed[k], report, made);
                        continue;
                    }
                    // Labels stay as they are; every other cell is read, header row included.
                    if (labels[k] != pii_type::none && (c.row <= first || context_of(k).label == pii_type::none))
                    {
                        values[k] = c.value;
                        continue;
                    }
                    const std::vector<pii_entity> entities =
                        to_write(c.value, detector_.detect(c.value, context_of(k)));
                    for (const pii_entity& e : entities)
                        if (!p_.keeps_in_clear(e.type)) report.count(e.type);
                    values[k] = text_values(c.value, entities, report);
                }
            }
            report.text_units += cells.size();

            std::string out;
            std::size_t pos = 0;
            for (std::size_t k = 0; k < cells.size(); ++k)
            {
                const cell& c = cells[k];
                if (values[k] == c.value) continue;
                /* A cell written without quotes keeps the quotes it holds as they are, as it was read,
                   unless its new value needs quotes: a delimiter, a line break or a quote at its start. */
                bool quote = c.quoted || (!values[k].empty() && values[k][0] == '"');
                for (char ch : values[k])
                    if (ch == delim || ch == '\n' || ch == '\r') quote = true;
                std::string raw;
                for (char ch : values[k])
                {
                    if (ch == '"' && quote) raw += "\"\"";
                    else raw.push_back(ch);
                }
                out.append(text, pos, c.raw_begin - pos);
                if (quote && !c.quoted) out += "\"" + raw + "\"";
                else out += raw;
                pos = c.raw_end;
            }
            out.append(text, pos, std::string::npos);
            return out;
        }

    // ------------------------------------------------------------------------------------
    // Markup: character data and attribute values only.

        static std::string lower_ascii(std::string s)
        {
            for (char& c : s) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            return s;
        }

        static bool inline_element(const std::string& local)
        {
            static const char* const names[] = { "a",    "abbr", "b",     "bdi",  "bdo",    "big",    "cite", "code",
                                                 "data", "dfn",  "em",    "font", "i",      "kbd",    "mark", "q",
                                                 "s",    "samp", "small", "span", "strike", "strong", "sub",  "sup",
                                                 "time", "tt",   "u",     "var",  "wbr",    "ins",    "del",  "label" };
            for (const char* n : names)
                if (local == n) return true;
            return false;
        }

        static bool skipped_attribute(const std::string& qname)
        {
            const std::string q = lower_ascii(qname);
            if (q == "xmlns" || q.compare(0, 6, "xmlns:") == 0 || q.compare(0, 4, "xml:") == 0) return true;
            static const char* const names[] = { "id",
                                                 "class",
                                                 "style",
                                                 "type",
                                                 "rel",
                                                 "lang",
                                                 "dir",
                                                 "width",
                                                 "height",
                                                 "colspan",
                                                 "rowspan",
                                                 "align",
                                                 "valign",
                                                 "border",
                                                 "cellpadding",
                                                 "cellspacing",
                                                 "charset",
                                                 "encoding",
                                                 "version",
                                                 "standalone",
                                                 "method",
                                                 "target",
                                                 "media",
                                                 "sizes",
                                                 "srcset",
                                                 "role",
                                                 "tabindex",
                                                 "for",
                                                 "http-equiv",
                                                 "name",
                                                 "viewbox",
                                                 "transform",
                                                 "fill",
                                                 "stroke",
                                                 "d",
                                                 "x",
                                                 "y",
                                                 "cx",
                                                 "cy",
                                                 "r",
                                                 "points",
                                                 "font-family",
                                                 "font-size",
                                                 "color",
                                                 "bgcolor",
                                                 "face",
                                                 "size",
                                                 "span",
                                                 "start",
                                                 "dx",
                                                 "dy",
                                                 "rx",
                                                 "ry",
                                                 "x1",
                                                 "x2",
                                                 "y1",
                                                 "y2",
                                                 "offset",
                                                 "stop-color",
                                                 "opacity",
                                                 "preserveaspectratio",
                                                 "crossorigin",
                                                 "integrity",
                                                 "async",
                                                 "defer",
                                                 "nonce",
                                                 "referrerpolicy",
                                                 "loading",
                                                 "decoding" };
            const std::string local = lower_ascii(xml_local_name(qname));
            for (const char* n : names)
                if (local == n) return true;
            return false;
        }

        // Namespace declarations and xml: attributes, never personal data.
        static bool namespace_attribute(const std::string& qname)
        {
            const std::string q = lower_ascii(qname);
            return q == "xmlns" || q.compare(0, 6, "xmlns:") == 0 || q.compare(0, 4, "xml:") == 0;
        }

        static bool uri_attribute(const std::string& qname)
        {
            const std::string local = lower_ascii(xml_local_name(qname));
            return local == "href" || local == "src" || local == "action" || local == "cite" || local == "data" ||
                   local == "poster" || local == "background" || local == "formaction" || local == "longdesc";
        }

        /* Character data with the five XML entities, numeric references and the HTML
           entities of the Latin repertoire decoded. When map is given, (*map)[k] is the
           offset in raw of the k-th byte of the result, plus a final entry raw.size():
           the bytes of a decoded reference all map to the reference. */
        static std::string markup_unescape(const std::string& raw, std::vector<std::size_t>* map = nullptr)
        {
            static const std::map<std::string, std::uint32_t> named = {
                { "amp", '&' },       { "lt", '<' },       { "gt", '>' },       { "quot", '"' },
                { "apos", '\'' },     { "nbsp", 0xA0 },    { "eacute", 0xE9 },  { "egrave", 0xE8 },
                { "ecirc", 0xEA },    { "euml", 0xEB },    { "agrave", 0xE0 },  { "aacute", 0xE1 },
                { "acirc", 0xE2 },    { "auml", 0xE4 },    { "atilde", 0xE3 },  { "aring", 0xE5 },
                { "ccedil", 0xE7 },   { "igrave", 0xEC },  { "iacute", 0xED },  { "icirc", 0xEE },
                { "iuml", 0xEF },     { "ograve", 0xF2 },  { "oacute", 0xF3 },  { "ocirc", 0xF4 },
                { "ouml", 0xF6 },     { "otilde", 0xF5 },  { "oslash", 0xF8 },  { "ugrave", 0xF9 },
                { "uacute", 0xFA },   { "ucirc", 0xFB },   { "uuml", 0xFC },    { "yacute", 0xFD },
                { "yuml", 0xFF },     { "ntilde", 0xF1 },  { "szlig", 0xDF },   { "aelig", 0xE6 },
                { "oelig", 0x153 },   { "Eacute", 0xC9 },  { "Egrave", 0xC8 },  { "Ecirc", 0xCA },
                { "Euml", 0xCB },     { "Agrave", 0xC0 },  { "Aacute", 0xC1 },  { "Acirc", 0xC2 },
                { "Auml", 0xC4 },     { "Ccedil", 0xC7 },  { "Icirc", 0xCE },   { "Iuml", 0xCF },
                { "Ocirc", 0xD4 },    { "Ouml", 0xD6 },    { "Ugrave", 0xD9 },  { "Ucirc", 0xDB },
                { "Uuml", 0xDC },     { "Ntilde", 0xD1 },  { "OElig", 0x152 },  { "AElig", 0xC6 },
                { "laquo", 0xAB },    { "raquo", 0xBB },   { "rsquo", 0x2019 }, { "lsquo", 0x2018 },
                { "ldquo", 0x201C },  { "rdquo", 0x201D }, { "ndash", 0x2013 }, { "mdash", 0x2014 },
                { "hellip", 0x2026 }, { "euro", 0x20AC },  { "copy", 0xA9 },    { "reg", 0xAE },
                { "deg", 0xB0 },      { "middot", 0xB7 },  { "bull", 0x2022 },  { "sect", 0xA7 },
                { "para", 0xB6 },     { "times", 0xD7 },   { "shy", 0xAD },     { "thinsp", 0x2009 }
            };
            std::string out;
            if (map) map->clear();
            for (std::size_t i = 0; i < raw.size(); ++i)
            {
                if (raw[i] == '&')
                {
                    std::size_t semi = std::string::npos;
                    for (std::size_t k = i + 1; k < raw.size() && k <= i + 32; ++k)
                    {
                        if (raw[k] == ';') { semi = k; break; }
                        if (raw[k] == '&' || raw[k] == '<') break;
                    }
                    if (semi != std::string::npos)
                    {
                        const std::string ent = raw.substr(i + 1, semi - i - 1);
                        std::uint32_t cp = 0;
                        if (!ent.empty() && ent[0] == '#')
                            cp = static_cast<std::uint32_t>(dlib::impl::xml_reference(ent));
                        else
                        {
                            const auto it = named.find(ent);
                            if (it != named.end()) cp = it->second;
                        }
                        if (cp != 0)
                        {
                            const std::size_t before = out.size();
                            privacy_impl::utf8_append(out, static_cast<char32_t>(cp));
                            if (map) map->insert(map->end(), out.size() - before, i);
                            i = semi;
                            continue;
                        }
                    }
                }
                out.push_back(raw[i]);
                if (map) map->push_back(i);
            }
            if (map) map->push_back(raw.size());
            return out;
        }

        struct markup_piece
        {
            std::size_t token;                  // text, CDATA or comment token
            std::size_t ub, ue;                 // range in the unit text
            std::size_t raw_begin, raw_end;     // the raw range it reads
            int mode;                           // 0 character data, 1 CDATA, 2 comment
        };

        // HTML and SVG documents, whose attributes include a known set of presentational ones.
        static bool html_like(const std::string& text)
        {
            const std::string head = lower_ascii(text.substr(0, 4096));
            if (head.find("<!doctype html") != std::string::npos || head.find("<html") != std::string::npos ||
                head.find("<svg") != std::string::npos || head.find("http://www.w3.org/1999/xhtml") != std::string::npos)
                return true;
            return false;
        }

        struct markup_unit
        {
            std::string text;
            std::vector<markup_piece> pieces;
        };

        std::vector<edit> unit_edits(const std::string& text, document_report& report, bool restoring) const
        {
            std::vector<edit> edits;
            if (restoring)
            {
                // The texts written in the place of pictures leave, with the tags they hold.
                std::vector<std::pair<std::size_t, std::size_t>> pictures;
                if (pictures_) pictures = pictures_(text);
                for (const auto& r : pictures)
                    if (r.first < r.second && r.second <= text.size())
                    {
                        edits.push_back(edit{ r.first, r.second, std::string() });
                        ++report.pictures_removed;
                    }
                pseudonymizer::restore_report rr;
                std::size_t inside = 0;
                for (const auto& o : p_.find_tags(text, &rr))
                {
                    bool covered = false;
                    for (const auto& r : pictures)
                        if (o.begin < r.second && r.first < o.end) covered = true;
                    if (covered) ++inside;
                    else edits.push_back(edit{ o.begin, o.end, o.original });
                }
                report.tags_restored += rr.restored - std::min(inside, rr.restored);
                report.tags_rejected += rr.rejected;
                // The substitutes of an untagged text, outside the tags and the pictures.
                if (text_resolve_)
                    for (const auto& o : p_.find_substitutes(text, text_resolve_))
                    {
                        bool covered = false;
                        for (const edit& ed : edits)
                            if (o.begin < ed.e && ed.b < o.end) covered = true;
                        if (covered) continue;
                        edits.push_back(edit{ o.begin, o.end, o.original });
                        ++report.substitutes_restored;
                    }
                std::sort(edits.begin(), edits.end(), [](const edit& a, const edit& b) { return a.b < b.b; });
            }
            else
            {
                for (const pii_entity& e : to_write(text, detector_.detect(text)))
                {
                    if (p_.keeps_in_clear(e.type)) continue;
                    edits.push_back(edit{ e.begin, e.end, text_value(e.type, text.substr(e.begin, e.end - e.begin),
                                                                     report) });
                    report.count(e.type);
                }
            }
            return edits;
        }

        std::string markup(const std::string& text, document_report& report, bool restoring)
        {
            /* HTML is read with the tolerance of HTML parsers; other markup must be
               well-formed XML, and is read the same way with a note when it is not. */
            const bool html = html_like(text);
            std::vector<xml_token> tokens;
            if (html)
            {
                tokens = html_tokenize(text);
            }
            else
            {
                try
                {
                    tokens = xml_tokenize(text);
                }
                catch (xml_error&)
                {
                    report.note("the markup is not well-formed XML; it was read with the tolerance of HTML parsers");
                    tokens = html_tokenize(text);
                }
            }

            /* Character data split by inline elements forms one unit. Script and style
               contents are read too: a name in a string literal is still a name. Each
               comment is a unit of its own. */
            std::vector<markup_unit> units(1);
            for (std::size_t k = 0; k < tokens.size(); ++k)
            {
                const xml_token& t = tokens[k];
                if (t.kind == xml_token::text || t.kind == xml_token::cdata)
                {
                    const std::string raw = text.substr(t.content_begin, t.content_end - t.content_begin);
                    const std::string value = t.kind == xml_token::cdata ? raw : markup_unescape(raw);
                    markup_unit& u = units.back();
                    const std::size_t b = u.text.size();
                    u.text += value;
                    u.pieces.push_back(markup_piece{ k, b, u.text.size(), t.content_begin, t.content_end,
                                                     t.kind == xml_token::cdata ? 1 : 0 });
                    continue;
                }
                if (t.kind == xml_token::start_tag || t.kind == xml_token::end_tag || t.kind == xml_token::empty_tag)
                {
                    const std::string local = lower_ascii(xml_local_name(t.name));
                    if (!inline_element(local) && !units.back().pieces.empty())
                        units.push_back(markup_unit());
                    continue;
                }
                if (!units.back().pieces.empty())
                    units.push_back(markup_unit());
                if (t.kind == xml_token::comment && t.end >= t.begin + 7 && text.compare(t.end - 3, 3, "-->") == 0)
                {
                    markup_unit u;
                    u.text = text.substr(t.begin + 4, t.end - 3 - (t.begin + 4));
                    u.pieces.push_back(markup_piece{ k, 0, u.text.size(), t.begin + 4, t.end - 3, 2 });
                    units.push_back(u);
                    units.push_back(markup_unit());
                }
            }

            // Attribute values, each a unit of its own.
            struct attribute_unit { std::size_t b, e; bool uri; std::string value; };
            std::vector<attribute_unit> attributes;
            for (const xml_token& t : tokens)
            {
                if (t.kind != xml_token::start_tag && t.kind != xml_token::empty_tag) continue;
                for (const xml_attribute& a : t.attributes)
                {
                    if ((html ? skipped_attribute(a.name) : namespace_attribute(a.name)) || a.value_end <= a.value_begin)
                        continue;
                    attributes.push_back(attribute_unit{ a.value_begin, a.value_end, uri_attribute(a.name),
                        markup_unescape(text.substr(a.value_begin, a.value_end - a.value_begin)) });
                }
            }

            if (!restoring)
            {
                detector_.begin_document();
                for (const markup_unit& u : units)
                    if (!u.text.empty()) detector_.learn(u.text);
                for (const attribute_unit& a : attributes)
                    detector_.learn(a.value);
            }

            xml_edit_list edits;
            for (const markup_unit& u : units)
            {
                if (u.text.empty()) continue;
                ++report.text_units;
                const std::vector<edit> found = unit_edits(u.text, report, restoring);
                if (found.empty()) continue;
                const std::vector<std::vector<edit>> local = split_edits(u, found);
                std::vector<std::string> written(u.pieces.size());
                for (std::size_t k = 0; k < u.pieces.size(); ++k)
                {
                    const markup_piece& pc = u.pieces[k];
                    const std::string raw = text.substr(pc.raw_begin, pc.raw_end - pc.raw_begin);
                    written[k] = local[k].empty() ? raw : rewrite_raw(raw, pc.mode, local[k]);
                }
                /* A CDATA section that anonymize() cut in two, because a tag ended right
                   before a '>', is joined again once the tag is restored, so that the
                   restored document is the original one byte for byte. */
                for (std::size_t k = 0; k < u.pieces.size();)
                {
                    std::size_t last = k;
                    std::string joined = written[k];
                    while (restoring && last + 1 < u.pieces.size() &&
                           split_cdata(text, u.pieces[last], u.pieces[last + 1]) &&
                           (joined + written[last + 1]).find("]]>") == std::string::npos)
                    {
                        joined += written[last + 1];
                        ++last;
                    }
                    if (last > k)
                        edits.replace(u.pieces[k].raw_begin, u.pieces[last].raw_end, joined);
                    else if (!local[k].empty())
                        edits.replace(u.pieces[k].raw_begin, u.pieces[k].raw_end, written[k]);
                    k = last + 1;
                }
            }
            for (const attribute_unit& a : attributes)
            {
                ++report.text_units;
                std::string value;
                if (restoring)
                {
                    pseudonymizer::restore_report rr;
                    value = with_text_substitutes(p_.restore(without_pictures(a.value, report), &rr), report);
                    report.tags_restored += rr.restored;
                    report.tags_rejected += rr.rejected;
                }
                else
                {
                    const std::vector<pii_entity> entities = to_write(a.value, detector_.detect(a.value));
                    std::size_t pos = 0;
                    for (const pii_entity& e : entities)
                    {
                        if (p_.keeps_in_clear(e.type)) continue;
                        report.count(e.type);
                        value.append(a.value, pos, e.begin - pos);
                        const std::string written =
                            text_value(e.type, a.value.substr(e.begin, e.end - e.begin), report);
                        if (a.uri)
                            value += untagged_text_ ? percent_encode_substitute(written) : percent_encode_tag(written);
                        else
                            value += written;
                        pos = e.end;
                    }
                    value.append(a.value, pos, std::string::npos);
                }
                if (value != a.value)
                    edits.replace(a.b, a.e, xml_escape_attribute(value));
            }
            return edits.apply(text);
        }

        /* The edits of a unit cut to its pieces, in piece coordinates. Each replacement
           goes to the piece holding the first character of its edit; the other pieces it
           covers only lose their characters. */
        static std::vector<std::vector<edit>> split_edits(const markup_unit& u, const std::vector<edit>& edits)
        {
            std::vector<std::vector<edit>> out(u.pieces.size());
            for (const edit& e : edits)
            {
                bool placed = false;
                for (std::size_t k = 0; k < u.pieces.size(); ++k)
                {
                    const markup_piece& pc = u.pieces[k];
                    if (pc.ue <= e.b || e.e <= pc.ub) continue;
                    const std::size_t b = std::max(e.b, pc.ub) - pc.ub;
                    const std::size_t end = std::min(e.e, pc.ue) - pc.ub;
                    out[k].push_back(edit{ b, end, placed ? std::string() : e.rep });
                    placed = true;
                }
            }
            return out;
        }

        /* Two CDATA pieces of a unit separated by "]]><![CDATA[", the first ending with "]]"
           and the second starting with '>': the cut anonymize() makes after a tag. */
        static bool split_cdata(const std::string& text, const markup_piece& a, const markup_piece& b)
        {
            static const std::string seam = "]]><![CDATA[";
            return a.mode == 1 && b.mode == 1 && b.raw_begin == a.raw_end + seam.size() &&
                   text.compare(a.raw_end, seam.size(), seam) == 0 && a.raw_end >= a.raw_begin + 2 &&
                   text.compare(a.raw_end - 2, 2, "]]") == 0 && b.raw_end > b.raw_begin && text[b.raw_begin] == '>';
        }

        /* Applies piece edits to the raw character data: only the edited ranges are
           rewritten, so entities and character references elsewhere stay as written.
           mode is 0 for character data, 1 for a CDATA section, 2 for a comment. */
        static std::string rewrite_raw(const std::string& raw, int mode, const std::vector<edit>& local)
        {
            std::vector<std::size_t> map;
            if (mode != 0)
            {
                for (std::size_t i = 0; i <= raw.size(); ++i) map.push_back(i);
            }
            else
            {
                markup_unescape(raw, &map);
            }
            std::string out;
            std::size_t pos = 0;
            std::vector<std::size_t> ends;
            for (const edit& e : local)
            {
                const std::size_t rb = map[e.b], re = map[e.e];
                if (rb < pos) continue;
                out.append(raw, pos, rb - pos);
                out += mode != 0 ? e.rep : xml_escape_text(e.rep);
                ends.push_back(out.size());
                pos = re;
            }
            out.append(raw, pos, std::string::npos);
            if (mode == 0)
            {
                /* Character data cannot hold "]]>": a tag followed by a '>' written as is
                   ("Dupont &lt;jean.dupont@acme.fr>") ends with the reference of its last
                   bracket, which every parser reads back as a bracket. */
                for (auto it = ends.rbegin(); it != ends.rend(); ++it)
                {
                    const std::size_t p = *it;
                    if (p == 0 || out[p - 1] != ']') continue;
                    const bool closes = (p >= 2 && out[p - 2] == ']' && out.compare(p, 1, ">") == 0) ||
                                        out.compare(p, 2, "]>") == 0;
                    if (closes) out.replace(p - 1, 1, "&#93;");
                }
            }
            if (mode == 2)
            {
                // A comment cannot hold "--".
                std::string::size_type at;
                while ((at = out.find("--")) != std::string::npos) out.replace(at, 2, "- -");
            }
            if (mode == 1)
            {
                // A replacement ending in "]]" must not close the section early.
                std::string safe;
                std::size_t from = 0, at;
                while ((at = out.find("]]>", from)) != std::string::npos)
                {
                    safe.append(out, from, at - from);
                    safe += "]]]]><![CDATA[>";
                    from = at + 3;
                }
                safe.append(out, from, std::string::npos);
                out.swap(safe);
            }
            return out;
        }

        const pseudonymizer& p_;
        pii_detector& detector_;
        std::string name_hint_;
        bool pdf_layout_ = true;
        document_image_describer describer_;
        zip_limits package_limits_;
        bool untagged_properties_ = false;
        pseudonymizer::substitute_resolver resolve_;
        bool untagged_text_ = false;
        pseudonymizer::substitute_resolver text_resolve_;
        document_picture_text_finder pictures_;
        std::vector<pii_imposed_column> columns_;
        // fold_key() of the names of columns_, in the same order.
        std::vector<std::string> column_keys_;
        bool ascii_1252_ = false;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DOCUMENT_ANONYMIZER_H_
