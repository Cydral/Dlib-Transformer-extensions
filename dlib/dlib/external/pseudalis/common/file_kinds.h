// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_FILE_KINDS_H_
#define PSEUDALIS_COMMON_FILE_KINDS_H_

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

/*
    The kinds of files Pseudalis protects, from their names: the engine checks the content
    against the kind of the name, an Office document, a PDF, a text, a picture or an archive.
    A name whose extension gives no kind is taken for a text when its content is text through
    and through, which the engine decides with dlib; the drive and the shell foresee it from the
    start of the file (text_head()), and leave the other files at once.  A name without
    extension goes to the engine, which takes its content for what it is.
*/

namespace pseudalis
{
    enum class name_kind { unsupported, document, picture, archive };

    /*
        The extension of a name in lower case: ".docx", ".tar.gz", or an empty string.  What follows
        the last dot is an extension when it holds 1 to 16 letters, digits, '-' or '_': "Lettre à
        M. Dupont" has none.
    */
    inline std::string name_extension(const std::string& name)
    {
        const size_t slash = name.find_last_of("/\\");
        std::string base = slash == std::string::npos ? name : name.substr(slash + 1);
        for (char& c : base) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (base.size() > 7 && base.compare(base.size() - 7, 7, ".tar.gz") == 0) return ".tar.gz";
        const size_t dot = base.rfind('.');
        if (dot == std::string::npos || dot == 0 || dot + 1 == base.size() || base.size() - dot - 1 > 16)
            return std::string();
        for (size_t i = dot + 1; i < base.size(); ++i)
        {
            const unsigned char c = static_cast<unsigned char>(base[i]);
            if (c >= 0x80 || (!std::isalnum(c) && c != '-' && c != '_')) return std::string();
        }
        return base.substr(dot);
    }

    // The families of the documents: a text (plain, delimited or markup), an Office document, a PDF.
    enum class document_family { none, text, office, pdf };

    namespace file_kinds_impl
    {
        const char* const texts[] = {
            ".txt", ".text", ".log", ".md", ".markdown", ".rst", ".csv", ".tsv", ".tab", ".json", ".jsonl", ".ndjson",
            ".xml", ".html", ".htm", ".xhtml", ".svg", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
            ".properties", ".sql", ".tex", ".srt", ".vtt" };
        const char* const offices[] = {
            ".docx", ".docm", ".dotx", ".dotm", ".xlsx", ".xlsm", ".xltx", ".xltm", ".pptx", ".pptm", ".potx", ".potm",
            ".ppsx", ".ppsm" };
        const char* const pictures[] = {
            ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".bmp", ".tif", ".tiff", ".webp" };
        const char* const archives[] = { ".zip", ".tar", ".tgz", ".tar.gz" };
    }

    inline document_family document_family_of_name(const std::string& name)
    {
        const std::string ext = name_extension(name);
        if (ext == ".pdf") return document_family::pdf;
        for (const char* e : file_kinds_impl::texts)
            if (ext == e) return document_family::text;
        for (const char* e : file_kinds_impl::offices)
            if (ext == e) return document_family::office;
        return document_family::none;
    }

    inline name_kind kind_of_name(const std::string& name)
    {
        if (document_family_of_name(name) != document_family::none) return name_kind::document;
        const std::string ext = name_extension(name);
        for (const char* e : file_kinds_impl::pictures)
            if (ext == e) return name_kind::picture;
        for (const char* e : file_kinds_impl::archives)
            if (ext == e) return name_kind::archive;
        return name_kind::unsupported;
    }

    // The extensions of the names Pseudalis protects, ".tar.gz" included.
    inline std::vector<std::string> supported_extensions()
    {
        std::vector<std::string> all(1, ".pdf");
        for (const char* e : file_kinds_impl::texts) all.push_back(e);
        for (const char* e : file_kinds_impl::offices) all.push_back(e);
        for (const char* e : file_kinds_impl::pictures) all.push_back(e);
        for (const char* e : file_kinds_impl::archives) all.push_back(e);
        return all;
    }

    inline bool supported_name(const std::string& name)
    {
        return kind_of_name(name) != name_kind::unsupported;
    }

    /*
        Whether the start of a file may be the start of a text: UTF-16, with its byte order mark or
        its zero bytes, whose code units hold no control character but the tabulation, the ends of
        lines and the form feed, no U+FFFE or U+FFFF and no unpaired surrogate, or else bytes with no
        such control character, which the engine also requires through the whole content.  The
        drive and the shell leave at once a file of an extension of no kind whose start is not a
        text, without reading the rest of it.
    */
    inline bool text_head(const char* data, size_t size)
    {
        const unsigned char* d = reinterpret_cast<const unsigned char*>(data);
        int order = 0;      // 1: UTF-16 little endian, 2: big endian
        const bool bom = size >= 2 && ((d[0] == 0xFF && d[1] == 0xFE) || (d[0] == 0xFE && d[1] == 0xFF));
        const size_t at = bom ? 2 : 0;
        if (bom)
        {
            order = d[0] == 0xFF ? 1 : 2;
        }
        else
        {
            const size_t pairs = std::min<size_t>(size, 4096) / 2;
            size_t zero_even = 0, zero_odd = 0;
            for (size_t i = 0; i < pairs; ++i)
            {
                if (d[2 * i] == 0) ++zero_even;
                if (d[2 * i + 1] == 0) ++zero_odd;
            }
            if (pairs >= 4 && 10 * zero_odd >= 4 * pairs && 20 * zero_even < pairs) order = 1;
            else if (pairs >= 4 && 10 * zero_even >= 4 * pairs && 20 * zero_odd < pairs) order = 2;
        }
        if (order != 0)
        {
            bool high = false;
            for (size_t i = at; i + 1 < size; i += 2)
            {
                const unsigned u = order == 1 ? d[i] | (static_cast<unsigned>(d[i + 1]) << 8)
                                              : (static_cast<unsigned>(d[i]) << 8) | d[i + 1];
                const bool low = u >= 0xDC00 && u <= 0xDFFF;
                if (high != low) return false;
                high = u >= 0xD800 && u <= 0xDBFF;
                if (low || high) continue;
                if (u < 0x20 ? u != '\t' && u != '\n' && u != '\r' && u != '\f'
                             : u == 0x7F || (u >= 0x80 && u < 0xA0) || u == 0xFFFE || u == 0xFFFF)
                    return false;
            }
            return true;
        }
        for (size_t i = 0; i < size; ++i)
        {
            const unsigned char c = d[i];
            if (c < 0x20 ? c != '\t' && c != '\n' && c != '\r' && c != '\f' : c == 0x7F) return false;
        }
        return true;
    }
}

#endif // PSEUDALIS_COMMON_FILE_KINDS_H_
