// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_FILE_KINDS_H_
#define PSEUDALIS_COMMON_FILE_KINDS_H_

#include <cctype>
#include <string>
#include <vector>

/*
    The kinds of files Pseudalis protects, from their names: the engine checks the content
    against the kind of the name, an Office document, a PDF, a text, a picture or an archive,
    and takes the kind of the content for a name without extension; the drive and the shell
    refuse the other names at once.
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
        Whether a name may hold a file Pseudalis protects: a name of its kinds, or a name without
        extension ("README", "LICENSE"), whose content then decides.  The other names are refused
        at once, whatever their content.
    */
    inline bool possible_name(const std::string& name)
    {
        return supported_name(name) || name_extension(name).empty();
    }
}

#endif // PSEUDALIS_COMMON_FILE_KINDS_H_
