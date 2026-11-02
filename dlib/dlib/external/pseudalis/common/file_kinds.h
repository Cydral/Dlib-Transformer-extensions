// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_FILE_KINDS_H_
#define PSEUDALIS_COMMON_FILE_KINDS_H_

#include <cctype>
#include <string>
#include <vector>

/*
    The kinds of files Pseudalis protects, from their names: the engine refines the answer
    from the content, the drive and the shell refuse the other names at once.
*/

namespace pseudalis
{
    enum class name_kind { unsupported, document, picture, archive };

    // The extension of a name in lower case: ".docx", ".tar.gz", or an empty string.
    inline std::string name_extension(const std::string& name)
    {
        const size_t slash = name.find_last_of("/\\");
        std::string base = slash == std::string::npos ? name : name.substr(slash + 1);
        for (char& c : base) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (base.size() > 7 && base.compare(base.size() - 7, 7, ".tar.gz") == 0) return ".tar.gz";
        const size_t dot = base.rfind('.');
        return dot == std::string::npos || dot == 0 ? std::string() : base.substr(dot);
    }

    namespace file_kinds_impl
    {
        const char* const documents[] = {
            ".txt", ".text", ".log", ".md", ".markdown", ".csv", ".tsv", ".tab", ".json", ".xml", ".html", ".htm",
            ".xhtml", ".svg", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".docx", ".docm", ".dotx", ".dotm", ".xlsx",
            ".xlsm", ".xltx", ".xltm", ".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm", ".pdf" };
        const char* const pictures[] = {
            ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".bmp", ".tif", ".tiff", ".webp" };
        const char* const archives[] = { ".zip", ".tar", ".tgz", ".tar.gz" };
    }

    inline name_kind kind_of_name(const std::string& name)
    {
        const std::string ext = name_extension(name);
        for (const char* e : file_kinds_impl::documents)
            if (ext == e) return name_kind::document;
        for (const char* e : file_kinds_impl::pictures)
            if (ext == e) return name_kind::picture;
        for (const char* e : file_kinds_impl::archives)
            if (ext == e) return name_kind::archive;
        return name_kind::unsupported;
    }

    // The extensions of the names Pseudalis protects, ".tar.gz" included.
    inline std::vector<std::string> supported_extensions()
    {
        std::vector<std::string> all;
        for (const char* e : file_kinds_impl::documents) all.push_back(e);
        for (const char* e : file_kinds_impl::pictures) all.push_back(e);
        for (const char* e : file_kinds_impl::archives) all.push_back(e);
        return all;
    }

    inline bool supported_name(const std::string& name)
    {
        return kind_of_name(name) != name_kind::unsupported;
    }
}

#endif // PSEUDALIS_COMMON_FILE_KINDS_H_
