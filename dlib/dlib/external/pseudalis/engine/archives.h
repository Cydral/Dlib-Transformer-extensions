// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_ENGINE_ARCHIVES_H_
#define PSEUDALIS_ENGINE_ARCHIVES_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

/*
    The archives the engine opens: ZIP, TAR and TAR compressed with gzip.  They are read and
    written in memory only, so that the content of an archive never reaches a disk in clear.
*/

namespace pseudalis
{
    enum class archive_kind { none, zip, tar, tar_gz };

    struct archive_error : public std::runtime_error
    {
        explicit archive_error(const std::string& what) : std::runtime_error(what) {}
    };

    struct archive_entry
    {
        std::string name;               // '/' separated, as the archive names it
        std::string data;
        bool directory = false;
        bool regular = true;            // false for the links and the devices of a TAR
        uint32_t mode = 0644;           // TAR
        int64_t mtime = 0;              // seconds since 1970, TAR
    };

    struct archive_limits
    {
        uint64_t max_entry_size = 512ull << 20;
        uint64_t max_total_size = 2048ull << 20;
        size_t max_entries = 20000;
    };

    // The kind of an archive from its content: the signatures of ZIP, gzip and TAR, else none.
    archive_kind archive_kind_of(const std::string& bytes);
    // The kind its name gives: .zip, .tar, .tar.gz or .tgz, else none.
    archive_kind archive_kind_of_name(const std::string& name);
    const char* archive_kind_name(archive_kind k);

    /*!
        ensures
            - Returns the entries of the archive, in their order.  A TAR entry named by a
              GNU long name or a PAX header takes that name; its PAX and GNU headers are not
              returned.  The entries of a ZIP whose name ends with '/' are directories.
        throws
            - archive_error when the archive is damaged, encrypted, or exceeds limits.
    !*/
    std::vector<archive_entry> read_archive(archive_kind k, const std::string& bytes,
        const archive_limits& limits = archive_limits());

    /*!
        ensures
            - Returns an archive of the kind k holding the entries whose regular is true,
              in their order: names longer than the TAR header allows get a PAX header.
    !*/
    std::string write_archive(archive_kind k, const std::vector<archive_entry>& entries);

    // gzip members, concatenated or not; decompression throws archive_error beyond max_size.
    std::string gzip_compress(const std::string& data);
    std::string gzip_decompress(const std::string& data, uint64_t max_size);
}

#endif // PSEUDALIS_ENGINE_ARCHIVES_H_
