// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_ZIP_ARCHIVE_ABSTRACT_H_
#ifdef DLIB_ZIP_ARCHIVE_ABSTRACT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "deflate_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class zip_error : public error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Thrown when an archive is malformed, uses an unsupported feature
                (encryption, a compression method other than stored or deflate, spanning)
                or exceeds the limits it is read under.
        !*/
    public:
        zip_error(const std::string& message);
    };

// ----------------------------------------------------------------------------------------

    struct zip_limits
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Bounds applied while reading an untrusted archive, checked against the
                central directory before any member is decoded:
                    - max_entry_size: the largest declared uncompressed size of a member.
                    - max_total_size: the largest sum of the declared uncompressed sizes.
                    - max_entries: the largest number of members.
                    - max_ratio: the largest ratio between the declared uncompressed size
                      of a deflated member and its compressed size plus one. DEFLATE
                      cannot exceed about 1032 to 1, so the default only rejects sizes
                      that no stream can produce.
                A member is decoded up to its declared size and no further.
        !*/
        std::uint64_t max_entry_size = 512ull << 20;
        std::uint64_t max_total_size = 2048ull << 20;
        std::size_t max_entries = 20000;
        std::uint64_t max_ratio = 1100;
    };

    struct zip_entry_info
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The central directory record of one member.
        !*/
        std::string name;
        std::uint16_t method = 0;          // 0 stored, 8 deflate
        std::uint16_t flags = 0;
        std::uint32_t crc32 = 0;
        std::uint64_t compressed_size = 0;
        std::uint64_t uncompressed_size = 0;
        std::uint64_t local_header_offset = 0;
    };

// ----------------------------------------------------------------------------------------

    class zip_archive_reader
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Read access to a ZIP archive held in memory (PKWARE APPNOTE 6.3), as used
                by the Office Open XML packages.

                The central directory is authoritative. A member's local header must
                carry its name and compression method, member names must be unique, and
                every member is verified against its CRC-32 and declared size when read.
                Every offset and size read from the archive is checked against the
                archive size. ZIP64 records are understood. Encrypted members and
                methods other than stored and deflate are refused.

                Member names are returned as stored. They are not sanitized: a name may
                be absolute or contain "..", so a caller that writes members to disk
                must validate them first.
        !*/
    public:
        explicit zip_archive_reader(
            const std::string& archive,
            const zip_limits& limits = zip_limits()
        );
        /*!
            ensures
                - Parses the central directory of archive, which is copied.
            throws
                - zip_error if the archive is malformed or exceeds limits.
        !*/

        const std::vector<zip_entry_info>& entries() const;
        /*!
            ensures
                - Returns the members in central directory order.
        !*/

        bool contains(const std::string& name) const;
        /*!
            ensures
                - Returns true if a member has exactly this name.
        !*/

        std::string read(const std::string& name) const;
        std::string read(std::size_t index) const;
        /*!
            ensures
                - Returns the uncompressed content of the member.
            throws
                - zip_error if the member does not exist, fails its checks or exceeds the
                  limits.
        !*/
    };

// ----------------------------------------------------------------------------------------

    class zip_archive_writer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Builds a ZIP archive in memory. Members are deflated unless compression
                does not pay, in which case they are stored. Every member carries the
                same neutral timestamp (1980-01-01 00:00) so that the archive does not
                record when its parts were produced.
        !*/
    public:
        zip_archive_writer(int level = 6);
        /*!
            ensures
                - Members added with compression are deflated at this level, clamped to
                  [0, 9].
        !*/

        void add(
            const std::string& name,
            const std::string& content,
            bool compress = true
        );
        /*!
            ensures
                - Appends the member, deflated if compress is true and deflating makes it
                  smaller, stored otherwise.
            throws
                - zip_error if name has been added before, is longer than 65535 bytes,
                  if the archive already holds 65534 members, if content holds 4 GiB or
                  more, or if the archive would exceed 4 GiB. The writer is then left
                  unchanged.
        !*/

        std::string finish();
        /*!
            ensures
                - Appends the central directory and returns the archive bytes.
                - The writer is reset afterwards.
            throws
                - zip_error if the central directory would take the archive above
                  4 GiB. The writer is then left unchanged.
        !*/
    };

// ----------------------------------------------------------------------------------------

    std::string read_file_bytes(const std::string& filename);
    void write_file_bytes(const std::string& filename, const std::string& bytes);
    /*!
        ensures
            - Whole-file binary I/O helpers.
        throws
            - zip_error if the file cannot be opened, read or written.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_ZIP_ARCHIVE_ABSTRACT_H_
