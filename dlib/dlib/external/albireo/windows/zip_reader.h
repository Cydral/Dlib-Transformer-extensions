// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_WINDOWS_ZIP_READER_H_
#define ALBIREO_WINDOWS_ZIP_READER_H_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace albireo
{

    struct zip_entry
    {
        std::string name;              // path inside the archive, '/' separated
        uint64_t compressed = 0;
        uint64_t size = 0;
        uint32_t crc = 0;
        uint16_t method = 0;           // 0 stored, 8 deflate
        uint64_t local_offset = 0;
        bool directory() const { return !name.empty() && name.back() == '/'; }
    };

    class zip_reader
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A ZIP archive held in memory (the payload of the installer, a resource of
                its program), read through its central directory, Zip64 included.  The
                entries are stored or deflated; each one is checked against its CRC-32
                as it is written.
        !*/
    public:
        bool open(const void* data, uint64_t size, std::string& error);
        const std::vector<zip_entry>& entries() const { return entries_; }
        uint64_t total_size() const;

        /*!
            ensures
                - Writes the content of e to the callback in pieces; returns false, with
                  the reason in error, when the entry is damaged or the callback refuses
                  a piece.
        !*/
        bool extract(const zip_entry& e, const std::function<bool(const void*, size_t)>& write,
            std::string& error) const;

    private:
        const unsigned char* data_ = nullptr;
        uint64_t size_ = 0;
        std::vector<zip_entry> entries_;
    };

}

#endif // ALBIREO_WINDOWS_ZIP_READER_H_
