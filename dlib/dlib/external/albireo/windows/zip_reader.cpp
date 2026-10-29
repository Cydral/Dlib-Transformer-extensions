// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "zip_reader.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <zlib.h>

namespace albireo
{

    namespace
    {
        uint16_t u16(const unsigned char* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
        uint32_t u32(const unsigned char* p)
        {
            return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
                | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
        }
        uint64_t u64(const unsigned char* p) { return u32(p) | (static_cast<uint64_t>(u32(p + 4)) << 32); }
    }

    bool zip_reader::open(const void* data, uint64_t size, std::string& error)
    {
        data_ = static_cast<const unsigned char*>(data);
        size_ = size;
        entries_.clear();
        if (size < 22)
        {
            error = "the archive is empty";
            return false;
        }
        // The end of central directory record, within the last 64 KB.
        int64_t end = -1;
        const uint64_t low = size > 65557 ? size - 65557 : 0;
        for (uint64_t i = size - 22 + 1; i-- > low;)
            if (u32(data_ + i) == 0x06054b50u)
            {
                end = static_cast<int64_t>(i);
                break;
            }
        if (end < 0)
        {
            error = "not a ZIP archive";
            return false;
        }
        uint64_t count = u16(data_ + end + 10);
        uint64_t cd_size = u32(data_ + end + 12);
        uint64_t cd_offset = u32(data_ + end + 16);
        // Zip64: the locator just before the record gives the Zip64 record.
        if (end >= 20 && u32(data_ + end - 20) == 0x07064b50u)
        {
            const uint64_t rec = u64(data_ + end - 20 + 8);
            if (rec + 56 <= size && u32(data_ + rec) == 0x06064b50u)
            {
                count = u64(data_ + rec + 32);
                cd_size = u64(data_ + rec + 40);
                cd_offset = u64(data_ + rec + 48);
            }
        }
        if (cd_offset + cd_size > size)
        {
            error = "the central directory is outside the archive";
            return false;
        }
        uint64_t p = cd_offset;
        for (uint64_t k = 0; k < count; ++k)
        {
            if (p + 46 > size || u32(data_ + p) != 0x02014b50u)
            {
                error = "damaged central directory";
                return false;
            }
            zip_entry e;
            e.method = u16(data_ + p + 10);
            e.crc = u32(data_ + p + 16);
            e.compressed = u32(data_ + p + 20);
            e.size = u32(data_ + p + 24);
            const uint16_t nlen = u16(data_ + p + 28), elen = u16(data_ + p + 30), clen = u16(data_ + p + 32);
            e.local_offset = u32(data_ + p + 42);
            if (p + 46 + nlen + elen + clen > size)
            {
                error = "damaged central directory";
                return false;
            }
            e.name.assign(reinterpret_cast<const char*>(data_ + p + 46), nlen);
            // The Zip64 extra field holds the values whose 32-bit field is saturated.
            const unsigned char* x = data_ + p + 46 + nlen;
            for (uint16_t off = 0; off + 4 <= elen;)
            {
                const uint16_t id = u16(x + off), len = u16(x + off + 2);
                if (id == 0x0001)
                {
                    uint16_t q = static_cast<uint16_t>(off + 4);
                    if (e.size == 0xFFFFFFFFu && q + 8 <= off + 4 + len) { e.size = u64(x + q); q += 8; }
                    if (e.compressed == 0xFFFFFFFFu && q + 8 <= off + 4 + len) { e.compressed = u64(x + q); q += 8; }
                    if (e.local_offset == 0xFFFFFFFFu && q + 8 <= off + 4 + len)
                    {
                        e.local_offset = u64(x + q);
                        q += 8;
                    }
                }
                off = static_cast<uint16_t>(off + 4 + len);
            }
            for (char& c : e.name)
                if (c == '\\') c = '/';
            entries_.push_back(e);
            p += 46 + nlen + elen + clen;
        }
        return true;
    }

    uint64_t zip_reader::total_size() const
    {
        uint64_t t = 0;
        for (const auto& e : entries_) t += e.size;
        return t;
    }

    bool zip_reader::extract(const zip_entry& e, const std::function<bool(const void*, size_t)>& write,
        std::string& error) const
    {
        if (e.local_offset + 30 > size_ || u32(data_ + e.local_offset) != 0x04034b50u)
        {
            error = e.name + ": damaged local header";
            return false;
        }
        const unsigned char* local = data_ + e.local_offset;
        const uint64_t start = e.local_offset + 30 + u16(local + 26) + u16(local + 28);
        if (start + e.compressed > size_)
        {
            error = e.name + ": truncated";
            return false;
        }
        const unsigned char* src = data_ + start;
        uLong crc = crc32(0L, Z_NULL, 0);
        if (e.method == 0)
        {
            uint64_t done = 0;
            while (done < e.size)
            {
                const size_t n = static_cast<size_t>(std::min<uint64_t>(e.size - done, 1u << 20));
                crc = crc32(crc, src + done, static_cast<uInt>(n));
                if (!write(src + done, n))
                {
                    error = e.name + ": cannot be written";
                    return false;
                }
                done += n;
            }
        }
        else if (e.method == 8)
        {
            z_stream z;
            std::memset(&z, 0, sizeof(z));
            if (inflateInit2(&z, -MAX_WBITS) != Z_OK)
            {
                error = "inflate";
                return false;
            }
            std::vector<unsigned char> out(1u << 20);
            uint64_t consumed = 0, produced = 0;
            int rc = Z_OK;
            while (rc != Z_STREAM_END)
            {
                if (z.avail_in == 0 && consumed < e.compressed)
                {
                    const uint64_t n = std::min<uint64_t>(e.compressed - consumed, 1u << 30);
                    z.next_in = const_cast<Bytef*>(src + consumed);
                    z.avail_in = static_cast<uInt>(n);
                    consumed += n;
                }
                z.next_out = out.data();
                z.avail_out = static_cast<uInt>(out.size());
                rc = inflate(&z, Z_NO_FLUSH);
                if (rc != Z_OK && rc != Z_STREAM_END)
                {
                    inflateEnd(&z);
                    error = e.name + ": damaged data";
                    return false;
                }
                const size_t n = out.size() - z.avail_out;
                if (n)
                {
                    crc = crc32(crc, out.data(), static_cast<uInt>(n));
                    produced += n;
                    if (!write(out.data(), n))
                    {
                        inflateEnd(&z);
                        error = e.name + ": cannot be written";
                        return false;
                    }
                }
                if (rc == Z_OK && n == 0 && z.avail_in == 0 && consumed >= e.compressed)
                {
                    inflateEnd(&z);
                    error = e.name + ": truncated data";
                    return false;
                }
            }
            inflateEnd(&z);
            if (produced != e.size)
            {
                error = e.name + ": wrong size";
                return false;
            }
        }
        else
        {
            error = e.name + ": unsupported compression";
            return false;
        }
        if (static_cast<uint32_t>(crc) != e.crc)
        {
            error = e.name + ": the CRC does not match";
            return false;
        }
        return true;
    }

}
