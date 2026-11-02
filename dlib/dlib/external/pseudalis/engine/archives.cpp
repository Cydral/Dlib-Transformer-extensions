// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "archives.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <dlib/crc32.h>
#include <dlib/zip/deflate.h>
#include <dlib/zip/zip_archive.h>

namespace pseudalis
{

    namespace
    {
        bool ends_with(const std::string& s, const char* suffix)
        {
            const size_t n = std::strlen(suffix);
            if (s.size() < n) return false;
            for (size_t i = 0; i < n; ++i)
                if (std::tolower(static_cast<unsigned char>(s[s.size() - n + i])) != suffix[i]) return false;
            return true;
        }

        // A TAR header: 512 bytes whose checksum adds up, the sum of its bytes with the checksum field as spaces.
        bool tar_header(const std::string& b, size_t at)
        {
            if (at + 512 > b.size()) return false;
            unsigned long sum = 0;
            for (size_t i = 0; i < 512; ++i)
                sum += (i >= 148 && i < 156) ? 32u : static_cast<unsigned char>(b[at + i]);
            unsigned long stored = 0;
            bool digits = false;
            for (size_t i = 148; i < 156; ++i)
            {
                const char c = b[at + i];
                if (c >= '0' && c <= '7')
                {
                    stored = stored * 8 + static_cast<unsigned long>(c - '0');
                    digits = true;
                }
                else if (digits) break;
            }
            return digits && stored == sum;
        }

        uint64_t tar_number(const std::string& b, size_t at, size_t n)
        {
            // Base 256 (GNU), the first byte marked with its high bit.
            if (static_cast<unsigned char>(b[at]) & 0x80)
            {
                uint64_t v = static_cast<unsigned char>(b[at]) & 0x7F;
                for (size_t i = 1; i < n; ++i) v = (v << 8) | static_cast<unsigned char>(b[at + i]);
                return v;
            }
            uint64_t v = 0;
            for (size_t i = 0; i < n; ++i)
            {
                const char c = b[at + i];
                if (c >= '0' && c <= '7') v = v * 8 + static_cast<uint64_t>(c - '0');
                else if (c == 0 || c == ' ') { if (v) break; }
                else break;
            }
            return v;
        }

        std::string tar_string(const std::string& b, size_t at, size_t n)
        {
            size_t len = 0;
            while (len < n && b[at + len] != '\0') ++len;
            return b.substr(at, len);
        }

        /*
            The records "<length> <key>=<value>\n" of a PAX header.  The length, decimal digits
            alone, counts the whole record: a record that does not end where it says, or that
            would not move forward, ends the reading.
        */
        void pax_records(const std::string& data, std::string& path, uint64_t& size, bool& has_size)
        {
            size_t p = 0;
            while (p < data.size())
            {
                const size_t sp = data.find(' ', p);
                if (sp == std::string::npos || sp == p || sp - p > 19) break;
                if (data.find_first_not_of("0123456789", p) != sp) break;
                const uint64_t len = std::strtoull(data.substr(p, sp - p).c_str(), nullptr, 10);
                if (len <= sp - p + 1 || len > data.size() - p || data[p + static_cast<size_t>(len) - 1] != '\n')
                    break;
                const std::string record = data.substr(sp + 1, static_cast<size_t>(p + len - sp - 2));
                const size_t eq = record.find('=');
                if (eq != std::string::npos)
                {
                    const std::string key = record.substr(0, eq), value = record.substr(eq + 1);
                    if (key == "path") path = value;
                    else if (key == "size")
                    {
                        size = std::strtoull(value.c_str(), nullptr, 10);
                        has_size = true;
                    }
                }
                p += static_cast<size_t>(len);
            }
        }

        /*
            Whether the name of an entry stays inside the folder where the archive opens: no
            absolute path, no drive, no backslash and no ".." among its components.  The
            entries named otherwise are left out of the archive written.
        */
        bool safe_entry_name(const std::string& name)
        {
            if (name.empty() || name[0] == '/' || name.find('\\') != std::string::npos ||
                name.find(':') != std::string::npos || name.find('\0') != std::string::npos)
                return false;
            size_t at = 0;
            while (at <= name.size())
            {
                const size_t slash = name.find('/', at);
                const size_t end = slash == std::string::npos ? name.size() : slash;
                if (name.compare(at, end - at, "..") == 0 && end - at == 2) return false;
                if (slash == std::string::npos) break;
                at = slash + 1;
            }
            return true;
        }

        std::vector<archive_entry> read_tar(const std::string& b, const archive_limits& limits)
        {
            std::vector<archive_entry> out;
            uint64_t total = 0;
            std::string long_name, pax_path;
            uint64_t pax_size = 0;
            bool pax_has_size = false;
            size_t at = 0;
            while (at + 512 <= b.size())
            {
                // Two blocks of zeros end the archive; one is enough here.
                bool zero = true;
                for (size_t i = 0; i < 512 && zero; ++i) zero = b[at + i] == '\0';
                if (zero) break;
                if (!tar_header(b, at)) throw archive_error("TAR: damaged header at offset " + std::to_string(at));
                const size_t hdr = at;
                const char type = b[hdr + 156];
                uint64_t size = tar_number(b, hdr + 124, 12);
                if (pax_has_size && type != 'x' && type != 'g') size = pax_size;
                const size_t data_at = at + 512;
                if (size > b.size() - data_at) throw archive_error("TAR: an entry goes past the end of the archive");
                at = data_at + static_cast<size_t>((size + 511) / 512 * 512);
                if (type == 'L' || type == 'x')
                {
                    // The headers of names are read here; a name of 64 KB is long enough.
                    if (size > (64u << 10)) throw archive_error("TAR: a header of names is too long");
                    const std::string data = b.substr(data_at, static_cast<size_t>(size));
                    if (type == 'L') long_name = tar_string(data, 0, data.size());
                    else pax_records(data, pax_path, pax_size, pax_has_size);
                    continue;
                }
                if (type == 'g' || type == 'K') continue;
                archive_entry e;
                const std::string prefix = b.compare(hdr + 257, 5, "ustar") == 0 ? tar_string(b, hdr + 345, 155)
                                                                                 : std::string();
                std::string name = tar_string(b, hdr, 100);
                if (!prefix.empty()) name = prefix + "/" + name;
                if (!long_name.empty()) name = long_name;
                if (!pax_path.empty()) name = pax_path;
                long_name.clear();
                pax_path.clear();
                pax_has_size = false;
                e.name = name;
                // The permissions alone: no set-user or set-group identifier, no sticky bit.
                e.mode = static_cast<uint32_t>(tar_number(b, hdr + 100, 8)) & 0777u;
                e.mtime = static_cast<int64_t>(tar_number(b, hdr + 136, 12));
                e.directory = type == '5' || (!name.empty() && name.back() == '/');
                e.regular = (type == '0' || type == '\0' || type == '7' || e.directory) && safe_entry_name(name);
                if (e.regular && !e.directory) e.data.assign(b, data_at, static_cast<size_t>(size));
                total += e.data.size();
                if (e.data.size() > limits.max_entry_size || total > limits.max_total_size)
                    throw archive_error("TAR: the archive exceeds the size limits");
                out.push_back(std::move(e));
                if (out.size() > limits.max_entries) throw archive_error("TAR: too many entries");
            }
            return out;
        }

        void put_octal(std::string& h, size_t at, size_t n, uint64_t v)
        {
            // n - 1 digits and a terminating NUL.
            std::string digits(n - 1, '0');
            for (size_t i = n - 1; i-- > 0 && v;)
            {
                digits[i] = static_cast<char>('0' + (v & 7));
                v >>= 3;
            }
            h.replace(at, n - 1, digits);
            h[at + n - 1] = '\0';
        }

        std::string tar_block(const std::string& name, uint64_t size, char type, uint32_t mode, int64_t mtime)
        {
            std::string h(512, '\0');
            h.replace(0, std::min<size_t>(name.size(), 100), name.substr(0, 100));
            put_octal(h, 100, 8, mode & 0777);
            put_octal(h, 108, 8, 0);
            put_octal(h, 116, 8, 0);
            put_octal(h, 124, 12, size);
            put_octal(h, 136, 12, static_cast<uint64_t>(mtime > 0 ? mtime : 0));
            h[156] = type;
            h.replace(257, 6, std::string("ustar\0", 6));
            h.replace(263, 2, "00");
            unsigned long sum = 0;
            for (size_t i = 0; i < 512; ++i) sum += (i >= 148 && i < 156) ? 32u : static_cast<unsigned char>(h[i]);
            put_octal(h, 148, 7, sum);
            h[155] = ' ';
            return h;
        }

        void pad(std::string& out)
        {
            if (out.size() % 512) out.append(512 - out.size() % 512, '\0');
        }

        std::string write_tar(const std::vector<archive_entry>& entries)
        {
            std::string out;
            for (const archive_entry& e : entries)
            {
                if (!e.regular) continue;
                std::string name = e.name;
                if (e.directory && (name.empty() || name.back() != '/')) name += '/';
                const char type = e.directory ? '5' : '0';
                const uint64_t size = e.directory ? 0 : e.data.size();
                if (name.size() > 100)
                {
                    // A PAX header carries the whole name; the header that follows keeps its end.
                    std::string record = " path=" + name + "\n";
                    size_t len = record.size();
                    while (std::to_string(len).size() + record.size() != len)
                        len = std::to_string(len).size() + record.size();
                    const std::string pax = std::to_string(len) + record;
                    out += tar_block("PaxHeaders/" + name.substr(name.size() - std::min<size_t>(name.size(), 80)),
                                     pax.size(), 'x', 0644, e.mtime);
                    out += pax;
                    pad(out);
                }
                out += tar_block(name.size() > 100 ? name.substr(name.size() - 100) : name, size, type,
                                 e.directory ? 0755 : (e.mode ? e.mode : 0644), e.mtime);
                if (!e.directory)
                {
                    out += e.data;
                    pad(out);
                }
            }
            out.append(1024, '\0');
            return out;
        }

        void put_le32(std::string& s, uint32_t v)
        {
            for (int k = 0; k < 4; ++k) s.push_back(static_cast<char>((v >> (8 * k)) & 0xFF));
        }

        uint32_t get_le32(const std::string& s, size_t at)
        {
            uint32_t v = 0;
            for (int k = 3; k >= 0; --k) v = (v << 8) | static_cast<unsigned char>(s[at + static_cast<size_t>(k)]);
            return v;
        }
    }

// ----------------------------------------------------------------------------------------

    archive_kind archive_kind_of(const std::string& bytes)
    {
        if (bytes.size() >= 4 && bytes.compare(0, 4, "PK\x03\x04", 4) == 0) return archive_kind::zip;
        if (bytes.size() >= 4 && bytes.compare(0, 4, "PK\x05\x06", 4) == 0) return archive_kind::zip;
        if (bytes.size() >= 3 && static_cast<unsigned char>(bytes[0]) == 0x1F &&
            static_cast<unsigned char>(bytes[1]) == 0x8B && bytes[2] == 8)
            return archive_kind::tar_gz;
        if (tar_header(bytes, 0)) return archive_kind::tar;
        return archive_kind::none;
    }

    archive_kind archive_kind_of_name(const std::string& name)
    {
        if (ends_with(name, ".zip")) return archive_kind::zip;
        if (ends_with(name, ".tar.gz") || ends_with(name, ".tgz")) return archive_kind::tar_gz;
        if (ends_with(name, ".tar")) return archive_kind::tar;
        return archive_kind::none;
    }

    const char* archive_kind_name(archive_kind k)
    {
        switch (k)
        {
            case archive_kind::zip: return "zip";
            case archive_kind::tar: return "tar";
            case archive_kind::tar_gz: return "tar.gz";
            default: return "none";
        }
    }

    std::string gzip_compress(const std::string& data)
    {
        std::string out("\x1F\x8B\x08\x00\x00\x00\x00\x00\x00\xFF", 10);
        out += dlib::deflate_raw(data, 6);
        dlib::crc32 crc;
        crc.add(data);
        put_le32(out, static_cast<uint32_t>(crc.get_checksum()));
        put_le32(out, static_cast<uint32_t>(data.size() & 0xFFFFFFFFu));
        return out;
    }

    std::string gzip_decompress(const std::string& data, uint64_t max_size)
    {
        std::string out;
        size_t at = 0;
        // Members follow one another; each is read where it lies, and their number is bounded.
        size_t members = 0;
        while (at + 18 <= data.size())
        {
            if (++members > 4096) throw archive_error("gzip: too many members");
            if (static_cast<unsigned char>(data[at]) != 0x1F || static_cast<unsigned char>(data[at + 1]) != 0x8B ||
                data[at + 2] != 8)
            {
                if (out.empty()) throw archive_error("gzip: not a gzip stream");
                break;   // trailing garbage after the last member
            }
            const unsigned char flags = static_cast<unsigned char>(data[at + 3]);
            size_t p = at + 10;
            if (flags & 4)
            {
                if (p + 2 > data.size()) throw archive_error("gzip: damaged header");
                p += 2 + (static_cast<unsigned char>(data[p]) |
                          (static_cast<size_t>(static_cast<unsigned char>(data[p + 1])) << 8));
            }
            for (int field : { 8, 16 })
                if (flags & field)
                {
                    while (p < data.size() && data[p] != '\0') ++p;
                    ++p;
                }
            if (flags & 2) p += 2;
            if (p >= data.size()) throw archive_error("gzip: damaged header");
            size_t consumed = 0;
            std::string member;
            try
            {
                member = dlib::inflate_raw(data.data() + p, data.size() - p,
                                           static_cast<size_t>(max_size - std::min<uint64_t>(max_size, out.size())),
                                           &consumed);
            }
            catch (std::exception& e)
            {
                throw archive_error(std::string("gzip: ") + e.what());
            }
            p += consumed;
            if (p + 8 > data.size()) throw archive_error("gzip: the stream is cut short");
            dlib::crc32 crc;
            crc.add(member);
            if (get_le32(data, p) != static_cast<uint32_t>(crc.get_checksum()))
                throw archive_error("gzip: the checksum does not match");
            out += member;
            if (out.size() > max_size) throw archive_error("gzip: the archive exceeds the size limits");
            at = p + 8;
        }
        if (out.empty() && !data.empty()) throw archive_error("gzip: the stream is cut short");
        return out;
    }

    std::vector<archive_entry> read_archive(archive_kind k, const std::string& bytes, const archive_limits& limits)
    {
        if (k == archive_kind::tar) return read_tar(bytes, limits);
        if (k == archive_kind::tar_gz)
            return read_tar(gzip_decompress(bytes, limits.max_total_size + (1u << 20)), limits);
        if (k != archive_kind::zip) throw archive_error("not an archive");
        std::vector<archive_entry> out;
        try
        {
            dlib::zip_limits zl;
            zl.max_entry_size = limits.max_entry_size;
            zl.max_total_size = limits.max_total_size;
            zl.max_entries = limits.max_entries;
            const dlib::zip_archive_reader z(bytes, zl);
            for (size_t i = 0; i < z.entries().size(); ++i)
            {
                const dlib::zip_entry_info& info = z.entries()[i];
                if (info.flags & 1) throw archive_error("ZIP: " + info.name + " is encrypted");
                archive_entry e;
                e.name = info.name;
                e.directory = !info.name.empty() && info.name.back() == '/';
                e.regular = safe_entry_name(info.name);
                if (e.regular && !e.directory) e.data = z.read(i);
                out.push_back(std::move(e));
            }
        }
        catch (archive_error&)
        {
            throw;
        }
        catch (std::exception& e)
        {
            throw archive_error(std::string("ZIP: ") + e.what());
        }
        return out;
    }

    std::string write_archive(archive_kind k, const std::vector<archive_entry>& entries)
    {
        if (k == archive_kind::tar) return write_tar(entries);
        if (k == archive_kind::tar_gz) return gzip_compress(write_tar(entries));
        if (k != archive_kind::zip) throw archive_error("not an archive");
        dlib::zip_archive_writer w(6);
        for (const archive_entry& e : entries)
        {
            if (!e.regular) continue;
            if (e.directory)
                w.add(e.name.empty() || e.name.back() != '/' ? e.name + "/" : e.name, std::string(), false);
            else w.add(e.name, e.data);
        }
        return w.finish();
    }

}
