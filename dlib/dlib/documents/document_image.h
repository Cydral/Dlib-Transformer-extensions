// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DOCUMENT_IMAGE_H_
#define DLIB_DOCUMENT_IMAGE_H_

#include "document_image_abstract.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "../crc32.h"
#include "../zip/deflate.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct document_image
    {
        std::string bytes;
        std::string media_type;
        std::string location;
    };

    typedef std::function<std::string(const document_image& image)> document_image_describer;

    typedef std::function<std::vector<std::pair<std::size_t, std::size_t>>(const std::string& text)>
        document_picture_text_finder;

// ----------------------------------------------------------------------------------------

    inline std::string image_media_type(const std::string& bytes)
    {
        auto starts = [&bytes](const char* sig, std::size_t n, std::size_t at = 0) {
            return bytes.size() >= at + n && bytes.compare(at, n, sig, n) == 0;
        };
        if (starts("\x89PNG\r\n\x1a\n", 8)) return "image/png";
        if (starts("\xFF\xD8\xFF", 3)) return "image/jpeg";
        if (starts("GIF87a", 6) || starts("GIF89a", 6)) return "image/gif";
        if (starts("BM", 2) && bytes.size() > 14) return "image/bmp";
        if (starts("II*\0", 4) || starts("MM\0*", 4)) return "image/tiff";
        if (starts("RIFF", 4) && starts("WEBP", 4, 8)) return "image/webp";
        if (starts("\x01\0\0\0", 4) && starts(" EMF", 4, 40)) return "image/x-emf";
        if (starts("\xD7\xCD\xC6\x9A", 4) || starts("\x01\0\x09\0", 4) || starts("\x02\0\x09\0", 4))
            return "image/x-wmf";
        return std::string();
    }

// ----------------------------------------------------------------------------------------

    inline std::string encode_png(const std::string& pixels, long width, long height, int channels)
    {
        if (width <= 0 || height <= 0 || channels < 1 || channels > 4) return std::string();
        const std::size_t row = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
        if (pixels.size() != row * static_cast<std::size_t>(height)) return std::string();
        auto be32 = [](std::string& s, std::uint32_t v) {
            for (int k = 3; k >= 0; --k) s.push_back(static_cast<char>((v >> (8 * k)) & 0xFF));
        };
        auto chunk = [&be32](std::string& out, const char* type, const std::string& data) {
            be32(out, static_cast<std::uint32_t>(data.size()));
            std::string body(type, 4);
            body += data;
            crc32 crc;
            crc.add(body);
            out += body;
            be32(out, crc.get_checksum());
        };
        // Each row starts with filter 0 (none): zlib does the work.
        std::string raw;
        raw.reserve((row + 1) * static_cast<std::size_t>(height));
        for (long y = 0; y < height; ++y)
        {
            raw.push_back('\0');
            raw.append(pixels, static_cast<std::size_t>(y) * row, row);
        }
        static const char color_types[] = { 0, 4, 2, 6 };
        std::string ihdr;
        be32(ihdr, static_cast<std::uint32_t>(width));
        be32(ihdr, static_cast<std::uint32_t>(height));
        ihdr.push_back(8);
        ihdr.push_back(color_types[channels - 1]);
        ihdr.append(3, '\0');
        std::string out("\x89PNG\r\n\x1a\n", 8);
        chunk(out, "IHDR", ihdr);
        chunk(out, "IDAT", zlib_compress(raw));
        chunk(out, "IEND", std::string());
        return out;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DOCUMENT_IMAGE_H_
