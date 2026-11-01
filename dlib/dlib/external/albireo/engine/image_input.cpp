// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "image_input.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>

#include <dlib/external/libjpeg/jpeglib.h>
#include <dlib/external/libpng/png.h>

namespace albireo
{

    namespace
    {
        [[noreturn]] void refuse(const std::string& message)
        {
            throw std::invalid_argument(message);
        }

        std::string lower(std::string s)
        {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        uint32_t read_be32(const uint8_t* p)
        {
            return static_cast<uint32_t>(p[0]) << 24 | static_cast<uint32_t>(p[1]) << 16
                | static_cast<uint32_t>(p[2]) << 8 | p[3];
        }

        uint16_t read_be16(const uint8_t* p)
        {
            return static_cast<uint16_t>(p[0] << 8 | p[1]);
        }

        // The name of a format the service does not read, from its signature.
        std::string other_format(const std::string& bytes)
        {
            auto starts = [&](const char* s, size_t at) {
                const size_t n = std::strlen(s);
                return bytes.size() >= at + n && bytes.compare(at, n, s) == 0;
            };
            if (starts("GIF8", 0)) return "GIF";
            if (starts("RIFF", 0) && starts("WEBP", 8)) return "WebP";
            if (starts("BM", 0)) return "BMP";
            if (starts("II*", 0) || starts("MM", 0)) return "TIFF";
            if (starts("ftyp", 4)) return "HEIF or AVIF";
            return "";
        }

        /*
            The orientation of an EXIF block (a TIFF header and its first directory), 1 when
            the block holds none or cannot be read.
        */
        int exif_orientation(const uint8_t* p, size_t n)
        {
            if (n < 8) return 1;
            const bool little = p[0] == 'I' && p[1] == 'I';
            if (!little && !(p[0] == 'M' && p[1] == 'M')) return 1;
            auto u16 = [&](size_t at) -> unsigned {
                return little ? static_cast<unsigned>(p[at] | p[at + 1] << 8)
                    : static_cast<unsigned>(p[at] << 8 | p[at + 1]);
            };
            auto u32 = [&](size_t at) -> uint32_t {
                return little ? static_cast<uint32_t>(p[at]) | static_cast<uint32_t>(p[at + 1]) << 8
                        | static_cast<uint32_t>(p[at + 2]) << 16 | static_cast<uint32_t>(p[at + 3]) << 24
                    : read_be32(p + at);
            };
            if (u16(2) != 42) return 1;
            const uint32_t ifd = u32(4);
            if (ifd > n - 2) return 1;
            const unsigned count = u16(ifd);
            for (unsigned i = 0; i < count; ++i)
            {
                const size_t e = ifd + 2 + static_cast<size_t>(i) * 12;
                if (e + 12 > n) break;
                if (u16(e) == 0x0112 && u16(e + 2) == 3)
                {
                    const unsigned v = u16(e + 8);
                    return v >= 1 && v <= 8 ? static_cast<int>(v) : 1;
                }
            }
            return 1;
        }

        bool swaps_axes(int orientation)
        {
            return orientation >= 5 && orientation <= 8;
        }

        void check_size(long width, long height, const image_limits& limits)
        {
            if (width <= 0 || height <= 0) refuse("the image has no pixels");
            if (width > limits.max_side || height > limits.max_side)
                refuse("the image measures " + std::to_string(width) + " x " + std::to_string(height)
                    + " pixels; a side holds " + std::to_string(limits.max_side) + " pixels at most");
            if (static_cast<double>(width) * height > static_cast<double>(limits.max_pixels))
                refuse("the image holds " + std::to_string(static_cast<long long>(width) * height)
                    + " pixels; an image holds " + std::to_string(limits.max_pixels / 1000000) + " megapixels at most");
        }

        image_header png_header(const std::string& bytes)
        {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(bytes.data());
            const size_t n = bytes.size();
            if (n < 33 || std::memcmp(p + 12, "IHDR", 4) != 0) refuse("the PNG image is damaged (no header)");
            image_header h;
            h.format = image_format::png;
            h.width = static_cast<long>(std::min<uint32_t>(read_be32(p + 16), 0x7FFFFFFF));
            h.height = static_cast<long>(std::min<uint32_t>(read_be32(p + 20), 0x7FFFFFFF));
            // An eXIf chunk before the pixels gives the orientation.
            size_t at = 8;
            while (at + 12 <= n)
            {
                const uint32_t len = read_be32(p + at);
                if (len > n - at - 12) break;
                if (std::memcmp(p + at + 4, "IDAT", 4) == 0 || std::memcmp(p + at + 4, "IEND", 4) == 0) break;
                if (std::memcmp(p + at + 4, "eXIf", 4) == 0)
                {
                    h.orientation = exif_orientation(p + at + 8, len);
                    break;
                }
                at += 12 + static_cast<size_t>(len);
            }
            return h;
        }

        image_header jpeg_header(const std::string& bytes)
        {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(bytes.data());
            const size_t n = bytes.size();
            image_header h;
            h.format = image_format::jpeg;
            size_t at = 2;
            bool sized = false;
            while (at + 4 <= n)
            {
                if (p[at] != 0xFF) refuse("the JPEG image is damaged (invalid marker)");
                const uint8_t marker = p[at + 1];
                if (marker == 0xFF) { ++at; continue; }                   // fill byte
                if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) { at += 2; continue; }
                if (marker == 0xD9 || marker == 0xDA) break;               // end of image, start of scan
                const size_t len = read_be16(p + at + 2);
                if (len < 2 || at + 2 + len > n) refuse("the JPEG image is damaged (truncated header)");
                const uint8_t* seg = p + at + 4;
                const size_t seg_len = len - 2;
                if (marker == 0xE1 && seg_len > 6 && std::memcmp(seg, "Exif\0\0", 6) == 0)
                    h.orientation = exif_orientation(seg + 6, seg_len - 6);
                const bool sof = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
                if (sof && !sized)
                {
                    if (seg_len < 5) refuse("the JPEG image is damaged (frame header)");
                    h.height = read_be16(seg + 1);
                    h.width = read_be16(seg + 3);
                    sized = true;
                }
                at += 2 + len;
            }
            if (!sized) refuse("the JPEG image is damaged (no frame header)");
            return h;
        }

        // Transparent pixels composited onto white, with the integer arithmetic of
        // Pillow's alpha_composite, which the reference preprocessing applies.
        void composite_on_white(const uint8_t* rgba, uint8_t* rgb, size_t pixels)
        {
            for (size_t i = 0; i < pixels; ++i)
            {
                const uint32_t a = rgba[4 * i + 3];
                for (int c = 0; c < 3; ++c)
                {
                    if (a == 255) { rgb[3 * i + c] = rgba[4 * i + c]; continue; }
                    // coef1 = a << 7, coef2 = (255 - a) << 7, rounded division by 255.
                    const uint32_t t = rgba[4 * i + c] * (a << 7) + 255u * ((255u - a) << 7) + (0x80u << 7);
                    rgb[3 * i + c] = static_cast<uint8_t>((((t >> 8) + t) >> 8) >> 7);
                }
            }
        }

        // PNG through libpng.  The decoding functions create every C++ object they hold before
        // their setjmp, which a failure of the library returns to.
        struct png_reader_state
        {
            const uint8_t* data;
            size_t size;
            size_t at;
            char message[256];
        };

        void albireo_png_error(png_structp png, png_const_charp message)
        {
            png_reader_state* s = static_cast<png_reader_state*>(png_get_error_ptr(png));
            std::snprintf(s->message, sizeof(s->message), "%s", message);
            png_longjmp(png, 1);
        }

        void albireo_png_warning(png_structp, png_const_charp)
        {
        }

        void albireo_png_read(png_structp png, png_bytep out, png_size_t n)
        {
            png_reader_state* s = static_cast<png_reader_state*>(png_get_io_ptr(png));
            if (n > s->size - s->at) png_error(png, "the file is truncated");
            std::memcpy(out, s->data + s->at, n);
            s->at += n;
        }

        // Returns 0, or 1 with the message of libpng in state.message, or 2 when memory is short.
        int decode_png_pixels(png_reader_state& state, const image_limits& limits, std::vector<uint8_t>& pixels,
            std::vector<png_bytep>& rows, long& width, long& height, int& channels)
        {
            png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, &state, albireo_png_error,
                albireo_png_warning);
            if (!png) return 2;
            png_infop info = png_create_info_struct(png);
            if (!info)
            {
                png_destroy_read_struct(&png, nullptr, nullptr);
                return 2;
            }
            if (setjmp(png_jmpbuf(png)))
            {
                png_destroy_read_struct(&png, &info, nullptr);
                return 1;
            }
            png_set_read_fn(png, &state, albireo_png_read);
            png_set_user_limits(png, static_cast<png_uint_32>(limits.max_side),
                static_cast<png_uint_32>(limits.max_side));
            png_set_chunk_malloc_max(png, 8u << 20);
            png_read_info(png, info);
            const png_uint_32 w = png_get_image_width(png, info);
            const png_uint_32 h = png_get_image_height(png, info);
            if (static_cast<double>(w) * h > static_cast<double>(limits.max_pixels)) png_error(png, "too many pixels");
            const int depth = png_get_bit_depth(png, info);
            const int type = png_get_color_type(png, info);
            if (depth == 16) png_set_strip_16(png);
            if (type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
            if (type == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
            if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
            if (type == PNG_COLOR_TYPE_GRAY || type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
            png_set_interlace_handling(png);
            png_read_update_info(png, info);
            const int c = png_get_channels(png, info);
            if ((c != 3 && c != 4) || png_get_rowbytes(png, info) != static_cast<size_t>(w) * c)
                png_error(png, "unexpected pixel layout");
            try
            {
                pixels.resize(static_cast<size_t>(w) * h * c);
                rows.resize(h);
            }
            catch (...)
            {
                png_destroy_read_struct(&png, &info, nullptr);
                return 2;
            }
            for (png_uint_32 y = 0; y < h; ++y) rows[y] = pixels.data() + static_cast<size_t>(y) * w * c;
            png_read_image(png, rows.data());
            png_destroy_read_struct(&png, &info, nullptr);
            width = static_cast<long>(w);
            height = static_cast<long>(h);
            channels = c;
            return 0;
        }

        rgb_image decode_png(const std::string& bytes, const image_limits& limits)
        {
            png_reader_state state;
            state.data = reinterpret_cast<const uint8_t*>(bytes.data());
            state.size = bytes.size();
            state.at = 0;
            state.message[0] = '\0';
            std::vector<uint8_t> pixels;
            std::vector<png_bytep> rows;
            long w = 0, h = 0;
            int c = 0;
            const int rc = decode_png_pixels(state, limits, pixels, rows, w, h, c);
            if (rc == 2) throw std::bad_alloc();
            if (rc != 0) refuse(std::string("the PNG image cannot be read (") + state.message + ")");
            rgb_image out;
            out.width = w;
            out.height = h;
            if (c == 3)
            {
                out.pixels.swap(pixels);
            }
            else
            {
                out.pixels.resize(static_cast<size_t>(w) * h * 3);
                composite_on_white(pixels.data(), out.pixels.data(), static_cast<size_t>(w) * h);
            }
            return out;
        }

        // JPEG through libjpeg, with the same discipline.
        struct jpeg_failure
        {
            jpeg_error_mgr mgr;
            std::jmp_buf jump;
            char message[JMSG_LENGTH_MAX];
        };

        void albireo_jpeg_error(j_common_ptr info)
        {
            jpeg_failure* f = reinterpret_cast<jpeg_failure*>(info->err);
            (*info->err->format_message)(info, f->message);
            std::longjmp(f->jump, 1);
        }

        void albireo_jpeg_message(j_common_ptr, int)
        {
        }

        int decode_jpeg_pixels(const std::string& bytes, const image_limits& limits, jpeg_failure& failure,
            std::vector<uint8_t>& pixels, long& width, long& height, bool& cmyk, bool& adobe)
        {
            jpeg_decompress_struct info;
            info.err = jpeg_std_error(&failure.mgr);
            failure.mgr.error_exit = albireo_jpeg_error;
            failure.mgr.emit_message = albireo_jpeg_message;
            failure.message[0] = '\0';
            if (setjmp(failure.jump))
            {
                jpeg_destroy_decompress(&info);
                return 1;
            }
            jpeg_create_decompress(&info);
            jpeg_mem_src(&info, reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
            jpeg_read_header(&info, TRUE);
            if (static_cast<long>(info.image_width) > limits.max_side
                || static_cast<long>(info.image_height) > limits.max_side
                || static_cast<double>(info.image_width) * info.image_height > static_cast<double>(limits.max_pixels))
            {
                std::snprintf(failure.message, sizeof(failure.message), "too many pixels");
                jpeg_destroy_decompress(&info);
                return 1;
            }
            const bool four = info.jpeg_color_space == JCS_CMYK || info.jpeg_color_space == JCS_YCCK;
            info.out_color_space = four ? JCS_CMYK : JCS_RGB;
            jpeg_start_decompress(&info);
            const size_t stride = static_cast<size_t>(info.output_width) * info.output_components;
            try
            {
                pixels.resize(stride * info.output_height);
            }
            catch (...)
            {
                jpeg_destroy_decompress(&info);
                return 2;
            }
            while (info.output_scanline < info.output_height)
            {
                JSAMPROW row = pixels.data() + stride * info.output_scanline;
                jpeg_read_scanlines(&info, &row, 1);
            }
            width = static_cast<long>(info.output_width);
            height = static_cast<long>(info.output_height);
            cmyk = four;
            adobe = info.saw_Adobe_marker != 0;
            jpeg_finish_decompress(&info);
            jpeg_destroy_decompress(&info);
            return 0;
        }

        rgb_image decode_jpeg(const std::string& bytes, const image_limits& limits)
        {
            jpeg_failure failure;
            std::vector<uint8_t> pixels;
            long w = 0, h = 0;
            bool cmyk = false, adobe = false;
            const int rc = decode_jpeg_pixels(bytes, limits, failure, pixels, w, h, cmyk, adobe);
            if (rc == 2) throw std::bad_alloc();
            if (rc != 0) refuse(std::string("the JPEG image cannot be read (") + failure.message + ")");
            rgb_image out;
            out.width = w;
            out.height = h;
            if (!cmyk)
            {
                out.pixels.swap(pixels);
                return out;
            }
            // CMYK as Pillow converts it, the files of Adobe storing the inverted values.
            const size_t n = static_cast<size_t>(w) * h;
            out.pixels.resize(n * 3);
            for (size_t i = 0; i < n; ++i)
            {
                uint32_t v[4];
                for (int c = 0; c < 4; ++c) v[c] = adobe ? 255u - pixels[4 * i + c] : pixels[4 * i + c];
                const uint32_t nk = 255u - v[3];
                for (int c = 0; c < 3; ++c)
                {
                    const uint32_t t = v[c] * nk + 128u;
                    const uint32_t m = ((t >> 8) + t) >> 8;
                    out.pixels[3 * i + c] = static_cast<uint8_t>(nk > m ? nk - m : 0u);
                }
            }
            return out;
        }

        // The image as displayed for an EXIF orientation, as Pillow's exif_transpose makes it.
        rgb_image orient(const rgb_image& src, int orientation)
        {
            if (orientation <= 1 || orientation > 8) return src;
            const long W = src.width, H = src.height;
            rgb_image out;
            out.width = swaps_axes(orientation) ? H : W;
            out.height = swaps_axes(orientation) ? W : H;
            out.pixels.resize(src.pixels.size());
            for (long y = 0; y < out.height; ++y)
                for (long x = 0; x < out.width; ++x)
                {
                    long sx = x, sy = y;
                    switch (orientation)
                    {
                    case 2: sx = W - 1 - x; break;                              // mirrored
                    case 3: sx = W - 1 - x; sy = H - 1 - y; break;              // rotated 180
                    case 4: sy = H - 1 - y; break;                              // flipped
                    case 5: sx = y; sy = x; break;                              // transposed
                    case 6: sx = y; sy = H - 1 - x; break;                      // rotated 90 clockwise
                    case 7: sx = W - 1 - y; sy = H - 1 - x; break;              // transversed
                    default: sx = W - 1 - y; sy = x; break;                     // rotated 90 anticlockwise
                    }
                    const uint8_t* s = src.pixels.data() + 3 * (static_cast<size_t>(sy) * W + sx);
                    uint8_t* d = out.pixels.data() + 3 * (static_cast<size_t>(y) * out.width + x);
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                }
            return out;
        }

        // Pillow's resampling: the coefficients of one axis.
        const int resample_bits = 22;

        double bicubic_filter(double x)
        {
            const double a = -0.5;
            if (x < 0.0) x = -x;
            if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
            if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
            return 0.0;
        }

        double bilinear_filter(double x)
        {
            if (x < 0.0) x = -x;
            return x < 1.0 ? 1.0 - x : 0.0;
        }

        long resample_coefficients(long in_size, long out_size, std::vector<long>& bounds,
            std::vector<int32_t>& weights, bool bilinear)
        {
            const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
            const double filterscale = std::max(1.0, scale);
            const double support = (bilinear ? 1.0 : 2.0) * filterscale;
            const long ksize = static_cast<long>(std::ceil(support)) * 2 + 1;
            std::vector<double> k(static_cast<size_t>(ksize));
            bounds.resize(static_cast<size_t>(out_size) * 2);
            weights.assign(static_cast<size_t>(out_size * ksize), 0);
            for (long xx = 0; xx < out_size; ++xx)
            {
                const double center = (static_cast<double>(xx) + 0.5) * scale;
                const double ss = 1.0 / filterscale;
                long xmin = static_cast<long>(center - support + 0.5);
                if (xmin < 0) xmin = 0;
                long xmax = static_cast<long>(center + support + 0.5);
                if (xmax > in_size) xmax = in_size;
                xmax -= xmin;
                double ww = 0.0;
                for (long x = 0; x < xmax; ++x)
                {
                    const double t = (static_cast<double>(x + xmin) - center + 0.5) * ss;
                    const double w = bilinear ? bilinear_filter(t) : bicubic_filter(t);
                    k[static_cast<size_t>(x)] = w;
                    ww += w;
                }
                for (long x = 0; x < xmax; ++x)
                {
                    const double w = ww != 0.0 ? k[static_cast<size_t>(x)] / ww : k[static_cast<size_t>(x)];
                    // Pillow adds or subtracts one half, then truncates toward zero.
                    weights[static_cast<size_t>(xx * ksize + x)] = static_cast<int32_t>(
                        w < 0.0 ? -0.5 + w * (1 << resample_bits) : 0.5 + w * (1 << resample_bits));
                }
                bounds[static_cast<size_t>(xx * 2)] = xmin;
                bounds[static_cast<size_t>(xx * 2 + 1)] = xmax;
            }
            return ksize;
        }

        inline uint8_t clip8(int32_t v)
        {
            v >>= resample_bits;
            return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
        }

        // The SHA-256 compression function.
        const uint32_t sha_k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };

        inline uint32_t rotr(uint32_t x, int n)
        {
            return (x >> n) | (x << (32 - n));
        }

        void sha256_block(uint32_t h[8], const uint8_t* p)
        {
            uint32_t w[64];
            for (int i = 0; i < 16; ++i) w[i] = read_be32(p + 4 * i);
            for (int i = 16; i < 64; ++i)
            {
                const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }
            uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
            for (int i = 0; i < 64; ++i)
            {
                const uint32_t t1 = k + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + sha_k[i]
                    + w[i];
                const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
                k = g;
                g = f;
                f = e;
                e = d + t1;
                d = c;
                c = b;
                b = a;
                a = t1 + t2;
            }
            h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
        }
    }

    std::string decode_image_data_url(const std::string& url, size_t max_bytes)
    {
        size_t start = 0;
        while (start < url.size() && std::isspace(static_cast<unsigned char>(url[start]))) ++start;
        const std::string head = lower(url.substr(start, 16));
        if (head.compare(0, 5, "data:") != 0)
        {
            if (head.compare(0, 7, "http://") == 0 || head.compare(0, 8, "https://") == 0
                || head.compare(0, 5, "file:") == 0)
                refuse("images are given as data URLs (data:image/png;base64,...); the service reads nothing from "
                    "the network or the disk for a request, so http, https and file addresses are refused");
            refuse("an image is given as a data URL (data:image/png;base64,... or data:image/jpeg;base64,...)");
        }
        const size_t comma = url.find(',', start);
        if (comma == std::string::npos) refuse("the data URL of an image holds no data");
        const std::string meta = lower(url.substr(start + 5, comma - start - 5));
        const size_t semi = meta.find(';');
        const std::string type = meta.substr(0, semi);
        if (type.compare(0, 6, "image/") != 0) refuse("the data URL does not hold an image (" + type + ")");
        if (meta.size() < 7 || meta.compare(meta.size() - 7, 7, ";base64") != 0)
            refuse("the data URL of an image must be encoded in base64");
        // The decoded size is checked before any decoding: three bytes per four characters.
        const size_t text = url.size() - comma - 1;
        if (text / 4 * 3 > max_bytes + 3)
            refuse("the image takes more than " + std::to_string(max_bytes >> 20) + " MB");
        std::string out;
        out.reserve(text / 4 * 3 + 3);
        uint32_t value = 0;
        int bits = 0;
        bool padding = false;
        for (size_t i = comma + 1; i < url.size(); ++i)
        {
            const unsigned char c = static_cast<unsigned char>(url[i]);
            int d;
            if (c >= 'A' && c <= 'Z') d = c - 'A';
            else if (c >= 'a' && c <= 'z') d = c - 'a' + 26;
            else if (c >= '0' && c <= '9') d = c - '0' + 52;
            else if (c == '+' || c == '-') d = 62;
            else if (c == '/' || c == '_') d = 63;
            else if (c == '=') { padding = true; continue; }
            else if (std::isspace(c)) continue;
            else refuse("the data URL of an image holds characters outside base64");
            if (padding) refuse("the data URL of an image holds characters after its padding");
            value = (value << 6) | static_cast<uint32_t>(d);
            bits += 6;
            if (bits >= 8)
            {
                bits -= 8;
                out += static_cast<char>((value >> bits) & 0xFF);
            }
        }
        if (out.size() > max_bytes) refuse("the image takes more than " + std::to_string(max_bytes >> 20) + " MB");
        if (out.empty()) refuse("the data URL of an image holds no data");
        return out;
    }

    image_header read_image_header(const std::string& bytes, const image_limits& limits)
    {
        if (bytes.size() > limits.max_bytes)
            refuse("the image takes more than " + std::to_string(limits.max_bytes >> 20) + " MB");
        static const unsigned char png_sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n' };
        image_header h;
        if (bytes.size() >= 8 && std::memcmp(bytes.data(), png_sig, 8) == 0)
        {
            h = png_header(bytes);
        }
        else if (bytes.size() >= 3 && static_cast<unsigned char>(bytes[0]) == 0xFF
            && static_cast<unsigned char>(bytes[1]) == 0xD8 && static_cast<unsigned char>(bytes[2]) == 0xFF)
        {
            h = jpeg_header(bytes);
        }
        else
        {
            const std::string other = other_format(bytes);
            refuse(other.empty() ? std::string("the image is neither a PNG nor a JPEG file")
                : "the image is a " + other + " file; the service reads PNG and JPEG images");
        }
        check_size(h.width, h.height, limits);
        if (swaps_axes(h.orientation)) std::swap(h.width, h.height);
        return h;
    }

    rgb_image decode_image(const std::string& bytes, const image_limits& limits, long fit_width, long fit_height)
    {
        const image_header h = read_image_header(bytes, limits);
        rgb_image img = h.format == image_format::png ? decode_png(bytes, limits) : decode_jpeg(bytes, limits);
        // The stored image is resized to the stored axes of the target, then oriented.
        if (fit_width > 0 && fit_height > 0)
        {
            const bool swap = swaps_axes(h.orientation);
            rgb_image small = resize_bicubic(img, swap ? fit_height : fit_width, swap ? fit_width : fit_height);
            img.clear();
            img = std::move(small);
        }
        if (h.orientation > 1) img = orient(img, h.orientation);
        return img;
    }

    void fit_image_size(long width, long height, long factor, long min_pixels, long max_pixels, long& out_width,
        long& out_height)
    {
        if (width <= 0 || height <= 0 || factor <= 0) refuse("the image has no pixels");
        if (static_cast<double>(std::max(width, height)) / static_cast<double>(std::min(width, height)) > 200.0)
            refuse("the sides of the image differ by a ratio over 200");
        const double f = static_cast<double>(factor);
        const double h = static_cast<double>(height), w = static_cast<double>(width);
        // Python's round(), halves to even, in the default rounding mode.
        long h_bar = static_cast<long>(std::nearbyint(h / f)) * factor;
        long w_bar = static_cast<long>(std::nearbyint(w / f)) * factor;
        if (static_cast<double>(h_bar) * w_bar > static_cast<double>(max_pixels))
        {
            const double beta = std::sqrt(h * w / static_cast<double>(max_pixels));
            h_bar = std::max(factor, static_cast<long>(std::floor(h / beta / f)) * factor);
            w_bar = std::max(factor, static_cast<long>(std::floor(w / beta / f)) * factor);
        }
        else if (static_cast<double>(h_bar) * w_bar < static_cast<double>(min_pixels))
        {
            const double beta = std::sqrt(static_cast<double>(min_pixels) / (h * w));
            h_bar = static_cast<long>(std::ceil(h * beta / f)) * factor;
            w_bar = static_cast<long>(std::ceil(w * beta / f)) * factor;
        }
        out_width = w_bar;
        out_height = h_bar;
    }

    rgb_image resize_bicubic(const rgb_image& src, long width, long height)
    {
        return resize_pillow(src, width, height, false);
    }

    rgb_image resize_bilinear(const rgb_image& src, long width, long height)
    {
        return resize_pillow(src, width, height, true);
    }

    rgb_image resize_pillow(const rgb_image& src, long width, long height, bool bilinear)
    {
        if (width <= 0 || height <= 0 || src.width <= 0 || src.height <= 0)
            throw std::invalid_argument("resize_pillow: empty image");
        if (width == src.width && height == src.height) return src;
        const int32_t round = 1 << (resample_bits - 1);

        // Horizontal pass, into an image of the target width and the source height.
        rgb_image mid;
        const rgb_image* in = &src;
        if (width != src.width)
        {
            std::vector<long> bounds;
            std::vector<int32_t> k;
            const long ksize = resample_coefficients(src.width, width, bounds, k, bilinear);
            mid.width = width;
            mid.height = src.height;
            mid.pixels.resize(static_cast<size_t>(width) * src.height * 3);
            for (long y = 0; y < src.height; ++y)
            {
                const uint8_t* row = src.pixels.data() + static_cast<size_t>(y) * src.width * 3;
                uint8_t* out = mid.pixels.data() + static_cast<size_t>(y) * width * 3;
                for (long x = 0; x < width; ++x)
                {
                    const long xmin = bounds[static_cast<size_t>(2 * x)];
                    const long xcnt = bounds[static_cast<size_t>(2 * x + 1)];
                    const int32_t* w = k.data() + x * ksize;
                    int32_t s0 = round, s1 = round, s2 = round;
                    const uint8_t* p = row + xmin * 3;
                    for (long i = 0; i < xcnt; ++i, p += 3)
                    {
                        s0 += p[0] * w[i];
                        s1 += p[1] * w[i];
                        s2 += p[2] * w[i];
                    }
                    out[3 * x] = clip8(s0);
                    out[3 * x + 1] = clip8(s1);
                    out[3 * x + 2] = clip8(s2);
                }
            }
            in = &mid;
        }
        if (height == in->height) return in == &mid ? mid : *in;

        // Vertical pass, a whole row at a time.
        std::vector<long> bounds;
        std::vector<int32_t> k;
        const long ksize = resample_coefficients(in->height, height, bounds, k, bilinear);
        rgb_image out;
        out.width = width;
        out.height = height;
        const size_t row_values = static_cast<size_t>(width) * 3;
        out.pixels.resize(row_values * static_cast<size_t>(height));
        std::vector<int32_t> acc(row_values);
        for (long y = 0; y < height; ++y)
        {
            const long ymin = bounds[static_cast<size_t>(2 * y)];
            const long ycnt = bounds[static_cast<size_t>(2 * y + 1)];
            const int32_t* w = k.data() + y * ksize;
            std::fill(acc.begin(), acc.end(), round);
            for (long i = 0; i < ycnt; ++i)
            {
                const uint8_t* row = in->pixels.data() + static_cast<size_t>(ymin + i) * row_values;
                const int32_t wi = w[i];
                for (size_t e = 0; e < row_values; ++e) acc[e] += row[e] * wi;
            }
            uint8_t* dst = out.pixels.data() + static_cast<size_t>(y) * row_values;
            for (size_t e = 0; e < row_values; ++e) dst[e] = clip8(acc[e]);
        }
        return out;
    }

    std::array<uint8_t, 32> sha256(const uint8_t* data, size_t n)
    {
        uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19 };
        size_t i = 0;
        for (; i + 64 <= n; i += 64) sha256_block(h, data + i);
        uint8_t tail[128] = { 0 };
        const size_t rest = n - i;
        if (rest) std::memcpy(tail, data + i, rest);
        tail[rest] = 0x80;
        const size_t blocks = rest + 9 > 64 ? 2 : 1;
        const uint64_t bits = static_cast<uint64_t>(n) * 8;
        for (int b = 0; b < 8; ++b) tail[blocks * 64 - 1 - b] = static_cast<uint8_t>(bits >> (8 * b));
        for (size_t b = 0; b < blocks; ++b) sha256_block(h, tail + 64 * b);
        std::array<uint8_t, 32> out;
        for (int w = 0; w < 8; ++w)
            for (int b = 0; b < 4; ++b)
                out[static_cast<size_t>(4 * w + b)] = static_cast<uint8_t>(h[w] >> (24 - 8 * b));
        return out;
    }

    std::array<uint8_t, 32> image_digest(const rgb_image& image)
    {
        // The size first, so that two images of the same pixels in other shapes differ.
        std::vector<uint8_t> data(16 + image.pixels.size());
        for (int b = 0; b < 8; ++b)
        {
            data[static_cast<size_t>(b)] = static_cast<uint8_t>(static_cast<uint64_t>(image.width) >> (8 * b));
            data[static_cast<size_t>(8 + b)] = static_cast<uint8_t>(static_cast<uint64_t>(image.height) >> (8 * b));
        }
        if (!image.pixels.empty()) std::memcpy(data.data() + 16, image.pixels.data(), image.pixels.size());
        return sha256(data.data(), data.size());
    }

}
