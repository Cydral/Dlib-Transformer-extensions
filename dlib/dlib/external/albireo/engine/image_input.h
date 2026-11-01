// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_IMAGE_INPUT_H_
#define ALBIREO_ENGINE_IMAGE_INPUT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace albireo
{

    /*!
        The images of the requests, from the data URL of a message to the pixels an image
        encoder reads.

        A request carries an image as a data URL ("data:image/png;base64,..."): the service
        reads nothing from the network or the disk on behalf of a request, so every other
        address is refused.  The bytes are a PNG or a JPEG file, recognized by their
        signature, and decoded by the libpng and libjpeg copies bundled with dlib.  The size
        of an image is read from its header before anything is decoded, so that the limits
        of image_limits refuse an image before its pixels take any memory.

        The decoded image is made of 8-bit RGB pixels, as the reference preprocessing of
        the vision models makes it: transparent pixels are composited onto white, a 16-bit
        PNG keeps the high byte of its samples, and the orientation that a JPEG or a PNG
        declares in its EXIF data is applied, as the browsers show the image.

        Every function throws std::invalid_argument, with a message for the user, when the
        image is refused or cannot be read.
    !*/

    struct image_limits
    {
        long max_side = 8192;                 // pixels on the longest side
        long max_pixels = 40000000;           // pixels in all
        size_t max_bytes = 20u << 20;         // bytes of the PNG or JPEG file
    };

    enum class image_format { png, jpeg };

    struct image_header
    {
        image_format format = image_format::png;
        long width = 0;                       // as displayed, the orientation applied
        long height = 0;
        int orientation = 1;                  // EXIF orientation, 1 to 8
    };

    struct rgb_image
    {
        long width = 0;
        long height = 0;
        std::vector<uint8_t> pixels;          // height rows of width RGB triplets

        void clear()
        {
            width = height = 0;
            std::vector<uint8_t>().swap(pixels);
        }
    };

    /*!
        ensures
            - Returns the bytes of the base64 data URL url, whose media type must be an image.
            - Refuses an http, https or file address, a data URL that is not in base64, and
              more than max_bytes bytes, before decoding them.
    !*/
    std::string decode_image_data_url(const std::string& url, size_t max_bytes);

    /*!
        ensures
            - Returns the format and the displayed size of the PNG or JPEG file in bytes,
              read from its header, and refuses a file beyond limits.
    !*/
    image_header read_image_header(const std::string& bytes, const image_limits& limits);

    /*!
        ensures
            - Decodes the PNG or JPEG file in bytes to RGB pixels, within limits.
            - When fit_width and fit_height are positive, the image is resized to them with
              resize_bicubic() before its orientation is applied, so that a large image never
              exists at full size once decoded; fit_width and fit_height are then the size
              of the image as displayed.
    !*/
    rgb_image decode_image(const std::string& bytes, const image_limits& limits, long fit_width = 0,
        long fit_height = 0);

    /*!
        ensures
            - Returns in out_width and out_height the size an image of width x height is
              resized to before an encoder reads it: both multiples of factor, within
              [min_pixels, max_pixels] pixels when the aspect ratio allows it, the aspect
              ratio kept as closely as possible.  This is the smart_resize rule of the
              reference preprocessing of Qwen2-VL and Qwen3-VL, with its rounding of halves
              to even.
            - Refuses an image whose sides differ by a ratio over 200.
    !*/
    void fit_image_size(long width, long height, long factor, long min_pixels, long max_pixels, long& out_width,
        long& out_height);

    /*!
        ensures
            - Returns src resized to width x height with the bicubic filter of Pillow
              (a = -0.5, its support widened by the scale factor when the image shrinks,
              weights in fixed point with 22 fractional bits, the horizontal pass first, each
              pass rounded to 8 bits), which the reference preprocessing uses; an image of
              the same size is returned as it is.
    !*/
    rgb_image resize_bicubic(const rgb_image& src, long width, long height);

    // The same with the bilinear filter of Pillow (a triangle of support 1, widened by the
    // scale factor when the image shrinks), which the preprocessing of LLaVA-NeXT models uses.
    rgb_image resize_bilinear(const rgb_image& src, long width, long height);

    // resize_bicubic() or resize_bilinear().
    rgb_image resize_pillow(const rgb_image& src, long width, long height, bool bilinear);

    // The SHA-256 digest of n bytes.
    std::array<uint8_t, 32> sha256(const uint8_t* data, size_t n);

    // The digest of the pixels of an image and its size, the key of its encodings.
    std::array<uint8_t, 32> image_digest(const rgb_image& image);

}

#endif // ALBIREO_ENGINE_IMAGE_INPUT_H_
