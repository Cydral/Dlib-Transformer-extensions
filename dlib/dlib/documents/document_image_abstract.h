// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DOCUMENT_IMAGE_ABSTRACT_H_
#ifdef DLIB_DOCUMENT_IMAGE_ABSTRACT_H_

#include <functional>
#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct document_image
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A picture found in a document: the bytes of its file, encoded (PNG, JPEG,
                GIF, BMP, TIFF, WebP, EMF, WMF...), its media type, and where it was found
                ("word/media/image1.png" in an Office package, "page 2" in a PDF).
        !*/
        std::string bytes;
        std::string media_type;     // "image/png", "image/jpeg"...
        std::string location;
    };

    typedef std::function<std::string(const document_image& image)> document_image_describer;
    /*!
        WHAT THIS OBJECT REPRESENTS
            A function that gives the text to write in the place of a picture of a
            document, a description for instance, or an empty string to leave the picture
            out without text.  The anonymizers write the text as it is returned: a
            function whose text may hold personal data protects it first.
    !*/

// ----------------------------------------------------------------------------------------

    std::string image_media_type(
        const std::string& bytes
    );
    /*!
        ensures
            - Returns the media type of an image file from its signature: "image/png",
              "image/jpeg", "image/gif", "image/bmp", "image/tiff", "image/webp",
              "image/x-emf", "image/x-wmf", or an empty string for anything else.
    !*/

    std::string encode_png(
        const std::string& pixels,
        long width,
        long height,
        int channels
    );
    /*!
        requires
            - pixels holds width*height pixels of channels bytes each, row after row, from
              the top: gray (1), gray and alpha (2), red green blue (3) or red green blue
              and alpha (4), 8 bits per sample.
        ensures
            - Returns the PNG file of these pixels, or an empty string when the sizes do not
              match or channels is not 1 to 4.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DOCUMENT_IMAGE_ABSTRACT_H_
