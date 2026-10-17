// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DEFLATE_ABSTRACT_H_
#ifdef DLIB_DEFLATE_ABSTRACT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include "../error.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class deflate_error : public error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Thrown when a compressed stream is malformed, truncated, fails its
                checksum, or would expand beyond the size limit given by the caller.
        !*/
    public:
        deflate_error(const std::string& message);
    };

// ----------------------------------------------------------------------------------------

    std::string inflate_raw(
        const std::string& data,
        std::size_t max_output_size = std::size_t(1) << 30,
        std::size_t* consumed = nullptr
    );
    /*!
        ensures
            - Decodes a raw DEFLATE stream (RFC 1951): stored, fixed Huffman and dynamic
              Huffman blocks, until the block flagged final.
            - If consumed != nullptr, #*consumed is the number of input bytes the stream
              occupied, trailing bits of the last byte included.
            - Back references are checked against the data produced so far, code tables
              are checked for over-subscription, input is checked for exhaustion after
              every code, and the output never exceeds max_output_size. Memory use is
              therefore bounded by max_output_size, and the work by the input size and
              max_output_size.
        throws
            - deflate_error on malformed or truncated input, or when the output would
              exceed max_output_size.
    !*/

    std::string inflate_raw_prefix(
        const std::string& data,
        std::size_t max_output_size,
        bool& complete
    );
    /*!
        ensures
            - Decodes data as inflate_raw() does, without throwing: returns the output
              produced before the first error, so that a truncated or damaged stream
              yields the bytes it encodes up to the damage.
            - #complete == true if the whole stream was decoded without error within
              max_output_size, false otherwise.
    !*/

    std::string deflate_raw(
        const std::string& data,
        int level = 6
    );
    /*!
        requires
            - 0 <= level <= 9
        ensures
            - Returns a raw DEFLATE stream that inflate_raw() decodes back to data.
            - level 0 emits stored blocks only. Levels 1 to 9 run an LZ77 matcher over a
              32 KiB window with hash chains whose length grows with the level, lazy
              evaluation from level 4, and emit for every block whichever of stored,
              fixed Huffman or dynamic Huffman coding is shortest. Dynamic code lengths
              are limited to 15 bits (7 for the code length alphabet) as RFC 1951
              requires.
    !*/

    std::uint32_t adler32(
        const void* data,
        std::size_t size,
        std::uint32_t initial = 1
    );
    /*!
        ensures
            - Returns the Adler-32 checksum of the bytes, continuing from initial.
    !*/

    std::string zlib_compress(
        const std::string& data,
        int level = 6
    );
    /*!
        ensures
            - Returns data wrapped in the zlib format (RFC 1950): a two byte header, a raw
              DEFLATE stream and the Adler-32 of data. This is the encoding of the PDF
              FlateDecode filter.
    !*/

    std::string zlib_decompress(
        const std::string& data,
        std::size_t max_output_size = std::size_t(1) << 30
    );
    /*!
        ensures
            - Decodes a zlib stream and verifies its Adler-32 checksum.
        throws
            - deflate_error on a bad header, a preset dictionary, malformed data, a
              missing or wrong checksum, or when the output would exceed max_output_size.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DEFLATE_ABSTRACT_H_
