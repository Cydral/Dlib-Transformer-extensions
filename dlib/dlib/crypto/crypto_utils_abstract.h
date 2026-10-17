// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CRYPTO_UTILS_ABSTRACT_H_
#ifdef DLIB_CRYPTO_UTILS_ABSTRACT_H_

#include <cstddef>
#include <string>
#include "../error.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class crypto_error : public error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The exception thrown by the components of dlib/crypto when they are given
                an invalid key, an out of range parameter, or when the operating system
                cannot supply random bytes.
        !*/
    public:
        crypto_error(const std::string& message);
    };

// ----------------------------------------------------------------------------------------

    void secure_wipe(
        void* data,
        std::size_t size
    );
    /*!
        ensures
            - Overwrites the size bytes starting at data with zeros through volatile
              stores, so the compiler cannot drop the writes as dead stores. Used by the
              crypto objects to erase key material in their destructors.
    !*/

    bool constant_time_equal(
        const void* a,
        const void* b,
        std::size_t size
    );
    /*!
        ensures
            - Returns true if the two buffers hold the same size bytes.
            - The running time depends on size only, whatever the position of the first
              difference, as a MAC comparison requires.
    !*/

// ----------------------------------------------------------------------------------------

    std::string to_hex(
        const void* data,
        std::size_t size
    );
    /*!
        ensures
            - Returns the lowercase hexadecimal rendering of the size bytes at data.
    !*/

    std::string to_hex(
        const std::string& bytes
    );
    /*!
        ensures
            - Returns to_hex(bytes.data(), bytes.size()).
    !*/

    std::string from_hex(
        const std::string& hex
    );
    /*!
        ensures
            - Returns the bytes encoded by hex. Upper and lower case digits are accepted
              and whitespace is ignored.
        throws
            - crypto_error if hex contains another character or an odd number of digits.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CRYPTO_UTILS_ABSTRACT_H_
