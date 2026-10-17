// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SECURE_RANDOM_ABSTRACT_H_
#ifdef DLIB_SECURE_RANDOM_ABSTRACT_H_

#include <cstddef>
#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    void generate_secure_random_bytes(
        void* buffer,
        std::size_t size
    );
    /*!
        ensures
            - Fills buffer with size bytes from the operating system's cryptographically
              secure generator: RtlGenRandom on Windows, arc4random_buf on the BSDs and
              macOS, getrandom on Linux, /dev/urandom elsewhere and on Linux kernels
              without getrandom. The device is read without buffering and opened with
              O_CLOEXEC where available.
            - dlib::rand and std::random_device are not used: the first is predictable,
              and the standard leaves the quality of the second to the implementation.
        throws
            - crypto_error if the generator cannot be read.
    !*/

    std::string generate_secure_random_bytes(
        std::size_t size
    );
    /*!
        ensures
            - Returns size bytes obtained as above.
        throws
            - crypto_error if the generator cannot be read.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SECURE_RANDOM_ABSTRACT_H_
