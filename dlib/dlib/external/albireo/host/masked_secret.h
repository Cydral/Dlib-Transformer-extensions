// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_MASKED_SECRET_H_
#define ALBIREO_HOST_MASKED_SECRET_H_

#include <cstddef>
#include <string>

namespace albireo
{

    /*!
        ensures
            - Returns the secret that the configuration built into the program masked (CMakeLists.txt
              and cmake/web_search_key.cmake): each of its size bytes xored with the bits 16 to 23 of
              the next value of the generator x = (x * 1103515245 + 12345) mod 2^31 started at seed,
              which the configuration runs the same way.  The password of the public link and the key
              of the search have seeds of their own.
    !*/
    inline std::string unmask_secret(const unsigned char* bytes, size_t size, unsigned long long seed)
    {
        std::string s(size, '\0');
        for (size_t i = 0; i < size; ++i)
        {
            seed = (seed * 1103515245ull + 12345ull) % 2147483648ull;
            s[i] = static_cast<char>(bytes[i] ^ static_cast<unsigned char>((seed >> 16) & 255));
        }
        return s;
    }

}

#endif // ALBIREO_HOST_MASKED_SECRET_H_
