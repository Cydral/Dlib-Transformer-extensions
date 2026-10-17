// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CRYPTO_UTILS_H_
#define DLIB_CRYPTO_UTILS_H_

#include "crypto_utils_abstract.h"

#include <cstddef>
#include <string>
#include "../error.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class crypto_error : public error
    {
    public:
        crypto_error(const std::string& message) : error(message) {}
    };

// ----------------------------------------------------------------------------------------

    inline void secure_wipe(
        void* data,
        std::size_t size
    )
    {
        volatile unsigned char* p = static_cast<volatile unsigned char*>(data);
        while (size--)
            *p++ = 0;
    }

    inline bool constant_time_equal(
        const void* a,
        const void* b,
        std::size_t size
    )
    {
        const unsigned char* pa = static_cast<const unsigned char*>(a);
        const unsigned char* pb = static_cast<const unsigned char*>(b);
        unsigned char diff = 0;
        for (std::size_t i = 0; i < size; ++i)
            diff |= static_cast<unsigned char>(pa[i] ^ pb[i]);
        return diff == 0;
    }

// ----------------------------------------------------------------------------------------

    inline std::string to_hex(
        const void* data,
        std::size_t size
    )
    {
        static const char digits[] = "0123456789abcdef";
        const unsigned char* p = static_cast<const unsigned char*>(data);
        std::string out;
        out.reserve(size * 2);
        for (std::size_t i = 0; i < size; ++i)
        {
            out.push_back(digits[p[i] >> 4]);
            out.push_back(digits[p[i] & 0x0F]);
        }
        return out;
    }

    inline std::string to_hex(
        const std::string& bytes
    )
    {
        return to_hex(bytes.data(), bytes.size());
    }

    inline std::string from_hex(
        const std::string& hex
    )
    {
        std::string out;
        int pending = -1;
        for (char ch : hex)
        {
            int v;
            if (ch >= '0' && ch <= '9') v = ch - '0';
            else if (ch >= 'a' && ch <= 'f') v = ch - 'a' + 10;
            else if (ch >= 'A' && ch <= 'F') v = ch - 'A' + 10;
            else if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') continue;
            else throw crypto_error("from_hex(): invalid hexadecimal character");

            if (pending < 0)
            {
                pending = v;
            }
            else
            {
                out.push_back(static_cast<char>((pending << 4) | v));
                pending = -1;
            }
        }
        if (pending >= 0)
            throw crypto_error("from_hex(): odd number of hexadecimal digits");
        return out;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CRYPTO_UTILS_H_
