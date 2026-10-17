// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SECURE_RANDOM_H_
#define DLIB_SECURE_RANDOM_H_

#include "secure_random_abstract.h"

#include <cstddef>
#include <string>
#include "crypto_utils.h"

#if defined(_WIN32)
#  include "../windows_magic.h"
#  include <windows.h>
extern "C" BOOLEAN NTAPI SystemFunction036(PVOID RandomBuffer, ULONG RandomBufferLength);
#  if defined(_MSC_VER)
#    pragma comment(lib, "advapi32.lib")
#  endif
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#  include <stdlib.h>
#  define DLIB_HAVE_ARC4RANDOM_BUF 1
#else
#  include <errno.h>
#  include <fcntl.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <sys/syscall.h>
#  endif
#  ifndef O_CLOEXEC
#    define O_CLOEXEC 0
#  endif
#endif

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline void generate_secure_random_bytes(
        void* buffer,
        std::size_t size
    )
    {
        if (size == 0)
            return;
#if defined(_WIN32)
        unsigned char* p = static_cast<unsigned char*>(buffer);
        while (size > 0)
        {
            const ULONG chunk = static_cast<ULONG>(size > 0x10000000 ? 0x10000000 : size);
            if (!SystemFunction036(p, chunk))
                throw crypto_error("generate_secure_random_bytes(): RtlGenRandom failed");
            p += chunk;
            size -= chunk;
        }
#elif defined(DLIB_HAVE_ARC4RANDOM_BUF)
        arc4random_buf(buffer, size);
#else
        // Unbuffered reads only, so that no copy of the bytes is left in a library buffer.
        unsigned char* p = static_cast<unsigned char*>(buffer);
#  if defined(__linux__) && defined(SYS_getrandom)
        while (size > 0)
        {
            const std::size_t chunk = size > 0x1000000 ? 0x1000000 : size;
            const long got = syscall(SYS_getrandom, p, chunk, 0);
            if (got < 0)
            {
                if (errno == EINTR)
                    continue;
                if (errno == ENOSYS)
                    break;      // kernels before 3.17: the device below
                throw crypto_error("generate_secure_random_bytes(): getrandom failed");
            }
            p += got;
            size -= static_cast<std::size_t>(got);
        }
        if (size == 0)
            return;
#  endif
        const int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            throw crypto_error("generate_secure_random_bytes(): cannot open /dev/urandom");
        while (size > 0)
        {
            const ssize_t got = ::read(fd, p, size);
            if (got < 0 && errno == EINTR)
                continue;
            if (got <= 0)
            {
                ::close(fd);
                throw crypto_error("generate_secure_random_bytes(): short read from /dev/urandom");
            }
            p += got;
            size -= static_cast<std::size_t>(got);
        }
        ::close(fd);
#endif
    }

    inline std::string generate_secure_random_bytes(
        std::size_t size
    )
    {
        std::string out(size, '\0');
        if (size > 0)
            generate_secure_random_bytes(&out[0], size);
        return out;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SECURE_RANDOM_H_
