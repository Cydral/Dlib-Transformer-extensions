// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/* Mutex type of the Win32 implementation of the Mbed TLS threading layer, included by
   mbedtls/threading.h when MBEDTLS_THREADING_ALT is defined (see dlib_mbedtls_config.h). */

#ifndef DLIB_MBEDTLS_THREADING_ALT_H
#define DLIB_MBEDTLS_THREADING_ALT_H

typedef struct mbedtls_threading_mutex_t
{
    void* lock;         /* an SRWLOCK, which holds exactly one pointer */
    char is_valid;
} mbedtls_threading_mutex_t;

/* dlib_mbedtls_threading_win32.c registers the implementation before the static
   constructors of the program run.  Naming its initializer here makes the linker take that
   object file out of the static library for every program that includes an Mbed TLS
   header, since nothing else refers to it. */
#if defined(_MSC_VER)
#if defined(_M_IX86)
#pragma comment(linker, "/include:_dlib_mbedtls_threading_win32_initializer")
#else
#pragma comment(linker, "/include:dlib_mbedtls_threading_win32_initializer")
#endif
#endif

#endif /* DLIB_MBEDTLS_THREADING_ALT_H */
