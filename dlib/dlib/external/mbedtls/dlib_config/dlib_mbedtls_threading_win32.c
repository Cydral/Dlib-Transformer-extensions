// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/* Win32 implementation of the Mbed TLS threading layer, for the compilers that provide no
   POSIX threads.  Mbed TLS calls these functions through pointers that
   mbedtls_threading_set_alt() sets, and fails every lock until it is called, so the
   registration runs before the static constructors of the program: a global object that
   opens a TLS session while the program starts finds the mutexes in place. */

#include "mbedtls/build_info.h"

#if defined(MBEDTLS_THREADING_ALT) && defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "mbedtls/threading.h"

static void dlib_mbedtls_mutex_init(mbedtls_threading_mutex_t* mutex)
{
    if (mutex == NULL)
        return;
    InitializeSRWLock((PSRWLOCK)&mutex->lock);
    mutex->is_valid = 1;
}

static void dlib_mbedtls_mutex_free(mbedtls_threading_mutex_t* mutex)
{
    if (mutex == NULL)
        return;
    mutex->is_valid = 0;
}

static int dlib_mbedtls_mutex_lock(mbedtls_threading_mutex_t* mutex)
{
    if (mutex == NULL || !mutex->is_valid)
        return MBEDTLS_ERR_THREADING_BAD_INPUT_DATA;
    AcquireSRWLockExclusive((PSRWLOCK)&mutex->lock);
    return 0;
}

static int dlib_mbedtls_mutex_unlock(mbedtls_threading_mutex_t* mutex)
{
    if (mutex == NULL || !mutex->is_valid)
        return MBEDTLS_ERR_THREADING_BAD_INPUT_DATA;
    ReleaseSRWLockExclusive((PSRWLOCK)&mutex->lock);
    return 0;
}

static void __cdecl dlib_mbedtls_threading_win32_setup(void)
{
    mbedtls_threading_set_alt(dlib_mbedtls_mutex_init, dlib_mbedtls_mutex_free,
                              dlib_mbedtls_mutex_lock, dlib_mbedtls_mutex_unlock);
}

#if defined(_MSC_VER)
/* .CRT$XCT sorts before .CRT$XCU, where the compiler places the dynamic initializers of
   C++ globals, so the registration precedes them. */
#pragma section(".CRT$XCT", read)
__declspec(allocate(".CRT$XCT")) void (__cdecl *dlib_mbedtls_threading_win32_initializer)(void) =
    dlib_mbedtls_threading_win32_setup;
#else
void (*dlib_mbedtls_threading_win32_initializer)(void) = dlib_mbedtls_threading_win32_setup;
__attribute__((constructor(101))) static void dlib_mbedtls_threading_win32_constructor(void)
{
    dlib_mbedtls_threading_win32_setup();
}
#endif

#else

/* ISO C forbids an empty translation unit. */
typedef int dlib_mbedtls_threading_win32_unused;

#endif
