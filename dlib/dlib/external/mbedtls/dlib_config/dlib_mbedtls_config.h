// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/* Adjustments to the default configuration of the Mbed TLS copy bundled with dlib.

   Mbed TLS reads this file after include/mbedtls/mbedtls_config.h because the dlib_mbedtls
   CMake target defines MBEDTLS_USER_CONFIG_FILE to name it.  Every translation unit that
   includes an Mbed TLS header must see the same configuration as the library, which is why
   that definition is a public property of the target rather than a private one. */

#ifndef DLIB_MBEDTLS_CONFIG_H
#define DLIB_MBEDTLS_CONFIG_H

/* Thread safety.  TLS 1.3 goes through the PSA cryptography layer, whose key store is
   global, so it has to be protected as soon as two threads hold TLS sessions, which is the
   case of every server that serves connections from a thread pool.  POSIX threads are used
   where the compiler provides them, the Win32 implementation of
   dlib_mbedtls_threading_win32.c otherwise. */
#define MBEDTLS_THREADING_C
#if defined(_MSC_VER)
#define MBEDTLS_THREADING_ALT
#else
#define MBEDTLS_THREADING_PTHREAD
#endif

/* The programs built on this copy open, read and write their sockets through
   cpp-httplib, which hands Mbed TLS its own send and receive callbacks. */
#undef MBEDTLS_NET_C

#endif /* DLIB_MBEDTLS_CONFIG_H */
