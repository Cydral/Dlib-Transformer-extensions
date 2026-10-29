Mbed TLS bundled with dlib
==========================

Version   : 3.6.7 (long-term support branch 3.6)
Origin    : https://github.com/Mbed-TLS/mbedtls, tag v3.6.7
            (commit 068ff080b369adfac81509f9b57b2afabaf82dc5)
License   : Apache-2.0 OR GPL-2.0-or-later, see LICENSE in this folder; dlib uses it under
            the Apache License 2.0

What this folder holds
----------------------

   include/       the public headers of the release (include/mbedtls, include/psa), as released
   library/       the sources of the library (*.c, *.h), as released
   dlib_config/   the files dlib adds:
                     dlib_mbedtls_config.h            adjustments to the default configuration
                     threading_alt.h                  mutex type of the Win32 threading layer
                     dlib_mbedtls_threading_win32.c   Win32 threading layer for MSVC

The build files, the tests, the programs, the scripts and the third party drivers of the
release are left out: dlib compiles the library with its own CMake target, and the default
configuration enables none of the third party drivers.

How it is used
--------------

The sources are compiled into the static library dlib_mbedtls, which the dlib::httplib target
links for its HTTPS support (see dlib/external/httplib/README.txt).  In a project build, the
library is compiled only when a program links dlib::httplib.  The option DLIB_HTTPLIB_TLS, ON by
default, decides whether dlib::httplib uses it.

Configuration
-------------

The target defines MBEDTLS_USER_CONFIG_FILE as "dlib_mbedtls_config.h", publicly, so that the
library and every translation unit that includes an Mbed TLS header see the same configuration.
That file changes the default configuration of the release in two places:

   - MBEDTLS_THREADING_C, with POSIX threads where the compiler provides them and the Win32
     layer of dlib_mbedtls_threading_win32.c with MSVC.  TLS 1.3 goes through the PSA layer,
     whose key store is global, and a server serves its connections from several threads.
   - MBEDTLS_NET_C is removed: cpp-httplib reads and writes the sockets itself and hands Mbed TLS
     its own send and receive callbacks.

The Win32 threading layer registers itself before the static constructors of a program run,
since Mbed TLS fails every lock until mbedtls_threading_set_alt() is called.  threading_alt.h
names its initializer to the MSVC linker, so that the object file is taken out of the static
library by every program that includes an Mbed TLS header.

Tests
-----

test_httplib in the dtest suite exercises the library through cpp-httplib: TLS 1.2 and 1.3
handshakes, certificate and host name verification, and concurrent sessions.

Updating
--------

Copy the include and library folders of a newer release of the 3.6 branch over these ones
(without their build files), update the version and origin above, check that the options of
dlib_mbedtls_config.h still exist under the same names, and run the dtest suite.
