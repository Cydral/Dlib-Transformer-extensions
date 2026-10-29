# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Cross compilation for 64-bit Windows with MinGW-w64 (POSIX thread model), from Linux:
#
#   cmake -S dlib/external/albireo -B build-albireo-win
#         -DCMAKE_TOOLCHAIN_FILE=dlib/external/albireo/cmake/mingw-w64.cmake

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(ALBIREO_MINGW_PREFIX x86_64-w64-mingw32)
set(CMAKE_C_COMPILER ${ALBIREO_MINGW_PREFIX}-gcc-posix)
set(CMAKE_CXX_COMPILER ${ALBIREO_MINGW_PREFIX}-g++-posix)
set(CMAKE_RC_COMPILER ${ALBIREO_MINGW_PREFIX}-windres)
set(CMAKE_FIND_ROOT_PATH /usr/${ALBIREO_MINGW_PREFIX})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
