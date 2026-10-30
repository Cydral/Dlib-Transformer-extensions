# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Downloads the files of the models that come with Albireo into the cache of the build,
# each one named by its SHA-256 and checked against it (cmake -P, CMake 3.19 or later):
#
#   cmake -DMANIFEST=<resources/builtin_models.json> -DCACHE=<folder> -P builtin_models.cmake
#
# A file already in the cache is kept; albireo-builtin then checks its hash before it
# converts it.  The downloads go through the proxy of the https_proxy variable, when set.

if (CMAKE_VERSION VERSION_LESS 3.19)
   message(FATAL_ERROR "the models that come with Albireo need CMake 3.19 or later")
endif()
file(READ "${MANIFEST}" manifest)
string(JSON count LENGTH "${manifest}" models)
file(MAKE_DIRECTORY "${CACHE}")
math(EXPR last "${count} - 1")
foreach (i RANGE ${last})
   string(JSON id GET "${manifest}" models ${i} id)
   string(JSON url GET "${manifest}" models ${i} from url)
   string(JSON sha GET "${manifest}" models ${i} from sha256)
   string(JSON size GET "${manifest}" models ${i} from size)
   set(target "${CACHE}/${sha}.gguf")
   if (EXISTS "${target}")
      file(SIZE "${target}" have)
      if (have EQUAL size)
         continue()
      endif()
      file(REMOVE "${target}")
   endif()
   message(STATUS "Downloading ${id}: ${url}")
   file(DOWNLOAD "${url}" "${target}.partial" EXPECTED_HASH SHA256=${sha} SHOW_PROGRESS TLS_VERIFY ON
      STATUS status)
   list(GET status 0 code)
   if (NOT code EQUAL 0)
      list(GET status 1 text)
      file(REMOVE "${target}.partial")
      message(FATAL_ERROR "${id} could not be downloaded from ${url}: ${text}")
   endif()
   file(RENAME "${target}.partial" "${target}")
endforeach()
