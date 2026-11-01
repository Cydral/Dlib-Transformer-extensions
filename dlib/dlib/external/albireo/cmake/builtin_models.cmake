# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Downloads the files of the models that come with Albireo into the cache of the build,
# each one named by its SHA-256 and checked against it (cmake -P, CMake 3.19 or later):
#
#   cmake -DMANIFEST=<resources/builtin_models.json> -DCACHE=<folder> -P builtin_models.cmake
#
# A model gives the file it is converted from ("from") and, for a model that reads images,
# its projector ("projector"), which Albireo ships as published.  A file already in the
# cache is kept when its SHA-256 matches, and downloaded again otherwise; albireo-builtin
# checks the hash again before it uses it.  The downloads go through the proxy of the
# https_proxy variable, when set, and stop after two minutes without data.

if (CMAKE_VERSION VERSION_LESS 3.19)
   message(FATAL_ERROR "the models that come with Albireo need CMake 3.19 or later")
endif()

# Downloads url into the cache as <sha>.gguf, unless a file of that size and hash is there already.
function(albireo_cache_file label url sha size)
   set(target "${CACHE}/${sha}.gguf")
   if (EXISTS "${target}")
      file(SIZE "${target}" have)
      if (have EQUAL size)
         file(SHA256 "${target}" have_sha)
         string(TOLOWER "${sha}" want_sha)
         if (have_sha STREQUAL want_sha)
            return()
         endif()
         message(STATUS "${target} does not match its SHA-256: downloaded again")
      endif()
      file(REMOVE "${target}")
   endif()
   message(STATUS "Downloading ${label}: ${url}")
   file(DOWNLOAD "${url}" "${target}.partial" EXPECTED_HASH SHA256=${sha} SHOW_PROGRESS TLS_VERIFY ON
      INACTIVITY_TIMEOUT 120 STATUS status)
   list(GET status 0 code)
   if (NOT code EQUAL 0)
      list(GET status 1 text)
      file(REMOVE "${target}.partial")
      message(FATAL_ERROR "${label} could not be downloaded from ${url}: ${text}")
   endif()
   file(RENAME "${target}.partial" "${target}")
endfunction()

file(READ "${MANIFEST}" manifest)
string(JSON count LENGTH "${manifest}" models)
file(MAKE_DIRECTORY "${CACHE}")
math(EXPR last "${count} - 1")
foreach (i RANGE ${last})
   string(JSON id GET "${manifest}" models ${i} id)
   string(JSON url GET "${manifest}" models ${i} from url)
   string(JSON sha GET "${manifest}" models ${i} from sha256)
   string(JSON size GET "${manifest}" models ${i} from size)
   albireo_cache_file("${id}" "${url}" "${sha}" "${size}")
   string(JSON projector ERROR_VARIABLE none GET "${manifest}" models ${i} projector)
   if (NOT none)
      string(JSON url GET "${projector}" url)
      string(JSON sha GET "${projector}" sha256)
      string(JSON size GET "${projector}" size)
      albireo_cache_file("the projector of ${id}" "${url}" "${sha}" "${size}")
   endif()
endforeach()
