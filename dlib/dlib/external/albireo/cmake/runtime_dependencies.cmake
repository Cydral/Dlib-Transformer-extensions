# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Copies next to an engine the libraries it needs at run time, other than those of the
# system, so that the engines folder of an installation is complete:
#
#   cmake -DENGINE=<engine library> -DDEST=<folder> -DDIRS=<dir|dir|...> [-DEXTRA=<glob|glob|...>]
#         [-DEXCLUDE=<regex|regex|...>] -P runtime_dependencies.cmake
#
# DIRS lists the folders where the libraries are searched, such as the bin folder of the
# CUDA toolkit or the libs folder of OpenVINO.  EXTRA lists the libraries the engine loads
# by name while it runs, which no import table shows: the plugins of OpenVINO, the parts of
# cuDNN, the runtime of the compiler.  Their own dependencies are copied as well.  EXCLUDE
# lists the names of the libraries left out, those the engine loads on first use only.

if (NOT ENGINE OR NOT DEST)
   message(FATAL_ERROR "runtime_dependencies.cmake needs ENGINE and DEST")
endif()
string(REPLACE "|" ";" dirs "${DIRS}")
string(REPLACE "|" ";" extra "${EXTRA}")
string(REPLACE "|" ";" exclude "${EXCLUDE}")

set(extra_files)
foreach (pattern IN LISTS extra)
   if (pattern)
      file(GLOB matches "${pattern}")
      list(APPEND extra_files ${matches})
   endif()
endforeach()

file(GET_RUNTIME_DEPENDENCIES
   LIBRARIES ${ENGINE} ${extra_files}
   RESOLVED_DEPENDENCIES_VAR resolved
   UNRESOLVED_DEPENDENCIES_VAR unresolved
   DIRECTORIES ${dirs}
   PRE_EXCLUDE_REGEXES ${exclude} "^api-ms-.*" "^ext-ms-.*" "^hvsifiletrust\\.dll$" "^ld-linux.*" "^libc\\.so.*"
      "^libm\\.so.*" "^libdl\\.so.*" "^libpthread\\.so.*" "^librt\\.so.*" "^libgcc_s\\.so.*" "^libstdc\\+\\+\\.so.*"
   POST_EXCLUDE_REGEXES ".*[/\\\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\].*" "^/lib/.*" "^/lib64/.*" "^/usr/lib/.*"
      "^/usr/lib64/.*")

set(copied 0)
foreach (f IN LISTS resolved extra_files)
   get_filename_component(name "${f}" NAME)
   if (NOT EXISTS "${DEST}/${name}" OR "${f}" IS_NEWER_THAN "${DEST}/${name}")
      file(COPY "${f}" DESTINATION "${DEST}")
      math(EXPR copied "${copied} + 1")
   endif()
endforeach()
message(STATUS "Run time libraries of ${ENGINE}: ${copied} copied to ${DEST}")
foreach (u IN LISTS unresolved)
   # Libraries of the driver, such as nvcuda.dll, come with the system of the user.
   if (NOT u MATCHES "^(nvcuda|nvml|OpenCL|ze_loader|ze_intel_.*|libcuda|libOpenCL)")
      message(WARNING "${ENGINE} needs ${u}, found in none of: ${dirs}")
   endif()
endforeach()
