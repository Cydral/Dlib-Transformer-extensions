# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# The key of the Brave Search API built into the service (host/web_access.h), included by CMakeLists.txt.  The
# key is read at each configuration from ALBIREO_WEB_SEARCH_KEY_FILE, a file outside the repository, without its
# byte order mark and its ends of line, and written masked, with a seed of its own, into a header of the build
# tree alone (generated/web_search_secret.h).  Without the file, the build carries no key, and the search of the
# service goes to Wikipedia by default.  Every installation of a build that carries the key shares its monthly
# credit.  No message of the configuration gives the key.

if (CMAKE_HOST_WIN32 AND DEFINED ENV{USERPROFILE})
   file(TO_CMAKE_PATH "$ENV{USERPROFILE}/keys/albireo-brave-search-key.txt" web_search_key_default)
else()
   set(web_search_key_default "$ENV{HOME}/keys/albireo-brave-search-key.txt")
endif()
set(ALBIREO_WEB_SEARCH_KEY_FILE "${web_search_key_default}" CACHE FILEPATH
   "File holding the key of the Brave Search API built into the service")
set(web_search_key_bytes "0")
set(web_search_key_size 0)
if (EXISTS "${ALBIREO_WEB_SEARCH_KEY_FILE}")
   set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ALBIREO_WEB_SEARCH_KEY_FILE}")
   file(READ "${ALBIREO_WEB_SEARCH_KEY_FILE}" web_search_key_hex HEX)
   # A byte order mark before the key and the ends of line after it are not part of it.
   string(REGEX REPLACE "^efbbbf" "" web_search_key_hex "${web_search_key_hex}")
   string(REGEX REPLACE "(0d|0a)+$" "" web_search_key_hex "${web_search_key_hex}")
   string(LENGTH "${web_search_key_hex}" web_search_key_digits)
   math(EXPR web_search_key_size "${web_search_key_digits} / 2")
   if (web_search_key_size GREATER 0)
      # Each byte masked by the generator that host/masked_secret.h runs again, from the seed of host/web_access.cpp.
      set(web_search_key_seed 20261102)
      set(web_search_key_list "")
      math(EXPR web_search_key_last "${web_search_key_size} - 1")
      foreach (i RANGE 0 ${web_search_key_last})
         math(EXPR web_search_key_at "${i} * 2")
         string(SUBSTRING "${web_search_key_hex}" ${web_search_key_at} 2 web_search_key_byte)
         math(EXPR web_search_key_seed "(${web_search_key_seed} * 1103515245 + 12345) % 2147483648")
         math(EXPR web_search_key_byte "0x${web_search_key_byte} ^ ((${web_search_key_seed} >> 16) & 255)")
         list(APPEND web_search_key_list ${web_search_key_byte})
      endforeach()
      string(REPLACE ";" ", " web_search_key_bytes "${web_search_key_list}")
      message(STATUS "Web search: the key of the Brave Search API is read from ${ALBIREO_WEB_SEARCH_KEY_FILE} "
         "and built into the service")
   else()
      message(STATUS "Web search: no key of the Brave Search API built in, ${ALBIREO_WEB_SEARCH_KEY_FILE} is empty")
   endif()
   unset(web_search_key_hex)
   unset(web_search_key_list)
else()
   message(STATUS "Web search: no key of the Brave Search API built in, ${ALBIREO_WEB_SEARCH_KEY_FILE} does not "
      "exist")
endif()
configure_file(${CMAKE_CURRENT_LIST_DIR}/web_search_secret.h.in ${CMAKE_BINARY_DIR}/generated/web_search_secret.h
   @ONLY)
