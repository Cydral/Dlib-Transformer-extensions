# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Packs the programs of a build into the payload of the installer (cmake -P, run in the
# folder of the payload): the service, the tray, the engines, the web application and the
# licenses, as a ZIP archive whose paths are those of the installation.

file(REMOVE_RECURSE ${STAGE})
file(MAKE_DIRECTORY ${STAGE}/licenses)
file(COPY ${BIN_DIR}/albireo-service.exe ${BIN_DIR}/albireo-tray.exe DESTINATION ${STAGE})
file(COPY ${BIN_DIR}/engines DESTINATION ${STAGE} FOLLOW_SYMLINK_CHAIN)
file(COPY ${SOURCE_DIR}/web DESTINATION ${STAGE})
# The engines of the other build trees; the licenses of their runtimes, in their licenses
# folder, join those of the installation.
foreach (dir ${EXTRA_ENGINE_DIRS})
   if (IS_DIRECTORY "${dir}")
      file(GLOB extra LIST_DIRECTORIES false "${dir}/*")
      if (extra)
         file(COPY ${extra} DESTINATION ${STAGE}/engines)
      endif()
      if (IS_DIRECTORY "${dir}/licenses")
         file(GLOB licenses "${dir}/licenses/*")
         file(COPY ${licenses} DESTINATION ${STAGE}/licenses)
      endif()
   endif()
endforeach()
configure_file(${SOURCE_DIR}/resources/license.txt ${STAGE}/licenses/albireo.txt COPYONLY)
configure_file(${DLIB_DIR}/../LICENSE.txt ${STAGE}/licenses/dlib.txt COPYONLY)
configure_file(${DLIB_DIR}/external/mbedtls/LICENSE ${STAGE}/licenses/mbedtls.txt COPYONLY)
configure_file(${DLIB_DIR}/external/httplib/LICENSE ${STAGE}/licenses/cpp-httplib.txt COPYONLY)
configure_file(${DLIB_DIR}/external/zlib/README ${STAGE}/licenses/zlib.txt COPYONLY)
configure_file(${DLIB_DIR}/external/sqlite/README.txt ${STAGE}/licenses/sqlite.txt COPYONLY)

file(GLOB entries RELATIVE ${STAGE} ${STAGE}/*)
file(REMOVE ${OUTPUT})
execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${STAGE}
   ${CMAKE_COMMAND} -E tar cf ${OUTPUT} --format=zip ${entries}
   RESULT_VARIABLE rc)
if (NOT rc EQUAL 0)
   message(FATAL_ERROR "the payload could not be packed")
endif()
