# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Packs the programs of a build into the payload of the installer (cmake -P): the service,
# its engines, the icon, the commands of the Explorer, the web application, the licenses
# and the installer of WinFsp, as a ZIP archive whose paths are those of the installation.

file(REMOVE_RECURSE ${STAGE})
file(MAKE_DIRECTORY ${STAGE}/licenses ${STAGE}/winfsp)
file(COPY ${BIN_DIR}/pseudalis-service.exe ${BIN_DIR}/pseudalis-tray.exe ${BIN_DIR}/pseudalis.exe
   ${BIN_DIR}/pseudalis-engine.dll ${BIN_DIR}/pseudalis-engine-avx2.dll DESTINATION ${STAGE})
file(COPY ${BIN_DIR}/web DESTINATION ${STAGE})
# The installer of WinFsp, checked again at the build: it may have changed since the configuration.
file(SHA256 "${WINFSP_MSI}" winfsp_msi_sha256)
if (DEFINED WINFSP_MSI_SHA256 AND NOT winfsp_msi_sha256 STREQUAL WINFSP_MSI_SHA256)
   message(FATAL_ERROR "${WINFSP_MSI} is not the installer of WinFsp that the build expects")
endif()
configure_file(${WINFSP_MSI} ${STAGE}/winfsp/winfsp.msi COPYONLY)
configure_file(${SOURCE_DIR}/resources/license.txt ${STAGE}/licenses/pseudalis.txt COPYONLY)
configure_file(${DLIB_DIR}/../LICENSE.txt ${STAGE}/licenses/dlib.txt COPYONLY)
configure_file(${DLIB_DIR}/external/mbedtls/LICENSE ${STAGE}/licenses/mbedtls.txt COPYONLY)
configure_file(${DLIB_DIR}/external/httplib/LICENSE ${STAGE}/licenses/cpp-httplib.txt COPYONLY)
configure_file(${DLIB_DIR}/external/zlib/README ${STAGE}/licenses/zlib.txt COPYONLY)
configure_file(${DLIB_DIR}/external/sqlite/README.txt ${STAGE}/licenses/sqlite.txt COPYONLY)
configure_file(${SWAGGER_DIR}/LICENSE ${STAGE}/licenses/swagger-ui.txt COPYONLY)
# WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos, https://github.com/winfsp/winfsp
if (EXISTS "${WINFSP_LICENSE}")
   configure_file(${WINFSP_LICENSE} ${STAGE}/licenses/winfsp.txt COPYONLY)
else()
   file(WRITE ${STAGE}/licenses/winfsp.txt "WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos\n"
      "https://github.com/winfsp/winfsp\n\nGPLv3 with the FLOSS exception: see the repository of WinFsp.\n")
endif()
# Where the source of the WinFsp carried lies (GPLv3, section 6).
file(WRITE ${STAGE}/licenses/winfsp-source.txt
   "WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos\n"
   "https://github.com/winfsp/winfsp\n\n"
   "Pseudalis carries the installer of WinFsp 2.1 (build 2.1.25156), winfsp\\winfsp.msi,\n"
   "unchanged, as the WinFsp project released it.  The source code of this version is\n"
   "published by the project at https://github.com/winfsp/winfsp/tree/v2.1 (tag v2.1), and\n"
   "the release at https://github.com/winfsp/winfsp/releases/tag/v2.1.\n")

file(GLOB entries RELATIVE ${STAGE} ${STAGE}/*)
file(REMOVE ${OUTPUT})
execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${STAGE}
   ${CMAKE_COMMAND} -E tar cf ${OUTPUT} --format=zip ${entries}
   RESULT_VARIABLE rc)
if (NOT rc EQUAL 0)
   message(FATAL_ERROR "the payload could not be packed")
endif()
