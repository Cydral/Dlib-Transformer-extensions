# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# The Windows programs: the tray application and the installer.  The installer carries
# the programs of the build as a ZIP payload; engines built in other build trees (CUDA,
# OpenVINO) join it through ALBIREO_EXTRA_ENGINE_DIRS, a list of folders whose content
# goes to the engines folder of the installation.

set(ALBIREO_EXTRA_ENGINE_DIRS "" CACHE STRING "Folders of engines built elsewhere, added to the installer")

set(ui_libraries comctl32 windowscodecs msimg32 ole32 oleaut32 shell32 uuid advapi32)

add_executable(albireo-tray WIN32 tray.cpp tray_windows.cpp ui.cpp strings.cpp service_client.cpp tray.rc)
target_link_libraries(albireo-tray PRIVATE albireo_common dlib::httplib ${ui_libraries})
target_compile_options(albireo-tray PRIVATE ${ALBIREO_WARNINGS})

set(ALBIREO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ALBIREO_WINDOWS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ALBIREO_PAYLOAD ${CMAKE_BINARY_DIR}/payload/albireo-payload.zip)
file(GLOB_RECURSE web_files ${ALBIREO_SOURCE_DIR}/web/*)
add_custom_command(OUTPUT ${ALBIREO_PAYLOAD}
   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/payload/stage
   COMMAND ${CMAKE_COMMAND} -DBIN_DIR=${CMAKE_BINARY_DIR}/bin -DSTAGE=${CMAKE_BINARY_DIR}/payload/stage
      -DOUTPUT=${ALBIREO_PAYLOAD} -DSOURCE_DIR=${ALBIREO_SOURCE_DIR} -DDLIB_DIR=${ALBIREO_DLIB_DIR}
      "-DEXTRA_ENGINE_DIRS=${ALBIREO_EXTRA_ENGINE_DIRS}" -P ${ALBIREO_SOURCE_DIR}/cmake/payload.cmake
   DEPENDS albireo-service albireo-tray albireo-engine-cpu albireo-engine-cpu-avx2 ${web_files}
      ${ALBIREO_SOURCE_DIR}/cmake/payload.cmake ${ALBIREO_SOURCE_DIR}/resources/license.txt
   COMMENT "Packing the programs into the payload of the installer")
add_custom_target(albireo-payload DEPENDS ${ALBIREO_PAYLOAD})
add_dependencies(albireo-payload albireo-web)

configure_file(setup.rc.in ${CMAKE_BINARY_DIR}/rc/setup.rc @ONLY)
set_source_files_properties(${CMAKE_BINARY_DIR}/rc/setup.rc PROPERTIES OBJECT_DEPENDS "${ALBIREO_PAYLOAD}")
add_executable(albireo-setup WIN32 setup.cpp setup_actions.cpp zip_reader.cpp ui.cpp strings.cpp
   ${ALBIREO_SOURCE_DIR}/host/certificates.cpp ${ALBIREO_SOURCE_DIR}/host/crypto.cpp ${CMAKE_BINARY_DIR}/rc/setup.rc)
add_dependencies(albireo-setup albireo-payload)
target_link_libraries(albireo-setup PRIVATE albireo_common albireo_zlib dlib::httplib ${ui_libraries} crypt32 ws2_32)
target_compile_options(albireo-setup PRIVATE ${ALBIREO_WARNINGS})

foreach (t albireo-tray albireo-setup)
   if (MSVC)
      # The manifest is a resource of the program.
      target_link_options(${t} PRIVATE /MANIFEST:NO)
   elseif (MINGW)
      target_link_options(${t} PRIVATE -municode)
   endif()
endforeach()
