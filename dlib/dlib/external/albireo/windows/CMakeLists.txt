# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# The Windows programs: the tray application and the installers.  The installer carries
# the programs of the build as a ZIP payload; engines built in other build trees (CUDA,
# OpenVINO) join it through ALBIREO_EXTRA_ENGINE_DIRS, a list of folders whose content
# goes to the engines folder of the installation.  albireo-setup-online.exe is that
# program, which the updates take; albireo-setup.exe adds the models that come with
# Albireo, for an installation complete without network.

set(ALBIREO_EXTRA_ENGINE_DIRS "" CACHE STRING "Folders of engines built elsewhere, added to the installer")

set(ui_libraries comctl32 windowscodecs ole32 oleaut32 shell32 uuid advapi32)

add_executable(albireo-tray WIN32 tray.cpp tray_windows.cpp ui.cpp strings.cpp service_client.cpp tray.rc)
target_link_libraries(albireo-tray PRIVATE albireo_common dlib::httplib ${ui_libraries})
target_compile_options(albireo-tray PRIVATE ${ALBIREO_WARNINGS})

set(ALBIREO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ALBIREO_WINDOWS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ALBIREO_PAYLOAD ${CMAKE_BINARY_DIR}/payload/albireo-payload.zip)
file(GLOB_RECURSE web_files ${ALBIREO_SOURCE_DIR}/web/*)
set(processor_engines albireo-engine-cpu albireo-engine-cpu-avx2)
foreach (e albireo-engine-cpu-avxvnni albireo-engine-cpu-avx512)
   if (TARGET ${e})
      list(APPEND processor_engines ${e})
   endif()
endforeach()
add_custom_command(OUTPUT ${ALBIREO_PAYLOAD}
   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/payload/stage
   COMMAND ${CMAKE_COMMAND} -DBIN_DIR=${CMAKE_BINARY_DIR}/bin -DSTAGE=${CMAKE_BINARY_DIR}/payload/stage
      -DOUTPUT=${ALBIREO_PAYLOAD} -DSOURCE_DIR=${ALBIREO_SOURCE_DIR} -DDLIB_DIR=${ALBIREO_DLIB_DIR}
      "-DEXTRA_ENGINE_DIRS=${ALBIREO_EXTRA_ENGINE_DIRS}" -P ${ALBIREO_SOURCE_DIR}/cmake/payload.cmake
   DEPENDS albireo-service albireo-tray ${processor_engines} ${web_files}
      ${ALBIREO_SOURCE_DIR}/cmake/payload.cmake ${ALBIREO_SOURCE_DIR}/resources/license.txt
   COMMENT "Packing the programs into the payload of the installer")
add_custom_target(albireo-payload DEPENDS ${ALBIREO_PAYLOAD})
add_dependencies(albireo-payload albireo-web)

configure_file(setup.rc.in ${CMAKE_BINARY_DIR}/rc/setup.rc @ONLY)
set_source_files_properties(${CMAKE_BINARY_DIR}/rc/setup.rc PROPERTIES OBJECT_DEPENDS "${ALBIREO_PAYLOAD}")
add_executable(albireo-setup-online WIN32 setup.cpp setup_actions.cpp zip_reader.cpp ui.cpp strings.cpp
   ${ALBIREO_SOURCE_DIR}/host/certificates.cpp ${ALBIREO_SOURCE_DIR}/host/crypto.cpp ${CMAKE_BINARY_DIR}/rc/setup.rc)
add_dependencies(albireo-setup-online albireo-payload)
target_link_libraries(albireo-setup-online PRIVATE albireo_common albireo_zlib dlib::httplib ${ui_libraries} crypt32
   ws2_32)
target_compile_options(albireo-setup-online PRIVATE ${ALBIREO_WARNINGS})

# The installer without models, which the updates take, and the complete installer: the
# same program followed by the models that come with Albireo (common/builtin_models.h), or
# its copy when the build prepares no models.  No target bears the name of the complete
# installer, which CMake would otherwise take for the file of that target.
set(complete_setup ${CMAKE_BINARY_DIR}/bin/albireo-setup.exe)
if (ALBIREO_BUILTIN_MODELS)
   add_custom_command(OUTPUT ${complete_setup}
      COMMAND ${ALBIREO_BUILTIN_COMMAND} --manifest ${ALBIREO_BUILTIN_MANIFEST} --cache ${ALBIREO_MODEL_CACHE}
         --setup $<TARGET_FILE:albireo-setup-online> --setup-out ${complete_setup}
      DEPENDS albireo-setup-online ${ALBIREO_BUILTIN_STAMP}
      COMMENT "Appending the models that come with Albireo to the installer"
      VERBATIM)
else()
   add_custom_command(OUTPUT ${complete_setup}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:albireo-setup-online> ${complete_setup}
      DEPENDS albireo-setup-online
      VERBATIM)
endif()
add_custom_target(albireo-setup-complete ALL DEPENDS ${complete_setup})

foreach (t albireo-tray albireo-setup-online)
   if (MSVC)
      # The manifest is a resource of the program; the libraries it imports are loaded from
      # System32 only (LOAD_LIBRARY_SEARCH_SYSTEM32), never from the folder of the program.
      target_link_options(${t} PRIVATE /MANIFEST:NO /DEPENDENTLOADFLAG:0x800)
   elseif (MINGW)
      target_link_options(${t} PRIVATE -municode)
   endif()
endforeach()

# The installer runs elevated from any folder, a folder of downloads for instance.  MinGW has
# no /DEPENDENTLOADFLAG: the system libraries it imports that Windows does not load from
# System32 by itself (bcrypt, crypt32) are delay-loaded instead, at their first call, once
# SetDefaultDllDirectories() limits the search to System32.  The delay-import libraries are
# made from the import libraries of the toolchain and take their names, ahead of them.
if (MINGW)
   set(delay_dir ${CMAKE_BINARY_DIR}/delay-import)
   file(MAKE_DIRECTORY ${delay_dir})
   set(delay_libraries)
   foreach (lib bcrypt crypt32)
      find_library(ALBIREO_IMPORT_${lib} ${lib})
      if (NOT ALBIREO_IMPORT_${lib} OR NOT CMAKE_DLLTOOL OR NOT CMAKE_NM)
         message(FATAL_ERROR "The delay-import library of ${lib} needs the import library, nm and dlltool of MinGW")
      endif()
      execute_process(COMMAND ${CMAKE_NM} -g --defined-only ${ALBIREO_IMPORT_${lib}} OUTPUT_VARIABLE symbols)
      string(REGEX MATCHALL "__imp_[A-Za-z0-9_]+" imports "${symbols}")
      list(REMOVE_DUPLICATES imports)
      list(TRANSFORM imports REPLACE "^__imp_" "")
      list(JOIN imports "\n" exports)
      file(WRITE ${delay_dir}/${lib}.def "LIBRARY ${lib}.dll\nEXPORTS\n${exports}\n")
      add_custom_command(OUTPUT ${delay_dir}/lib${lib}.a
         COMMAND ${CMAKE_DLLTOOL} -d ${delay_dir}/${lib}.def -y ${delay_dir}/lib${lib}.a
         DEPENDS ${delay_dir}/${lib}.def
         COMMENT "Making the delay-import library of ${lib}.dll")
      list(APPEND delay_libraries ${delay_dir}/lib${lib}.a)
   endforeach()
   add_custom_target(albireo-delay-import DEPENDS ${delay_libraries})
   add_dependencies(albireo-setup-online albireo-delay-import)
   target_link_directories(albireo-setup-online BEFORE PRIVATE ${delay_dir})
   set_property(TARGET albireo-setup-online APPEND PROPERTY LINK_DEPENDS ${delay_libraries})
endif()
