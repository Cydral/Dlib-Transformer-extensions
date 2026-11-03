# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# The Windows programs: the icon (pseudalis-tray.exe), the commands of the Explorer
# (pseudalis.exe) and the installer (pseudalis-setup.exe), which carries the programs of
# the build, the web application and the licenses as a ZIP payload.

set(ui_libraries comctl32 windowscodecs ole32 oleaut32 shell32 uuid advapi32)

set(PSEUDALIS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(PSEUDALIS_WINDOWS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(pseudalis-tray WIN32 tray.cpp ui.cpp strings.cpp service_client.cpp tray.rc)
target_link_libraries(pseudalis-tray PRIVATE pseudalis_common dlib::httplib ${ui_libraries})
target_compile_options(pseudalis-tray PRIVATE ${PSEUDALIS_WARNINGS})
# The images the resources embed: the Makefile and Ninja generators follow only the files they include.
set(tray_images about.png mark.png pseudalis.ico tray.ico tray-off.ico tray-update.ico)
list(TRANSFORM tray_images PREPEND ${PSEUDALIS_SOURCE_DIR}/resources/)
set_source_files_properties(tray.rc PROPERTIES OBJECT_DEPENDS "${tray_images}")

add_executable(pseudalis-shell WIN32 shell.cpp ui.cpp strings.cpp service_client.cpp
   ${PSEUDALIS_SOURCE_DIR}/host/crypto.cpp shell.rc)
set_target_properties(pseudalis-shell PROPERTIES OUTPUT_NAME pseudalis)
target_include_directories(pseudalis-shell PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(pseudalis-shell PRIVATE pseudalis_common dlib::httplib ${ui_libraries} crypt32)
target_compile_options(pseudalis-shell PRIVATE ${PSEUDALIS_WARNINGS})
set_source_files_properties(shell.rc PROPERTIES OBJECT_DEPENDS ${PSEUDALIS_SOURCE_DIR}/resources/pseudalis.ico)

# The libraries the programs and the engines import: those of Windows alone, the runtime of the
# compiler linked statically, so that the installer carries no library of a third party
# (tools/check_imports.cpp, through cmake/check_imports.cmake).  The check runs on Windows, or
# under the emulator of a cross build (CMAKE_CROSSCOMPILING_EMULATOR, Wine for instance).
add_executable(pseudalis-check-imports ${PSEUDALIS_SOURCE_DIR}/tools/check_imports.cpp)
target_compile_options(pseudalis-check-imports PRIVATE ${PSEUDALIS_WARNINGS})
set(check_imports)
if (NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR)
   set(check_command ${CMAKE_COMMAND} -DCHECK=$<TARGET_FILE:pseudalis-check-imports>
      "-DEMULATOR=${CMAKE_CROSSCOMPILING_EMULATOR}")
   set(checked $<TARGET_FILE:pseudalis-service> $<TARGET_FILE:pseudalis-tray> $<TARGET_FILE:pseudalis-shell>
      $<TARGET_FILE:pseudalis-engine> $<TARGET_FILE:pseudalis-engine-avx2>)
   list(JOIN checked "|" checked)
   set(check_imports COMMAND ${check_command} "-DFILES=${checked}" -P ${PSEUDALIS_SOURCE_DIR}/cmake/check_imports.cmake)
else()
   message(STATUS "Libraries of the programs: not checked, this cross build runs no program of Windows "
      "(CMAKE_CROSSCOMPILING_EMULATOR)")
endif()

# The payload of the installer.
set(PSEUDALIS_PAYLOAD ${CMAKE_BINARY_DIR}/payload/pseudalis-payload.zip)
# The build checks the list again, so that a file added to or removed from web needs no new configuration.
file(GLOB_RECURSE web_files CONFIGURE_DEPENDS ${PSEUDALIS_SOURCE_DIR}/web/*)
add_custom_command(OUTPUT ${PSEUDALIS_PAYLOAD}
   ${check_imports}
   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/payload/stage
   COMMAND ${CMAKE_COMMAND} -DBIN_DIR=${CMAKE_BINARY_DIR}/bin -DSTAGE=${CMAKE_BINARY_DIR}/payload/stage
      -DOUTPUT=${PSEUDALIS_PAYLOAD} -DSOURCE_DIR=${PSEUDALIS_SOURCE_DIR} -DDLIB_DIR=${PSEUDALIS_DLIB_DIR}
      -DSWAGGER_DIR=${PSEUDALIS_ALBIREO_DIR}/web/docs
      -P ${PSEUDALIS_SOURCE_DIR}/cmake/payload.cmake
   DEPENDS pseudalis-service pseudalis-tray pseudalis-shell pseudalis-engine pseudalis-engine-avx2
      pseudalis-check-imports ${web_files}
      ${PSEUDALIS_SOURCE_DIR}/cmake/payload.cmake ${PSEUDALIS_SOURCE_DIR}/cmake/check_imports.cmake
      ${PSEUDALIS_SOURCE_DIR}/resources/license.txt
   COMMENT "Packing the programs into the payload of the installer"
   # The list of the files checked holds '|', which a shell would read without quotes.
   VERBATIM)
add_custom_target(pseudalis-payload DEPENDS ${PSEUDALIS_PAYLOAD})
add_dependencies(pseudalis-payload pseudalis-web)

configure_file(setup.rc.in ${CMAKE_BINARY_DIR}/rc/setup.rc @ONLY)
set(setup_resources ${PSEUDALIS_PAYLOAD} ${PSEUDALIS_SOURCE_DIR}/resources/setup-side.png
   ${PSEUDALIS_SOURCE_DIR}/resources/pseudalis.ico ${PSEUDALIS_SOURCE_DIR}/resources/license.txt)
set_source_files_properties(${CMAKE_BINARY_DIR}/rc/setup.rc PROPERTIES OBJECT_DEPENDS "${setup_resources}")
add_executable(pseudalis-setup WIN32 setup.cpp setup_actions.cpp zip_reader.cpp ui.cpp strings.cpp
   ${PSEUDALIS_SOURCE_DIR}/host/certificates.cpp ${PSEUDALIS_SOURCE_DIR}/host/crypto.cpp
   ${CMAKE_BINARY_DIR}/rc/setup.rc)
add_dependencies(pseudalis-setup pseudalis-payload)
target_include_directories(pseudalis-setup PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(pseudalis-setup PRIVATE pseudalis_common pseudalis_zlib dlib::httplib ${ui_libraries} crypt32
   ws2_32)
target_compile_options(pseudalis-setup PRIVATE ${PSEUDALIS_WARNINGS})
if (check_imports)
   add_dependencies(pseudalis-setup pseudalis-check-imports)
   add_custom_command(TARGET pseudalis-setup POST_BUILD
      COMMAND ${check_command} -DFILES=$<TARGET_FILE:pseudalis-setup> -DREMOVE=$<TARGET_FILE:pseudalis-setup>
         -P ${PSEUDALIS_SOURCE_DIR}/cmake/check_imports.cmake
      COMMENT "Checking the libraries the installer imports"
      VERBATIM)
endif()

foreach (t pseudalis-tray pseudalis-shell pseudalis-setup)
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
      find_library(PSEUDALIS_IMPORT_${lib} ${lib})
      if (NOT PSEUDALIS_IMPORT_${lib} OR NOT CMAKE_DLLTOOL OR NOT CMAKE_NM)
         message(FATAL_ERROR "The delay-import library of ${lib} needs the import library, nm and dlltool of MinGW")
      endif()
      execute_process(COMMAND ${CMAKE_NM} -g --defined-only ${PSEUDALIS_IMPORT_${lib}} OUTPUT_VARIABLE symbols)
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
   add_custom_target(pseudalis-delay-import DEPENDS ${delay_libraries})
   add_dependencies(pseudalis-setup pseudalis-delay-import)
   target_link_directories(pseudalis-setup BEFORE PRIVATE ${delay_dir})
   set_property(TARGET pseudalis-setup APPEND PROPERTY LINK_DEPENDS ${delay_libraries})
endif()
