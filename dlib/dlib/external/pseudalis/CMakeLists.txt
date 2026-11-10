# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Pseudalis: the protection of the personal data of documents on Windows, built on
# dlib/privacy.
#
#   cmake -S dlib/dlib/external/pseudalis -B build-pseudalis
#   cmake --build build-pseudalis --config Release
#
# Every program lands in <build>/bin, the layout of an installation: the service, its two
# engines (pseudalis-engine.dll for every x64 processor, pseudalis-engine-avx2.dll for the
# processors with AVX2), the web application, the tray, the shell program and, on Windows,
# the installer pseudalis-setup.exe, which carries all of them.
#
# Outside Windows the service builds with its engine inside (PSEUDALIS_ENGINE_STATIC), its
# drives working as folders without letters, for the tests (tools/test_linux.sh).

cmake_minimum_required(VERSION 3.16)
project(pseudalis VERSION 1.0.9 LANGUAGES C CXX)

option(PSEUDALIS_TESTS "Build the test programs" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
   set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The programs carry their runtime: nothing to install beside them.
if (POLICY CMP0091)
   cmake_policy(SET CMP0091 NEW)
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
if (MSVC)
   foreach (option /utf-8 /bigobj /W3 /permissive-)
      add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:${option}>)
   endforeach()
   add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
elseif (MINGW)
   add_link_options(-static -static-libgcc -static-libstdc++ $<$<CONFIG:Release>:-s>)
   add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if (WIN32)
   add_compile_definitions(UNICODE _UNICODE _WIN32_WINNT=0x0A00)
endif()

get_filename_component(PSEUDALIS_DLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(PSEUDALIS_INCLUDE_DIR "${PSEUDALIS_DLIB_DIR}/.." ABSOLUTE)
get_filename_component(PSEUDALIS_ALBIREO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../albireo" ABSOLUTE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach (config ${CMAKE_CONFIGURATION_TYPES})
   string(TOUPPER ${config} upper)
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${upper} ${CMAKE_BINARY_DIR}/bin)
   set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${upper} ${CMAKE_BINARY_DIR}/bin)
endforeach()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
   set(PSEUDALIS_WARNINGS -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

# dlib: its HTTPS target (cpp-httplib and Mbed TLS) for the programs; the engines compile its
# sources themselves, each with the instructions of its processors.
set(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "" FORCE)
set(DLIB_USE_CUDA OFF CACHE BOOL "" FORCE)
set(DLIB_USE_NPU OFF CACHE BOOL "" FORCE)
set(DLIB_USE_BLAS OFF CACHE BOOL "" FORCE)
set(DLIB_USE_LAPACK OFF CACHE BOOL "" FORCE)
set(DLIB_JPEG_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_PNG_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_GIF_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_WEBP_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_JXL_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_LINK_WITH_SQLITE3 OFF CACHE BOOL "" FORCE)
set(DLIB_USE_FFMPEG OFF CACHE BOOL "" FORCE)
set(DLIB_USE_MKL_FFT OFF CACHE BOOL "" FORCE)
add_subdirectory(${PSEUDALIS_DLIB_DIR} dlib_build EXCLUDE_FROM_ALL)
find_package(Threads REQUIRED)

# The models of the detector: the header the example slm_document_anonymizer_ex embeds.
set(PSEUDALIS_MODELS_HEADER "${PSEUDALIS_INCLUDE_DIR}/examples/slm_document_anonymizer_ex.h" CACHE FILEPATH
   "Header of the models compiled into the engines (slm_document_anonymizer_ex embed)")
if (NOT EXISTS "${PSEUDALIS_MODELS_HEADER}")
   message(FATAL_ERROR "No models header at ${PSEUDALIS_MODELS_HEADER}: set PSEUDALIS_MODELS_HEADER")
endif()

# The build of the programs and of the engines, new at each configuration of the build tree: the
# service loads only an engine of its own build, never one that another installation or another
# build left beside it (common/engine_loader.cpp).
string(TIMESTAMP PSEUDALIS_BUILD_TIME "%Y%m%d%H%M%S" UTC)
string(RANDOM LENGTH 8 ALPHABET "0123456789abcdef" PSEUDALIS_BUILD_RANDOM)
set(PSEUDALIS_BUILD_STAMP "${PSEUDALIS_BUILD_TIME}-${PSEUDALIS_BUILD_RANDOM}")
configure_file(cmake/build_stamp.h.in ${CMAKE_BINARY_DIR}/generated/build_stamp.h @ONLY)

# ---------------------------------------------------------------------------------------------
# The engines.

set(PSEUDALIS_ENGINE_SOURCES engine/engine.cpp engine/archives.cpp engine/engine_api.cpp
   ${PSEUDALIS_DLIB_DIR}/all/source.cpp)
# The sources of dlib keep the warnings of their own build.
if (MSVC)
   set_source_files_properties(${PSEUDALIS_DLIB_DIR}/all/source.cpp PROPERTIES COMPILE_OPTIONS "/W0")
else()
   set_source_files_properties(${PSEUDALIS_DLIB_DIR}/all/source.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()
set(PSEUDALIS_ENGINE_DEFINITIONS PSEUDALIS_ENGINE_BUILD DLIB_NO_GUI_SUPPORT DLIB_DISABLE_ASSERTS
   "PSEUDALIS_MODELS_HEADER=\"${PSEUDALIS_MODELS_HEADER}\"")
if (MSVC)
   set(PSEUDALIS_ISA_avx2 /arch:AVX2)
else()
   set(PSEUDALIS_ISA_avx2 -mavx2 -mfma -mf16c)
endif()

function(pseudalis_engine target isa)
   add_library(${target} SHARED ${PSEUDALIS_ENGINE_SOURCES})
   target_include_directories(${target} PRIVATE ${PSEUDALIS_INCLUDE_DIR} ${CMAKE_BINARY_DIR}/generated)
   target_compile_definitions(${target} PRIVATE ${PSEUDALIS_ENGINE_DEFINITIONS})
   target_compile_options(${target} PRIVATE ${isa})
   target_link_libraries(${target} PRIVATE Threads::Threads)
   set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET hidden PREFIX "")
   if (WIN32)
      set(PSEUDALIS_RC_FILE "${target}.dll")
      set(PSEUDALIS_RC_DESCRIPTION "Pseudalis engine")
      configure_file(resources/engine.rc.in ${CMAKE_BINARY_DIR}/rc/${target}.rc @ONLY)
      target_sources(${target} PRIVATE ${CMAKE_BINARY_DIR}/rc/${target}.rc)
      target_link_libraries(${target} PRIVATE ws2_32 winmm imm32)
   else()
      set_target_properties(${target} PROPERTIES PREFIX "lib")
   endif()
endfunction()

pseudalis_engine(pseudalis-engine "")
pseudalis_engine(pseudalis-engine-avx2 "${PSEUDALIS_ISA_avx2}")

# ---------------------------------------------------------------------------------------------
# The programs.

add_library(pseudalis_sqlite STATIC ${PSEUDALIS_DLIB_DIR}/external/sqlite/sqlite3.c)
target_include_directories(pseudalis_sqlite PUBLIC ${PSEUDALIS_DLIB_DIR}/external/sqlite)
target_compile_definitions(pseudalis_sqlite PRIVATE SQLITE_THREADSAFE=1 SQLITE_DQS=0 SQLITE_DEFAULT_MEMSTATUS=0
   SQLITE_DEFAULT_WAL_SYNCHRONOUS=1 SQLITE_LIKE_DOESNT_MATCH_BLOBS SQLITE_TEMP_STORE=2 SQLITE_SECURE_DELETE
   SQLITE_ENABLE_API_ARMOR SQLITE_OMIT_LOAD_EXTENSION)
if (NOT WIN32)
   target_link_libraries(pseudalis_sqlite PUBLIC ${CMAKE_DL_LIBS} m)
endif()

set(zlib_dir ${PSEUDALIS_DLIB_DIR}/external/zlib)
add_library(pseudalis_zlib STATIC ${zlib_dir}/adler32.c ${zlib_dir}/compress.c ${zlib_dir}/crc32.c
   ${zlib_dir}/deflate.c ${zlib_dir}/infback.c ${zlib_dir}/inffast.c ${zlib_dir}/inflate.c ${zlib_dir}/inftrees.c
   ${zlib_dir}/trees.c ${zlib_dir}/uncompr.c ${zlib_dir}/zutil.c)
target_include_directories(pseudalis_zlib PUBLIC ${zlib_dir})
if (MSVC)
   target_compile_options(pseudalis_zlib PRIVATE /W0)
else()
   target_compile_options(pseudalis_zlib PRIVATE -w)
endif()

# Code shared by the service, the tray, the shell program and the installer.
add_library(pseudalis_common STATIC common/log.cpp common/platform.cpp)
target_include_directories(pseudalis_common PUBLIC ${PSEUDALIS_INCLUDE_DIR})
target_compile_options(pseudalis_common PRIVATE ${PSEUDALIS_WARNINGS})
target_link_libraries(pseudalis_common PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if (WIN32)
   target_link_libraries(pseudalis_common PUBLIC shell32 ole32 bcrypt advapi32)
endif()

# The public key of the update feed, which pseudalis-sign keygen writes once, beside the private
# key that the publisher keeps outside the repository (README.md, Publishing a version).
set(PSEUDALIS_FEED_PUBLIC_KEY_FILE "${CMAKE_CURRENT_SOURCE_DIR}/resources/feed-public-key.pem" CACHE FILEPATH
   "Public key (PEM) of the update feed")
set(PSEUDALIS_FEED_PUBLIC_KEY_TEXT "")
if (EXISTS "${PSEUDALIS_FEED_PUBLIC_KEY_FILE}")
   set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PSEUDALIS_FEED_PUBLIC_KEY_FILE}")
   file(STRINGS "${PSEUDALIS_FEED_PUBLIC_KEY_FILE}" feed_key_lines)
   foreach (line ${feed_key_lines})
      string(APPEND PSEUDALIS_FEED_PUBLIC_KEY_TEXT "${line}\\n")
   endforeach()
   message(STATUS "Update feed: public key read from ${PSEUDALIS_FEED_PUBLIC_KEY_FILE}")
else()
   message(STATUS "Update feed: no public key at ${PSEUDALIS_FEED_PUBLIC_KEY_FILE}, the programs install no update")
endif()
configure_file(cmake/feed_public_key.h.in ${CMAKE_BINARY_DIR}/generated/feed_public_key.h @ONLY)

# The service.
set(PSEUDALIS_SERVICE_SOURCES common/engine_loader.cpp host/main.cpp host/service.cpp host/server.cpp
   host/settings.cpp host/store.cpp host/vault.cpp host/keys.cpp host/albireo_client.cpp host/updater.cpp
   host/http_fetch.cpp host/crypto.cpp host/certificates.cpp)
# The drive: a folder of the data folder, watched by the service, with its letter on Windows.
list(APPEND PSEUDALIS_SERVICE_SOURCES host/drive_folders.cpp)
if (WIN32)
   list(APPEND PSEUDALIS_SERVICE_SOURCES resources/service.rc)
else()
   list(APPEND PSEUDALIS_SERVICE_SOURCES ${PSEUDALIS_ENGINE_SOURCES})
endif()
add_executable(pseudalis-service ${PSEUDALIS_SERVICE_SOURCES})
target_include_directories(pseudalis-service PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(pseudalis-service PRIVATE pseudalis_common pseudalis_sqlite dlib::httplib)
if (WIN32)
   target_link_libraries(pseudalis-service PRIVATE winhttp crypt32)
   if (MSVC)
      # The libraries the service imports are loaded from System32 only (LOAD_LIBRARY_SEARCH_SYSTEM32).
      target_link_options(pseudalis-service PRIVATE /DEPENDENTLOADFLAG:0x800)
   endif()
else()
   # The engine inside the program, outside Windows.
   target_compile_definitions(pseudalis-service PRIVATE PSEUDALIS_ENGINE_STATIC ${PSEUDALIS_ENGINE_DEFINITIONS})
endif()
target_compile_options(pseudalis-service PRIVATE ${PSEUDALIS_WARNINGS})

# The signatures of the update feed, for its publisher.
add_executable(pseudalis-sign tools/pseudalis_sign.cpp host/crypto.cpp)
target_include_directories(pseudalis-sign PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(pseudalis-sign PRIVATE pseudalis_common dlib::httplib)
target_compile_options(pseudalis-sign PRIVATE ${PSEUDALIS_WARNINGS})

# The web application and the documentation of the interface, next to the service; Swagger UI
# comes from the copy of Albireo in this repository.
set(swagger_dir ${PSEUDALIS_ALBIREO_DIR}/web/docs)
add_custom_target(pseudalis-web ALL
   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/web ${CMAKE_BINARY_DIR}/bin/web
   COMMAND ${CMAKE_COMMAND} -E copy ${swagger_dir}/swagger-ui-bundle.js ${swagger_dir}/swagger-ui.css
      ${swagger_dir}/LICENSE ${swagger_dir}/NOTICE ${CMAKE_BINARY_DIR}/bin/web/docs
   COMMENT "Copying the web application")
add_dependencies(pseudalis-service pseudalis-web)

if (WIN32)
   add_subdirectory(windows)
endif()

# ---------------------------------------------------------------------------------------------
# Tests.

if (PSEUDALIS_TESTS)
   add_executable(test_engine tests/test_engine.cpp ${PSEUDALIS_ENGINE_SOURCES})
   target_include_directories(test_engine PRIVATE ${PSEUDALIS_INCLUDE_DIR} ${CMAKE_BINARY_DIR}/generated)
   target_compile_definitions(test_engine PRIVATE ${PSEUDALIS_ENGINE_DEFINITIONS})
   target_link_libraries(test_engine PRIVATE Threads::Threads)
   if (WIN32)
      target_link_libraries(test_engine PRIVATE ws2_32 winmm imm32)
   endif()
   add_executable(test_host tests/test_host.cpp host/vault.cpp host/keys.cpp host/store.cpp host/crypto.cpp
      host/settings.cpp)
   target_include_directories(test_host PRIVATE ${CMAKE_BINARY_DIR}/generated)
   target_link_libraries(test_host PRIVATE pseudalis_common pseudalis_sqlite dlib::httplib)
   if (WIN32)
      target_link_libraries(test_host PRIVATE crypt32)
   endif()
   target_compile_options(test_host PRIVATE ${PSEUDALIS_WARNINGS})
endif()
