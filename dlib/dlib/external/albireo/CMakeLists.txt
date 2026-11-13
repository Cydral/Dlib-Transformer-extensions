# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Albireo: the local inference service built on dlib.
#
# One build tree per kind of engine, since dlib is configured for one primary target:
#
#   cmake -S dlib/external/albireo -B build-albireo                       (service, tray, installer,
#                                                                          processor engines, tests)
#   cmake -S dlib/external/albireo -B build-albireo-cuda -DALBIREO_PROGRAMS=OFF -DALBIREO_ENGINE_CUDA=ON
#   cmake -S dlib/external/albireo -B build-albireo-ov -DALBIREO_PROGRAMS=OFF -DALBIREO_ENGINE_OPENVINO=ON
#
# Every program lands in <build>/bin and every engine in <build>/bin/engines, the layout
# of an installation; the web application is copied to <build>/bin/web.  The installer
# takes the engines of the other build trees through ALBIREO_EXTRA_ENGINE_DIRS, a list of
# folders whose files go into the engines folder of the installation.

cmake_minimum_required(VERSION 3.16)
project(albireo VERSION 1.0.20 LANGUAGES C CXX)

# The version is written in four places that must agree: project() above, the numbers of
# common/version.h, which the version resources of the programs and of the installer carry and
# which the installer compares before it replaces an installation, the text of common/version.h,
# which the service, the engines and the update feed give, and the description of the interface.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/common/version.h" version_lines REGEX "^#define ALBIREO_VERSION")
set(version_numbers)
foreach (part MAJOR MINOR PATCH)
   string(REGEX MATCH "#define ALBIREO_VERSION_${part} +([0-9]+)" found "${version_lines}")
   list(APPEND version_numbers "${CMAKE_MATCH_1}")
endforeach()
string(REPLACE ";" "." version_numbers "${version_numbers}")
string(REGEX MATCH "#define ALBIREO_VERSION +\"([0-9.]+)\"" found "${version_lines}")
set(version_text "${CMAKE_MATCH_1}")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/web/openapi.json" openapi_text)
string(REGEX MATCH "\"version\" *: *\"([0-9.]+)\"" found "${openapi_text}")
set(openapi_version "${CMAKE_MATCH_1}")
if (NOT version_numbers STREQUAL PROJECT_VERSION OR NOT version_text STREQUAL PROJECT_VERSION OR
    NOT openapi_version STREQUAL PROJECT_VERSION)
   message(FATAL_ERROR "the version differs between project() (${PROJECT_VERSION}), the numbers of "
      "common/version.h (${version_numbers}), its text (${version_text}) and web/openapi.json "
      "(${openapi_version}): the four must agree")
endif()

option(ALBIREO_PROGRAMS "Build the service, the tray, the installer and the processor engines" ON)
option(ALBIREO_ENGINE_CUDA "Build the engine for NVIDIA graphics processors (dlib built with CUDA)" OFF)
option(ALBIREO_ENGINE_OPENVINO "Build the engine for Intel NPU and graphics processors (dlib built with OpenVINO)" OFF)
option(ALBIREO_ENGINE_DLIB "Build an engine whose products go through the host code of dlib (validation)" OFF)
option(ALBIREO_TESTS "Build the test programs" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
   set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The programs carry their runtime: nothing to install beside them.  The OpenVINO engine
# is the exception: it shares C++ objects with the runtime of OpenVINO, built with the
# shared runtime, and takes the same; its folder then receives the runtime libraries.
if (POLICY CMP0091)
   cmake_policy(SET CMP0091 NEW)
endif()
if (ALBIREO_ENGINE_OPENVINO)
   set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
else()
   set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
if (MSVC)
   # Options of the C and C++ compiler, which nvcc would take for source files.
   foreach (option /utf-8 /bigobj /W3 /permissive-)
      add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:${option}>)
   endforeach()
   add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
elseif (MINGW)
   add_link_options(-static -static-libgcc -static-libstdc++)
   add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if (WIN32)
   add_compile_definitions(UNICODE _UNICODE _WIN32_WINNT=0x0A00)
endif()

get_filename_component(ALBIREO_DLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(ALBIREO_INCLUDE_DIR "${ALBIREO_DLIB_DIR}/.." ABSOLUTE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(ALBIREO_ENGINE_DIR ${CMAKE_BINARY_DIR}/bin/engines)
foreach (cfg ${CMAKE_CONFIGURATION_TYPES})
   string(TOUPPER ${cfg} CFG)
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

# Puts the executables of a target in a folder of the build tree, for every configuration.
function(albireo_output_dir target dir)
   set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${dir})
   foreach (cfg ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER ${cfg} CFG)
      set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${CFG} ${dir})
   endforeach()
endfunction()

if (ALBIREO_ENGINE_CUDA AND ALBIREO_ENGINE_OPENVINO)
   message(FATAL_ERROR "The CUDA and OpenVINO engines need two build trees: dlib is configured for one of them.")
endif()

# dlib: the tensor engines link the library, configured for their device; the other
# programs only use its headers and its HTTPS target.  The processor side of the engines
# is their own, so dlib needs no BLAS: the engines then depend on no library beside the
# runtime of their device.
set(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "" FORCE)
set(DLIB_USE_CUDA ${ALBIREO_ENGINE_CUDA} CACHE BOOL "" FORCE)
set(DLIB_USE_NPU ${ALBIREO_ENGINE_OPENVINO} CACHE BOOL "" FORCE)
if (ALBIREO_ENGINE_CUDA)
   set(DLIB_BACKENDS "CPU/CUDA" CACHE STRING "" FORCE)
elseif (ALBIREO_ENGINE_OPENVINO)
   set(DLIB_BACKENDS "CPU/NPU" CACHE STRING "" FORCE)
else()
   set(DLIB_BACKENDS "CPU" CACHE STRING "" FORCE)
endif()
set(DLIB_USE_BLAS OFF CACHE BOOL "" FORCE)
set(DLIB_USE_LAPACK OFF CACHE BOOL "" FORCE)
set(DLIB_JPEG_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_PNG_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_LINK_WITH_SQLITE3 OFF CACHE BOOL "" FORCE)
set(DLIB_USE_FFMPEG OFF CACHE BOOL "" FORCE)
set(DLIB_JXL_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_WEBP_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_GIF_SUPPORT OFF CACHE BOOL "" FORCE)
set(DLIB_USE_MKL_FFT OFF CACHE BOOL "" FORCE)
if (ALBIREO_ENGINE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
   # A distributed engine carries the machine code of the cards of desktops and laptops, from
   # the GeForce GTX 900 series on (Maxwell 5.0, Pascal 6.1, Volta 7.0, Turing 7.5, Ampere 8.6,
   # Ada 8.9, Blackwell 12.0), and the intermediate code of the newest one, which the driver
   # compiles for the cards that follow.  CUDA 13 compiles for Turing and later only.
   find_package(CUDAToolkit)
   if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
      set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 89-real 120-real 120-virtual)
   elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
      set(CMAKE_CUDA_ARCHITECTURES 50-real 61-real 70-real 75-real 86-real 89-real 120-real 120-virtual)
   else()
      set(CMAKE_CUDA_ARCHITECTURES 50-real 61-real 70-real 75-real 86-real 89-real 89-virtual)
   endif()
endif()
add_subdirectory(${ALBIREO_DLIB_DIR} dlib_build EXCLUDE_FROM_ALL)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
   set(ALBIREO_WARNINGS -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

# zlib from dlib/external, for the programs and the image decoders of the engines.
set(zlib_dir ${ALBIREO_DLIB_DIR}/external/zlib)
add_library(albireo_zlib STATIC ${zlib_dir}/adler32.c ${zlib_dir}/compress.c ${zlib_dir}/crc32.c
   ${zlib_dir}/deflate.c ${zlib_dir}/infback.c ${zlib_dir}/inffast.c ${zlib_dir}/inflate.c ${zlib_dir}/inftrees.c
   ${zlib_dir}/trees.c ${zlib_dir}/uncompr.c ${zlib_dir}/zutil.c)
target_include_directories(albireo_zlib PUBLIC ${zlib_dir})

# The PNG and JPEG decoders of the engines, which read the images of the requests: the copies
# of libpng and libjpeg that dlib bundles, compiled into every engine, whatever the system holds.
set(png_dir ${ALBIREO_DLIB_DIR}/external/libpng)
set(jpeg_dir ${ALBIREO_DLIB_DIR}/external/libjpeg)
add_library(albireo_image STATIC
   ${png_dir}/png.c ${png_dir}/pngerror.c ${png_dir}/pngget.c ${png_dir}/pngmem.c ${png_dir}/pngpread.c
   ${png_dir}/pngread.c ${png_dir}/pngrio.c ${png_dir}/pngrtran.c ${png_dir}/pngrutil.c ${png_dir}/pngset.c
   ${png_dir}/pngtrans.c ${png_dir}/pngwio.c ${png_dir}/pngwrite.c ${png_dir}/pngwtran.c ${png_dir}/pngwutil.c
   ${jpeg_dir}/jaricom.c ${jpeg_dir}/jcomapi.c ${jpeg_dir}/jdapimin.c ${jpeg_dir}/jdapistd.c ${jpeg_dir}/jdarith.c
   ${jpeg_dir}/jdatasrc.c ${jpeg_dir}/jdcoefct.c ${jpeg_dir}/jdcolor.c ${jpeg_dir}/jddctmgr.c ${jpeg_dir}/jdhuff.c
   ${jpeg_dir}/jdinput.c ${jpeg_dir}/jdmainct.c ${jpeg_dir}/jdmarker.c ${jpeg_dir}/jdmaster.c ${jpeg_dir}/jdmerge.c
   ${jpeg_dir}/jdpostct.c ${jpeg_dir}/jdsample.c ${jpeg_dir}/jerror.c ${jpeg_dir}/jidctflt.c ${jpeg_dir}/jidctfst.c
   ${jpeg_dir}/jidctint.c ${jpeg_dir}/jmemmgr.c ${jpeg_dir}/jmemnobs.c ${jpeg_dir}/jquant1.c ${jpeg_dir}/jquant2.c
   ${jpeg_dir}/jutils.c)
target_include_directories(albireo_image PRIVATE ${png_dir} ${jpeg_dir})
target_link_libraries(albireo_image PUBLIC albireo_zlib)
if (MSVC)
   # The simplified interface of libpng takes the magnitude of a negative row stride by
   # negating its unsigned value, on purpose (C4146).
   target_compile_options(albireo_image PRIVATE /wd4146)
endif()
# Compiled into the shared engines, whose own symbols alone are visible.
set_target_properties(albireo_zlib albireo_image PROPERTIES POSITION_INDEPENDENT_CODE ON
   C_VISIBILITY_PRESET hidden)

# Inference engines.
set(ALBIREO_ENGINE_SOURCES
   engine/chat_format.cpp
   engine/cpu_runtime.cpp
   engine/engine.cpp
   engine/engine_api.cpp
   engine/image_input.cpp
   engine/model_config.cpp
   engine/model_file.cpp
   engine/sampler.cpp
   engine/vision_encoder.cpp
   common/system_info.cpp)

# The instructions of the processor engines.  An engine of the AVX2 family is built for AVX2,
# FMA and F16C, and its quantized kernels (engine/qkernels.cpp) for the instructions of its
# variant: the description of the library then runs on any processor of the family, and says
# whether the kernels run on it.  MSVC defines no macro for VNNI, which the build names.
include(CheckCXXCompilerFlag)
if (MSVC)
   set(ALBIREO_ISA_avx2 /arch:AVX2)
   set(ALBIREO_ISA_avxvnni /arch:AVX2)
   set(ALBIREO_ISA_avxvnni_DEFS ALBIREO_AVX_VNNI)
   # The intrinsics of AVX-VNNI came with Visual Studio 2022.
   if (MSVC_VERSION GREATER_EQUAL 1930)
      set(ALBIREO_HAS_avxvnni ON)
   endif()
else()
   set(ALBIREO_ISA_avx2 -mavx2 -mfma -mf16c)
   set(ALBIREO_ISA_avxvnni ${ALBIREO_ISA_avx2} -mavxvnni)
   check_cxx_compiler_flag(-mavxvnni ALBIREO_HAS_avxvnni)
endif()
set(ALBIREO_HAS_portable ON)
set(ALBIREO_HAS_avx2 ON)

# The quantized kernels of a family of processors, shared by the engines and the tests.
function(albireo_kernels family)
   set(target albireo-kernels-${family})
   if (TARGET ${target})
      return()
   endif()
   add_library(${target} OBJECT engine/qkernels.cpp)
   target_include_directories(${target} PRIVATE ${ALBIREO_INCLUDE_DIR})
   target_compile_options(${target} PRIVATE ${ALBIREO_WARNINGS} ${ALBIREO_ISA_${family}})
   target_compile_definitions(${target} PRIVATE ${ALBIREO_ISA_${family}_DEFS})
   set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON
      C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endfunction()

# An engine library; kernels is the family of its quantized kernels, or "inline" for kernels
# compiled with the library itself (the device engines, built for AVX2 with dlib).
function(albireo_engine variant kernels)
   set(target albireo-engine-${variant})
   if (kernels STREQUAL "inline")
      add_library(${target} SHARED ${ALBIREO_ENGINE_SOURCES} engine/qkernels.cpp ${ARGN})
   else()
      albireo_kernels(${kernels})
      add_library(${target} SHARED ${ALBIREO_ENGINE_SOURCES} $<TARGET_OBJECTS:albireo-kernels-${kernels}> ${ARGN})
      if (NOT kernels STREQUAL "portable")
         target_compile_options(${target} PRIVATE ${ALBIREO_ISA_avx2})
      endif()
   endif()
   target_include_directories(${target} PRIVATE ${ALBIREO_INCLUDE_DIR})
   target_compile_definitions(${target} PRIVATE ALBIREO_ENGINE_BUILD ALBIREO_ENGINE_VARIANT="${variant}")
   target_compile_options(${target} PRIVATE ${ALBIREO_WARNINGS})
   target_link_libraries(${target} PRIVATE albireo_image)
   set_target_properties(${target} PROPERTIES
      C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
      LIBRARY_OUTPUT_DIRECTORY ${ALBIREO_ENGINE_DIR}
      RUNTIME_OUTPUT_DIRECTORY ${ALBIREO_ENGINE_DIR})
   foreach (cfg ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER ${cfg} CFG)
      set_target_properties(${target} PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY_${CFG} ${ALBIREO_ENGINE_DIR} RUNTIME_OUTPUT_DIRECTORY_${CFG} ${ALBIREO_ENGINE_DIR})
   endforeach()
   if (WIN32)
      set_target_properties(${target} PROPERTIES PREFIX "")
      # Version information of the library, as the installer and the tray read it.
      set(ALBIREO_RC_DESCRIPTION "Albireo inference engine (${variant})")
      set(ALBIREO_RC_FILE "${target}.dll")
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/resources/engine.rc.in ${CMAKE_BINARY_DIR}/rc/${target}.rc @ONLY)
      target_sources(${target} PRIVATE ${CMAKE_BINARY_DIR}/rc/${target}.rc)
      target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/resources)
   else()
      find_package(Threads REQUIRED)
      target_link_libraries(${target} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
   endif()
   if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32 AND NOT APPLE)
      # The engine exports the functions of engine_api.h only, so that none of its symbols
      # binds to the program.  The processor engines also carry their C++ runtime, whatever
      # the system provides; the device engines share the one of the libraries of their
      # device, whose exceptions must unwind through a single runtime.
      file(WRITE ${CMAKE_BINARY_DIR}/albireo-engine.map "{ global: albireo_engine_*; local: *; };\n")
      target_link_options(${target} PRIVATE -Wl,--exclude-libs,ALL
         -Wl,--version-script=${CMAKE_BINARY_DIR}/albireo-engine.map -Wl,-Bsymbolic)
      if (NOT ALBIREO_DEVICE_ENGINE)
         target_link_options(${target} PRIVATE -static-libstdc++ -static-libgcc)
      endif()
   endif()
endfunction()

if (ALBIREO_PROGRAMS)
   # Portable engine: the instructions of every x86-64 processor.
   albireo_engine(cpu portable)
   # Engine for the processors with AVX2, FMA and F16C (2013 and later).
   albireo_engine(cpu-avx2 avx2)
   # AVX-VNNI: Intel Core 12th generation and later, AMD Zen 5.  The processors with AVX-512
   # and without AVX-VNNI run the engine of AVX2.
   if (ALBIREO_HAS_avxvnni)
      albireo_engine(cpu-avxvnni avxvnni)
   else()
      message(STATUS "The compiler lacks AVX-VNNI: no cpu-avxvnni engine.")
   endif()
endif()

# The libraries a device engine loads at run time, copied next to it so that the engines
# folder of an installation is complete: the runtime of CUDA and cuDNN, or that of OpenVINO
# with the plugins of the NPU and of the graphics processor.
if (WIN32)
   option(ALBIREO_COPY_RUNTIME "Copy next to the device engines the libraries they load" ON)
else()
   option(ALBIREO_COPY_RUNTIME "Copy next to the device engines the libraries they load" OFF)
endif()
# The runtime of the compiler, which the libraries of the devices built with it load.  Its
# folder is found again at each configuration, since an update of Visual Studio replaces it,
# and the configuration stops when a library of it is missing.
set(ALBIREO_COMPILER_RUNTIME)
if (MSVC AND (ALBIREO_ENGINE_CUDA OR ALBIREO_ENGINE_OPENVINO))
   set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
   set(CMAKE_INSTALL_UCRT_LIBRARIES FALSE)
   unset(MSVC_REDIST_DIR CACHE)
   include(InstallRequiredSystemLibraries)
   set(ALBIREO_COMPILER_RUNTIME ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
   foreach (required msvcp140.dll vcruntime140.dll vcruntime140_1.dll)
      set(found FALSE)
      foreach (library ${ALBIREO_COMPILER_RUNTIME})
         get_filename_component(name "${library}" NAME)
         string(TOLOWER "${name}" name)
         if (name STREQUAL required AND EXISTS "${library}")
            set(found TRUE)
         endif()
      endforeach()
      if (NOT found)
         message(FATAL_ERROR "${required}, of the runtime of the compiler that the device engines load, was not "
            "found in the redistributable files of Visual Studio (MSVC_REDIST_DIR: ${MSVC_REDIST_DIR}): repair the "
            "C++ workload of Visual Studio, or of its Build Tools")
      endif()
   endforeach()
endif()
# The licenses of the runtime libraries, copied into the licenses folder beside the engines,
# which the installer moves to the licenses folder of the installation: the first existing
# file of a list, under a name of the installation.
set(ALBIREO_RUNTIME_LICENSES "" CACHE STRING "License files of the device runtimes, added to the licenses folder")
function(albireo_copy_license target name)
   foreach (f ${ARGN})
      if (EXISTS "${f}")
         add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/licenses"
            COMMAND ${CMAKE_COMMAND} -E copy "${f}" "$<TARGET_FILE_DIR:${target}>/licenses/${name}"
            VERBATIM)
         return()
      endif()
   endforeach()
   message(STATUS "No license file found for ${name}; set ALBIREO_RUNTIME_LICENSES to add it to the installer")
endfunction()

function(albireo_copy_runtime target dirs extra exclude)
   if (NOT ALBIREO_COPY_RUNTIME)
      return()
   endif()
   string(REPLACE ";" "|" dirs_arg "${dirs}")
   string(REPLACE ";" "|" extra_arg "${extra};${ALBIREO_COMPILER_RUNTIME}")
   string(REPLACE ";" "|" exclude_arg "${exclude}")
   add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} "-DENGINE=$<TARGET_FILE:${target}>" "-DDEST=$<TARGET_FILE_DIR:${target}>"
         "-DDIRS=${dirs_arg}" "-DEXTRA=${extra_arg}" "-DEXCLUDE=${exclude_arg}"
         -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/runtime_dependencies.cmake
      VERBATIM)
endfunction()

# The runtime libraries of the device of this tree, which its engine and its test programs
# load: the folders they come from, more files to copy, the names to leave out, and the
# libraries loaded on first use only (MSVC).
set(ALBIREO_RUNTIME_DIRS)
set(ALBIREO_RUNTIME_EXTRA)
set(ALBIREO_RUNTIME_EXCLUDE)
set(ALBIREO_DELAYED_LIBRARIES)

function(albireo_device_runtime target)
   foreach (name ${ALBIREO_DELAYED_LIBRARIES})
      target_link_options(${target} PRIVATE "/DELAYLOAD:${name}")
   endforeach()
   if (ALBIREO_DELAYED_LIBRARIES)
      target_link_libraries(${target} PRIVATE delayimp)
   endif()
   if (ALBIREO_RUNTIME_DIRS)
      albireo_copy_runtime(${target} "${ALBIREO_RUNTIME_DIRS}" "${ALBIREO_RUNTIME_EXTRA}" "${ALBIREO_RUNTIME_EXCLUDE}")
   endif()
endfunction()

# Engines whose products go to a device through the tensors of dlib.  Their processor side
# is the one of cpu-avx2, which every machine with such a device has, or the quantized kernels
# of the family given after the variant (avxvnni), the rest of the library staying at AVX2.
function(albireo_tensor_engine variant)
   set(ALBIREO_DEVICE_ENGINE ON)
   set(family "${ARGN}")
   if (family)
      albireo_engine(${variant} ${family} engine/tensor_backend.cpp)
   else()
      albireo_engine(${variant} inline engine/tensor_backend.cpp)
   endif()
   set(target albireo-engine-${variant})
   foreach (f ${ALBIREO_RUNTIME_LICENSES})
      get_filename_component(name "${f}" NAME)
      albireo_copy_license(${target} "${name}" "${f}")
   endforeach()
   target_compile_definitions(${target} PRIVATE ALBIREO_TENSOR_BACKEND)
   target_link_libraries(${target} PRIVATE dlib::dlib)
   target_compile_options(${target} PRIVATE ${ALBIREO_ISA_avx2})
endfunction()

if (ALBIREO_ENGINE_CUDA)
   if (NOT DLIB_USE_CUDA)
      message(FATAL_ERROR "dlib could not be configured with CUDA (see the messages above): the CUDA engine needs "
         "the CUDA toolkit and cuDNN.")
   endif()
   find_package(CUDAToolkit REQUIRED)
   albireo_tensor_engine(cuda)
   target_link_libraries(albireo-engine-cuda PRIVATE CUDA::cudart)
   # The folders of cuDNN, then those of the toolkit, which may carry another version of cuDNN
   # whose libraries the search must not take.  cuDNN loads its parts by name, those of the
   # tensor descriptors and of the elementary operations being the ones the engine uses, copied
   # from the folders of cuDNN alone.
   set(cudnn_dirs ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64)
   if (CUDNN_LIBRARY_PATH)
      get_filename_component(cudnn_lib_dir "${CUDNN_LIBRARY_PATH}" DIRECTORY)
      string(REGEX REPLACE "/lib(/|$)" "/bin\\1" cudnn_bin_dir "${cudnn_lib_dir}")
      get_filename_component(cudnn_bin_parent "${cudnn_bin_dir}" DIRECTORY)
      set(cudnn_dirs "${cudnn_bin_dir}" "${cudnn_bin_parent}")
   endif()
   set(ALBIREO_RUNTIME_DIRS ${cudnn_dirs} ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64)
   list(REMOVE_DUPLICATES ALBIREO_RUNTIME_DIRS)
   foreach (dir ${cudnn_dirs})
      list(APPEND ALBIREO_RUNTIME_EXTRA "${dir}/cudnn_graph64_*.dll" "${dir}/cudnn_ops64_*.dll")
   endforeach()
   # dlib links the random numbers and the solvers of CUDA, which the engine never calls:
   # loaded on first use only, they need not be installed.
   if (MSVC)
      file(GLOB unused_libraries "${CUDAToolkit_BIN_DIR}/curand64_*.dll" "${CUDAToolkit_BIN_DIR}/cusolver64_*.dll"
         "${CUDAToolkit_BIN_DIR}/x64/curand64_*.dll" "${CUDAToolkit_BIN_DIR}/x64/cusolver64_*.dll")
      foreach (library ${unused_libraries})
         get_filename_component(name "${library}" NAME)
         list(APPEND ALBIREO_DELAYED_LIBRARIES "${name}")
         list(APPEND ALBIREO_RUNTIME_EXCLUDE "^${name}$")
      endforeach()
   endif()
   albireo_device_runtime(albireo-engine-cuda)

   # The oldest cards the engine runs on, as ten times the major version of the compute
   # capability plus its minor version: the lowest architecture of its kernels, raised to
   # Turing when the cuDNN copied beside it is 9.12 or later, which left the older cards.
   set(ALBIREO_CUDA_MIN_CC 0)
   foreach (arch ${CMAKE_CUDA_ARCHITECTURES})
      set(cc 0)
      if (arch MATCHES "^([0-9]+)")
         set(cc ${CMAKE_MATCH_1})
      elseif (arch STREQUAL "all" OR arch STREQUAL "all-major")
         if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
            set(cc 75)
         else()
            set(cc 50)
         endif()
      endif()
      if (cc GREATER 0 AND (ALBIREO_CUDA_MIN_CC EQUAL 0 OR cc LESS ALBIREO_CUDA_MIN_CC))
         set(ALBIREO_CUDA_MIN_CC ${cc})
      endif()
   endforeach()
   set(cudnn_version "unknown")
   if (CUDNN_INCLUDE_PATH AND EXISTS "${CUDNN_INCLUDE_PATH}/cudnn_version.h")
      file(STRINGS "${CUDNN_INCLUDE_PATH}/cudnn_version.h" cudnn_defines REGEX "#define CUDNN_(MAJOR|MINOR) +[0-9]+")
      string(REGEX REPLACE ".*CUDNN_MAJOR +([0-9]+).*" "\\1" cudnn_major "${cudnn_defines}")
      string(REGEX REPLACE ".*CUDNN_MINOR +([0-9]+).*" "\\1" cudnn_minor "${cudnn_defines}")
      set(cudnn_version "${cudnn_major}.${cudnn_minor}")
      if (cudnn_version VERSION_GREATER_EQUAL 9.12 AND ALBIREO_CUDA_MIN_CC LESS 75)
         set(ALBIREO_CUDA_MIN_CC 75)
      endif()
   endif()
   target_compile_definitions(albireo-engine-cuda PRIVATE ALBIREO_CUDA_MIN_CC=${ALBIREO_CUDA_MIN_CC})
   message(STATUS "CUDA engine: CUDA ${CUDAToolkit_VERSION}, cuDNN ${cudnn_version}, architectures "
      "${CMAKE_CUDA_ARCHITECTURES}")
   if (ALBIREO_CUDA_MIN_CC GREATER 0)
      math(EXPR cc_major "${ALBIREO_CUDA_MIN_CC} / 10")
      math(EXPR cc_minor "${ALBIREO_CUDA_MIN_CC} % 10")
      message(STATUS "CUDA engine: cards of compute capability ${cc_major}.${cc_minor} and later")
   endif()
   # The oldest cards of the build, which tools/build_windows.ps1 reports at its end.
   set(ALBIREO_CUDA_MIN_CC_BUILT ${ALBIREO_CUDA_MIN_CC} CACHE INTERNAL "Oldest compute capability of the CUDA engine")
   if (ALBIREO_CUDA_MIN_CC GREATER 61)
      message(WARNING "The CUDA engine of this build leaves out the GeForce GTX 900 and 10 series (Maxwell, "
         "Pascal), which need the architectures 50 and 61 with CUDA 12.8 or 12.9 and cuDNN 9.11 or earlier "
         "(architectures ${CMAKE_CUDA_ARCHITECTURES}, cuDNN ${cudnn_version}).")
   endif()
   # nvcc warns that it will stop compiling for the cards before Turing.
   target_compile_options(dlib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
   if (ALBIREO_COPY_RUNTIME)
      albireo_copy_license(albireo-engine-cuda cuda-eula.txt "${CUDAToolkit_BIN_DIR}/../EULA.txt")
      albireo_copy_license(albireo-engine-cuda cudnn-license.txt "${cudnn_bin_parent}/../LICENSE"
         "${cudnn_bin_parent}/../../LICENSE" "${cudnn_bin_dir}/../LICENSE")
   endif()
endif()
if (ALBIREO_ENGINE_OPENVINO)
   if (NOT DLIB_USE_NPU)
      message(FATAL_ERROR "dlib could not be configured with OpenVINO (see the messages above): the OpenVINO engine "
         "needs the OpenVINO runtime, from its archive or from pip install openvino.")
   endif()
   # One engine per family of processor kernels, so that the NPU and the graphics processor
   # join the richest kernels of the processor: the service runs the one this processor takes.
   albireo_tensor_engine(openvino)
   set(openvino_engines albireo-engine-openvino)
   if (ALBIREO_HAS_avxvnni)
      albireo_tensor_engine(openvino-avxvnni avxvnni)
      list(APPEND openvino_engines albireo-engine-openvino-avxvnni)
   endif()
   # The folder of the runtime of OpenVINO, which loads the plugins of the devices by name.
   find_package(OpenVINO QUIET COMPONENTS Runtime PATHS "${OpenVINO_DIR}" NO_DEFAULT_PATH)
   set(openvino_dir)
   if (TARGET openvino::runtime)
      foreach (cfg RELEASE RELWITHDEBINFO NOCONFIG)
         get_target_property(location openvino::runtime IMPORTED_LOCATION_${cfg})
         if (location AND NOT openvino_dir)
            get_filename_component(openvino_dir "${location}" DIRECTORY)
         endif()
      endforeach()
   endif()
   if (NOT openvino_dir AND ALBIREO_COPY_RUNTIME)
      message(FATAL_ERROR "The folder of the OpenVINO runtime was not found (find_package(OpenVINO) with "
         "OpenVINO_DIR \"${OpenVINO_DIR}\"): its libraries cannot join the engines. Set OpenVINO_DIR to the "
         "cmake folder of the runtime, or -DALBIREO_COPY_RUNTIME=OFF for engines that use the runtime in place.")
   endif()
   if (openvino_dir)
      set(ALBIREO_RUNTIME_DIRS "${openvino_dir}")
      # TBB lies beside the libraries of OpenVINO in its Python package, and in the 3rdparty
      # folder of its archive (runtime/3rdparty/tbb/bin next to runtime/bin/intel64/Release).
      get_filename_component(tbb_dir "${openvino_dir}/../../../3rdparty/tbb/bin" ABSOLUTE)
      set(tbbbind_libraries)
      if (IS_DIRECTORY "${tbb_dir}")
         list(APPEND ALBIREO_RUNTIME_DIRS "${tbb_dir}")
         # The library with which TBB reads the cores of a hybrid processor, without its debug build.
         file(GLOB tbbbind_libraries "${tbb_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}tbbbind*${CMAKE_SHARED_LIBRARY_SUFFIX}")
         list(FILTER tbbbind_libraries EXCLUDE REGEX "_debug${CMAKE_SHARED_LIBRARY_SUFFIX}$")
      endif()
      # The plugins of the NPU and of the graphics processor, and that of the processor, which runs the
      # graphs of the models (engine/ov_graph.h) on any machine.
      foreach (device npu gpu cpu)
         set(plugin "${CMAKE_SHARED_LIBRARY_PREFIX}openvino_intel_${device}_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}")
         list(APPEND ALBIREO_RUNTIME_EXTRA "${openvino_dir}/${plugin}")
      endforeach()
      # What the plugins load by name besides: the compiler of the NPU plugin, its loader and
      # the runtime of its graphs (OpenVINO 2025 and later), the library with which TBB reads
      # the cores of a hybrid processor, and the table of the tuned kernels of the GPU plugin.
      foreach (part npu_compiler npu_compiler_loader npu_vm_runtime)
         list(APPEND ALBIREO_RUNTIME_EXTRA
            "${openvino_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}openvino_intel_${part}${CMAKE_SHARED_LIBRARY_SUFFIX}")
      endforeach()
      list(APPEND ALBIREO_RUNTIME_EXTRA "${openvino_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}tbbbind*"
         ${tbbbind_libraries} "${openvino_dir}/cache.json")
      foreach (engine ${openvino_engines})
         albireo_device_runtime(${engine})
      endforeach()
      if (ALBIREO_COPY_RUNTIME)
         # The license of the Python package, or of the archive distribution.
         file(GLOB openvino_licenses "${openvino_dir}/../../openvino-*.dist-info/licenses/LICENSE"
            "${openvino_dir}/../../openvino-*.dist-info/LICENSE" "${openvino_dir}/../../../../docs/licensing/LICENSE")
         albireo_copy_license(albireo-engine-openvino openvino-license.txt ${openvino_licenses})
      endif()
   endif()
   if (TARGET openvino::runtime)
      # The graph of the whole decoder (engine/ov_graph.h): engine/ov_graph.cpp alone includes the
      # headers of OpenVINO, which need C++17, and is compiled on its own so that the other sources
      # keep C++14; the engines link the runtime library itself.
      add_library(albireo-ov-graph OBJECT engine/ov_graph.cpp)
      set_target_properties(albireo-ov-graph PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
         POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden
         VISIBILITY_INLINES_HIDDEN ON)
      target_include_directories(albireo-ov-graph PRIVATE ${ALBIREO_INCLUDE_DIR})
      target_compile_options(albireo-ov-graph PRIVATE ${ALBIREO_WARNINGS} ${ALBIREO_ISA_avx2})
      target_link_libraries(albireo-ov-graph PRIVATE openvino::runtime)
      foreach (engine ${openvino_engines})
         target_sources(${engine} PRIVATE $<TARGET_OBJECTS:albireo-ov-graph>)
         target_link_libraries(${engine} PRIVATE $<TARGET_LINKER_FILE:openvino::runtime>)
         target_compile_definitions(${engine} PRIVATE ALBIREO_OV_GRAPH)
      endforeach()
   else()
      message(STATUS "The OpenVINO runtime was not found by find_package(OpenVINO): the engines carry no graph of "
         "the whole decoder.")
   endif()
endif()
if (ALBIREO_ENGINE_DLIB)
   albireo_tensor_engine(dlib)
endif()

if (ALBIREO_PROGRAMS)
   # Libraries compiled into the programs: SQLite and zlib (above) from dlib/external.
   add_library(albireo_sqlite STATIC ${ALBIREO_DLIB_DIR}/external/sqlite/sqlite3.c)
   target_include_directories(albireo_sqlite PUBLIC ${ALBIREO_DLIB_DIR}/external/sqlite)
   target_compile_definitions(albireo_sqlite PRIVATE SQLITE_THREADSAFE=1 SQLITE_DQS=0 SQLITE_DEFAULT_MEMSTATUS=0
      SQLITE_DEFAULT_WAL_SYNCHRONOUS=1 SQLITE_LIKE_DOESNT_MATCH_BLOBS SQLITE_TEMP_STORE=2 SQLITE_SECURE_DELETE
      SQLITE_ENABLE_API_ARMOR SQLITE_ENABLE_FTS5 SQLITE_ENABLE_MATH_FUNCTIONS SQLITE_OMIT_LOAD_EXTENSION)
   if (NOT WIN32)
      target_link_libraries(albireo_sqlite PUBLIC ${CMAKE_DL_LIBS} m)
   endif()

   # Code shared by the service, the tray and the installer.
   add_library(albireo_common STATIC common/log.cpp common/platform.cpp common/system_info.cpp
      common/engine_loader.cpp)
   target_include_directories(albireo_common PUBLIC ${ALBIREO_INCLUDE_DIR})
   target_compile_options(albireo_common PRIVATE ${ALBIREO_WARNINGS})
   if (WIN32)
      target_link_libraries(albireo_common PUBLIC shell32 ole32 bcrypt advapi32)
   else()
      find_package(Threads REQUIRED)
      target_link_libraries(albireo_common PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
   endif()

   # The speech synthesis that reads the answers aloud: SVOX Pico in its own program,
   # albireo-speech, which the service starts for each reading, and its voices of six languages.
   set(pico_dir ${ALBIREO_DLIB_DIR}/external/svox_pico)
   file(GLOB pico_sources ${pico_dir}/lib/*.c)
   add_library(albireo_pico STATIC ${pico_sources})
   target_include_directories(albireo_pico PUBLIC ${pico_dir}/lib)
   # Third-party code of 2009, compiled as it is published: its warnings are not ours to fix.
   # With Visual Studio, /W0 replaces the level of the directory instead of following it, which
   # would print the notice that an option overrides another for each of its files.
   if (MSVC)
      get_target_property(pico_options albireo_pico COMPILE_OPTIONS)
      list(FILTER pico_options EXCLUDE REGEX "/W[0-4]")
      set_target_properties(albireo_pico PROPERTIES COMPILE_OPTIONS "${pico_options};/W0")
   else()
      target_compile_options(albireo_pico PRIVATE -w)
   endif()

   # The program of the speech synthesis: the synthesizer and the runtime of the compiler, no
   # other library, so that it starts with a token of low integrity from the service (host/speech.h).
   add_executable(albireo-speech host/speech_worker.cpp host/speech_text.cpp)
   target_link_libraries(albireo-speech PRIVATE albireo_pico)
   target_include_directories(albireo-speech PRIVATE ${ALBIREO_INCLUDE_DIR})
   target_compile_options(albireo-speech PRIVATE ${ALBIREO_WARNINGS})
   if (WIN32)
      target_sources(albireo-speech PRIVATE resources/speech.rc)
   endif()

   # The password of the public link of the published files (host/public_link.h), read from a
   # file outside the repository at each configuration and built into the service, masked: the
   # header generated lies in the build tree alone.  Without the file, the service reaches the
   # link without password.
   if (WIN32 AND DEFINED ENV{USERPROFILE})
      file(TO_CMAKE_PATH "$ENV{USERPROFILE}/keys/albireo-public-link.txt" public_link_default)
   else()
      set(public_link_default "$ENV{HOME}/keys/albireo-public-link.txt")
   endif()
   set(ALBIREO_PUBLIC_LINK_PASSWORD_FILE "${public_link_default}" CACHE FILEPATH
      "File holding the password of the public link of the published files")
   set(public_link_bytes "0")
   set(public_link_size 0)
   if (EXISTS "${ALBIREO_PUBLIC_LINK_PASSWORD_FILE}")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ALBIREO_PUBLIC_LINK_PASSWORD_FILE}")
      file(READ "${ALBIREO_PUBLIC_LINK_PASSWORD_FILE}" public_link_hex HEX)
      # A byte order mark before the password and the ends of line after it are not part of it.
      string(REGEX REPLACE "^efbbbf" "" public_link_hex "${public_link_hex}")
      string(REGEX REPLACE "(0d|0a)+$" "" public_link_hex "${public_link_hex}")
      string(LENGTH "${public_link_hex}" public_link_digits)
      math(EXPR public_link_size "${public_link_digits} / 2")
      if (public_link_size GREATER 0)
         # Each byte masked by the generator that host/public_link.cpp runs again to unmask it.
         set(public_link_seed 20261101)
         set(public_link_list "")
         math(EXPR public_link_last "${public_link_size} - 1")
         foreach (i RANGE 0 ${public_link_last})
            math(EXPR public_link_at "${i} * 2")
            string(SUBSTRING "${public_link_hex}" ${public_link_at} 2 public_link_byte)
            math(EXPR public_link_seed "(${public_link_seed} * 1103515245 + 12345) % 2147483648")
            math(EXPR public_link_byte "0x${public_link_byte} ^ ((${public_link_seed} >> 16) & 255)")
            list(APPEND public_link_list ${public_link_byte})
         endforeach()
         string(REPLACE ";" ", " public_link_bytes "${public_link_list}")
      endif()
      message(STATUS "Public link: password read from ${ALBIREO_PUBLIC_LINK_PASSWORD_FILE}")
   elseif (WIN32 AND NOT CMAKE_CROSSCOMPILING)
      # The installed services would read the protected link in vain.
      message(WARNING "Public link: no password, ${ALBIREO_PUBLIC_LINK_PASSWORD_FILE} does not exist; the "
         "service built reaches the public link without password")
   else()
      message(STATUS "Public link: no password, ${ALBIREO_PUBLIC_LINK_PASSWORD_FILE} does not exist")
   endif()
   configure_file(cmake/public_link_secret.h.in ${CMAKE_BINARY_DIR}/generated/public_link_secret.h @ONLY)

   # The service.
   add_executable(albireo-service host/main.cpp host/service.cpp host/server.cpp host/settings.cpp host/store.cpp
      host/models.cpp host/updater.cpp host/engine_host.cpp host/http_fetch.cpp host/public_link.cpp host/crypto.cpp
      host/certificates.cpp host/speech.cpp host/speech_text.cpp host/web_access.cpp)
   target_include_directories(albireo-service PRIVATE ${CMAKE_BINARY_DIR}/generated)
   target_link_libraries(albireo-service PRIVATE albireo_common albireo_sqlite albireo_zlib dlib::httplib)
   target_compile_options(albireo-service PRIVATE ${ALBIREO_WARNINGS})
   if (WIN32)
      target_sources(albireo-service PRIVATE resources/service.rc)
      # crypt32: the key of the server encrypted with DPAPI, the authority in the trusted roots.
      target_link_libraries(albireo-service PRIVATE winhttp crypt32)
   endif()

   # The signatures of the update feed and of the model catalog, for their publisher.
   add_executable(albireo-sign tools/albireo_sign.cpp host/crypto.cpp)
   target_link_libraries(albireo-sign PRIVATE albireo_common dlib::httplib)
   target_compile_options(albireo-sign PRIVATE ${ALBIREO_WARNINGS})

   # The preparation of the models that come with Albireo: checked and converted with the
   # code of the engine, then given to the programs of the tree and to the installer.
   albireo_kernels(portable)
   add_executable(albireo-builtin tools/albireo_builtin.cpp engine/chat_format.cpp engine/model_config.cpp
      engine/model_file.cpp $<TARGET_OBJECTS:albireo-kernels-portable> host/crypto.cpp)
   target_link_libraries(albireo-builtin PRIVATE albireo_common dlib::httplib)
   target_compile_options(albireo-builtin PRIVATE ${ALBIREO_WARNINGS})

   # The models that come with Albireo (resources/builtin_models.json): downloaded once into
   # the cache (cmake/builtin_models.cmake), checked and converted by albireo-builtin, then
   # given to the "models" folder of the programs of the tree and, on Windows, to the
   # complete installer.  A cross build takes the albireo-builtin of the build machine.
   if (WIN32 AND NOT CMAKE_CROSSCOMPILING)
      set(builtin_default ON)
   else()
      set(builtin_default OFF)
   endif()
   option(ALBIREO_BUILTIN_MODELS "Prepare the models that come with Albireo and put them in the installer"
      ${builtin_default})
   if (WIN32 AND DEFINED ENV{LOCALAPPDATA})
      file(TO_CMAKE_PATH "$ENV{LOCALAPPDATA}/Albireo/build-cache" cache_default)
   elseif (DEFINED ENV{XDG_CACHE_HOME})
      set(cache_default "$ENV{XDG_CACHE_HOME}/albireo")
   else()
      set(cache_default "$ENV{HOME}/.cache/albireo")
   endif()
   set(ALBIREO_MODEL_CACHE "${cache_default}" CACHE PATH
      "Folder of the downloaded and converted models that come with Albireo")
   set(ALBIREO_BUILTIN_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/resources/builtin_models.json" CACHE FILEPATH
      "Manifest of the models that come with Albireo")
   set(ALBIREO_BUILTIN_TOOL "" CACHE FILEPATH "albireo-builtin of the build machine, for a cross build")
   if (ALBIREO_BUILTIN_MODELS)
      if (CMAKE_VERSION VERSION_LESS 3.19)
         message(FATAL_ERROR "the models that come with Albireo need CMake 3.19 or later "
            "(-DALBIREO_BUILTIN_MODELS=OFF builds without them)")
      endif()
      if (ALBIREO_BUILTIN_TOOL)
         set(ALBIREO_BUILTIN_COMMAND "${ALBIREO_BUILTIN_TOOL}")
         set(builtin_tool_target)
      elseif (CMAKE_CROSSCOMPILING)
         message(FATAL_ERROR "a cross build prepares its models with the albireo-builtin of the build machine: "
            "set ALBIREO_BUILTIN_TOOL")
      else()
         set(ALBIREO_BUILTIN_COMMAND $<TARGET_FILE:albireo-builtin>)
         set(builtin_tool_target albireo-builtin)
      endif()
      set(ALBIREO_BUILTIN_STAMP ${CMAKE_BINARY_DIR}/bin/models/builtin.json)
      add_custom_command(OUTPUT ${ALBIREO_BUILTIN_STAMP}
         COMMAND ${CMAKE_COMMAND} -DMANIFEST=${ALBIREO_BUILTIN_MANIFEST} -DCACHE=${ALBIREO_MODEL_CACHE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/builtin_models.cmake
         COMMAND ${ALBIREO_BUILTIN_COMMAND} --manifest ${ALBIREO_BUILTIN_MANIFEST} --cache ${ALBIREO_MODEL_CACHE}
            --models-out ${CMAKE_BINARY_DIR}/bin/models
         DEPENDS ${ALBIREO_BUILTIN_MANIFEST} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/builtin_models.cmake
            ${builtin_tool_target}
         COMMENT "Preparing the models that come with Albireo"
         VERBATIM)
      add_custom_target(albireo-builtin-models ALL DEPENDS ${ALBIREO_BUILTIN_STAMP})
   endif()

   # The web application and the documentation of the interface, next to the service.
   add_custom_target(albireo-web ALL
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/web ${CMAKE_BINARY_DIR}/bin/web
      COMMENT "Copying the web application")
   add_dependencies(albireo-service albireo-web)
   # The program and the voices of the speech synthesis, next to the service.
   add_custom_target(albireo-speech-voices ALL
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${pico_dir}/lang ${CMAKE_BINARY_DIR}/bin/speech
      COMMENT "Copying the voices of the speech synthesis")
   add_dependencies(albireo-service albireo-speech albireo-speech-voices)

   if (WIN32)
      add_subdirectory(windows)
   endif()
endif()

if (ALBIREO_TESTS AND (ALBIREO_ENGINE_CUDA OR ALBIREO_ENGINE_OPENVINO OR ALBIREO_ENGINE_DLIB))
   # The products of a device against those of the processor kernels, on a real model.
   add_executable(test_offload tests/test_offload.cpp engine/cpu_runtime.cpp engine/model_config.cpp
      engine/model_file.cpp engine/qkernels.cpp engine/tensor_backend.cpp common/system_info.cpp)
   target_include_directories(test_offload PRIVATE ${ALBIREO_INCLUDE_DIR})
   target_compile_definitions(test_offload PRIVATE ALBIREO_TENSOR_BACKEND)
   target_link_libraries(test_offload PRIVATE dlib::dlib)
   if (ALBIREO_ENGINE_CUDA)
      target_link_libraries(test_offload PRIVATE CUDA::cudart)
      target_compile_definitions(test_offload PRIVATE ALBIREO_CUDA_MIN_CC=${ALBIREO_CUDA_MIN_CC})
   endif()
   if (MSVC)
      target_compile_options(test_offload PRIVATE /arch:AVX2)
   else()
      target_compile_options(test_offload PRIVATE -mavx2 -mfma -mf16c)
      find_package(Threads REQUIRED)
      target_link_libraries(test_offload PRIVATE Threads::Threads)
   endif()
   albireo_output_dir(test_offload ${CMAKE_BINARY_DIR}/tests)
   albireo_device_runtime(test_offload)

   # The operations of dlib on the weights in their GGUF formats, against the processor kernels.
   add_executable(test_device_ops tests/test_device_ops.cpp engine/qkernels.cpp common/system_info.cpp)
   target_include_directories(test_device_ops PRIVATE ${ALBIREO_INCLUDE_DIR})
   target_link_libraries(test_device_ops PRIVATE dlib::dlib)
   if (ALBIREO_ENGINE_CUDA)
      target_link_libraries(test_device_ops PRIVATE CUDA::cudart)
   endif()
   if (MSVC)
      target_compile_options(test_device_ops PRIVATE /arch:AVX2)
   else()
      target_compile_options(test_device_ops PRIVATE -mavx2 -mfma -mf16c)
      target_link_libraries(test_device_ops PRIVATE Threads::Threads)
   endif()
   albireo_output_dir(test_device_ops ${CMAKE_BINARY_DIR}/tests)
   albireo_device_runtime(test_device_ops)

   if (TARGET albireo-ov-graph)
      # The graph of the whole decoder against the processor runtime, on a real model.
      add_executable(test_ov_graph tests/test_ov_graph.cpp engine/cpu_runtime.cpp engine/model_config.cpp
         engine/model_file.cpp engine/qkernels.cpp common/system_info.cpp $<TARGET_OBJECTS:albireo-ov-graph>)
      target_include_directories(test_ov_graph PRIVATE ${ALBIREO_INCLUDE_DIR})
      target_link_libraries(test_ov_graph PRIVATE $<TARGET_LINKER_FILE:openvino::runtime>)
      target_compile_options(test_ov_graph PRIVATE ${ALBIREO_WARNINGS} ${ALBIREO_ISA_avx2})
      if (NOT WIN32)
         target_link_libraries(test_ov_graph PRIVATE Threads::Threads)
      endif()
      albireo_output_dir(test_ov_graph ${CMAKE_BINARY_DIR}/tests)
      albireo_device_runtime(test_ov_graph)
   endif()
endif()

if (ALBIREO_TESTS AND ALBIREO_PROGRAMS)
   # The files the service writes in its models folder and reads in its imports, the downloads and the
   # updates from the public link, the settings and the web access, with a local server.
   add_executable(test_host tests/test_host.cpp host/models.cpp host/store.cpp host/http_fetch.cpp
      host/public_link.cpp host/crypto.cpp host/settings.cpp host/web_access.cpp host/updater.cpp)
   target_include_directories(test_host PRIVATE ${CMAKE_BINARY_DIR}/generated)
   target_link_libraries(test_host PRIVATE albireo_common albireo_sqlite albireo_zlib dlib::httplib)
   if (WIN32)
      # The proxy of WinHTTP, which the downloads read as the service does (host/http_fetch.cpp).
      target_link_libraries(test_host PRIVATE winhttp)
   endif()
   target_compile_options(test_host PRIVATE ${ALBIREO_WARNINGS})
   albireo_output_dir(test_host ${CMAKE_BINARY_DIR}/tests)
   # test_model_file: damaged model files; test_engine_api: the C interface of an engine library.
   foreach (t test_qkernels test_runtime test_tokenizer engine_chat bench_runtime test_model_file test_engine_api)
      if (t STREQUAL "engine_chat" OR t STREQUAL "test_engine_api")
         add_executable(${t} tests/${t}.cpp)
         target_link_libraries(${t} PRIVATE albireo_common)
      else()
         albireo_kernels(avx2)
         add_executable(${t} tests/${t}.cpp engine/cpu_runtime.cpp engine/model_config.cpp engine/model_file.cpp
            $<TARGET_OBJECTS:albireo-kernels-avx2> common/system_info.cpp)
         target_include_directories(${t} PRIVATE ${ALBIREO_INCLUDE_DIR})
         target_compile_options(${t} PRIVATE ${ALBIREO_ISA_avx2})
         if (NOT WIN32)
            target_link_libraries(${t} PRIVATE Threads::Threads)
         endif()
      endif()
      albireo_output_dir(${t} ${CMAKE_BINARY_DIR}/tests)
   endforeach()
   # The chat format of the agents: the prompts with tools, and the tool calls read back.
   add_executable(test_chat_format tests/test_chat_format.cpp engine/chat_format.cpp)
   target_include_directories(test_chat_format PRIVATE ${ALBIREO_INCLUDE_DIR})
   albireo_output_dir(test_chat_format ${CMAKE_BINARY_DIR}/tests)
   # The draws of the sampler with the drafts of the answers.
   add_executable(test_sampler tests/test_sampler.cpp engine/sampler.cpp)
   target_compile_options(test_sampler PRIVATE ${ALBIREO_WARNINGS})
   albireo_output_dir(test_sampler ${CMAKE_BINARY_DIR}/tests)
   # The kernels of AVX-VNNI: their checks and their speed, on the processors that have its
   # instructions (the others skip them).
   if (ALBIREO_HAS_avxvnni)
      albireo_kernels(avxvnni)
      foreach (t test_qkernels bench_runtime)
         add_executable(${t}_avxvnni tests/${t}.cpp engine/cpu_runtime.cpp engine/model_config.cpp
            engine/model_file.cpp $<TARGET_OBJECTS:albireo-kernels-avxvnni> common/system_info.cpp)
         target_include_directories(${t}_avxvnni PRIVATE ${ALBIREO_INCLUDE_DIR})
         target_compile_options(${t}_avxvnni PRIVATE ${ALBIREO_ISA_avx2})
         if (NOT WIN32)
            target_link_libraries(${t}_avxvnni PRIVATE Threads::Threads)
         endif()
         albireo_output_dir(${t}_avxvnni ${CMAKE_BINARY_DIR}/tests)
      endforeach()
   endif()
endif()
