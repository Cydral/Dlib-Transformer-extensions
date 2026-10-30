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
project(albireo VERSION 1.0.6 LANGUAGES C CXX)

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

# Inference engines.
set(ALBIREO_ENGINE_SOURCES
   engine/chat_format.cpp
   engine/cpu_runtime.cpp
   engine/engine.cpp
   engine/engine_api.cpp
   engine/model_config.cpp
   engine/model_file.cpp
   engine/sampler.cpp
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
   set(ALBIREO_ISA_avx512 /arch:AVX512)
   set(ALBIREO_ISA_avx512_DEFS ALBIREO_AVX512_VNNI)
   # The intrinsics of AVX-VNNI came with Visual Studio 2022.
   if (MSVC_VERSION GREATER_EQUAL 1930)
      set(ALBIREO_HAS_avxvnni ON)
   endif()
   set(ALBIREO_HAS_avx512 ON)
else()
   set(ALBIREO_ISA_avx2 -mavx2 -mfma -mf16c)
   set(ALBIREO_ISA_avxvnni ${ALBIREO_ISA_avx2} -mavxvnni)
   set(ALBIREO_ISA_avx512 ${ALBIREO_ISA_avx2} -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni)
   # The 256-bit VNNI kernels through the AVX-512 form of their instructions, to test them
   # on the processors that have AVX-512 VNNI and lack AVX-VNNI.
   set(ALBIREO_ISA_vnni256 ${ALBIREO_ISA_avx2} -mavx512vl -mavx512vnni)
   set(ALBIREO_ISA_vnni256_DEFS ALBIREO_QK_VNNI_EVEX)
   check_cxx_compiler_flag(-mavxvnni ALBIREO_HAS_avxvnni)
   check_cxx_compiler_flag(-mavx512vnni ALBIREO_HAS_avx512)
   set(ALBIREO_HAS_vnni256 ${ALBIREO_HAS_avx512})
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
   # AVX-VNNI: Intel Core 12th generation and later, AMD Zen 5.
   if (ALBIREO_HAS_avxvnni)
      albireo_engine(cpu-avxvnni avxvnni)
   else()
      message(STATUS "The compiler lacks AVX-VNNI: no cpu-avxvnni engine.")
   endif()
   # AVX-512 with VNNI: Intel Xeon from Cascade Lake on, Core 10th and 11th generation for
   # laptops and desktops (Ice Lake, Tiger Lake, Rocket Lake), AMD Zen 4 and Zen 5.
   if (ALBIREO_HAS_avx512)
      albireo_engine(cpu-avx512 avx512)
   else()
      message(STATUS "The compiler lacks AVX-512 VNNI: no cpu-avx512 engine.")
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
# The runtime of the compiler, which the libraries of the devices built with it load.
set(ALBIREO_COMPILER_RUNTIME)
if (MSVC AND (ALBIREO_ENGINE_CUDA OR ALBIREO_ENGINE_OPENVINO))
   set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
   set(CMAKE_INSTALL_UCRT_LIBRARIES FALSE)
   include(InstallRequiredSystemLibraries)
   set(ALBIREO_COMPILER_RUNTIME ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
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

# The runtime libraries of the device of this tree, which its engine and its test program
# load: the folders they come from, more files to copy, the names to leave out, and the
# libraries loaded on first use only (MSVC).  The test program receives more files than
# the engine: the CPU plugin of OpenVINO, for a machine without NPU or Intel graphics.
set(ALBIREO_RUNTIME_DIRS)
set(ALBIREO_RUNTIME_EXTRA)
set(ALBIREO_RUNTIME_EXCLUDE)
set(ALBIREO_DELAYED_LIBRARIES)
set(ALBIREO_TEST_RUNTIME_EXTRA)

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
# of the family given after the variant (avxvnni, avx512), the rest of the library staying at
# AVX2.
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
   # The folders of the toolkit and of cuDNN; cuDNN loads its parts by name, those of the
   # tensor descriptors and of the elementary operations being the ones the engine uses.
   set(ALBIREO_RUNTIME_DIRS ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64)
   if (CUDNN_LIBRARY_PATH)
      get_filename_component(cudnn_lib_dir "${CUDNN_LIBRARY_PATH}" DIRECTORY)
      string(REGEX REPLACE "/lib(/|$)" "/bin\\1" cudnn_bin_dir "${cudnn_lib_dir}")
      get_filename_component(cudnn_bin_parent "${cudnn_bin_dir}" DIRECTORY)
      list(APPEND ALBIREO_RUNTIME_DIRS "${cudnn_bin_dir}" "${cudnn_bin_parent}")
   endif()
   foreach (dir ${ALBIREO_RUNTIME_DIRS})
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
   if (ALBIREO_CUDA_MIN_CC GREATER_EQUAL 75)
      message(STATUS "CUDA engine: the GeForce GTX 900 and 10 series need CUDA 12.8 or 12.9 with cuDNN 9.11 "
         "or earlier")
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
   if (ALBIREO_HAS_avx512)
      albireo_tensor_engine(openvino-avx512 avx512)
      list(APPEND openvino_engines albireo-engine-openvino-avx512)
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
   if (openvino_dir)
      set(ALBIREO_RUNTIME_DIRS "${openvino_dir}")
      foreach (device npu gpu)
         set(plugin "${CMAKE_SHARED_LIBRARY_PREFIX}openvino_intel_${device}_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}")
         list(APPEND ALBIREO_RUNTIME_EXTRA "${openvino_dir}/${plugin}")
      endforeach()
      foreach (engine ${openvino_engines})
         albireo_device_runtime(${engine})
      endforeach()
      set(ALBIREO_TEST_RUNTIME_EXTRA
         "${openvino_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}openvino_intel_cpu_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}")
      if (ALBIREO_COPY_RUNTIME)
         # The license of the Python package, or of the archive distribution.
         file(GLOB openvino_licenses "${openvino_dir}/../../openvino-*.dist-info/licenses/LICENSE"
            "${openvino_dir}/../../openvino-*.dist-info/LICENSE" "${openvino_dir}/../../../../docs/licensing/LICENSE")
         albireo_copy_license(albireo-engine-openvino openvino-license.txt ${openvino_licenses})
      endif()
   endif()
endif()
if (ALBIREO_ENGINE_DLIB)
   albireo_tensor_engine(dlib)
endif()

if (ALBIREO_PROGRAMS)
   # Libraries compiled into the programs: SQLite and zlib from dlib/external.
   add_library(albireo_sqlite STATIC ${ALBIREO_DLIB_DIR}/external/sqlite/sqlite3.c)
   target_include_directories(albireo_sqlite PUBLIC ${ALBIREO_DLIB_DIR}/external/sqlite)
   target_compile_definitions(albireo_sqlite PRIVATE SQLITE_THREADSAFE=1 SQLITE_DQS=0 SQLITE_DEFAULT_MEMSTATUS=0
      SQLITE_DEFAULT_WAL_SYNCHRONOUS=1 SQLITE_LIKE_DOESNT_MATCH_BLOBS SQLITE_TEMP_STORE=2 SQLITE_SECURE_DELETE
      SQLITE_ENABLE_API_ARMOR SQLITE_ENABLE_FTS5 SQLITE_ENABLE_MATH_FUNCTIONS SQLITE_OMIT_LOAD_EXTENSION)
   if (NOT WIN32)
      target_link_libraries(albireo_sqlite PUBLIC ${CMAKE_DL_LIBS} m)
   endif()

   set(zlib_dir ${ALBIREO_DLIB_DIR}/external/zlib)
   add_library(albireo_zlib STATIC ${zlib_dir}/adler32.c ${zlib_dir}/compress.c ${zlib_dir}/crc32.c
      ${zlib_dir}/deflate.c ${zlib_dir}/infback.c ${zlib_dir}/inffast.c ${zlib_dir}/inflate.c ${zlib_dir}/inftrees.c
      ${zlib_dir}/trees.c ${zlib_dir}/uncompr.c ${zlib_dir}/zutil.c)
   target_include_directories(albireo_zlib PUBLIC ${zlib_dir})

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

   # The service.
   add_executable(albireo-service host/main.cpp host/service.cpp host/server.cpp host/settings.cpp host/store.cpp
      host/models.cpp host/updater.cpp host/engine_host.cpp host/http_fetch.cpp host/crypto.cpp host/certificates.cpp)
   target_link_libraries(albireo-service PRIVATE albireo_common albireo_sqlite albireo_zlib dlib::httplib)
   target_compile_options(albireo-service PRIVATE ${ALBIREO_WARNINGS})
   if (WIN32)
      target_sources(albireo-service PRIVATE resources/service.rc)
      target_link_libraries(albireo-service PRIVATE winhttp)
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
   list(APPEND ALBIREO_RUNTIME_EXTRA ${ALBIREO_TEST_RUNTIME_EXTRA})
   albireo_device_runtime(test_offload)
endif()

if (ALBIREO_TESTS AND ALBIREO_PROGRAMS)
   foreach (t test_qkernels test_runtime test_tokenizer engine_chat bench_runtime)
      if (t STREQUAL "engine_chat")
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
   # The kernels of the other families: their checks and their speed, on the processors
   # that have their instructions (the others skip them).
   foreach (family avxvnni avx512 vnni256)
      if (NOT ALBIREO_HAS_${family})
         continue()
      endif()
      albireo_kernels(${family})
      foreach (t test_qkernels bench_runtime)
         add_executable(${t}_${family} tests/${t}.cpp engine/cpu_runtime.cpp engine/model_config.cpp
            engine/model_file.cpp $<TARGET_OBJECTS:albireo-kernels-${family}> common/system_info.cpp)
         target_include_directories(${t}_${family} PRIVATE ${ALBIREO_INCLUDE_DIR})
         target_compile_options(${t}_${family} PRIVATE ${ALBIREO_ISA_avx2})
         if (NOT WIN32)
            target_link_libraries(${t}_${family} PRIVATE Threads::Threads)
         endif()
         albireo_output_dir(${t}_${family} ${CMAKE_BINARY_DIR}/tests)
      endforeach()
   endforeach()
endif()
