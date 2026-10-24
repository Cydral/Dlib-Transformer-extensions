# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Finds the OpenVINO runtime.
#
# dlib uses the OpenVINO runtime to route computations to the NPU and the
# integrated GPU of the processor.  This script looks for its CMake package in
# the usual CMake locations (OpenVINO_DIR as a CMake or environment variable,
# CMAKE_PREFIX_PATH, INTEL_OPENVINO_DIR as set by the setupvars scripts of the
# archive distributions), then in the openvino Python package of the Python
# interpreter found on the PATH, which a
#    python -m pip install openvino
# provides with its headers, libraries and CMake files.
#
#  OpenVINO_FOUND            - True if a usable OpenVINO runtime was found.
#  OpenVINO_VERSION          - Its version.
#  DLIB_OPENVINO_LIBRARY_DIR - The folder holding its shared libraries, which
#                              programs load at run time.

message(STATUS "Searching for OpenVINO")

set(dlib_openvino_hints)
if (DEFINED ENV{INTEL_OPENVINO_DIR})
   file(TO_CMAKE_PATH "$ENV{INTEL_OPENVINO_DIR}" dlib_openvino_root)
   list(APPEND dlib_openvino_hints "${dlib_openvino_root}/runtime/cmake")
endif()
find_package(OpenVINO QUIET COMPONENTS Runtime HINTS ${dlib_openvino_hints})

if (NOT OpenVINO_FOUND)
   # The interpreter that drives a Python build of dlib comes first, then the one on the PATH.
   set(dlib_openvino_python)
   foreach (dlib_python_var Python3_EXECUTABLE Python_EXECUTABLE PYTHON_EXECUTABLE)
      if (NOT dlib_openvino_python AND ${dlib_python_var})
         set(dlib_openvino_python "${${dlib_python_var}}")
      endif()
   endforeach()
   if (NOT dlib_openvino_python)
      if (WIN32)
         find_program(DLIB_OPENVINO_PYTHON NAMES python python3)
      else()
         find_program(DLIB_OPENVINO_PYTHON NAMES python3 python)
      endif()
      mark_as_advanced(DLIB_OPENVINO_PYTHON)
      set(dlib_openvino_python "${DLIB_OPENVINO_PYTHON}")
   endif()
   if (dlib_openvino_python)
      # find_spec locates the package without loading its native modules.
      execute_process(
         COMMAND "${dlib_openvino_python}" -c
            "import importlib.util, os; s = importlib.util.find_spec('openvino'); print(os.path.join(os.path.dirname(s.origin), 'cmake') if s and s.origin else '', end='')"
         OUTPUT_VARIABLE dlib_openvino_python_cmake_dir
         RESULT_VARIABLE dlib_openvino_python_result
         ERROR_QUIET
         TIMEOUT 60)
      if (dlib_openvino_python_result EQUAL 0 AND EXISTS "${dlib_openvino_python_cmake_dir}/OpenVINOConfig.cmake")
         file(TO_CMAKE_PATH "${dlib_openvino_python_cmake_dir}" dlib_openvino_python_cmake_dir)
         find_package(OpenVINO QUIET COMPONENTS Runtime PATHS "${dlib_openvino_python_cmake_dir}" NO_DEFAULT_PATH)
      endif()
   endif()
endif()

if (OpenVINO_FOUND AND NOT TARGET openvino::runtime)
   set(OpenVINO_FOUND 0)
endif()

if (OpenVINO_FOUND)
   # Check that this compiler builds and links a program against the runtime.
   set(dlib_openvino_test_source "${PROJECT_BINARY_DIR}/test_for_openvino/test_for_openvino.cpp")
   file(WRITE "${dlib_openvino_test_source}"
      "#include <openvino/openvino.hpp>\n"
      "#include <openvino/op/ops.hpp>\n"
      "int main()\n"
      "{\n"
      "    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});\n"
      "    auto y = std::make_shared<ov::op::v0::Relu>(x);\n"
      "    auto model = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x});\n"
      "    ov::Core core;\n"
      "    return model->get_parameters().size() == 1 ? 0 : 1;\n"
      "}\n")
   try_compile(test_for_openvino_worked
      "${PROJECT_BINARY_DIR}/test_for_openvino_build"
      SOURCES "${dlib_openvino_test_source}"
      LINK_LIBRARIES openvino::runtime
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      OUTPUT_VARIABLE dlib_openvino_test_output)
   if (NOT test_for_openvino_worked)
      set(OpenVINO_FOUND 0)
      message(STATUS "Found OpenVINO ${OpenVINO_VERSION} in ${OpenVINO_DIR}, but a test program could not be built against it with this compiler.")
   endif()
endif()

if (OpenVINO_FOUND)
   set(DLIB_OPENVINO_LIBRARY_DIR)
   get_target_property(dlib_openvino_configs openvino::runtime IMPORTED_CONFIGURATIONS)
   foreach (dlib_openvino_config RELEASE ${dlib_openvino_configs})
      if (NOT DLIB_OPENVINO_LIBRARY_DIR)
         get_target_property(dlib_openvino_location openvino::runtime IMPORTED_LOCATION_${dlib_openvino_config})
         if (dlib_openvino_location)
            get_filename_component(DLIB_OPENVINO_LIBRARY_DIR "${dlib_openvino_location}" DIRECTORY)
         endif()
      endif()
   endforeach()
   message(STATUS "Found OpenVINO ${OpenVINO_VERSION} in ${OpenVINO_DIR}")
else()
   message(" *****************************************************************************")
   message(" *** No OpenVINO runtime found, so dlib will not route computations to the ***")
   message(" *** NPU or the integrated GPU.  The simplest way to get one is the Python ***")
   message(" *** package:                                                              ***")
   message(" ***    python -m pip install openvino                                     ***")
   message(" *** You can also point OpenVINO_DIR to the folder holding                 ***")
   message(" *** OpenVINOConfig.cmake.  Then re-run cmake with -DDLIB_USE_NPU=ON.      ***")
   message(" *****************************************************************************")
endif()
