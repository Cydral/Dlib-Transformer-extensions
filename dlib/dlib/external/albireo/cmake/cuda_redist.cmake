# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Lays out in a cache the CUDA toolkit and cuDNN with which the CUDA engine is built on Windows,
# from the redistributable archives that NVIDIA publishes for windows-x86_64, for a computer
# that has neither of them installed (cmake -P, CMake 3.19 or later):
#
#   cmake -DCACHE=<folder> [-DCUDA_VERSION=12.9.1] [-DCUDNN_VERSION=9.11.0] [-DCUDNN_ONLY=ON]
#         -P cuda_redist.cmake
#
# The manifests of the two releases give each archive with its size and its SHA-256.  The
# archives are downloaded once into <CACHE>/archives, checked against their hash, and downloaded
# again when it does not match.  The folders of each archive are then merged into
# <CACHE>/cuda-<version>, a standalone toolkit for the toolset cuda=<folder> of the Visual
# Studio generators, which find its integration in extras/visual_studio_integration/MSBuildExtensions,
# and into <CACHE>/cudnn-<version>-cuda<major>.  A stamp in each folder records the versions,
# the hashes of the archives and the hash of this script: a second run finds it, and downloads
# and extracts nothing.  The last lines give the two folders and the bytes downloaded:
#
#   -- CUDA_REDIST_TOOLKIT=<folder>
#   -- CUDA_REDIST_CUDNN=<folder>
#   -- CUDA_REDIST_DOWNLOADED=<bytes>
#
# CUDNN_ONLY lays out cuDNN alone, in its build for the major version of CUDA_VERSION, for an
# installed toolkit whose own cuDNN leaves out cards the engine serves; CUDA_REDIST_TOOLKIT is
# then not printed.
#
# The toolkit holds what the engine is built with: the compiler (cuda_nvcc), the runtime
# (cuda_cudart), the headers of CCCL that those of the runtime include (cuda_cccl), cuBLAS,
# cuRAND and cuSOLVER, which dlib links, and the integration of Visual Studio.  cuSPARSE, which
# cuSOLVER loads, and nvJitLink, which cuSPARSE loads, stay out: FindCUDAToolkit links them to
# CUDA::cusolver only when it finds them, and the engine, which loads cuSOLVER on first use only,
# never calls it.
# The downloads go through the proxy of the https_proxy variable, when set, and stop after two
# minutes without data.

if (CMAKE_VERSION VERSION_LESS 3.19)
   message(FATAL_ERROR "cuda_redist.cmake needs CMake 3.19 or later")
endif()
if (NOT CACHE)
   message(FATAL_ERROR "cuda_redist.cmake needs CACHE")
endif()
if (NOT CUDA_VERSION)
   set(CUDA_VERSION 12.9.1)
endif()
if (NOT CUDNN_VERSION)
   set(CUDNN_VERSION 9.11.0)
endif()
file(TO_CMAKE_PATH "${CACHE}" CACHE)
string(REGEX MATCH "^[0-9]+" cuda_major "${CUDA_VERSION}")
set(redist_url https://developer.download.nvidia.com/compute)
set(toolkit "${CACHE}/cuda-${CUDA_VERSION}")
set(cudnn "${CACHE}/cudnn-${CUDNN_VERSION}-cuda${cuda_major}")
set(downloaded 0)

# One run at a time in the cache, so that a run started during another finds the folders complete.
file(MAKE_DIRECTORY "${CACHE}")
file(LOCK "${CACHE}/redist.lock" GUARD PROCESS TIMEOUT 3600 RESULT_VARIABLE lock_result)
if (NOT lock_result EQUAL 0)
   message(FATAL_ERROR "${CACHE} stays locked by another run: ${lock_result}")
endif()

# Reads into out the manifest of a release of a product (cuda, cudnn), downloaded once.
function(albireo_redist_manifest product version out)
   set(file "${CACHE}/manifests/${product}-${version}.json")
   set(url "${redist_url}/${product}/redist/redistrib_${version}.json")
   if (NOT EXISTS "${file}")
      message(STATUS "Downloading ${url}")
      file(DOWNLOAD "${url}" "${file}.partial" TLS_VERIFY ON INACTIVITY_TIMEOUT 120 STATUS status)
      list(GET status 0 code)
      if (NOT code EQUAL 0)
         list(GET status 1 text)
         file(REMOVE "${file}.partial")
         message(FATAL_ERROR "${url} could not be downloaded: ${text}")
      endif()
      file(RENAME "${file}.partial" "${file}")
   endif()
   file(READ "${file}" json)
   string(JSON label ERROR_VARIABLE error GET "${json}" release_label)
   if (error OR NOT label STREQUAL version)
      # A file of another content, such as a page of a proxy, is deleted for the next run to download it again.
      file(REMOVE "${file}")
      message(FATAL_ERROR "${url} gave a file that is not the manifest of ${product} ${version}, which the next run "
         "downloads again")
   endif()
   set(${out} "${json}" PARENT_SCOPE)
endfunction()

# Appends to the list out the archive of a component in a manifest, as
# <component>|<url>|<sha256>|<size>, under the members given after out: windows-x86_64, then
# cuda<major> for cuDNN, which publishes one archive per major version of CUDA.
function(albireo_redist_entry json product component out)
   string(JSON path ERROR_VARIABLE error GET "${json}" ${component} ${ARGN} relative_path)
   if (error)
      message(FATAL_ERROR "the manifest of ${product} has no archive ${component} ${ARGN}: ${error}")
   endif()
   string(JSON sha GET "${json}" ${component} ${ARGN} sha256)
   string(JSON size GET "${json}" ${component} ${ARGN} size)
   set(${out} ${${out}} "${component}|${redist_url}/${product}/redist/${path}|${sha}|${size}" PARENT_SCOPE)
endfunction()

# Moves the files under the folder from to the same places under the folder to.
function(albireo_redist_move from to)
   file(GLOB_RECURSE files LIST_DIRECTORIES false RELATIVE "${from}" "${from}/*")
   foreach (f IN LISTS files)
      get_filename_component(dir "${to}/${f}" DIRECTORY)
      file(MAKE_DIRECTORY "${dir}")
      file(RENAME "${from}/${f}" "${to}/${f}")
   endforeach()
endfunction()

# Lays out the folder dest from its archives, unless its stamp records them already.  The
# LICENSE of the archive of license_component becomes license_name, where an installation has
# its license.
function(albireo_redist_folder dest label archives license_component license_name)
   file(SHA256 "${CMAKE_CURRENT_LIST_FILE}" script_sha)
   string(REPLACE ";" "\n" stamp "${label};${archives};${script_sha}\n")
   if (EXISTS "${dest}/albireo-redist.stamp")
      file(READ "${dest}/albireo-redist.stamp" have)
      if (have STREQUAL stamp)
         return()
      endif()
   endif()
   # Every archive first, so that a download cut short leaves no folder half laid out.
   foreach (archive IN LISTS archives)
      string(REPLACE "|" ";" fields "${archive}")
      list(GET fields 1 url)
      list(GET fields 2 sha)
      list(GET fields 3 size)
      get_filename_component(name "${url}" NAME)
      set(file "${CACHE}/archives/${name}")
      if (EXISTS "${file}")
         file(SIZE "${file}" have_size)
         set(have_sha "")
         if (have_size EQUAL size)
            file(SHA256 "${file}" have_sha)
         endif()
         if (NOT have_sha STREQUAL sha)
            message(STATUS "${file} does not match its SHA-256: downloaded again")
            file(REMOVE "${file}")
         endif()
      endif()
      if (NOT EXISTS "${file}")
         math(EXPR mb "(${size} + 524287) / 1048576")
         message(STATUS "Downloading ${name}, ${mb} MB")
         # A transfer cut short or a file of another hash stops the script, with the reason; the
         # partial file it leaves is replaced by the next download.
         file(REMOVE "${file}.partial")
         file(DOWNLOAD "${url}" "${file}.partial" EXPECTED_HASH SHA256=${sha} SHOW_PROGRESS TLS_VERIFY ON
            INACTIVITY_TIMEOUT 120)
         file(RENAME "${file}.partial" "${file}")
         math(EXPR downloaded "${downloaded} + ${size}")
      endif()
   endforeach()
   file(REMOVE_RECURSE "${dest}")
   file(MAKE_DIRECTORY "${dest}")
   set(work "${CACHE}/x")
   foreach (archive IN LISTS archives)
      string(REPLACE "|" ";" fields "${archive}")
      list(GET fields 0 component)
      list(GET fields 1 url)
      get_filename_component(name "${url}" NAME)
      message(STATUS "Extracting ${name}")
      file(REMOVE_RECURSE "${work}")
      if (component STREQUAL "visual_studio_integration")
         # Its extensions of MSBuild and its license, without the templates of Visual Studio projects.
         file(ARCHIVE_EXTRACT INPUT "${CACHE}/archives/${name}" DESTINATION "${work}"
            PATTERNS "*/LICENSE" "*/MSBuildExtensions")
      else()
         file(ARCHIVE_EXTRACT INPUT "${CACHE}/archives/${name}" DESTINATION "${work}")
      endif()
      # The archive holds one folder, named after it, whose folders are merged into dest.
      file(GLOB top LIST_DIRECTORIES true "${work}/*")
      list(LENGTH top count)
      if (NOT count EQUAL 1 OR NOT IS_DIRECTORY "${top}")
         message(FATAL_ERROR "${name} does not hold a single folder")
      endif()
      foreach (sub bin include lib nvvm)
         if (IS_DIRECTORY "${top}/${sub}")
            albireo_redist_move("${top}/${sub}" "${dest}/${sub}")
         endif()
      endforeach()
      if (IS_DIRECTORY "${top}/visual_studio_integration/MSBuildExtensions")
         albireo_redist_move("${top}/visual_studio_integration/MSBuildExtensions"
            "${dest}/extras/visual_studio_integration/MSBuildExtensions")
      endif()
      if (component STREQUAL license_component)
         file(RENAME "${top}/LICENSE" "${dest}/${license_name}")
      endif()
   endforeach()
   file(REMOVE_RECURSE "${work}")
   file(WRITE "${dest}/albireo-redist.stamp" "${stamp}")
   set(downloaded ${downloaded} PARENT_SCOPE)
endfunction()

if (NOT CUDNN_ONLY)
   albireo_redist_manifest(cuda ${CUDA_VERSION} cuda_manifest)
   set(toolkit_archives)
   foreach (component cuda_nvcc cuda_cudart cuda_cccl libcublas libcurand libcusolver visual_studio_integration)
      albireo_redist_entry("${cuda_manifest}" cuda ${component} toolkit_archives windows-x86_64)
   endforeach()
endif()
albireo_redist_manifest(cudnn ${CUDNN_VERSION} cudnn_manifest)
set(cudnn_archives)
albireo_redist_entry("${cudnn_manifest}" cudnn cudnn cudnn_archives windows-x86_64 cuda${cuda_major})

# The EULA of the toolkit is the LICENSE of each of its archives, in revisions of several dates:
# the one of cuBLAS, whose libraries make most of what the installer carries, becomes EULA.txt,
# where an installed toolkit has it.
if (NOT CUDNN_ONLY)
   albireo_redist_folder("${toolkit}" "CUDA ${CUDA_VERSION}" "${toolkit_archives}" libcublas EULA.txt)
endif()
albireo_redist_folder("${cudnn}" "cuDNN ${CUDNN_VERSION}" "${cudnn_archives}" cudnn LICENSE)

if (NOT CUDNN_ONLY)
   message(STATUS "CUDA_REDIST_TOOLKIT=${toolkit}")
endif()
message(STATUS "CUDA_REDIST_CUDNN=${cudnn}")
message(STATUS "CUDA_REDIST_DOWNLOADED=${downloaded}")
