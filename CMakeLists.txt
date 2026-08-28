#   Copyright (c) 2026 Scripps Research, Forli Lab.
#   All rights reserved.
#
#   Author: Niccolo Bruciaferri
#
#   This library is free software; you can redistribute it and/or
#   modify it under the terms of the GNU Lesser General Public
#   License as published by the Free Software Foundation; either
#   version 2.1 of the License, or (at your option) any later version.
#
#   This library is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public
#   License along with this library; if not, write to the Free Software
#   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# CUDA architecture selection.
#
# This must be resolved BEFORE project(), because CMake already passes
# -gencode flags during CUDA compiler detection: a bad value there fails as
# "Check for working CUDA compiler ... broken" instead of as a normal error.
#
# Do NOT hardcode a list and do NOT simply skip this when the variable is
# already set. Which architectures nvcc accepts changes between toolkit major
# versions (CUDA 13 dropped Maxwell/Pascal/Volta, i.e. everything below
# compute_75), and CMake's own "all" keyword expands from a built-in table
# that can disagree with the installed toolkit. Either one produces
#     nvcc fatal : Unsupported gpu architecture 'compute_50'
# Environments frequently inject a value (CMAKE_ARGS / CUDAARCHS / -D from a
# build frontend such as scikit-build-core), so we always ask the toolkit what
# it really supports and then validate/repair the requested value.
#
# Override at configure time for a specific build, e.g.:
#   -DCMAKE_CUDA_ARCHITECTURES=native   (only the GPU(s) present locally)
#   -DCMAKE_CUDA_ARCHITECTURES="90;120" (an explicit subset; still validated)
# ---------------------------------------------------------------------------
if(DEFINED CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CUDA_COMPILER}")
  set(_mcswell_nvcc "${CMAKE_CUDA_COMPILER}")
elseif(DEFINED ENV{CUDACXX} AND EXISTS "$ENV{CUDACXX}")
  set(_mcswell_nvcc "$ENV{CUDACXX}")
else()
  find_program(_mcswell_nvcc NAMES nvcc HINTS ENV CUDA_HOME ENV CUDA_PATH
               PATH_SUFFIXES bin)
endif()

set(_mcswell_arch_supported "")
if(_mcswell_nvcc)
  execute_process(
    COMMAND "${_mcswell_nvcc}" --list-gpu-arch
    OUTPUT_VARIABLE _mcswell_nvcc_out
    RESULT_VARIABLE _mcswell_nvcc_res
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(_mcswell_nvcc_res EQUAL 0)
    string(REGEX MATCHALL "compute_[0-9]+" _mcswell_arch_matches "${_mcswell_nvcc_out}")
    foreach(_mcswell_arch IN LISTS _mcswell_arch_matches)
      string(REGEX REPLACE "^compute_" "" _mcswell_arch "${_mcswell_arch}")
      list(APPEND _mcswell_arch_supported "${_mcswell_arch}")
    endforeach()
  endif()
endif()

if(_mcswell_arch_supported)
  list(REMOVE_DUPLICATES _mcswell_arch_supported)
  list(SORT _mcswell_arch_supported COMPARE NATURAL)
else()
  # nvcc could not be queried (too old for --list-gpu-arch, or not found yet).
  # Fall back to the range every supported toolkit (CUDA >= 12.0) accepts.
  set(_mcswell_arch_supported "75;80;86;89;90")
endif()

string(TOLOWER "${CMAKE_CUDA_ARCHITECTURES}" _mcswell_arch_request)

if(_mcswell_arch_request STREQUAL "native")
  # Resolved by nvcc against the local GPU; nothing to validate here.
elseif(NOT _mcswell_arch_request
       OR _mcswell_arch_request STREQUAL "all"
       OR _mcswell_arch_request STREQUAL "all-major")
  # Native (SASS) code for every architecture this toolkit supports, plus a
  # PTX-only ("virtual") fallback for the newest one. PTX is forward
  # compatible, so a GPU newer than anything listed here JIT-compiles that PTX
  # at first kernel launch instead of failing with "no kernel image is
  # available for this device".
  list(GET _mcswell_arch_supported -1 _mcswell_arch_newest)
  set(_mcswell_arch_final ${_mcswell_arch_supported} "${_mcswell_arch_newest}-virtual")
  set(CMAKE_CUDA_ARCHITECTURES "${_mcswell_arch_final}"
      CACHE STRING "CUDA architectures to generate device code for" FORCE)
else()
  set(_mcswell_arch_final "")
  set(_mcswell_arch_dropped "")
  foreach(_mcswell_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX MATCH "^[0-9]+" _mcswell_arch_num "${_mcswell_arch}")
    if(_mcswell_arch_num AND _mcswell_arch_num IN_LIST _mcswell_arch_supported)
      list(APPEND _mcswell_arch_final "${_mcswell_arch}")
    else()
      list(APPEND _mcswell_arch_dropped "${_mcswell_arch}")
    endif()
  endforeach()

  if(_mcswell_arch_dropped)
    message(WARNING
      "Ignoring CUDA architecture(s) '${_mcswell_arch_dropped}': not supported "
      "by ${_mcswell_nvcc} (supports ${_mcswell_arch_supported}).")
  endif()

  if(NOT _mcswell_arch_final)
    list(GET _mcswell_arch_supported -1 _mcswell_arch_newest)
    set(_mcswell_arch_final ${_mcswell_arch_supported} "${_mcswell_arch_newest}-virtual")
  endif()

  set(CMAKE_CUDA_ARCHITECTURES "${_mcswell_arch_final}"
      CACHE STRING "CUDA architectures to generate device code for" FORCE)
endif()

message(STATUS "MCSwell: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

project(mcswell_cpp LANGUAGES CXX CUDA)

# Enforce the CUDA Toolkit floor documented in README.md instead of letting
# an older toolkit fail later with a confusing, unrelated error.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
  message(FATAL_ERROR
    "MCSwell requires CUDA Toolkit >= 12.0 (found ${CMAKE_CUDA_COMPILER_VERSION} "
    "at ${CMAKE_CUDA_COMPILER}). See README.md for installation instructions.")
endif()

# A conda/mamba-provided nvcc (commonly pulled in transitively by OpenMM)
# resolved via PATH instead of the system CUDA Toolkit is a known source of
# glibc/nvcc header conflicts (_Float64/_Float128/issignaling errors from
# <bits/mathcalls-narrow.h>, <bits/iscanonical.h>, <math.h>) on newer distros
# such as Ubuntu 24.04 -- reproduced and confirmed against this project's own
# CMakeLists.txt (see issue #4). Warn early instead of letting users hit a
# cryptic compiler-detection failure.
if(CMAKE_CUDA_COMPILER MATCHES "(conda|mamba)[a-zA-Z0-9_]*[/\\]envs[/\\]|[/\\](miniconda3|anaconda3|miniforge3|mambaforge)[/\\]")
  message(WARNING
    "CMAKE_CUDA_COMPILER (${CMAKE_CUDA_COMPILER}) appears to come from a "
    "conda/mamba environment rather than the system CUDA Toolkit. This is a "
    "known source of glibc/nvcc header conflicts on newer distros (see "
    "README.md, 'Remove conda CUDA packages'). If configuration fails with "
    "_Float64/_Float128/issignaling errors, remove the conda-provided "
    "cuda-nvcc package or set -DCMAKE_CUDA_COMPILER=/path/to/system/nvcc "
    "explicitly (e.g. /usr/bin/nvcc or /usr/local/cuda/bin/nvcc).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ---- Python extension (note the leading underscore) ----
pybind11_add_module(_mcswell_cpp
    src/bindings.cpp
    src/insertion_points.cpp
    src/run_mcswell.cpp
    src/cuda/cuda_kernels.cu
    src/cuda/launch.cu
)

set_target_properties(_mcswell_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mcswell_cpp"
)

target_include_directories(_mcswell_cpp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src   # keep if you still include headers from src (ideally none)
)

target_link_libraries(_mcswell_cpp PRIVATE
    CUDA::cudart
    CUDA::curand
)

if (NOT MSVC)
  target_compile_options(_mcswell_cpp PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

# --- Water model selection ---
set(WATER_MODEL "TIP4P" CACHE STRING "Water model: TIP3P or TIP4P")
set_property(CACHE WATER_MODEL PROPERTY STRINGS TIP3P TIP4P)

if(WATER_MODEL STREQUAL "TIP3P")
  target_compile_definitions(_mcswell_cpp PRIVATE TIP3P)
  target_compile_definitions(_mcswell_cpp PRIVATE WATER_MODEL_NAME="${WATER_MODEL}")
elseif(WATER_MODEL STREQUAL "TIP4P")
  target_compile_definitions(_mcswell_cpp PRIVATE TIP4P)
  target_compile_definitions(_mcswell_cpp PRIVATE WATER_MODEL_NAME="${WATER_MODEL}")
else()
  message(FATAL_ERROR "Invalid WATER_MODEL='${WATER_MODEL}'. Use TIP3P or TIP4P.")
endif()

install(TARGETS _mcswell_cpp
        LIBRARY DESTINATION mcswell_cpp
        RUNTIME DESTINATION mcswell_cpp
        ARCHIVE DESTINATION mcswell_cpp.)

set_target_properties(_mcswell_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mcswell_cpp"
)

target_include_directories(_mcswell_cpp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src   # keep if you still include headers from src (ideally none)
)

target_link_libraries(_mcswell_cpp PRIVATE
    CUDA::cudart
    CUDA::curand
)

if (NOT MSVC)
  target_compile_options(_mcswell_cpp PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

# --- Water model selection ---
set(WATER_MODEL "TIP4P" CACHE STRING "Water model: TIP3P or TIP4P")
set_property(CACHE WATER_MODEL PROPERTY STRINGS TIP3P TIP4P)

if(WATER_MODEL STREQUAL "TIP3P")
  target_compile_definitions(_mcswell_cpp PRIVATE TIP3P)
  target_compile_definitions(_mcswell_cpp PRIVATE WATER_MODEL_NAME="${WATER_MODEL}")
elseif(WATER_MODEL STREQUAL "TIP4P")
  target_compile_definitions(_mcswell_cpp PRIVATE TIP4P)
  target_compile_definitions(_mcswell_cpp PRIVATE WATER_MODEL_NAME="${WATER_MODEL}")
else()
  message(FATAL_ERROR "Invalid WATER_MODEL='${WATER_MODEL}'. Use TIP3P or TIP4P.")
endif()

install(TARGETS _mcswell_cpp
        LIBRARY DESTINATION mcswell_cpp
        RUNTIME DESTINATION mcswell_cpp
        ARCHIVE DESTINATION mcswell_cpp.)
