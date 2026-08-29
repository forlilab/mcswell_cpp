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

# Must be set BEFORE project() so CMake uses it during CUDA compiler detection.
# Native (SASS) code for every desktop/datacenter architecture from Maxwell
# through Hopper, which every CUDA >= 12.0 toolkit can target, plus a
# PTX-only ("virtual") fallback for the newest of them (Hopper). PTX is
# forward-compatible: a GPU newer than anything listed here (e.g. Blackwell,
# or an architecture that doesn't exist yet) will JIT-compile that PTX at
# first kernel launch instead of failing with "no kernel image is available
# for this device."
#
# Override at configure time for a specific build, e.g.:
#   -DCMAKE_CUDA_ARCHITECTURES=native        (only the GPU(s) present locally)
#   -DCMAKE_CUDA_ARCHITECTURES="90;100;120"  (add native Blackwell codegen,
#                                              requires CUDA >= 12.8)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86;89;90;90-virtual")
endif()

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

# ---- nanoflann (header-only KD-tree, used by the post-simulation analysis
#      library below) -----------------------------------------------------
# Fetched as plain sources via FetchContent_Populate (not
# FetchContent_MakeAvailable) so that nanoflann's own CMakeLists.txt --
# which adds its own example/test targets and pollutes directory-scope
# compile options -- is never processed. We only need the single header.
include(FetchContent)
FetchContent_Declare(
    nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG        v1.10.0
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(nanoflann)
if(NOT nanoflann_POPULATED)
    FetchContent_Populate(nanoflann)
endif()

add_library(nanoflann INTERFACE)
target_include_directories(nanoflann INTERFACE "${nanoflann_SOURCE_DIR}/include")

# ---- Post-simulation analysis library ------------------------------------
# Pure C++17, no CUDA: operates on host-memory snapshot data already
# produced by the GPU titration kernel (see run_mcswell_gpu_titration in
# src/run_mcswell.cpp). Kept as its own target so it can be built and unit
# tested with a plain C++ compiler, without a CUDA toolchain or GPU.
add_library(mcswell_analysis STATIC
    src/analysis/kdtree.cpp
    src/analysis/frame_extraction.cpp
    src/analysis/density_grid.cpp
    src/analysis/binomial_fit.cpp
)

target_include_directories(mcswell_analysis PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mcswell_analysis PUBLIC nanoflann)
target_compile_features(mcswell_analysis PUBLIC cxx_std_17)

if (NOT MSVC)
  target_compile_options(mcswell_analysis PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

option(MCSWELL_BUILD_TESTS "Build standalone unit tests for the CUDA-free analysis library" OFF)
if(MCSWELL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

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
    mcswell_analysis
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
