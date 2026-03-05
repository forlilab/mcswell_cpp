cmake_minimum_required(VERSION 3.18)
project(mcswell_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build SASS for common GPU architectures + PTX for forward compatibility.
# Override at configure time with -DCMAKE_CUDA_ARCHITECTURES="80;90" etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86;89;90")
endif()

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
set(WATER_MODEL "TIP3P" CACHE STRING "Water model: TIP3P, TIP3PFP or TIP4P")
set_property(CACHE WATER_MODEL PROPERTY STRINGS TIP3P TIP3PFP)

if(WATER_MODEL STREQUAL "TIP3P")
  target_compile_definitions(_mcswell_cpp PRIVATE TIP3P)
  target_compile_definitions(_mcswell_cpp PRIVATE WATER_MODEL_NAME="${WATER_MODEL}")
elseif(WATER_MODEL STREQUAL "TIP3PFP")
  target_compile_definitions(_mcswell_cpp PRIVATE TIP3PFP)
  target_compile_definitions(_mcswell_cpp PRIVATE WATER_MODEL_NAME="${WATER_MODEL}")
elseif(WATER_MODEL STREQUAL "TIP4P")
  target_compile_definitions(_mcswell_cpp PRIVATE TIP4P)
  target_compile_definitions(_mcswell_cpp PRIVATE WATER_MODEL_NAME="${WATER_MODEL}")
else()
  message(FATAL_ERROR "Invalid WATER_MODEL='${WATER_MODEL}'. Use TIP3P, TIP3PFP or TIP4P.")
endif()

install(TARGETS _mcswell_cpp
        LIBRARY DESTINATION mcswell_cpp
        RUNTIME DESTINATION mcswell_cpp
        ARCHIVE DESTINATION mcswell_cpp.)
