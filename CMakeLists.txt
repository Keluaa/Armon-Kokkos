cmake_minimum_required(VERSION 3.16)

project(Armon-Kokkos LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos Modules
set(Kokkos_ENABLE_HWLOC ON)

if(Kokkos_ENABLE_HIP)
    set(Kokkos_ARCH_VEGA908) # Mi100 GPU
endif()

if(Kokkos_ENABLE_CUDA)
    set(Kokkos_ENABLE_CUDA_CONSTEXPR ON)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON)
    set(Kokkos_ENABLE_LIBRT OFF)
    set(Kokkos_ENABLE_LIBDL OFF)
endif()

if(Kokkos_ENABLE_SIMD)
    set(Kokkos_ENABLE_PRAGMA_IVDEP ON)
#    set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON)
endif()

# Project Modules
add_subdirectory(modules/kokkos)

# Project Sources
include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_subdirectory(src)
