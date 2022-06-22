cmake_minimum_required(VERSION 3.16)

project(Armon-Kokkos LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos Modules
set(Kokkos_ENABLE_HWLOC ON CACHE BOOL "Kokkos_ENABLE_HWLOC" FORCE)

if(Kokkos_ENABLE_HIP)
    set(Kokkos_ARCH_VEGA908 ON CACHE BOOL "Kokkos_ARCH_VEGA908" FORCE) # Mi100 GPU
endif()

if(Kokkos_ENABLE_CUDA)
    execute_process(COMMAND hostname RESULT_VARIABLE HOSTNAME)
    if(HOSTNAME MATCHES "^login")
        # The GPU architecture will not be detected automatically, set it to AMPERE 80 by default
        set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "Kokkos_ARCH_AMPERE80" FORCE)  # A100 GPU
    endif()

    set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Kokkos_ENABLE_CUDA_CONSTEXPR" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Kokkos_ENABLE_CUDA_LAMBDA" FORCE)
    set(Kokkos_ENABLE_LIBRT OFF CACHE BOOL "Kokkos_ENABLE_LIBRT" FORCE)
    set(Kokkos_ENABLE_LIBDL OFF CACHE BOOL "Kokkos_ENABLE_LIBDL" FORCE)
endif()

if(Kokkos_ENABLE_OPENMP)
    # Some functions used in the Kokkos OpenMP backend are marked as deprecated, this is to remove the warnings about them
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

#if(Kokkos_ENABLE_SIMD)
#    set(Kokkos_ENABLE_PRAGMA_SIMD ON)
#    set(Kokkos_ENABLE_PRAGMA_IVDEP ON)
#    set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON)
#    set(KOKKOS_ENABLE_PRAGMA_UNROLL ON)
#endif()

# Project Modules
add_subdirectory(modules/kokkos)

# Project Options
option(USE_SINGLE_PRECISION "Use float instead of double" OFF)

# Project Sources
include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_subdirectory(src)
