cmake_minimum_required(VERSION 3.16)

project(Armon-Kokkos LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Kokkos Modules
set(Kokkos_ENABLE_HWLOC ON CACHE BOOL "Kokkos_ENABLE_HWLOC" FORCE)
set(Kokkos_ENABLE_LIBRT OFF CACHE BOOL "Kokkos_ENABLE_LIBRT" FORCE)
set(Kokkos_ENABLE_LIBDL OFF CACHE BOOL "Kokkos_ENABLE_LIBDL" FORCE)

if(Kokkos_ENABLE_HIP)
    set(Kokkos_ARCH_VEGA908 ON) # Mi100 GPU
    #set(Kokkos_ARCH_VEGA90A ON) # Mi250 GPU

    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS ON CACHE BOOL "Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS" FORCE)
    endif()
endif()

if(Kokkos_ENABLE_CUDA)
    execute_process(COMMAND hostname RESULT_VARIABLE HOSTNAME)
    if(HOSTNAME MATCHES "^login")
        # The GPU architecture will not be detected automatically, set it to AMPERE 80 by default
        set(Kokkos_ARCH_AMPERE80 ON)  # A100 GPU
    endif()

    set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Kokkos_ENABLE_CUDA_CONSTEXPR" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Kokkos_ENABLE_CUDA_LAMBDA" FORCE)
endif()

if(Kokkos_ENABLE_OPENMP)
    # Some functions used in the Kokkos OpenMP backend are marked as deprecated, this is to remove the warnings about them
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

    set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION" FORCE)
endif()


if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT Kokkos_ENABLE_OPENMP AND NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP)
    # Enable all debug options when running in serial mode only
    set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Kokkos_ENABLE_DEBUG" FORCE)
    set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "Kokkos_ENABLE_DEBUG_BOUNDS_CHECK" FORCE)
    # Disable fast math in order to catch NANs more easily
    set(OPTIONS "-fno-fast-math")
else()
    set(Kokkos_ENABLE_DEBUG OFF CACHE BOOL "Kokkos_ENABLE_DEBUG" FORCE)
    set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK OFF CACHE BOOL "Kokkos_ENABLE_DEBUG_BOUNDS_CHECK" FORCE)

    # Optimize for the native architecture when using only the CPU
    if(Kokkos_ENABLE_OPENMP AND NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP)
        set(Kokkos_ARCH_NATIVE ON)
    else()
        set(Kokkos_ARCH_NATIVE OFF)
    endif()

    set(OPTIONS "-ffast-math")
endif()


# Project Options
option(USE_SINGLE_PRECISION "Use float instead of double" OFF)

if(USE_SINGLE_PRECISION)
    set(PRECISION 1)
else()
    set(PRECISION 0)
endif()

# Project Modules
add_subdirectory(modules/kokkos)

# Project Sources
include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_subdirectory(src)
add_subdirectory(tests)
