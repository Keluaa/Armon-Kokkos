cmake_minimum_required(VERSION 3.16)

project(Armon-Kokkos LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -funroll-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -funroll-loops")

# Kokkos Modules
set(Kokkos_ENABLE_HWLOC OFF CACHE BOOL "Kokkos_ENABLE_HWLOC" FORCE)
set(Kokkos_ENABLE_LIBRT OFF CACHE BOOL "Kokkos_ENABLE_LIBRT" FORCE)
set(Kokkos_ENABLE_LIBDL ON CACHE BOOL "Kokkos_ENABLE_LIBDL" FORCE)

if(Kokkos_ENABLE_HIP)
    execute_process(COMMAND hostname RESULT_VARIABLE HOSTNAME)
    if(HOSTNAME MATCHES "^login")
        set(Kokkos_ARCH_VEGA908 ON)  # MI100
        # set(Kokkos_ARCH_VEGA90A ON)  # MI250
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
    set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION" FORCE)
endif()


if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT Kokkos_ENABLE_OPENMP AND NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP)
    # Enable all debug options when running in serial mode only
    set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Kokkos_ENABLE_DEBUG" FORCE)
    set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "Kokkos_ENABLE_DEBUG_BOUNDS_CHECK" FORCE)
    set(Kokkos_ARCH_NATIVE OFF CACHE BOOL "Kokkos_ARCH_NATIVE" FORCE)
    # Disable fast math in order to catch NANs more easily
    set(OPTIONS "-fno-fast-math")
else()
    set(Kokkos_ENABLE_DEBUG OFF CACHE BOOL "Kokkos_ENABLE_DEBUG" FORCE)
    set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK OFF CACHE BOOL "Kokkos_ENABLE_DEBUG_BOUNDS_CHECK" FORCE)

    # Optimize for the native architecture when using only the CPU
    set(Kokkos_ARCH_NATIVE ON CACHE BOOL "Kokkos_ARCH_NATIVE" FORCE)

    set(OPTIONS "-ffast-math")
endif()


# Project Options
option(USE_SINGLE_PRECISION "Use float instead of double" OFF)
option(USE_NVTX "Enable NVTX tracing" OFF)
option(USE_SIMD_KERNELS "Use team vector loops instead of normal loops" OFF)

if(USE_SINGLE_PRECISION)
    set(PRECISION 1)
else()
    set(PRECISION 0)
endif()

if(USE_NVTX)
    set(NVTX 1)
else()
    set(NVTX 0)
endif()

if(USE_SIMD_KERNELS)
    set(SIMD_KERNELS 1)
else()
    set(SIMD_KERNELS 0)
endif()


include(detect_simd_size.cmake)


if (NOT DEFINED ARMON_KOKKOS_ROOT)
    set(ARMON_KOKKOS_ROOT "${PROJECT_SOURCE_DIR}/modules/kokkos")
endif()


# Project Modules
add_subdirectory(${ARMON_KOKKOS_ROOT} lib)

# Project Sources
include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_subdirectory(src)
add_subdirectory(tests)
