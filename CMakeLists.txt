cmake_minimum_required(VERSION 3.16)

project(Armon-Kokkos LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -funroll-loops")

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

    set(OPTIONS "-ffast-math")
endif()

# Project Options
option(USE_SINGLE_PRECISION "Use float instead of double" OFF)
option(USE_NVTX "Enable NVTX tracing" OFF)
option(USE_SIMD_KERNELS "Use team vector loops instead of normal loops" OFF)
option(USE_2D_ITER "Use 2 Kokkos::RangePolicy instead of 1 Kokkos::RangePolicy" OFF)
option(USE_MD_ITER "Use Kokkos::MDRangePolicy instead of Kokkos::RangePolicy" OFF)
option(BALANCE_MD_ITER "Use a mediocre load balancing algorithm for OpenMP threads when USE_MD_ITER=ON" OFF)
option(CHECK_VIEW_ORDER "Check if Kokkos::View parameters in kernels are in the correct order" OFF)
option(TRY_ALL_CALLS "Surround all exported kernel functions with a try-catch block" OFF)

if(USE_SIMD_KERNELS AND (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP))
    message(WARNING "SIMD kernels are extremely slow on GPU")
endif()

if((USE_SIMD_KERNELS AND USE_MD_ITER) OR (USE_SIMD_KERNELS AND USE_2D_ITER) OR (USE_2D_ITER AND USE_MD_ITER))
    message(FATAL_ERROR "Only one of USE_SIMD_KERNELS, USE_2D_ITER and USE_MD_ITER can be ON")
endif()

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

if(USE_2D_ITER)
    set(2D_ITER 1)
else()
    set(2D_ITER 0)
endif()

if(USE_MD_ITER)
    set(MD_ITER 1)
else()
    set(MD_ITER 0)
endif()

if (BALANCE_MD_ITER)
    set(BALANCE_MD_ITER 1)
else()
    set(BALANCE_MD_ITER 0)
endif()


include(detect_simd_size.cmake)

# Important: required to build shared libraries using Kokkos
set(BUILD_SHARED_LIBS ON)

# Project Modules
if (NOT DEFINED Kokkos_ROOT)
    add_subdirectory("${PROJECT_SOURCE_DIR}/modules/kokkos" lib)  # Use Kokkos in-tree
else()
    find_package(Kokkos REQUIRED)  # Use the provided Kokkos installation
endif()

# Project Sources
include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_subdirectory(src)
add_subdirectory(tests)
