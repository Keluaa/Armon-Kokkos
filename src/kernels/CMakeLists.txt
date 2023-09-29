
add_library(armon_kernels SHARED
        common.h
        kernels.h
        parallel_kernels.h
        limiters.h
        test_cases.h
        indexing.cpp indexing.h
        acoustic.cpp
        fluid_eos.cpp
        euler_projection.cpp
        init_test.cpp
        halo_exchange.cpp
        reductions.cpp)
target_link_libraries(armon_kernels PRIVATE Kokkos::kokkos)
target_compile_options(armon_kernels PRIVATE
        $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>)
target_compile_definitions(armon_kernels PRIVATE
        PREFERRED_SIMD_SIZE=${PREFERRED_SIMD_SIZE})

if (USE_SINGLE_PRECISION)
    target_compile_definitions(armon_kernels PRIVATE USE_SINGLE_PRECISION)
endif()

if (USE_SIMD_KERNELS)
    target_compile_definitions(armon_kernels PRIVATE USE_SIMD_KERNELS)
endif()
