
add_library(armon_kernels SHARED
        common.h
        kernels.h
        parallel_kernels.h
        limiters.h
        test_cases.h
        indexing.cpp indexing.h
        utils.cpp utils.h
        acoustic.cpp
        fluid_eos.cpp
        euler_projection.cpp
        init_test.cpp
        halo_exchange.cpp
        reductions.cpp)
target_link_libraries(armon_kernels PRIVATE Kokkos::kokkos)
target_compile_options(armon_kernels PRIVATE
        $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>
        $<IF:$<STREQUAL:$<CXX_COMPILER_ID>,Clang>,-Wno-gnu-zero-variadic-macro-arguments,>
        ${OPTIONS})
target_compile_definitions(armon_kernels PRIVATE
        USE_SIMD_KERNELS=${SIMD_KERNELS}
        USE_2D_ITER=${2D_ITER}
        USE_MD_ITER=${MD_ITER}
        BALANCE_MD_ITER=${BALANCE_MD_ITER}
        PREFERRED_SIMD_SIZE=${PREFERRED_SIMD_SIZE}
        USE_SINGLE_PRECISION=${PRECISION}
        CHECK_VIEW_ORDER
        TRY_ALL_CALLS)
