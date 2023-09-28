
option(FORCE_AVX2   "Force the use of 32 bytes teams, as if AVX2 extensions were available"   OFF)
option(FORCE_AVX512 "Force the use of 64 bytes teams, as if AVX512 extensions were available" OFF)

execute_process(COMMAND $ENV{SHELL} -c "lscpu | grep avx2" OUTPUT_QUIET RESULT_VARIABLE IS_AVX2_DETECTED)
execute_process(COMMAND $ENV{SHELL} -c "lscpu | grep avx512" OUTPUT_QUIET RESULT_VARIABLE IS_AVX512_DETECTED)

if (IS_AVX2_DETECTED EQUAL 0)
    set(IS_AVX2_DETECTED ON)
else ()
    set(IS_AVX2_DETECTED OFF)
endif ()

if (IS_AVX512_DETECTED EQUAL 0)
    set(IS_AVX512_DETECTED ON)
else ()
    set(IS_AVX512_DETECTED OFF)
endif ()

if (FORCE_AVX2 OR IS_AVX2_DETECTED)
    set(HAS_AVX2 ON)
else ()
    set(HAS_AVX2 OFF)
endif ()

if (FORCE_AVX512 OR IS_AVX512_DETECTED)
    set(HAS_AVX512 ON)
else ()
    set(HAS_AVX512 OFF)
endif ()

if (HAS_AVX512)
    set(PREFERRED_SIMD_SIZE 64)
elseif (HAS_AVX2)
    set(PREFERRED_SIMD_SIZE 32)
else ()
    set(PREFERRED_SIMD_SIZE 8)
endif ()

message("PREFERRED_SIMD_SIZE: ${PREFERRED_SIMD_SIZE}")
