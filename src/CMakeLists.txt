
if(Kokkos_ENABLE_CUDA)
    set(LIB_NAME_2D "armon_cuda_2D")
elseif(Kokkos_ENABLE_HIP)
    set(LIB_NAME_2D "armon_hip_2D")
elseif(Kokkos_ENABLE_OPENMP)
    set(LIB_NAME_2D "armon_openmp_2D")
else()
    set(LIB_NAME_2D "armon_serial_2D")
endif()

set(LIB_NAME_2D ${LIB_NAME_2D} PARENT_SCOPE)
set(EXE_NAME_2D "${LIB_NAME_2D}.exe")

# 2D library & executable

add_subdirectory(kernels)

add_library(${LIB_NAME_2D}
        armon_2D.cpp armon_2D.h
        parameters.cpp parameters.h
        ranges.cpp ranges.h
        kernels/indexing.h kernels/common.h
        data.h
        io.cpp io.h
        utils.h)
target_link_libraries(${LIB_NAME_2D} PRIVATE Kokkos::kokkos armon_kernels)
target_compile_options(${LIB_NAME_2D} PRIVATE ${OPTIONS})
target_include_directories(${LIB_NAME_2D} PRIVATE kernels)

target_compile_definitions(${LIB_NAME_2D} PUBLIC
        USE_SINGLE_PRECISION=${PRECISION}
        USE_NVTX=${NVTX}
        USE_SIMD_KERNELS=${SIMD_KERNELS}
        PREFERRED_SIMD_SIZE=${PREFERRED_SIMD_SIZE})

add_executable(${EXE_NAME_2D}
        main_2D.cpp)
target_link_libraries(${EXE_NAME_2D} PRIVATE ${LIB_NAME_2D} PRIVATE Kokkos::kokkos)
