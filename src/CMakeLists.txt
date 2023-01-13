
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

add_library(${LIB_NAME_2D}
        armon_2D.cpp
        parameters.cpp parameters.h
        indexing.cpp indexing.h
        data.h
        armon_2D.h
        kernel_info.cpp kernel_info.h
        io.cpp io.h
        utils.h)
target_link_libraries(${LIB_NAME_2D} PRIVATE Kokkos::kokkos)
target_compile_options(${LIB_NAME_2D} PRIVATE ${OPTIONS})

add_executable(${EXE_NAME_2D}
        main_2D.cpp)
target_link_libraries(${EXE_NAME_2D} PRIVATE ${LIB_NAME_2D} PRIVATE Kokkos::kokkos)
target_compile_definitions(${LIB_NAME_2D} PRIVATE USE_SINGLE_PRECISION=${PRECISION})
