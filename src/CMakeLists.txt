
if(Kokkos_ENABLE_CUDA)
    set(EXE_NAME "armon_cuda.exe")
elseif(Kokkos_ENABLE_HIP)
    set(EXE_NAME "armon_hip.exe")
elseif(Kokkos_ENABLE_OPENMP)
    set(EXE_NAME "armon_openmp.exe")
else()
    set(EXE_NAME "armon_serial.exe")
endif()

add_executable(${EXE_NAME} "armon.cpp")
target_link_libraries(${EXE_NAME} PRIVATE Kokkos::kokkos)

if(USE_SINGLE_PRECISION)
    target_compile_definitions(${EXE_NAME} PRIVATE USE_SINGLE_PRECISION=1)
else()
    target_compile_definitions(${EXE_NAME} PRIVATE USE_SINGLE_PRECISION=0)
endif()
