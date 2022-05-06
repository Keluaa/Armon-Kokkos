
if (Kokkos_ENABLE_CUDA)
    set(EXE_NAME "armon_cuda.exe")
elseif(Kokkos_ENABLE_OPENMP)
    set(EXE_NAME "armon_openmp.exe")
else()
    set(EXE_NAME "armon_serial.exe")
endif()

add_executable(${EXE_NAME} "armon.cpp")
target_link_libraries(${EXE_NAME} PRIVATE Kokkos::kokkos)
