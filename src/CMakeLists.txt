
if (Kokkos_ENABLE_CUDA)
    set(EXE_NAME "armon_cuda")
elseif(Kokkos_ENABLE_OPENMP)
    set(EXE_NAME "armon_openmp")
else()
    set(EXE_NAME "armon_serial")
endif()

add_executable(${EXE_NAME} "armon.cpp")
target_link_libraries(${EXE_NAME} PRIVATE Kokkos::kokkos)

target_compile_definitions(${EXE_NAME} PRIVATE
        USE_GPU=${Kokkos_ENABLE_CUDA}
        USE_THREADING=${Kokkos_ENABLE_OPENMP})
