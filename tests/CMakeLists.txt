
set(TESTS_NAME_2D "${LIB_NAME_2D}_tests.exe")

add_executable(${TESTS_NAME_2D}
        armon_2D_tests.cpp
        reference.cpp)
target_link_libraries(${TESTS_NAME_2D} PUBLIC ${LIB_NAME_2D} PRIVATE Kokkos::kokkos)
target_include_directories(${TESTS_NAME_2D} PUBLIC ../src)
target_compile_definitions(${TESTS_NAME_2D}
        PRIVATE USE_SINGLE_PRECISION=${PRECISION}
        PRIVATE PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")
