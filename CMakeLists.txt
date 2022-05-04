cmake_minimum_required(VERSION 3.16)

project(Armon-Kokkos LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)


add_subdirectory(modules/kokkos)

include_directories(${Kokkos_INCLUDE_DIRS_RET})

add_subdirectory(src)