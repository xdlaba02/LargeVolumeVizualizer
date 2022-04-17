cmake_minimum_required(VERSION 3.20)

project(LargeVolumeVizualizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wfatal-errors -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -march=native -DNDEBUG")

add_subdirectory(liblvf)
add_subdirectory(tools)
add_subdirectory(extra)
