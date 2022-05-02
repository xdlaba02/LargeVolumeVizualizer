# @author Drahomír Dlabaja (xdlaba02)
# @date 2. 5. 2022
# @copyright 2022 Drahomír Dlabaja

cmake_minimum_required(VERSION 3.20)

project(LargeVolumeVizualizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wfatal-errors")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -march=native -DNDEBUG -s")

add_subdirectory(liblvf)
add_subdirectory(tools)
add_subdirectory(extra)
