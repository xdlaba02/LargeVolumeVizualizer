# @author Drahomír Dlabaja (xdlaba02)
# @date 11. 5. 2023
# @copyright 2023 Drahomír Dlabaja

cmake_minimum_required(VERSION 3.20)

project(LargeVolumeVizualizer)

set(CMAKE_COLOR_DIAGNOSTICS       ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	add_compile_options(-Wall -Wextra -Wpedantic -Wfatal-errors)
	add_compile_options($<$<CONFIG:RELEASE>:-fopenmp>)
	add_compile_options($<$<CONFIG:RELEASE>:-march=native>)

	add_link_options($<$<CONFIG:RELEASE>:-fopenmp>)
endif()

# Default constructors for glm > 0.9.9.0
add_compile_definitions(GLM_FORCE_CTOR_INIT)

set(liblvf_DIR ${CMAKE_CURRENT_SOURCE_DIR}/liblvf)

add_subdirectory(tools)
add_subdirectory(extra)
