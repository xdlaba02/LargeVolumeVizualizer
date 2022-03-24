cmake_minimum_required(VERSION 3.20)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

add_library(glm INTERFACE)
target_include_directories(glm SYSTEM INTERFACE glm)

add_library(Vc STATIC Vc/src/const.cpp)
target_include_directories(Vc SYSTEM PUBLIC Vc)

add_subdirectory(glfw)

add_subdirectory(imgui)
