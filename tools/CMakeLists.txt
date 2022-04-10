cmake_minimum_required(VERSION 3.20)

add_subdirectory(external)

find_package(OpenGL REQUIRED)

add_executable(vizualize     vizualize.cc)
add_executable(vizualize_raw vizualize_raw.cc)
add_executable(process       process.cc)

target_link_libraries(vizualize     imgui glfw OpenGL::GL lvf)
target_link_libraries(vizualize_raw imgui glfw OpenGL::GL lvf)
target_link_libraries(process       lvf)
