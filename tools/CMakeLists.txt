# @author Drahomír Dlabaja (xdlaba02)
# @date 11. 5. 2023
# @copyright 2023 Drahomír Dlabaja

cmake_minimum_required(VERSION 3.20)

find_package(glfw3  REQUIRED)
find_package(imgui  REQUIRED)
find_package(OpenGL REQUIRED)
find_package(liblvf REQUIRED)


add_library(imgui_glfw_opengl2)

target_sources(imgui_glfw_opengl2 
    PRIVATE
        ${CMAKE_BINARY_DIR}/bindings/imgui_impl_glfw.cpp 
        ${CMAKE_BINARY_DIR}/bindings/imgui_impl_glfw.h 
        ${CMAKE_BINARY_DIR}/bindings/imgui_impl_opengl2.cpp 
        ${CMAKE_BINARY_DIR}/bindings/imgui_impl_opengl2.h
)

target_link_libraries(imgui_glfw_opengl2 PUBLIC imgui::imgui glfw OpenGL::GL)

target_include_directories(imgui_glfw_opengl2 INTERFACE ${CMAKE_BINARY_DIR})


add_executable(vizualize     vizualize.cc)
add_executable(vizualize_raw vizualize_raw.cc)
add_executable(process       process.cc)

target_link_libraries(vizualize     PRIVATE liblvf::liblvf imgui_glfw_opengl2 glfw OpenGL::GL)
target_link_libraries(vizualize_raw PRIVATE liblvf::liblvf imgui_glfw_opengl2 glfw OpenGL::GL)
target_link_libraries(process       PRIVATE liblvf::liblvf)
