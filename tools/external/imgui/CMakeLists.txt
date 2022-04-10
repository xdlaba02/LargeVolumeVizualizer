cmake_minimum_required(VERSION 3.20)
add_library(imgui imgui/imgui.cpp
                  imgui/imgui_draw.cpp
                  imgui/imgui_tables.cpp
                  imgui/imgui_widgets.cpp
                  imgui/backends/imgui_impl_glfw.cpp
                  imgui/backends/imgui_impl_opengl2.cpp)

target_include_directories(imgui PUBLIC imgui/ INTERFACE .)
