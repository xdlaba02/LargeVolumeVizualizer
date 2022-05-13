# @author Drahomír Dlabaja (xdlaba02)
# @date 2. 5. 2022
# @copyright 2022 Drahomír Dlabaja

cmake_minimum_required(VERSION 3.17)

add_library(imgui imgui/imgui.cpp
                  imgui/imgui_draw.cpp
                  imgui/imgui_tables.cpp
                  imgui/imgui_widgets.cpp
                  imgui/backends/imgui_impl_glfw.cpp
                  imgui/backends/imgui_impl_opengl2.cpp)

target_include_directories(imgui PUBLIC imgui/ INTERFACE .)
