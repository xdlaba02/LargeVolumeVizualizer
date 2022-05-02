# @author Drahomír Dlabaja (xdlaba02)
# @date 2. 5. 2022
# @copyright 2022 Drahomír Dlabaja

cmake_minimum_required(VERSION 3.20)

add_subdirectory(external)

add_library(lvf INTERFACE)

target_link_libraries(lvf INTERFACE Vc glm)

target_include_directories(lvf INTERFACE interface)
