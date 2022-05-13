# @author Drahomír Dlabaja (xdlaba02)
# @date 2. 5. 2022
# @copyright 2022 Drahomír Dlabaja

cmake_minimum_required(VERSION 3.20)

add_library(glm INTERFACE)
target_include_directories(glm SYSTEM INTERFACE glm)

add_library(Vc STATIC Vc/src/const.cpp)
target_include_directories(Vc SYSTEM PUBLIC Vc)
