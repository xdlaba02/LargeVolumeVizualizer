# @author Drahomír Dlabaja (xdlaba02)
# @date 11. 5. 2023
# @copyright 2023 Drahomír Dlabaja

cmake_minimum_required(VERSION 3.20)

find_package(Vc REQUIRED)
find_package(glm REQUIRED)

add_library(liblvf INTERFACE)

target_link_libraries(liblvf INTERFACE Vc::Vc glm::glm)

target_include_directories(liblvf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(liblvf INTERFACE cxx_std_20)