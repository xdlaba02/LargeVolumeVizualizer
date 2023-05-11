# @author Drahomír Dlabaja (xdlaba02)
# @date 11. 5. 2023
# @copyright 2023 Drahomír Dlabaja

cmake_minimum_required(VERSION 3.20)

add_executable(raw raw.cc)
add_executable(blocks blocks.cc)
add_executable(tree tree.cc)
add_executable(tree_real tree_real.cc)

target_link_libraries(raw       liblvf::liblvf)
target_link_libraries(blocks    liblvf::liblvf)
target_link_libraries(tree      liblvf::liblvf)
target_link_libraries(tree_real liblvf::liblvf)
