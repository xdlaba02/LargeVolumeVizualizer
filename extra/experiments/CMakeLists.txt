cmake_minimum_required(VERSION 3.20)

add_executable(raw raw.cc)
add_executable(blocks blocks.cc)
add_executable(tree tree.cc)
add_executable(tree_reduced tree_reduced.cc)
add_executable(tree_real tree_real.cc)

target_link_libraries(raw lvf)
target_link_libraries(blocks lvf)
target_link_libraries(tree lvf)
target_link_libraries(tree_reduced lvf)
target_link_libraries(tree_real lvf)
