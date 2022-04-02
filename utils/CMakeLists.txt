cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenGL REQUIRED)

add_executable(random_sapling_test random_sampling.cc)
add_executable(ray_integration_test ray_integration.cc)
add_executable(mapped_coherent_test mapped_coherent.cc)
add_executable(ldexp_test ldexp.cc)
add_executable(sort_three_values_test sort_three_values.cc)

target_link_libraries(random_sapling_test Vc)
target_link_libraries(ray_integration_test Vc glm)
target_link_libraries(mapped_coherent_test Vc glm)

target_include_directories(random_sapling_test PUBLIC ../src)
target_include_directories(ray_integration_test PUBLIC ../src)
target_include_directories(mapped_coherent_test PUBLIC ../src)
