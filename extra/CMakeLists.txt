cmake_minimum_required(VERSION 3.20)

add_executable(blocked_integration_test blocked_integration.cc)
add_executable(naive_integration_test   naive_integration.cc)

add_executable(ldexp_test             ldexp.cc)
add_executable(sort_three_values_test sort_three_values.cc)
add_executable(test_volume_generator  test_volume_generator.cc)

target_link_libraries(blocked_integration_test Vc glm lvf)
target_link_libraries(naive_integration_test   Vc glm lvf)
target_link_libraries(test_volume_generator    lvf)
