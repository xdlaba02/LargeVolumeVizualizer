cmake_minimum_required(VERSION 3.20)

add_executable(ldexp_test             ldexp.cc)
add_executable(sort_three_values_test sort_three_values.cc)
add_executable(test_volume_generator  test_volume_generator.cc)
add_executable(preintegrate  preintegrate.cc)

add_executable(histrogram  histrogram.cc)
add_executable(histrogram16  histrogram16.cc)
add_executable(int16touint16  int16touint16.cc)

target_link_libraries(test_volume_generator    lvf)
target_link_libraries(histrogram    lvf)
target_link_libraries(histrogram16    lvf)
target_link_libraries(preintegrate    lvf)

add_subdirectory(experiments)
