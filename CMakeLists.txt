cmake_minimum_required(VERSION 3.20)

project(LargeVolumeVizualizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wfatal-errors -fopenmp -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

add_subdirectory(external)

# include libraries headers as system to supress warnings
function(target_link_libraries_system target)
  set(libs ${ARGN})
  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${target} SYSTEM PRIVATE ${lib_include_dirs})
    target_link_libraries(${target} ${lib})
  endforeach(lib)
endfunction(target_link_libraries_system)


# ugly but easy, ok for smol project
file(GLOB SRC_FILES CONFIGURE_DEPENDS src/*.cc)

find_package(OpenGL REQUIRED)

add_executable(vizualize src/vizualize.cc)
add_executable(vizualize_raw src/vizualize_raw.cc)
add_executable(process src/process.cc)
add_executable(test_volume_generator test/test_volume_generator.cc)

target_link_libraries_system(vizualize glm glfw OpenGL::GL Vc)
target_link_libraries_system(vizualize_raw glm glfw OpenGL::GL Vc)

target_include_directories(vizualize PUBLIC src)
target_include_directories(vizualize_raw PUBLIC src)
target_include_directories(process PUBLIC src)
target_include_directories(test_volume_generator PUBLIC src)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
