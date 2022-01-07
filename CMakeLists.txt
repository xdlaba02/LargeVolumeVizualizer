cmake_minimum_required(VERSION 3.20)

project(LargeVolumeVizualizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wfatal-errors -fopenmp -g")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

add_subdirectory(external)
#add_subdirectory(data)

# ugly but easy, ok for smol project
file(GLOB SRC_FILES CONFIGURE_DEPENDS src/*.cc)

find_package(OpenGL REQUIRED)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} glm glfw OpenGL::GL Vc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
