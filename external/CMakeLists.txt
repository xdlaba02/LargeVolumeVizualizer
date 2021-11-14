cmake_minimum_required(VERSION 3.20)

add_subdirectory(glm)
add_subdirectory(glfw)

unset(BUILD_TESTING)
add_subdirectory(Vc)
