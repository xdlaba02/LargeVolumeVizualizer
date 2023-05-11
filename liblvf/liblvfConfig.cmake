# @author Drahomír Dlabaja (xdlaba02)
# @date 11. 5. 2023
# @copyright 2023 Drahomír Dlabaja

if (NOT TARGET liblvf::liblvf)
  string(REGEX REPLACE ${CMAKE_SOURCE_DIR} "" RELATIVE_ENGINE_DIR ${CMAKE_CURRENT_LIST_DIR})
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR} ${CMAKE_BINARY_DIR}${RELATIVE_ENGINE_DIR})
  add_library(liblvf::liblvf ALIAS liblvf)
endif()
