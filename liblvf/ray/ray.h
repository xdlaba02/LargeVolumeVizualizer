/**
* @file ray.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Data structures for ray operations.
*/

#pragma once

#include <glm/glm.hpp>

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
  glm::vec3 direction_inverse;
};

struct RayRange {
  float min;
  float max;
};
