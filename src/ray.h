#pragma once

#include <glm/glm.hpp>

struct Ray {
  Ray(const glm::vec3 &origin, const glm::vec3 &direction)
      : origin(origin)
      , direction(direction)
      , direction_inverse(1.f / direction) {}
      
  const glm::vec3 origin;
  const glm::vec3 direction;
  const glm::vec3 direction_inverse;
};
