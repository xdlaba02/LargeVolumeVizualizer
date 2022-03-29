#pragma once

#include <glm/glm.hpp>

inline float distance_aabb_point(const glm::vec3 &point, const glm::vec3 &min, const glm::vec3 &max) {
  glm::vec3 vec = glm::max(min - point, glm::max(glm::vec3(0.f), point - max));
  return glm::length(vec);
}

inline float max_distance_aabb_point(const glm::vec3 &point, const glm::vec3 &min, const glm::vec3 &max) {
  return glm::length(glm::max(max - point, point - min));
}
