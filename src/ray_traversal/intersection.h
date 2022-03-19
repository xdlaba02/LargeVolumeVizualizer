#pragma once

#include <glm/glm.hpp>

#include <array>

inline void intersect_aabb_ray(const glm::vec3& origin, glm::vec3 ray_direction_inverse, const glm::vec3 &min, const glm::vec3 &max, float& tmin, float& tmax) {
  std::array<glm::vec3, 2> ts { (min - origin) * ray_direction_inverse,
                                (max - origin) * ray_direction_inverse };

  auto direction_negative = glm::lessThan(ray_direction_inverse, glm::vec3(0.f));

  tmin = std::max(std::max(ts[direction_negative.x].x,  ts[direction_negative.y].y),  ts[direction_negative.z].z);
  tmax = std::min(std::min(ts[!direction_negative.x].x, ts[!direction_negative.y].y), ts[!direction_negative.z].z);

  tmin = std::max(tmin, 0.f); // solves when ray hits box from behind
}
