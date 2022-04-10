#pragma once

#include "ray.h"

#include <glm/glm.hpp>

#include <array>

inline RayRange intersect_aabb_ray(const Ray &ray, const glm::vec3 &min, const glm::vec3 &max) {
  std::array<glm::vec3, 2> ts { (min - ray.origin) * ray.direction_inverse,
                                (max - ray.origin) * ray.direction_inverse };

  auto direction_negative = glm::lessThan(ray.direction_inverse, glm::vec3(0.f));

  return {
    std::max(0.f, std::max(std::max(ts[direction_negative.x].x,  ts[direction_negative.y].y),  ts[direction_negative.z].z)),
                  std::min(std::min(ts[!direction_negative.x].x, ts[!direction_negative.y].y), ts[!direction_negative.z].z)
  };
}
