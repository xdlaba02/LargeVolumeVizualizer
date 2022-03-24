#pragma once

#include "ray_simd.h"

inline simd::RayRange intersect_aabb_ray(const simd::Ray &ray, const glm::vec3 &min, const glm::vec3 &max) {
  simd::vec3 tmins = (simd::vec3(min) - ray.origin) * ray.direction_inverse;
  simd::vec3 tmaxs = (simd::vec3(max) - ray.origin) * ray.direction_inverse;

  for (uint8_t i = 0; i < 3; ++i) {
    swap(tmins[i], tmaxs[i], ray.direction_inverse[i] < 0.f);
  }

  simd::RayRange range { tmins.x, tmaxs.x };

  range.min(tmins.y > range.min) = tmins.y;
  range.max(tmaxs.y < range.max) = tmaxs.y;

  range.min(tmins.z > range.min) = tmins.z;
  range.max(tmaxs.z < range.max) = tmaxs.z;

  range.min(range.min < 0.f) = 0.f;

  return range;
}
