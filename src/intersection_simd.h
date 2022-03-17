#pragma once

#include "simd.h"

#include <glm/glm.hpp>

inline void intersect_aabb_ray(const glm::vec<3, simd::float_v> &origin, const glm::vec<3, simd::float_v> &ray_direction_inverse, const glm::vec3 &min, const glm::vec3 &max, simd::float_v& tmin, simd::float_v& tmax) {
  glm::vec<3, simd::float_v> tmins = (glm::vec<3, simd::float_v>(min) - origin) * ray_direction_inverse;
  glm::vec<3, simd::float_v> tmaxs = (glm::vec<3, simd::float_v>(max) - origin) * ray_direction_inverse;

  for (uint32_t i = 0; i < 3; ++i) {
    simd::swap(tmins[i], tmaxs[i], ray_direction_inverse[i] < 0.f);
  }

  tmin = tmins[0];
  tmax = tmaxs[0];

  for (uint32_t i = 1; i < 3; ++i) {
    tmin(tmins[i] > tmin) = tmins[i];
    tmax(tmaxs[i] < tmax) = tmaxs[i];
  }

  tmin(tmin < 0.f) = 0.f;
}
