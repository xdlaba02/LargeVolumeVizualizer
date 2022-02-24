#pragma once

#include "simd.h"

#include <glm/glm.hpp>

inline void intersect_aabb_rays_single_origin(const glm::vec3& origin, glm::vec<3, simd::float_v> ray_direction_inverses, const glm::vec3 &min, const glm::vec3 &max, simd::float_v& tmins, simd::float_v& tmaxs) {
  tmins = -std::numeric_limits<float>::infinity();
  tmaxs = std::numeric_limits<float>::infinity();

  for (uint32_t i = 0; i < 3; ++i) {
    simd::float_v t0 = (min[i] - origin[i]) * ray_direction_inverses[i];
    simd::float_v t1 = (max[i] - origin[i]) * ray_direction_inverses[i];

    simd::swap(t0, t1, ray_direction_inverses[i] < 0.f);

    tmins(t0 > tmins) = t0;
    tmaxs(t1 < tmaxs) = t1;
  }
}

inline void intersect_aabb_ray(const glm::vec3& origin, glm::vec3 ray_direction_inverse, const glm::vec3 &min, const glm::vec3 &max, float& tmin, float& tmax) {
  tmin = -std::numeric_limits<float>::infinity();
  tmax = std::numeric_limits<float>::infinity();

  for (uint32_t i = 0; i < 3; ++i) {
    float t0 = (min[i] - origin[i]) * ray_direction_inverse[i];
    float t1 = (max[i] - origin[i]) * ray_direction_inverse[i];

    if (ray_direction_inverse[i] < 0.f) {
      tmin = std::max(tmin, t1);
      tmax = std::min(tmax, t0);
    }
    else {
      tmin = std::max(tmin, t0);
      tmax = std::min(tmax, t1);
    }
  }
}
