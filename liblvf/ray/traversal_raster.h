/**
* @file traversal_raster.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Experimental function that traverses raster defined by integer numbers grid along ray on specified interval.
* This traversal is broken but in theory, it isn't. One must imagine Sisyphus happy.
*/

#pragma once

#include "ray.h"

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename F>
inline void ray_raster_traversal(const Ray &ray, const RayRange &range, const F& callback) {
  glm::vec3 pos = ray.origin + ray.direction * range.min;

  // this is wrong for negative directions because block[i] might be == to size[i], otherwise it should work
  glm::vec<3, uint32_t> block = glm::vec<3, uint32_t>(pos);

  glm::vec3 next_max = (glm::vec3(block) - ray.origin) * ray.direction_inverse;

  glm::vec3 delta = glm::abs(ray.direction_inverse);

  glm::vec<3, int8_t> step;

  for (uint8_t i = 0; i < 3; i++) {
    if (ray.direction[i] < 0.f) {
      step[i] = -1;
    }
    else {
      step[i] = 1;
      next_max[i] += ray.direction_inverse[i];
    }
  }

  auto idx_of_min = [](const glm::vec3 &vec) {
    return std::array<uint8_t, 8> { 2, 1, 2, 1, 2, 2, 0, 0 }[((vec[0] < vec[1]) << 2) | ((vec[0] < vec[2]) << 1) | ((vec[1] < vec[2]) << 0)];
  };

  uint8_t axis = idx_of_min(next_max);

  RayRange child_range { range.min, next_max[axis] };

  while (child_range.max < range.max && callback(child_range, block)) {
    child_range.min = child_range.max;

    next_max[axis] += delta[axis];
    block[axis] += step[axis];

    axis = idx_of_min(next_max);
    child_range.max = next_max[axis];
  }
}
