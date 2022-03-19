#pragma once

#include "ray.h"

#include <utils/utils.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

// Interactive isosurface ray tracing of large octree volumes
// https://www.researchgate.net/publication/310054812_Interactive_isosurface_ray_tracing_of_large_octree_volumes
template <typename F>
void ray_octree_traversal(const Ray &ray, const RayRange &range, glm::vec3 cell, uint32_t layer, const F &callback) {
  if (callback(range, cell, layer)) {
    // TODO precompute?
    float child_size = approx_exp2(-layer - 1);

    glm::vec3 center = cell + child_size;

    glm::vec3 tcenter = (center - ray.origin) * ray.direction_inverse;

    // fast sort axis by tcenter

    bool gt01 = tcenter[0] > tcenter[1];
    bool gt02 = tcenter[0] > tcenter[2];
    bool gt12 = tcenter[1] > tcenter[2];

    std::array<uint8_t, 3> axis;

    axis[ gt01 +  gt02] = 0;
    axis[!gt01 +  gt12] = 1;
    axis[!gt02 + !gt12] = 2;

    // TODO precompute?
    for (uint8_t i = 0; i < 3; i++) {
      if (ray.direction[i] < 0.f) {
        std::swap(cell[i], center[i]);
      }
    }

    float tmin = range.min;

    for (uint8_t i = 0; i < 3; i++) {
      float tmax = std::min(tcenter[axis[i]], range.max);

      if (tmin < tmax) {
        ray_octree_traversal(ray, { tmin, tmax }, cell, layer + 1, callback);
        tmin = tmax;
      }

      // outside the condition because the way of swapping child cells by direction
      cell[axis[i]] = center[axis[i]];
    }

    if (tmin < range.max) {
      ray_octree_traversal(ray, { tmin, range.max }, cell, layer + 1, callback);
    }
  }
}
