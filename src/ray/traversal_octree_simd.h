#pragma once

#include "ray_simd.h"

#include <utils/utils.h>
#include <utils/simd.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>
#include <concepts>

template <typename F>
concept RayOctreeTraversalSimdCallback = std::invocable<F, const simd::RayRange &, const simd::vec3 &, uint32_t, simd::float_m>;

template <RayOctreeTraversalSimdCallback F>
void ray_octree_traversal(const simd::Ray &ray, const simd::RayRange &range, simd::vec3 cell, uint32_t layer, simd::float_m mask, const F &callback) {
  callback(range, cell, layer, mask);

  if (mask.isEmpty()) {
    return;
  }

  // TODO precompute?
  float child_size = exp2i(-layer - 1);

  simd::vec3 center;
  simd::vec3 tcenter;
  std::array<simd::uint32_v, 3> axis;

  center = cell + simd::float_v(child_size);

  tcenter = (center - ray.origin) * ray.direction_inverse;

  // fast sort axis by tcenter
  simd::uint32_v gt01 = 0;
  simd::uint32_v gt02 = 0;
  simd::uint32_v gt12 = 0;

  gt01(tcenter[0] > tcenter[1]) = 1;
  gt02(tcenter[0] > tcenter[2]) = 1;
  gt12(tcenter[1] > tcenter[2]) = 1;

  simd::uint32_v idx0 = 0 + gt01 + gt02;
  simd::uint32_v idx1 = 1 - gt01 + gt12;
  simd::uint32_v idx2 = 2 - gt02 - gt12;

  for (uint32_t k = 0; k < simd::len; k++) {
    axis[idx0[k]][k] = 0;
    axis[idx1[k]][k] = 1;
    axis[idx2[k]][k] = 2;
  }

  for (uint8_t i = 0; i < 3; i++) {
    swap(cell[i], center[i], ray.direction[i] < 0.f);
  }

  simd::RayRange child_range = range;

  for (uint8_t i = 0; i < 3; i++) {

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask[k]) {
        child_range.max[k] = tcenter[axis[i][k]][k];
      }
    }

    child_range.max = min(child_range.max, range.max);

    simd::float_m child_in_range = mask && (child_range.min < child_range.max);

    if (child_in_range.isNotEmpty()) {
      ray_octree_traversal(ray, child_range, cell, layer + 1, child_in_range, callback);
      child_range.min = child_range.max;
    }

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask[k]) {
        cell[axis[i][k]][k] = center[axis[i][k]][k];
      }
    }
  }

  child_range.max = range.max;
  simd::float_m child_in_range = mask && child_range.min < child_range.max;

  if (child_in_range.isNotEmpty()) {
    ray_octree_traversal(ray, child_range, cell, layer + 1, child_in_range, callback);
  }
}
