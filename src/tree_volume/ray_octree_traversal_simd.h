#pragma once

#include "../utils.h"
#include "../simd.h"

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

using Vec3Vec = glm::vec<3, simd::float_v>;
using Vec4Vec = glm::vec<4, simd::float_v>;

struct RayVec {
  Vec3Vec origin;
  Vec3Vec direction;
  Vec3Vec direction_inverse;
};

struct RayRangeVec {
  simd::float_v min;
  simd::float_v max;
};

template <typename F>
void ray_octree_traversal(const RayVec &ray_vec, const RayRangeVec &range_vec, Vec3Vec cell_vec, uint32_t layer, simd::float_m mask_vec, const F &callback) {
  callback(range_vec, cell_vec, layer, mask_vec);

  if (mask_vec.isEmpty()) {
    return;
  }

  // TODO precompute?
  float child_size = approx_exp2(-layer - 1);

  Vec3Vec center_vec;
  Vec3Vec tcenter_vec;
  std::array<simd::uint32_v, 3> axis_vec;

  center_vec = cell_vec + simd::float_v(child_size);

  tcenter_vec = (center_vec - ray_vec.origin) * ray_vec.direction_inverse;

  // fast sort axis by tcenter_vec
  simd::uint32_v gt01 = 0;
  simd::uint32_v gt02 = 0;
  simd::uint32_v gt12 = 0;

  gt01(tcenter_vec[0] > tcenter_vec[1]) = 1;
  gt02(tcenter_vec[0] > tcenter_vec[2]) = 1;
  gt12(tcenter_vec[1] > tcenter_vec[2]) = 1;

  simd::uint32_v idx0 = 0 + gt01 + gt02;
  simd::uint32_v idx1 = 1 - gt01 + gt12;
  simd::uint32_v idx2 = 2 - gt02 - gt12;

  for (uint32_t k = 0; k < simd::len; k++) {
    axis_vec[idx0[k]][k] = 0;
    axis_vec[idx1[k]][k] = 1;
    axis_vec[idx2[k]][k] = 2;
  }

  for (uint8_t i = 0; i < 3; i++) {
    simd::swap(cell_vec[i], center_vec[i], ray_vec.direction[i] < 0.f);
  }

  RayRangeVec child_range_vec = range_vec;

  for (uint8_t i = 0; i < 3; i++) {

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask_vec[k]) {
        uint32_t axis = axis_vec[i][k];
        child_range_vec.max[k] = tcenter_vec[axis][k];
      }
    }

    child_range_vec.max = simd::min(child_range_vec.max, range_vec.max);

    simd::float_m child_in_range = mask_vec && (child_range_vec.min < child_range_vec.max);

    if (child_in_range.isNotEmpty()) {
      ray_octree_traversal(ray_vec, child_range_vec, cell_vec, layer + 1, child_in_range, callback);
      child_range_vec.min = child_range_vec.max;
    }

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask_vec[k]) {
        uint32_t axis = axis_vec[i][k];
        cell_vec[axis][k] = center_vec[axis][k];
      }
    }
  }

  child_range_vec.max = range_vec.max;
  simd::float_m child_in_range = mask_vec && child_range_vec.min < child_range_vec.max;

  if (child_in_range.isNotEmpty()) {
    ray_octree_traversal(ray_vec, child_range_vec, cell_vec, layer + 1, child_in_range, callback);
  }
}
