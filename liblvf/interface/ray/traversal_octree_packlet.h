#pragma once

#include "ray_simd.h"

#include <utils/fast_exp2.h>
#include <utils/simd.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

// Sqaure packlet of simd::len * simd::len rays.
using RayPacklet = std::array<simd::Ray, simd::len>;
using MaskPacklet = std::array<simd::float_m, simd::len>;
using RayRangePacklet = std::array<simd::RayRange, simd::len>;
using Vec3Packlet = std::array<simd::vec3, simd::len>;
using Vec4Packlet = std::array<simd::vec4, simd::len>;
using AxisPacklet = std::array<std::array<simd::uint32_v, 3>, simd::len>;
using FloatPacklet = std::array<simd::float_v, simd::len>;

template <typename F>
concept rayOctreeTraversalPackletCallback = std::invocable<F, const RayPacklet &, const Vec3Packlet &, uint8_t, MaskPacklet &>;

template <typename F>
void ray_octree_traversal(const RayPacklet &ray_packlet, const RayRangePacklet &range_packlet, Vec3Packlet cell_packlet, uint8_t layer, MaskPacklet mask_packlet, const F &callback) {
  callback(range_packlet, cell_packlet, layer, mask_packlet);

  bool should_recurse = false;
  for (uint32_t j = 0; j < simd::len; j++) {
    if (mask_packlet[j].isNotEmpty()) {
      should_recurse = true;
      break;
    }
  }

  if (!should_recurse) {
    return;
  }

  // TODO precompute?
  float child_size = exp2i(-layer - 1);

  Vec3Packlet center_packlet;
  Vec3Packlet tcenter_packlet;
  AxisPacklet axis_packlet;

  for (uint32_t j = 0; j < simd::len; j++) {
    if (mask_packlet[j].isEmpty()) {
      continue;
    }

    center_packlet[j] = cell_packlet[j] + simd::float_v(child_size);

    tcenter_packlet[j] = (center_packlet[j] - ray_packlet[j].origin) * ray_packlet[j].direction_inverse;

    // fast sort axis by tcenter_packlet
    simd::uint32_v gt01 = 0;
    simd::uint32_v gt02 = 0;
    simd::uint32_v gt12 = 0;

    gt01(tcenter_packlet[j][0] > tcenter_packlet[j][1]) = 1;
    gt02(tcenter_packlet[j][0] > tcenter_packlet[j][2]) = 1;
    gt12(tcenter_packlet[j][1] > tcenter_packlet[j][2]) = 1;

    simd::uint32_v idx0 = 0 + gt01 + gt02;
    simd::uint32_v idx1 = 1 - gt01 + gt12;
    simd::uint32_v idx2 = 2 - gt02 - gt12;

    for (uint32_t k = 0; k < simd::len; k++) {
      axis_packlet[j][idx0[k]][k] = 0;
      axis_packlet[j][idx1[k]][k] = 1;
      axis_packlet[j][idx2[k]][k] = 2;
    }

    for (uint8_t i = 0; i < 3; i++) {
      swap(cell_packlet[j][i], center_packlet[j][i], ray_packlet[j].direction[i] < 0.f);
    }
  }

  RayRangePacklet child_range_packlet = range_packlet;

  for (uint8_t i = 0; i < 3; i++) {

    MaskPacklet child_in_range;
    bool packlet_not_empty = false;

    for (uint8_t j = 0; j < simd::len; j++) {

      for (uint8_t k = 0; k < simd::len; k++) {
        if (mask_packlet[j][k]) {
          uint8_t axis = axis_packlet[j][i][k];
          child_range_packlet[j].max[k] = tcenter_packlet[j][axis][k];
        }
      }

      child_range_packlet[j].max = std::min(child_range_packlet[j].max, range_packlet[j].max);

      child_in_range[j] = mask_packlet[j] && (child_range_packlet[j].min < child_range_packlet[j].max);
      packlet_not_empty = packlet_not_empty || mask_packlet[j].isNotEmpty();
    }

    if (packlet_not_empty) {
      ray_octree_traversal(ray_packlet, child_range_packlet, cell_packlet, layer + 1, child_in_range, callback);

      for (uint8_t j = 0; j < simd::len; j++) {
        child_range_packlet[j].min = child_range_packlet[j].max;
      }
    }

    for (uint8_t j = 0; j < simd::len; j++) {
      for (uint8_t k = 0; k < simd::len; k++) {
        if (mask_packlet[j][k]) {
          uint8_t axis = axis_packlet[j][i][k];
          cell_packlet[j][axis][k] = center_packlet[j][axis][k];
        }
      }
    }
  }

  MaskPacklet child_in_range;
  bool packlet_not_empty = false;

  for (uint8_t j = 0; j < simd::len; j++) {
    child_range_packlet[j].max = range_packlet[j].max;
    child_in_range[j] = mask_packlet[j] && (child_range_packlet[j].min < child_range_packlet[j].max);
    packlet_not_empty = packlet_not_empty || mask_packlet[j].isNotEmpty();
  }

  if (packlet_not_empty) {
    ray_octree_traversal(ray_packlet, child_range_packlet, cell_packlet, layer + 1, child_in_range, callback);
  }
}
