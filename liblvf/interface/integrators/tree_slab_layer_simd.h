/**
* @file tree_slab_layer_simd.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Function that integrates one layer of a tree volume with vector of rays.
*/

#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler_simd.h>

#include <ray/ray_simd.h>
#include <ray/intersection_simd.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, uint32_t N, typename F>
simd::vec4 integrate_tree_slab_layer_simd(const TreeVolume<T, N> &volume, const simd::Ray &ray, simd::float_m mask, uint8_t layer, float step, float terminate_thresh, const F &transfer_function) {
  simd::vec4 dst(0.f, 0.f, 0.f, 1.f);

  simd::RayRange range = intersect_aabb_ray(ray, {0, 0, 0}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac});

  mask = mask && range.min < range.max;

  uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

  if (!mask.isEmpty()) {
    simd::RayRange slab_range { range.min, range.min };
    simd::float_v slab_start_value {};

    while (!mask.isEmpty()) {
      simd::vec3 sample_pos = ray.origin + ray.direction * slab_range.max;

      simd::float_v slab_end_value = sample(volume, sample_pos.x, sample_pos.y, sample_pos.z, layer_index, mask);

      blend(transfer_function(slab_end_value, slab_start_value, mask), dst, slab_range.max - slab_range.min, mask);

      slab_start_value = slab_end_value;
      slab_range.min = slab_range.max;
      slab_range.max = slab_range.max + step;

      mask = mask && slab_range.max < range.max && dst.a > terminate_thresh;
    }
  }

  return dst;
}
