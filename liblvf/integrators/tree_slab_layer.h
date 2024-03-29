/**
* @file tree_slab_layer.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Function that integrates one layer of a tree volume with a ray.
*/


#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler.h>

#include <ray/intersection.h>
#include <ray/ray.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, uint32_t N, typename F>
glm::vec4 integrate_tree_slab_layer(const TreeVolume<T, N> &volume, const Ray &ray, uint8_t layer, float step, float terminate_thresh, const F &transfer_function) {
  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  RayRange range = intersect_aabb_ray(ray, {0, 0, 0}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac});

  uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

  if (range.min < range.max) {
    RayRange slab_range { range.min, range.min };

    float slab_start_value {};

    while (slab_range.max < range.max && dst.a > terminate_thresh) {
      glm::vec3 sample_pos = ray.origin + ray.direction * slab_range.max;

      float slab_end_value = sample(volume, sample_pos.x, sample_pos.y, sample_pos.z, layer_index);

      blend(transfer_function(slab_end_value, slab_start_value), dst, slab_range.max - slab_range.min);

      slab_start_value = slab_end_value;
      slab_range.min = slab_range.max;
      slab_range.max = slab_range.max + step;
    }
  }

  return dst;
}
