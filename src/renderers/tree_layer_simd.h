#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler_simd.h>

#include <ray/ray_simd.h>
#include <ray/intersection_simd.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
simd::vec4 render_layer(const TreeVolume<T> &volume, const simd::Ray &ray, simd::float_m mask, uint8_t layer, float step, const TransferFunctionType &transfer_function) {
  simd::vec4 dst(0.f, 0.f, 0.f, 1.f);

  simd::RayRange range {};

  intersect_aabb_ray(ray.origin, ray.direction_inverse, {0, 0, 0}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, range.min, range.max);

  mask = mask && range.min < range.max;

  if (!mask.isEmpty()) {
    simd::float_v stepsize = step * exp2i(layer);

    simd::RayRange slab_range { range.min, range.min };
    simd::float_v slab_start_value {};

    while (!mask.isEmpty()) {
      simd::vec3 sample_pos = ray.origin + ray.direction * slab_range.max;

      simd::float_v slab_end_value = sample(volume, sample_pos.x, sample_pos.y, sample_pos.z, layer, mask);

      blend(transfer_function(slab_end_value, slab_start_value, mask), dst, slab_range.max - slab_range.min, mask);

      slab_start_value = slab_end_value;
      slab_range.min = slab_range.max;
      slab_range.max = slab_range.max + stepsize;

      mask = mask && slab_range.max < range.max && dst.a > 1.f / 256.f;
    }
  }

  return dst;
}
