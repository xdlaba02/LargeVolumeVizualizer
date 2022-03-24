#pragma once

#include <tree_volume/tree_volume.h>
#include <tree_volume/sampler.h>

#include <ray/intersection.h>
#include <ray/ray.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
glm::vec4 render_layer(const TreeVolume<T> &volume, const Ray &ray, uint8_t layer, float step, const TransferFunctionType &transfer_function) {
  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  RayRange range;

  intersect_aabb_ray(ray.origin, ray.direction_inverse, {0, 0, 0}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, range.min, range.max);

  if (range.min < range.max) {
    float stepsize = step * approx_exp2(layer);

    RayRange slab_range { range.min, range.min };

    float slab_start_value {};

    while (slab_range.max < range.max && dst.a > 1.f / 256.f) {
      glm::vec3 sample_pos = ray.origin + ray.direction * slab_range.max;

      float slab_end_value = sample(volume, sample_pos.x, sample_pos.y, sample_pos.z, layer);

      blend(transfer_function(slab_end_value, slab_start_value), dst, slab_range.max - slab_range.min);

      slab_start_value = slab_end_value;
      slab_range.min = slab_range.max;
      slab_range.max = slab_range.max + stepsize;
    }
  }


  return dst;
}
