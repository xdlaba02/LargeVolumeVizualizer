#pragma once

#include <raw_volume/raw_volume.h>
#include <raw_volume/sampler.h>

#include <ray/intersection.h>
#include <ray/ray.h>

#include <utils/utils.h>
#include <utils/blend.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
glm::vec4 integrate_raw_slab(const RawVolume<T> &volume, const Ray &ray, float step, float terminate_thresh, const TransferFunctionType &transfer_function) {
  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  RayRange range = intersect_aabb_ray(ray, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f});

  if (range.min < range.max) {
    RayRange slab_range { range.min, range.min };

    float slab_start_value {};

    while (slab_range.max < range.max) {

      if (dst.a < terminate_thresh) {
        break;
      }

      glm::vec3 sample_pos = ray.origin + ray.direction * slab_range.max;

      float slab_end_value = sample(volume, sample_pos.x, sample_pos.y, sample_pos.z);

      blend(transfer_function(slab_start_value, slab_end_value), dst, slab_range.max - slab_range.min);

      slab_start_value = slab_end_value;
      slab_range.min = slab_range.max;
      slab_range.max = slab_range.max + step;
    }
  }

  return dst;
}
