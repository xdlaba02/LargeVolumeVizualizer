#pragma once

#include <raw_volume/raw_volume.h>
#include <raw_volume/sampler_simd.h>

#include <ray/intersection_simd.h>
#include <ray/ray_simd.h>

#include <utils/blend_simd.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
simd::vec4 integrate_raw_slab_simd(const RawVolume<T> &volume, const simd::Ray &ray, float step, float terminate_thresh, simd::float_m mask, const TransferFunctionType &transfer_function) {
  simd::vec4 dst = {0.f, 0.f, 0.f, 1.f};

  simd::RayRange range = intersect_aabb_ray(ray, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f});

  mask = mask && range.min < range.max;

  if (mask.isNotEmpty()) {
    simd::RayRange slab_range { range.min, range.min };

    simd::float_v slab_start_value {};

    for (mask = mask & slab_range.max < range.max & dst.a >= terminate_thresh; mask.isNotEmpty(); mask &= slab_range.max < range.max & dst.a >= terminate_thresh) {
      simd::vec3 sample_pos = ray.origin + ray.direction * slab_range.max;

      simd::float_v slab_end_value = sample(volume, sample_pos.x, sample_pos.y, sample_pos.z, mask);

      blend(transfer_function(slab_start_value, slab_end_value, mask), dst, slab_range.max - slab_range.min, mask);

      slab_start_value = slab_end_value;
      slab_range.min = slab_range.max;
      slab_range.max = slab_range.max + step;
    }
  }

  return dst;
}
