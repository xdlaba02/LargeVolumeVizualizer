#pragma once

#include "tree_volume.h"
#include "sampler_simd.h"

#include <ray_traversal/intersection_simd.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
glm::vec<4, simd::float_v> integrate(const TreeVolume<T> &volume, const glm::vec3 &origin, const glm::vec<3, simd::float_v> &direction, simd::float_m mask, uint8_t layer, float step, const TransferFunctionType &transfer_function) {
  simd::float_v tmin, tmax;
  intersect_aabb_ray(origin, simd::float_v(1.f) / direction, {0, 0, 0}, {1, 1, 1}, tmin, tmax);

  simd::float_v stepsize = step * simd::float_v(layer + 1);

  glm::vec<4, simd::float_v> dst(0.f, 0.f, 0.f, 1.f);

  simd::float_v prev_value {};

  mask = mask && tmin < tmax;

  if (!mask.isEmpty()) {
    glm::vec<3, simd::float_v> sample = glm::vec<3, simd::float_v>(origin) + direction * tmin;
    prev_value = volume.sample_volume(sample.x, sample.y, sample.z, layer, tmin < tmax);
  }

  tmin += stepsize;
  mask = mask && tmin < tmax;

  while (!mask.isEmpty()) {
    glm::vec<3, simd::float_v> sample = glm::vec<3, simd::float_v>(origin) + direction * tmin;

    simd::float_v value = volume.sample_volume(sample.x, sample.y, sample.z, layer, mask);

    glm::vec<4, simd::float_v> src = transfer_function(value, prev_value, mask);

    simd::float_m alpha_mask = mask && src.a > 0.f;

    if (!alpha_mask.isEmpty()) {
      simd::float_v alpha = simd::float_v(1.f) - Vc::exp(-src.a * stepsize);

      simd::float_v coef = alpha * dst.a;

      dst.r(alpha_mask) += src.r * coef;
      dst.g(alpha_mask) += src.g * coef;
      dst.b(alpha_mask) += src.b * coef;
      dst.a(alpha_mask) *= 1 - alpha;

      mask &= dst.a > 1.f / 256.f;
    }

    prev_value = value;
    tmin += stepsize;
    mask = mask && tmin < tmax;
  }

  return dst;
}
