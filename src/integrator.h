#pragma once

#include "tree_volume/tree_volume.h"
#include "tree_volume/sampler.h"
#include "intersection.h"

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
glm::vec4 integrate(const TreeVolume<T> &volume, const glm::vec3 &origin, const glm::vec3 &direction, uint8_t layer, float step, const TransferFunctionType &transfer_function) {
  float tmin, tmax;
  intersect_aabb_ray(origin, 1.f / direction, {0, 0, 0}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, tmin, tmax);

  float stepsize = step * (layer + 1);

  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  float prev_value {};

  if (tmin < tmax) {
    glm::vec3 sample = origin + direction * tmin;
    prev_value = ::sample(volume, sample.x, sample.y, sample.z, layer);
  }

  tmin += stepsize;

  while (tmin < tmax) {
    glm::vec3 sample = origin + direction * tmin;

    float value = ::sample(volume, sample.x, sample.y, sample.z, layer);

    auto src = transfer_function(value, prev_value);

    if (src.a > 0.f) {
      float alpha = 1.f - std::exp(-src.a * stepsize);

      float coef = alpha * dst.a;

      dst.r += src.r * coef;
      dst.g += src.g * coef;
      dst.b += src.b * coef;
      dst.a *= 1 - alpha;

      if (dst.a <= 1.f / 256.f) {
        break;
      }
    }

    prev_value = value;
    tmin += stepsize;
  }

  return dst;
}
