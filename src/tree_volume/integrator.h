#pragma once

#include "tree_volume.h"
#include "sampler.h"
#include "blend.h"

#include <ray/intersection.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
glm::vec4 integrate(const TreeVolume<T> &volume, const glm::vec3 &origin, const glm::vec3 &direction, uint8_t layer, float step, const TransferFunctionType &transfer_function) {
  float tmin, tmax;
  intersect_aabb_ray(origin, 1.f / direction, {0, 0, 0}, { volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac}, tmin, tmax);

  float stepsize = step * approx_exp2(layer);

  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  float prev_value {};

  if (tmin < tmax) {
    glm::vec3 sample = origin + direction * tmin;
    prev_value = ::sample(volume, sample.x, sample.y, sample.z, layer);
  }

  tmin += stepsize;

  while (tmin < tmax && dst.a > 1.f / 256.f) {
    glm::vec3 sample_pos = origin + direction * tmin;

    float value = sample(volume, sample_pos.x, sample_pos.y, sample_pos.z, layer);

    blend(transfer_function(value, prev_value), dst, stepsize);

    prev_value = value;
    tmin += stepsize;
  }

  return dst;
}
