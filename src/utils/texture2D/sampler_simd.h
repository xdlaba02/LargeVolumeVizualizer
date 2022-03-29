#pragma once

#include "texture2D.h"

#include <utils/glm_simd.h>

#include <cstdint>

namespace simd {
  struct SampleInfo {
    uint32_v pix[2][2];
    float_v frac[2];
  };
}

simd::SampleInfo sample_info(uint32_t width, uint32_t height, simd::float_v x, simd::float_v y) {
  simd::SampleInfo info;

  simd::float_v denorm_x = x * width  - 0.5f;
  simd::float_v denorm_y = y * height - 0.5f;

  info.pix[0][0] = denorm_x;
  info.pix[0][1] = min(denorm_x + 1.f, width - 1);

  info.pix[1][0] = denorm_y;
  info.pix[1][1] = min(denorm_y + 1.f, height - 1);

  info.frac[0] = denorm_x - info.pix[0][0];
  info.frac[1] = denorm_y - info.pix[1][0];

  return info;
}

template <typename T>
inline simd::float_v sample(const Texture2D<T> &texture, const simd::SampleInfo &info, const simd::float_m &mask) {
  simd::float_v acc[2][2];

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      acc[0][0][k] = texture(info.pix[0][0][k], info.pix[1][0][k]);
      acc[0][1][k] = texture(info.pix[0][1][k], info.pix[1][0][k]);
      acc[1][0][k] = texture(info.pix[0][0][k], info.pix[1][1][k]);
      acc[1][1][k] = texture(info.pix[0][1][k], info.pix[1][1][k]);
    }
  }

  acc[0][0] += (acc[0][1] - acc[0][0]) * info.frac[0];
  acc[1][0] += (acc[1][1] - acc[1][0]) * info.frac[0];

  return acc[0][0] + (acc[1][0] - acc[0][0]) * info.frac[1];
};

template <typename T>
inline simd::float_v sample(const Texture2D<T> &texture, const simd::float_v &x, const simd::float_v &y, const simd::float_m &mask) {
  return sample(texture, sample_info(texture.width(), texture.height(), x, y), mask);
};
