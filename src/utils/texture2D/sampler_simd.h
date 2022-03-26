#pragma once

#include "texture2D.h"

#include <utils/glm_simd.h>

#include <cstdint>

template <typename T>
inline simd::float_v sample(const Texture2D<T> &texture, const simd::float_v &x, const simd::float_v &y, const simd::float_m &mask) {
  simd::float_v denorm_x = x * texture.width() - 0.5f;
  simd::float_v denorm_y = y * texture.height() - 0.5f;

  simd::uint32_v pix_x[2];
  simd::uint32_v pix_y[2];

  pix_x[0] = denorm_x;
  pix_x[1] = min(denorm_x + 1.f, texture.width() - 1);

  pix_y[0] = denorm_y;
  pix_y[1] = min(denorm_y + 1.f, texture.height() - 1);

  simd::float_v acc[2][2];

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      acc[0][0][k] = texture(pix_x[0][k], pix_y[0][k]);
      acc[0][1][k] = texture(pix_x[1][k], pix_y[0][k]);
      acc[1][0][k] = texture(pix_x[0][k], pix_y[1][k]);
      acc[1][1][k] = texture(pix_x[1][k], pix_y[1][k]);
    }
  }

  simd::float_v frac_x = denorm_x - pix_x[0];

  acc[0][0] += (acc[0][1] - acc[0][0]) * frac_x;
  acc[1][0] += (acc[1][1] - acc[1][0]) * frac_x;

  simd::float_v frac_y = denorm_y - pix_y[0];

  return acc[0][0] + (acc[1][0] - acc[0][0]) * frac_y;
};
