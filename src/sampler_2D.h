#pragma once

#include "simd.h"

#include <cstdint>

template <uint64_t STRIDE>
inline simd::float_v sampler2D(const float *data, const simd::float_v &xs, const simd::float_v &ys, const simd::float_m &mask) {
  simd::uint32_v pix_xs = xs;
  simd::uint32_v pix_ys = ys;

  simd::float_v frac_xs = xs - pix_xs;
  simd::float_v frac_ys = ys - pix_ys;

  simd::float_v accs[2][2];

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      uint64_t base = pix_ys[k] * STRIDE + pix_xs[k];

      accs[0][0][k] = data[base];
      accs[0][1][k] = data[base + 1];
      accs[1][0][k] = data[base + STRIDE];
      accs[1][1][k] = data[base + STRIDE + 1];
    }
  }

  accs[0][0] += (accs[0][1] - accs[0][0]) * frac_xs;
  accs[1][0] += (accs[1][1] - accs[1][0]) * frac_xs;

  return accs[0][0] + (accs[1][0] - accs[0][0]) * frac_ys;
};

template <uint64_t STRIDE>
inline float sampler2D(const float *data, float x, float y) {
  uint32_t pix_x = x;
  uint32_t pix_y = y;

  float frac_x = x - pix_x;
  float frac_y = y - pix_y;

  float acc[2][2];

  uint64_t base = pix_y * STRIDE + pix_x;

  acc[0][0] = data[base];
  acc[0][1] = data[base + 1];
  acc[1][0] = data[base + STRIDE];
  acc[1][1] = data[base + STRIDE + 1];

  acc[0][0] += (acc[0][1] - acc[0][0]) * frac_x;
  acc[1][0] += (acc[1][1] - acc[1][0]) * frac_x;

  return acc[0][0] + (acc[1][0] - acc[0][0]) * frac_y;
};
