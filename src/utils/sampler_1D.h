#pragma once

#include "simd.h"

inline simd::float_v sampler1D(const float *data, const simd::float_v &values, const simd::float_m &mask) {
  simd::uint32_v pix = values;

  simd::float_v accs[2];

  accs[0].gather(data, pix + 0, mask);
  accs[1].gather(data, pix + 1, mask);

  return accs[0] + (accs[1] - accs[0]) * (values - pix);
};

inline float sampler1D(const float *data, float value) {
  uint32_t pix = value;
  return data[pix + 0] + (data[pix + 1] - data[pix + 0]) * (value - pix);
};
