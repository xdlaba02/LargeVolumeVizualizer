#pragma once

#include <utils/glm_simd.h>

void blend(const simd::vec4 &src_vec, simd::vec4 &dst_vec, simd::float_v stepsize, const simd::float_m &mask_vec) {
  simd::float_v alpha = exp(-src_vec.a * stepsize);

  simd::float_v coef = (1.f - alpha) * dst_vec.a;

  dst_vec.r(mask_vec) += src_vec.r * coef;
  dst_vec.g(mask_vec) += src_vec.g * coef;
  dst_vec.b(mask_vec) += src_vec.b * coef;
  dst_vec.a(mask_vec) *= alpha;
};
