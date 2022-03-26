#pragma once

#include <utils/glm_simd.h>

void blend(const simd::vec4 &src, simd::vec4 &dst, simd::float_v stepsize, const simd::float_m &mask) {
  simd::float_v alpha = exp(-src.a * stepsize);

  simd::float_v coef = (1.f - alpha) * dst.a;

  dst.r(mask) += src.r * coef;
  dst.g(mask) += src.g * coef;
  dst.b(mask) += src.b * coef;
  dst.a(mask) *= alpha;
};
