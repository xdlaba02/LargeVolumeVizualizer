#pragma once

#include <utils/simd.h>

#include <glm/glm.hpp>

void blend(const glm::vec<4, simd::float_v> &src_vec, glm::vec<4, simd::float_v> &dst_vec, simd::float_v stepsize, const simd::float_m &mask_vec) {
  simd::float_v alpha = simd::exp(-src_vec.a * stepsize);

  simd::float_v coef = (1.f - alpha) * dst_vec.a;

  dst_vec.r(mask_vec) += src_vec.r * coef;
  dst_vec.g(mask_vec) += src_vec.g * coef;
  dst_vec.b(mask_vec) += src_vec.b * coef;
  dst_vec.a(mask_vec) *= alpha;
};
