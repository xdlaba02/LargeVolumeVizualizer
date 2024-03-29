/**
* @file blend.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Functin that performs integration step with specific size and blends the result into the output color.
*/

#pragma once

#include <glm/glm.hpp>

void blend(const glm::vec4 &src, glm::vec4 &dst, float stepsize) {
  float alpha = std::exp(-src.a * stepsize);

  float coef = (1.f - alpha) * dst.a;

  dst.r += src.r * coef;
  dst.g += src.g * coef;
  dst.b += src.b * coef;
  dst.a *= alpha;
};
