#pragma once

#include "texture2D.h"

template <typename T>
inline float sample(const Texture2D<T> &texture, float x, float y) {
  float denorm_x = x * texture.width() - 0.5f;
  float denorm_y = y * texture.height() - 0.5f;

  uint32_t pix_x[2];
  uint32_t pix_y[2];

  pix_x[0] = denorm_x;
  pix_x[1] = std::min<uint32_t>(denorm_x + 1.f, texture.width() - 1);

  pix_y[0] = denorm_y;
  pix_y[1] = std::min<uint32_t>(denorm_y + 1.f, texture.height() - 1);

  float acc[2][2];

  acc[0][0] = texture(pix_x[0], pix_y[0]);
  acc[0][1] = texture(pix_x[1], pix_y[0]);
  acc[1][0] = texture(pix_x[0], pix_y[1]);
  acc[1][1] = texture(pix_x[1], pix_y[1]);

  float frac_x = denorm_x - pix_x[0];
  float frac_y = denorm_y - pix_y[0];

  acc[0][0] += (acc[0][1] - acc[0][0]) * frac_x;
  acc[1][0] += (acc[1][1] - acc[1][0]) * frac_x;

  return acc[0][0] + (acc[1][0] - acc[0][0]) * frac_y;
};
