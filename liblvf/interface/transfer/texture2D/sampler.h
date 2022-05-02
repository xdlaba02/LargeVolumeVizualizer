/**
* @file sampler.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Functions for sampling 2D texture with bilinear interpolation.
* Could be used to sample multiple same-size textures.
*/

#pragma once

#include "texture2D.h"

struct SampleInfo {
  uint32_t pix[2][2];
  float frac[2];
};

SampleInfo sample_info(uint32_t width, uint32_t height, float x, float y) {
  SampleInfo info;

  float denorm_x = x * width  - 0.5f;
  float denorm_y = y * height - 0.5f;

  info.pix[0][0] = denorm_x;
  info.pix[0][1] = std::min<uint32_t>(denorm_x + 1.f, width - 1);

  info.pix[1][0] = denorm_y;
  info.pix[1][1] = std::min<uint32_t>(denorm_y + 1.f, height - 1);

  info.frac[0] = denorm_x - info.pix[0][0];
  info.frac[1] = denorm_y - info.pix[1][0];

  return info;
}

template <typename T>
inline float sample(const Texture2D<T> &texture, const SampleInfo &info) {
  float acc[2][2];

  acc[0][0] = texture(info.pix[0][0], info.pix[1][0]);
  acc[0][1] = texture(info.pix[0][1], info.pix[1][0]);
  acc[1][0] = texture(info.pix[0][0], info.pix[1][1]);
  acc[1][1] = texture(info.pix[0][1], info.pix[1][1]);

  acc[0][0] += (acc[0][1] - acc[0][0]) * info.frac[0];
  acc[1][0] += (acc[1][1] - acc[1][0]) * info.frac[0];

  return acc[0][0] + (acc[1][0] - acc[0][0]) * info.frac[1];
};

template <typename T>
inline float sample(const Texture2D<T> &texture, float x, float y) {
  return sample(texture, sample_info(texture.width(), texture.height(), x, y));
};
