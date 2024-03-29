/**
* @file sampler.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Function that samples raw volume with trilinear interpolation.
*/

#pragma once

#include "raw_volume.h"

#include <utils/linear_interpolation.h>

template <typename T>
inline float sample(const RawVolume<T> &volume, float x, float y, float z) {
  float denorm_x = x * volume.width  - 0.5f;
  float denorm_y = y * volume.height - 0.5f;
  float denorm_z = z * volume.depth  - 0.5f;

  uint32_t vox_x[2];
  uint32_t vox_y[2];
  uint32_t vox_z[2];

  vox_x[0] = denorm_x;
  vox_y[0] = denorm_y;
  vox_z[0] = denorm_z;

  vox_x[1] = std::min<uint32_t>(denorm_x + 1.f, volume.width  - 1);
  vox_y[1] = std::min<uint32_t>(denorm_y + 1.f, volume.height - 1);
  vox_z[1] = std::min<uint32_t>(denorm_z + 1.f, volume.depth  - 1);

  float frac_x = denorm_x - vox_x[0];
  float frac_y = denorm_y - vox_y[0];
  float frac_z = denorm_z - vox_z[0];

  float acc[2][2][2];

  for (uint8_t z = 0; z < 2; z++) {
    for (uint8_t y = 0; y < 2; y++) {
      for (uint8_t x = 0; x < 2; x++) {
        acc[z][y][x] = volume.data[volume.voxel_handle(vox_x[x], vox_y[y], vox_z[z])];
      }
    }
  }

  interpolate(acc, frac_x, frac_y, frac_z);

  return acc[0][0][0];
};
