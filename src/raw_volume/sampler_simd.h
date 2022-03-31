#pragma once

#include "raw_volume.h"

#include <utils/simd.h>

template <typename T>
inline simd::float_v sample(const RawVolume<T> &volume, const simd::float_v &x, const simd::float_v &y, const simd::float_v &z, const simd::float_m &mask) {
  simd::float_v denorm_x = x * volume.width  - 0.5f;
  simd::float_v denorm_y = y * volume.height - 0.5f;
  simd::float_v denorm_z = z * volume.depth  - 0.5f;

  simd::uint32_v vox_x[2];
  simd::uint32_v vox_y[2];
  simd::uint32_v vox_z[2];

  vox_x[0] = denorm_x;
  vox_y[0] = denorm_y;
  vox_z[0] = denorm_z;

  vox_x[1] = min(denorm_x + 1.f, volume.width  - 1);
  vox_y[1] = min(denorm_y + 1.f, volume.height - 1);
  vox_z[1] = min(denorm_z + 1.f, volume.depth  - 1);

  simd::float_v frac_x = denorm_x - vox_x[0];
  simd::float_v frac_y = denorm_y - vox_y[0];
  simd::float_v frac_z = denorm_z - vox_z[0];

  uint64_t base = volume.voxel_handle(vox_x, vox_y, vox_z);

  simd::float_v acc[2][2][2];

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      for (uint8_t z = 0; z < 2; z++) {
        for (uint8_t y = 0; y < 2; y++) {
          for (uint8_t x = 0; x < 2; x++) {
            acc[z][y][x][k] = volume.data[volume.voxel_handle(vox_x[x][k], vox_y[y][k], vox_z[z][k])];
          }
        }
      }
    }
  }

  acc[0][0][0] = acc[0][0][0] + (acc[0][0][1] - acc[0][0][0]) * frac_x;
  acc[0][1][0] = acc[0][1][0] + (acc[0][1][1] - acc[0][1][0]) * frac_x;
  acc[1][0][0] = acc[1][0][0] + (acc[1][0][1] - acc[1][0][0]) * frac_x;
  acc[1][1][0] = acc[1][1][0] + (acc[1][1][1] - acc[1][1][0]) * frac_x;

  acc[0][0][0] += (acc[0][1][0] - acc[0][0][0]) * frac_y;
  acc[1][0][0] += (acc[1][1][0] - acc[1][0][0]) * frac_y;

  acc[0][0][0] += (acc[1][0][0] - acc[0][0][0]) * frac_z;

  return acc[0][0][0];
};
