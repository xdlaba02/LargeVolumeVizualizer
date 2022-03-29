#pragma once

#include "raw_volume.h"

#include <utils/simd.h>

template <typename T>
inline simd::float_v sample(const RawVolume<T> &volume, const simd::float_v &x, const simd::float_v &y, const simd::float_v &z, const simd::float_m &mask) {
  simd::float_v denorm_x = x * volume.width  - 0.5f;
  simd::float_v denorm_y = y * volume.height - 0.5f;
  simd::float_v denorm_z = z * volume.depth  - 0.5f;

  simd::uint32_v vox_x = denorm_x;
  simd::uint32_v vox_y = denorm_y;
  simd::uint32_v vox_z = denorm_z;

  simd::float_v frac_x = denorm_x - vox_x;
  simd::float_v frac_y = denorm_y - vox_y;
  simd::float_v frac_z = denorm_z - vox_z;

  simd::float_m incrementable_x = vox_x < (volume.width - 1);
  simd::float_m incrementable_y = vox_y < (volume.height - 1);
  simd::float_m incrementable_z = vox_z < (volume.depth - 1);

  simd::float_v acc[2][2][2];

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      uint64_t base = volume.voxel_handle(vox_x[k], vox_y[k], vox_z[k]);

      uint64_t x_offset = incrementable_x[k] * 1;
      uint64_t y_offset = incrementable_y[k] * volume.width;
      uint64_t z_offset = incrementable_z[k] * volume.stride;

      acc[0][0][0][k] = volume.data[base];
      acc[0][0][1][k] = volume.data[base + x_offset];
      acc[0][1][0][k] = volume.data[base + y_offset];
      acc[0][1][1][k] = volume.data[base + y_offset + x_offset];
      acc[1][0][0][k] = volume.data[base + z_offset];
      acc[1][0][1][k] = volume.data[base + z_offset + x_offset];
      acc[1][1][0][k] = volume.data[base + z_offset + y_offset];
      acc[1][1][1][k] = volume.data[base + z_offset + y_offset + x_offset];
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
