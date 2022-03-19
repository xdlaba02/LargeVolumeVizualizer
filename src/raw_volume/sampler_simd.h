#pragma once

#include "raw_volume.h"

#include <utils/simd.h>

template <typename T>
inline simd::float_v sample(const RawVolume<T> &volume, const simd::float_v &x, const simd::float_v &y, const simd::float_v &z, const simd::float_m &mask) {
  simd::uint32_v pix_x = x;
  simd::uint32_v pix_y = y;
  simd::uint32_v pix_z = z;

  simd::float_v frac_x = x - pix_x;
  simd::float_v frac_y = y - pix_y;
  simd::float_v frac_z = z - pix_z;

  simd::float_m incrementable_x = pix_x < (volume.width - 1);
  simd::float_m incrementable_y = pix_y < (volume.height - 1);
  simd::float_m incrementable_z = pix_z < (volume.depth - 1);

  simd::int32_v buffer[2][2][2];

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      uint64_t base = volume.voxel_handle(pix_x[k], pix_y[k], pix_z[k]);

      uint64_t x_offset = incrementable_x[k] * 1;
      uint64_t y_offset = incrementable_y[k] * volume.width;
      uint64_t z_offset = incrementable_z[k] * volume.stride;

      buffer[0][0][0][k] = volume.data[base];
      buffer[0][0][1][k] = volume.data[base + x_offset];
      buffer[0][1][0][k] = volume.data[base + y_offset];
      buffer[0][1][1][k] = volume.data[base + y_offset + x_offset];
      buffer[1][0][0][k] = volume.data[base + z_offset];
      buffer[1][0][1][k] = volume.data[base + z_offset + x_offset];
      buffer[1][1][0][k] = volume.data[base + z_offset + y_offset];
      buffer[1][1][1][k] = volume.data[base + z_offset + y_offset + x_offset];
    }
  }

  simd::float_v acc[2][2];

  acc[0][0] = buffer[0][0][0] + (buffer[0][0][1] - buffer[0][0][0]) * frac_x;
  acc[0][1] = buffer[0][1][0] + (buffer[0][1][1] - buffer[0][1][0]) * frac_x;
  acc[1][0] = buffer[1][0][0] + (buffer[1][0][1] - buffer[1][0][0]) * frac_x;
  acc[1][1] = buffer[1][1][0] + (buffer[1][1][1] - buffer[1][1][0]) * frac_x;

  acc[0][0] += (acc[0][1] - acc[0][0]) * frac_y;
  acc[1][0] += (acc[1][1] - acc[1][0]) * frac_y;

  acc[0][0] += (acc[1][0] - acc[0][0]) * frac_z;

  return acc[0][0];
};
