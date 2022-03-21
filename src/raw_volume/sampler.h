#pragma once

#include "raw_volume.h"

template <typename T>
inline float sample(const RawVolume<T> &volume, float x, float y, float z) {
  uint32_t pix_x = x;
  uint32_t pix_y = y;
  uint32_t pix_z = z;

  float frac_x = x - pix_x;
  float frac_y = y - pix_y;
  float frac_z = z - pix_z;

  int32_t buffer[2][2][2];

  uint64_t base = volume.voxel_handle(pix_x, pix_y, pix_z);

  uint64_t x_offset = pix_x < (volume.width - 1);
  uint64_t y_offset = pix_y < (volume.height - 1) * volume.width;
  uint64_t z_offset = pix_z < (volume.depth - 1) * volume.stride;

  buffer[0][0][0] = volume.data[base];
  buffer[0][0][1] = volume.data[base + x_offset];
  buffer[0][1][0] = volume.data[base + y_offset];
  buffer[0][1][1] = volume.data[base + y_offset + x_offset];
  buffer[1][0][0] = volume.data[base + z_offset];
  buffer[1][0][1] = volume.data[base + z_offset + x_offset];
  buffer[1][1][0] = volume.data[base + z_offset + y_offset];
  buffer[1][1][1] = volume.data[base + z_offset + y_offset + x_offset];

  float acc[2][2];

  acc[0][0] = buffer[0][0][0] + (buffer[0][0][1] - buffer[0][0][0]) * frac_x;
  acc[0][1] = buffer[0][1][0] + (buffer[0][1][1] - buffer[0][1][0]) * frac_x;
  acc[1][0] = buffer[1][0][0] + (buffer[1][0][1] - buffer[1][0][0]) * frac_x;
  acc[1][1] = buffer[1][1][0] + (buffer[1][1][1] - buffer[1][1][0]) * frac_x;

  acc[0][0] += (acc[0][1] - acc[0][0]) * frac_y;
  acc[1][0] += (acc[1][1] - acc[1][0]) * frac_y;

  acc[0][0] += (acc[1][0] - acc[0][0]) * frac_z;

  return acc[0][0];
};
