#pragma once

#include "tree_volume.h"

#include <utils/morton.h>
#include <utils/fast_exp2.h>
#include <utils/linear_interpolation.h>

// expects coordinates from interval <-.5f, TreeVolume<T, N>::SUBVOLUME_SIDE - .5f>
// can safely handle values from interval (-1.f, TreeVolume<T, N>::SUBVOLUME_SIDE) due to truncation used
template <typename T, uint32_t N>
inline float sample(const TreeVolume<T, N> &volume, uint64_t block_handle, float denorm_x, float denorm_y, float denorm_z) {
  uint32_t vox_x[2];
  uint32_t vox_y[2];
  uint32_t vox_z[2];

  vox_x[0] = denorm_x;
  vox_y[0] = denorm_y;
  vox_z[0] = denorm_z;

  vox_x[1] = denorm_x + 1.f;
  vox_y[1] = denorm_y + 1.f;
  vox_z[1] = denorm_z + 1.f;

  float frac_x = denorm_x - vox_x[0];
  float frac_y = denorm_y - vox_y[0];
  float frac_z = denorm_z - vox_z[0];

  vox_x[0] = Morton<N>::interleave(vox_x[0]);
  vox_y[0] = Morton<N>::interleave(vox_y[0]);
  vox_z[0] = Morton<N>::interleave(vox_z[0]);

  vox_x[1] = Morton<N>::interleave(vox_x[1]);
  vox_y[1] = Morton<N>::interleave(vox_y[1]);
  vox_z[1] = Morton<N>::interleave(vox_z[1]);

  float acc[2][2][2];

  for (uint8_t z = 0; z < 2; z++) {
    for (uint8_t y = 0; y < 2; y++) {
      for (uint8_t x = 0; x < 2; x++) {
        acc[z][y][x] = volume.block(block_handle)[Morton<N>::combine_interleaved(vox_x[x], vox_y[y], vox_z[z])];
      }
    }
  }

  interpolate(acc, frac_x, frac_y, frac_z);

  return acc[0][0][0];
}

// expects coordinates from interval <0, 1>
template <typename T, uint32_t N>
inline float sample(const TreeVolume<T, N> &volume, float x, float y, float z, uint8_t layer) {
  uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

  float denorm_x = x * exp2i(layer_index) * TreeVolume<T, N>::SUBVOLUME_SIDE - 0.5f;
  float denorm_y = y * exp2i(layer_index) * TreeVolume<T, N>::SUBVOLUME_SIDE - 0.5f;
  float denorm_z = z * exp2i(layer_index) * TreeVolume<T, N>::SUBVOLUME_SIDE - 0.5f;

  uint32_t vox_x = denorm_x;
  uint32_t vox_y = denorm_y;
  uint32_t vox_z = denorm_z;

  uint32_t block_x = vox_x / TreeVolume<T, N>::SUBVOLUME_SIDE;
  uint32_t block_y = vox_y / TreeVolume<T, N>::SUBVOLUME_SIDE;
  uint32_t block_z = vox_z / TreeVolume<T, N>::SUBVOLUME_SIDE;

  const typename TreeVolume<T, N>::Node &node = volume.node(volume.info.node_handle(block_x, block_y, block_z, layer));

  if (node.min == node.max) {
    return node.min;
  }
  else {
    // reminder from division
    float in_block_x = denorm_x - block_x * TreeVolume<T, N>::SUBVOLUME_SIDE;
    float in_block_y = denorm_y - block_y * TreeVolume<T, N>::SUBVOLUME_SIDE;
    float in_block_z = denorm_z - block_z * TreeVolume<T, N>::SUBVOLUME_SIDE;

    return sample(volume, node.block_handle, in_block_x, in_block_y, in_block_z);
  }
};
