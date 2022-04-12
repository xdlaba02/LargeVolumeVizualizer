#pragma once

#include "tree_volume.h"

#include <utils/morton.h>
#include <utils/linear_interpolation.h>
#include <utils/fast_exp2.h>
#include <utils/simd.h>

// expects coordinates from interval <-.5f, TreeVolume<T, N>::SUBVOLUME_SIDE - .5f>
// can safely handle values from interval (-1.f, TreeVolume<T, N>::SUBVOLUME_SIDE) due to padding and truncation used
template <typename T, uint32_t N>
inline simd::float_v sample(const TreeVolume<T, N> &volume, const std::array<uint64_t, simd::len> &block_handle, const simd::float_v &denorm_x, const simd::float_v &denorm_y, const simd::float_v &denorm_z, const simd::float_m &mask) {

  simd::uint32_v indices[2][2][2];

  simd::float_v frac_x;
  simd::float_v frac_y;
  simd::float_v frac_z;

  {
    simd::uint32_v vox_x[2];
    simd::uint32_v vox_y[2];
    simd::uint32_v vox_z[2];

    vox_x[0] = denorm_x;
    vox_y[0] = denorm_y;
    vox_z[0] = denorm_z;

    vox_x[1] = denorm_x + 1.f;
    vox_y[1] = denorm_y + 1.f;
    vox_z[1] = denorm_z + 1.f;

    frac_x = denorm_x - vox_x[0];
    frac_y = denorm_y - vox_y[0];
    frac_z = denorm_z - vox_z[0];

    vox_x[0] = Morton<N>::interleave(vox_x[0]);
    vox_y[0] = Morton<N>::interleave(vox_y[0]);
    vox_z[0] = Morton<N>::interleave(vox_z[0]);

    vox_x[1] = Morton<N>::interleave(vox_x[1]);
    vox_y[1] = Morton<N>::interleave(vox_y[1]);
    vox_z[1] = Morton<N>::interleave(vox_z[1]);

    for (uint8_t z = 0; z < 2; z++) {
      for (uint8_t y = 0; y < 2; y++) {
        for (uint8_t x = 0; x < 2; x++) {
          indices[z][y][x] = Morton<N>::combine_interleaved(vox_x[x], vox_y[y], vox_z[z]);
        }
      }
    }
  }

  simd::float_v acc[2][2][2];

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      for (uint8_t z = 0; z < 2; z++) {
        for (uint8_t y = 0; y < 2; y++) {
          for (uint8_t x = 0; x < 2; x++) {
            acc[z][y][x][k] = volume.block(block_handle[k])[indices[z][y][x][k]];
          }
        }
      }
    }
  }

  interpolate(acc, frac_x, frac_y, frac_z);

  return acc[0][0][0];
}

// expects coordinates from interval <0, 1>
template <typename T, uint32_t N>
inline simd::float_v sample(const TreeVolume<T, N> &volume, const simd::float_v &x, const simd::float_v &y, const simd::float_v &z, uint8_t layer, simd::float_m mask) {
  uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

  simd::float_v denorm_x = x * exp2i(layer_index) * TreeVolume<T, N>::SUBVOLUME_SIDE - 0.5f;
  simd::float_v denorm_y = y * exp2i(layer_index) * TreeVolume<T, N>::SUBVOLUME_SIDE - 0.5f;
  simd::float_v denorm_z = z * exp2i(layer_index) * TreeVolume<T, N>::SUBVOLUME_SIDE - 0.5f;

  simd::uint32_v vox_x = denorm_x;
  simd::uint32_v vox_y = denorm_y;
  simd::uint32_v vox_z = denorm_z;

  simd::uint32_v block_x = div<TreeVolume<T, N>::SUBVOLUME_SIDE>(vox_x);
  simd::uint32_v block_y = div<TreeVolume<T, N>::SUBVOLUME_SIDE>(vox_y);
  simd::uint32_v block_z = div<TreeVolume<T, N>::SUBVOLUME_SIDE>(vox_z);

  simd::uint32_v min, max;
  std::array<uint64_t, simd::len> block_handles;

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      const typename TreeVolume<T, N>::Node &node = volume.node(volume.info.node_handle(block_x[k], block_y[k], block_z[k], layer));
      min[k] = node.min;
      max[k] = node.max;
      block_handles[k] = node.block_handle;
    }
  }

  simd::float_v samples = min;

  mask &= min != max;

  if (mask.isNotEmpty()) {
    simd::float_v in_block_x = denorm_x - block_x * TreeVolume<T, N>::SUBVOLUME_SIDE;
    simd::float_v in_block_y = denorm_y - block_y * TreeVolume<T, N>::SUBVOLUME_SIDE;
    simd::float_v in_block_z = denorm_z - block_z * TreeVolume<T, N>::SUBVOLUME_SIDE;

    samples(mask) = sample(volume, block_handles, in_block_x, in_block_y, in_block_z, mask);
  }

  return samples;
}
