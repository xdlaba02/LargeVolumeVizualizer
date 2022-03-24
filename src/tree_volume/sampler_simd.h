#pragma once

#include "tree_volume.h"

#include <utils/morton.h>
#include <utils/utils.h>
#include <utils/simd.h>

// expects coordinates from interval <-.5f, TreeVolume<T>::SUBVOLUME_SIDE - .5f>
// can safely handle values from interval (-1.f, TreeVolume<T>::SUBVOLUME_SIDE) due to padding and truncation used
template <typename T>
inline simd::float_v sample(const TreeVolume<T> &volume, const std::array<uint64_t, simd::len> &block_handle, const simd::float_v &denorm_x, const simd::float_v &denorm_y, const simd::float_v &denorm_z, const simd::float_m &mask) {
  simd::uint32_v denorm_x_low = denorm_x;
  simd::uint32_v denorm_y_low = denorm_y;
  simd::uint32_v denorm_z_low = denorm_z;

  simd::uint32_v denorm_x_high = denorm_x + 1.f;
  simd::uint32_v denorm_y_high = denorm_y + 1.f;
  simd::uint32_v denorm_z_high = denorm_z + 1.f;

  simd::float_v frac_x = denorm_x - denorm_x_low;
  simd::float_v frac_y = denorm_y - denorm_y_low;
  simd::float_v frac_z = denorm_z - denorm_z_low;

  simd::uint32_v denorm_x_low_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_x_low);
  simd::uint32_v denorm_y_low_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_y_low);
  simd::uint32_v denorm_z_low_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_z_low);

  simd::uint32_v denorm_x_high_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_x_high);
  simd::uint32_v denorm_y_high_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_y_high);
  simd::uint32_v denorm_z_high_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_z_high);

  simd::uint32_v morton_indices[2][2][2];

  morton_indices[0][0][0] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_low_interleaved);
  morton_indices[0][0][1] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_low_interleaved);
  morton_indices[0][1][0] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_low_interleaved);
  morton_indices[0][1][1] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_low_interleaved);
  morton_indices[1][0][0] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_high_interleaved);
  morton_indices[1][0][1] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_high_interleaved);
  morton_indices[1][1][0] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_high_interleaved);
  morton_indices[1][1][1] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_high_interleaved);

  simd::int32_v buffers[2][2][2];

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      buffers[0][0][0][k] = volume.blocks[block_handle[k]][morton_indices[0][0][0][k]];
      buffers[0][0][1][k] = volume.blocks[block_handle[k]][morton_indices[0][0][1][k]];
      buffers[0][1][0][k] = volume.blocks[block_handle[k]][morton_indices[0][1][0][k]];
      buffers[0][1][1][k] = volume.blocks[block_handle[k]][morton_indices[0][1][1][k]];
      buffers[1][0][0][k] = volume.blocks[block_handle[k]][morton_indices[1][0][0][k]];
      buffers[1][0][1][k] = volume.blocks[block_handle[k]][morton_indices[1][0][1][k]];
      buffers[1][1][0][k] = volume.blocks[block_handle[k]][morton_indices[1][1][0][k]];
      buffers[1][1][1][k] = volume.blocks[block_handle[k]][morton_indices[1][1][1][k]];
    }
  }

  simd::float_v accs[2][2];

  accs[0][0] = buffers[0][0][0] + (buffers[0][0][1] - buffers[0][0][0]) * frac_x;
  accs[0][1] = buffers[0][1][0] + (buffers[0][1][1] - buffers[0][1][0]) * frac_x;
  accs[1][0] = buffers[1][0][0] + (buffers[1][0][1] - buffers[1][0][0]) * frac_x;
  accs[1][1] = buffers[1][1][0] + (buffers[1][1][1] - buffers[1][1][0]) * frac_x;

  accs[0][0] += (accs[0][1] - accs[0][0]) * frac_y;
  accs[1][0] += (accs[1][1] - accs[1][0]) * frac_y;

  accs[0][0] += (accs[1][0] - accs[0][0]) * frac_z;

  return accs[0][0];
}

// expects coordinates from interval <0, 1>
template <typename T>
inline simd::float_v sample(const TreeVolume<T> &volume, const simd::float_v &x, const simd::float_v &y, const simd::float_v &z, uint8_t layer, const simd::float_m &mask) {
  simd::uint32_v width;
  simd::uint32_v height;
  simd::uint32_v depth;

  uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      width[k]  = exp2i(layer_index) * TreeVolume<T>::SUBVOLUME_SIDE;
      height[k] = exp2i(layer_index) * TreeVolume<T>::SUBVOLUME_SIDE;
      depth[k]  = exp2i(layer_index) * TreeVolume<T>::SUBVOLUME_SIDE;
    }
  }

  simd::float_v denorm_x = x * width  - 0.5f;
  simd::float_v denorm_y = y * height - 0.5f;
  simd::float_v denorm_z = z * depth  - 0.5f;

  simd::uint32_v vox_x = denorm_x;
  simd::uint32_v vox_y = denorm_y;
  simd::uint32_v vox_z = denorm_z;

  simd::uint32_v block_x = div<TreeVolume<T>::SUBVOLUME_SIDE>(vox_x);
  simd::uint32_v block_y = div<TreeVolume<T>::SUBVOLUME_SIDE>(vox_y);
  simd::uint32_v block_z = div<TreeVolume<T>::SUBVOLUME_SIDE>(vox_z);

  simd::uint32_v min, max;
  std::array<uint64_t, simd::len> block_handles;

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      const typename TreeVolume<T>::Node &node = volume.nodes[volume.info.node_handle(block_x[k], block_y[k], block_z[k], layer)];
      min[k] = node.min;
      max[k] = node.max;
      block_handles[k] = node.block_handle;
    }
  }

  simd::float_v samples = min;

  simd::float_m block_exists = (min != max) && mask;

  if (!block_exists.isEmpty()) {
    simd::float_v in_block_x = denorm_x - block_x * TreeVolume<T>::SUBVOLUME_SIDE;
    simd::float_v in_block_y = denorm_y - block_y * TreeVolume<T>::SUBVOLUME_SIDE;
    simd::float_v in_block_z = denorm_z - block_z * TreeVolume<T>::SUBVOLUME_SIDE;

    samples(block_exists) = sample(volume, block_handles, in_block_x, in_block_y, in_block_z, block_exists);
  }

  return samples;
}
