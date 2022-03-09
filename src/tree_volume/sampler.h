#pragma once

#include "tree_volume.h"

// expects coordinates from interval <-.5f, TreeVolume<T>::SUBVOLUME_SIDE - .5f>
// can safely handle values from interval (-1.f, TreeVolume<T>::SUBVOLUME_SIDE) due to truncation used
template <typename T>
inline float sample_block(const TreeVolume<T> &volume, uint64_t block_handle, float denorm_x, float denorm_y, float denorm_z) {
  uint32_t denorm_x_low = denorm_x;
  uint32_t denorm_y_low = denorm_y;
  uint32_t denorm_z_low = denorm_z;

  uint32_t denorm_x_high = denorm_x + 1.f;
  uint32_t denorm_y_high = denorm_y + 1.f;
  uint32_t denorm_z_high = denorm_z + 1.f;

  float frac_x = denorm_x - denorm_x_low;
  float frac_y = denorm_y - denorm_y_low;
  float frac_z = denorm_z - denorm_z_low;

  uint32_t denorm_x_low_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_x_low);
  uint32_t denorm_y_low_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_y_low);
  uint32_t denorm_z_low_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_z_low);

  uint32_t denorm_x_high_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_x_high);
  uint32_t denorm_y_high_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_y_high);
  uint32_t denorm_z_high_interleaved = Morton<TreeVolume<T>::BLOCK_BITS>::interleave(denorm_z_high);

  uint32_t morton_indices[2][2][2];

  morton_indices[0][0][0] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_low_interleaved);
  morton_indices[0][0][1] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_low_interleaved);
  morton_indices[0][1][0] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_low_interleaved);
  morton_indices[0][1][1] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_low_interleaved);
  morton_indices[1][0][0] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_high_interleaved);
  morton_indices[1][0][1] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_high_interleaved);
  morton_indices[1][1][0] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_high_interleaved);
  morton_indices[1][1][1] = Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_high_interleaved);

  int32_t buffers[2][2][2];

  buffers[0][0][0] = volume.blocks[block_handle][morton_indices[0][0][0]];
  buffers[0][0][1] = volume.blocks[block_handle][morton_indices[0][0][1]];
  buffers[0][1][0] = volume.blocks[block_handle][morton_indices[0][1][0]];
  buffers[0][1][1] = volume.blocks[block_handle][morton_indices[0][1][1]];
  buffers[1][0][0] = volume.blocks[block_handle][morton_indices[1][0][0]];
  buffers[1][0][1] = volume.blocks[block_handle][morton_indices[1][0][1]];
  buffers[1][1][0] = volume.blocks[block_handle][morton_indices[1][1][0]];
  buffers[1][1][1] = volume.blocks[block_handle][morton_indices[1][1][1]];

  float accs[2][2];

  accs[0][0] = buffers[0][0][0] + (buffers[0][0][1] - buffers[0][0][0]) * frac_x;
  accs[0][1] = buffers[0][1][0] + (buffers[0][1][1] - buffers[0][1][0]) * frac_x;
  accs[1][0] = buffers[1][0][0] + (buffers[1][0][1] - buffers[1][0][0]) * frac_x;
  accs[1][1] = buffers[1][1][0] + (buffers[1][1][1] - buffers[1][1][0]) * frac_x;

  accs[0][0] += (accs[0][1] - accs[0][0]) * frac_y;
  accs[1][0] += (accs[1][1] - accs[1][0]) * frac_y;

  accs[0][0] += (accs[1][0] - accs[0][0]) * frac_z;

  return accs[0][0];
}

// expects coordinates from interval <-.5f, TreeVolume<T>::SUBVOLUME_SIDE - .5f>
// can safely handle values from interval (-1.f, TreeVolume<T>::SUBVOLUME_SIDE) due to padding and truncation used
template <typename T>
inline simd::float_v sample_block(const TreeVolume<T> &volume, const std::array<uint64_t, simd::len> &block_handle, const simd::float_v &denorm_x, const simd::float_v &denorm_y, const simd::float_v &denorm_z, const simd::float_m &mask) {
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
inline float sample_volume(const TreeVolume<T> &volume, float x, float y, float z, uint8_t layer) {
  float denorm_x = x * volume.info.layers[layer].width  - 0.5f;
  float denorm_y = y * volume.info.layers[layer].height - 0.5f;
  float denorm_z = z * volume.info.layers[layer].depth  - 0.5f;

  uint32_t vox_x = denorm_x;
  uint32_t vox_y = denorm_y;
  uint32_t vox_z = denorm_z;

  uint32_t block_x = vox_x / TreeVolume<T>::SUBVOLUME_SIDE;
  uint32_t block_y = vox_y / TreeVolume<T>::SUBVOLUME_SIDE;
  uint32_t block_z = vox_z / TreeVolume<T>::SUBVOLUME_SIDE;

  const typename TreeVolume<T>::Node &node = volume.nodes[volume.info.node_handle(block_x, block_y, block_z, layer)];

  if (node.min == node.max) {
    return node.min;
  }
  else {
    // reminder from division
    float in_block_x = denorm_x - block_x * TreeVolume<T>::SUBVOLUME_SIDE;
    float in_block_y = denorm_y - block_y * TreeVolume<T>::SUBVOLUME_SIDE;
    float in_block_z = denorm_z - block_z * TreeVolume<T>::SUBVOLUME_SIDE;

    return sample_block(node.block_handle, in_block_x, in_block_y, in_block_z);
  }
};

// expects coordinates from interval <0, 1>
template <typename T>
inline simd::float_v sample_volume(const TreeVolume<T> &volume, const simd::float_v &x, const simd::float_v &y, const simd::float_v &z, const simd::uint32_v &layer, const simd::float_m &mask) {
  simd::uint32_v width;
  simd::uint32_v height;
  simd::uint32_v depth;

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      width[k]  = volume.info.layers[layer[k]].width;
      height[k] = volume.info.layers[layer[k]].height;
      depth[k]  = volume.info.layers[layer[k]].depth;
    }
  }

  simd::float_v denorm_x = x * width  - 0.5f;
  simd::float_v denorm_y = y * height - 0.5f;
  simd::float_v denorm_z = z * depth  - 0.5f;

  simd::uint32_v vox_x = denorm_x;
  simd::uint32_v vox_y = denorm_y;
  simd::uint32_v vox_z = denorm_z;

  simd::uint32_v block_x = simd::fast_div<TreeVolume<T>::SUBVOLUME_SIDE>(vox_x);
  simd::uint32_v block_y = simd::fast_div<TreeVolume<T>::SUBVOLUME_SIDE>(vox_y);
  simd::uint32_v block_z = simd::fast_div<TreeVolume<T>::SUBVOLUME_SIDE>(vox_z);

  simd::uint32_v min, max;
  std::array<uint64_t, simd::len> block_handles;

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      const typename TreeVolume<T>::Node &node = volume.nodes[volume.info.node_handle(block_x[k], block_y[k], block_z[k], layer[k])];
      min[k] = node.min;
      max[k] = node.max;
      block_handles[k] = node.block_handle;
    }
  }

  simd::float_v samples = min;

  simd::float_m integrate = (min != max) & mask;

  if (!integrate.isEmpty()) {
    simd::float_v in_block_x = denorm_x - block_x * TreeVolume<T>::SUBVOLUME_SIDE;
    simd::float_v in_block_y = denorm_y - block_y * TreeVolume<T>::SUBVOLUME_SIDE;
    simd::float_v in_block_z = denorm_z - block_z * TreeVolume<T>::SUBVOLUME_SIDE;

    samples(integrate) = sample_block(volume, block_handles, in_block_x, in_block_y, in_block_z, integrate);
  }

  return samples;
}
