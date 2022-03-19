#pragma once

#include "tree_volume.h"

#include <utils/morton.h>
#include <utils/utils.h>

struct Samplet {
  int32_t data[2][2][2];
  float frac[3];
};

// expects coordinates from interval <-.5f, TreeVolume<T>::SUBVOLUME_SIDE - .5f>
// can safely handle values from interval (-1.f, TreeVolume<T>::SUBVOLUME_SIDE) due to truncation used
template <typename T>
inline Samplet samplet(const TreeVolume<T> &volume, uint64_t block_handle, float denorm_x, float denorm_y, float denorm_z) {
  uint32_t voxs_x[2] { static_cast<uint32_t>(denorm_x), static_cast<uint32_t>(denorm_x + 1.f) };
  uint32_t voxs_y[2] { static_cast<uint32_t>(denorm_y), static_cast<uint32_t>(denorm_y + 1.f) };
  uint32_t voxs_z[2] { static_cast<uint32_t>(denorm_z), static_cast<uint32_t>(denorm_z + 1.f) };

  uint32_t voxs_x_interleaved[2] { Morton<TreeVolume<T>::BLOCK_BITS>::interleave(voxs_x[0]), Morton<TreeVolume<T>::BLOCK_BITS>::interleave(voxs_x[1]) };
  uint32_t voxs_y_interleaved[2] { Morton<TreeVolume<T>::BLOCK_BITS>::interleave(voxs_y[0]), Morton<TreeVolume<T>::BLOCK_BITS>::interleave(voxs_y[1]) };
  uint32_t voxs_z_interleaved[2] { Morton<TreeVolume<T>::BLOCK_BITS>::interleave(voxs_z[0]), Morton<TreeVolume<T>::BLOCK_BITS>::interleave(voxs_z[1]) };

  Samplet output;

  for (uint8_t z = 0; z < 2; z++) {
    for (uint8_t y = 0; y < 2; y++) {
      for (uint8_t x = 0; x < 2; x++) {
        output.data[z][y][x] = volume.blocks[block_handle][Morton<TreeVolume<T>::BLOCK_BITS>::combine_interleaved(voxs_x_interleaved[x], voxs_y_interleaved[y], voxs_z_interleaved[z])];
      }
    }
  }

  output.frac[0] = denorm_x - voxs_x[0];
  output.frac[1] = denorm_y - voxs_y[0];
  output.frac[2] = denorm_z - voxs_z[0];

  return output;
}

inline float linterp(const Samplet &samplet) {
  float accs[2][2];

  accs[0][0] = samplet.data[0][0][0] + (samplet.data[0][0][1] - samplet.data[0][0][0]) * samplet.frac[0];
  accs[0][1] = samplet.data[0][1][0] + (samplet.data[0][1][1] - samplet.data[0][1][0]) * samplet.frac[0];
  accs[1][0] = samplet.data[1][0][0] + (samplet.data[1][0][1] - samplet.data[1][0][0]) * samplet.frac[0];
  accs[1][1] = samplet.data[1][1][0] + (samplet.data[1][1][1] - samplet.data[1][1][0]) * samplet.frac[0];

  accs[0][0] += (accs[0][1] - accs[0][0]) * samplet.frac[1];
  accs[1][0] += (accs[1][1] - accs[1][0]) * samplet.frac[1];

  accs[0][0] += (accs[1][0] - accs[0][0]) * samplet.frac[2];

  return accs[0][0];
}

inline std::array<float, 3> gradient(const Samplet &samplet) {
  std::array<float, 3> grad {};

  int32_t diff[2][2][3];

  for (uint8_t y = 0; y < 2; y++) {
    for (uint8_t x = 0; x < 2; x++) {
      diff[y][x][0] = samplet.data[y][x][0] - samplet.data[y][x][1];
      diff[y][x][1] = samplet.data[y][0][x] - samplet.data[y][1][x];
      diff[y][x][2] = samplet.data[0][y][x] - samplet.data[1][y][x];
    }
  }

  static constinit uint8_t low_frac_idx[3] { 1, 0, 0 };
  static constinit uint8_t high_frac_idx[3] { 2, 2, 1 };

  for (uint8_t i = 0; i < 3; i++) {
    float acc0 = diff[0][1][i] + (diff[0][0][i] - diff[0][1][i]) * 0.5f;
    float acc1 = diff[1][1][i] + (diff[1][0][i] - diff[1][1][i]) * 0.5f;

    grad[i] = acc1 + (acc0 - acc1) * 0.5f;
  }

  return grad;
}

// expects coordinates from interval <-.5f, TreeVolume<T>::SUBVOLUME_SIDE - .5f>
// can safely handle values from interval (-1.f, TreeVolume<T>::SUBVOLUME_SIDE) due to truncation used
template <typename T>
inline float sample(const TreeVolume<T> &volume, uint64_t block_handle, float denorm_x, float denorm_y, float denorm_z) {
  return linterp(samplet(volume, block_handle, denorm_x, denorm_y, denorm_z));
}

// expects coordinates from interval <0, 1>
template <typename T>
inline float sample(const TreeVolume<T> &volume, float x, float y, float z, uint8_t layer) {
  uint8_t layer_index = std::size(volume.info.layers) - 1 - layer;

  float denorm_x = x * approx_exp2(layer_index) * float(TreeVolume<T>::SUBVOLUME_SIDE) - 0.5f;
  float denorm_y = y * approx_exp2(layer_index) * float(TreeVolume<T>::SUBVOLUME_SIDE) - 0.5f;
  float denorm_z = z * approx_exp2(layer_index) * float(TreeVolume<T>::SUBVOLUME_SIDE) - 0.5f;

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

    return linterp(samplet(volume, node.block_handle, in_block_x, in_block_y, in_block_z));
  }
};
