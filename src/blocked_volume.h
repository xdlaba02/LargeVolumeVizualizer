#pragma once

#include "mapped_file.h"
#include "endian.h"
#include "simd.h"
#include "morton.h"
#include "preintegrated_transfer_function.h"
#include "raster_traversal.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <cstddef>

#include <algorithm>

template <typename T>
class BlockedVolume {
public:
  static constexpr uint32_t BLOCK_BITS = 4;
  static constexpr uint32_t BLOCK_SIDE = 1 << BLOCK_BITS;
  static constexpr uint32_t BLOCK_SIZE = BLOCK_SIDE * BLOCK_SIDE * BLOCK_SIDE;
  static constexpr uint32_t BLOCK_BYTES = BLOCK_SIZE * sizeof(T);

  static constexpr uint32_t SUBVOLUME_SIDE = BLOCK_SIDE - 1;
  static constexpr uint32_t SUBVOLUME_SIZE = SUBVOLUME_SIDE * SUBVOLUME_SIDE * SUBVOLUME_SIDE;

  struct Info {
    Info(uint64_t width, uint64_t height, uint64_t depth):
      width(width),
      height(height),
      depth(depth),
      width_in_blocks((width  + SUBVOLUME_SIDE - 1) / SUBVOLUME_SIDE),
      height_in_blocks((height + SUBVOLUME_SIDE - 1) / SUBVOLUME_SIDE),
      depth_in_blocks((depth  + SUBVOLUME_SIDE - 1) / SUBVOLUME_SIDE),
      stride_in_blocks(width_in_blocks * height_in_blocks),
      size_in_blocks(stride_in_blocks * depth_in_blocks),
      size_in_bytes(size_in_blocks * BLOCK_BYTES) {}

    const uint32_t width;
    const uint32_t height;
    const uint32_t depth;

    const uint32_t width_in_blocks;
    const uint32_t height_in_blocks;
    const uint32_t depth_in_blocks;

    const uint64_t stride_in_blocks;
    const uint64_t size_in_blocks;

    const uint64_t size_in_bytes;
  };

  using Block = LE<T>[BLOCK_SIZE];

  struct Node {
    T min, max;
    uint64_t block_handle;
  };

  BlockedVolume(const char *blocks_file_name, const char *metadata_file_name, uint64_t width, uint64_t height, uint64_t depth):
      info(width, height, depth) {

    m_data_file.open(blocks_file_name, 0, info.size_in_bytes, MappedFile::READ, MappedFile::SHARED);

    if (!m_data_file) {
      return;
    }

    m_metadata_file.open(metadata_file_name, 0, info.size_in_blocks * 2 + info.size_in_blocks * sizeof(uint64_t), MappedFile::READ, MappedFile::SHARED);

    if (!m_metadata_file) {
      m_data_file.close();
      return;
    }

    m_blocks  = reinterpret_cast<const Block *>(m_data_file.data());
    m_mins    = reinterpret_cast<const LE<T> *>(m_metadata_file.data());
    m_maxs    = m_mins + info.size_in_blocks;
    m_block_handles = reinterpret_cast<const LE<uint64_t> *>(m_maxs + info.size_in_blocks);
  }

  inline operator bool() const { return m_data_file && m_metadata_file; }

  inline Node node(uint32_t x, uint32_t y, uint32_t z) const {
    uint64_t i = z * info.stride_in_blocks + y * info.width_in_blocks + x;
    return { m_mins[i], m_maxs[i], m_block_handles[i] };
  }

  inline float sample_volume(float denorm_x, float denorm_y, float denorm_z) const {
    uint32_t vox_x = denorm_x;
    uint32_t vox_y = denorm_y;
    uint32_t vox_z = denorm_z;

    uint32_t block_x = vox_x / SUBVOLUME_SIDE;
    uint32_t block_y = vox_y / SUBVOLUME_SIDE;
    uint32_t block_z = vox_z / SUBVOLUME_SIDE;

    Node node = this->node(block_x, block_y, block_z);

    if (node.min == node.max) {
      return node.min;
    }
    else {
      // reminder from division
      float in_block_x = denorm_x - block_x * SUBVOLUME_SIDE;
      float in_block_y = denorm_y - block_y * SUBVOLUME_SIDE;
      float in_block_z = denorm_z - block_z * SUBVOLUME_SIDE;

      return sample_block(node.block_handle, in_block_x, in_block_y, in_block_z);
    }
  };

  inline simd::float_v sample_volume(const simd::float_v &denorm_x, const simd::float_v &denorm_y, const simd::float_v &denorm_z, const simd::float_m &mask) const {
    simd::uint32_v vox_x = denorm_x;
    simd::uint32_v vox_y = denorm_y;
    simd::uint32_v vox_z = denorm_z;

    simd::uint32_v block_x = simd::fast_div<SUBVOLUME_SIDE>(vox_x);
    simd::uint32_v block_y = simd::fast_div<SUBVOLUME_SIDE>(vox_y);
    simd::uint32_v block_z = simd::fast_div<SUBVOLUME_SIDE>(vox_z);

    simd::float_v samples;

    simd::uint32_v min, max;
    std::array<uint64_t, simd::len> block_handles;

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask[k]) {
        Node node = node(block_x[k], block_y[k], block_z[k]);
        min[k] = node.min;
        max[k] = node.max;
        block_handles[k] = node.block_handle;
      }
    }

    simd::float_m same = min == max;
    simd::float_m different = !same;

    if (!same.isEmpty()) {
      samples(same) = min;
    }

    if (!different.isEmpty()) {
      simd::float_v in_block_x = denorm_x - block_x * SUBVOLUME_SIDE;
      simd::float_v in_block_y = denorm_y - block_y * SUBVOLUME_SIDE;
      simd::float_v in_block_z = denorm_z - block_z * SUBVOLUME_SIDE;

      samples(different) = sample_block(block_handles, in_block_x, in_block_y, in_block_z, mask & different);
    }

    return samples;
  }

  inline float sample_block(uint64_t block_handle, float denorm_x, float denorm_y, float denorm_z) const {
    uint32_t denorm_x_low = denorm_x;
    uint32_t denorm_y_low = denorm_y;
    uint32_t denorm_z_low = denorm_z;

    uint32_t denorm_x_high = denorm_x + 1.f;
    uint32_t denorm_y_high = denorm_y + 1.f;
    uint32_t denorm_z_high = denorm_z + 1.f;

    float frac_x = denorm_x - denorm_x_low;
    float frac_y = denorm_y - denorm_y_low;
    float frac_z = denorm_z - denorm_z_low;

    uint32_t denorm_x_low_interleaved = morton::interleave_4b_3d(denorm_x_low);
    uint32_t denorm_y_low_interleaved = morton::interleave_4b_3d(denorm_y_low);
    uint32_t denorm_z_low_interleaved = morton::interleave_4b_3d(denorm_z_low);

    uint32_t denorm_x_high_interleaved = morton::interleave_4b_3d(denorm_x_high);
    uint32_t denorm_y_high_interleaved = morton::interleave_4b_3d(denorm_y_high);
    uint32_t denorm_z_high_interleaved = morton::interleave_4b_3d(denorm_z_high);

    uint32_t morton_indices[2][2][2];

    morton_indices[0][0][0] = morton::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_low_interleaved);
    morton_indices[0][0][1] = morton::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_low_interleaved);
    morton_indices[0][1][0] = morton::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_low_interleaved);
    morton_indices[0][1][1] = morton::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_low_interleaved);
    morton_indices[1][0][0] = morton::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_high_interleaved);
    morton_indices[1][0][1] = morton::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_high_interleaved);
    morton_indices[1][1][0] = morton::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_high_interleaved);
    morton_indices[1][1][1] = morton::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_high_interleaved);

    int32_t buffers[2][2][2];

    buffers[0][0][0] = m_blocks[block_handle][morton_indices[0][0][0]];
    buffers[0][0][1] = m_blocks[block_handle][morton_indices[0][0][1]];
    buffers[0][1][0] = m_blocks[block_handle][morton_indices[0][1][0]];
    buffers[0][1][1] = m_blocks[block_handle][morton_indices[0][1][1]];
    buffers[1][0][0] = m_blocks[block_handle][morton_indices[1][0][0]];
    buffers[1][0][1] = m_blocks[block_handle][morton_indices[1][0][1]];
    buffers[1][1][0] = m_blocks[block_handle][morton_indices[1][1][0]];
    buffers[1][1][1] = m_blocks[block_handle][morton_indices[1][1][1]];

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

  inline simd::float_v sample_block(const std::array<uint64_t, simd::len> &block_handle, const simd::float_v &denorm_x, const float &denorm_y, const float &denorm_z, const simd::float_m &mask) const {
    simd::uint32_v denorm_x_low = denorm_x;
    simd::uint32_v denorm_y_low = denorm_y;
    simd::uint32_v denorm_z_low = denorm_z;

    simd::uint32_v denorm_x_high = denorm_x + 1.f;
    simd::uint32_v denorm_y_high = denorm_y + 1.f;
    simd::uint32_v denorm_z_high = denorm_z + 1.f;

    simd::float_v frac_x = denorm_x - denorm_x_low;
    simd::float_v frac_y = denorm_y - denorm_y_low;
    simd::float_v frac_z = denorm_z - denorm_z_low;

    simd::uint32_v denorm_x_low_interleaved = morton::interleave_4b_3d(denorm_x_low);
    simd::uint32_v denorm_y_low_interleaved = morton::interleave_4b_3d(denorm_y_low);
    simd::uint32_v denorm_z_low_interleaved = morton::interleave_4b_3d(denorm_z_low);

    simd::uint32_v denorm_x_high_interleaved = morton::interleave_4b_3d(denorm_x_high);
    simd::uint32_v denorm_y_high_interleaved = morton::interleave_4b_3d(denorm_y_high);
    simd::uint32_v denorm_z_high_interleaved = morton::interleave_4b_3d(denorm_z_high);

    simd::uint32_v morton_indices[2][2][2];

    morton_indices[0][0][0] = morton::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_low_interleaved);
    morton_indices[0][0][1] = morton::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_low_interleaved);
    morton_indices[0][1][0] = morton::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_low_interleaved);
    morton_indices[0][1][1] = morton::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_low_interleaved);
    morton_indices[1][0][0] = morton::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_high_interleaved);
    morton_indices[1][0][1] = morton::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_high_interleaved);
    morton_indices[1][1][0] = morton::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_high_interleaved);
    morton_indices[1][1][1] = morton::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_high_interleaved);

    simd::int32_v buffers[2][2][2];

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask[k]) {
        buffers[0][0][0][k] = m_blocks[block_handle[k]][morton_indices[0][0][0][k]];
        buffers[0][0][1][k] = m_blocks[block_handle[k]][morton_indices[0][0][1][k]];
        buffers[0][1][0][k] = m_blocks[block_handle[k]][morton_indices[0][1][0][k]];
        buffers[0][1][1][k] = m_blocks[block_handle[k]][morton_indices[0][1][1][k]];
        buffers[1][0][0][k] = m_blocks[block_handle[k]][morton_indices[1][0][0][k]];
        buffers[1][0][1][k] = m_blocks[block_handle[k]][morton_indices[1][0][1][k]];
        buffers[1][1][0][k] = m_blocks[block_handle[k]][morton_indices[1][1][0][k]];
        buffers[1][1][1][k] = m_blocks[block_handle[k]][morton_indices[1][1][1][k]];
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

public:
  const Info info;

private:
  MappedFile m_data_file;
  MappedFile m_metadata_file;

  const Block *m_blocks;
  const LE<T> *m_mins;
  const LE<T> *m_maxs;
  const LE<uint64_t> *m_block_handles;
};
