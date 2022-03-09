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
    struct Layer {
      Layer(uint32_t width_in_blocks, uint32_t height_in_blocks, uint32_t depth_in_blocks):
        width_in_blocks(width_in_blocks),
        height_in_blocks(height_in_blocks),
        depth_in_blocks(depth_in_blocks),
        stride_in_blocks(width_in_blocks * height_in_blocks),
        size_in_blocks(stride_in_blocks * depth_in_blocks) {}

      uint32_t width_in_blocks;
      uint32_t height_in_blocks;
      uint32_t depth_in_blocks;

      uint64_t stride_in_blocks;
      uint64_t size_in_blocks;

      inline uint64_t node_handle(uint32_t x, uint32_t y, uint32_t z) const {
        return z * stride_in_blocks + y * width_in_blocks + x;
      }
    };

    Info(uint32_t width, uint32_t height, uint32_t depth)
    {
        size_in_blocks = 0;

        uint32_t width_in_blocks  = (width  + SUBVOLUME_SIDE - 1) / SUBVOLUME_SIDE;
        uint32_t height_in_blocks = (height + SUBVOLUME_SIDE - 1) / SUBVOLUME_SIDE;
        uint32_t depth_in_blocks  = (depth  + SUBVOLUME_SIDE - 1) / SUBVOLUME_SIDE;

        while (width_in_blocks > 1 || height_in_blocks > 1 || depth_in_blocks > 1) {
          layers.emplace_back(width_in_blocks, height_in_blocks, depth_in_blocks);
          layer_offsets.push_back(size_in_blocks);
          size_in_blocks += layers.back().size_in_blocks;

          ++width_in_blocks  >>= 1;
          ++height_in_blocks >>= 1;
          ++depth_in_blocks  >>= 1;
        }

        layers.emplace_back(width, height, depth);
        layer_offsets.push_back(size_in_blocks);

        float octree_size = (1 << (std::size(layers) - 1)) * SUBVOLUME_SIDE;

        width_frac = width   / octree_size;
        height_frac = height / octree_size;
        depth_frac = depth   / octree_size;

        size_in_blocks += layers.back().size_in_blocks;
    }

    inline uint64_t node_handle(uint32_t x, uint32_t y, uint32_t z, uint8_t layer) const {
      return layer_offsets[layer] + layers[layer].node_handle(x, y, z);
    }

    std::vector<Layer> layers;
    std::vector<uint64_t> layer_offsets;

    float width_frac;
    float height_frac;
    float depth_frac;

    uint64_t size_in_blocks;
  };

  using Block = LE<T>[BLOCK_SIZE];

  struct __attribute__ ((packed)) Node {
    LE<uint64_t> block_handle;
    LE<T> min, max;
  };

  BlockedVolume(const char *blocks_file_name, const char *metadata_file_name, uint64_t width, uint64_t height, uint64_t depth):
      info(width, height, depth) {

    m_data_file.open(blocks_file_name, 0, info.size_in_blocks * BLOCK_BYTES, MappedFile::READ, MappedFile::SHARED);
    m_metadata_file.open(metadata_file_name, 0, info.size_in_blocks * sizeof(Node), MappedFile::READ, MappedFile::SHARED);

    if (!m_data_file || !m_metadata_file) {
      m_data_file.close();
      m_metadata_file.close();
      return;
    }

    blocks  = reinterpret_cast<const Block *>(m_data_file.data());
    nodes  = reinterpret_cast<const Node *>(m_metadata_file.data());
  }

  inline operator bool() const { return m_data_file && m_metadata_file; }

  // expects coordinates from interval <0, 1>
  inline float sample_volume(float x, float y, float z, uint8_t layer) const {
    float denorm_x = x * info.layers[layer].width  - 0.5f;
    float denorm_y = y * info.layers[layer].height - 0.5f;
    float denorm_z = z * info.layers[layer].depth  - 0.5f;

    uint32_t vox_x = denorm_x;
    uint32_t vox_y = denorm_y;
    uint32_t vox_z = denorm_z;

    uint32_t block_x = vox_x / SUBVOLUME_SIDE;
    uint32_t block_y = vox_y / SUBVOLUME_SIDE;
    uint32_t block_z = vox_z / SUBVOLUME_SIDE;

    const Node &node = nodes[info.node_handle(block_x, block_y, block_z, layer)];

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

  // expects coordinates from interval <0, 1>
  inline simd::float_v sample_volume(const simd::float_v &x, const simd::float_v &y, const simd::float_v &z, const simd::uint32_v &layer, const simd::float_m &mask) const {
    simd::uint32_v width;
    simd::uint32_v height;
    simd::uint32_v depth;

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask[k]) {
        width[k]  = info.layers[layer[k]].width;
        height[k] = info.layers[layer[k]].height;
        depth[k]  = info.layers[layer[k]].depth;
      }
    }

    simd::float_v denorm_x = x * width  - 0.5f;
    simd::float_v denorm_y = y * height - 0.5f;
    simd::float_v denorm_z = z * depth  - 0.5f;

    simd::uint32_v vox_x = denorm_x;
    simd::uint32_v vox_y = denorm_y;
    simd::uint32_v vox_z = denorm_z;

    simd::uint32_v block_x = simd::fast_div<SUBVOLUME_SIDE>(vox_x);
    simd::uint32_v block_y = simd::fast_div<SUBVOLUME_SIDE>(vox_y);
    simd::uint32_v block_z = simd::fast_div<SUBVOLUME_SIDE>(vox_z);

    simd::uint32_v min, max;
    std::array<uint64_t, simd::len> block_handles;

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask[k]) {
        const Node &node = nodes[info.node_handle(block_x[k], block_y[k], block_z[k], layer[k])];
        min[k] = node.min;
        max[k] = node.max;
        block_handles[k] = node.block_handle;
      }
    }

    simd::float_v samples = min;

    simd::float_m integrate = (min != max) & mask;

    if (!integrate.isEmpty()) {
      simd::float_v in_block_x = denorm_x - block_x * SUBVOLUME_SIDE;
      simd::float_v in_block_y = denorm_y - block_y * SUBVOLUME_SIDE;
      simd::float_v in_block_z = denorm_z - block_z * SUBVOLUME_SIDE;

      samples(integrate) = sample_block(block_handles, in_block_x, in_block_y, in_block_z, integrate);
    }

    return samples;
  }

  // expects coordinates from interval <-.5f, SUBVOLUME_SIDE - .5f>
  // can safely handle values from interval (-1.f, SUBVOLUME_SIDE) due to truncation used
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

    uint32_t denorm_x_low_interleaved = Morton<BLOCK_BITS>::interleave(denorm_x_low);
    uint32_t denorm_y_low_interleaved = Morton<BLOCK_BITS>::interleave(denorm_y_low);
    uint32_t denorm_z_low_interleaved = Morton<BLOCK_BITS>::interleave(denorm_z_low);

    uint32_t denorm_x_high_interleaved = Morton<BLOCK_BITS>::interleave(denorm_x_high);
    uint32_t denorm_y_high_interleaved = Morton<BLOCK_BITS>::interleave(denorm_y_high);
    uint32_t denorm_z_high_interleaved = Morton<BLOCK_BITS>::interleave(denorm_z_high);

    uint32_t morton_indices[2][2][2];

    morton_indices[0][0][0] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_low_interleaved);
    morton_indices[0][0][1] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_low_interleaved);
    morton_indices[0][1][0] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_low_interleaved);
    morton_indices[0][1][1] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_low_interleaved);
    morton_indices[1][0][0] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_high_interleaved);
    morton_indices[1][0][1] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_high_interleaved);
    morton_indices[1][1][0] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_high_interleaved);
    morton_indices[1][1][1] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_high_interleaved);

    int32_t buffers[2][2][2];

    buffers[0][0][0] = blocks[block_handle][morton_indices[0][0][0]];
    buffers[0][0][1] = blocks[block_handle][morton_indices[0][0][1]];
    buffers[0][1][0] = blocks[block_handle][morton_indices[0][1][0]];
    buffers[0][1][1] = blocks[block_handle][morton_indices[0][1][1]];
    buffers[1][0][0] = blocks[block_handle][morton_indices[1][0][0]];
    buffers[1][0][1] = blocks[block_handle][morton_indices[1][0][1]];
    buffers[1][1][0] = blocks[block_handle][morton_indices[1][1][0]];
    buffers[1][1][1] = blocks[block_handle][morton_indices[1][1][1]];

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

  // expects coordinates from interval <-.5f, SUBVOLUME_SIDE - .5f>
  // can safely handle values from interval (-1.f, SUBVOLUME_SIDE) due to padding and truncation used
  inline simd::float_v sample_block(const std::array<uint64_t, simd::len> &block_handle, const simd::float_v &denorm_x, const simd::float_v &denorm_y, const simd::float_v &denorm_z, const simd::float_m &mask) const {
    simd::uint32_v denorm_x_low = denorm_x;
    simd::uint32_v denorm_y_low = denorm_y;
    simd::uint32_v denorm_z_low = denorm_z;

    simd::uint32_v denorm_x_high = denorm_x + 1.f;
    simd::uint32_v denorm_y_high = denorm_y + 1.f;
    simd::uint32_v denorm_z_high = denorm_z + 1.f;

    simd::float_v frac_x = denorm_x - denorm_x_low;
    simd::float_v frac_y = denorm_y - denorm_y_low;
    simd::float_v frac_z = denorm_z - denorm_z_low;

    simd::uint32_v denorm_x_low_interleaved = Morton<BLOCK_BITS>::interleave(denorm_x_low);
    simd::uint32_v denorm_y_low_interleaved = Morton<BLOCK_BITS>::interleave(denorm_y_low);
    simd::uint32_v denorm_z_low_interleaved = Morton<BLOCK_BITS>::interleave(denorm_z_low);

    simd::uint32_v denorm_x_high_interleaved = Morton<BLOCK_BITS>::interleave(denorm_x_high);
    simd::uint32_v denorm_y_high_interleaved = Morton<BLOCK_BITS>::interleave(denorm_y_high);
    simd::uint32_v denorm_z_high_interleaved = Morton<BLOCK_BITS>::interleave(denorm_z_high);

    simd::uint32_v morton_indices[2][2][2];

    morton_indices[0][0][0] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_low_interleaved);
    morton_indices[0][0][1] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_low_interleaved);
    morton_indices[0][1][0] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_low_interleaved);
    morton_indices[0][1][1] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_low_interleaved);
    morton_indices[1][0][0] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_low_interleaved,  denorm_z_high_interleaved);
    morton_indices[1][0][1] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_low_interleaved,  denorm_z_high_interleaved);
    morton_indices[1][1][0] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_low_interleaved,  denorm_y_high_interleaved, denorm_z_high_interleaved);
    morton_indices[1][1][1] = Morton<BLOCK_BITS>::combine_interleaved(denorm_x_high_interleaved, denorm_y_high_interleaved, denorm_z_high_interleaved);

    simd::int32_v buffers[2][2][2];

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask[k]) {
        buffers[0][0][0][k] = blocks[block_handle[k]][morton_indices[0][0][0][k]];
        buffers[0][0][1][k] = blocks[block_handle[k]][morton_indices[0][0][1][k]];
        buffers[0][1][0][k] = blocks[block_handle[k]][morton_indices[0][1][0][k]];
        buffers[0][1][1][k] = blocks[block_handle[k]][morton_indices[0][1][1][k]];
        buffers[1][0][0][k] = blocks[block_handle[k]][morton_indices[1][0][0][k]];
        buffers[1][0][1][k] = blocks[block_handle[k]][morton_indices[1][0][1][k]];
        buffers[1][1][0][k] = blocks[block_handle[k]][morton_indices[1][1][0][k]];
        buffers[1][1][1][k] = blocks[block_handle[k]][morton_indices[1][1][1][k]];
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
  const Block *blocks;
  const Node *nodes;

private:
  MappedFile m_data_file;
  MappedFile m_metadata_file;
};
