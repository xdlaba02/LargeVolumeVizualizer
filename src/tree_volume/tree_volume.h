#pragma once

#include "../mapped_file.h"
#include "../endian.h"
#include "../simd.h"
#include "../morton.h"
#include "../preintegrated_transfer_function.h"
#include "../raster_traversal.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <cstddef>

#include <algorithm>

template <typename T>
class TreeVolume {
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

  TreeVolume(const char *blocks_file_name, const char *metadata_file_name, uint64_t width, uint64_t height, uint64_t depth):
      info(width, height, depth) {

    m_data_file.open(blocks_file_name, 0, info.size_in_blocks * BLOCK_BYTES, MappedFile::READ, MappedFile::SHARED);
    m_metadata_file.open(metadata_file_name, 0, info.size_in_blocks * sizeof(Node), MappedFile::READ, MappedFile::SHARED);

    if (!m_data_file) {
      throw std::runtime_error(std::string("Unable to map '") + blocks_file_name + "'!");
    }

    if (!m_metadata_file) {
      throw std::runtime_error(std::string("Unable to map '") + metadata_file_name + "'!");
    }

    blocks  = reinterpret_cast<const Block *>(m_data_file.data());
    nodes  = reinterpret_cast<const Node *>(m_metadata_file.data());
  }

public:
  const Info info;
  const Block *blocks;
  const Node *nodes;

private:
  MappedFile m_data_file;
  MappedFile m_metadata_file;
};
