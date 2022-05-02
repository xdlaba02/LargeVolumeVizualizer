/**
* @file tree_volume.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Class that maps tree volume files into the virtual memory.
* The class provides structure that describes memory layout of the metadata file.
* The metadata file describes layout of the data file.
*/

#pragma once

#include <utils/mapped_file.h>
#include <utils/endian.h>

#include <cstdint>
#include <cstddef>

#include <algorithm>
#include <filesystem>

template <typename T, uint32_t N>
class TreeVolume {
public:
  static constexpr uint32_t BLOCK_BITS = N;
  static constexpr uint32_t BLOCK_SIDE = 1 << BLOCK_BITS;
  static constexpr uint32_t BLOCK_SIZE = BLOCK_SIDE * BLOCK_SIDE * BLOCK_SIDE;
  static constexpr uint32_t BLOCK_BYTES = BLOCK_SIZE * sizeof(T);

  static constexpr uint32_t SUBVOLUME_SIDE = BLOCK_SIDE - 1;
  static constexpr uint32_t SUBVOLUME_SIZE = SUBVOLUME_SIDE * SUBVOLUME_SIDE * SUBVOLUME_SIDE;

  struct Info {
    struct Layer {

      Layer(uint32_t width_in_nodes, uint32_t height_in_nodes, uint32_t depth_in_nodes):
        width_in_nodes(width_in_nodes),
        height_in_nodes(height_in_nodes),
        depth_in_nodes(depth_in_nodes),
        stride_in_nodes(static_cast<uint64_t>(width_in_nodes) * height_in_nodes),
        size_in_nodes(stride_in_nodes * depth_in_nodes) {}

      uint32_t width_in_nodes;
      uint32_t height_in_nodes;
      uint32_t depth_in_nodes;

      uint64_t stride_in_nodes;
      uint64_t size_in_nodes;

      inline uint64_t node_handle(uint32_t x, uint32_t y, uint32_t z) const {
        return z * stride_in_nodes + y * width_in_nodes + x;
      }
    };

    Info(uint32_t width, uint32_t height, uint32_t depth)
    {
        size_in_nodes = 0;

        uint32_t width_in_nodes  = (width  + SUBVOLUME_SIDE - 1) / SUBVOLUME_SIDE;
        uint32_t height_in_nodes = (height + SUBVOLUME_SIDE - 1) / SUBVOLUME_SIDE;
        uint32_t depth_in_nodes  = (depth  + SUBVOLUME_SIDE - 1) / SUBVOLUME_SIDE;

        while (width_in_nodes > 1 || height_in_nodes > 1 || depth_in_nodes > 1) {
          layers.emplace_back(width_in_nodes, height_in_nodes, depth_in_nodes);
          layer_offsets.push_back(size_in_nodes);
          size_in_nodes += layers.back().size_in_nodes;

          ++width_in_nodes  >>= 1;
          ++height_in_nodes >>= 1;
          ++depth_in_nodes  >>= 1;
        }

        layers.emplace_back(width_in_nodes, height_in_nodes, depth_in_nodes);
        layer_offsets.push_back(size_in_nodes);

        float octree_size = (1 << (std::size(layers) - 1)) * SUBVOLUME_SIDE;

        width_frac = width   / octree_size;
        height_frac = height / octree_size;
        depth_frac = depth   / octree_size;

        size_in_nodes += layers.back().size_in_nodes;
    }

    inline uint64_t node_handle(uint32_t x, uint32_t y, uint32_t z, uint8_t layer) const {
      return layer_offsets[layer] + layers[layer].node_handle(x, y, z);
    }

    std::vector<Layer> layers;
    std::vector<uint64_t> layer_offsets;

    float width_frac;
    float height_frac;
    float depth_frac;

    uint64_t size_in_nodes;
  };

  using Block = LE<T>[BLOCK_SIZE];

  struct __attribute__ ((packed)) Node {
    LE<uint64_t> block_handle;
    LE<T> min, max;
  };

  TreeVolume(const char *blocks_file_name, const char *metadata_file_name, uint32_t width, uint32_t height, uint32_t depth):
      info(width, height, depth) {

    size_t block_size = std::filesystem::file_size(blocks_file_name);
    size_t metadata_size = std::filesystem::file_size(metadata_file_name);

    m_data_file.open(blocks_file_name, 0, block_size, MappedFile::READ, MappedFile::SHARED);
    m_metadata_file.open(metadata_file_name, 0, metadata_size, MappedFile::READ, MappedFile::SHARED);

    if (!m_data_file) {
      throw std::runtime_error(std::string("Unable to map '") + blocks_file_name + "'!");
    }

    if (!m_metadata_file) {
      throw std::runtime_error(std::string("Unable to map '") + metadata_file_name + "'!");
    }

    m_blocks = reinterpret_cast<const Block *>(m_data_file.data());
    m_nodes  = reinterpret_cast<const Node *>(m_metadata_file.data());
  }

  inline const Node &node(uint64_t node_handle) const { return m_nodes[node_handle]; }
  inline const Block &block(uint64_t block_handle) const { return m_blocks[block_handle]; }

  const Info info;

private:
  const Block *m_blocks;
  const Node *m_nodes;

  MappedFile m_data_file;
  MappedFile m_metadata_file;
};
