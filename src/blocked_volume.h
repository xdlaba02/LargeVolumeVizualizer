#pragma once

#include "mapped_file.h"
#include "endian.h"

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
    T min;
    T max;
    const Block &block;
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
    m_offsets = reinterpret_cast<const LE<uint64_t> *>(m_maxs + info.size_in_blocks);
  }

  operator bool() const { return m_data_file && m_metadata_file; }

  Node node(uint32_t x, uint32_t y, uint32_t z) const {
    uint64_t index = z * info.stride_in_blocks + y * info.width_in_blocks + x;
    return { m_mins[index], m_maxs[index], m_blocks[m_offsets[index]] };
  }

  const Info info;

private:
  MappedFile m_data_file;
  MappedFile m_metadata_file;

  const Block *m_blocks;
  const LE<T> *m_mins;
  const LE<T> *m_maxs;
  const LE<uint64_t> *m_offsets;
};
