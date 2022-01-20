#pragma once

#include "mapped_file.h"
#include "byteswap.h"

#include <cstdint>
#include <cstddef>

#include <bit>
#include <algorithm>

template <typename T>
using Block = T[4096];

template <typename T>
class BlockedVolume {
public:
  BlockedVolume(const char *blocks_file_name, const char *metadata_file_name, uint64_t width, uint64_t height, uint64_t depth) {
    m_width = width;
    m_height = height;
    m_depth = depth;

    m_width_in_blocks  = (m_width  + 14) / 15;
    m_height_in_blocks = (m_height + 14) / 15;
    m_depth_in_blocks  = (m_depth  + 14) / 15;

    m_stride_in_blocks = m_width_in_blocks * m_height_in_blocks;
    m_size_in_blocks = m_stride_in_blocks * m_depth_in_blocks;

    m_size_in_bytes = m_size_in_blocks * 16 * 16 * 16 * sizeof(T);

    m_data_file.open(blocks_file_name, 0, m_size_in_bytes, MappedFile::READ, MappedFile::SHARED);

    if (!m_data_file) {
      return;
    }

    m_metadata_file.open(metadata_file_name, 0, m_size_in_blocks * 2 + m_size_in_blocks * sizeof(uint64_t), MappedFile::READ, MappedFile::SHARED);

    if (!m_metadata_file) {
      m_data_file.close();
      return;
    }

    m_blocks  = reinterpret_cast<const Block<T> *>(m_data_file.data());
    m_mins    = reinterpret_cast<const T *>(m_metadata_file.data());
    m_maxs    = m_mins + m_size_in_blocks;
    m_offsets = reinterpret_cast<const uint64_t *>(m_maxs + m_size_in_blocks);
  }

  operator bool() const { return m_data_file && m_metadata_file; }

  T min(uint64_t i) const {
    if constexpr (std::endian::native == std::endian::big) {
      return byteswap(m_mins[i]);
    }
    else {
      return m_mins[i];
    }
  }

  T max(uint64_t i) const {
    if constexpr (std::endian::native == std::endian::big) {
      return byteswap(m_maxs[i]);
    }
    else {
      return m_maxs[i];
    }
  }

  uint64_t offset(uint64_t i) const {
    if constexpr (std::endian::native == std::endian::big) {
      return byteswap(m_offsets[i]);
    }
    else {
      return m_offsets[i];
    }
  }

  const Block<T> &block(uint64_t offset) const { return m_blocks[offset]; }

  constexpr uint32_t width()  const { return m_width; }
  constexpr uint32_t height() const { return m_height; }
  constexpr uint32_t depth()  const { return m_depth; }

  constexpr uint32_t width_in_blocks()  const { return m_width_in_blocks; }
  constexpr uint32_t height_in_blocks() const { return m_height_in_blocks; }
  constexpr uint32_t depth_in_blocks()  const { return m_depth_in_blocks; }

  constexpr uint32_t stride_in_blocks()  const { return m_stride_in_blocks; }
  constexpr uint32_t size_in_blocks() const { return m_size_in_blocks; }

private:
  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_depth;

  uint32_t m_width_in_blocks;
  uint32_t m_height_in_blocks;
  uint32_t m_depth_in_blocks;

  uint64_t m_stride_in_blocks;
  uint64_t m_size_in_blocks;

  uint64_t m_size_in_bytes;

  MappedFile m_data_file;
  MappedFile m_metadata_file;

  const Block<T> *m_blocks;
  const T *m_mins;
  const T *m_maxs;
  const uint64_t *m_offsets;
};
