#pragma once

#include "mapped_file.h"
#include "byteswap.h"

#include <cstdint>
#include <cstddef>

#include <bit>
#include <algorithm>

template <typename T>
class RawVolume {
public:
  RawVolume(const char *file_name, uint64_t width, uint64_t height, uint64_t depth):
      m_width(width),
      m_height(height),
      m_depth(depth),
      m_stride(width * height),
      m_size_bytes(width * height * depth * sizeof(T)),
      m_file(file_name, 0, m_size_bytes, MappedFile::READ, MappedFile::SHARED)
  {}

  const T *data() const { return m_file.data(); }
  operator bool() const { return m_file; }

  constexpr uint32_t width()  const { return m_width; }
  constexpr uint32_t height() const { return m_height; }
  constexpr uint32_t depth()  const { return m_depth; }

  constexpr uint32_t xStride() const { return 1; }
  constexpr uint32_t yStride() const { return m_width; }
  constexpr uint32_t zStride() const { return m_stride; }

  T operator()(uint32_t x, uint32_t y, uint32_t z) const {
    return (*this)[z * m_stride + y * m_width + x];
  }

  T operator[](uint64_t i) const {
    T value = static_cast<const T *>(m_file.data())[i];

    if constexpr (std::endian::native == std::endian::big) {
      return byteswap(value);
    }
    else {
      return value;
    }
  }

private:
  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_depth;

  uint64_t m_stride;
  uint64_t m_size_bytes;

  MappedFile m_file;
};
