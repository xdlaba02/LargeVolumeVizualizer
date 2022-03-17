#pragma once

#include "../mapped_file.h"
#include "../endian.h"

#include <cstdint>

#include <stdexcept>

template <typename T>
class RawVolume {
public:
  RawVolume(const char *file_name, uint64_t width, uint64_t height, uint64_t depth):
      width(width),
      height(height),
      depth(depth),
      stride(width * height),
      size(width * height * depth),
      m_file(file_name, 0, size * sizeof(T), MappedFile::READ, MappedFile::SHARED)
  {
    if (!m_file) {
      throw std::runtime_error(std::string("Unable to open '") + file_name + "'!");
    }

    data = reinterpret_cast<const LE<T> *>(m_file.data());
  }

  uint64_t voxel_handle(uint32_t x, uint32_t y, uint32_t z) const {
    return z * stride + y * width + x;
  }

public:
  const uint32_t width;
  const uint32_t height;
  const uint32_t depth;
  const uint64_t stride;
  const uint64_t size;

  const LE<T> *data;

private:

  MappedFile m_file;
};
