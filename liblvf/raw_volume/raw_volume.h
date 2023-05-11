/**
* @file raw_volume.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Class that maps raw volume file into the virtual memory.
*/

#pragma once

#include <utils/mapped_file.h>
#include <utils/endian.h>

#include <cstdint>

#include <stdexcept>

template <typename T>
class RawVolume {
public:
  RawVolume(const char *file_name, uint32_t width, uint32_t height, uint32_t depth):
      width(width),
      height(height),
      depth(depth),
      stride(static_cast<uint64_t>(width) * height),
      size(stride * depth),
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
