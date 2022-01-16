#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstddef>

#include <bit>
#include <algorithm>

template <typename T>
class RawVolume {
public:
  RawVolume() = default;

  RawVolume(const char *file_name, uint64_t width, uint64_t height, uint64_t depth):
      m_width(width),
      m_height(height),
      m_depth(depth),
      m_stride(width * height),
      m_size(width * height * depth * sizeof(T)) {

    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
      return;
    }

    m_data = static_cast<T *>(mmap(nullptr, m_size, PROT_READ, MAP_SHARED, fd, 0));

    close(fd);

    if (m_data == MAP_FAILED) {
      m_data = nullptr;
      return;
    }
  }

  ~RawVolume() {
    if (m_data) {
      munmap(m_data, m_size);
    }
  }

  const T *data() const { return m_data; }
  operator bool() const { return m_data; }

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
    if constexpr (std::endian::native == std::endian::big) {
      T value = m_data[i];
      uint8_t *bytes = reinterpret_cast<uint8_t *>(&value);
      std::reverse(bytes, bytes + sizeof(T));
      return value;
    }

    else if constexpr (std::endian::native == std::endian::little) {
      return m_data[i];
    }
  }

private:
  T *m_data = nullptr;

  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_depth;

  uint64_t m_stride;
  uint64_t m_size;
};
