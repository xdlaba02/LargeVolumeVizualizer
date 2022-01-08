#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstddef>

class RawVolume {
public:
  RawVolume() = default;

  RawVolume(const char *file_name, uint32_t width, uint32_t height, uint32_t depth, uint32_t bytes_per_voxel = 1):
      m_width(width),
      m_height(height),
      m_depth(depth),
      m_bytes_per_voxel(bytes_per_voxel) {

    if (bytes_per_voxel != 1 && bytes_per_voxel != 2) {
      return;
    }

    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
      return;
    }

    m_data = mmap(nullptr, width * height * depth * bytes_per_voxel, PROT_READ, MAP_SHARED, fd, 0);

    close(fd);

    if (m_data == MAP_FAILED) {
      m_data = nullptr;
      return;
    }
  }

  ~RawVolume() {
    if (m_data) {
      munmap(m_data, m_width * m_height * m_depth * m_bytes_per_voxel);
    }
  }

  operator void *() const { return m_data; }

  uint32_t width()  const { return m_width; }
  uint32_t height() const { return m_height; }
  uint32_t depth()  const { return m_depth; }

  uint16_t operator()(uint32_t x, uint32_t y, uint32_t z) const {
    switch (m_bytes_per_voxel) {
      case 1:
        return static_cast<uint8_t *>(m_data)[(z * m_height + y) * m_width + x];
      case 2:
        return static_cast<uint16_t *>(m_data)[(z * m_height + y) * m_width + x];
      default:
        return 0;
    }
  }

private:
  void  *m_data = nullptr;
  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_depth;
  uint32_t m_bytes_per_voxel;
};
