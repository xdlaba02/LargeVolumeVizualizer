#pragma once

#include "mapped_file.h"
#include "endian.h"
#include "simd.h"

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
  {
    m_data  = reinterpret_cast<const LE<T> *>(m_file.data());
  }

  operator bool() const { return m_file; }

  simd::float_v samples(simd::float_v xs, simd::float_v ys, simd::float_v zs, simd::float_m mask) {
    simd::uint32_v pix_xs = xs;
    simd::uint32_v pix_ys = ys;
    simd::uint32_v pix_zs = zs;

    simd::float_v frac_xs = xs - pix_xs;
    simd::float_v frac_ys = ys - pix_ys;
    simd::float_v frac_zs = zs - pix_zs;

    simd::float_m incrementable_xs = pix_xs < (m_width - 1);
    simd::float_m incrementable_ys = pix_ys < (m_height - 1);
    simd::float_m incrementable_zs = pix_zs < (m_depth - 1);

    simd::int32_v buffers[2][2][2];

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask[k]) {
        uint64_t base = pix_zs[k] * m_stride + pix_ys[k] * m_width + pix_xs[k];

        uint64_t x_offset = incrementable_xs[k] * 1;
        uint64_t y_offset = incrementable_ys[k] * m_width;
        uint64_t z_offset = incrementable_zs[k] * m_stride;

        buffers[0][0][0][k] = m_data[base];
        buffers[0][0][1][k] = m_data[base + x_offset];
        buffers[0][1][0][k] = m_data[base + y_offset];
        buffers[0][1][1][k] = m_data[base + y_offset + x_offset];
        buffers[1][0][0][k] = m_data[base + z_offset];
        buffers[1][0][1][k] = m_data[base + z_offset + x_offset];
        buffers[1][1][0][k] = m_data[base + z_offset + y_offset];
        buffers[1][1][1][k] = m_data[base + z_offset + y_offset + x_offset];
      }
    }

    simd::float_v accs[2][2];

    accs[0][0] = buffers[0][0][0] + (buffers[0][0][1] - buffers[0][0][0]) * frac_xs;
    accs[0][1] = buffers[0][1][0] + (buffers[0][1][1] - buffers[0][1][0]) * frac_xs;
    accs[1][0] = buffers[1][0][0] + (buffers[1][0][1] - buffers[1][0][0]) * frac_xs;
    accs[1][1] = buffers[1][1][0] + (buffers[1][1][1] - buffers[1][1][0]) * frac_xs;

    accs[0][0] += (accs[0][1] - accs[0][0]) * frac_ys;
    accs[1][0] += (accs[1][1] - accs[1][0]) * frac_ys;

    accs[0][0] += (accs[1][0] - accs[0][0]) * frac_zs;

    return accs[0][0];
  };

  T operator()(uint32_t x, uint32_t y, uint32_t z) const {
    return m_data[z * m_stride + y * m_width + x];
  }

private:
  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_depth;

  uint64_t m_stride;
  uint64_t m_size_bytes;

  MappedFile m_file;
  const LE<T> *m_data;
};
