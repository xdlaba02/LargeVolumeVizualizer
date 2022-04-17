#pragma once

#include <utils/morton.h>
#include <utils/simd.h>

#include <cstdint>
#include <cstddef>

template <typename T, size_t BLOCK_BITS>
void sample_morton_scalar(const T *data, uint64_t block_index, float denorm_x, float denorm_y, float denorm_z) {
  static constexpr const size_t BLOCK_SIZE = size_t{1} << (BLOCK_BITS * 3);

  uint32_t vox_x[2];
  uint32_t vox_y[2];
  uint32_t vox_z[2];

  vox_x[0] = denorm_x;
  vox_y[0] = denorm_y;
  vox_z[0] = denorm_z;

  vox_x[1] = denorm_x + 1.f;
  vox_y[1] = denorm_y + 1.f;
  vox_z[1] = denorm_z + 1.f;

  vox_x[0] = Morton<BLOCK_BITS>::interleave(vox_x[0]);
  vox_y[0] = Morton<BLOCK_BITS>::interleave(vox_y[0]);
  vox_z[0] = Morton<BLOCK_BITS>::interleave(vox_z[0]);

  vox_x[1] = Morton<BLOCK_BITS>::interleave(vox_x[1]);
  vox_y[1] = Morton<BLOCK_BITS>::interleave(vox_y[1]);
  vox_z[1] = Morton<BLOCK_BITS>::interleave(vox_z[1]);

  static float v __attribute__((used)) {};

  for (uint8_t z = 0; z < 2; z++) {
    for (uint8_t y = 0; y < 2; y++) {
      for (uint8_t x = 0; x < 2; x++) {
        v += data[block_index * BLOCK_SIZE + Morton<BLOCK_BITS>::combine_interleaved(vox_x[x], vox_y[y], vox_z[z])];
      }
    }
  }
}

template <typename T, size_t BLOCK_BITS>
void sample_linear_scalar(const T *data, uint64_t block_index, float denorm_x, float denorm_y, float denorm_z) {
  static constexpr const size_t BLOCK_SIZE = size_t{1} << (BLOCK_BITS * 3);

  uint32_t vox_x[2];
  uint32_t vox_y[2];
  uint32_t vox_z[2];

  vox_x[0] = denorm_x;
  vox_y[0] = denorm_y;
  vox_z[0] = denorm_z;

  vox_x[1] = denorm_x + 1.f;
  vox_y[1] = denorm_y + 1.f;
  vox_z[1] = denorm_z + 1.f;

  static float v __attribute__((used)) {};

  for (uint8_t z = 0; z < 2; z++) {
    for (uint8_t y = 0; y < 2; y++) {
      for (uint8_t x = 0; x < 2; x++) {
        v += data[block_index * BLOCK_SIZE + vox_x[x] * 1 + vox_y[y] * (1 << BLOCK_BITS) + vox_z[z] * (1 << BLOCK_BITS * 2)];
      }
    }
  }
}

template <typename T, size_t BLOCK_BITS>
void sample_morton_simd(const T *data, const std::array<uint64_t, simd::len> &block_indices, const simd::float_v &denorm_x, const simd::float_v &denorm_y, const simd::float_v &denorm_z, const simd::float_m &mask) {
  static constexpr const size_t BLOCK_SIZE = size_t{1} << (BLOCK_BITS * 3);

  simd::uint32_v indices[2][2][2];

  {
    simd::uint32_v vox_x[2];
    simd::uint32_v vox_y[2];
    simd::uint32_v vox_z[2];

    vox_x[0] = denorm_x;
    vox_y[0] = denorm_y;
    vox_z[0] = denorm_z;

    vox_x[1] = denorm_x + 1.f;
    vox_y[1] = denorm_y + 1.f;
    vox_z[1] = denorm_z + 1.f;

    vox_x[0] = Morton<BLOCK_BITS>::interleave(vox_x[0]);
    vox_y[0] = Morton<BLOCK_BITS>::interleave(vox_y[0]);
    vox_z[0] = Morton<BLOCK_BITS>::interleave(vox_z[0]);

    vox_x[1] = Morton<BLOCK_BITS>::interleave(vox_x[1]);
    vox_y[1] = Morton<BLOCK_BITS>::interleave(vox_y[1]);
    vox_z[1] = Morton<BLOCK_BITS>::interleave(vox_z[1]);

    for (uint8_t z = 0; z < 2; z++) {
      for (uint8_t y = 0; y < 2; y++) {
        for (uint8_t x = 0; x < 2; x++) {
          indices[z][y][x] = Morton<BLOCK_BITS>::combine_interleaved(vox_x[x], vox_y[y], vox_z[z]);
        }
      }
    }
  }

  static simd::float_v v __attribute__((used)) {};

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      for (uint8_t z = 0; z < 2; z++) {
        for (uint8_t y = 0; y < 2; y++) {
          for (uint8_t x = 0; x < 2; x++) {
            v += data[block_indices[k] * BLOCK_SIZE + indices[z][y][x][k]];
          }
        }
      }
    }
  }
}

template <typename T, size_t BLOCK_BITS>
void sample_linear_simd(const T *data, const std::array<uint64_t, simd::len> &block_indices, const simd::float_v &denorm_x, const simd::float_v &denorm_y, const simd::float_v &denorm_z, const simd::float_m &mask) {
  static constexpr const size_t BLOCK_SIZE = size_t{1} << (BLOCK_BITS * 3);

  simd::uint32_v indices[2][2][2];

  {
    simd::uint32_v vox_x[2];
    simd::uint32_v vox_y[2];
    simd::uint32_v vox_z[2];

    vox_x[0] = denorm_x;
    vox_y[0] = denorm_y;
    vox_z[0] = denorm_z;

    vox_x[1] = denorm_x + 1.f;
    vox_y[1] = denorm_y + 1.f;
    vox_z[1] = denorm_z + 1.f;

    for (uint8_t z = 0; z < 2; z++) {
      for (uint8_t y = 0; y < 2; y++) {
        for (uint8_t x = 0; x < 2; x++) {
          indices[z][y][x] = vox_x[x] * 1 + vox_y[y] * (1 << BLOCK_BITS) + vox_z[z] * (1 << BLOCK_BITS * 2);
        }
      }
    }
  }

  static simd::float_v v __attribute__((used)) {};

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      for (uint8_t z = 0; z < 2; z++) {
        for (uint8_t y = 0; y < 2; y++) {
          for (uint8_t x = 0; x < 2; x++) {
            v += data[block_indices[k] * BLOCK_SIZE + indices[z][y][x][k]];
          }
        }
      }
    }
  }
}

template <typename T>
void sample_raw_scalar(const T *data, uint32_t width, uint32_t height, uint32_t depth, float x, float y, float z) {
  float denorm_x = x * width  - 0.5f;
  float denorm_y = y * height - 0.5f;
  float denorm_z = z * depth  - 0.5f;

  uint32_t vox_x[2];
  uint32_t vox_y[2];
  uint32_t vox_z[2];

  vox_x[0] = denorm_x;
  vox_y[0] = denorm_y;
  vox_z[0] = denorm_z;

  vox_x[1] = std::min<uint32_t>(denorm_x + 1.f, width  - 1);
  vox_y[1] = std::min<uint32_t>(denorm_y + 1.f, height - 1);
  vox_z[1] = std::min<uint32_t>(denorm_z + 1.f, depth  - 1);

  static float v __attribute__((used)) {};

  for (uint8_t z = 0; z < 2; z++) {
    for (uint8_t y = 0; y < 2; y++) {
      for (uint8_t x = 0; x < 2; x++) {
        v += data[vox_x[x] + vox_y[y] * width + vox_z[z] * width * height];
      }
    }
  }
}

template <typename T>
void sample_raw_simd(const T *data, uint32_t width, uint32_t height, uint32_t depth, const simd::float_v &x, const simd::float_v &y, const simd::float_v &z, const simd::float_m &mask) {
  simd::float_v denorm_x = x * width  - 0.5f;
  simd::float_v denorm_y = y * height - 0.5f;
  simd::float_v denorm_z = z * depth  - 0.5f;

  simd::uint32_v vox_x[2];
  simd::uint32_v vox_y[2];
  simd::uint32_v vox_z[2];

  vox_x[0] = denorm_x;
  vox_y[0] = denorm_y;
  vox_z[0] = denorm_z;

  vox_x[1] = min(denorm_x + 1.f, width  - 1);
  vox_y[1] = min(denorm_y + 1.f, height - 1);
  vox_z[1] = min(denorm_z + 1.f, depth  - 1);

  static simd::float_v v __attribute__((used)) {};

  for (uint32_t k = 0; k < simd::len; k++) {
    if (mask[k]) {
      for (uint8_t z = 0; z < 2; z++) {
        for (uint8_t y = 0; y < 2; y++) {
          for (uint8_t x = 0; x < 2; x++) {
            v += data[vox_x[x][k] + vox_y[y][k] * width + vox_z[z][k] * width * height];
          }
        }
      }
    }
  }
}

template <size_t BITS, typename F>
void sample_blocked_scalar(uint32_t width_in_blocks, uint32_t height_in_blocks, uint32_t depth_in_blocks, float x, float y, float z, const F &func) {
  static constexpr const uint32_t SUBVOLUME_SIDE = (uint32_t(1) << BITS) - 1;

  float denorm_x = x * width_in_blocks  * SUBVOLUME_SIDE  - 0.5f;
  float denorm_y = y * height_in_blocks * SUBVOLUME_SIDE - 0.5f;
  float denorm_z = z * depth_in_blocks  * SUBVOLUME_SIDE  - 0.5f;

  uint32_t vox_x = denorm_x;
  uint32_t vox_y = denorm_y;
  uint32_t vox_z = denorm_z;

  uint32_t block_x = vox_x / SUBVOLUME_SIDE;
  uint32_t block_y = vox_y / SUBVOLUME_SIDE;
  uint32_t block_z = vox_z / SUBVOLUME_SIDE;

  float in_block_x = denorm_x - block_x * SUBVOLUME_SIDE;
  float in_block_y = denorm_y - block_y * SUBVOLUME_SIDE;
  float in_block_z = denorm_z - block_z * SUBVOLUME_SIDE;

  func(block_x, block_y, block_z, in_block_x, in_block_y, in_block_z);
}

template <size_t BITS, typename F>
void sample_blocked_simd(uint32_t width_in_blocks, uint32_t height_in_blocks, uint32_t depth_in_blocks, simd::float_v x, simd::float_v y, simd::float_v z, const F &func) {
  static constexpr const uint32_t SUBVOLUME_SIDE = (uint32_t(1) << BITS) - 1;

  simd::float_v denorm_x = x * width_in_blocks  * SUBVOLUME_SIDE - 0.5f;
  simd::float_v denorm_y = y * height_in_blocks * SUBVOLUME_SIDE - 0.5f;
  simd::float_v denorm_z = z * depth_in_blocks  * SUBVOLUME_SIDE - 0.5f;

  simd::uint32_v vox_x = denorm_x;
  simd::uint32_v vox_y = denorm_y;
  simd::uint32_v vox_z = denorm_z;

  simd::uint32_v block_x = div<SUBVOLUME_SIDE>(vox_x);
  simd::uint32_v block_y = div<SUBVOLUME_SIDE>(vox_y);
  simd::uint32_v block_z = div<SUBVOLUME_SIDE>(vox_z);

  simd::float_v in_block_x = denorm_x - block_x * simd::float_v(SUBVOLUME_SIDE);
  simd::float_v in_block_y = denorm_y - block_y * simd::float_v(SUBVOLUME_SIDE);
  simd::float_v in_block_z = denorm_z - block_z * simd::float_v(SUBVOLUME_SIDE);

  func(block_x, block_y, block_z, in_block_x, in_block_y, in_block_z);
}
