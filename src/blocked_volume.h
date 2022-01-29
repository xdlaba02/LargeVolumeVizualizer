#pragma once

#include "mapped_file.h"
#include "intersection.h"
#include "endian.h"
#include "simd.h"
#include "morton.h"
#include "preintegrated_transfer_function.h"

#include <glm/glm.hpp>

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

  BlockedVolume(const char *blocks_file_name, const char *metadata_file_name, uint64_t width, uint64_t height, uint64_t depth):
      m_info(width, height, depth) {

    m_data_file.open(blocks_file_name, 0, m_info.size_in_bytes, MappedFile::READ, MappedFile::SHARED);

    if (!m_data_file) {
      return;
    }

    m_metadata_file.open(metadata_file_name, 0, m_info.size_in_blocks * 2 + m_info.size_in_blocks * sizeof(uint64_t), MappedFile::READ, MappedFile::SHARED);

    if (!m_metadata_file) {
      m_data_file.close();
      return;
    }

    m_blocks  = reinterpret_cast<const Block *>(m_data_file.data());
    m_mins    = reinterpret_cast<const LE<T> *>(m_metadata_file.data());
    m_maxs    = m_mins + m_info.size_in_blocks;
    m_offsets = reinterpret_cast<const LE<uint64_t> *>(m_maxs + m_info.size_in_blocks);
  }

  operator bool() const { return m_data_file && m_metadata_file; }

  simd::float_v samples(simd::float_v xs, simd::float_v ys, simd::float_v zs, simd::float_m mask) const {
    simd::uint32_v pix_xs = xs;
    simd::uint32_v pix_ys = ys;
    simd::uint32_v pix_zs = zs;

    simd::float_v frac_xs = xs - pix_xs;
    simd::float_v frac_ys = ys - pix_ys;
    simd::float_v frac_zs = zs - pix_zs;

    simd::uint32_v block_xs = simd::fast_div<SUBVOLUME_SIDE>(pix_xs);
    simd::uint32_v block_ys = simd::fast_div<SUBVOLUME_SIDE>(pix_ys);
    simd::uint32_v block_zs = simd::fast_div<SUBVOLUME_SIDE>(pix_zs);

    // reminder from division by 15 todo make universal
    simd::uint32_v in_block_xs = pix_xs - ((block_xs << 4) - block_xs);
    simd::uint32_v in_block_ys = pix_ys - ((block_ys << 4) - block_ys);
    simd::uint32_v in_block_zs = pix_zs - ((block_zs << 4) - block_zs);

    simd::uint32_v in_block_xs0_interleaved = morton::interleave_4b_3d(in_block_xs + 0);
    simd::uint32_v in_block_xs1_interleaved = morton::interleave_4b_3d(in_block_xs + 1);
    simd::uint32_v in_block_ys0_interleaved = morton::interleave_4b_3d(in_block_ys + 0);
    simd::uint32_v in_block_ys1_interleaved = morton::interleave_4b_3d(in_block_ys + 1);
    simd::uint32_v in_block_zs0_interleaved = morton::interleave_4b_3d(in_block_zs + 0);
    simd::uint32_v in_block_zs1_interleaved = morton::interleave_4b_3d(in_block_zs + 1);

    simd::uint32_v offsets[2][2][2];

    offsets[0][0][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
    offsets[0][0][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
    offsets[0][1][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
    offsets[0][1][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
    offsets[1][0][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
    offsets[1][0][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
    offsets[1][1][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);
    offsets[1][1][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);

    simd::int32_v buffers[2][2][2];

    for (uint32_t k = 0; k < simd::len; k++) {
      if (mask[k]) {
        uint64_t index = block_zs[k] * m_info.stride_in_blocks + block_ys[k] * m_info.width_in_blocks + block_xs[k];
        T min = m_mins[index];
        T max = m_maxs[index];

        if (min == max) {
          buffers[0][0][0][k] = min;
          buffers[0][0][1][k] = min;
          buffers[0][1][0][k] = min;
          buffers[0][1][1][k] = min;
          buffers[1][0][0][k] = min;
          buffers[1][0][1][k] = min;
          buffers[1][1][0][k] = min;
          buffers[1][1][1][k] = min;
        }
        else {
          const Block &block = m_blocks[m_offsets[index]];

          buffers[0][0][0][k] = block[offsets[0][0][0][k]];
          buffers[0][0][1][k] = block[offsets[0][0][1][k]];
          buffers[0][1][0][k] = block[offsets[0][1][0][k]];
          buffers[0][1][1][k] = block[offsets[0][1][1][k]];
          buffers[1][0][0][k] = block[offsets[1][0][0][k]];
          buffers[1][0][1][k] = block[offsets[1][0][1][k]];
          buffers[1][1][0][k] = block[offsets[1][1][0][k]];
          buffers[1][1][1][k] = block[offsets[1][1][1][k]];
        }
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

  float sample(float x, float y, float z) const {
    uint32_t pix_x = x;
    uint32_t pix_y = y;
    uint32_t pix_z = z;

    float frac_x = x - pix_x;
    float frac_y = y - pix_y;
    float frac_z = z - pix_z;

    uint32_t block_x = pix_x / SUBVOLUME_SIDE;
    uint32_t block_y = pix_y / SUBVOLUME_SIDE;
    uint32_t block_z = pix_z / SUBVOLUME_SIDE;

    // reminder from division
    uint32_t in_block_x = pix_x - ((block_x << 4) - block_x);
    uint32_t in_block_y = pix_y - ((block_y << 4) - block_y);
    uint32_t in_block_z = pix_z - ((block_z << 4) - block_z);

    uint32_t in_block_xs0_interleaved = morton::interleave_4b_3d(in_block_x + 0);
    uint32_t in_block_xs1_interleaved = morton::interleave_4b_3d(in_block_x + 1);
    uint32_t in_block_ys0_interleaved = morton::interleave_4b_3d(in_block_y + 0);
    uint32_t in_block_ys1_interleaved = morton::interleave_4b_3d(in_block_y + 1);
    uint32_t in_block_zs0_interleaved = morton::interleave_4b_3d(in_block_z + 0);
    uint32_t in_block_zs1_interleaved = morton::interleave_4b_3d(in_block_z + 1);

    uint32_t offsets[2][2][2];

    offsets[0][0][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
    offsets[0][0][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
    offsets[0][1][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
    offsets[0][1][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
    offsets[1][0][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
    offsets[1][0][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
    offsets[1][1][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);
    offsets[1][1][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);

    int32_t buffers[2][2][2];

    uint64_t index = block_z * m_info.stride_in_blocks + block_y * m_info.width_in_blocks + block_x;
    T min = m_mins[index];
    T max = m_maxs[index];

    if (min == max) {
      buffers[0][0][0] = min;
      buffers[0][0][1] = min;
      buffers[0][1][0] = min;
      buffers[0][1][1] = min;
      buffers[1][0][0] = min;
      buffers[1][0][1] = min;
      buffers[1][1][0] = min;
      buffers[1][1][1] = min;
    }
    else {
      const Block &block = m_blocks[m_offsets[index]];

      buffers[0][0][0] = block[offsets[0][0][0]];
      buffers[0][0][1] = block[offsets[0][0][1]];
      buffers[0][1][0] = block[offsets[0][1][0]];
      buffers[0][1][1] = block[offsets[0][1][1]];
      buffers[1][0][0] = block[offsets[1][0][0]];
      buffers[1][0][1] = block[offsets[1][0][1]];
      buffers[1][1][0] = block[offsets[1][1][0]];
      buffers[1][1][1] = block[offsets[1][1][1]];
    }

    float accs[2][2];

    accs[0][0] = buffers[0][0][0] + (buffers[0][0][1] - buffers[0][0][0]) * frac_x;
    accs[0][1] = buffers[0][1][0] + (buffers[0][1][1] - buffers[0][1][0]) * frac_x;
    accs[1][0] = buffers[1][0][0] + (buffers[1][0][1] - buffers[1][0][0]) * frac_x;
    accs[1][1] = buffers[1][1][0] + (buffers[1][1][1] - buffers[1][1][0]) * frac_x;

    accs[0][0] += (accs[0][1] - accs[0][0]) * frac_y;
    accs[1][0] += (accs[1][1] - accs[1][0]) * frac_y;

    accs[0][0] += (accs[1][0] - accs[0][0]) * frac_z;

    return accs[0][0];
  };

  glm::vec4 integrate(const glm::vec3 &origin, const glm::vec3 &direction, float stepsize, const PreintegratedTransferFunction<uint8_t> &transfer_r, const PreintegratedTransferFunction<uint8_t> &transfer_g, const PreintegratedTransferFunction<uint8_t> &transfer_b, const PreintegratedTransferFunction<uint8_t> &transfer_a) const {

    glm::vec3 step = direction * stepsize;
    glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

    traverse(origin, direction, [&](const glm::vec<3, uint32_t> &block_pos, const glm::vec3 &in_block_pos, float tmax) {
      //std::cerr << block_pos.x << " " << block_pos.y << " " << block_pos.z << " | " << in_block_pos.x << " " << in_block_pos.y << " " << in_block_pos.z << " | " << tmax << "\n";

      uint64_t index = block_pos.z * m_info.stride_in_blocks + block_pos.y * m_info.width_in_blocks + block_pos.x;

      T min = m_mins[index];
      T max = m_maxs[index];

      if (min == max) { // fast integration
        float a = transfer_a(min, max);

        if (a > 0.f) { // empty space skipping
          float r = transfer_r(min, max);
          float g = transfer_g(min, max);
          float b = transfer_b(min, max);

          float alpha = 1.f - std::exp(-a * tmax);

          float coef = alpha * dst.a;

          dst.r += r * coef;
          dst.g += g * coef;
          dst.b += b * coef;
          dst.a *= 1 - alpha;
        }
      }
      else { // integration by sampling

        float prev_value = sample_block(m_blocks[m_offsets[index]], in_block_pos);

        glm::vec3 sample = in_block_pos + step;
        float tmin = stepsize;

        while (tmin <= tmax) {
          float value = sample_block(m_blocks[m_offsets[index]], sample);

          float a = transfer_a(value, prev_value);

          if (a > 0.f) { // empty sample skipping
            float r = transfer_r(value, prev_value);
            float g = transfer_g(value, prev_value);
            float b = transfer_b(value, prev_value);

            float alpha = 1.f - std::exp(-a * std::min(stepsize, tmax - tmin));

            float coef = alpha * dst.a;

            dst.r += r * coef;
            dst.g += g * coef;
            dst.b += b * coef;
            dst.a *= 1 - alpha;
          }

          prev_value = value;

          tmin += stepsize;
          sample += step;
        }
      }

      return dst.a > 1.f / 256.f;
    });

    return dst;
  }

  float width() const { return m_info.width; }
  float height() const { return m_info.height; }
  float depth() const { return m_info.depth; }

private:
  MappedFile m_data_file;
  MappedFile m_metadata_file;

  const Info m_info;

  const Block *m_blocks;
  const LE<T> *m_mins;
  const LE<T> *m_maxs;
  const LE<uint64_t> *m_offsets;

  template <typename F>
  inline void traverse(const glm::vec3 &origin, const glm::vec3 &direction, const F& callback) const {

    glm::vec3 t_delta;         // SUBVOLUME_SIDE / direction
    glm::vec3 t_next_crossing; //

    glm::vec<3, int32_t> cell;

    glm::vec<3, int32_t> step;
    glm::vec<3, int32_t> stop;

    float t, t_end;

    {
      glm::vec<3, uint32_t> size = { m_info.width, m_info.height, m_info.depth };
      glm::vec<3, uint32_t> size_in_blocks = { m_info.width_in_blocks, m_info.height_in_blocks, m_info.depth_in_blocks };

      glm::vec3 direction_inverse = 1.f / direction;

      intersect_aabb_ray(origin, direction_inverse, {0, 0, 0}, size, t, t_end);

      if (t >= t_end) {
        return;
      }

      glm::vec3 sample = origin + direction * t;

      for (uint8_t i = 0; i < 3; i++) {
        cell[i] = std::clamp<int32_t>(sample[i] / SUBVOLUME_SIDE, 0, size_in_blocks[i] - 1);

        if (direction[i] < 0.f) {
            t_delta[i] = -float(SUBVOLUME_SIDE) * direction_inverse[i];
            t_next_crossing[i] = t + ((cell[i] + 0) * float(SUBVOLUME_SIDE) - sample[i]) * direction_inverse[i];
            step[i] = -1;
            stop[i] = 0;
        }
        else {
            t_delta[i] = float(SUBVOLUME_SIDE) * direction_inverse[i];
            t_next_crossing[i] = t + ((cell[i] + 1) * float(SUBVOLUME_SIDE) - sample[i]) * direction_inverse[i];
            step[i] = 1;
            stop[i] = size_in_blocks[i] - 1;
        }
      }
    }


    auto nearest_axis = [&]() {
      uint8_t k = ((t_next_crossing[0] < t_next_crossing[1]) << 2)
                | ((t_next_crossing[0] < t_next_crossing[2]) << 1)
                | ((t_next_crossing[1] < t_next_crossing[2]) << 0);

      constexpr uint8_t arr[8] { 2, 1, 2, 1, 2, 2, 0, 0 };

      return arr[k];
    };

    uint8_t axis = nearest_axis();

    while (cell[axis] != stop[axis]) {
      if (!callback(cell, origin + direction * t - glm::vec3(cell * int32_t(SUBVOLUME_SIDE)), t_next_crossing[axis] - t)) {
        return;
      }

      t = t_next_crossing[axis];

      t_next_crossing[axis] += t_delta[axis];

      cell[axis] += step[axis];

      axis = nearest_axis();
    }

    callback(cell, origin + direction * t - glm::vec3(cell * int32_t(SUBVOLUME_SIDE)), t_end - t);
  }

  static inline float sample_block(const Block &block, const glm::vec3 &pos) {
    glm::vec<3, uint32_t> pix = pos;

    glm::vec3 frac = pos - glm::vec3(pix);

    uint32_t in_block_xs0_interleaved = morton::interleave_4b_3d(pix.x + 0);
    uint32_t in_block_xs1_interleaved = morton::interleave_4b_3d(pix.x + 1);
    uint32_t in_block_ys0_interleaved = morton::interleave_4b_3d(pix.y + 0);
    uint32_t in_block_ys1_interleaved = morton::interleave_4b_3d(pix.y + 1);
    uint32_t in_block_zs0_interleaved = morton::interleave_4b_3d(pix.z + 0);
    uint32_t in_block_zs1_interleaved = morton::interleave_4b_3d(pix.z + 1);

    uint32_t offsets[2][2][2];

    offsets[0][0][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
    offsets[0][0][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
    offsets[0][1][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
    offsets[0][1][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
    offsets[1][0][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
    offsets[1][0][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
    offsets[1][1][0] = morton::combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);
    offsets[1][1][1] = morton::combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);

    int32_t buffers[2][2][2];

    buffers[0][0][0] = block[offsets[0][0][0]];
    buffers[0][0][1] = block[offsets[0][0][1]];
    buffers[0][1][0] = block[offsets[0][1][0]];
    buffers[0][1][1] = block[offsets[0][1][1]];
    buffers[1][0][0] = block[offsets[1][0][0]];
    buffers[1][0][1] = block[offsets[1][0][1]];
    buffers[1][1][0] = block[offsets[1][1][0]];
    buffers[1][1][1] = block[offsets[1][1][1]];

    float accs[2][2];

    accs[0][0] = buffers[0][0][0] + (buffers[0][0][1] - buffers[0][0][0]) * frac.x;
    accs[0][1] = buffers[0][1][0] + (buffers[0][1][1] - buffers[0][1][0]) * frac.x;
    accs[1][0] = buffers[1][0][0] + (buffers[1][0][1] - buffers[1][0][0]) * frac.x;
    accs[1][1] = buffers[1][1][0] + (buffers[1][1][1] - buffers[1][1][0]) * frac.x;

    accs[0][0] += (accs[0][1] - accs[0][0]) * frac.y;
    accs[1][0] += (accs[1][1] - accs[1][0]) * frac.y;

    accs[0][0] += (accs[1][0] - accs[0][0]) * frac.z;

    return accs[0][0];
  }
};
