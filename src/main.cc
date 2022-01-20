
#include "raw_volume.h"
#include "morton.h"
#include "blocked_volume.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <GLFW/glfw3.h>

#include <Vc/Vc>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

using Vc::float_v;
using Vc::float_m;

constexpr uint32_t simdlen = float_v::size();

using uint32_v = Vc::SimdArray<uint32_t, simdlen>;
using  int32_v = Vc::SimdArray<int32_t, simdlen>;

// TODO preintegrated
// TODO pyramid
// TODO min max tree

template <typename ColorType>
class ColorGradient1D {
    std::vector<std::pair<float, ColorType>> m_colors;

    using ColorsIt = typename decltype(m_colors)::const_iterator;

    ColorsIt itrateToNearestValue(float value) const {
        for (auto it = std::cbegin(m_colors); it != std::cend(m_colors); ++it) {
            if (it->first >= value) {
                return it;
            }
        }

        return std::cend(m_colors);
    }

public:
    void setColor(float value, ColorType color) {
        m_colors.insert(itrateToNearestValue(value), { value, color });
    }

    ColorType color(float value) const {
        auto it = itrateToNearestValue(value);
        auto lowerIt = std::max(std::begin(m_colors), it - 1);
        auto upperIt = std::min(it, std::end(m_colors) - 1);
        float diff = upperIt->first - lowerIt->first;
        float frac = diff ? (value - lowerIt->first) / diff : 0.5f;
        return lowerIt->second * (1 - frac) + upperIt->second * frac;
    }
};


inline void intersect_aabb_rays_single_origin(const glm::vec3& origin, glm::vec<3, float_v> ray_directions, const glm::vec3 &min, const glm::vec3 &max, float_v& tmins, float_v& tmaxs) {
  ray_directions = glm::vec<3, float_v>{1, 1, 1} / ray_directions;

  tmins = -std::numeric_limits<float>::infinity();
  tmaxs = std::numeric_limits<float>::infinity();

  for (uint32_t i = 0; i < 3; ++i) {
    float_v t0 = (min[i] - origin[i]) * ray_directions[i];
    float_v t1 = (max[i] - origin[i]) * ray_directions[i];

    float_m swap_mask = ray_directions[i] < 0.f;

    t0(swap_mask) = t0 + t1;
    t1(swap_mask) = t0 - t1;
    t0(swap_mask) = t0 - t1;

    tmins(t0 > tmins) = t0;
    tmaxs(t1 < tmaxs) = t1;
  }
}

#if 1

// fast division by 15 in 32 bit register, maximum viable number this can divide correctly is 74908, which should be sufficient
template <typename T>
inline T div_by_15(T n) {
  return (n * 0x8889) >> 19;
}

#else

template <typename T>
inline T div_by_15(T n) {
  return n / 15;
}

#endif

inline float_v sampler1D(const float *data, float_v values) {
  uint32_v pix = values;

  float_v accs[2];

  accs[0].gather(data, pix);
  accs[1].gather(data, pix + 1);

  return accs[0] + (accs[1] - accs[0]) * (values - pix);
};

template <size_t W, size_t H>
inline float_v sampler2D(const float *data, float_v xs, float_v ys, float_m mask) {
  uint32_v pix_xs = xs;
  uint32_v pix_ys = ys;

  float_v frac_xs = xs - pix_xs;
  float_v frac_ys = ys - pix_ys;

  float_v accs[2][2];

  for (uint32_t k = 0; k < simdlen; k++) {
    if (mask[k]) {
        uint64_t base = pix_ys[k] * W + pix_xs[k];

        accs[0][0][k] = data[base];
        accs[0][1][k] = data[base + 1];
        accs[1][0][k] = data[base + W];
        accs[1][1][k] = data[base + W + 1];
    }
  }

  accs[0][0] += (accs[0][1] - accs[0][0]) * frac_xs;
  accs[1][0] += (accs[1][1] - accs[1][0]) * frac_xs;

  return accs[0][0] + (accs[1][0] - accs[0][0]) * frac_ys;
};

template <typename T>
inline float_v rawVolumeSampler(const RawVolume<T> &volume, float_v xs, float_v ys, float_v zs, float_m mask) {
  xs *= volume.width() - 1;
  ys *= volume.height() - 1;
  zs *= volume.depth() - 1;

  uint32_v pix_xs = xs;
  uint32_v pix_ys = ys;
  uint32_v pix_zs = zs;

  float_v frac_xs = xs - pix_xs;
  float_v frac_ys = ys - pix_ys;
  float_v frac_zs = zs - pix_zs;

  float_m incrementable_xs = pix_xs < (volume.width() - 1);
  float_m incrementable_ys = pix_ys < (volume.height() - 1);
  float_m incrementable_zs = pix_zs < (volume.depth() - 1);

  int32_v buffers[2][2][2];

  for (uint32_t k = 0; k < simdlen; k++) {
    if (mask[k]) {
      uint64_t base = pix_zs[k] * volume.zStride() + pix_ys[k] * volume.yStride() + pix_xs[k] * volume.xStride();

      uint64_t x_offset = incrementable_xs[k] * volume.xStride();
      uint64_t y_offset = incrementable_ys[k] * volume.yStride();
      uint64_t z_offset = incrementable_zs[k] * volume.zStride();

      buffers[0][0][0][k] = volume[base];
      buffers[0][0][1][k] = volume[base + x_offset];
      buffers[0][1][0][k] = volume[base + y_offset];
      buffers[0][1][1][k] = volume[base + y_offset + x_offset];
      buffers[1][0][0][k] = volume[base + z_offset];
      buffers[1][0][1][k] = volume[base + z_offset + x_offset];
      buffers[1][1][0][k] = volume[base + z_offset + y_offset];
      buffers[1][1][1][k] = volume[base + z_offset + y_offset + x_offset];
    }
  }

  float_v accs[2][2];

  accs[0][0] = buffers[0][0][0] + (buffers[0][0][1] - buffers[0][0][0]) * frac_xs;
  accs[0][1] = buffers[0][1][0] + (buffers[0][1][1] - buffers[0][1][0]) * frac_xs;
  accs[1][0] = buffers[1][0][0] + (buffers[1][0][1] - buffers[1][0][0]) * frac_xs;
  accs[1][1] = buffers[1][1][0] + (buffers[1][1][1] - buffers[1][1][0]) * frac_xs;

  accs[0][0] += (accs[0][1] - accs[0][0]) * frac_ys;
  accs[1][0] += (accs[1][1] - accs[1][0]) * frac_ys;

  accs[0][0] += (accs[1][0] - accs[0][0]) * frac_zs;

  return accs[0][0];
};

inline float_v blockedVolumeSampler(const BlockedVolume<uint8_t> &volume, float_v xs, float_v ys, float_v zs, float_m mask) {
  xs *= volume.width() - 1;
  ys *= volume.height() - 1;
  zs *= volume.depth() - 1;

  uint32_v pix_xs = xs;
  uint32_v pix_ys = ys;
  uint32_v pix_zs = zs;

  float_v frac_xs = xs - pix_xs;
  float_v frac_ys = ys - pix_ys;
  float_v frac_zs = zs - pix_zs;

  uint32_v block_xs = div_by_15(pix_xs);
  uint32_v block_ys = div_by_15(pix_ys);
  uint32_v block_zs = div_by_15(pix_zs);

  // reminder from division
  uint32_v in_block_xs = pix_xs - ((block_xs << 4) - block_xs);
  uint32_v in_block_ys = pix_ys - ((block_ys << 4) - block_ys);
  uint32_v in_block_zs = pix_zs - ((block_zs << 4) - block_zs);

  uint32_v in_block_xs0_interleaved = morton::interleave_4b_3d(in_block_xs + 0);
  uint32_v in_block_xs1_interleaved = morton::interleave_4b_3d(in_block_xs + 1);
  uint32_v in_block_ys0_interleaved = morton::interleave_4b_3d(in_block_ys + 0);
  uint32_v in_block_ys1_interleaved = morton::interleave_4b_3d(in_block_ys + 1);
  uint32_v in_block_zs0_interleaved = morton::interleave_4b_3d(in_block_zs + 0);
  uint32_v in_block_zs1_interleaved = morton::interleave_4b_3d(in_block_zs + 1);

  uint32_v offsets[2][2][2];

  offsets[0][0][0] = morton::morton_combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
  offsets[0][0][1] = morton::morton_combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
  offsets[0][1][0] = morton::morton_combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
  offsets[0][1][1] = morton::morton_combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
  offsets[1][0][0] = morton::morton_combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
  offsets[1][0][1] = morton::morton_combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
  offsets[1][1][0] = morton::morton_combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);
  offsets[1][1][1] = morton::morton_combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);

  int32_v buffers[2][2][2];

  for (uint32_t k = 0; k < simdlen; k++) {
    if (mask[k]) {
      uint64_t block_index = block_zs[k] * volume.stride_in_blocks() + block_ys[k] * volume.width_in_blocks() + block_xs[k];

      uint8_t min = volume.min(block_index);
      uint8_t max = volume.max(block_index);

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
        const Block<uint8_t> &block = volume.block(volume.offset(block_index));

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

  float_v accs[2][2];

  accs[0][0] = buffers[0][0][0] + (buffers[0][0][1] - buffers[0][0][0]) * frac_xs;
  accs[0][1] = buffers[0][1][0] + (buffers[0][1][1] - buffers[0][1][0]) * frac_xs;
  accs[1][0] = buffers[1][0][0] + (buffers[1][0][1] - buffers[1][0][0]) * frac_xs;
  accs[1][1] = buffers[1][1][0] + (buffers[1][1][1] - buffers[1][1][0]) * frac_xs;

  accs[0][0] += (accs[0][1] - accs[0][0]) * frac_ys;
  accs[1][0] += (accs[1][1] - accs[1][0]) * frac_ys;

  accs[0][0] += (accs[1][0] - accs[0][0]) * frac_zs;

  return accs[0][0];
};

template <typename T>
using Block = T[4096];

int main(int argc, char *argv[]) {
  if (!glfwInit()) {
      printf("Couldn't init GLFW\n");
      return 1;
  }

  constexpr uint32_t window_width = 1920;
  constexpr uint32_t window_height = 1080;

  GLFWwindow *window = glfwCreateWindow(window_width, window_height, "Ahoj", NULL, NULL);
  if (!window) {
      printf("Couldn't open window\n");
      return 1;
  }

  glfwMakeContextCurrent(window);

  uint32_t width;
  uint32_t height;
  uint32_t depth;

  {
    std::stringstream wstream(argv[3]);
    std::stringstream hstream(argv[4]);
    std::stringstream dstream(argv[5]);
    wstream >> width;
    hstream >> height;
    dstream >> depth;
  }

  BlockedVolume<uint8_t> blocked_volume(argv[1], argv[2], width, height, depth);

  if (!blocked_volume) {
    std::cerr << "blocked_volume failed\n";
    return 1;
  }

  std::array<float, 257 * 257> preintegratedTransferR {};
  std::array<float, 257 * 257> preintegratedTransferG {};
  std::array<float, 257 * 257> preintegratedTransferB {};
  std::array<float, 257 * 257> preintegratedTransferA {};

  {
      ColorGradient1D<glm::vec3> colorGradient {};

      colorGradient.setColor(80.f,  {0.75f, 0.5f, 0.25f});
      colorGradient.setColor(82.f,  {1.00f, 1.0f, 0.85f});

      ColorGradient1D<float> alphaGradient {};

      /*
      alphaGradient.setColor(151.f,  0.0f);
      alphaGradient.setColor(152.f,  1.0f);
      */
      alphaGradient.setColor(40.f,  000.0f);
      alphaGradient.setColor(60.f,  001.0f);
      alphaGradient.setColor(63.f,  005.f);
      alphaGradient.setColor(80.f,  000.0f);
      alphaGradient.setColor(82.f,  100.0f);

      for (uint32_t x = 0; x < 257; x++) {
        glm::vec3 color(0.f, 0.f, 0.f);
        float alpha = 0.f;

        for (uint32_t y = x; y < 257; y++) {
          color += colorGradient.color(y);
          alpha += alphaGradient.color(y);

          preintegratedTransferR[y * 257 + x] = preintegratedTransferR[x * 257 + y] = color.r / ((y + 1) - x);
          preintegratedTransferG[y * 257 + x] = preintegratedTransferG[x * 257 + y] = color.g / ((y + 1) - x);
          preintegratedTransferB[y * 257 + x] = preintegratedTransferB[x * 257 + y] = color.b / ((y + 1) - x);
          preintegratedTransferA[y * 257 + x] = preintegratedTransferA[x * 257 + y] = alpha   / ((y + 1) - x);
        }
      }
  }

  std::vector<uint8_t> raster(window_width * window_height * 3);

  uint64_t rgb_x_stride = 3;
  uint64_t rgb_y_stride = window_width * rgb_x_stride;

  uint32_v pixel_offsets = uint32_v::IndexesFromZero() * 3;

  float time = 0.f;
  while (!glfwWindowShouldClose(window)) {

    glm::mat4 model = glm::rotate(glm::mat4(1.0f), -time, glm::vec3(0.0, 1.0, 0.0)) * glm::rotate(glm::mat4(1.0f), glm::radians(90.f), glm::vec3(1.0, 0.0, 0.0));

    glm::vec3 origin = glm::vec3(1, 0, 0) * 2.f;
    glm::mat4 view = glm::lookAt(origin, { 0, 0, 0 }, {0, 1, 0 }) * model;

    origin = glm::vec4(origin, 1) * model;


    #pragma omp parallel for schedule(dynamic)
    for (uint32_t j = 0; j < window_height; j++) {

      constexpr float cameraFOV = std::tan(45 * M_PI / 180 * 0.5);

      float y = (2 * (j + 0.5) / window_height - 1) * cameraFOV;

      for (uint32_t i = 0; i < window_width; i += simdlen) {
        float_v is = float_v::IndexesFromZero() + i;

        constexpr float aspect = float(window_width) / float(window_height);

        float_v xs = (2 * (is + 0.5f) / float(window_width) - 1) * aspect * cameraFOV;

        glm::vec<3, float_v> ray_directions {};

        for (uint32_t k = 0; k < simdlen; k++) {
          glm::vec4 direction = glm::normalize(glm::vec4(xs[k], y, -1, 0) * view);
          ray_directions.x[k] = direction.x;
          ray_directions.y[k] = direction.y;
          ray_directions.z[k] = direction.z;
        }

        float_v tmins, tmaxs;
        intersect_aabb_rays_single_origin(origin, ray_directions, {-.5, -.5, -.5}, {+.5, +.5, +.5}, tmins, tmaxs);

        glm::vec<4, float_v> dsts(0.f, 0.f, 0.f, 1.f);

        constexpr float stepsize = 0.002f;

        float_v prev_values {};

        {
          glm::vec<3, float_v> vs = glm::vec<3, float_v>(origin) + ray_directions * tmins + float_v(0.5f);
          prev_values = blockedVolumeSampler(blocked_volume, vs.x, vs.y, vs.z, tmins <= tmaxs);
          tmins += stepsize;
        }

        for (float_m mask = tmins <= tmaxs; !mask.isEmpty(); mask &= tmins <= tmaxs) {
          glm::vec<3, float_v> vs = glm::vec<3, float_v>(origin) + ray_directions * tmins + float_v(0.5f);

          float_v values = blockedVolumeSampler(blocked_volume, vs.x, vs.y, vs.z, mask);

          float_v a = sampler2D<257, 257>(preintegratedTransferA.data(), values, prev_values, mask);

          float_m alpha_mask = a > 0.f;

          if (!alpha_mask.isEmpty()) {
            float_v r = sampler2D<257, 257>(preintegratedTransferR.data(), values, prev_values, mask & alpha_mask);
            float_v g = sampler2D<257, 257>(preintegratedTransferG.data(), values, prev_values, mask & alpha_mask);
            float_v b = sampler2D<257, 257>(preintegratedTransferB.data(), values, prev_values, mask & alpha_mask);

            a = float_v(1.f) - Vc::exp(-a * Vc::min(stepsize, tmaxs - tmins));

            float_v coef = a * dsts.a;

            // Evaluate the current opacity
            dsts.r(mask) += r * coef;
            dsts.g(mask) += g * coef;
            dsts.b(mask) += b * coef;
            dsts.a(mask) *= 1 - a;

            mask &= dsts.a > 0.01f;
          }

          prev_values = values;

          tmins += stepsize;
        }

        dsts.r *= 255;
        dsts.g *= 255;
        dsts.b *= 255;

        uint8_t *rgb_start = raster.data() + j * rgb_y_stride + i * rgb_x_stride;

        dsts.r.scatter(rgb_start + 0, pixel_offsets);
        dsts.g.scatter(rgb_start + 1, pixel_offsets);
        dsts.b.scatter(rgb_start + 2, pixel_offsets);
      }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, raster.data());
    glfwSwapBuffers(window);
    glfwPollEvents();

    time += 0.1f;
  }
}
