
#include "raw_volume.h"
#include "morton.h"
#include "blocked_volume.h"
#include "linear_gradient.h"
#include "preintegrated_transfer_function.h"
#include "fast_div.h"
#include "simd.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

// TODO pyramid
// TODO min max tree

inline void intersect_aabb_rays_single_origin(const glm::vec3& origin, glm::vec<3, simd::float_v> ray_directions, const glm::vec3 &min, const glm::vec3 &max, simd::float_v& tmins, simd::float_v& tmaxs) {
  ray_directions = glm::vec<3, simd::float_v>{1, 1, 1} / ray_directions;

  tmins = -std::numeric_limits<float>::infinity();
  tmaxs = std::numeric_limits<float>::infinity();

  for (uint32_t i = 0; i < 3; ++i) {
    simd::float_v t0 = (min[i] - origin[i]) * ray_directions[i];
    simd::float_v t1 = (max[i] - origin[i]) * ray_directions[i];

    simd::swap(t0, t1, ray_directions[i] < 0.f);

    tmins(t0 > tmins) = t0;
    tmaxs(t1 < tmaxs) = t1;
  }
}

template <typename T>
inline simd::float_v rawVolumeSampler(const RawVolume<T> &volume, simd::float_v xs, simd::float_v ys, simd::float_v zs, simd::float_m mask) {
  xs *= volume.info.width - 1;
  ys *= volume.info.height - 1;
  zs *= volume.info.depth - 1;

  simd::uint32_v pix_xs = xs;
  simd::uint32_v pix_ys = ys;
  simd::uint32_v pix_zs = zs;

  simd::float_v frac_xs = xs - pix_xs;
  simd::float_v frac_ys = ys - pix_ys;
  simd::float_v frac_zs = zs - pix_zs;

  simd::float_m incrementable_xs = pix_xs < (volume.info.width - 1);
  simd::float_m incrementable_ys = pix_ys < (volume.info.height - 1);
  simd::float_m incrementable_zs = pix_zs < (volume.info.depth - 1);

  simd::int32_v buffers[2][2][2];

  for (uint32_t k = 0; k < simd::len; k++) {
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

inline simd::float_v blockedVolumeSampler(const BlockedVolume<uint8_t> &volume, simd::float_v xs, simd::float_v ys, simd::float_v zs, simd::float_m mask) {
  xs *= volume.info.width - 1;
  ys *= volume.info.height - 1;
  zs *= volume.info.depth - 1;

  simd::uint32_v pix_xs = xs;
  simd::uint32_v pix_ys = ys;
  simd::uint32_v pix_zs = zs;

  simd::float_v frac_xs = xs - pix_xs;
  simd::float_v frac_ys = ys - pix_ys;
  simd::float_v frac_zs = zs - pix_zs;

  simd::uint32_v block_xs = fast_div<15>(pix_xs);
  simd::uint32_v block_ys = fast_div<15>(pix_ys);
  simd::uint32_v block_zs = fast_div<15>(pix_zs);

  // reminder from division
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
      BlockedVolume<uint8_t>::Node node = volume.node(block_xs[k], block_ys[k], block_zs[k]);

      if (node.min == node.max) {
        buffers[0][0][0][k] = node.min;
        buffers[0][0][1][k] = node.min;
        buffers[0][1][0][k] = node.min;
        buffers[0][1][1][k] = node.min;
        buffers[1][0][0][k] = node.min;
        buffers[1][0][1][k] = node.min;
        buffers[1][1][0][k] = node.min;
        buffers[1][1][1][k] = node.min;
      }
      else {
        const BlockedVolume<uint8_t>::Block &block = node.block;

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

  std::map<float, glm::vec3> color_map {
    {80.f,  {0.75f, 0.5f, 0.25f}},
    {82.f,  {1.00f, 1.0f, 0.85f}}
  };

  std::map<float, float> alpha_map {
    {40.f,  000.0f},
    {60.f,  001.0f},
    {63.f,  005.0f},
    {80.f,  000.0f},
    {82.f,  100.0f},
    //{151.f,  0.0f},
    //{152.f,  1.0f},
  };

  PreintegratedTransferFunction<uint8_t> preintegratedTransferR([&](float v){ return linear_gradient(color_map, v).r; });
  PreintegratedTransferFunction<uint8_t> preintegratedTransferG([&](float v){ return linear_gradient(color_map, v).g; });
  PreintegratedTransferFunction<uint8_t> preintegratedTransferB([&](float v){ return linear_gradient(color_map, v).b; });
  PreintegratedTransferFunction<uint8_t> preintegratedTransferA([&](float v){ return linear_gradient(alpha_map, v); });

  std::vector<uint8_t> raster(window_width * window_height * 3);

  uint64_t rgb_x_stride = 3;
  uint64_t rgb_y_stride = window_width * rgb_x_stride;

  simd::uint32_v pixel_offsets = simd::uint32_v::IndexesFromZero() * 3;

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

      for (uint32_t i = 0; i < window_width; i += simd::len) {
        simd::float_v is = simd::float_v::IndexesFromZero() + i;

        constexpr float aspect = float(window_width) / float(window_height);

        simd::float_v xs = (2 * (is + 0.5f) / float(window_width) - 1) * aspect * cameraFOV;

        glm::vec<3, simd::float_v> ray_directions {};

        for (uint32_t k = 0; k < simd::len; k++) {
          glm::vec4 direction = glm::normalize(glm::vec4(xs[k], y, -1, 0) * view);
          ray_directions.x[k] = direction.x;
          ray_directions.y[k] = direction.y;
          ray_directions.z[k] = direction.z;
        }

        simd::float_v tmins, tmaxs;
        intersect_aabb_rays_single_origin(origin, ray_directions, {-.5, -.5, -.5}, {+.5, +.5, +.5}, tmins, tmaxs);

        glm::vec<4, simd::float_v> dsts(0.f, 0.f, 0.f, 1.f);

        constexpr float stepsize = 0.002f;

        simd::float_v prev_values {};

        {
          glm::vec<3, simd::float_v> vs = glm::vec<3, simd::float_v>(origin) + ray_directions * tmins + simd::float_v(0.5f);
          prev_values = blockedVolumeSampler(blocked_volume, vs.x, vs.y, vs.z, tmins <= tmaxs);
          tmins += stepsize;
        }

        for (simd::float_m mask = tmins <= tmaxs; !mask.isEmpty(); mask &= tmins <= tmaxs) {
          glm::vec<3, simd::float_v> vs = glm::vec<3, simd::float_v>(origin) + ray_directions * tmins + simd::float_v(0.5f);

          simd::float_v values = blockedVolumeSampler(blocked_volume, vs.x, vs.y, vs.z, mask);

          simd::float_v a = preintegratedTransferA(values, prev_values, mask);

          simd::float_m alpha_mask = a > 0.f;

          if (!alpha_mask.isEmpty()) {
            simd::float_v r = preintegratedTransferR(values, prev_values, mask & alpha_mask);
            simd::float_v g = preintegratedTransferG(values, prev_values, mask & alpha_mask);
            simd::float_v b = preintegratedTransferB(values, prev_values, mask & alpha_mask);

            a = simd::float_v(1.f) - Vc::exp(-a * Vc::min(stepsize, tmaxs - tmins));

            simd::float_v coef = a * dsts.a;

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
