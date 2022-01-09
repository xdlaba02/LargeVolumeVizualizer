
#include "raw_volume.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <GLFW/glfw3.h>

#include <Vc/Vc>

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

using Vc::float_v;
using Vc::float_m;

constexpr uint32_t simdlen = float_v::size();

using uint32_v = Vc::SimdArray<uint32_t, simdlen>;
using int32_v = Vc::SimdArray<int32_t, simdlen>;

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

template <typename T>
inline constexpr T interleave_4b_3d(T v) {
  v = (v | (v << 4)) & 0x0C3;
  return (v | (v << 2)) & 0x249;
}

template <typename T>
inline constexpr T morton_combine_interleaved(T x, T y, T z) {
  return x | (y << 1) | (z << 2);
}

template <typename T>
inline constexpr T morton_index_4b_3d(T x, T y, T z) {
  return morton_combine_interleaved(interleave_4b_3d(x), interleave_4b_3d(y), interleave_4b_3d(z));
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

inline float_v sampler3D(const RawVolume &volume, float_v xs, float_v ys, float_v zs, float_m mask) {
  const uint8_t *data = static_cast<uint8_t *>((void *)volume);
  uint32_t width = volume.width();
  uint32_t height = volume.height();
  uint32_t depth = volume.depth();

  uint32_v pix_xs = xs;
  uint32_v pix_ys = ys;
  uint32_v pix_zs = zs;

  float_v frac_xs = xs - pix_xs;
  float_v frac_ys = ys - pix_ys;
  float_v frac_zs = zs - pix_zs;

  float_m incrementable_xs = pix_xs < (width - 1);
  float_m incrementable_ys = pix_ys < (height - 1);
  float_m incrementable_zs = pix_zs < (depth - 1);

  int32_v buffers[2][2][2];

  uint32_v indices_z = (simd_cast<uint32_v>(pix_zs) * height + pix_ys) * width + pix_xs;
  for (uint32_t z = 0; z < 2; z++) {

    uint32_v indices_yz = indices_z;
    for (uint32_t y = 0; y < 2; y++) {

      uint32_v indices_xyz = indices_yz;
      for (uint32_t x = 0; x < 2; x++) {

        for (uint32_t k = 0; k < simdlen; k++) {
          if (mask[k]) {
            buffers[z][y][x][k] = data[indices_xyz[k]];
          }
        }

        indices_xyz(incrementable_xs) += 1;
      }

      indices_yz(incrementable_ys) += width;
    }

    indices_z(incrementable_zs) += width * height;
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

inline float_v sampler3DBlocked(const uint8_t *volume, uint32_t block_z_stride, uint32_t block_y_stride, uint32_t block_x_stride, float_v xs, float_v ys, float_v zs, float_m mask) {
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

  uint32_v block_start_index = block_zs * block_z_stride + block_ys * block_y_stride + block_xs * block_x_stride;

  uint32_v indices[2][2][2];

  uint32_v in_block_xs0_interleaved = interleave_4b_3d(in_block_xs + 0);
  uint32_v in_block_xs1_interleaved = interleave_4b_3d(in_block_xs + 1);
  uint32_v in_block_ys0_interleaved = interleave_4b_3d(in_block_ys + 0);
  uint32_v in_block_ys1_interleaved = interleave_4b_3d(in_block_ys + 1);
  uint32_v in_block_zs0_interleaved = interleave_4b_3d(in_block_zs + 0);
  uint32_v in_block_zs1_interleaved = interleave_4b_3d(in_block_zs + 1);

  indices[0][0][0] = block_start_index + morton_combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
  indices[0][0][1] = block_start_index + morton_combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs0_interleaved);
  indices[0][1][0] = block_start_index + morton_combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
  indices[0][1][1] = block_start_index + morton_combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs0_interleaved);
  indices[1][0][0] = block_start_index + morton_combine_interleaved(in_block_xs0_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
  indices[1][0][1] = block_start_index + morton_combine_interleaved(in_block_xs1_interleaved, in_block_ys0_interleaved, in_block_zs1_interleaved);
  indices[1][1][0] = block_start_index + morton_combine_interleaved(in_block_xs0_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);
  indices[1][1][1] = block_start_index + morton_combine_interleaved(in_block_xs1_interleaved, in_block_ys1_interleaved, in_block_zs1_interleaved);

  int32_v buffers[2][2][2];

  for (uint32_t k = 0; k < simdlen; k++) {
    if (mask[k]) {
      buffers[0][0][0][k] = volume[indices[0][0][0][k]];
      buffers[0][0][1][k] = volume[indices[0][0][1][k]];
      buffers[0][1][0][k] = volume[indices[0][1][0][k]];
      buffers[0][1][1][k] = volume[indices[0][1][1][k]];
      buffers[1][0][0][k] = volume[indices[1][0][0][k]];
      buffers[1][0][1][k] = volume[indices[1][0][1][k]];
      buffers[1][1][0][k] = volume[indices[1][1][0][k]];
      buffers[1][1][1][k] = volume[indices[1][1][1][k]];
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
    std::stringstream wstream(argv[2]);
    std::stringstream hstream(argv[3]);
    std::stringstream dstream(argv[4]);
    wstream >> width;
    hstream >> height;
    dstream >> depth;
  }


  RawVolume volume(argv[1], width, height, depth);
  if (!volume) {
    std::cerr << "couldn't open volume\n";
    return 1;
  }

  uint32_t x_stride = 1;
  uint32_t y_stride = width * x_stride;
  uint32_t z_stride = height * y_stride;

  uint32_t width_in_blocks  = (width  + 14) / 15;
  uint32_t height_in_blocks = (height + 14) / 15;
  uint32_t depth_in_blocks  = (depth  + 14) / 15;

  uint64_t block_x_stride = 4096;
  uint64_t block_y_stride = width_in_blocks * block_x_stride;
  uint64_t block_z_stride = height_in_blocks * block_y_stride;

  uint64_t blocked_size = depth_in_blocks * block_z_stride;

  std::string blocked_file_name = std::string(argv[1]) + ".blocked";

  uint8_t *blocked_volume {};
  {
    if (access(blocked_file_name.c_str(), F_OK)) {
      int fd = open(blocked_file_name.c_str(), O_RDWR | O_CREAT, 00777);
      lseek(fd, blocked_size - 1, SEEK_SET);
      write(fd, "", 1);

      blocked_volume = (uint8_t *)mmap(nullptr, blocked_size, PROT_WRITE, MAP_SHARED, fd, 0);

      close(fd);

      if (blocked_volume == MAP_FAILED) {
        std::cerr << "mmap: " << std::strerror(errno) << '\n';
        return 1;
      }

      for (uint32_t block_z = 0; block_z < depth_in_blocks; block_z++) {
        for (uint32_t block_y = 0; block_y < height_in_blocks; block_y++) {
          for (uint32_t block_x = 0; block_x < width_in_blocks; block_x++) {

            uint8_t *block = &blocked_volume[block_z * block_z_stride + block_y * block_y_stride + block_x * block_x_stride];

            for (uint32_t z = 0; z < 16; z++) {

              uint32_t z_interleaved = interleave_4b_3d(z);
              uint64_t original_z = std::min(block_z * 15 + z, depth - 1);

              for (uint32_t y = 0; y < 16; y++) {

                uint32_t y_interleaved = interleave_4b_3d(y);
                uint64_t original_y = std::min(block_y * 15 + y, height - 1);

                for (uint32_t x = 0; x < 16; x++) {

                  uint32_t x_interleaved = interleave_4b_3d(x);
                  uint64_t original_x = std::min(block_x * 15 + x, width - 1);

                  uint32_t in_block_index = morton_combine_interleaved(x_interleaved, y_interleaved, z_interleaved);
                  uint64_t original_index = original_z * z_stride + original_y * y_stride + original_x * x_stride;

                  block[in_block_index] = ((const uint8_t *)(const void *)volume)[original_index];
                }
              }
            }

          }
        }
      }

      munmap(blocked_volume, blocked_size);
    }
  }

  {
    int fd = open(blocked_file_name.c_str(), O_RDONLY);

    blocked_volume = (uint8_t *)mmap(nullptr, blocked_size, PROT_READ, MAP_SHARED, fd, 0);

    close(fd);

    if (blocked_volume == MAP_FAILED) {
      std::cerr << "mmap: " << std::strerror(errno) << '\n';
      return 1;
    }
  }


  std::array<float, 257> transferR {};
  std::array<float, 257> transferG {};
  std::array<float, 257> transferB {};
  std::array<float, 257> transferA {};

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

      for (uint32_t i = 0; i < 256; i++) {
        glm::vec3 color = colorGradient.color(i);
        float alpha = alphaGradient.color(i);
        transferR[i] = color.r;
        transferG[i] = color.g;
        transferB[i] = color.b;
        transferA[i] = alpha;
      }
  }

  std::vector<uint8_t> raster(window_width * window_height * 3);

  size_t rgb_x_stride = 3;
  size_t rgb_y_stride = window_width * rgb_x_stride;

  uint32_v pixel_offsets = uint32_v::IndexesFromZero() * 3;

  float time = 0.f;
  while (!glfwWindowShouldClose(window) && time < 10.f) {

    glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0.0, 1.0, 0.0)) * glm::rotate(glm::mat4(1.0f), glm::radians(90.f), glm::vec3(1.0, 0.0, 0.0));

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

        constexpr float stepsize = 0.01f;

        for (float_m mask = tmins <= tmaxs; !mask.isEmpty(); mask &= tmins <= tmaxs) {
          float_v steps = Vc::min(stepsize, tmaxs - tmins);

          glm::vec<3, float_v> vs = glm::vec<3, float_v>(origin) + ray_directions * (tmins + steps / 2.f) + float_v(0.5f);

          float_v values = sampler3DBlocked(blocked_volume, block_z_stride, block_y_stride, block_x_stride, vs.x * (width - 1), vs.y * (height - 1), vs.z * (depth - 1), mask);
          //float_v values = sampler3D(volume, vs.x * (width - 1), vs.y * (height - 1), vs.z * (depth - 1), mask);

          float_v r = sampler1D(transferR.data(), values);
          float_v g = sampler1D(transferG.data(), values);
          float_v b = sampler1D(transferB.data(), values);
          float_v a = sampler1D(transferA.data(), values);

          a = float_v(1.f) - Vc::exp(-a * stepsize);

          // Evaluate the current opacity
          dsts.r(mask) += r * a * dsts.a;
          dsts.g(mask) += g * a * dsts.a;
          dsts.b(mask) += b * a * dsts.a;
          dsts.a(mask) *= 1 - a;

          mask &= dsts.a > 0.01f;

          tmins(mask) += stepsize;
        }

        float_v rs = (dsts.r) * 255;
        float_v gs = (dsts.g) * 255;
        float_v bs = (dsts.b) * 255;

        uint8_t *rgb_start = raster.data() + j * rgb_y_stride + i * rgb_x_stride;

        rs.scatter(rgb_start + 0, pixel_offsets);
        gs.scatter(rgb_start + 1, pixel_offsets);
        bs.scatter(rgb_start + 2, pixel_offsets);
      }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, raster.data());
    glfwSwapBuffers(window);
    glfwPollEvents();

    time += 0.1f;
  }
}
