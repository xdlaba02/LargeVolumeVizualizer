
#include "raw_volume.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <GLFW/glfw3.h>

#include <Vc/Vc>

#include <iostream>
#include <vector>
#include <cmath>

using Vc::float_v;
using Vc::float_m;

constexpr std::size_t simdlen = float_v::size();

using uint32_v = Vc::SimdArray<uint32_t, simdlen>;

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


void intersect_aabb_rays_single_origin(const glm::vec3& origin, glm::vec<3, float_v> ray_directions, const glm::vec3 &min, const glm::vec3 &max, float_v& tmins, float_v& tmaxs) {
  ray_directions = glm::vec<3, float_v>{1, 1, 1} / ray_directions;

  tmins = -std::numeric_limits<float>::infinity();
  tmaxs = std::numeric_limits<float>::infinity();

  for (std::size_t i = 0; i < 3; ++i) {
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

float_v sampler1D(const float *data, uint32_t size, float_v values, float_m mask) {
  uint32_v pix = values;
  float_v frac = values - pix;

  float_v accs[2];

  accs[0].gather(data, pix, mask);

  pix(pix < (size - 1))++;

  accs[1].gather(data, pix, mask);

  return accs[0] * (1 - frac) + accs[1] * frac;
};

float_v sampler3D(const RawVolume &volume, float_v xs, float_v ys, float_v zs, float_m mask) {
  const uint8_t *data = static_cast<uint8_t *>((void *)volume);
  uint32_t width = volume.width();
  uint32_t height = volume.height();
  uint32_t depth = volume.depth();

  xs *= width - 1;
  ys *= height - 1;
  zs *= depth - 1;

  uint32_v pix_xs = xs;
  uint32_v pix_ys = ys;
  uint32_v pix_zs = zs;

  float_v frac_xs = xs - pix_xs;
  float_v frac_ys = ys - pix_ys;
  float_v frac_zs = zs - pix_zs;

  float_m incrementable_xs = pix_xs < (width - 1);
  float_m incrementable_ys = pix_ys < (height - 1);
  float_m incrementable_zs = pix_zs < (depth - 1);

  float_v accs[2][2][2];

  float_v::IndexType indices_z = (pix_zs * height + pix_ys) * width + pix_xs;
  for (size_t z = 0; z < 2; z++) {

    float_v::IndexType indices_yz = indices_z;
    for (size_t y = 0; y < 2; y++) {

      float_v::IndexType indices_xyz = indices_yz;
      for (size_t x = 0; x < 2; x++) {

        for (uint32_t k = 0; k < simdlen; k++) {
          if (mask[k]) {
            accs[z][y][x][k] = data[indices_xyz[k]];
          }
        }

        indices_xyz(incrementable_xs) += 1;
      }

      indices_yz(incrementable_ys) += width;
    }

    indices_z(incrementable_zs) += width * height;
  }

  accs[0][0][0] = accs[0][0][0] * (1 - frac_xs) + accs[0][0][1] * frac_xs;
  accs[0][1][0] = accs[0][1][0] * (1 - frac_xs) + accs[0][1][1] * frac_xs;
  accs[1][0][0] = accs[1][0][0] * (1 - frac_xs) + accs[1][0][1] * frac_xs;
  accs[1][1][0] = accs[1][1][0] * (1 - frac_xs) + accs[1][1][1] * frac_xs;

  accs[0][0][0] = accs[0][0][0] * (1 - frac_ys) + accs[0][1][0] * frac_ys;
  accs[1][0][0] = accs[1][0][0] * (1 - frac_ys) + accs[1][1][0] * frac_ys;

  accs[0][0][0] = accs[0][0][0] * (1 - frac_zs) + accs[1][0][0] * frac_zs;

  return accs[0][0][0];
};

int main(int argc, char *argv[]) {
  if (!glfwInit()) {
      printf("Couldn't init GLFW\n");
      return 1;
  }

  constexpr std::size_t width = 1920;
  constexpr std::size_t height = 1080;

  GLFWwindow *window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
  if (!window) {
      printf("Couldn't open window\n");
      return 1;
  }

  glfwMakeContextCurrent(window);

  RawVolume volume(argv[1], 256, 256, 256);
  if (!volume) {
    return 1;
  }

  std::array<float, 256> transferR {};
  std::array<float, 256> transferG {};
  std::array<float, 256> transferB {};
  std::array<float, 256> transferA {};

  {
      ColorGradient1D<glm::vec3> colorGradient {};

      colorGradient.setColor(80.f,  {0.75f, 0.5f, 0.25f});
      colorGradient.setColor(82.f,  {1.00f, 1.0f, 0.85f});

      ColorGradient1D<float> alphaGradient {};

      alphaGradient.setColor(40.f,  0.0f);
      alphaGradient.setColor(60.f,  0.1f);
      alphaGradient.setColor(63.f,  0.05f);
      alphaGradient.setColor(80.f,  0.00f);
      alphaGradient.setColor(82.f,  2.00f);
      alphaGradient.setColor(255.f, 5.00f);

      for (size_t i = 0; i < 256; i++) {
        glm::vec3 color = colorGradient.color(i);
        float alpha = alphaGradient.color(i);
        transferR[i] = color.r;
        transferG[i] = color.g;
        transferB[i] = color.b;
        transferA[i] = alpha;
      }
  }

  std::vector<uint8_t> raster(width * height * 3);

  float time = 0.f;
  while (!glfwWindowShouldClose(window)) {

    glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0.0, 1.0, 0.0)) * glm::rotate(glm::mat4(1.0f), glm::radians(90.f), glm::vec3(1.0, 0.0, 0.0)) * glm::scale(glm::mat4(1.0f), glm::vec3(1.25f, 1.f, 1.f));

    glm::vec3 origin = glm::vec3(1, 0, 0) * 2.f;
    glm::mat4 view = glm::lookAt(origin, { 0, 0, 0 }, {0, 1, 0 }) * model;

    origin = glm::vec4(origin, 1) * model;

    #pragma omp parallel for
    for (uint32_t j = 0; j < height; j++) {

      constexpr float cameraFOV = std::tan(45 * M_PI / 180 * 0.5);

      float y = (2 * (j + 0.5) / height - 1) * cameraFOV;

      for (uint32_t i = 0; i < width; i += simdlen) {
        float_v is = float_v::IndexesFromZero() + i;

        constexpr float aspect = float(width) / float(height);

        float_v xs = (2 * (is + 0.5f) / float(width) - 1) * aspect * cameraFOV;

        glm::vec<3, float_v> ray_directions {};

        for (std::size_t k = 0; k < simdlen; k++) {
          glm::vec4 direction = glm::normalize(glm::vec4(xs[k], y, -1, 0) * view);
          ray_directions.x[k] = direction.x;
          ray_directions.y[k] = direction.y;
          ray_directions.z[k] = direction.z;
        }

        float_v tmins, tmaxs;
        intersect_aabb_rays_single_origin(origin, ray_directions, {-.5, -.5, -.5}, {+.5, +.5, +.5}, tmins, tmaxs);

        glm::vec<4, float_v> dsts(0.f);

        constexpr float stepsize = 0.01f;

        for (float_m mask = tmins <= tmaxs; !mask.isEmpty(); mask &= tmins <= tmaxs) {
          float_v steps = Vc::min(stepsize, tmaxs - tmins);

          glm::vec<3, float_v> vs = glm::vec<3, float_v>(origin) + ray_directions * (tmins + steps / 2.f) + glm::vec<3, float_v>{float_v(0.5f), float_v(0.5f), float_v(0.5f)};

          float_v values = sampler3D(volume, vs.x, vs.y, vs.z, mask);

          glm::vec<4, float_v> srcs {
            sampler1D(transferR.data(), transferR.size(), values, mask),
            sampler1D(transferG.data(), transferG.size(), values, mask),
            sampler1D(transferB.data(), transferB.size(), values, mask),
            sampler1D(transferA.data(), transferA.size(), values, mask)
          };

          srcs.a = float_v(1.f) - Vc::exp(-srcs.a * steps * 5);

          srcs.r *= srcs.a;
          srcs.g *= srcs.a;
          srcs.b *= srcs.a;

          // Evaluate the current opacity
          glm::vec<4, float_v> tmp = (1 - dsts.a) * srcs;
          dsts.r(mask) += tmp.r;
          dsts.g(mask) += tmp.g;
          dsts.b(mask) += tmp.b;
          dsts.a(mask) += tmp.a;

          mask &= dsts.a < 0.99f;

          tmins(mask) += stepsize;
        }

        float_v rs = (dsts.r + (1 - dsts.a)) * 255;
        float_v gs = (dsts.g + (1 - dsts.a)) * 255;
        float_v bs = (dsts.b + (1 - dsts.a)) * 255;

        rs.scatter(raster.data() + (j * width + i) * 3 + 0, uint32_v::IndexesFromZero() * 3);
        gs.scatter(raster.data() + (j * width + i) * 3 + 1, uint32_v::IndexesFromZero() * 3);
        bs.scatter(raster.data() + (j * width + i) * 3 + 2, uint32_v::IndexesFromZero() * 3);
      }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, raster.data());
    glfwSwapBuffers(window);
    glfwPollEvents();

    time += 0.1f;
  }
}
