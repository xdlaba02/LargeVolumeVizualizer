
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

    void clear() { m_colors.clear(); }
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

float_v sampler3D(const RawVolume &volume, float_v xs, float_v ys, float_v zs, float_m mask) {
  xs *= 255.f;
  ys *= 255.f;
  zs *= 255.f;

  uint32_v pix_xs = xs;
  uint32_v pix_ys = ys;
  uint32_v pix_zs = zs;

  float_v frac_xs = xs - pix_xs;
  float_v frac_ys = ys - pix_ys;
  float_v frac_zs = zs - pix_zs;

  const uint8_t *data = static_cast<uint8_t *>((void *)volume);
  uint32_t width = volume.width();
  uint32_t height = volume.height();
  uint32_t depth = volume.depth();

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

        indices_xyz(pix_xs < (width - 2)) += 1; // FIXME
      }

      indices_yz(pix_ys < (height - 2)) += width; // FIXME
    }

    indices_z(pix_zs < (depth - 2)) += height; // FIXME
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
  glm::vec3 volume_min {-.5, -.5, -.5};
  glm::vec3 volume_max {+.5, +.5, +.5};

  constexpr std::size_t width = 1920;
  constexpr std::size_t height = 1080;
  constexpr float stepsize = 0.1f;

  std::vector<uint8_t> raster(width * height * 3);

  if (!glfwInit()) {
      printf("Couldn't init GLFW\n");
      return 1;
  }

  GLFWwindow *window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
  if (!window) {
      printf("Couldn't open window\n");
      return 1;
  }

  glfwMakeContextCurrent(window);

  constexpr float aspect = float(width) / float(height);
  float cameraFOV = std::tan(45 * M_PI / 180 * 0.5);

  RawVolume volume(argv[1], 256, 256, 256);
  if (!volume) {
    return 1;
  }

  ColorGradient1D<glm::vec4> transfer {};

  transfer.setColor(35.f,  {0.0f, 0.90f, 0.0f, 0.0f});
  transfer.setColor(40.f,  {0.0f, 0.90f, 0.0f, 0.5f});
  transfer.setColor(58.f,  {0.0f, 0.90f, 0.0f, 0.5f});
  transfer.setColor(59.f,  {0.0f, 0.00f, 0.0f, 0.5f});
  transfer.setColor(60.f,  {0.7f, 0.35f, 0.0f, 0.5f});
  transfer.setColor(255.f, {0.9f, 0.55f, 0.0f, 0.8f});

  uint32_v offsets {};

  for (std::size_t k = 0; k < simdlen; k++) {
    offsets[k] = k * 3;
  }

  float time = 0.f;
  while (!glfwWindowShouldClose(window)) {

    glm::vec3 origin = glm::normalize(glm::vec3(std::sin(time) * 2, std::cos(time), std::cos(time) * 2)) * (std::sin(time / 4) + 3);

    glm::mat4 view = glm::lookAt(origin, { 0, 0, 0 }, {0, 1, 0 });

    #pragma omp parallel for
    for (uint32_t j = 0; j < height; j++) {
      float y = (2 * (j + 0.5) / height - 1) * cameraFOV;

      for (uint32_t i = 0; i < width; i += simdlen) {
        float_v is = float_v(Vc::IndexesFromZero) + i;

        float_v xs = (2 * (is + 0.5f) / float(width) - 1) * aspect * cameraFOV;

        glm::vec<3, float_v> ray_directions {};

        for (std::size_t k = 0; k < simdlen; k++) {
          glm::vec4 direction = glm::normalize(glm::vec4(xs[k], y, -1, 0) * view);
          ray_directions.x[k] = direction.x;
          ray_directions.y[k] = direction.y;
          ray_directions.z[k] = direction.z;
        }

        float_v tmins, tmaxs;
        intersect_aabb_rays_single_origin(origin, ray_directions, volume_min, volume_max, tmins, tmaxs);

        glm::vec<4, float_v> dsts(0.f);

        for (float_m mask = tmins <= tmaxs; !mask.isEmpty(); mask = tmins <= tmaxs) {
          float_v steps = Vc::min(stepsize, tmaxs - tmins);

          glm::vec<3, float_v> vs = glm::vec<3, float_v>(origin) + ray_directions * (tmins + steps / 2.f) + float_v(0.5f);

          float_v values = sampler3D(volume, vs.x, vs.y, vs.z, mask);

          glm::vec<4, float_v> srcs {};
          for (std::size_t k = 0; k < simdlen; k++) {
            glm::vec4 src = transfer.color(values[k]);
            srcs.x[k] = src.x;
            srcs.y[k] = src.y;
            srcs.z[k] = src.z;
            srcs.w[k] = src.w;
          }

          srcs.a *= steps;

          srcs.r *= srcs.a;
          srcs.g *= srcs.a;
          srcs.b *= srcs.a;

          // Evaluate the current opacity
          glm::vec<4, float_v> tmp = (1 - dsts.a) * srcs;
          dsts.x(mask) += tmp.x;
          dsts.y(mask) += tmp.y;
          dsts.z(mask) += tmp.z;
          dsts.w(mask) += tmp.w;

          tmins += stepsize;
        }

        float_v rs = (dsts.r + (1 - dsts.a)) * 255;
        float_v gs = (dsts.g + (1 - dsts.a)) * 255;
        float_v bs = (dsts.b + (1 - dsts.a)) * 255;

        rs.scatter(raster.data() + (j * width + i) * 3 + 0, offsets);
        gs.scatter(raster.data() + (j * width + i) * 3 + 1, offsets);
        bs.scatter(raster.data() + (j * width + i) * 3 + 2, offsets);
      }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, raster.data());
    glfwSwapBuffers(window);
    glfwPollEvents();

    time += 0.1f;
  }
}
