
#include "raw_volume.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <GLFW/glfw3.h>

#include <experimental/simd>

#include <iostream>
#include <vector>
#include <cmath>

using simdline = std::experimental::native_simd;

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


bool intersect_aabb_ray(const glm::vec3& origin, const glm::vec3& direction, const glm::vec3 &min, const glm::vec3 &max, float& tMin, float& tMax) {
  glm::vec3 inv_direction = glm::vec3{1, 1, 1} / direction;

  tMin = -std::numeric_limits<float>::infinity();
  tMax = std::numeric_limits<float>::infinity();

  for (size_t i = 0; i < 3; ++i) {
    float t0, t1;

    if (inv_direction[i] >= 0.f) {
      t0 = (min[i] - origin[i]) * inv_direction[i];
      t1 = (max[i] - origin[i]) * inv_direction[i];
    }
    else {
      t1 = (min[i] - origin[i]) * inv_direction[i];
      t0 = (max[i] - origin[i]) * inv_direction[i];
    }

    tMin = t0 > tMin ? t0 : tMin;
    tMax = t1 < tMax ? t1 : tMax;
  }

  return tMax + std::numeric_limits<float>::epsilon() >= tMin;
}

int main(int argc, char *argv[]) {

  glm::vec3 volume_min {-.5, -.5, -.5};
  glm::vec3 volume_max {+.5, +.5, +.5};

  constexpr size_t width = 1920;
  constexpr size_t height = 1080;

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

  auto sampler = [&volume](float x, float y, float z) {
    x *= 255;
    y *= 255;
    z *= 255;

    size_t pix_x = x;
    size_t pix_y = y;
    size_t pix_z = z;

    float frac_x = x - pix_x;
    float frac_y = y - pix_y;
    float frac_z = z - pix_z;

    float acc0 = volume(pix_x, pix_y,     pix_z    ) * (1 - frac_x) + volume(pix_x + 1, pix_y    , pix_z    ) * frac_x;
    float acc1 = volume(pix_x, pix_y + 1, pix_z    ) * (1 - frac_x) + volume(pix_x + 1, pix_y + 1, pix_z    ) * frac_x;
    float acc2 = volume(pix_x, pix_y,     pix_z + 1) * (1 - frac_x) + volume(pix_x + 1, pix_y    , pix_z + 1) * frac_x;
    float acc3 = volume(pix_x, pix_y + 1, pix_z + 1) * (1 - frac_x) + volume(pix_x + 1, pix_y + 1, pix_z + 1) * frac_x;

    acc0 = acc0 * (1 - frac_y) + acc1 * frac_y;
    acc1 = acc2 * (1 - frac_y) + acc3 * frac_y;

    acc0 = acc0 * (1 - frac_z) + acc1 * frac_z;

    return acc0;
  };

  ColorGradient1D<glm::vec4> transfer {};

  transfer.setColor(35.f,  {0.0f, 0.90f, 0.0f, 0.0f});
  transfer.setColor(40.f,  {0.0f, 0.90f, 0.0f, 0.5f});
  transfer.setColor(58.f,  {0.0f, 0.90f, 0.0f, 0.5f});
  transfer.setColor(59.f,  {0.0f, 0.00f, 0.0f, 0.5f});
  transfer.setColor(60.f,  {0.7f, 0.35f, 0.0f, 0.5f});
  transfer.setColor(255.f, {0.9f, 0.55f, 0.0f, 0.8f});

  float time = 0.f;
  while (!glfwWindowShouldClose(window)) {

    glm::vec3 origin = glm::normalize(glm::vec3(std::sin(time) * 2, std::cos(time), std::cos(time) * 2)) * (std::sin(time / 4) + 3);

    glm::mat4 view = glm::lookAt(origin, { 0, 0, 0 }, {0, 1, 0 });

    #pragma omp parallel for
    for (uint32_t j = 0; j < height; j++) {
      float y = (2 * (j + 0.5) / height - 1) * cameraFOV;

      for (uint32_t i = 0; i < width; i++) {
        float x = (2 * (i + 0.5) / width - 1) * aspect * cameraFOV;

        glm::vec3 ray_direction = glm::normalize(glm::vec4(x, y, -1, 0) * view);

        float tmin, tmax;
        intersect_aabb_ray(origin, ray_direction, volume_min, volume_max, tmin, tmax);

        glm::vec4 dst(0.f);

        constexpr float stepsize = 0.1f;

        while (tmin <= tmax) {
          float step = std::min(stepsize, tmax - tmin);
          glm::vec3 v = glm::vec3(origin + ray_direction * (tmin + step / 2.f)) + 0.5f;

          float value = sampler(v.x, v.y, v.z);

          glm::vec4 src = transfer.color(value);

          src.a = 1.f - std::pow(1.f - src.a, step);

          src.r *= src.a;
          src.g *= src.a;
          src.b *= src.a;

          // Evaluate the current opacity
          dst += (1 - dst.a) * src;

          tmin += stepsize;
        }

        raster[(j * width + i) * 3 + 0] = (dst.r + (1 - dst.a)) * 255;
        raster[(j * width + i) * 3 + 1] = (dst.g + (1 - dst.a)) * 255;
        raster[(j * width + i) * 3 + 2] = (dst.b + (1 - dst.a)) * 255;
      }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, raster.data());
    glfwSwapBuffers(window);
    glfwPollEvents();

    time += 0.1f;
  }
}
