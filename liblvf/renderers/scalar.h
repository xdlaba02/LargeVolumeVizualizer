/**
* @file scalar.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Function that generates rays for a viewport with specific persepective projection and view matrix.
*/

#pragma once

#include <utils/ray_generator.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename F>
void render_scalar(uint32_t width, uint32_t height, float fov, const glm::mat4 &vmt, uint8_t *rgb_buffer, const F &integrator) {
  glm::mat4 ray_transform = glm::inverse(vmt);

  glm::vec3 ray_origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

  RayGenerator ray_generator(width, height, fov);

  #pragma omp parallel for schedule(dynamic)
  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {

      glm::vec3 dir = ray_transform * ray_generator(x, y);

      glm::vec4 output = integrator({ ray_origin, dir, 1.f / dir });

      rgb_buffer[y * width * 3 + x * 3 + 0] = output.r * 255.f;
      rgb_buffer[y * width * 3 + x * 3 + 1] = output.g * 255.f;
      rgb_buffer[y * width * 3 + x * 3 + 2] = output.b * 255.f;
    }
  }
}
