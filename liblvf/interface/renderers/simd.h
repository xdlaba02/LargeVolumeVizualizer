#pragma once

#include <components/ray_generator.h>

#include <components/simd.h>
#include <components/glm_simd.h>

template <typename F>
void render_simd(uint32_t width, uint32_t height, float fov, const glm::mat4 &vmt, uint8_t *rgb_buffer, const F &integrator) {
  glm::mat4 ray_transform = glm::inverse(vmt);

  glm::vec3 ray_origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

  RayGenerator ray_generator(width, height, fov);

  #pragma omp parallel for schedule(dynamic)
  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x += simd::len) {

      simd::float_m mask {};
      simd::vec3 ray_direction {};

      for (uint32_t k = 0; k < simd::len && x + k < width; k++) {
        glm::vec3 dir = ray_transform * ray_generator(x + k, y);

        ray_direction.x[k] = dir.x;
        ray_direction.y[k] = dir.y;
        ray_direction.z[k] = dir.z;

        mask[k] = true;
      }

      simd::vec4 output = integrator({ ray_origin, ray_direction, simd::float_v(1.f) / ray_direction }, mask);

      for (uint32_t k = 0; k < simd::len; k++) {
        if (mask[k]) {
          rgb_buffer[y * width * 3 + (x + k) * 3 + 0] = output.r[k] * 255.f;
          rgb_buffer[y * width * 3 + (x + k) * 3 + 1] = output.g[k] * 255.f;
          rgb_buffer[y * width * 3 + (x + k) * 3 + 2] = output.b[k] * 255.f;
        }
      }
    }
  }
}
