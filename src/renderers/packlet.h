#pragma once

#include <utils/ray_generator.h>

#include <ray/traversal_octree_packlet.h>

template <typename F>
void render_packlet(uint32_t width, uint32_t height, float fov, const glm::mat4 &model, const glm::mat4 &view, uint8_t *rgb_buffer, const F &integrator) {
  glm::mat4 ray_transform = glm::inverse(view * model);

  glm::vec3 ray_origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

  RayGenerator ray_generator(width, height, fov);

  #pragma omp parallel for schedule(dynamic)
  for (uint32_t y = 0; y < height; y += simd::len) {
    for (uint32_t x = 0; x < width; x += simd::len) {

      RayPacklet ray_packlet {};
      MaskPacklet mask_packlet {};

      for (uint32_t yy = 0; yy < simd::len && y + yy < height; yy++) {

        simd::vec3 ray_direction;
        for (uint32_t xx = 0; xx < simd::len && x + xx < width; xx++) {
          glm::vec3 dir = ray_transform * ray_generator(x + xx, y + yy);

          ray_direction.x[xx] = dir.x;
          ray_direction.y[xx] = dir.y;
          ray_direction.z[xx] = dir.z;

          mask_packlet[yy][xx] = true;
        }

        ray_packlet[yy] = { ray_origin, ray_direction, simd::float_v(1.f) / ray_direction };
      }

      Vec4Packlet output_packlet = integrator(ray_packlet, mask_packlet);

      for (uint32_t yy = 0; yy < simd::len; yy++) {
        for (uint32_t xx = 0; xx < simd::len; xx++) {
          if (mask_packlet[yy][xx]) {
            rgb_buffer[(y + yy) * width * 3 + (x + xx) * 3 + 0] = output_packlet[yy].r[xx] * 255.f;
            rgb_buffer[(y + yy) * width * 3 + (x + xx) * 3 + 1] = output_packlet[yy].g[xx] * 255.f;
            rgb_buffer[(y + yy) * width * 3 + (x + xx) * 3 + 2] = output_packlet[yy].b[xx] * 255.f;
          }
        }
      }
    }
  }
}
