#pragma once

#include <ray/traversal_octree_packlet.h>

#include <utils/ray_generator.h>
#include <utils/simd.h>
#include <utils/glm_simd.h>

#include <glm/glm.hpp>

#include <cstdint>

template <typename F>
void render_scalar(uint32_t width, uint32_t height, float fov, const glm::mat4 &vmt, const F &integrator) {
  glm::mat4 ray_transform = glm::inverse(vmt);

  glm::vec3 ray_origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

  RayGenerator ray_generator(width, height, fov);

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      glm::vec3 dir = ray_transform * ray_generator(x, y);
      integrator({ ray_origin, dir, 1.f / dir });
    }
  }
}

template <typename F>
void render_simd(uint32_t width, uint32_t height, float fov, const glm::mat4 &vmt, const F &integrator) {
  glm::mat4 ray_transform = glm::inverse(vmt);

  glm::vec3 ray_origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

  RayGenerator ray_generator(width, height, fov);

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x += simd::len) {

      simd::float_m mask {};
      simd::vec3 ray_direction;

      for (uint32_t k = 0; k < simd::len && x + k < width; k++) {
        glm::vec3 dir = ray_transform * ray_generator(x + k, y);

        ray_direction.x[k] = dir.x;
        ray_direction.y[k] = dir.y;
        ray_direction.z[k] = dir.z;

        mask[k] = true;
      }

      integrator({ ray_origin, ray_direction, simd::float_v(1.f) / ray_direction }, mask);
    }
  }
}


template <typename F>
void render_packlet(uint32_t width, uint32_t height, float fov, const glm::mat4 &vmt, const F &integrator) {
  glm::mat4 ray_transform = glm::inverse(vmt);
  
  glm::vec3 ray_origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

  RayGenerator ray_generator(width, height, fov);

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

      integrator(ray_packlet, mask_packlet);
    }
  }
}
