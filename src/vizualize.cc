
#include "vizualize_args.h"
#include "glfw.h"

#include <tree_volume/tree_volume.h>
#include <tree_volume/renderer_tree.h>
#include <tree_volume/renderer_tree_simd.h>
#include <tree_volume/renderer_tree_packlet.h>
#include <tree_volume/renderer_layer.h>
#include <tree_volume/renderer_layer_simd.h>
#include <tree_volume/renderer_layer_dda.h>

#include <utils/linear_gradient.h>
#include <utils/preintegrated_transfer_function.h>
#include <utils/ray_generator.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

#include <iostream>
#include <vector>
#include <chrono>

// Phong shading
// TODO Determinace uzlu
// TODO Rozhrani
// TODO Precomputes?
// TODO Camera class
// TODO Nicer Sampler1D and Sampler2D

int main(int argc, char *argv[]) {
  const char *processed_volume;
  const char *processed_metadata;
  uint32_t width;
  uint32_t height;
  uint32_t depth;

  if (!parse_args(argc, argv, processed_volume, processed_metadata, width, height, depth)) {
    return 1;
  }

  TreeVolume<uint8_t> volume(processed_volume, processed_metadata, width, height, depth);

  std::map<float, glm::vec3> color_map {
    #if 1
    {80.f,  {0.75f, 0.5f, 0.25f}},
    {82.f,  {1.00f, 1.0f, 0.85f}}
    #else
    {0.f,  {1.00f, 0.0f, 0.0f}},
    {89.f,  {0.00f, 1.0f, 0.0f}},
    {178.f,  {0.00f, 0.0f, 1.0f}},
    #endif
  };

  std::map<float, float> alpha_map {
    #if 1
    {40.f,  000.f},
    {60.f,  001.f},
    {63.f,  005.f},
    {80.f,  000.f},
    {82.f,  100.f},
    #else
    {0.f,   0.f},
    {89.f,  5.f},
    {178.f, 0.f},
    #endif
  };

  PreintegratedTransferFunction<uint8_t> preintegrated_r([&](float v){ return linear_gradient(color_map, v).r; });
  PreintegratedTransferFunction<uint8_t> preintegrated_g([&](float v){ return linear_gradient(color_map, v).g; });
  PreintegratedTransferFunction<uint8_t> preintegrated_b([&](float v){ return linear_gradient(color_map, v).b; });
  PreintegratedTransferFunction<uint8_t> preintegrated_a([&](float v){ return linear_gradient(alpha_map, v); });

  auto transfer_function = [&](const auto &v0, const auto &v1, const auto &mask) {
    return Vec4Vec(preintegrated_r(v0, v1, mask), preintegrated_g(v0, v1, mask), preintegrated_b(v0, v1, mask), preintegrated_a(v0, v1, mask));
  };

  auto transfer_function_scalar = [&](const auto &v0, const auto &v1) {
    return glm::vec4(preintegrated_r(v0, v1), preintegrated_g(v0, v1), preintegrated_b(v0, v1), preintegrated_a(v0, v1));
  };

  GLFW::Window window(1920, 1080, "Volumetric Vizualizer");

  auto prev_time = std::chrono::steady_clock::now();

  glm::vec2 prev_pos;
  window.getCursor(prev_pos.x, prev_pos.y);

  float yaw = -90;
  float pitch = 0;

  glm::vec3 camera_pos   = glm::vec3(0.0f, 0.0f, 2.0f);
  constexpr glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::vec3 volume_pos   = glm::vec3(0.0f, 0.0f, 0.0f);

  glm::vec3 volume_size(1.f / volume.info.width_frac, 1.f / volume.info.height_frac, 1.f / volume.info.depth_frac);

  float t = 0.f;
  while (!window.shouldClose()) {

    auto time = std::chrono::steady_clock::now();
    float delta = std::chrono::duration_cast<std::chrono::milliseconds>(time - prev_time).count() / 1000.f;
    std::cerr << delta << "\n";
    prev_time = time;

    t += delta;

    {
      glm::vec2 pos;

      window.getCursor(pos.x, pos.y);

      constexpr float sensitivity = 0.1f;
      glm::vec2 offset = (pos - prev_pos) * sensitivity;

      prev_pos = pos;

      yaw   += offset.x;
      pitch -= offset.y;

      pitch = std::clamp(pitch, -89.f, 89.f);
    }

    glm::vec3 camera_front = glm::normalize(glm::vec3{
      std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch)),
      std::sin(glm::radians(pitch)),
      std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch)),
    });

    {
      const float speed = 1.f * delta;

      if (window.getKey(GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        window.shouldClose(true);
      }

      if (window.getKey(GLFW_KEY_W) == GLFW_PRESS) {
        camera_pos += speed * camera_front;
      }

      if (window.getKey(GLFW_KEY_S) == GLFW_PRESS) {
        camera_pos -= speed * camera_front;
      }

      if (window.getKey(GLFW_KEY_A) == GLFW_PRESS) {
        camera_pos -= glm::normalize(glm::cross(camera_front, camera_up)) * speed;
      }

      if (window.getKey(GLFW_KEY_D) == GLFW_PRESS) {
        camera_pos += glm::normalize(glm::cross(camera_front, camera_up)) * speed;
      }

      if (window.getKey(GLFW_KEY_Q) == GLFW_PRESS) {
        volume_pos.x -= speed;
      }

      if (window.getKey(GLFW_KEY_E) == GLFW_PRESS) {
        volume_pos.x += speed;
      }
    }

    // TODO hide transforms into object and camera class
    // by default, the volume is rendered in the interval [0, volume.info.frac]
    glm::mat4 model = glm::translate(glm::mat4(1.f), volume_pos)
                    * glm::rotate(glm::mat4(1.f), 0.f, glm::vec3(0.f, 1.f, 0.f))
                    * glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f))
                    * glm::translate(glm::mat4(1.f), glm::vec3(-0.5f))
                    * glm::scale(glm::mat4(1.f), volume_size);

    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);

    glm::mat4 ray_transform = glm::inverse(view * model);

    glm::vec3 ray_origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

    RayGenerator ray_generator(window.width(), window.height(), 45.f);

    float step = 0.001f;

    #pragma omp parallel for schedule(dynamic)
    for (uint32_t y = 0; y < window.height(); y += simd::len) {
      for (uint32_t x = 0; x < window.width(); x += simd::len) {

        RayPacklet ray_packlet {};
        MaskPacklet mask_packlet {};
        Vec4Packlet output_packlet {};

        for (uint32_t yy = 0; yy < simd::len && y + yy < window.height(); yy++) {

          Vec3Vec ray_direction;
          for (uint32_t xx = 0; xx < simd::len && x + xx < window.width(); xx++) {
            glm::vec3 dir = ray_transform * ray_generator(x + xx, y + yy);

            /*
            glm::vec4 output = render_layer(volume, { ray_origin, dir, 1.f / dir }, 0, step, transfer_function_scalar);
            glm::vec4 output = render_tree(volume, { ray_origin, dir, 1.f / dir }, step, transfer_function_scalar);
            */
            glm::vec4 output = render_layer_dda(volume, { ray_origin, dir, 1.f / dir }, 0, step, transfer_function_scalar); // FIXME not working again
            output_packlet[yy].x[xx] = output.x;
            output_packlet[yy].y[xx] = output.y;
            output_packlet[yy].z[xx] = output.z;

            ray_direction.x[xx] = dir.x;
            ray_direction.y[xx] = dir.y;
            ray_direction.z[xx] = dir.z;

            mask_packlet[yy][xx] = true;
          }

          ray_packlet[yy] = { ray_origin, ray_direction, simd::float_v(1.f) / ray_direction };

          //output_packlet[yy] = render_tree(volume, ray_packlet[yy], step, mask_packlet[yy], transfer_function);
          //output_packlet[yy] = render_layer(volume, ray_packlet[yy], mask_packlet[yy], 0, step, transfer_function);
        }

        //output_packlet = render_tree(volume, ray_packlet, step, mask_packlet, transfer_function);

        for (uint32_t yy = 0; yy < simd::len; yy++) {
          output_packlet[yy].r *= 255.f;
          output_packlet[yy].g *= 255.f;
          output_packlet[yy].b *= 255.f;

          for (uint32_t xx = 0; xx < simd::len; xx++) {
            if (mask_packlet[yy][xx]) {
              window.raster(x + xx, y + yy, 0) = output_packlet[yy].r[xx];
              window.raster(x + xx, y + yy, 1) = output_packlet[yy].g[xx];
              window.raster(x + xx, y + yy, 2) = output_packlet[yy].b[xx];
            }
          }
        }
      }
    }

    window.swapBuffers();
    window.pollEvents();
  }
}
