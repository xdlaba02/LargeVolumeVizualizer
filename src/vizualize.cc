
#include "vizualize_args.h"
#include "glfw.h"

#include <tree_volume/tree_volume.h>

#include <renderers/tree.h>
#include <renderers/tree_simd.h>
#include <renderers/tree_packlet.h>
#include <renderers/tree_layer.h>
#include <renderers/tree_layer_simd.h>
#include <renderers/tree_layer_dda.h>

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
    {0.f,   {1.0f, 0.0f, 0.0f}},
    {1.f,   {1.0f, 0.5f, 0.5f}},

    {2.f,   {1.0f, 1.0f, 0.0f}},
    {3.f,   {1.0f, 1.0f, 0.5f}},

    {4.f,   {0.0f, 1.0f, 0.0f}},
    {5.f,   {0.5f, 1.0f, 0.5f}},

    {6.f,   {0.0f, 1.0f, 1.0f}},
    {7.f,   {0.5f, 1.0f, 1.0f}},

    {8.f,   {0.0f, 0.0f, 1.0f}},
    {9.f,   {0.5f, 0.5f, 1.0f}},

    {10.f,   {1.0f, 0.0f, 1.0f}},
    {11.f,   {1.0f, 0.5f, 1.0f}},

    {12.f,   {1.0f, 0.0f, 0.0f}},
    {13.f,   {1.0f, 0.5f, 0.5f}},

    {14.f,   {1.0f, 1.0f, 0.0f}},
    {15.f,   {1.0f, 1.0f, 0.5f}},

    {16.f,   {0.0f, 1.0f, 0.0f}},
    {17.f,   {0.5f, 1.0f, 0.5f}},

    {18.f,   {0.0f, 1.0f, 1.0f}},
    {19.f,   {0.5f, 1.0f, 1.0f}},

    {20.f,   {0.0f, 0.0f, 1.0f}},
    {21.f,   {0.5f, 0.5f, 1.0f}},

    {22.f,   {1.0f, 0.0f, 1.0f}},
    {23.f,   {1.0f, 0.5f, 1.0f}},
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
    {0.f,   10000.f},
#endif
  };

  PreintegratedTransferFunction<uint8_t> preintegrated_r([&](float v){ return linear_gradient(color_map, v).r; });
  PreintegratedTransferFunction<uint8_t> preintegrated_g([&](float v){ return linear_gradient(color_map, v).g; });
  PreintegratedTransferFunction<uint8_t> preintegrated_b([&](float v){ return linear_gradient(color_map, v).b; });
  PreintegratedTransferFunction<uint8_t> preintegrated_a([&](float v){ return linear_gradient(alpha_map, v); });

  auto transfer_function = [&](const auto &v0, const auto &v1, const auto &mask) {
    return simd::vec4(preintegrated_r(v0, v1, mask), preintegrated_g(v0, v1, mask), preintegrated_b(v0, v1, mask), preintegrated_a(v0, v1, mask));
  };

  auto transfer_function_scalar = [&](const auto &v0, const auto &v1) {
    return glm::vec4(preintegrated_r(v0, v1), preintegrated_g(v0, v1), preintegrated_b(v0, v1), preintegrated_a(v0, v1));
  };

  GLFW::Window window(1920, 1080, "Volumetric Vizualizer");

  std::vector<uint8_t> raster(window.width() * window.height() * 3);

  auto prev_time = std::chrono::steady_clock::now();

  glm::vec2 prev_pos;
  window.getCursor(prev_pos.x, prev_pos.y);

  float yaw = -90;
  float pitch = 0;

  glm::vec3 camera_pos   = glm::vec3(0.0f, 0.0f, 2.0f);
  constexpr glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::vec3 volume_pos   = glm::vec3(0.0f, 0.0f, 0.0f);

  glm::vec3 volume_size(1.f / volume.info.width_frac, 1.f / volume.info.height_frac, 1.f / volume.info.depth_frac);

  bool imgui_show = true;

  float step = 0.001f;
  float lod = 0.05f;
  float fov = 45.f;
  float skip_thresh = 0.01f;

  float t = 0.f;
  while (!window.shouldClose()) {

    window.makeContextCurrent();


    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::Begin("Configuration", &imgui_show, 0)) {
      ImGui::SliderFloat("Step", &step, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Level of Detail", &lod, 0.f, 1.f, "%.2f", ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Camera FOV", &fov, 0.f, 180.f, "%.0f");
      ImGui::SliderFloat("Block skip threshold", &skip_thresh, 0.f, 1.f, "%.2f");
      ImGui::End();
    }

    auto time = std::chrono::steady_clock::now();
    float delta = std::chrono::duration_cast<std::chrono::milliseconds>(time - prev_time).count() / 1000.f;
    std::cerr << delta << "\n";
    prev_time = time;

    t += delta;

    {
      glm::vec2 pos;

      window.getCursor(pos.x, pos.y);

      glm::vec2 offset = pos - prev_pos;

      prev_pos = pos;

      if (window.getMouseButton(GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        window.setCursorMode(GLFW_CURSOR_DISABLED);

        constexpr float sensitivity = 0.1f;

        yaw   += offset.x * sensitivity;
        pitch -= offset.y * sensitivity;

        pitch = std::clamp(pitch, -89.f, 89.f);
      }
      else {
        window.setCursorMode(GLFW_CURSOR_NORMAL);
      }
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
                    * glm::rotate(glm::mat4(1.f), t, glm::vec3(0.f, 1.f, 0.f))
                    * glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f))
                    * glm::translate(glm::mat4(1.f), glm::vec3(-0.5f))
                    * glm::scale(glm::mat4(1.f), volume_size);

    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);

    glm::mat4 ray_transform = glm::inverse(view * model);

    glm::vec3 ray_origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

    RayGenerator ray_generator(window.width(), window.height(), fov);

    #pragma omp parallel for schedule(dynamic)
    for (uint32_t y = 0; y < window.height(); y += simd::len) {
      for (uint32_t x = 0; x < window.width(); x += simd::len) {

        RayPacklet ray_packlet {};
        MaskPacklet mask_packlet {};
        Vec4Packlet output_packlet {};

        for (uint32_t yy = 0; yy < simd::len && y + yy < window.height(); yy++) {

          simd::vec3 ray_direction;
          for (uint32_t xx = 0; xx < simd::len && x + xx < window.width(); xx++) {
            glm::vec3 dir = ray_transform * ray_generator(x + xx, y + yy);

            /*
            glm::vec4 output = render_layer_dda(volume, { ray_origin, dir, 1.f / dir }, 0, step, transfer_function_scalar);
            glm::vec4 output = render_layer(volume, { ray_origin, dir, 1.f / dir }, 0, step, transfer_function_scalar);
            */
            glm::vec4 output = render_tree(volume, { ray_origin, dir, 1.f / dir }, step, skip_thresh, transfer_function_scalar, [&](const glm::vec3 &cell, uint8_t layer) {
              float child_size = exp2i(-layer - 1);

              float block_size = glm::length(volume_size * child_size);
              float block_distance = glm::length(volume_size * (ray_origin - cell + child_size));

              float perceived_size = block_size / block_distance;

              return layer < std::size(volume.info.layers) - 1 && perceived_size > lod;
            });

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
              raster[(y + yy) * window.width() * 3 + (x + xx) * 3 + 0] = output_packlet[yy].r[xx];
              raster[(y + yy) * window.width() * 3 + (x + xx) * 3 + 1] = output_packlet[yy].g[xx];
              raster[(y + yy) * window.width() * 3 + (x + xx) * 3 + 2] = output_packlet[yy].b[xx];
            }
          }
        }
      }
    }

    ImGui::Render();

    glDrawPixels(window.width(), window.height(), GL_RGB, GL_UNSIGNED_BYTE, raster.data());

    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    window.swapBuffers();

    GLFW::pollEvents();
  }
}
