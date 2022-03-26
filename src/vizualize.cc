
#include "vizualize_args.h"
#include "glfw.h"

#include <tree_volume/tree_volume.h>

#include <renderers/scalar.h>
#include <renderers/simd.h>
#include <renderers/packlet.h>

#include <integrators/tree_slab.h>
#include <integrators/tree_slab_simd.h>
#include <integrators/tree_slab_packlet.h>
#include <integrators/tree_slab_layer.h>
#include <integrators/tree_slab_layer_simd.h>
#include <integrators/tree_slab_layer_dda.h>

#include <utils/linear_gradient.h>
#include <utils/preintegrate_function.h>
#include <utils/ray_generator.h>

#include <utils/texture2D/texture2D.h>
#include <utils/texture2D/sampler.h>
#include <utils/texture2D/sampler_simd.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

#include <iostream>
#include <vector>
#include <chrono>

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
#if 0
    {80.f,  {0.75f, 0.5f, 0.25f}},
    {82.f,  {1.00f, 1.0f, 0.85f}}
#else
    {00.f,   {1.0f, 0.0f, 0.0f}},
    {01.f,   {1.0f, 1.0f, 0.0f}},
    {02.f,   {0.0f, 1.0f, 0.0f}},
    {03.f,   {0.0f, 1.0f, 1.0f}},
    {04.f,   {0.0f, 0.0f, 1.0f}},
    {05.f,   {1.0f, 0.0f, 1.0f}},
    {06.f,   {1.0f, 0.0f, 0.0f}},
    {07.f,   {1.0f, 1.0f, 0.0f}},
    {08.f,   {0.0f, 1.0f, 0.0f}},
    {09.f,   {0.0f, 1.0f, 1.0f}},
    {10.f,   {0.0f, 0.0f, 1.0f}},
    {11.f,   {1.0f, 0.0f, 1.0f}},
#endif
  };

  std::map<float, float> alpha_map {
#if 0
    {40.f,  000.f},
    {60.f,  001.f},
    {63.f,  005.f},
    {80.f,  000.f},
    {82.f,  100.f},
#else
    {0.f,   100000.f},
#endif
  };

  static constinit uint32_t pre_size = 256;

  Texture2D<float> transfer_r = preintegrate_function(pre_size, [&](float v){ return linear_gradient(color_map, v).r; });
  Texture2D<float> transfer_g = preintegrate_function(pre_size, [&](float v){ return linear_gradient(color_map, v).g; });
  Texture2D<float> transfer_b = preintegrate_function(pre_size, [&](float v){ return linear_gradient(color_map, v).b; });
  Texture2D<float> transfer_a = preintegrate_function(pre_size, [&](float v){ return linear_gradient(alpha_map, v); });

  auto transfer_function_vector = [&](simd::float_v begin, simd::float_v end, const simd::float_m &mask) -> simd::vec4 {
    // convert to normalized coordinates
    begin = (begin + 0.5f) / 256.f;
    end   = (end   + 0.5f) / 256.f;

    return {
      sample(transfer_r, begin, end, mask),
      sample(transfer_g, begin, end, mask),
      sample(transfer_b, begin, end, mask),
      sample(transfer_a, begin, end, mask)
    };
  };

  auto transfer_function_scalar = [&](float begin, float end) -> glm::vec4 {
    // convert to normalized coordinates
    begin = (begin + 0.5f) / 256.f;
    end   = (end   + 0.5f) / 256.f;

    return {
      sample(transfer_r, begin, end),
      sample(transfer_g, begin, end),
      sample(transfer_b, begin, end),
      sample(transfer_a, begin, end)
    };
  };

  GLFW::Window window(1920, 1080, "Volumetric Vizualizer");

  std::vector<uint8_t> raster(window.width() * window.height() * 3);

  auto prev_time = std::chrono::steady_clock::now();

  glm::vec2 prev_pos;
  window.getCursor(prev_pos.x, prev_pos.y);

  float yaw = -90;
  float pitch = 0;

  glm::vec3 camera_pos          = glm::vec3(0.0f, 0.0f, 2.0f);
  constexpr glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::vec3 volume_pos = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 volume_size(1.f / volume.info.width_frac, 1.f / volume.info.height_frac, 1.f / volume.info.depth_frac);


  bool imgui_show = true;

  float step = 0.001f;
  float lod = 0.05f;
  float fov = 45.f;
  int scalar_layer = 0;
  float terminate_thresh = 0.01f;

  enum: int {
    RENDERER_LAYER_DDA,
    RENDERER_LAYER_SCALAR,
    RENDERER_LAYER_VECTOR,
    RENDERER_TREE_SCALAR,
    RENDERER_TREE_VECTOR,
    RENDERER_TREE_PACKLET,
  } renderer = RENDERER_TREE_VECTOR;

  float t = 0.f;
  while (!window.shouldClose()) {
    auto time = std::chrono::steady_clock::now();
    float delta = std::chrono::duration_cast<std::chrono::milliseconds>(time - prev_time).count() / 1000.f;
    prev_time = time;

    t += delta;

    // INPUT HANDLING PART

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
                    //* glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f))
                    //* glm::translate(glm::mat4(1.f), glm::vec3(-0.5f))
                    * glm::scale(glm::mat4(1.f), volume_size);

    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);

    // RENDERING PART

    window.makeContextCurrent();

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::Begin("Configuration", &imgui_show, 0)) {
      ImGui::SetNextItemOpen(true);
      if (ImGui::TreeNode("Info")) {
        ImGui::Text("%3.1f FPS", 1.f / delta);
        ImGui::TreePop();
      }

      if (ImGui::TreeNode("Renderer")) {
        ImGui::SliderFloat("Step", &step, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Ray terminate threshold", &terminate_thresh, 0.f, 1.f, "%.2f");
        ImGui::RadioButton("Scalar Layer", reinterpret_cast<int *>(&renderer), RENDERER_LAYER_SCALAR);
        ImGui::RadioButton("Vector Layer", reinterpret_cast<int *>(&renderer), RENDERER_LAYER_VECTOR);
        ImGui::RadioButton("Scalar Layer DDA", reinterpret_cast<int *>(&renderer), RENDERER_LAYER_DDA);
        ImGui::RadioButton("Scalar Tree", reinterpret_cast<int *>(&renderer), RENDERER_TREE_SCALAR);
        ImGui::RadioButton("Vector Tree", reinterpret_cast<int *>(&renderer), RENDERER_TREE_VECTOR);
        ImGui::RadioButton("Packlet Tree", reinterpret_cast<int *>(&renderer), RENDERER_TREE_PACKLET);

        if (renderer == RENDERER_TREE_SCALAR || renderer == RENDERER_TREE_VECTOR || renderer == RENDERER_TREE_PACKLET) {
          ImGui::SliderFloat("Level of Detail", &lod, 0.f, 1.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        }
        else if (renderer == RENDERER_LAYER_SCALAR || renderer == RENDERER_LAYER_VECTOR || renderer == RENDERER_LAYER_DDA) {
          ImGui::SliderInt("Layer", &scalar_layer, 0, std::size(volume.info.layers) - 1);
        }
      }

      if (ImGui::TreeNode("Camera")) {
        ImGui::SliderFloat("FOV", &fov, 0.f, 180.f, "%.0f");
      }
      ImGui::End();
    }

    ImGui::Render();

    switch (renderer) {
      case RENDERER_LAYER_DDA:
        render_scalar(window.width(), window.height(), fov, model, view, raster.data(), [&](const Ray &ray) {
          return integrate_tree_slab_layer_dda(volume, ray, scalar_layer, step, terminate_thresh, transfer_function_scalar);
        });
      break;

      case RENDERER_LAYER_SCALAR:
        render_scalar(window.width(), window.height(), fov, model, view, raster.data(), [&](const Ray &ray) {
          return integrate_tree_slab_layer(volume, ray, scalar_layer, step, terminate_thresh, transfer_function_scalar);
        });
      break;

      case RENDERER_LAYER_VECTOR:
        render_simd(window.width(), window.height(), fov, model, view, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
          return integrate_tree_slab_layer_simd(volume, ray, mask, scalar_layer, step, terminate_thresh, transfer_function_vector);
        });
      break;

      case RENDERER_TREE_SCALAR:
        render_scalar(window.width(), window.height(), fov, model, view, raster.data(), [&](const Ray &ray) {
          return integrate_tree_slab(volume, ray, step, terminate_thresh, transfer_function_scalar, [&](const glm::vec3 &cell, uint8_t layer) {
            float child_size = exp2i(-layer - 1);

            float block_size = glm::length(volume_size * child_size);
            float block_distance = glm::length(volume_size * (ray.origin - cell + child_size));

            float perceived_size = block_size / block_distance;

            return layer == std::size(volume.info.layers) - 1 || perceived_size <= lod;
          });
        });
      break;

      case RENDERER_TREE_VECTOR:
        render_simd(window.width(), window.height(), fov, model, view, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
          return integrate_tree_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector, [&](const simd::vec3 &cell, uint8_t layer, const simd::float_m &) {
            float child_size = exp2i(-layer - 1);

            float block_size = glm::length(child_size * 2);
            simd::vec3 block_vec = ray.origin + simd::float_v(child_size) - cell;
            simd::float_v block_distance = sqrt(block_vec.x * block_vec.x + block_vec.y * block_vec.y + block_vec.z * block_vec.z);

            simd::float_v perceived_size = simd::float_v(block_size) / block_distance;

            return simd::float_m(layer == std::size(volume.info.layers) - 1) || perceived_size <= lod;
          });
        });
      break;

      case RENDERER_TREE_PACKLET:
        render_packlet(window.width(), window.height(), fov, model, view, raster.data(), [&](const RayPacklet &ray_packlet, const MaskPacklet &mask_packlet) {
          return integrate_tree_slab_packlet(volume, ray_packlet, step, terminate_thresh, mask_packlet, transfer_function_vector, [&](const Vec3Packlet &cell_packlet, uint8_t layer, const MaskPacklet &mask_packlet) {
            float child_size = exp2i(-layer - 1);

            float block_size = glm::length(child_size * 2);

            MaskPacklet output_mask_packlet = mask_packlet;

            for (uint8_t j = 0; j < simd::len; j++) {
              if (mask_packlet[j].isNotEmpty()) {
                simd::vec3 block_vec = ray_packlet[j].origin + simd::float_v(child_size) - cell_packlet[j];
                simd::float_v block_distance = sqrt(block_vec.x * block_vec.x + block_vec.y * block_vec.y + block_vec.z * block_vec.z);

                simd::float_v perceived_size = simd::float_v(block_size) / block_distance;

                output_mask_packlet[j] = simd::float_m(layer == std::size(volume.info.layers) - 1) || perceived_size <= lod;
              }
            }

            return output_mask_packlet;
          });
        });
      break;

      default:
      break;
    }

    glDrawPixels(window.width(), window.height(), GL_RGB, GL_UNSIGNED_BYTE, raster.data());
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    window.swapBuffers();

    GLFW::pollEvents();
  }
}
