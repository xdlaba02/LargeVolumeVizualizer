#include "tf1d.h"
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

#include <transfer/piecewise_linear.h>
#include <transfer/preintegrate_function.h>

#include <components/ray_generator.h>

#include <components/texture2D/texture2D.h>
#include <components/texture2D/sampler.h>
#include <components/texture2D/sampler_simd.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>

void print_usage(const char *argv0) {
  std::cerr << "Usage: \n";
  std::cerr << argv0 << " <processed-volume> <processed-metadata> <transfer-function> <width> <height> <depth>\n";
}

bool parse_args(int argc, char *argv[], const char *&processed_volume, const char *&processed_metadata, const char *&transfer_function, uint32_t &width, uint32_t &height, uint32_t &depth) {
  if (argc != 7) {
    print_usage(argv[0]);
    return false;
  }

  processed_volume   = argv[1];
  processed_metadata = argv[2];
  transfer_function  = argv[3];

  {
    std::stringstream arg {};
    arg << argv[4];
    arg >> width;
    if (!arg) {
      print_usage(argv[0]);
      return false;
    }
  }

  {
    std::stringstream arg {};
    arg << argv[5];
    arg >> height;
    if (!arg) {
      print_usage(argv[0]);
      return false;
    }
  }

  {
    std::stringstream arg {};
    arg << argv[6];
    arg >> depth;
    if (!arg) {
      print_usage(argv[0]);
      return false;
    }
  }

  return true;
}

int main(int argc, char *argv[]) {
  const char *processed_volume;
  const char *processed_metadata;
  const char *transfer_function;
  uint32_t width;
  uint32_t height;
  uint32_t depth;

  if (!parse_args(argc, argv, processed_volume, processed_metadata, transfer_function, width, height, depth)) {
    return 1;
  }

  TreeVolume<uint8_t> volume(processed_volume, processed_metadata, width, height, depth);

  TF1D tf1d = TF1D::load_from_file(transfer_function);

  static constinit uint32_t pre_size = 256;

  Texture2D<float> transfer_r = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.rgb, v * 256 / pre_size).r; });
  Texture2D<float> transfer_g = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.rgb, v * 256 / pre_size).g; });
  Texture2D<float> transfer_b = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.rgb, v * 256 / pre_size).b; });
  Texture2D<float> transfer_a = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.a,   v * 256 / pre_size);   });

  auto transfer_function_vector = [&](const simd::float_v &begin, const simd::float_v &end, const simd::float_m &mask) -> simd::vec4 {
    simd::SampleInfo info = sample_info(pre_size, pre_size, (begin + 0.5f) / 256.f, (end + 0.5f) / 256.f);

    return {
      sample(transfer_r, info, mask),
      sample(transfer_g, info, mask),
      sample(transfer_b, info, mask),
      sample(transfer_a, info, mask)
    };
  };

  auto transfer_function_scalar = [&](float begin, float end) -> glm::vec4 {
    SampleInfo info = sample_info(pre_size, pre_size, (begin + 0.5f) / 256.f, (end + 0.5f) / 256.f);

    return {
      sample(transfer_r, info),
      sample(transfer_g, info),
      sample(transfer_b, info),
      sample(transfer_a, info)
    };
  };

  GLFW::Window window(640, 480, "Volumetric Vizualizer");

  std::vector<uint8_t> raster(window.width() * window.height() * 3);

  glm::vec2 prev_cursor_pos;

  const glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::vec3 camera_pos = glm::vec3(0.0f, 0.0f, 2.0f);
  float camera_yaw     = -90;
  float camera_pitch   = 0;

  // Tree volume renderes expects the ray intersecting from (0 .. 1), but the interval the volume is in is in range (0 .. volume.info.frac*) due to the layers being power of two sizes.
  // We need to adjust the transformation by this fraction.
  const glm::vec3 volume_frac     = glm::vec3(volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac);

  glm::vec3 volume_scale    = glm::vec3(1.0f, 1.0f, 1.0f);
  glm::vec3 volume_pos      = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 volume_rotation = glm::vec3(0.0f, 0.0f, 0.0f);

  bool imgui_show = true;

  float step = 0.001f;
  float quality = 0.01f;
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
  auto prev_time = std::chrono::steady_clock::now();
  while (!window.shouldClose()) {
    auto time = std::chrono::steady_clock::now();
    float delta = std::chrono::duration_cast<std::chrono::milliseconds>(time - prev_time).count() / 1000.f;
    prev_time = time;

    t += delta;

    // INPUT HANDLING PART

    {
      glm::vec2 pos;

      window.getCursor(pos.x, pos.y);

      glm::vec2 offset = pos - prev_cursor_pos;

      prev_cursor_pos = pos;

      if (window.getMouseButton(GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        window.setCursorMode(GLFW_CURSOR_DISABLED);

        constexpr float sensitivity = 0.1f;

        camera_yaw   += offset.x * sensitivity;
        camera_pitch -= offset.y * sensitivity;

        camera_pitch = std::clamp(camera_pitch, -89.f, 89.f);
      }
      else {
        window.setCursorMode(GLFW_CURSOR_NORMAL);
      }
    }

    glm::vec3 camera_front = glm::normalize(glm::vec3{
      std::cos(glm::radians(camera_yaw)) * std::cos(glm::radians(camera_pitch)),
      std::sin(glm::radians(camera_pitch)),
      std::sin(glm::radians(camera_yaw)) * std::cos(glm::radians(camera_pitch)),
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
    }

    // RENDERING PART

    window.makeContextCurrent();

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::Begin("Configuration", &imgui_show, 0)) {
      if (ImGui::TreeNode("Info")) {
        if (delta < 1.f) {
          ImGui::Text("%3.1f FPS", 1.f / delta);
        }
        else {
          ImGui::Text("%3.1f SPF", delta);
        }
        ImGui::TreePop();
      }

      if (ImGui::TreeNode("Renderer")) {
        ImGui::SliderFloat("Step", &step, 0.0001f, 1.f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Ray terminate threshold", &terminate_thresh, 0.f, 1.f, "%.2f");

        ImGui::RadioButton("Scalar Layer", reinterpret_cast<int *>(&renderer), RENDERER_LAYER_SCALAR);
        ImGui::RadioButton("Vector Layer", reinterpret_cast<int *>(&renderer), RENDERER_LAYER_VECTOR);
        ImGui::RadioButton("Scalar Layer DDA", reinterpret_cast<int *>(&renderer), RENDERER_LAYER_DDA);
        ImGui::RadioButton("Scalar Tree", reinterpret_cast<int *>(&renderer), RENDERER_TREE_SCALAR);
        ImGui::RadioButton("Vector Tree", reinterpret_cast<int *>(&renderer), RENDERER_TREE_VECTOR);
        ImGui::RadioButton("Packlet Tree", reinterpret_cast<int *>(&renderer), RENDERER_TREE_PACKLET);

        if (renderer == RENDERER_TREE_SCALAR || renderer == RENDERER_TREE_VECTOR || renderer == RENDERER_TREE_PACKLET) {
          //ImGui::SliderFloat("Quality", &quality, 0.0001f, 1.f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Quality", &quality, 0.f, .1f, "%.4f");
        }
        else if (renderer == RENDERER_LAYER_SCALAR || renderer == RENDERER_LAYER_VECTOR || renderer == RENDERER_LAYER_DDA) {
          ImGui::SliderInt("Layer", &scalar_layer, 0, std::size(volume.info.layers) - 1);
        }
        ImGui::TreePop();
      }

      if (ImGui::TreeNode("Camera")) {
        ImGui::SliderFloat("FOV", &fov, 0.f, 180.f, "%.0f");
        ImGui::BulletText("Position: ");
        ImGui::DragFloat("X", &camera_pos.x, 0.005f);
        ImGui::DragFloat("Y", &camera_pos.y, 0.005f);
        ImGui::DragFloat("Z", &camera_pos.z, 0.005f);
        ImGui::BulletText("Rotation: ");
        ImGui::DragFloat("Yaw", &camera_yaw, 1);
        ImGui::SliderFloat("Pitch", &camera_pitch, -89, 89);
        ImGui::TreePop();
      }

      if (ImGui::TreeNode("Volume")) {
        ImGui::Text("Volume resolution: %dx%dx%d", width, height, depth);
        ImGui::BulletText("Position: ");
        ImGui::DragFloat("X###PositionX", &volume_pos.x, 0.005f);
        ImGui::DragFloat("Y###PositionY", &volume_pos.y, 0.005f);
        ImGui::DragFloat("Z###PositionZ", &volume_pos.z, 0.005f);
        ImGui::BulletText("Rotation: ");
        ImGui::DragFloat("X###RotationX", &volume_rotation.x, 1.f);
        ImGui::DragFloat("Y###RotationY", &volume_rotation.y, 1.f);
        ImGui::DragFloat("Z###RotationZ", &volume_rotation.z, 1.f);
        ImGui::BulletText("Size: ");
        ImGui::DragFloat("X###ScaleX", &volume_scale.x, 0.005f);
        ImGui::DragFloat("Y###ScaleY", &volume_scale.y, 0.005f);
        ImGui::DragFloat("Z###ScaleZ", &volume_scale.z, 0.005f);
        ImGui::TreePop();
      }

      ImGui::End();
    }

    ImGui::Render();

    // converts volume from volume space [-.5, .5] to texture space [0, volume_frac];
    glm::mat4 texture =
      glm::translate(glm::mat4(1.f), glm::vec3(-.5f)) * //shifts the visible part of the volume the center
      glm::scale(glm::mat4(1.f), 1.f / volume_frac); //transforms the visible part of the volume to [0..1]

    glm::mat4 model =
      glm::translate(glm::mat4(1.f), volume_pos) *
      glm::rotate(glm::mat4(1.f), glm::radians(volume_rotation.x), glm::vec3(1.f, 0.f, 0.f)) *
      glm::rotate(glm::mat4(1.f), glm::radians(volume_rotation.y), glm::vec3(0.f, 1.f, 0.f)) *
      glm::rotate(glm::mat4(1.f), glm::radians(volume_rotation.z), glm::vec3(0.f, 0.f, 1.f)) *
      glm::scale(glm::mat4(1.f), volume_scale);

    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);

    glm::mat4 mt = model * texture;

    glm::mat4 vmt = view * mt; // converts volume from world space to camera space

    float projected_size = quality / glm::distance(camera_pos, volume_pos); // projected size ve stredu. Vepredu budou jemnejsi bloky, vzadu hrubsi

    switch (renderer) {
      case RENDERER_LAYER_DDA:
        render_scalar(window.width(), window.height(), fov, vmt, raster.data(), [&](const Ray &ray) {
          return integrate_tree_slab_layer_dda(volume, ray, scalar_layer, step, terminate_thresh, transfer_function_scalar);
        });
      break;

      case RENDERER_LAYER_SCALAR:
        render_scalar(window.width(), window.height(), fov, vmt, raster.data(), [&](const Ray &ray) {
          return integrate_tree_slab_layer(volume, ray, scalar_layer, step, terminate_thresh, transfer_function_scalar);
        });
      break;

      case RENDERER_LAYER_VECTOR:
        render_simd(window.width(), window.height(), fov, vmt, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
          return integrate_tree_slab_layer_simd(volume, ray, mask, scalar_layer, step, terminate_thresh, transfer_function_vector);
        });
      break;

      case RENDERER_TREE_SCALAR: {
        glm::mat4 mt = model * texture;
        render_scalar(window.width(), window.height(), fov, vmt, raster.data(), [&](const Ray &ray) {
          return integrate_tree_slab(volume, ray, step, terminate_thresh, transfer_function_scalar, [&](const glm::vec3 &cell, uint8_t layer) {
            float block_distance = glm::distance(camera_pos, glm::vec3(mt * glm::vec4(cell + exp2i(-layer - 1), 1.f))); // convert to world space
            return layer == std::size(volume.info.layers) - 1 || exp2i(-layer) <= block_distance * projected_size;
          });
        });
      }

      break;

      case RENDERER_TREE_VECTOR: {
        render_simd(window.width(), window.height(), fov, vmt, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
          return integrate_tree_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector, [&](const simd::vec3 &cell, uint8_t layer, const simd::float_m &mask) {
            float cell_size = exp2i(-layer);
            float child_size = exp2i(-layer - 1);

            simd::float_v block_distance;

            for (uint8_t k = 0; k < simd::len; k++) {
              if (mask[k]) {
                glm::vec3 single_cell { cell.x[k], cell.y[k], cell.z[k] };
                block_distance[k] = glm::distance(camera_pos, glm::vec3(mt * glm::vec4(single_cell + child_size, 1.f))); // convert to world space
              }
            }

            return simd::float_m(layer == std::size(volume.info.layers) - 1) || cell_size <= block_distance * projected_size;
          });
        });
      }

      break;

      case RENDERER_TREE_PACKLET: {
        render_packlet(window.width(), window.height(), fov, vmt, raster.data(), [&](const RayPacklet &ray_packlet, const MaskPacklet &mask_packlet) {
          return integrate_tree_slab_packlet(volume, ray_packlet, step, terminate_thresh, mask_packlet, transfer_function_vector, [&](const Vec3Packlet &cell_packlet, uint8_t layer, const MaskPacklet &mask_packlet) {
            float cell_size = exp2i(-layer);
            float child_size = exp2i(-layer - 1);

            MaskPacklet output_mask_packlet = mask_packlet;

            for (uint8_t j = 0; j < simd::len; j++) {
              if (mask_packlet[j].isNotEmpty()) {
                simd::float_v block_distance;

                for (uint8_t k = 0; k < simd::len; k++) {
                  if (mask_packlet[j][k]) {
                    glm::vec3 single_cell { cell_packlet[j].x[k], cell_packlet[j].y[k], cell_packlet[j].z[k] };
                    block_distance[k] = glm::distance(camera_pos, glm::vec3(mt * glm::vec4(single_cell + child_size, 1.f))); // convert to world space
                  }
                }

                output_mask_packlet[j] = simd::float_m(layer == std::size(volume.info.layers) - 1) || cell_size <= block_distance * projected_size;
              }
            }

            return output_mask_packlet;
          });
        });
      }
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
