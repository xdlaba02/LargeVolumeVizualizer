
#include "glfw.h"
#include "tf1d.h"

#include <raw_volume/raw_volume.h>

#include <renderers/scalar.h>
#include <renderers/simd.h>

#include <integrators/raw_slab.h>
#include <integrators/raw_slab_simd.h>

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
#include <sstream>
#include <vector>
#include <chrono>

int main(int argc, char *argv[]) {
  uint32_t width;
  uint32_t height;
  uint32_t depth;

  {
    std::stringstream wstream(argv[3]);
    std::stringstream hstream(argv[4]);
    std::stringstream dstream(argv[5]);
    wstream >> width;
    hstream >> height;
    dstream >> depth;
  }

  RawVolume<uint8_t> volume(argv[1], width, height, depth);

  TF1D tf = TF1D::load_from_file(argv[2]);

  static constinit uint32_t pre_size = 256;

  Texture2D<float> transfer_r = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf.rgb, v).r; });
  Texture2D<float> transfer_g = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf.rgb, v).g; });
  Texture2D<float> transfer_b = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf.rgb, v).b; });
  Texture2D<float> transfer_a = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf.a,   v); });

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

  GLFW::Window window(1920, 1080, "Volumetric Vizualizer");

  std::vector<uint8_t> raster(window.width() * window.height() * 3);

  glm::vec2 prev_cursor_pos;

  const glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::vec3 camera_pos = glm::vec3(0.0f, 0.0f, 2.0f);
  float camera_yaw     = -90;
  float camera_pitch   = 0;

  glm::vec3 volume_scale    = glm::vec3(1.0f, 1.0f, 1.0f);
  glm::vec3 volume_pos      = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 volume_rotation = glm::vec3(0.0f, 0.0f, 0.0f);

  float step = 0.001f;
  float fov = 45.f;
  float terminate_thresh = 0.01f;

  float t = 0.f;
  auto prev_time = std::chrono::steady_clock::now();
  while (!window.shouldClose()) {
    auto time = std::chrono::steady_clock::now();
    float delta = std::chrono::duration_cast<std::chrono::milliseconds>(time - prev_time).count() / 1000.f;
    prev_time = time;

    std::cerr << 1 / delta << " FPS\n";

    t += delta;

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

    glm::mat4 model =
      glm::translate(glm::mat4(1.f), volume_pos) *
      glm::rotate(glm::mat4(1.f), glm::radians(volume_rotation.x), glm::vec3(1.f, 0.f, 0.f)) *
      glm::rotate(glm::mat4(1.f), glm::radians(volume_rotation.y), glm::vec3(0.f, 1.f, 0.f)) *
      glm::rotate(glm::mat4(1.f), glm::radians(volume_rotation.z), glm::vec3(0.f, 0.f, 1.f)) *
      glm::scale(glm::mat4(1.f), volume_scale) *
      glm::translate(glm::mat4(1.f), glm::vec3(-0.5f));

    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);

    render_scalar(window.width(), window.height(), fov, view * model, raster.data(), [&](const Ray &ray) {
      return integrate_raw_slab(volume, ray, step, terminate_thresh, transfer_function_scalar);
    });

    /*
    render_simd(window.width(), window.height(), fov, view * model, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
      return integrate_raw_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector);
    });
    */

    window.makeContextCurrent();

    glDrawPixels(window.width(), window.height(), GL_RGB, GL_UNSIGNED_BYTE, raster.data());

    window.swapBuffers();
    GLFW::pollEvents();
  }
}
