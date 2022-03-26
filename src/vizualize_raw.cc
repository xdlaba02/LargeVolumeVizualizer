
#include "glfw.h"
#include "tf1d.h"

#include <raw_volume/raw_volume.h>

#include <integrators/raw_slab.h>

#include <utils/piecewise_linear.h>
#include <utils/preintegrate_function.h>
#include <utils/ray_generator.h>

#include <utils/texture2D/texture2D.h>
#include <utils/texture2D/sampler.h>

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

  auto transfer_function_scalar = [&](float begin, float end) -> glm::vec4 {
    begin = (begin + 0.5f) / 256.f;
    end   = (end   + 0.5f) / 256.f;

    return {
      sample(transfer_r, begin, end),
      sample(transfer_g, begin, end),
      sample(transfer_b, begin, end),
      sample(transfer_a, begin, end)
    };
  };

  GLFW::Window window(640, 480, "Volumetric Vizualizer");

  std::vector<uint8_t> raster(window.width() * window.height() * 3);

  glm::vec2 prev_cursor_pos;

  float camera_yaw = -90;
  float camera_pitch = 0;

  glm::vec3 camera_pos   = glm::vec3(0.0f, 0.0f, 2.0f);
  glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::vec3 volume_pos = glm::vec3(0.0f, 0.0f, 0.0f);

  float t = 0.f;
  auto prev_time = std::chrono::steady_clock::now();
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
      glm::vec2 offset = (pos - prev_cursor_pos) * sensitivity;

      prev_cursor_pos = pos;

      camera_yaw   += offset.x;
      camera_pitch -= offset.y;

      camera_pitch = std::clamp(camera_pitch, -89.f, 89.f);
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
                    * glm::translate(glm::mat4(1.f), glm::vec3(-0.5f));

    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);

    glm::mat4 ray_transform = glm::inverse(view * model);

    glm::vec3 ray_origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

    RayGenerator ray_generator(window.width(), window.height(), 45.f);

    float step = 0.001f;

    #pragma omp parallel for schedule(dynamic)
    for (uint32_t y = 0; y < window.height(); y++) {
      for (uint32_t x = 0; x < window.width(); x++) {
        glm::vec3 dir = ray_transform * ray_generator(x, y);
        glm::vec4 output = integrate_raw_slab(volume, { ray_origin, dir, 1.f / dir }, step, transfer_function_scalar);

        raster[y * window.width() * 3 + x * 3 + 0] = output.r * 255;
        raster[y * window.width() * 3 + x * 3 + 1] = output.g * 255;
        raster[y * window.width() * 3 + x * 3 + 2] = output.b * 255;
      }
    }

    window.makeContextCurrent();
    window.swapBuffers();
    GLFW::pollEvents();
  }
}
