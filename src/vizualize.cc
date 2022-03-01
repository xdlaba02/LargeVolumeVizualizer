
#include "renderer.h"
#include "blocked_volume.h"
#include "linear_gradient.h"
#include "preintegrated_transfer_function.h"
#include "vizualize_args.h"
#include "glfw.h"
#include "simd.h"
#include "intersection.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

#include <iostream>
#include <vector>
#include <chrono>

// TODO pyramid
// TODO min max tree

int main(int argc, char *argv[]) {
  const char *processed_volume;
  const char *processed_metadata;
  uint32_t width;
  uint32_t height;
  uint32_t depth;

  if (!parse_args(argc, argv, processed_volume, processed_metadata, width, height, depth)) {
    return 1;
  }

  BlockedVolume<uint8_t> volume(processed_volume, processed_metadata, width, height, depth);

  if (!volume) {
    std::cerr << "ERROR: Unable to open volume!\n";
    return 1;
  }

  GLFW glfw(1920, 1080, "Volumetric Vizualizer");

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

  auto transfer_function = [&](float v0, float v1) {
    return glm::vec4(preintegrated_r(v0, v1), preintegrated_g(v0, v1), preintegrated_b(v0, v1), preintegrated_a(v0, v1));
  };

  auto prev_time = std::chrono::steady_clock::now();

  glm::vec2 prev_pos;
  glfw.getCursor(prev_pos.x, prev_pos.y);

  float yaw = -90;
  float pitch = 0;

  glm::vec3 camera_pos   = glm::vec3(0.0f, 0.0f, 2.0f);
  constexpr glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::vec3 volume_pos   = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 volume_scale = glm::vec3(1.0f, 1.0f, 1.0f);

  float t = 0.f;
  while (!glfw.shouldClose()) {

    auto time = std::chrono::steady_clock::now();
    float delta = std::chrono::duration_cast<std::chrono::milliseconds>(time - prev_time).count() / 1000.f;
    std::cerr << delta << "\n";
    prev_time = time;

    t += delta;

    {
      glm::vec2 pos;

      glfw.getCursor(pos.x, pos.y);

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

      if (glfw.getKey(GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfw.shouldClose(true);
      }

      if (glfw.getKey(GLFW_KEY_W) == GLFW_PRESS) {
        camera_pos += speed * camera_front;
      }

      if (glfw.getKey(GLFW_KEY_S) == GLFW_PRESS) {
        camera_pos -= speed * camera_front;
      }

      if (glfw.getKey(GLFW_KEY_A) == GLFW_PRESS) {
        camera_pos -= glm::normalize(glm::cross(camera_front, camera_up)) * speed;
      }

      if (glfw.getKey(GLFW_KEY_D) == GLFW_PRESS) {
        camera_pos += glm::normalize(glm::cross(camera_front, camera_up)) * speed;
      }

      if (glfw.getKey(GLFW_KEY_Q) == GLFW_PRESS) {
        volume_pos.x -= speed;
      }

      if (glfw.getKey(GLFW_KEY_E) == GLFW_PRESS) {
        volume_pos.x += speed;
      }

      if (glfw.getKey(GLFW_KEY_UP) == GLFW_PRESS) {
        volume_scale += 0.1 * delta;
      }

      if (glfw.getKey(GLFW_KEY_DOWN) == GLFW_PRESS) {
        volume_scale -= 0.1 * delta;
      }

      if (glfw.getKey(GLFW_KEY_RIGHT) == GLFW_PRESS) {
        volume_scale.x += 0.1 * delta;
      }

      if (glfw.getKey(GLFW_KEY_LEFT) == GLFW_PRESS) {
        volume_scale.x -= 0.1 * delta;
      }
    }

    glm::mat4 model = glm::translate(glm::mat4(1.f), volume_pos)
                    * glm::rotate(glm::mat4(1.f), t, glm::vec3(0.f, 1.f, 0.f))
                    * glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f))
                    * glm::scale(glm::mat4(1.f), volume_scale);

    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);

    render(volume, view * model, glfw.width(), glfw.height(), 45.f, 1.f/160.f, transfer_function, [&](uint32_t x, uint32_t y, const glm::vec4 &output) {
      uint8_t *triplet = glfw.raster(x, y);
      triplet[0] = output.r * 255.f;
      triplet[1] = output.g * 255.f;
      triplet[2] = output.b * 255.f;
    });

    glfw.swapBuffers();
    glfw.pollEvents();
  }
}
