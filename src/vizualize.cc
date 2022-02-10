
#include "integrator.h"
#include "blocked_volume.h"
#include "linear_gradient.h"
#include "preintegrated_transfer_function.h"
#include "vizualize_args.h"
#include "glfw.h"
#include "simd.h"
#include "intersection.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

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

  std::map<float, glm::vec3> color_map {
    #if 0
    {80.f,  {0.75f, 0.5f, 0.25f}},
    {82.f,  {1.00f, 1.0f, 0.85f}}
    #else
    {0.f,  {1.00f, 0.0f, 0.0f}},
    {89.f,  {0.00f, 1.0f, 0.0f}},
    {178.f,  {0.00f, 0.0f, 1.0f}},
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
    {0.f,   0.f},
    {89.f,  5.f},
    {178.f, 0.f},
    #endif
  };

  auto transfer_function = [&](float v) {
    return glm::vec4(linear_gradient(color_map, v), linear_gradient(alpha_map, v));
  };

  Integrator<uint8_t> integrator(transfer_function, 0.005);

  GLFW glfw(1920, 1080, "Volumetric Vizualizer");

  auto prev_time = std::chrono::steady_clock::now();

  glm::vec2 prev_pos;
  glfw.getCursor(prev_pos.x, prev_pos.y);

  float yaw = -90;
  float pitch = 0;

  glm::vec3 camera_pos   = glm::vec3(0.0f, 0.0f, 2.0f);
  constexpr glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::vec3 volume_pos   = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 volume_scale = glm::vec3(1.0f, 1.0f, 1.0f);

  constexpr float yFOV = std::tan(45 * M_PI / 180 * 0.5);
  float aspect = float(glfw.width()) / float(glfw.height());

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

    // object space - volume in 3D interval <0, 1>
    // normalized object space - volume in 3D interval <-0.5, 0.5> - good for transformations because origin is at center of the volume.

    glm::mat4 norm = glm::translate(glm::mat4(1.f), glm::vec3(-0.5f, -0.5f, -0.5f));

    glm::mat4 norm_invese = glm::inverse(norm);

    glm::mat4 model = glm::translate(glm::mat4(1.f), volume_pos)
                    * glm::rotate(glm::mat4(1.f), t, glm::vec3(0.f, 1.f, 0.f))
                    * glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(1.f, 0.f, 0.f))
                    * glm::scale(glm::mat4(1.f), volume_scale);

    glm::mat4 model_inverse = glm::inverse(model);

    glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
    glm::mat4 view_inverse = glm::inverse(view);

    glm::mat4 norm_model_inverse = norm_invese * model_inverse;
    glm::mat4 norm_model_view_inverse = norm_model_inverse * view_inverse;

    glm::vec3 ray_origin = norm_model_inverse * glm::vec4(camera_pos, 1); // camera_pos is in world space, transform to object space

    #pragma omp parallel for schedule(dynamic)
    for (uint32_t j = 0; j < glfw.height(); j++) {
      float y = (2 * (j + 0.5) / glfw.height() - 1) * yFOV;

      for (uint32_t i = 0; i < glfw.width(); i += simd::len) {
        simd::float_v is = simd::float_v::IndexesFromZero() + i;
        simd::float_v xs = (2 * (is + 0.5f) / float(glfw.width()) - 1) * aspect * yFOV;

        glm::vec<3, simd::float_v> directions {};
        for (uint32_t k = 0; k < simd::len; k++) {
          glm::vec3 direction = norm_model_view_inverse * glm::normalize(glm::vec4(xs[k], y, -1, 0)); // generate ray normalized in view space and transform to object space
#if 1
          glm::vec<4, simd::uint32_v> dsts = integrator.integrate(volume, ray_origin, direction, int(t) % std::size(volume.info.layers)) * 255.f;
#else
          directions.x[k] = direction.x;
          directions.y[k] = direction.y;
          directions.z[k] = direction.z;
        }

        glm::vec<4, simd::uint32_v> dsts = integrator.integrate(volume, ray_origin, directions, simd::uint32_v(int(t) % std::size(volume.info.layers))) * simd::float_v(255.f);

        for (uint32_t k = 0; k < simd::len; k++) {
#endif
          uint8_t *triplet = glfw.raster(i, j) + k * 3;
          triplet[0] = dsts.r[k];
          triplet[1] = dsts.g[k];
          triplet[2] = dsts.b[k];
        }
      }
    }

    glfw.swapBuffers();
    glfw.pollEvents();
  }
}
