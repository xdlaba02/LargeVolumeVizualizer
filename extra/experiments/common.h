#include "../timer.h"

#include "../dummy/render.h"
#include "../dummy/integrate.h"
#include "../dummy/sample.h"

#include <ray/ray.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>

#include <random>
#include <fstream>
#include <sstream>
#include <iostream>

static const constexpr char *tmp_file_name = "tmpfile.tmp";

static const constexpr uint32_t viewport_width = 1024;
static const constexpr uint32_t viewport_height = 1024;
static const constexpr float viewport_fov = 90.f;
static const constexpr size_t n = 3;

static const constexpr float step = 0.01f;

// converts volume from volume space [-.5, .5] to texture space [0, 1];
static const glm::mat4 model = glm::translate(glm::mat4(1.f), glm::vec3(-.5f));

void parse_args(int argc, const char *argv[], uint32_t &width, uint32_t &height, uint32_t &depth, uint32_t &bytes_per_voxel) {
  if (argc != 5) {
    throw std::runtime_error("Wrong number of arguments!");
  }

  {
    std::stringstream arg {};
    arg << argv[1];
    arg >> width;
    if (!arg) {
      throw std::runtime_error("Unable to parse width!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[2];
    arg >> height;
    if (!arg) {
      throw std::runtime_error("Unable to parse height!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[3];
    arg >> depth;
    if (!arg) {
      throw std::runtime_error("Unable to parse depth!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[4];
    arg >> bytes_per_voxel;
    if (!arg) {
      throw std::runtime_error("Unable to parse bytes per voxel!");
    }
  }
}

template <typename F>
void generate_transforms(const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(0.f, 1.f);
  std::normal_distribution<float> normal {};

  for (uint64_t i = 0; i < n; i++) {
    glm::vec3 origin = glm::normalize(glm::vec3(normal(re), normal(re), normal(re)));

    static const glm::mat4 view = glm::lookAt(origin, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    func(view * model);
  }
}

template <typename F>
void test_scalar(const F &func) {
  generate_transforms([&](const glm::mat4 &vm) {
    render_scalar(viewport_width, viewport_height, viewport_fov, vm, func);
  });
}

template <typename F>
void test_simd(const F &func) {
  generate_transforms([&](const glm::mat4 &vm) {
    render_simd(viewport_width, viewport_height, viewport_fov, vm, func);
  });
}

template <typename F>
void test_packlet(const F &func) {
  generate_transforms([&](const glm::mat4 &vm) {
    render_packlet(viewport_width, viewport_height, viewport_fov, vm, func);
  });
}
