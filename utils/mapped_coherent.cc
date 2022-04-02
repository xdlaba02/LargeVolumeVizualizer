
#include "sample.h"
#include "integrate.h"
#include "render.h"
#include "timer.h"

#include <ray/ray.h>
#include <ray/ray_simd.h>
#include <ray/intersection.h>
#include <ray/intersection_simd.h>

#include <utils/mapped_file.h>
#include <utils/ray_generator.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>

#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>

static const constexpr uint32_t viewport_width = 1024;
static const constexpr uint32_t viewport_height = 1024;
static const constexpr float viewport_fov = 45.f;
static const constexpr float step = 0.001f;
static const constexpr glm::vec3 camera_pos = glm::vec3(1, 1, 1);

// converts volume from volume space [-.5, .5] to texture space [0, 1];
static const glm::mat4 model =
  glm::translate(glm::mat4(1.f), glm::vec3(-.5f));

static const glm::mat4 view = glm::lookAt(glm::vec3(1, 1, 1), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

static const glm::mat4 vm = view * model;

template <typename F>
double test_scalar(const F &func) {
  static float dummy __attribute__((used)) {};
  return measure_ns([&]{
    render_scalar(viewport_width, viewport_height, viewport_fov, vm, [&](const Ray &ray) {
      integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
        dummy += func(pos);
      });
    });
  });
}

template <typename F>
double test_simd(const F &func) {
  static simd::float_v dummy __attribute__((used)) {};
  return measure_ns([&]{
    render_simd(viewport_width, viewport_height, viewport_fov, vm, [&](const simd::Ray &ray, const simd::float_m &mask) {
      integrate_simd(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
        dummy += func(pos, mask);
      });
    });
  });
}

template <typename F>
double test_packlet(const F &func) {
  static simd::float_v dummy __attribute__((used)) {};

  return measure_ns([&]{
    render_packlet(viewport_width, viewport_height, viewport_fov, vm, [&](const RayPacklet &ray, const MaskPacklet &mask) {
      integrate_packlet(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
        dummy += func(pos, mask);
      });
    });
  });
}

void test_raw(const uint8_t *image_data, uint32_t volume_side) {
  size_t samples = 0;

  test_scalar([&](const glm::vec3 &) {
    samples++;
    return 0.f;
  });

  std::cerr << "raw scalar " << test_scalar([&](const glm::vec3 &pos) {
    return sample_raw_scalar(image_data, volume_side, volume_side, volume_side, pos.x, pos.y, pos.z);
  }) / samples << " ns per sample\n";

  std::cerr << "raw simd "  << test_simd([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_raw_simd(image_data, volume_side, volume_side, volume_side, pos.x, pos.y, pos.z, mask);
  }) /samples << " ns per sample\n";

  std::cerr << "raw packlet "  << test_packlet([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_raw_simd(image_data, volume_side, volume_side, volume_side, pos.x, pos.y, pos.z, mask);
  }) / samples << " ns per sample\n";
}

template <size_t BITS>
void test_blocked(const uint8_t *image_data, uint32_t volume_side) {
  uint32_t side_in_blocks = (volume_side + (1 << BITS) - 1) / (1 << BITS);

  size_t samples = 0;

  test_scalar([&](const glm::vec3 &) {
    samples++;
    return 0.f;
  });

  std::cout << "blocked morton scalar " << BITS << ": " << test_scalar([&](const glm::vec3 &pos) {
    return sample_blocked_scalar<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](uint64_t block_index, float in_block_x, float in_block_y, float in_block_z) {
      return sample_morton_scalar<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z);
    });
  }) / samples << " ns per sample\n";

  std::cout << "blocked linear scalar " << BITS << ": " << test_scalar([&](const glm::vec3 &pos) {
    return sample_blocked_scalar<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](uint64_t block_index, float in_block_x, float in_block_y, float in_block_z) {
      return sample_linear_scalar<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z);
    });

  }) / samples << " ns per sample\n";

  std::cout << "blocked morton simd " << BITS << ": " << test_simd([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_blocked_simd<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
      return sample_morton_simd<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });
  }) / samples << " ns per sample\n";

  std::cout << "blocked linear simd " << BITS << ": " << test_simd([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_blocked_simd<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
      return sample_linear_simd<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });

  }) / samples << " ns per sample\n";

  std::cout << "blocked morton packlet " << BITS << ": " << test_packlet([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_blocked_simd<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
      return sample_morton_simd<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });
  }) / samples << " ns per sample\n";

  std::cout << "blocked linear packlet " << BITS << ": " << test_packlet([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_blocked_simd<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
      return sample_linear_simd<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });

  }) / samples << " ns per sample\n";
}

int main(int argc, const char *argv[]) {
#if 0
  if (argc != 3) {
    throw std::runtime_error("Param cnt");
  }

  uint64_t side;

  {
    std::stringstream arg {};
    arg << argv[1];
    arg >> side;
    if (!arg) {
      throw std::runtime_error("Param type");
    }
  }

  uint64_t physical_side = ((side + 6) / 7) * 8;

  {
    std::ofstream tmp(argv[2], std::ios::binary);
    if (!tmp) {
      throw std::runtime_error(std::string("Unable to open '") + argv[2] + "'!");
    }

    tmp.seekp(physical_side * physical_side * physical_side - 1);
    tmp.write("", 1);
  }

  MappedFile tmp(argv[2], 0, physical_side * physical_side * physical_side, MappedFile::READ, MappedFile::SHARED);
  if (!tmp) {
    throw std::runtime_error(std::string("Unable to map '") + argv[2] + "'!");
  }

  uint8_t *data = reinterpret_cast<uint8_t *>(tmp.data());
#else
    uint64_t side = 1024;
    uint64_t physical_side = ((side + 6) / 7) * 8;
    std::vector<uint8_t> image(physical_side * physical_side * physical_side);

    uint8_t *data = image.data();
#endif

  test_raw(data, side);
  test_blocked<3>(data, side);
  test_blocked<4>(data, side);
  test_blocked<5>(data, side);
  test_blocked<6>(data, side);
  test_blocked<7>(data, side);
  test_blocked<8>(data, side);
  test_blocked<9>(data, side);
  test_blocked<10>(data, side);

  return 0;
}
