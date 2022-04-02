
#include "sample.h"
#include "integrate.h"
#include "timer.h"

#include <ray/ray.h>
#include <ray/ray_simd.h>
#include <ray/intersection.h>
#include <ray/intersection_simd.h>

#include <cstdint>

#include <vector>
#include <iostream>
#include <random>

template <typename F>
float test_scalar(uint64_t n, const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(0.f, 1.f);
  std::normal_distribution<float> normal {};

  static float dummy __attribute__((used)) {};

  return measure_ns([&]{
    for (uint64_t i = 0; i < n * simd::len; i++) {
      glm::vec3 origin = glm::normalize(glm::vec3(normal(re), normal(re), normal(re)));
      glm::vec3 direction = glm::normalize(glm::vec3(rd(re), rd(re), rd(re)) - origin);

      dummy += func({origin, direction, 1.f / direction});
    }
  });
}

template <typename F>
float test_simd(uint64_t n, const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(0.f, 1.f);
  std::normal_distribution<float> normal {};

  static simd::float_v dummy __attribute__((used)) {};

  return measure_ns([&]{
    for (uint64_t i = 0; i < n; i++) {
      simd::vec3 origin, direction;
      for (uint32_t k = 0; k < simd::len; k++) {
        glm::vec3 local_origin = glm::normalize(glm::vec3(normal(re), normal(re), normal(re)));
        glm::vec3 local_direction = glm::normalize(glm::vec3(rd(re), rd(re), rd(re)) - local_origin);

        origin.x[k] = local_origin.x;
        origin.y[k] = local_origin.y;
        origin.z[k] = local_origin.z;

        direction.x[k] = local_direction.x;
        direction.y[k] = local_direction.y;
        direction.z[k] = local_direction.z;
      }

      dummy += func({origin, direction, simd::float_v(1.f) / direction});
    }
  });
}

void test_raw(const uint8_t *image_data, uint32_t volume_side, float step, size_t n) {
  {
    size_t samples = 0;
    std::cout << "raw     scalar:\t\t\t" << test_scalar(n, [&](const Ray &ray) {
      float acc {};
      integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
        samples++;
        acc += sample_raw_scalar(image_data, volume_side, volume_side, volume_side, pos.x, pos.y, pos.z);
      });
      return acc;
    }) / samples << " ns per sample\n";
  }

  {
    size_t samples = 0;
    std::cout << "raw     simd:\t\t\t" << test_simd(n, [&](const simd::Ray &ray) {
      simd::float_v acc {};
      integrate_simd(ray, step, simd::float_m{true}, [&](const simd::vec3 &pos, const simd::float_m &mask) {
        samples++;
        acc += sample_raw_simd(image_data, volume_side, volume_side, volume_side, pos.x, pos.y, pos.z, mask);
      });
      return acc;
    }) / (samples * simd::len) << " ns per sample\n";
  }
}

template <size_t BITS>
void test_blocked(const uint8_t *image_data, uint32_t volume_side, float step, size_t n) {
  uint32_t side_in_blocks = (volume_side + (1 << BITS) - 1) / (1 << BITS);

  {
    size_t samples = 0;

    std::cout << "blocked morton scalar " << BITS << ":\t" << test_scalar(n, [&](const Ray &ray) {
      float acc {};
      integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
        samples++;
        acc += sample_blocked_scalar<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](uint64_t block_index, float in_block_x, float in_block_y, float in_block_z) {
          return sample_morton_scalar<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z);
        });
      });
      return acc;
    }) / samples << " ns per sample\n";

  }

  {
    size_t samples = 0;

    std::cout << "blocked linear scalar " << BITS << ":\t" << test_scalar(n, [&](const Ray &ray) {
      float acc {};
      integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
        samples++;
        acc += sample_blocked_scalar<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](uint64_t block_index, float in_block_x, float in_block_y, float in_block_z) {
          return sample_linear_scalar<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z);
        });
      });
      return acc;
    }) / samples << " ns per sample\n";
  }

  {
    size_t samples = 0;

    std::cout << "blocked morton simd   " << BITS << ":\t" << test_simd(n, [&](const simd::Ray &ray) {
      simd::float_v acc {};
      integrate_simd(ray, step, simd::float_m(true), [&](const simd::vec3 &pos, simd::float_m &mask) {
        samples++;
        acc += sample_blocked_simd<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_indices, simd::float_v in_block_x, simd::float_v in_block_y, simd::float_v in_block_z) {
          return sample_morton_simd<BITS>(image_data, block_indices, in_block_x, in_block_y, in_block_z, mask);
        });
      });
      return acc;
    }) / (samples * simd::len) << " ns per sample\n";
  }

  {
    size_t samples = 0;

    std::cout << "blocked linear simd   " << BITS << ":\t" << test_simd(n, [&](const simd::Ray &ray) {
      simd::float_v acc {};
      integrate_simd(ray, step, simd::float_m(true), [&](const simd::vec3 &pos, simd::float_m &mask) {
        samples++;
        acc += sample_blocked_simd<(1 << BITS) - 1>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_indices, simd::float_v in_block_x, simd::float_v in_block_y, simd::float_v in_block_z) {
          return sample_linear_simd<BITS>(image_data, block_indices, in_block_x, in_block_y, in_block_z, mask);
        });
      });
      return acc;
    }) / (samples * simd::len) << " ns per sample\n";
  }
}

int main(void) {

  uint64_t n = 10000;

  uint32_t max_volume_side = 1176;
  std::vector<uint8_t> image(max_volume_side * max_volume_side * max_volume_side);

  float val {};
  simd::float_v sval {};

  uint32_t volume_side = 1024;

  std::cout << "Volume size: " << volume_side << " x " << volume_side << " x " << volume_side << "\n";

  for (float step: {0.005, 0.001, 0.0005}) {
    std::cout << "Step: " << step << "\n";

    test_raw(image.data(), volume_side, step, n);
    test_blocked<3>(image.data(), volume_side, step, n);
    test_blocked<4>(image.data(), volume_side, step, n);
    test_blocked<5>(image.data(), volume_side, step, n);
    test_blocked<6>(image.data(), volume_side, step, n);
    test_blocked<7>(image.data(), volume_side, step, n);
    test_blocked<8>(image.data(), volume_side, step, n);
    test_blocked<9>(image.data(), volume_side, step, n);
    test_blocked<10>(image.data(), volume_side, step, n);

    std::cout << "\n";
  }


  std::cerr << val;
}
