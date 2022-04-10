
#include "sample.h"
#include "integrate.h"
#include "render.h"
#include "timer.h"

#include <components/ray/ray.h>
#include <components/ray/ray_simd.h>
#include <components/ray/intersection.h>
#include <components/ray/intersection_simd.h>

#include <components/mapped_file.h>
#include <components/ray_generator.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>

#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>

static const constexpr uint32_t viewport_width = 32;
static const constexpr uint32_t viewport_height = 32;
static const constexpr float viewport_fov = 1.4f;
static const constexpr size_t n = 100;

static const constexpr float step = 0.001f;

// converts volume from volume space [-.5, .5] to texture space [0, 1];
static const glm::mat4 model = glm::translate(glm::mat4(1.f), glm::vec3(-.5f));

template <typename F>
void generate_transforms(const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(0.f, 1.f);
  std::normal_distribution<float> normal {};

  for (uint64_t i = 0; i < n; i++) {
    glm::vec3 origin = glm::normalize(glm::vec3(normal(re), normal(re), normal(re))) * 2.f;

    static const glm::mat4 view = glm::lookAt(origin, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    func(view * model);
  }
}

template <typename F>
double test_scalar(const F &func) {
  static float dummy __attribute__((used)) {};

  return measure_ns([&]{
    generate_transforms([&](const glm::mat4 &vm) {
      render_scalar(viewport_width, viewport_height, viewport_fov, vm, [&](const Ray &ray) {
        integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
          dummy += func(pos);
        });
      });
    });
  });
}

template <typename F>
double test_simd(const F &func) {
  static simd::float_v dummy __attribute__((used)) {};

  return measure_ns([&]{
    generate_transforms([&](const glm::mat4 &vm) {
      render_simd(viewport_width, viewport_height, viewport_fov, vm, [&](const simd::Ray &ray, const simd::float_m &mask) {
        integrate_simd(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
          dummy += func(pos, mask);
        });
      });
    });
  });
}

template <typename F>
double test_packlet(const F &func) {
  static simd::float_v dummy __attribute__((used)) {};

  return measure_ns([&]{
    generate_transforms([&](const glm::mat4 &vm) {
      render_packlet(viewport_width, viewport_height, viewport_fov, vm, [&](const RayPacklet &ray, const MaskPacklet &mask) {
        integrate_packlet(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
          dummy += func(pos, mask);
        });
      });
    });
  });
}

template <size_t BITS>
void test(const uint8_t *volume_data) {
  size_t samples = 0;

  test_scalar([&](const glm::vec3 &pos) {
    samples++;
    return sample_raw_scalar(volume_data, 1 << BITS, 1 << BITS, 1 << BITS, pos.x, pos.y, pos.z);
  }); // MEM INIT

  std::cout << (1 << BITS) << " ";

  std::cout << test_scalar([&](const glm::vec3 &pos) {
    return sample_raw_scalar(volume_data, 1 << BITS, 1 << BITS, 1 << BITS, pos.x, pos.y, pos.z);
  }) / samples << " ";

  std::cout << test_simd([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_raw_simd(volume_data, 1 << BITS, 1 << BITS, 1 << BITS, pos.x, pos.y, pos.z, mask);
  }) / samples << " ";

  std::cout << test_packlet([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_raw_simd(volume_data, 1 << BITS, 1 << BITS, 1 << BITS, pos.x, pos.y, pos.z, mask);
  }) / samples << " ";

  std::cout << "\n";
}

int main(void) {
  uint64_t max_volume_side = 1 << 12;

  uint64_t physical_volume_size = max_volume_side * max_volume_side * max_volume_side;

  const char *tmpfile = "tmpfile.dat";

  {
    std::ofstream volume(tmpfile, std::ios::binary);
    if (!volume) {
      throw std::runtime_error(std::string("Unable to open '") + tmpfile + "'!");
    }

    volume.seekp(physical_volume_size - 1);
    volume.write("", 1);
  }

  MappedFile volume(tmpfile, 0, physical_volume_size, MappedFile::READ, MappedFile::SHARED);
  if (!volume) {
    throw std::runtime_error(std::string("Unable to open '") + tmpfile + "'!");
  }

  const uint8_t *volume_data = reinterpret_cast<const uint8_t *>(volume.data());

  std::cout << "# size scalar simd packlet\n";

  test<3>(volume_data);
  test<4>(volume_data);
  test<5>(volume_data);
  test<6>(volume_data);
  test<7>(volume_data);
  test<8>(volume_data);
  test<9>(volume_data);
  test<10>(volume_data);
  test<11>(volume_data);
  test<12>(volume_data);

  return 0;
}
