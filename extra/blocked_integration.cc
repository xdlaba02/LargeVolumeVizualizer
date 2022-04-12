
#include "timer.h"

#include "dummy/sample.h"
#include "dummy/integrate.h"
#include "dummy/render.h"

#include <ray/ray.h>
#include <ray/ray_simd.h>
#include <ray/intersection.h>
#include <ray/intersection_simd.h>

#include <utils/mapped_file.h>
#include <utils/ray_generator.h>
#include <utils/mapped_file.h>

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
static const constexpr size_t n = 2;

static const constexpr float step = 0.001f;

// converts volume from volume space [-.5, .5] to texture space [0, 1];
static const glm::mat4 model = glm::translate(glm::mat4(1.f), glm::vec3(-.5f));

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
double test_scalar(const F &func) {
  return measure_ns([&]{
    generate_transforms([&](const glm::mat4 &vm) {
      render_scalar(viewport_width, viewport_height, viewport_fov, vm, [&](const Ray &ray) {
        integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
          func(pos);
        });
      });
    });
  });
}

template <typename F>
double test_simd(const F &func) {
  return measure_ns([&]{
    generate_transforms([&](const glm::mat4 &vm) {
      render_simd(viewport_width, viewport_height, viewport_fov, vm, [&](const simd::Ray &ray, const simd::float_m &mask) {
        integrate_simd(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
          func(pos, mask);
        });
      });
    });
  });
}

template <typename F>
double test_packlet(const F &func) {
  return measure_ns([&]{
    generate_transforms([&](const glm::mat4 &vm) {
      render_packlet(viewport_width, viewport_height, viewport_fov, vm, [&](const RayPacklet &ray, const MaskPacklet &mask) {
        integrate_packlet(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
          func(pos, mask);
        });
      });
    });
  });
}

template <size_t BITS>
void test(const uint8_t *image_data, uint64_t volume_side) {

  static constexpr const uint64_t block_side = uint64_t(1) << BITS;
  static constexpr const uint64_t subvolume_side = block_side - 1;

  uint64_t side_in_blocks = (volume_side + subvolume_side - 1) / subvolume_side;

  size_t samples = 0;

  test_scalar([&](const glm::vec3 &pos) {
    return sample_blocked_scalar<subvolume_side>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](uint64_t block_index, float in_block_x, float in_block_y, float in_block_z) {
      samples++;
      return sample_morton_scalar<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z);
    });
  }); // MEM INIT

  std::cout << (1 << BITS) << " ";

  std::cout << test_scalar([&](const glm::vec3 &pos) {
    return sample_blocked_scalar<subvolume_side>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](uint64_t block_index, float in_block_x, float in_block_y, float in_block_z) {
      return sample_linear_scalar<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z);
    });
  }) / samples << " ";

  std::cout << test_scalar([&](const glm::vec3 &pos) {
    return sample_blocked_scalar<subvolume_side>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](uint64_t block_index, float in_block_x, float in_block_y, float in_block_z) {
      return sample_morton_scalar<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z);
    });
  }) / samples << " ";

  std::cout << test_simd([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_blocked_simd<subvolume_side>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
      return sample_linear_simd<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });

  }) / samples << " ";

  std::cout << test_simd([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_blocked_simd<subvolume_side>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
      return sample_morton_simd<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });
  }) / samples << " ";

  std::cout << test_packlet([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_blocked_simd<subvolume_side>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
      return sample_linear_simd<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });

  }) / samples << " ";

  std::cout << test_packlet([&](const simd::vec3 &pos, const simd::float_m &mask) {
    return sample_blocked_simd<subvolume_side>(side_in_blocks, side_in_blocks, side_in_blocks, pos.x, pos.y, pos.z, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
      return sample_morton_simd<BITS>(image_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });
  }) / samples << " ";

  std::cout << "\n";
}

int main(void) {
  uint64_t volume_side = 1 << 10;

  uint64_t physical_volume_side = ((volume_side + 1022) / 1023) * 1024;

  uint64_t physical_volume_size = physical_volume_side * physical_volume_side * physical_volume_side;

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

  std::cout << "# volume size: " << volume_side << " x " << volume_side << " x " << volume_side << "\n";

  std::cout << "# size linear-scalar morton-scalar linear-simd morton-simd linear-packlet morton-packlet\n";
  test<3>(volume_data, volume_side);
  test<4>(volume_data, volume_side);
  test<5>(volume_data, volume_side);
  test<6>(volume_data, volume_side);
  test<7>(volume_data, volume_side);
  test<8>(volume_data, volume_side);
  test<9>(volume_data, volume_side);
  test<10>(volume_data, volume_side);

  return 0;
}
