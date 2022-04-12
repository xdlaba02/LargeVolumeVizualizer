#include "../timer.h"

#include "../dummy/render.h"
#include "../dummy/integrate.h"
#include "../dummy/sample.h"

#include <raw_volume/raw_volume.h>
#include <tree_volume/tree_volume.h>

#include <ray/ray.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>

#include <random>
#include <fstream>
#include <sstream>
#include <iostream>

static const constexpr uint32_t viewport_width = 1920;
static const constexpr uint32_t viewport_height = 1080;
static const constexpr float viewport_fov = 45.f;
static const constexpr size_t n = 3;

static const constexpr float step = 0.001f;

// converts volume from volume space [-.5, .5] to texture space [0, 1];
static const glm::mat4 model = glm::translate(glm::mat4(1.f), glm::vec3(-.5f));

void parse_args(int argc, const char *argv[], const char *&raw_volume, uint32_t &width, uint32_t &height, uint32_t &depth, uint32_t &bytes_per_voxel) {
  if (argc != 6) {
    throw std::runtime_error("Wrong number of arguments!");
  }

  raw_volume = argv[1];

  {
    std::stringstream arg {};
    arg << argv[2];
    arg >> width;
    if (!arg) {
      throw std::runtime_error("Unable to parse width!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[3];
    arg >> height;
    if (!arg) {
      throw std::runtime_error("Unable to parse height!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[4];
    arg >> depth;
    if (!arg) {
      throw std::runtime_error("Unable to parse depth!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[5];
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
    render_scalar(viewport_width, viewport_height, viewport_fov, vm, [&](const Ray &ray) {
      integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
        func(pos);
      });
    });
  });
}

template <typename F>
void test_simd(const F &func) {
  generate_transforms([&](const glm::mat4 &vm) {
    render_simd(viewport_width, viewport_height, viewport_fov, vm, [&](const simd::Ray &ray, const simd::float_m &mask) {
      integrate_simd(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
        func(pos, mask);
      });
    });
  });
}

template <typename F>
void test_packlet(const F &func) {
  generate_transforms([&](const glm::mat4 &vm) {
    render_packlet(viewport_width, viewport_height, viewport_fov, vm, [&](const RayPacklet &ray, const MaskPacklet &mask) {
      integrate_packlet(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
        func(pos, mask);
      });
    });
  });
}

template <typename T>
void test_raw(const RawVolume<T> &volume) {
  const uint8_t *volume_data = reinterpret_cast<const uint8_t *>(volume.data);

  size_t samples = 0;

  test_scalar([&](const glm::vec3 &pos) {
    samples++;
    return sample_raw_scalar(volume_data, volume.width, volume.height, volume.depth, pos.x, pos.y, pos.z);
  }); // MEM INIT

  std::cout << "raw 0 0";

  std::cout << measure_ns([&]{
    test_scalar([&](const glm::vec3 &pos) {
      return sample_raw_scalar(volume_data, volume.width, volume.height, volume.depth, pos.x, pos.y, pos.z);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    test_simd([&](const simd::vec3 &pos, const simd::float_m &mask) {
      return sample_raw_simd(volume_data, volume.width, volume.height, volume.depth, pos.x, pos.y, pos.z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    test_packlet([&](const simd::vec3 &pos, const simd::float_m &mask) {
      return sample_raw_simd(volume_data, volume.width, volume.height, volume.depth, pos.x, pos.y, pos.z, mask);
    });
  }) / samples << " ";

  std::cout << "\n";
}

template <typename T, uint32_t N>
void test_blocked(const RawVolume<T> &volume) {
  const uint8_t *volume_data = reinterpret_cast<const uint8_t *>(volume.data);

  std::cout << measure_ns([&]{
    process_volume<T, N>(volume.width, volume.height, volume.depth, "tmp.data", "tmp.metadata", [&](uint32_t x, uint32_t y, uint32_t z) {
      return volume.data[volume.voxel_handle(x, y, z)];
    });
  }) << " ";

  size_t samples = 0;

  test_scalar([&](const glm::vec3 &pos) {
    samples++;
    return sample_raw_scalar(volume_data, volume.width, volume.height, volume.depth, pos.x, pos.y, pos.z);
  }); // MEM INIT

  std::cout << measure_ns([&]{
    test_scalar([&](const glm::vec3 &pos) {
      return sample_raw_scalar(volume_data, volume.width, volume.height, volume.depth, pos.x, pos.y, pos.z);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    test_simd([&](const simd::vec3 &pos, const simd::float_m &mask) {
      return sample_raw_simd(volume_data, volume.width, volume.height, volume.depth, pos.x, pos.y, pos.z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    test_packlet([&](const simd::vec3 &pos, const simd::float_m &mask) {
      return sample_raw_simd(volume_data, volume.width, volume.height, volume.depth, pos.x, pos.y, pos.z, mask);
    });
  }) / samples << " ";

  std::cout << "\n";
}

int main(int argc, const char *argv[]) {
  const char *raw_volume_file_name;
  uint32_t width, height, depth, bytes_per_voxel;

  parse_args(argc, argv, raw_volume_file_name, width, height, depth, bytes_per_voxel);

  std::cout << "# " << raw_volume_file_name << "\n";
  std::cout << "# type build overhead scalar simd packlet\n";

  if (bytes_per_voxel == 1) {
    RawVolume<uint8_t> volume(raw_volume_file_name, width, height, depth);
    test_raw(volume);
  }
  else if (bytes_per_voxel == 2) {
    RawVolume<uint16_t> volume(raw_volume_file_name, width, height, depth);
    test_raw(volume);
  }
  else {
    throw std::runtime_error("Only one or two bytes per voxel!");
  }

  return 0;
}
