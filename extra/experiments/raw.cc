#include "common.h"

template <typename T>
void test_raw(uint32_t width, uint32_t height, uint32_t depth) {
  {
    std::ofstream data(tmp_file_name, std::ios::binary);
    if (!data) {
      throw std::runtime_error(std::string("Unable to open '") + tmp_file_name + "'!");
    }

    data.seekp(uint64_t(width) * height * depth * sizeof(T) - 1);
    data.write("", 1);
  }

  MappedFile volume(tmp_file_name, 0, uint64_t(width) * height * depth * sizeof(T), MappedFile::READ, MappedFile::SHARED);

  const T *volume_data = reinterpret_cast<const T *>(volume.data());

  size_t samples = 0;

  test_scalar([&](const Ray &ray) {
    integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
      samples++;
      return sample_raw_scalar(volume_data, width, height, depth, pos.x, pos.y, pos.z);
    });
  }); // MEM INIT

  std::cout << "raw ";

  std::cout << measure_ns([&]{
    test_scalar([&](const Ray &ray) {
      integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
        return sample_raw_scalar(volume_data, width, height, depth, pos.x, pos.y, pos.z);
      });
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    test_simd([&](const simd::Ray &ray, const simd::float_m &mask) {
      integrate_simd(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
        return sample_raw_simd(volume_data, width, height, depth, pos.x, pos.y, pos.z, mask);
      });
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    test_packlet([&](const RayPacklet &ray, const MaskPacklet &mask) {
      integrate_packlet(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
        return sample_raw_simd(volume_data, width, height, depth, pos.x, pos.y, pos.z, mask);
      });
    });
  }) / samples << " ";

  std::cout << "\n";
}

int main(int argc, const char *argv[]) {
  uint32_t width, height, depth, bytes_per_voxel;

  parse_args(argc, argv, width, height, depth, bytes_per_voxel);

  std::cout << "#" << width << " " << height << " " << depth << " " << bytes_per_voxel << "\n";
  std::cout << "# type scalar simd packlet\n";

  if (bytes_per_voxel == 1) {
    test_raw<uint8_t>(width, height, depth);
  }
  else if (bytes_per_voxel == 2) {
    test_raw<uint16_t>(width, height, depth);
  }
  else {
    throw std::runtime_error("Only one or two bytes per voxel!");
  }

  return 0;
}
