#include "common.h"

template <typename T, uint32_t N, typename F>
void scalar_samples(const typename TreeVolume<T, N>::Info &info, const F &func) {
  test_scalar([&](const Ray &ray) {
    integrate_scalar(ray, step, [&](const glm::vec3 &pos) {
      sample_blocked_scalar<N>(info.layers[0].width_in_nodes, info.layers[0].height_in_nodes, info.layers[0].depth_in_nodes, pos.x, pos.y, pos.z, [&](uint32_t block_x, uint32_t block_y, uint32_t block_z, float in_block_x, float in_block_y, float in_block_z) {
        func(info.node_handle(block_x, block_y, block_z, 0), in_block_x, in_block_y, in_block_z);
      });
    });
  });
}

template <typename T, uint32_t N, typename F>
void vector_samples(const typename TreeVolume<T, N>::Info &info, const F &func) {
  test_simd([&](const simd::Ray &ray, const simd::float_m &mask) {
    integrate_simd(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
      sample_blocked_simd<N>(info.layers[0].width_in_nodes, info.layers[0].height_in_nodes, info.layers[0].depth_in_nodes, pos.x, pos.y, pos.z, [&](const simd::uint32_v &block_x, const simd::uint32_v &block_y, const simd::uint32_v &block_z, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
        std::array<uint64_t, simd::len> block_indices;

        for (uint8_t k = 0; k < simd::len; k++) {
          block_indices[k] = info.node_handle(block_x[k], block_y[k], block_z[k], 0);
        }

        func(block_indices, in_block_x, in_block_y, in_block_z, mask);
      });
    });
  });
}


template <typename T, uint32_t N, typename F>
void packlet_samples(const typename TreeVolume<T, N>::Info &info, const F &func) {
  test_packlet([&](const RayPacklet &ray, const MaskPacklet &mask) {
    integrate_packlet(ray, step, mask, [&](const simd::vec3 &pos, const simd::float_m &mask) {
      sample_blocked_simd<N>(info.layers[0].width_in_nodes, info.layers[0].height_in_nodes, info.layers[0].depth_in_nodes, pos.x, pos.y, pos.z, [&](const simd::uint32_v &block_x, const simd::uint32_v &block_y, const simd::uint32_v &block_z, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z) {
        std::array<uint64_t, simd::len> block_indices;

        for (uint8_t k = 0; k < simd::len; k++) {
          block_indices[k] = info.node_handle(block_x[k], block_y[k], block_z[k], 0);
        }

        func(block_indices, in_block_x, in_block_y, in_block_z, mask);
      });
    });
  });
}

template <typename T, uint32_t N>
void test_blocks(uint32_t width, uint32_t height, uint32_t depth) {

  typename TreeVolume<T, N>::Info info(width, height, depth);

  {
    std::ofstream data(tmp_file_name, std::ios::binary);
    if (!data) {
      throw std::runtime_error(std::string("Unable to open '") + tmp_file_name + "'!");
    }

    data.seekp(info.layers[0].size_in_nodes * TreeVolume<T, N>::BLOCK_BYTES - 1);
    data.write("", 1);
  }

  MappedFile volume(tmp_file_name, 0, info.layers[0].size_in_nodes * TreeVolume<T, N>::BLOCK_BYTES, MappedFile::READ, MappedFile::SHARED);

  const T *volume_data = reinterpret_cast<const T *>(volume.data());

  size_t samples = 0;

  scalar_samples<T, N>(info, [&](uint64_t node_handle, float in_block_x, float in_block_y, float in_block_z) {
    samples++;
    sample_linear_scalar<T, N>(volume_data, node_handle, in_block_x, in_block_y, in_block_z);
  }); // MEM INIT


  std::cout << "blocks" << N << " ";

  std::cout << (info.layers[0].size_in_nodes * TreeVolume<T, N>::BLOCK_SIZE) / (float(width) * height * depth) << " ";

  std::cout << measure_ns([&]{
    scalar_samples<T, N>(info, [&](uint64_t node_handle, float in_block_x, float in_block_y, float in_block_z) {
      sample_linear_scalar<T, N>(volume_data, node_handle, in_block_x, in_block_y, in_block_z);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    scalar_samples<T, N>(info, [&](uint64_t node_handle, float in_block_x, float in_block_y, float in_block_z) {
      sample_morton_scalar<T, N>(volume_data, node_handle, in_block_x, in_block_y, in_block_z);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    vector_samples<T, N>(info, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z, const simd::float_m &mask) {
      sample_linear_simd<T, N>(volume_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    vector_samples<T, N>(info, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z, const simd::float_m &mask) {
      sample_morton_simd<T, N>(volume_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    packlet_samples<T, N>(info, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z, const simd::float_m &mask) {
      sample_linear_simd<T, N>(volume_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    packlet_samples<T, N>(info, [&](const std::array<uint64_t, simd::len> &block_index, const simd::float_v &in_block_x, const simd::float_v & in_block_y, const simd::float_v & in_block_z, const simd::float_m &mask) {
      sample_morton_simd<T, N>(volume_data, block_index, in_block_x, in_block_y, in_block_z, mask);
    });
  }) / samples << " ";

  std::cout << "\n";
}

template <typename T>
void test(uint32_t width, uint32_t height, uint32_t depth) {
  test_blocks<T, 3>(width, height, depth);
  test_blocks<T, 4>(width, height, depth);
  test_blocks<T, 5>(width, height, depth);
  test_blocks<T, 6>(width, height, depth);
  test_blocks<T, 7>(width, height, depth);
  test_blocks<T, 8>(width, height, depth);
  test_blocks<T, 9>(width, height, depth);
  test_blocks<T,10>(width, height, depth);
}

int main(int argc, const char *argv[]) {
  uint32_t width, height, depth, bytes_per_voxel;

  parse_args(argc, argv, width, height, depth, bytes_per_voxel);

  std::cout << "#" << width << " " << height << " " << depth << " " << bytes_per_voxel << "\n";
  std::cout << "# type overhead linear-scalar morton-scalar linear-simd morton-simd linear-packlet morton-packlet\n";

  if (bytes_per_voxel == 1) {
    test<uint8_t>(width, height, depth);
  }
  else if (bytes_per_voxel == 2) {
    test<uint16_t>(width, height, depth);
  }
  else {
    throw std::runtime_error("Only one or two bytes per voxel!");
  }

  return 0;
}
