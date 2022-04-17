#include "common.h"

template <typename T, uint32_t N, typename F>
void scalar_samples(const typename TreeVolume<T, N>::Info &info, const F &func) {
  const glm::vec3 volume_frac = glm::vec3(info.width_frac, info.height_frac, info.depth_frac);
  const glm::mat4 texture = glm::scale(glm::mat4(1.f), 1.f / volume_frac);

  generate_transforms([&](const glm::mat4 &vm) {
    render_scalar(viewport_width, viewport_height, viewport_fov, vm * texture, [&](const Ray &ray) {
      integrate_tree_scalar<T, N>(info, ray, step, [&](const glm::vec3 &, uint32_t layer) {
        return layer == std::size(info.layers) - 1;
      }, [&](const glm::vec3 &node_pos, uint32_t layer_index, const glm::vec3 &in_block_pos) {
        func(info.node_handle(node_pos.x, node_pos.y, node_pos.z, layer_index), in_block_pos);
      });
    });
  });
}

template <typename T, uint32_t N, typename F>
void simd_samples(const typename TreeVolume<T, N>::Info &info, const F &func) {
  const glm::vec3 volume_frac = glm::vec3(info.width_frac, info.height_frac, info.depth_frac);
  const glm::mat4 texture = glm::scale(glm::mat4(1.f), 1.f / volume_frac);

  generate_transforms([&](const glm::mat4 &vm) {
    render_simd(viewport_width, viewport_height, viewport_fov, vm * texture, [&](const simd::Ray &ray, const simd::float_m &mask) {
      integrate_tree_simd<T, N>(info, ray, step, mask, [&](const simd::vec3 &, uint32_t layer, const simd::float_m &) {
        return simd::float_m(layer == std::size(info.layers) - 1);
      }, [&](const simd::vec3 &node_pos, uint32_t layer_index, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
        std::array<uint64_t, simd::len> block_indices;

        for (uint8_t k = 0; k < simd::len; k++) {
          block_indices[k] = info.node_handle(node_pos.x[k], node_pos.y[k], node_pos.z[k], layer_index);
        }

        func(block_indices, in_block_pos, mask);
      });
    });
  });
}

template <typename T, uint32_t N, typename F>
void packlet_samples(const typename TreeVolume<T, N>::Info &info, const F &func) {
  const glm::vec3 volume_frac = glm::vec3(info.width_frac, info.height_frac, info.depth_frac);
  const glm::mat4 texture = glm::scale(glm::mat4(1.f), 1.f / volume_frac);

  generate_transforms([&](const glm::mat4 &vm) {
    render_packlet(viewport_width, viewport_height, viewport_fov, vm * texture, [&](const RayPacklet &ray, const MaskPacklet &mask) {
      integrate_tree_packlet<T, N>(info, ray, step, mask, [&](const Vec3Packlet &, uint32_t layer, const MaskPacklet &) {
        MaskPacklet m;
        std::fill(std::begin(m), std::end(m), simd::float_m(layer == std::size(info.layers) - 1));
        return m;
      }, [&](const simd::vec3 &node_pos, uint32_t layer_index, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
        std::array<uint64_t, simd::len> block_indices;

        for (uint8_t k = 0; k < simd::len; k++) {
          if (mask[k]) {
            block_indices[k] = info.node_handle(node_pos.x[k], node_pos.y[k], node_pos.z[k], layer_index);
          }
        }

        func(block_indices, in_block_pos, mask);
      });
    });
  });
}

template <typename T, uint32_t N>
void test_tree(uint32_t width, uint32_t height, uint32_t depth) {

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

  scalar_samples<T, N>(info, [&](uint64_t node_handle, const glm::vec3 &in_block_pos) {
    samples++;
    sample_morton_scalar<T, N>(volume_data, node_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);
  });

  std::cout << "tree" << N << " ";

  std::cout << (info.layers[0].size_in_nodes * TreeVolume<T, N>::BLOCK_SIZE) / (float(width) * height * depth) << " "; // WORKS only for full resolution rendering bc other layers not used

  std::cout << measure_ns([&]{
    scalar_samples<T, N>(info, [&](uint64_t node_handle, const glm::vec3 &in_block_pos) {
      sample_linear_scalar<T, N>(volume_data, node_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    scalar_samples<T, N>(info, [&](uint64_t node_handle, const glm::vec3 &in_block_pos) {
      sample_morton_scalar<T, N>(volume_data, node_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    simd_samples<T, N>(info, [&](const std::array<uint64_t, simd::len> &block_indices, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
      sample_linear_simd<T, N>(volume_data, block_indices, in_block_pos.x, in_block_pos.y, in_block_pos.z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    simd_samples<T, N>(info, [&](const std::array<uint64_t, simd::len> &block_indices, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
      sample_morton_simd<T, N>(volume_data, block_indices, in_block_pos.x, in_block_pos.y, in_block_pos.z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    packlet_samples<T, N>(info, [&](const std::array<uint64_t, simd::len> &block_indices, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
      sample_linear_simd<T, N>(volume_data, block_indices, in_block_pos.x, in_block_pos.y, in_block_pos.z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    packlet_samples<T, N>(info, [&](const std::array<uint64_t, simd::len> &block_indices, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
      sample_morton_simd<T, N>(volume_data, block_indices, in_block_pos.x, in_block_pos.y, in_block_pos.z, mask);
    });
  }) / samples << " ";

  std::cout << "\n";
}

template <typename T>
void test(uint32_t width, uint32_t height, uint32_t depth) {
  test_tree<T, 3>(width, height, depth);
  test_tree<T, 4>(width, height, depth);
  test_tree<T, 5>(width, height, depth);
  test_tree<T, 6>(width, height, depth);
  test_tree<T, 7>(width, height, depth);
  test_tree<T, 8>(width, height, depth);
  test_tree<T, 9>(width, height, depth);
  test_tree<T, 10>(width, height, depth);
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
