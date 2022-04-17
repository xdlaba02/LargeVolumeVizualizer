#include "common.h"

#include <utils/scan_tree.h>

void parse_args_reduced(int argc, const char *argv[], uint32_t &width, uint32_t &height, uint32_t &depth, uint32_t &bytes_per_voxel, float &quality) {
  if (argc != 6) {
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

  {
    std::stringstream arg {};
    arg << argv[5];
    arg >> quality;
    if (!arg) {
      throw std::runtime_error("Unable to parse quality!");
    }
  }
}

template <typename F>
void generate_origins(const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(0.f, 1.f);
  std::normal_distribution<float> normal {};

  for (uint64_t i = 0; i < n; i++) {
    func(glm::normalize(glm::vec3(normal(re), normal(re), normal(re))));
  }
}

template <typename T, uint32_t N, typename F>
void scalar_samples(const typename TreeVolume<T, N>::Info &info, float quality, const F &func) {
  glm::vec3 volume_frac = glm::vec3(info.width_frac, info.height_frac, info.depth_frac);
  glm::mat4 texture = glm::scale(glm::mat4(1.f), 1.f / volume_frac);

  glm::mat4 mt = model * texture;


  generate_origins([&](const glm::vec3 &origin) {
    glm::mat4 view = glm::lookAt(origin, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    float projected_size = quality * TreeVolume<T, N>::BLOCK_SIDE / glm::distance(origin, glm::vec3(0.0f, 0.0f, 0.0f)); // projected size ve stredu. Vepredu budou jemnejsi bloky, vzadu hrubsi

    render_scalar(viewport_width, viewport_height, viewport_fov, view * mt, [&](const Ray &ray) {
      integrate_tree_scalar<T, N>(info, ray, step, [&](const glm::vec3 &cell, uint32_t layer) {
        float cell_size = exp2i(-layer);
        float child_size = exp2i(-layer - 1);

        glm::vec3 cell_center = mt * glm::vec4(cell + child_size, 1.f);  // convert to world space

        float block_distance = glm::distance(origin, cell_center);

        return layer == std::size(info.layers) - 1 || cell_size <= block_distance * projected_size;
      }, [&](const glm::vec3 &node_pos, uint32_t layer_index, const glm::vec3 &in_block_pos) {
        func(info.node_handle(node_pos.x, node_pos.y, node_pos.z, layer_index), in_block_pos);
      });
    });
  });
}

template <typename T, uint32_t N, typename F>
void simd_samples(const typename TreeVolume<T, N>::Info &info, float quality, const F &func) {
  glm::vec3 volume_frac = glm::vec3(info.width_frac, info.height_frac, info.depth_frac);
  glm::mat4 texture = glm::scale(glm::mat4(1.f), 1.f / volume_frac);

  glm::mat4 mt = model * texture;

  generate_origins([&](const glm::vec3 &origin) {
    glm::mat4 view = glm::lookAt(origin, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    float projected_size = quality * TreeVolume<T, N>::BLOCK_SIDE / glm::distance(origin, glm::vec3(0.0f, 0.0f, 0.0f)); // projected size ve stredu. Vepredu budou jemnejsi bloky, vzadu hrubsi

    render_simd(viewport_width, viewport_height, viewport_fov, view * mt, [&](const simd::Ray &ray, const simd::float_m &mask) {
      integrate_tree_simd<T, N>(info, ray, step, mask, [&](const simd::vec3 &cell, uint32_t layer, const simd::float_m &mask) {
        float cell_size = exp2i(-layer);
        float child_size = exp2i(-layer - 1);

        simd::float_v block_distance;

        for (uint8_t k = 0; k < simd::len; k++) {
          if (mask[k]) {
            glm::vec3 single_cell { cell.x[k], cell.y[k], cell.z[k] };
            glm::vec3 cell_center = mt * glm::vec4(single_cell + child_size, 1.f);  // convert to world space

            block_distance[k] = glm::distance(origin, cell_center); // convert to world space
          }
        }

        return simd::float_m(layer == std::size(info.layers) - 1) || cell_size <= block_distance * projected_size;
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
void packlet_samples(const typename TreeVolume<T, N>::Info &info, float quality, const F &func) {
  glm::vec3 volume_frac = glm::vec3(info.width_frac, info.height_frac, info.depth_frac);
  glm::mat4 texture = glm::scale(glm::mat4(1.f), 1.f / volume_frac);

  glm::mat4 mt = model * texture;

  generate_origins([&](const glm::vec3 &origin) {
    glm::mat4 view = glm::lookAt(origin, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    float projected_size = quality * TreeVolume<T, N>::BLOCK_SIDE / glm::distance(origin, glm::vec3(0.0f, 0.0f, 0.0f)); // projected size ve stredu. Vepredu budou jemnejsi bloky, vzadu hrubsi

    render_packlet(viewport_width, viewport_height, viewport_fov, view * mt, [&](const RayPacklet &ray, const MaskPacklet &mask) {
      integrate_tree_packlet<T, N>(info, ray, step, mask, [&](const Vec3Packlet &cell, uint32_t layer, const MaskPacklet &mask) {
        float cell_size = exp2i(-layer);
        float child_size = exp2i(-layer - 1);

        MaskPacklet output_mask = mask;

        for (uint8_t j = 0; j < simd::len; j++) {
          if (mask[j].isNotEmpty()) {
            simd::float_v block_distance;

            for (uint8_t k = 0; k < simd::len; k++) {
              if (mask[j][k]) {
                glm::vec3 single_cell { cell[j].x[k], cell[j].y[k], cell[j].z[k] };
                glm::vec3 cell_center = mt * glm::vec4(single_cell + child_size, 1.f);  // convert to world space

                block_distance[k] = glm::distance(origin, cell_center); // convert to world space
              }
            }

            output_mask[j] = simd::float_m(layer == std::size(info.layers) - 1) || cell_size <= block_distance * projected_size;
          }
        }

        return output_mask;
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
void test_tree(uint32_t width, uint32_t height, uint32_t depth, float quality) {

  typename TreeVolume<T, N>::Info info(width, height, depth);

  {
    std::ofstream data(tmp_file_name, std::ios::binary);
    if (!data) {
      throw std::runtime_error(std::string("Unable to open '") + tmp_file_name + "'!");
    }

    data.seekp(info.size_in_nodes * TreeVolume<T, N>::BLOCK_BYTES - 1);
    data.write("", 1);
  }

  MappedFile volume(tmp_file_name, 0, info.size_in_nodes * TreeVolume<T, N>::BLOCK_BYTES, MappedFile::READ, MappedFile::SHARED);

  const T *volume_data = reinterpret_cast<const T *>(volume.data());

  size_t samples = 0;

  scalar_samples<T, N>(info, quality, [&](uint64_t node_handle, const glm::vec3 &in_block_pos) {
    samples++;
    sample_morton_scalar<T, N>(volume_data, node_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);
  });

  std::cout << "tree" << N << " ";

  std::cout << (info.size_in_nodes * TreeVolume<T, N>::BLOCK_SIZE) / (float(width) * height * depth) << " "; // WORKS only for full resolution rendering bc other layers not used

  uint64_t used_blocks = 0;
  generate_origins([&](const glm::vec3 &origin) {
    glm::vec3 volume_frac = glm::vec3(info.width_frac, info.height_frac, info.depth_frac);
    glm::mat4 texture = glm::scale(glm::mat4(1.f), 1.f / volume_frac);

    glm::mat4 mt = model * texture;

    float projected_size = quality * TreeVolume<T, N>::BLOCK_SIDE / glm::distance(origin, glm::vec3(0.0f, 0.0f, 0.0f)); // projected size ve stredu. Vepredu budou jemnejsi bloky, vzadu hrubsi

    scan_tree<T, N>(info, {}, 0, [&](const glm::vec3 &cell, uint8_t layer) {
      float cell_size = exp2i(-layer);
      float child_size = exp2i(-layer - 1);

      glm::vec3 cell_center = mt * glm::vec4(cell + child_size, 1.f);  // convert to world space

      float block_distance = glm::distance(origin, cell_center);

      if (layer == std::size(info.layers) - 1 || cell_size <= block_distance * projected_size) {
        used_blocks++;
        return false;
      }
      else {
        return true;
      }
    });
  });

  std::cout << used_blocks * TreeVolume<T, N>::BLOCK_BYTES / n << " ";

  std::cout << measure_ns([&]{
    scalar_samples<T, N>(info, quality, [&](uint64_t node_handle, const glm::vec3 &in_block_pos) {
      sample_linear_scalar<T, N>(volume_data, node_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    scalar_samples<T, N>(info, quality, [&](uint64_t node_handle, const glm::vec3 &in_block_pos) {
      sample_morton_scalar<T, N>(volume_data, node_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    simd_samples<T, N>(info, quality, [&](const std::array<uint64_t, simd::len> &block_indices, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
      sample_linear_simd<T, N>(volume_data, block_indices, in_block_pos.x, in_block_pos.y, in_block_pos.z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    simd_samples<T, N>(info, quality, [&](const std::array<uint64_t, simd::len> &block_indices, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
      sample_morton_simd<T, N>(volume_data, block_indices, in_block_pos.x, in_block_pos.y, in_block_pos.z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    packlet_samples<T, N>(info, quality, [&](const std::array<uint64_t, simd::len> &block_indices, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
      sample_linear_simd<T, N>(volume_data, block_indices, in_block_pos.x, in_block_pos.y, in_block_pos.z, mask);
    });
  }) / samples << " ";

  std::cout << measure_ns([&]{
    packlet_samples<T, N>(info, quality, [&](const std::array<uint64_t, simd::len> &block_indices, const simd::vec3 &in_block_pos, const simd::float_m &mask) {
      sample_morton_simd<T, N>(volume_data, block_indices, in_block_pos.x, in_block_pos.y, in_block_pos.z, mask);
    });
  }) / samples << " ";

  std::cout << "\n";
}

template <typename T, uint32_t N>
void print_workset_size(uint32_t width, uint32_t height, uint32_t depth, float quality) {
  typename TreeVolume<T, N>::Info info(width, height, depth);

  uint64_t used_blocks = 0;
  generate_origins([&](const glm::vec3 &origin) {
    glm::vec3 volume_frac = glm::vec3(info.width_frac, info.height_frac, info.depth_frac);
    glm::mat4 texture = glm::scale(glm::mat4(1.f), 1.f / volume_frac);

    glm::mat4 mt = model * texture;

    float projected_size = quality * TreeVolume<T, N>::BLOCK_SIDE / glm::distance(origin, glm::vec3(0.0f, 0.0f, 0.0f)); // projected size ve stredu. Vepredu budou jemnejsi bloky, vzadu hrubsi

    scan_tree<T, N>(info, {}, 0, [&](const glm::vec3 &cell, uint8_t layer) {
      float cell_size = exp2i(-layer);
      float child_size = exp2i(-layer - 1);

      glm::vec3 cell_center = mt * glm::vec4(cell + child_size, 1.f);  // convert to world space

      float block_distance = glm::distance(origin, cell_center);

      if (layer == std::size(info.layers) - 1 || cell_size <= block_distance * projected_size) {
        used_blocks++;
        return false;
      }
      else {
        return true;
      }
    });
  });

  std::cout << "tree" << N << " ";

  std::cout << used_blocks * TreeVolume<T, N>::BLOCK_BYTES / n << "\n";
}

template <typename T>
void workset_sizes(uint32_t width, uint32_t height, uint32_t depth, float quality) {
  print_workset_size<T, 3>(width, height, depth, quality);
  print_workset_size<T, 4>(width, height, depth, quality);
  print_workset_size<T, 5>(width, height, depth, quality);
  print_workset_size<T, 6>(width, height, depth, quality);
  print_workset_size<T, 7>(width, height, depth, quality);
  print_workset_size<T, 8>(width, height, depth, quality);
}

template <typename T>
void test(uint32_t width, uint32_t height, uint32_t depth, float quality) {
  test_tree<T, 3>(width, height, depth, quality);
  test_tree<T, 4>(width, height, depth, quality);
  test_tree<T, 5>(width, height, depth, quality);
  test_tree<T, 6>(width, height, depth, quality);
  test_tree<T, 7>(width, height, depth, quality);
  test_tree<T, 8>(width, height, depth, quality);
}

int main(int argc, const char *argv[]) {
  uint32_t width, height, depth, bytes_per_voxel;

  float quality = 0.025f;

  parse_args_reduced(argc, argv, width, height, depth, bytes_per_voxel, quality);

  std::cout << "#" << width << " " << height << " " << depth << " " << bytes_per_voxel << "\n";
  std::cout << "# type overhead workset-bytes linear-scalar morton-scalar linear-simd morton-simd linear-packlet morton-packlet\n";

  if (bytes_per_voxel == 1) {
    workset_sizes<uint8_t>(width, height, depth, quality);
    test<uint8_t>(width, height, depth, quality);
  }
  else if (bytes_per_voxel == 2) {
    workset_sizes<uint16_t>(width, height, depth, quality);
    test<uint16_t>(width, height, depth, quality);
  }
  else {
    throw std::runtime_error("Only one or two bytes per voxel!");
  }

  return 0;
}
