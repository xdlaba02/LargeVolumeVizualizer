#include "common.h"
#include "../../tools/tf1d.h"

#include <raw_volume/raw_volume.h>

#include <tree_volume/tree_volume.h>
#include <tree_volume/processor.h>

#include <utils/scan_tree.h>

#include <renderers/scalar.h>
#include <renderers/simd.h>
#include <renderers/packlet.h>

#include <integrators/tree_slab.h>
#include <integrators/tree_slab_simd.h>
#include <integrators/tree_slab_packlet.h>
#include <integrators/tree_slab_layer.h>
#include <integrators/tree_slab_layer_simd.h>
#include <integrators/tree_slab_layer_dda.h>

#include <transfer/piecewise_linear.h>
#include <transfer/preintegrate_function.h>

#include <transfer/texture2D/texture2D.h>
#include <transfer/texture2D/sampler.h>
#include <transfer/texture2D/sampler_simd.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>

template <typename F>
void generate_origins(const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(0.f, 1.f);
  std::normal_distribution<float> normal {};

  for (uint64_t i = 0; i < n; i++) {
    func(glm::normalize(glm::vec3(normal(re), normal(re), normal(re))));
  }
}

void parse_args(int argc, const char *argv[], const char *&raw_volume, const char *&transfer_function, uint32_t &width, uint32_t &height, uint32_t &depth, uint32_t &bytes_per_voxel, float &step, float &quality, uint32_t &window_width, uint32_t &window_height) {
  if (argc != 11) {
    throw std::runtime_error("Wrong number of arguments!");
  }

  raw_volume        = argv[1];
  transfer_function = argv[6];

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
      throw std::runtime_error("Unable to parse bytes per sample!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[7];
    arg >> step;
    if (!arg) {
      throw std::runtime_error("Unable to parse step!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[8];
    arg >> quality;
    if (!arg) {
      throw std::runtime_error("Unable to parse quality!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[9];
    arg >> window_width;
    if (!arg) {
      throw std::runtime_error("Unable to parse window width!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[10];
    arg >> window_height;
    if (!arg) {
      throw std::runtime_error("Unable to parse window height!");
    }
  }
}

double MSE(const uint8_t *a, const uint8_t *b, size_t size) {
  double se = 0.;

  for (size_t i = 0; i < size; i++) {
    se += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
  }

  return se / size;
}

double PSNR(double mse) {
  return 10 * log10(255 * 255 / mse);
}

template <typename T, size_t N>
void vizualization_benchmark(const char *raw_volume_file_name, const char *transfer_function_file_name, uint32_t width, uint32_t height, uint32_t depth, float step, float quality, uint32_t window_width, uint32_t window_height) {
  RawVolume<T> raw_volume(raw_volume_file_name, width, height, depth);

  std::cout << (1 << N) << " ";

  std::cout << measure_ms([&]{
    process_volume<T, N>(width, height, depth, "TMP.proc", "TMP.meta", [&](uint32_t x, uint32_t y, uint32_t z) {
      return raw_volume.data[raw_volume.voxel_handle(x, y, z)];
    });
  }) / 1000 << " ";

  TreeVolume<T, N> volume("TMP.proc", "TMP.meta", width, height, depth);

  size_t orig_size = std::filesystem::file_size(raw_volume_file_name);
  size_t tree_size = std::filesystem::file_size("TMP.proc") + std::filesystem::file_size("TMP.meta");

  std::cout << double(tree_size) / orig_size << " ";

  TF1D tf1d = TF1D::load_from_file(transfer_function_file_name);

  static constinit uint32_t pre_size = 256;

  Texture2D<float> transfer_r = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.rgb, v * 256 / pre_size).r; });
  Texture2D<float> transfer_g = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.rgb, v * 256 / pre_size).g; });
  Texture2D<float> transfer_b = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.rgb, v * 256 / pre_size).b; });
  Texture2D<float> transfer_a = preintegrate_function(pre_size, [&](float v){ return piecewise_linear(tf1d.a,   v * 256 / pre_size);   });

  auto transfer_function_vector = [&](const simd::float_v &begin, const simd::float_v &end, const simd::float_m &mask) -> simd::vec4 {
    simd::SampleInfo info = sample_info(pre_size, pre_size, (begin + 0.5f) / (1 << (sizeof(T) * 8)), (end + 0.5f) / (1 << (sizeof(T) * 8)));

    return {
      sample(transfer_r, info, mask),
      sample(transfer_g, info, mask),
      sample(transfer_b, info, mask),
      sample(transfer_a, info, mask)
    };
  };

  auto transfer_function_scalar = [&](float begin, float end) -> glm::vec4 {
    SampleInfo info = sample_info(pre_size, pre_size, (begin + 0.5f) / (1 << (sizeof(T) * 8)), (end + 0.5f) / (1 << (sizeof(T) * 8)));

    return {
      sample(transfer_r, info),
      sample(transfer_g, info),
      sample(transfer_b, info),
      sample(transfer_a, info)
    };
  };

  size_t window_size = size_t(window_width) * window_height * 3;

  std::vector<uint8_t> raster_reference(window_size);
  std::vector<uint8_t> raster(window_size);

  const glm::vec3 volume_frac = glm::vec3(volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac);

  const glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  float terminate_thresh = 0.01f;

  // converts volume from volume space [-.5, .5] to texture space [0, volume_frac];
  glm::mat4 mt =
    glm::translate(glm::mat4(1.f), glm::vec3(-.5f)) * //shifts the visible part of the volume the center
    glm::scale(glm::mat4(1.f), 1.f / volume_frac); //transforms the visible part of the volume to [0..1]

  double time_scalar = 0.f;
  double time_simd = 0.f;
  double time_packlet = 0.f;

  double mse_scalar = 0.f;
  double mse_simd = 0.f;
  double mse_packlet = 0.f;

  generate_origins([&](const glm::vec3 &origin) {
    glm::mat4 view = glm::lookAt(origin, glm::vec3(0.0f, 0.0f, 0.0f), camera_up);

    glm::mat4 vmt = view * mt; // converts volume from world space to camera space

    float projected_size = quality * TreeVolume<T, N>::BLOCK_SIDE / glm::distance(origin, glm::vec3(0.0f, 0.0f, 0.0f)); // projected size ve stredu. Vepredu budou jemnejsi bloky, vzadu hrubsi

    render_simd(window_width, window_height, viewport_fov, vmt, raster_reference.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
      return integrate_tree_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector, [&](const simd::vec3 &, uint8_t layer, const simd::float_m &) {
        return simd::float_m(layer == std::size(volume.info.layers) - 1);
      });
    });

    std::cerr << "A\n";

    time_scalar += measure_ms([&]{
      render_scalar(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const Ray &ray) {
        return integrate_tree_slab(volume, ray, step, terminate_thresh, transfer_function_scalar, [&](const glm::vec3 &cell, uint8_t layer) {
          float cell_size = exp2i(-layer);
          float child_size = exp2i(-layer - 1);

          glm::vec3 cell_center = mt * glm::vec4(cell + child_size, 1.f);  // convert to world space

          float block_distance = glm::distance(origin, cell_center);

          return layer == std::size(volume.info.layers) - 1 || cell_size <= block_distance * projected_size;
        });
      });
    });

    std::cerr << "B\n";

    mse_scalar += MSE(raster_reference.data(), raster.data(), window_size);

    time_simd += measure_ms([&]{

      render_simd(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
        return integrate_tree_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector, [&](const simd::vec3 &cell, uint8_t layer, const simd::float_m &mask) {
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

          return simd::float_m(layer == std::size(volume.info.layers) - 1) || cell_size <= block_distance * projected_size;
        });
      });
    });

    std::cerr << "C\n";

    mse_simd += MSE(raster_reference.data(), raster.data(), window_size);

    time_packlet += measure_ms([&]{

      render_packlet(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const RayPacklet &ray_packlet, const MaskPacklet &mask_packlet) {
        return integrate_tree_slab_packlet(volume, ray_packlet, step, terminate_thresh, mask_packlet, transfer_function_vector, [&](const Vec3Packlet &cell_packlet, uint8_t layer, const MaskPacklet &mask_packlet) {
          float cell_size = exp2i(-layer);
          float child_size = exp2i(-layer - 1);

          MaskPacklet output_mask_packlet = mask_packlet;

          for (uint8_t j = 0; j < simd::len; j++) {
            if (mask_packlet[j].isNotEmpty()) {
              simd::float_v block_distance;

              for (uint8_t k = 0; k < simd::len; k++) {
                if (mask_packlet[j][k]) {
                  glm::vec3 single_cell { cell_packlet[j].x[k], cell_packlet[j].y[k], cell_packlet[j].z[k] };
                  glm::vec3 cell_center = mt * glm::vec4(single_cell + child_size, 1.f);  // convert to world space

                  block_distance[k] = glm::distance(origin, cell_center); // convert to world space
                }
              }

              output_mask_packlet[j] = simd::float_m(layer == std::size(volume.info.layers) - 1) || cell_size <= block_distance * projected_size;
            }
          }

          return output_mask_packlet;
        });
      });
    });

    mse_packlet += MSE(raster_reference.data(), raster.data(), window_size);
  });

  std::cerr << "D\n";

  std::cout << time_scalar / n / 1000 << " " << PSNR(mse_scalar / n) << " ";
  std::cout << time_simd / n / 1000 << " " << PSNR(mse_simd / n) << " ";
  std::cout << time_packlet / n / 1000 << " " << PSNR(mse_packlet / n) << "\n";

}

template <typename T>
void benchmarks(const char *raw_volume_file_name, const char *transfer_function_file_name, uint32_t width, uint32_t height, uint32_t depth, float step, float quality, uint32_t image_width, uint32_t image_height) {
  //vizualization_benchmark<T, 3>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, quality, image_width, image_height);
  //vizualization_benchmark<T, 4>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, quality, image_width, image_height);
  //vizualization_benchmark<T, 5>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, quality, image_width, image_height);
  //vizualization_benchmark<T, 6>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, quality, image_width, image_height);
  vizualization_benchmark<T, 7>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, quality, image_width, image_height);
  vizualization_benchmark<T, 8>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, quality, image_width, image_height);
}

int main(int argc, const char *argv[]) {
  try {
    const char *raw_volume;
    const char *transfer_function;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t bytes_per_voxel;
    float step;
    float quality;
    uint32_t image_width;
    uint32_t image_height;

    parse_args(argc, argv, raw_volume, transfer_function, width, height, depth, bytes_per_voxel, step, quality, image_width, image_height);

    std::cout << "# size process_time overhead scalar-time scalar-psnr simd-time simd-psnr packlet-time packlet-psnr\n";

    if (bytes_per_voxel == 1) {
      benchmarks<uint8_t>(raw_volume, transfer_function, width, height, depth, step, quality, image_width, image_height);
    }
    else if (bytes_per_voxel == 2) {
      benchmarks<uint16_t>(raw_volume, transfer_function, width, height, depth, step, quality, image_width, image_height);
    }
    else {
      throw std::runtime_error("Only one or two bytes per voxel!");
    }

  }
  catch (const std::runtime_error& e) {
    std::cerr << e.what() << "\n";
    std::cerr << "Usage: \n";
    std::cerr << argv[0] << " <raw-volume> <width> <height> <depth> <bytes-per-voxel> <transfer-function>\n";
  }

  return 0;
}
