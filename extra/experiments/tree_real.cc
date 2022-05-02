#include "common.h"
#include "../../tools/tf1d.h"

#include <raw_volume/raw_volume.h>

#include <tree_volume/tree_volume.h>
#include <tree_volume/processor.h>

#include <renderers/scalar.h>
#include <renderers/simd.h>
#include <renderers/packlet.h>

#include <integrators/tree_slab.h>
#include <integrators/tree_slab_simd.h>
#include <integrators/tree_slab_packlet.h>

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

size_t num = 0.f;

void parse_args(int argc, const char *argv[], const char *&raw_volume, const char *&transfer_function, uint32_t &width, uint32_t &height, uint32_t &depth, uint32_t &bytes_per_voxel, float &step, uint32_t &window_width, uint32_t &window_height) {
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
    arg >> window_width;
    if (!arg) {
      throw std::runtime_error("Unable to parse window width!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[9];
    arg >> window_height;
    if (!arg) {
      throw std::runtime_error("Unable to parse window height!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[10];
    arg >> num;
    if (!arg) {
      throw std::runtime_error("Unable to parse num!");
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
void vizualization_benchmark(const char *raw_volume_file_name, const char *transfer_function_file_name, uint32_t width, uint32_t height, uint32_t depth, float step, uint32_t window_width, uint32_t window_height) {
  RawVolume<T> raw_volume(raw_volume_file_name, width, height, depth);

  std::cout << "# " << width << " " << height << " " << depth << " " << sizeof(T) << " " << (1 << N) << "\n";

  std::cout << "# " << measure_ms([&]{
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

  glm::vec3 volume_frac = glm::vec3(volume.info.width_frac, volume.info.height_frac, volume.info.depth_frac);

  glm::vec3 camera_up = glm::vec3(0.0f, 1.0f, 0.0f);

  float terminate_thresh = 0.01f;

  // converts volume from volume space [-.5, .5] to texture space [0, volume_frac];
  glm::mat4 texture =
    glm::translate(glm::mat4(1.f), glm::vec3(-.5f)) * //shifts the visible part of the volume the center
    glm::scale(glm::mat4(1.f), 1.f / volume_frac); //transforms the visible part of the volume to [0..1]

  glm::mat4 view = glm::lookAt(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), camera_up);

  struct Info {
    double time_scalar = 0.f;
    double time_simd = 0.f;
    double time_packlet = 0.f;

    double mse_scalar = 0.f;
    double mse_simd = 0.f;
    double mse_packlet = 0.f;
  };

  double fullscreen_time = 0.f;
  std::map<size_t, Info> layer_times;
  std::map<float, Info> quality_times;

  for (size_t i = 0; i < num; i++) {
    glm::mat4 model = glm::rotate(glm::mat4(1.f), i * 0.314f, glm::vec3(0.f, 1.f, 0.f));

    glm::mat4 vmt = view * model * texture; // converts volume from world space to camera space

    fullscreen_time += measure_ms([&]{
      render_simd(window_width, window_height, viewport_fov, vmt, raster_reference.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
        return integrate_tree_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector, [&](const simd::float_v &, uint8_t layer, const simd::float_m &) {
          return simd::float_m(layer == std::size(volume.info.layers) - 1);
        });
      });
    });

    for (uint8_t desired_layer = 0; desired_layer < std::size(volume.info.layers) - 1; desired_layer++) {
      render_scalar(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const Ray &ray) {
        return integrate_tree_slab(volume, ray, step, terminate_thresh, transfer_function_scalar, [&](const float &, uint32_t layer) {
          return layer == desired_layer;
        });
      });

      layer_times[desired_layer].time_scalar += measure_ms([&]{
        render_scalar(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const Ray &ray) {
          return integrate_tree_slab(volume, ray, step, terminate_thresh, transfer_function_scalar, [&](const float &, uint32_t layer) {
            return layer == desired_layer;
          });
        });
      });

      layer_times[desired_layer].mse_scalar += MSE(raster_reference.data(), raster.data(), window_size);

      render_simd(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
        return integrate_tree_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector, [&](const simd::float_v &, uint32_t layer, const simd::float_m &) {
          return simd::float_m(layer == desired_layer);
        });
      });

      layer_times[desired_layer].time_simd += measure_ms([&]{
        render_simd(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
          return integrate_tree_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector, [&](const simd::float_v &, uint32_t layer, const simd::float_m &) {
            return simd::float_m(layer == desired_layer);
          });
        });
      });

      layer_times[desired_layer].mse_simd += MSE(raster_reference.data(), raster.data(), window_size);

      render_packlet(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const RayPacklet &ray_packlet, const MaskPacklet &mask_packlet) {
        return integrate_tree_slab_packlet(volume, ray_packlet, step, terminate_thresh, mask_packlet, transfer_function_vector, [&](const Vec4Packlet &, uint32_t layer, const MaskPacklet &mask) {
          MaskPacklet output_mask = mask;

          for (uint8_t j = 0; j < simd::len; j++) {
            if (mask[j].isNotEmpty()) {
              output_mask[j] = simd::float_m(layer == desired_layer);
            }
          }

          return output_mask;
        });
      });

      layer_times[desired_layer].time_packlet += measure_ms([&]{
        render_packlet(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const RayPacklet &ray_packlet, const MaskPacklet &mask_packlet) {
          return integrate_tree_slab_packlet(volume, ray_packlet, step, terminate_thresh, mask_packlet, transfer_function_vector, [&](const Vec4Packlet &, uint32_t layer, const MaskPacklet &mask) {
            MaskPacklet output_mask = mask;

            for (uint8_t j = 0; j < simd::len; j++) {
              if (mask[j].isNotEmpty()) {
                output_mask[j] = simd::float_m(layer == desired_layer);
              }
            }

            return output_mask;
          });
        });
      });

      layer_times[desired_layer].mse_packlet += MSE(raster_reference.data(), raster.data(), window_size);
    }

    for (float quality: {1.f, 2.f, 4.f, 6.f, 8.f, 10.f, 12.f, 14.f, 16.f, 18.f, 20.f, 22.f, 24.f, 26.f, 28.f, 30.f, 32.f, 34.f}) {
      render_scalar(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const Ray &ray) {
        return integrate_tree_slab(volume, ray, step, terminate_thresh, transfer_function_scalar, [&](const float &alpha, uint32_t layer) {
          float desired_layer = -log2(1 - alpha) * quality;
          return layer == std::size(volume.info.layers) - 1 || desired_layer <= layer;
        });
      });

      quality_times[quality].time_scalar += measure_ms([&]{
        render_scalar(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const Ray &ray) {
          return integrate_tree_slab(volume, ray, step, terminate_thresh, transfer_function_scalar, [&](const float &alpha, uint32_t layer) {
            float desired_layer = -log2(1 - alpha) * quality;
            return layer == std::size(volume.info.layers) - 1 || desired_layer <= layer;
          });
        });
      });

      quality_times[quality].mse_scalar += MSE(raster_reference.data(), raster.data(), window_size);

      render_simd(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
        return integrate_tree_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector, [&](const simd::float_v &alpha, uint32_t layer, const simd::float_m &) {
          simd::float_v desired_layer = -log2(simd::float_v(1) - alpha) * quality;
          return simd::float_m(layer == std::size(volume.info.layers) - 1) || desired_layer <= layer;
        });
      });

      quality_times[quality].time_simd += measure_ms([&]{
        render_simd(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const simd::Ray &ray, const simd::float_m &mask) {
          return integrate_tree_slab_simd(volume, ray, step, terminate_thresh, mask, transfer_function_vector, [&](const simd::float_v &alpha, uint32_t layer, const simd::float_m &) {
            simd::float_v desired_layer = -log2(simd::float_v(1) - alpha) * quality;
            return simd::float_m(layer == std::size(volume.info.layers) - 1) || desired_layer <= layer;
          });
        });
      });

      quality_times[quality].mse_simd += MSE(raster_reference.data(), raster.data(), window_size);

      render_packlet(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const RayPacklet &ray_packlet, const MaskPacklet &mask_packlet) {
        return integrate_tree_slab_packlet(volume, ray_packlet, step, terminate_thresh, mask_packlet, transfer_function_vector, [&](const Vec4Packlet &rgba, uint32_t layer, const MaskPacklet &mask) {
          MaskPacklet output_mask {};

          for (uint8_t j = 0; j < simd::len; j++) {
            if (mask[j].isNotEmpty()) {
              simd::float_v desired_layer = -log2(simd::float_v(1) - rgba[j].a) * quality;
              output_mask[j] = simd::float_m(layer == std::size(volume.info.layers) - 1) || desired_layer <= layer;
            }
          }

          return output_mask;
        });
      });

      quality_times[quality].time_packlet += measure_ms([&]{
        render_packlet(window_width, window_height, viewport_fov, vmt, raster.data(), [&](const RayPacklet &ray_packlet, const MaskPacklet &mask_packlet) {
          return integrate_tree_slab_packlet(volume, ray_packlet, step, terminate_thresh, mask_packlet, transfer_function_vector, [&](const Vec4Packlet &rgba, uint32_t layer, const MaskPacklet &mask) {
            MaskPacklet output_mask {};

            for (uint8_t j = 0; j < simd::len; j++) {
              if (mask[j].isNotEmpty()) {
                simd::float_v desired_layer = -log2(simd::float_v(1) - rgba[j].a) * quality;
                output_mask[j] = simd::float_m(layer == std::size(volume.info.layers) - 1) || desired_layer <= layer;
              }
            }

            return output_mask;
          });
        });
      });

      quality_times[quality].mse_packlet += MSE(raster_reference.data(), raster.data(), window_size);
    }
  }

  std::cout << fullscreen_time / (num * window_width * window_height) << "\n";

  std::cout << "# layer renderer\n";
  std::cout << "# layer scalar-time scalar-psnr simd-time simd-psnr packlet-time packlet-psnr\n";

  for (const auto &[layer, info]: layer_times) {
    std::cout << layer << " ";
    std::cout << info.time_scalar / (num * window_width * window_height) << " " << PSNR(info.mse_scalar / num) << " ";
    std::cout << info.time_simd / (num * window_width * window_height) << " " << PSNR(info.mse_simd / num) << " ";
    std::cout << info.time_packlet / (num * window_width * window_height) << " " << PSNR(info.mse_packlet / num) << "\n";
  }

  std::cout << "# alpha renderer\n";
  std::cout << "# quality scalar-time scalar-psnr simd-time simd-psnr packlet-time packlet-psnr\n";

  for (const auto &[quality, info]: quality_times) {
    std::cout << double(quality) << " ";
    std::cout << info.time_scalar  / (num * window_width * window_height) << " " << PSNR(info.mse_scalar  / num) << " ";
    std::cout << info.time_simd    / (num * window_width * window_height) << " " << PSNR(info.mse_simd    / num) << " ";
    std::cout << info.time_packlet / (num * window_width * window_height) << " " << PSNR(info.mse_packlet / num) << "\n";
  }

  std::cout << "\n";
}

template <typename T>
void benchmarks(const char *raw_volume_file_name, const char *transfer_function_file_name, uint32_t width, uint32_t height, uint32_t depth, float step, uint32_t image_width, uint32_t image_height) {
  vizualization_benchmark<T, 4>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, image_width, image_height);
  vizualization_benchmark<T, 5>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, image_width, image_height);
  vizualization_benchmark<T, 6>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, image_width, image_height);
  vizualization_benchmark<T, 7>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, image_width, image_height);
  vizualization_benchmark<T, 8>(raw_volume_file_name, transfer_function_file_name, width, height, depth, step, image_width, image_height);
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
    uint32_t image_width;
    uint32_t image_height;

    parse_args(argc, argv, raw_volume, transfer_function, width, height, depth, bytes_per_voxel, step, image_width, image_height);

    if (bytes_per_voxel == 1) {
      benchmarks<uint8_t>(raw_volume, transfer_function, width, height, depth, step, image_width, image_height);
    }
    else if (bytes_per_voxel == 2) {
      benchmarks<uint16_t>(raw_volume, transfer_function, width, height, depth, step, image_width, image_height);
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
