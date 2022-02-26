
#include "blocked_volume.h"
#include "intersection.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cstddef>

#include <numeric>

struct RayRange {
  float min;
  float max;
};

struct Ray {
  Ray(const glm::vec3 &origin, const glm::vec3 &direction): origin(origin), direction(direction), direction_inverse(1.f / direction) {}
  const glm::vec3 origin;
  const glm::vec3 direction;
  const glm::vec3 direction_inverse;
};

// Interactive isosurface ray tracing of large octree volumes
// https://www.researchgate.net/publication/310054812_Interactive_isosurface_ray_tracing_of_large_octree_volumes

// TODO everything is in cannonical blocked volume space. Test if it is possible to use it in interval [0, 1] and possibly keep it in integer arithmetics
template <typename F>
void recursive_integrate(const Ray &ray, const RayRange &range, const glm::vec<3, uint32_t> &cell, uint32_t layer, const F &callback) {
  if (callback(range, cell, layer)) {
    uint32_t child_bit = 1 << layer;

    glm::vec<3, uint32_t> center = cell | child_bit;

    glm::vec3 tcenter = (glm::vec3(center) - ray.origin) * ray.direction_inverse;

    // fast sort axis by tcenter
    std::array<uint8_t, 3> axis;

    bool gt01 = tcenter[0] > tcenter[1];
    bool gt02 = tcenter[0] > tcenter[2];
    bool gt12 = tcenter[1] > tcenter[2];

    axis[ gt01 +  gt02] = 0;
    axis[!gt01 +  gt12] = 1;
    axis[!gt02 + !gt12] = 2;

    glm::vec3 penter = ray.origin + ray.direction * range.min;

    glm::vec<3, uint32_t> child_cell {
      penter.x > center.x ? center.x : cell.x,
      penter.y > center.y ? center.y : cell.y,
      penter.z > center.z ? center.z : cell.z
    };

    float tmin = range.min;
    for (uint8_t i = 0; i < 3; i++) {
      float tmax = std::min(tcenter[axis[i]], range.max);

      if (tmin < tmax) {
        recursive_integrate(ray, { tmin, tmax }, child_cell, layer - 1, callback);
        tmin = tmax;

        // TODO is this right?
        child_cell[axis[i]] ^= child_bit;
      }
    }

    if (tmin < range.max) {
      recursive_integrate(ray, { tmin, range.max }, child_cell, layer - 1, callback);
    }
  }
}

template <typename T, typename TransferFunctionType, typename OutputFunctionType>
void render(const BlockedVolume<T> &volume, const glm::mat4 &mv, uint32_t width, uint32_t height, float yfov_degrees, float step, const TransferFunctionType &transfer_function, const OutputFunctionType &output_function) {

  glm::vec3 size_in_blocks = glm::vec3(1 << (std::size(volume.info.layers) - 1));

  // This transfroms ray from cannocical clip space to cannonical blocked volume space where volume is inside boundig box [0, 2^(num_layers - 1)]
  glm::mat4 ray_transform = glm::scale(glm::mat4(1.f), size_in_blocks) * glm::translate(glm::mat4(1.f), glm::vec3(0.5f, 0.5f, 0.5f)) * glm::inverse(mv);

  glm::vec3 origin = ray_transform * glm::vec4(0.f, 0.f, 0.f, 1.f);

  float yfov = glm::radians(yfov_degrees);
  float yfov_coef = std::tan(yfov / 2.f);
  float xfov_coef = yfov * width / height;

  float width_coef_avg  = xfov_coef / width;
  float width_coef  = 2.f * width_coef_avg;
  float width_shift = width_coef_avg - xfov_coef;

  float height_coef_avg = yfov_coef / height;
  float height_coef = 2.f * height_coef_avg;
  float height_shift = height_coef_avg - yfov_coef;

  for (uint32_t j = 0; j < height; j++) {
    float y = j * height_coef + height_shift;

    for (uint32_t i = 0; i < width; i++) {
      float x = i * width_coef + width_shift;

      Ray ray(origin, ray_transform * glm::normalize(glm::vec4(x, y, -1.f, 0.f)));

      float tmin {};
      float tmax {};

      intersect_aabb_ray(ray.origin, ray.direction_inverse, {}, size_in_blocks, tmin, tmax);

      glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

      auto integrate = [&](float v0, float v1, float step) {
        auto rgba = transfer_function(v0, v1);

        if (rgba.a > 0.f) { // empty space skipping
          float alpha = 1.f - std::exp(-rgba.a * step);

          float coef = alpha * dst.a;

          dst.r += rgba.r * coef;
          dst.g += rgba.g * coef;
          dst.b += rgba.b * coef;
          dst.a *= 1.f - alpha;
        }
      };

      if (tmin < tmax) {
        recursive_integrate(ray, { tmin, tmax }, { 0, 0, 0 }, std::size(volume.info.layers) - 1, [&](const RayRange &range, const glm::vec<3, uint32_t> &cell, uint32_t layer) {
          //cell in cannonical blocked volume space [0, 2^(num_layers - 1)]
          //block in layer space [0, 2^(num_layers - layer - 1)]
          auto block = cell >> layer;

          const auto &node = volume.nodes[volume.info.node_handle(block.x, block.y, block.z, layer)];

          if (node.min == node.max) { // fast integration
            integrate(node.min, node.max, range.max - range.min);
          }
          else { // integration by sampling

            // dumb condition that causes the recursion to go all the way down.
            // TODO do something elaborate here
            if (layer) {
              return true;
            }

            float t = range.min;
            float prev_value;

            // TODO move prev outside and initialize it somehow. Same with t;

            // direction and everything is in cannonical BLOCKED volume space, but we need to step the position in cannocical volume space

            {
              glm::vec3 pos = (ray.origin + ray.direction * t) / float(1 << layer);
              glm::vec3 in_block_pos = (pos - glm::vec3(block)) * float(BlockedVolume<T>::SUBVOLUME_SIDE) - .5f;
              prev_value = volume.sample_block(node.block_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);
            }

            while (t < range.max) {
              float next_step = std::min(step * (layer + 1), range.max - t);

              t += next_step;

              // fixme division
              glm::vec3 pos = (ray.origin + ray.direction * t) / float(1 << layer);
              glm::vec3 in_block_pos = (pos - glm::vec3(block)) * float(BlockedVolume<T>::SUBVOLUME_SIDE) - .5f;

              float value = volume.sample_block(node.block_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);

              integrate(value, prev_value, next_step);

              prev_value = value;
            }
          }

          return false;
        });
      }

      output_function(i, j, dst);
    }
  }
}
