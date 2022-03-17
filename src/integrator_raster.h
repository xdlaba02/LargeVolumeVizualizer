#pragma once

#include "tree_volume/tree_volume.h"
#include "ray_raster_traversal.h"
#include "intersection.h"

#include <glm/glm.hpp>

#include <cstdint>

template <typename T, typename TransferFunctionType>
glm::vec4 integrate(const TreeVolume<T> &volume, const glm::vec3 &origin, const glm::vec3 &direction, uint8_t layer, float step, const TransferFunctionType &transfer_function) {

  glm::vec4 dst(0.f, 0.f, 0.f, 1.f);

  const glm::vec3 size = {volume.info.layers[0].width, volume.info.layers[0].height, volume.info.layers[0].depth};
  const glm::vec3 size_in_blocks = {volume.info.layers[0].width_in_blocks, volume.info.layers[0].height_in_blocks, volume.info.layers[0].depth_in_blocks};

  ray_raster_traversal<TreeVolume<T>::SUBVOLUME_SIDE>(origin, direction, size, size_in_blocks, [&](const glm::vec<3, uint32_t> &block_pos, const glm::vec3 &in_block_pos, float tmax) {

    typename TreeVolume<T>::Node node = volume.nodes[volume.info.node_handle(block_pos.x, block_pos.y, block_pos.z, 0)];

    if (node.min == node.max) { // fast integration
      float a = m_transfer_a(node.min, node.max);

      if (a > 0.f) { // empty space skipping
        float r = m_transfer_r(node.min, node.max);
        float g = m_transfer_g(node.min, node.max);
        float b = m_transfer_b(node.min, node.max);

        float alpha = 1.f - std::exp(-a * tmax);

        float coef = alpha * dst.a;

        dst.r += r * coef;
        dst.g += g * coef;
        dst.b += b * coef;
        dst.a *= 1 - alpha;
      }
    }
    else { // integration by sampling
      float prev_value = volume.sample_block(node.block_handle, in_block_pos.x, in_block_pos.y, in_block_pos.z);

      float tmin = 0.f;

      while (tmin < tmax) {
        float next_step = std::min(m_stepsize, tmax - tmin);

        tmin += next_step;

        glm::vec3 current_pos = in_block_pos + direction * tmin;

        float value = volume.sample_block(node.block_handle, current_pos.x, current_pos.y, current_pos.z);

        float a = m_transfer_a(value, prev_value);

        if (a > 0.f) { // empty sample skipping
          float r = m_transfer_r(value, prev_value);
          float g = m_transfer_g(value, prev_value);
          float b = m_transfer_b(value, prev_value);

          float alpha = 1.f - std::exp(-a * next_step);

          float coef = alpha * dst.a;

          dst.r += r * coef;
          dst.g += g * coef;
          dst.b += b * coef;
          dst.a *= 1 - alpha;
        }

        prev_value = value;
      }
    }

    return dst.a > 1.f / 256.f;
  });

  return dst;
}
