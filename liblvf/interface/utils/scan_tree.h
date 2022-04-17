#pragma once

#include <tree_volume/tree_volume.h>

#include <utils/fast_exp2.h>

#include <glm/glm.hpp>

#include <cstdint>

#include <array>

template <typename T, uint32_t N, typename F>
void scan_tree(const typename TreeVolume<T, N>::Info &info, glm::vec3 cell, uint8_t layer, const F &func) {
  glm::vec3 node_pos = cell * exp2i(layer);

  uint8_t layer_index = std::size(info.layers) - 1 - layer;

  if (node_pos.x >= info.layers[layer_index].width_in_nodes
  || (node_pos.y >= info.layers[layer_index].height_in_nodes)
  || (node_pos.z >= info.layers[layer_index].depth_in_nodes)) {
    return;
  }

  if (func(cell, layer)) {
    float child_size = exp2i(-layer - 1);

    glm::vec3 center = cell + child_size;

    for (uint8_t z = 0; z < 2; z++) {
      for (uint8_t y = 0; y < 2; y++) {
        for (uint8_t x = 0; x < 2; x++) {
          scan_tree<T, N>(info, cell, layer + 1, func);
          std::swap(cell.x, center.x);
        }
        std::swap(cell.y, center.y);
      }
      std::swap(cell.z, center.z);
    }
  }
}
