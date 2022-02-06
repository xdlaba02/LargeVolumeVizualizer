#pragma once

#include "intersection.h"

#include <glm/glm.hpp>

#include <cstdint>

template <uint32_t CELL_SIZE, typename F>
inline void raster_traversal(const glm::vec3 &origin, const glm::vec3 &direction, const glm::vec<3, uint32_t> &size, const glm::vec<3, uint32_t> &size_in_cells, const F& callback) {

  glm::vec3 t_delta;
  glm::vec3 t_next_crossing;

  glm::vec<3, uint32_t> cell;
  glm::vec<3, int8_t> step;
  glm::vec<3, uint32_t> stop;

  float t, t_end;

  {
    glm::vec3 direction_inverse = 1.f / direction;

    intersect_aabb_ray(origin, direction_inverse, {0, 0, 0}, size, t, t_end);

    if (t >= t_end) {
      return;
    }

    glm::vec3 denorm = origin + direction * t - .5f;
    glm::vec<3, uint32_t> vox = glm::vec<3, uint32_t>(denorm);

    for (uint8_t i = 0; i < 3; i++) {
      cell[i] = vox[i] / CELL_SIZE;

      t_next_crossing[i] = t + (cell[i] * CELL_SIZE - denorm[i]) * direction_inverse[i];

      if (direction[i] < 0.f) {
        step[i] = -1;
        stop[i] = 0;
        t_delta[i] = -float(CELL_SIZE) * direction_inverse[i];
      }
      else {
        step[i] = 1;
        stop[i] = size_in_cells[i] - 1;
        t_delta[i] = float(CELL_SIZE) * direction_inverse[i];
        t_next_crossing[i] += t_delta[i];
      }
    }
  }

  auto nearest_axis = [&]() {
    uint8_t k = ((t_next_crossing[0] < t_next_crossing[1]) << 2)
              | ((t_next_crossing[0] < t_next_crossing[2]) << 1)
              | ((t_next_crossing[1] < t_next_crossing[2]) << 0);

    constexpr uint8_t arr[8] { 2, 1, 2, 1, 2, 2, 0, 0 };

    return arr[k];
  };

  uint8_t axis = nearest_axis();

  while (cell[axis] != stop[axis]) {

    if (!callback(cell, origin + direction * t - glm::vec3(cell * CELL_SIZE) - .5f, t_next_crossing[axis] - t)) {
      return;
    }

    t = t_next_crossing[axis];

    t_next_crossing[axis] += t_delta[axis];

    cell[axis] += step[axis];

    axis = nearest_axis();
  }

  callback(cell, origin + direction * t - glm::vec3(cell * CELL_SIZE) - .5f, t_end - t);
}
