/**
* @file linear_interpolation.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Generic template function that performs trilinear interpolation with 7 linear interpolations.
*/

#pragma once

#include <cstdint>

template <typename T>
void interpolate(T acc[2][2][2], const T &frac_x, const T &frac_y, const T &frac_z) {
  for (uint8_t z: {0, 1}) {
    for (uint8_t y: {0, 1}) {
      acc[z][y][0] += (acc[z][y][1] - acc[z][y][0]) * frac_x;
    }
    acc[z][0][0] += (acc[z][1][0] - acc[z][0][0]) * frac_y;
  }
  acc[0][0][0] += (acc[1][0][0] - acc[0][0][0]) * frac_z;
}
