/**
* @file preintegrate_function.h
* @author Drahomír Dlabaja (xdlaba02)
* @date 2. 5. 2022
* @copyright 2022 Drahomír Dlabaja
* @brief Function that integrates all possible intervals of a function on interval <0, 1> with specified resolution.
*/

#pragma once

#include "texture2D/texture2D.h"

#include <cstdint>

template <typename F>
Texture2D<float> preintegrate_function(uint32_t size, const F &func) {
  Texture2D<float> texture(size, size);

  for (uint32_t y = 0; y < size; y++) {
    float acc = func(float(y) / (size - 1));
    texture(y, y) = acc;
    for (uint32_t x = y + 1; x < size; x++) {
      acc += func(float(x) / (size - 1));
      texture(x, y) = texture(y, x) = acc / (x - y + 1);
    }
  }

  return texture;
}
