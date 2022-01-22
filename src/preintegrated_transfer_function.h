#pragma once

#include "simd.h"
#include "sampler_2D.h"
#include "fast_div.h"

#include <cstdint>

template <typename T>
class PreintegratedTransferFunction {
public:
  template <typename F>
  PreintegratedTransferFunction(const F &func) {
    for (uint32_t y = 0; y < N + 1; y++) {
      auto acc = func(y);
      m_data[y][y] = acc;
      for (uint32_t x = y + 1; x < N + 1; x++) {
        acc += func(x);
        m_data[x][y] = m_data[y][x] = acc / (x - y + 1);
      }
    }
  }

  simd::float_v operator()(simd::float_v v0, simd::float_v v1, simd::float_m mask) {
    return sampler2D<S>(reinterpret_cast<float *>(m_data), v0 / sizeof(T), v1 / sizeof(T), mask);
  }

  float operator()(float v0, float v1) {
    return sampler2D<S>(reinterpret_cast<float *>(m_data), v0 / sizeof(T), v1 / sizeof(T));
  }
private:
  static constexpr uint32_t N = 256;
  static constexpr uint32_t S = N + 1;
  float m_data[S][S];
};
