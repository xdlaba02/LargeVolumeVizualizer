#include <cstdint>

#include <array>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>

std::array<uint8_t, 3> sort_stl(const std::array<float, 3> &in) {
  std::array<uint8_t, 3> out {};

  std::iota(std::begin(out), std::end(out), 0);
  std::sort(std::begin(out), std::end(out), [&](uint8_t l, uint8_t r) { return in[l] < in[r]; });

  return out;
}

std::array<uint8_t, 3> sort_order(const std::array<float, 3> &in) {
  std::array<uint8_t, 3> out {};

  uint8_t order = ((in[0] < in[1]) << 2) | ((in[0] < in[2]) << 1) | (in[1] < in[2]);

  out[0] = std::array<uint8_t, 8>{ 2, 1, 0, 1, 2, 0, 0, 0 }[order];
  out[1] = std::array<uint8_t, 8>{ 1, 2, 1, 0, 0, 1, 2, 1 }[order];
  out[2] = std::array<uint8_t, 8>{ 0, 0, 2, 2, 1, 2, 1, 2 }[order];

  return out;
}

std::array<uint8_t, 3> sort_gt(const std::array<float, 3> &in) {
  std::array<uint8_t, 3> out {};

  uint8_t gt01 = in[0] > in[1];
  uint8_t gt02 = in[0] > in[2];
  uint8_t gt12 = in[1] > in[2];

  out[ gt01 +  gt02] = 0;
  out[!gt01 +  gt12] = 1;
  out[!gt02 + !gt12] = 2;

  return out;
}

template <typename F>
float test(uint64_t n, const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(0.f, 100.f);

  std::array<uint8_t, 3> order;

  auto start = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < n; i++) {
    order = func({ rd(re), rd(re), rd(re) });
  }

  return std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
}

int main(void) {
  uint64_t n = 100000000;
  std::cout << "stl: " << test(n, sort_stl) << " s\n";
  std::cout << "order: " << test(n, sort_order) << " s\n";
  std::cout << "gt: " << test(n, sort_gt) << " s\n";
}
