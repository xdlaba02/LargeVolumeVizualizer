
#include "sample.h"
#include "timer.h"

#include <cstdint>

#include <vector>
#include <iostream>
#include <random>

static const size_t n = 1000000;

template <size_t BLOCK_BITS, typename F>
float test_scalar(const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(-.5f, (1 << BLOCK_BITS) - 1.5f);

  std::vector<uint8_t> block(1 << (BLOCK_BITS * 3));

  static float dummy __attribute__((used)) {};

  return measure_ns([&]{
    for (uint64_t i = 0; i < n; i++) {
      dummy += func(block.data(), 0, rd(re), rd(re), rd(re));
    }
  });
}

template <size_t BLOCK_BITS, typename F>
float test_simd(const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(-.5f, (1 << BLOCK_BITS) - 1.5f);

  std::vector<uint8_t> block(1 << (BLOCK_BITS * 3));

  static simd::float_v dummy __attribute__((used)) {};

  std::array<uint64_t, simd::len> indices {};

  return measure_ns([&]{
    for (uint64_t i = 0; i < n; i++) {
      simd::float_v x, y, z;

      for (uint8_t k = 0; k < simd::len; k++) {
        x = rd(re);
        y = rd(re);
        z = rd(re);
      }

      dummy += func(block.data(), indices, x, y, x, simd::float_m(true));
    }
  });
}

template <size_t N>
void test() {
  std::cout << "linear scalar " << N << ":\t" << test_scalar<N>(sample_linear_scalar<N>) / n << " ns\n";
  std::cout << "morton scalar " << N << ":\t" << test_scalar<N>(sample_morton_scalar<N>) / n << " ns\n";
  std::cout << "linear simd   " << N << ":\t" << test_simd<N>(sample_linear_simd<N>) / (n * simd::len) << " ns\n";
  std::cout << "morton simd   " << N << ":\t" << test_simd<N>(sample_morton_simd<N>) / (n * simd::len) << " ns\n";
}

int main(void) {
  test<3>();
  test<4>();
  test<5>();
  test<6>();
  test<7>();
  test<8>();
  test<9>();
  test<10>();
}
