
#include "sample.h"
#include "timer.h"

#include <cstdint>

#include <vector>
#include <iostream>
#include <random>

template <size_t BLOCK_BITS, typename F>
float test_scalar(uint64_t n, const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(-.5f, (1 << BLOCK_BITS) - 1.5f);

  std::vector<uint8_t> block(1 << (BLOCK_BITS * 3));

  float val {};

  return measure_ns([&]{
    for (uint64_t i = 0; i < n; i++) {
      val += func(block.data(), 0, rd(re), rd(re), rd(re));
    }
  });
}

template <size_t BLOCK_BITS, typename F>
float test_simd(uint64_t n, const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(-.5f, (1 << BLOCK_BITS) - 1.5f);

  std::vector<uint8_t> block(1 << (BLOCK_BITS * 3));

  simd::float_v val {};

  simd::float_m dummy = simd::float_v(0.f) == simd::float_v(0.f);

  std::array<uint64_t, simd::len> indices {};

  return measure_ns([&]{
    for (uint64_t i = 0; i < n; i++) {
      simd::float_v x, y, z;

      for (uint8_t k = 0; k < simd::len; k++) {
        x = rd(re);
        y = rd(re);
        z = rd(re);
      }

      val += func(block.data(), indices, x, y, x, dummy);
    }
  });
}

int main(void) {
  uint64_t n = 1000000;

  std::cout << "SCALAR:\n";
  std::cout << "linear 3: " << test_scalar<3>(n, sample_linear_scalar<3>) / n << " ns\n";
  std::cout << "linear 4: " << test_scalar<4>(n, sample_linear_scalar<4>) / n << " ns\n";
  std::cout << "linear 5: " << test_scalar<5>(n, sample_linear_scalar<5>) / n << " ns\n";
  std::cout << "linear 6: " << test_scalar<6>(n, sample_linear_scalar<6>) / n << " ns\n";
  std::cout << "linear 7: " << test_scalar<7>(n, sample_linear_scalar<7>) / n << " ns\n";
  std::cout << "linear 8: " << test_scalar<8>(n, sample_linear_scalar<8>) / n << " ns\n";
  std::cout << "linear 9: " << test_scalar<9>(n, sample_linear_scalar<9>) / n << " ns\n";
  std::cout << "linear 10: " << test_scalar<10>(n, sample_linear_scalar<10>) / n << " ns\n";


  std::cout << "\n";

  std::cout << "SCALAR:\n";
  std::cout << "morton 3: " << test_scalar<3>(n, sample_morton_scalar<3>) / n << " ns\n";
  std::cout << "morton 4: " << test_scalar<4>(n, sample_morton_scalar<4>) / n << " ns\n";
  std::cout << "morton 5: " << test_scalar<5>(n, sample_morton_scalar<5>) / n << " ns\n";
  std::cout << "morton 6: " << test_scalar<6>(n, sample_morton_scalar<6>) / n << " ns\n";
  std::cout << "morton 7: " << test_scalar<7>(n, sample_morton_scalar<7>) / n << " ns\n";
  std::cout << "morton 8: " << test_scalar<8>(n, sample_morton_scalar<8>) / n << " ns\n";
  std::cout << "morton 9: " << test_scalar<9>(n, sample_morton_scalar<9>) / n << " ns\n";
  std::cout << "morton 10: " << test_scalar<10>(n, sample_morton_scalar<10>) / n << " ns\n";

  std::cout << "\n";

  std::cout << "SIMD:\n";
  std::cout << "linear 3: " << test_simd<3>(n, sample_linear_simd<3>) / (n * simd::len) << " ns\n";
  std::cout << "linear 4: " << test_simd<4>(n, sample_linear_simd<4>) / (n * simd::len) << " ns\n";
  std::cout << "linear 5: " << test_simd<5>(n, sample_linear_simd<5>) / (n * simd::len) << " ns\n";
  std::cout << "linear 6: " << test_simd<6>(n, sample_linear_simd<6>) / (n * simd::len) << " ns\n";
  std::cout << "linear 7: " << test_simd<7>(n, sample_linear_simd<7>) / (n * simd::len) << " ns\n";
  std::cout << "linear 8: " << test_simd<8>(n, sample_linear_simd<8>) / (n * simd::len) << " ns\n";
  std::cout << "linear 9: " << test_simd<9>(n, sample_linear_simd<9>) / (n * simd::len) << " ns\n";
  std::cout << "linear 10: " << test_simd<10>(n, sample_linear_simd<10>) / (n * simd::len) << " ns\n";

  std::cout << "\n";

  std::cout << "SIMD:\n";
  std::cout << "morton 3: " << test_simd<3>(n, sample_morton_simd<3>) / (n * simd::len) << " ns\n";
  std::cout << "morton 4: " << test_simd<4>(n, sample_morton_simd<4>) / (n * simd::len) << " ns\n";
  std::cout << "morton 5: " << test_simd<5>(n, sample_morton_simd<5>) / (n * simd::len) << " ns\n";
  std::cout << "morton 6: " << test_simd<6>(n, sample_morton_simd<6>) / (n * simd::len) << " ns\n";
  std::cout << "morton 7: " << test_simd<7>(n, sample_morton_simd<7>) / (n * simd::len) << " ns\n";
  std::cout << "morton 8: " << test_simd<8>(n, sample_morton_simd<8>) / (n * simd::len) << " ns\n";
  std::cout << "morton 9: " << test_simd<9>(n, sample_morton_simd<9>) / (n * simd::len) << " ns\n";
  std::cout << "morton 10: " << test_simd<10>(n, sample_morton_simd<10>) / (n * simd::len) << " ns\n";
}
