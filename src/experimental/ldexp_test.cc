#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

float my_ldexp(float num, int i) {
  reinterpret_cast<uint32_t &>(num) += num ? i << 23 : 0;
  return num;
}

constexpr float exp2(int i) {
  float num = 1.f;
  reinterpret_cast<uint32_t &>(num) += i << 23;
  return num;
}

constexpr float exp2v2(int32_t i) {
  uint32_t val = uint32_t(0x3f800000) + (i << 23);
  return reinterpret_cast<float &>(val);
}

constexpr float mult_ldexp(float num, int i) {
  return num * exp2(i);
}

template <typename F>
float test(uint64_t n, const F &func) {
  std::default_random_engine re(0);
  std::uniform_real_distribution<float> rd(-1024.f, 1024.f);

  auto start = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < n; i++) {
    for (int i = -64; i < 64; i++) {
      volatile float out = func(rd(re), i);
    }
  }

  return std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
}

int main(void) {

  float x = 0.125f;

  std::cerr << "ldexp test - x = " << x << "\n";

  for (int i = -128; i < 127; i++) {

    float std = std::ldexp(x, i);
    float my = my_ldexp(x, i);

    if (std != my) {
      std::cerr << i << " - std: " << std << " vs my:" << my << "\n";
    }
  }

  std::cerr << "exp2 test \n";

  for (int i = -128; i < 127; i++) {

    float std = std::exp2(i);
    float my = exp2(i);
    float myv2 = exp2v2(i);

    if (std != my) {
      std::cerr << i << " - std: " << std << " vs my:" << my << "\n";
    }

    if (std != myv2) {
      std::cerr << i << " - std: " << std << " vs myv2:" << myv2 << "\n";
    }
  }



  uint64_t n = 1000000;
  std::cout << "stl: " << test(n, [](float f, int i){ return std::ldexp(f, i);}) << " s\n";
  std::cout << "my: " << test(n, my_ldexp) << " s\n";
  std::cout << "mult: " << test(n, mult_ldexp) << " s\n";
  std::cout << "exp2: " << test(n, [](float f, int i){ return exp2(i);}) << " s\n";
  std::cout << "exp2v2: " << test(n, [](float f, int i){ return exp2v2(i);}) << " s\n";
  std::cout << "std exp2: " << test(n, [](float f, int i){ return std::exp2(i);}) << " s\n";

  return 0;
}
