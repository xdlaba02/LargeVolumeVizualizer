#pragma once

#include <chrono>

template <typename F>
double measure_ns(const F &func) {
  auto start = std::chrono::steady_clock::now();
  func();
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
}

template <typename F>
double measure_ms(const F &func) {
  auto start = std::chrono::steady_clock::now();
  func();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
