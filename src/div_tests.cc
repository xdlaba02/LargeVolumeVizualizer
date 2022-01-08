#include <iostream>
#include <limits>
#include <stdint.h>

uint32_t div_by_15_44(uint32_t n) {
  return (n * 0x23) >> 9;
}

uint32_t div_by_15_103(uint32_t n) {
  return (n * 0x45) >> 10;
}

uint32_t div_by_15_298(uint32_t n) {
  return (n * 0x89) >> 11;
}

uint32_t div_by_15_643(uint32_t n) {
  return (n * 0x223) >> 13;
}

uint32_t div_by_15_1498(uint32_t n) {
  return (n * 0x445) >> 14;
}

uint32_t div_by_15_4693(uint32_t n) {
  return (n * 0x889) >> 15;
}

uint32_t div_by_15_74908(uint32_t n) {
  return (n * 0x8889) >> 19;
}

uint32_t div_by_15_fullprecision(uint32_t n) {
  uint32_t q = (n >> 4) + (n >> 8);
  q += (q >> 8);
  q += (q >> 16);
  //uint32_t err = n - q * 15;
  uint32_t err = n - ((q << 4) - q);
  return q + ((err * 0x45) >> 10);
}

void division_test() {
  for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); i++) {
    if (div_by_15_fullprecision(i) != (i / 15)) {
      std::cout << i << " / 15 = " << div_by_15_fullprecision(i) << " vs in reality it is " << i / 15 << "\n";
      return;
    }
  }
  std::cout << "end\n";
}
