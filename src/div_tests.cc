#include "fast_div.h"

#include <iostream>
#include <cstdint>

int main() {
  using TestedType = uint32_t;
  constexpr TestedType D = 7;

  for (uint64_t i = 0; i < std::numeric_limits<uint32_t>::max(); i++) {
    if (fast_div<D, uint32_t>(i) != (i / D)) {
      std::cout << "Failed at " << i << "\n";
      break;
    }

    i++;
  }

  return 0;
}
