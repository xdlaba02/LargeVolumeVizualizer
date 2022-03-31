#include <fstream>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    return 1;
  }

  std::ifstream input(argv[1],  std::ios::in  | std::ios::binary);
  std::ofstream output(argv[2], std::ios::out | std::ios::binary);

  if (!input || !output) {
    return 2;
  }

  uint16_t val;

  while (input.read(reinterpret_cast<char*>(&val), sizeof(val))) {
    uint8_t reduced_val = val >> 8;
    output.write(reinterpret_cast<const char *>(&reduced_val), sizeof(reduced_val));
  }

  return 0;
}
