#include <utils/mapped_file.h>

#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <filesystem>

int main(int argc, char *argv[]) {
  if (argc != 6) {
    return 1;
  }

  float min, max;
  size_t width, height;

  std::stringstream args {};
  args << argv[2] << " " << argv[3] << " " << argv[4] << " " << argv[5];
  args >> min >> max >> width >> height;

  size_t size = std::filesystem::file_size(argv[1]);

  MappedFile input(argv[1], 0, size, MappedFile::READ, MappedFile::SHARED);

  if (!input) {
    return 2;
  }

  const uint8_t *input_data = reinterpret_cast<const uint8_t *>(input.data());

  std::vector<size_t> hist(height);

  #pragma omp parallel
  {
    std::vector<size_t> local_hist(height);

    #pragma omp for nowait
    for (size_t i = 0; i < size; i++) {
      float norm_val = input_data[i] / 255.f;

      if (norm_val >= min && norm_val <= max) {
        local_hist[(norm_val - min) * height / (max - min)]++;
      }
    }

    #pragma omp critical
    {
      for (size_t i = 0; i < height; i++) {
        hist[i] += local_hist[i];
      }
    }
  }

  size_t maxv = 0;
  for (size_t i = 0; i < height; i++) {
    maxv = std::max(maxv, hist[i]);
  }

  for (size_t i = 0; i < height; i++) {
    std::cout << std::fixed << std::setprecision(6) << std::setw(8) << i * (max - min) / height + min << " ";
    if (maxv) {
      for (size_t j = 0; j < hist[i] * width / maxv; j++) {
        std::cout << "#";
      }
    }
    std::cout << "\n";
  }

  return 0;
}
