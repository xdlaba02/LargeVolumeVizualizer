
#include <raw_volume/raw_volume.h>

#include <tree_volume/processor.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

void parse_args(int argc, const char *argv[], const char *&raw_volume, const char *&processed_volume, const char *&processed_metadata, uint32_t &width, uint32_t &height, uint32_t &depth) {
  if (argc != 7) {
    throw std::runtime_error("Wrong number of arguments!");
  }

  raw_volume         = argv[1];
  processed_volume   = argv[5];
  processed_metadata = argv[6];

  {
    std::stringstream arg {};
    arg << argv[2];
    arg >> width;
    if (!arg) {
      throw std::runtime_error("Unable to parse width!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[3];
    arg >> height;
    if (!arg) {
      throw std::runtime_error("Unable to parse height!");
    }
  }

  {
    std::stringstream arg {};
    arg << argv[4];
    arg >> depth;
    if (!arg) {
      throw std::runtime_error("Unable to parse depth!");
    }
  }
}

int main(int argc, const char *argv[]) {
  try {
    uint32_t width;
    uint32_t height;
    uint32_t depth;

    const char *raw_volume_file_name;
    const char *processed_volume_file_name;
    const char *metadata_file_name;

    parse_args(argc, argv, raw_volume_file_name, processed_volume_file_name, metadata_file_name, width, height, depth);

    RawVolume<uint8_t> volume(raw_volume_file_name, width, height, depth);

    process_volume<uint8_t>(width, height, depth, processed_volume_file_name, metadata_file_name, [&](uint32_t x, uint32_t y, uint32_t z) {
      return volume.data[volume.voxel_handle(x, y, z)];
    });
  }
  catch (const std::runtime_error& e) {
    std::cerr << e.what() << "\n";
    std::cerr << "Usage: \n";
    std::cerr << argv[0] << " <raw-volume> <width> <height> <depth> <processed-volume> <processed-metadata>\n";
  }
}
