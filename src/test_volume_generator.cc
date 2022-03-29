
#include <tree_volume/tree_volume.h>

#include <utils/mapped_file.h>
#include <utils/morton.h>
#include <utils/endian.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

int main(int argc, char *argv[]) {
  uint32_t width;
  uint32_t height;
  uint32_t depth;

  {
    std::stringstream wstream(argv[1]);
    std::stringstream hstream(argv[2]);
    std::stringstream dstream(argv[3]);
    wstream >> width;
    hstream >> height;
    dstream >> depth;
  }

  TreeVolume<uint8_t>::Info info(width, height, depth);

  {
    std::ofstream metadata(argv[4], std::ios::binary);
    if (!metadata) {
      throw std::runtime_error(std::string("Unable to open '") + argv[6] + "'!");
    }

    metadata.seekp(info.size_in_blocks * sizeof(TreeVolume<uint8_t>::Node) - 1);
    metadata.write("", 1);
  }

  MappedFile metadata(argv[4], 0, info.size_in_blocks * sizeof(TreeVolume<uint8_t>::Node), MappedFile::WRITE, MappedFile::SHARED);
  if (!metadata) {
    throw std::runtime_error(std::string("Unable to map '") + argv[6] + "'!");
  }

  TreeVolume<uint8_t>::Node *nodes = reinterpret_cast<TreeVolume<uint8_t>::Node *>(metadata.data());

  std::ofstream tree_volume(argv[5], std::ofstream::binary);
  if (!tree_volume) {
    throw std::runtime_error(std::string("Unable to open '") + argv[5] + "'!");
  }


  uint64_t block_handle = 0;

  for (uint8_t layer_index = 0; layer_index < std::size(info.layers); layer_index++) {
    uint8_t layer = std::size(info.layers) - layer_index - 1;
    
    std::cerr << "layer_index: " << (int)layer_index << "\n";
    std::cerr << info.layers[layer_index].width_in_blocks << " " << info.layers[layer_index].height_in_blocks << " " << info.layers[layer_index].depth_in_blocks << "\n";
    #pragma omp parallel for
    for (uint32_t block_z = 0; block_z < info.layers[layer_index].depth_in_blocks; block_z++) {
      for (uint32_t block_y = 0; block_y < info.layers[layer_index].height_in_blocks; block_y++) {
        for (uint32_t block_x = 0; block_x < info.layers[layer_index].width_in_blocks; block_x++) {

          TreeVolume<uint8_t>::Node &node = nodes[info.node_handle(block_x, block_y, block_z, layer_index)];

          node.block_handle = 0;
          node.min    = 255;
          node.max    = 0;

          TreeVolume<uint8_t>::Block block {};


          uint32_t block_val = layer * 2 + ((block_x & 1) ^ (block_y & 1) ^ (block_z & 1));

          for (uint32_t i = 0; i < TreeVolume<uint8_t>::BLOCK_SIZE; i++) {
            block[i] = block_val;
          }

          #pragma omp critical
          {
            node.block_handle = block_handle++;
            tree_volume.write((const char *)block, sizeof(block));
          }
        }
      }
    }
  }
}
