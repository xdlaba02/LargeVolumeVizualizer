
#include "raw_volume.h"
#include "blocked_volume.h"
#include "morton.h"
#include "endian.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

int main(int argc, char *argv[]) {
  if (argc != 7) {
    std::cerr << "args\n";
    return 1;
  }

  uint32_t width;
  uint32_t height;
  uint32_t depth;

  {
    std::stringstream wstream(argv[2]);
    std::stringstream hstream(argv[3]);
    std::stringstream dstream(argv[4]);
    wstream >> width;
    hstream >> height;
    dstream >> depth;
  }

  RawVolume<uint8_t> volume(argv[1], width, height, depth);
  if (!volume) {
    std::cerr << "couldn't open volume\n";
    return 1;
  }

  BlockedVolume<uint8_t>::Info info(width, height, depth);

  std::vector<uint8_t> mins(info.size_in_blocks);
  std::vector<uint8_t> maxs(info.size_in_blocks);
  std::vector<uint64_t> offsets(info.size_in_blocks);

  std::ofstream blocked_volume(argv[5], std::ofstream::binary);
  if (!blocked_volume) {
    std::cerr << "failed creating the blocked volume \n";
    return 1;
  }

  uint64_t offset = 0;

  for (uint32_t block_z = 0; block_z < info.depth_in_blocks; block_z++) {
    for (uint32_t block_y = 0; block_y < info.height_in_blocks; block_y++) {
      for (uint32_t block_x = 0; block_x < info.width_in_blocks; block_x++) {

        uint64_t block_index = block_z * info.stride_in_blocks + block_y * info.width_in_blocks + block_x;

        uint8_t &min = mins[block_index];
        uint8_t &max = maxs[block_index];

        min = 255;
        max = 0;

        BlockedVolume<uint8_t>::Block block {};
        for (uint32_t i = 0; i < BlockedVolume<uint8_t>::BLOCK_SIZE; i++) {
          uint32_t x, y, z;
          morton::from_index_4b_3d(i, x, y, z);

          uint32_t original_x = std::min(block_x * BlockedVolume<uint8_t>::SUBVOLUME_SIDE + x, width - 1);
          uint32_t original_y = std::min(block_y * BlockedVolume<uint8_t>::SUBVOLUME_SIDE + y, height - 1);
          uint32_t original_z = std::min(block_z * BlockedVolume<uint8_t>::SUBVOLUME_SIDE + z, depth - 1);

          block[i] = volume(original_x, original_y, original_z);

          min = std::min<uint8_t>(min, block[i]);
          max = std::max<uint8_t>(max, block[i]);

        }

        if (min != max) {
          offsets[block_index] = offset++;
          blocked_volume.write((const char *)block, sizeof(block));
        }
      }
    }
  }

  /*

  uint32_t layer_width  = width_in_blocks;
  uint32_t layer_height = height_in_blocks;
  uint32_t layer_depth  = depth_in_blocks;

  while (layer_width > 1 || layer_height > 1 || layer_depth > 1) {
    layer_width  = (layer_width  + 1) >> 1;
    layer_height = (layer_height + 1) >> 1;
    layer_depth  = (layer_depth  + 1) >> 1;

    nodes.emplace_back();

    for (uint32_t block_z = 0; block_z < layer_depth; block_z++) {
      for (uint32_t block_y = 0; block_y < layer_height; block_y++) {
        for (uint32_t block_x = 0; block_x < layer_width; block_x++) {

          Node &node = nodes.back().emplace_back();

          node.min = 255;
          node.max = 0;

          uint32_t block_index = 0;
          Block<uint8_t> block {};

          for (uint32_t source_z = block_z << 1; source_z < std::min(layer_depth - 1, (block_z << 1) + 1); source_z++) {
            for (uint32_t source_y = block_y << 1; source_y < std::min(layer_height - 1, (block_y << 1) + 1); source_y++) {
              for (uint32_t source_x = block_x << 1; source_x < std::min(layer_width - 1, (block_x << 1) + 1); source_x++) {

                const Node &source_node = nodes[nodes.size() - 2][source_z * layer_depth * layer_width + source_y * layer_width + source_x];

                node.min = std::min(node.min, source_node.min);
                node.max = std::max(node.max, source_node.max);

                if (source_node.min != source_node.max) {
                  const Block<uint8_t> &source_block = blocks[source_node.block_offset];

                  for (uint32_t i = 0; i < std::size(block) >> 3; i++ ) {
                    uint16_t value {};

                    for (uint8_t j = 0; j < 8; j++) {
                      value += source_block[(i << 3) + j];
                    }

                    block[block_index++] = value >> 3;
                  }
                }
                else {
                  for (uint32_t i = 0; i < std::size(block) >> 3; i++ ) {
                    block[block_index++] = source_node.min;
                  }
                }
              }
            }
          }

          if (node.min != node.max) {
            node.block_offset = blocks.size();
            blocks.push_back(block);
          }
        }
      }
    }
  }

  */

  std::ofstream metadata(argv[6]);
  if (!metadata) {
    std::cerr << "failed creating the metadata file\n";
    return 1;
  }

  metadata.write(reinterpret_cast<const char *>(mins.data()), info.size_in_blocks);
  metadata.write(reinterpret_cast<const char *>(maxs.data()), info.size_in_blocks);
  metadata.write(reinterpret_cast<const char *>(offsets.data()), info.size_in_blocks * sizeof(uint64_t));
}