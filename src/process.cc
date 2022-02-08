
#include "raw_volume.h"
#include "blocked_volume.h"
#include "mapped_file.h"
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

  std::vector<uint64_t> layer_sizes {};

  {
      uint32_t layer_width  = info.width_in_blocks;
      uint32_t layer_height = info.height_in_blocks;
      uint32_t layer_depth  = info.depth_in_blocks;

      while (layer_width > 1 || layer_height > 1 || layer_depth > 1) {

        layer_sizes.push_back(layer_width * layer_height * layer_depth);

        ++layer_width  >>= 1;
        ++layer_height >>= 1;
        ++layer_depth  >>= 1;
      }

      layer_sizes.push_back(layer_width * layer_height * layer_depth);
  }

  std::vector<uint64_t> layer_offsets {};

  uint64_t pyramid_size_in_blocks = 0;

  for (uint64_t layer_size: layer_sizes) {
    layer_offsets.push_back(pyramid_size_in_blocks);
    pyramid_size_in_blocks += layer_size;
  }

  {
    std::ofstream metadata(argv[6], std::ios::binary);
    if (!metadata) {
      std::cerr << "failed creating the metadata file\n";
      return 1;
    }

    metadata.seekp(pyramid_size_in_blocks * sizeof(BlockedVolume<uint8_t>::Node) - 1);
    metadata.write("", 1);
  }

  MappedFile metadata(argv[6], 0, pyramid_size_in_blocks * sizeof(BlockedVolume<uint8_t>::Node), MappedFile::WRITE, MappedFile::SHARED);
  if (!metadata) {
    std::cerr << "failed mapping the metadata file\n";
    return 1;
  }

  BlockedVolume<uint8_t>::Node *nodes = reinterpret_cast<BlockedVolume<uint8_t>::Node *>(metadata.data());

  std::ofstream blocked_volume(argv[5], std::ofstream::binary);
  if (!blocked_volume) {
    std::cerr << "failed creating the blocked volume \n";
    return 1;
  }

  uint8_t layer = 0;

  uint64_t layer_width  = info.width_in_blocks;
  uint64_t layer_height = info.height_in_blocks;
  uint64_t layer_depth  = info.depth_in_blocks;

  uint64_t block_handle = 0;

  for (uint32_t block_z = 0; block_z < layer_depth; block_z++) {
    for (uint32_t block_y = 0; block_y < layer_height; block_y++) {
      for (uint32_t block_x = 0; block_x < layer_width; block_x++) {

        uint64_t node_index = block_z * layer_width * layer_height + block_y * layer_width + block_x;

        BlockedVolume<uint8_t>::Node &node = nodes[layer_offsets[layer] + node_index];

        node.block_handle = 0;
        node.min    = 255;
        node.max    = 0;

        BlockedVolume<uint8_t>::Block block {};

        for (uint32_t i = 0; i < BlockedVolume<uint8_t>::BLOCK_SIZE; i++) {
          uint32_t x, y, z;
          morton::from_index_4b_3d(i, x, y, z);

          uint32_t original_x = std::min(block_x * BlockedVolume<uint8_t>::SUBVOLUME_SIDE + x, width - 1);
          uint32_t original_y = std::min(block_y * BlockedVolume<uint8_t>::SUBVOLUME_SIDE + y, height - 1);
          uint32_t original_z = std::min(block_z * BlockedVolume<uint8_t>::SUBVOLUME_SIDE + z, depth - 1);

          block[i] = volume(original_x, original_y, original_z);

          node.min = std::min<uint8_t>(node.min, block[i]);
          node.max = std::max<uint8_t>(node.max, block[i]);
        }

        if (node.min != node.max) {
          node.block_handle = block_handle++;
          blocked_volume.write((const char *)block, sizeof(block));
        }
      }
    }
  }

  layer++;

  while (layer < std::size(layer_offsets)) {
    uint64_t next_layer_width  = (layer_width  + 1) >> 1;
    uint64_t next_layer_height = (layer_height + 1) >> 1;
    uint64_t next_layer_depth  = (layer_depth  + 1) >> 1;

    blocked_volume.flush();

    MappedFile blocks_file(argv[5], 0, blocked_volume.tellp(), MappedFile::READ, MappedFile::SHARED);

    if (!blocks_file) {
      std::cerr << "failed to map processed blocks\n";
      return 1;
    }

    const BlockedVolume<uint8_t>::Block *blocks = reinterpret_cast<const BlockedVolume<uint8_t>::Block *>(blocks_file.data());

    for (uint32_t block_z = 0; block_z < next_layer_depth; block_z++) {
      for (uint32_t block_y = 0; block_y < next_layer_height; block_y++) {
        for (uint32_t block_x = 0; block_x < next_layer_width; block_x++) {

          uint64_t node_index = block_z * next_layer_width * next_layer_height + block_y * next_layer_width + block_x;

          BlockedVolume<uint8_t>::Node &node = nodes[layer_offsets[layer] + node_index];

          node.block_handle = 0;
          node.min    = 255;
          node.max    = 0;

          BlockedVolume<uint8_t>::Block block {};

          uint32_t voxel_index = 0;

          for (uint32_t source_z = block_z << 1; source_z < std::min<uint32_t>((block_z << 1) + 1, layer_depth); source_z++) {
            for (uint32_t source_y = block_y << 1; source_y < std::min<uint32_t>((block_y << 1) + 1, layer_height); source_y++) {
              for (uint32_t source_x = block_x << 1; source_x < std::min<uint32_t>((block_x << 1) + 1, layer_width); source_x++) {

                uint64_t source_node_index = source_z * layer_width * layer_height + source_y * layer_width + source_x;

                const BlockedVolume<uint8_t>::Node &source_node = nodes[layer_offsets[layer - 1] + source_node_index];

                if (source_node.min != source_node.max) {
                  const BlockedVolume<uint8_t>::Block &source_block = blocks[source_node.block_handle];

                  for (uint32_t i = 0; i < BlockedVolume<uint8_t>::BLOCK_SIZE >> 3; i++) {
                    uint32_t value {};

                    for (uint8_t j = 0; j < 8; j++) {
                      value += source_block[(i << 3) + j];
                    }

                    block[voxel_index++] = value >> 3;
                  }
                }
                else {
                  for (uint32_t i = 0; i < BlockedVolume<uint8_t>::BLOCK_SIZE >> 3; i++) {
                    block[voxel_index++] = source_node.min;
                  }
                }

                node.min = std::min(node.min, source_node.min);
                node.max = std::max(node.max, source_node.max);
              }
            }
          }

          if (node.min != node.max) {
            node.block_handle = block_handle++;
            blocked_volume.write((const char *)block, sizeof(block));
          }
        }
      }
    }

    layer_width  = next_layer_width;
    layer_height = next_layer_height;
    layer_depth  = next_layer_depth;

    layer++;
  }
}
