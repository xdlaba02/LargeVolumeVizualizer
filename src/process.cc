
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

  {
    std::ofstream metadata(argv[6], std::ios::binary);
    if (!metadata) {
      std::cerr << "failed creating the metadata file\n";
      return 1;
    }

    metadata.seekp(info.size_in_blocks * sizeof(BlockedVolume<uint8_t>::Node) - 1);
    metadata.write("", 1);
  }

  MappedFile metadata(argv[6], 0, info.size_in_blocks * sizeof(BlockedVolume<uint8_t>::Node), MappedFile::WRITE, MappedFile::SHARED);
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

  uint64_t block_handle = 0;

  #pragma omp parallel for
  for (uint32_t block_z = 0; block_z < info.layers[0].depth_in_blocks; block_z++) {
    for (uint32_t block_y = 0; block_y < info.layers[0].height_in_blocks; block_y++) {
      for (uint32_t block_x = 0; block_x < info.layers[0].width_in_blocks; block_x++) {

        BlockedVolume<uint8_t>::Node &node = nodes[info.node_handle(block_x, block_y, block_z, 0)];

        node.block_handle = 0;
        node.min    = 255;
        node.max    = 0;

        BlockedVolume<uint8_t>::Block block {};

        for (uint32_t i = 0; i < BlockedVolume<uint8_t>::BLOCK_SIZE; i++) {
          uint32_t x, y, z;
          Morton<BlockedVolume<uint8_t>::BLOCK_BITS>::from_index(i, x, y, z);

          uint32_t original_x = std::min(block_x * BlockedVolume<uint8_t>::SUBVOLUME_SIDE + x, width - 1);
          uint32_t original_y = std::min(block_y * BlockedVolume<uint8_t>::SUBVOLUME_SIDE + y, height - 1);
          uint32_t original_z = std::min(block_z * BlockedVolume<uint8_t>::SUBVOLUME_SIDE + z, depth - 1);

          block[i] = volume(original_x, original_y, original_z);

          node.min = std::min<uint8_t>(node.min, block[i]);
          node.max = std::max<uint8_t>(node.max, block[i]);
        }

        if (node.min != node.max) {
          #pragma omp critical
          {
            node.block_handle = block_handle++;
            blocked_volume.write((const char *)block, sizeof(block));
          }
        }
      }
    }
  }

  for (uint8_t layer = 1; layer < std::size(info.layers); layer++) {
    blocked_volume.flush();

    MappedFile blocks_file(argv[5], 0, blocked_volume.tellp(), MappedFile::READ, MappedFile::SHARED);

    if (!blocks_file) {
      std::cerr << "failed to map processed blocks\n";
      return 1;
    }

    const BlockedVolume<uint8_t>::Block *blocks = reinterpret_cast<const BlockedVolume<uint8_t>::Block *>(blocks_file.data());

    #pragma omp parallel for
    for (uint32_t block_z = 0; block_z < info.layers[layer].depth_in_blocks; block_z++) {

      uint32_t block_start_z = block_z * BlockedVolume<uint8_t>::SUBVOLUME_SIDE;

      for (uint32_t block_y = 0; block_y < info.layers[layer].height_in_blocks; block_y++) {

        uint32_t block_start_y = block_y * BlockedVolume<uint8_t>::SUBVOLUME_SIDE;

        for (uint32_t block_x = 0; block_x < info.layers[layer].width_in_blocks; block_x++) {

          uint32_t block_start_x = block_x * BlockedVolume<uint8_t>::SUBVOLUME_SIDE;

          BlockedVolume<uint8_t>::Node &node = nodes[info.node_handle(block_x, block_y, block_z, layer)];

          node.block_handle = 0;
          node.min    = 255;
          node.max    = 0;

          BlockedVolume<uint8_t>::Block block {};

          for (uint32_t in_block_z = 0; in_block_z < BlockedVolume<uint8_t>::BLOCK_SIDE; in_block_z++) {

            uint32_t voxel_z = block_start_z + in_block_z;
            uint32_t original_voxel_z = voxel_z << 1;

            for (uint32_t in_block_y = 0; in_block_y < BlockedVolume<uint8_t>::BLOCK_SIDE; in_block_y++) {

              uint32_t voxel_y = block_start_y + in_block_y;
              uint32_t original_voxel_y = voxel_y << 1;

              for (uint32_t in_block_x = 0; in_block_x < BlockedVolume<uint8_t>::BLOCK_SIDE; in_block_x++) {

                uint32_t voxel_x = block_start_x + in_block_x;
                uint32_t original_voxel_x = voxel_x << 1;

                uint32_t accumulator = 0;

                for (uint32_t original_z: {original_voxel_z, original_voxel_z + 1}) {

                  uint32_t original_block_z = std::min(original_z / BlockedVolume<uint8_t>::SUBVOLUME_SIDE, info.layers[layer - 1].depth_in_blocks  - 1);
                  uint32_t original_in_block_z = std::min(original_z - original_block_z * BlockedVolume<uint8_t>::SUBVOLUME_SIDE, BlockedVolume<uint8_t>::SUBVOLUME_SIDE - 1);

                  for (uint32_t original_y: {original_voxel_y, original_voxel_y + 1}) {

                    uint32_t original_block_y = std::min(original_y / BlockedVolume<uint8_t>::SUBVOLUME_SIDE, info.layers[layer - 1].height_in_blocks - 1);
                    uint32_t original_in_block_y = std::min(original_y - original_block_y * BlockedVolume<uint8_t>::SUBVOLUME_SIDE, BlockedVolume<uint8_t>::SUBVOLUME_SIDE - 1);

                    for (uint32_t original_x: {original_voxel_x, original_voxel_x + 1}) {

                      uint32_t original_block_x = std::min(original_x / BlockedVolume<uint8_t>::SUBVOLUME_SIDE, info.layers[layer - 1].width_in_blocks  - 1);
                      uint32_t original_in_block_x = std::min(original_x - original_block_x * BlockedVolume<uint8_t>::SUBVOLUME_SIDE, BlockedVolume<uint8_t>::SUBVOLUME_SIDE - 1);

                      BlockedVolume<uint8_t>::Node &original_node = nodes[info.node_handle(original_block_x, original_block_y, original_block_z, layer - 1)];

                      if (original_node.min != original_node.max) {
                        accumulator += blocks[original_node.block_handle][Morton<BlockedVolume<uint8_t>::BLOCK_BITS>::to_index(original_in_block_x, original_in_block_y, original_in_block_z)];
                      }
                      else {
                        accumulator += original_node.min;
                      }
                    }
                  }
                }

                uint32_t value = accumulator >> 3;

                node.min = std::min<uint32_t>(node.min, value);
                node.max = std::max<uint32_t>(node.max, value);

                block[Morton<BlockedVolume<uint8_t>::BLOCK_BITS>::to_index(in_block_x, in_block_y, in_block_z)] = value;
              }
            }
          }

          for (uint32_t original_block_z_unaligned = (block_z << 1); original_block_z_unaligned < (block_z << 1) + 2; original_block_z_unaligned++) {

            uint32_t original_block_z = std::min(original_block_z_unaligned, info.layers[layer - 1].depth_in_blocks  - 1);

            for (uint32_t original_block_y_unaligned = (block_y << 1); original_block_y_unaligned < (block_y << 1) + 2; original_block_y_unaligned++) {

              uint32_t original_block_y = std::min(original_block_y_unaligned, info.layers[layer - 1].height_in_blocks - 1);

              for (uint32_t original_block_x_unaligned = (block_x << 1); original_block_x_unaligned < (block_x << 1) + 2; original_block_x_unaligned++) {

                uint32_t original_block_x = std::min(original_block_x_unaligned, info.layers[layer - 1].width_in_blocks  - 1);

                BlockedVolume<uint8_t>::Node &original_node = nodes[info.node_handle(original_block_x, original_block_y, original_block_z, layer - 1)];

                node.min = std::min(node.min, original_node.min);
                node.max = std::max(node.max, original_node.max);
              }
            }
          }

          if (node.min != node.max) {
            #pragma omp critical
            {
              node.block_handle = block_handle++;
              blocked_volume.write((const char *)block, sizeof(block));
            }
          }
        }
      }
    }
  }
}
