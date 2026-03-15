#include <fmt/ostream.h>
#include <cstdint>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;

int main() {

    // Create a device mesh
    constexpr int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();

    // Create input and output DRAM buffers
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t number_of_tiles = 64;
    constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM
    };

    distributed::ReplicatedBufferConfig dram_buffer_config{
        .size = number_of_tiles * tile_size_bytes
    };

    auto src0_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer  = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

    // Generate test data
    constexpr float value_to_add = -1.0f;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<bfloat16> src0_data(elements_per_tile * number_of_tiles);
    std::vector<bfloat16> src1_data(elements_per_tile * number_of_tiles, bfloat16(value_to_add));

    for(auto& value : src0_data) {
        value = bfloat16(distribution(rng));
    }

    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_data, false);

    return 0;
}



