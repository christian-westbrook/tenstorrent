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

	constexpr int device_id = 0;
	auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();

    constexpr uint32_t num_tiles = 50;
    constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
    constexpr uint32_t dram_buffer_size = tile_size_bytes * num_tiles;

    // Create an L1 buffer
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::L1
    };

    distributed::ReplicatedBufferConfig l1_buffer_config{
        .size = tile_size_bytes
    };

    auto l1_buffer = distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());

    fmt::print("-------------------------------------------------------\n");
    fmt::print("Created an L1 buffer of size: {} bytes\n", l1_buffer->size());
    fmt::print("-------------------------------------------------------\n");

    // Create two DRAM buffers
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    distributed::ReplicatedBufferConfig dram_buffer_config{
        .size = dram_buffer_size
    };

    auto input_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto output_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

    fmt::print("\n");
    fmt::print("-------------------------------------------------------\n");
    fmt::print("Created an input DRAM buffer of size: {} bytes\n", input_dram_buffer->size());
    fmt::print("-------------------------------------------------------\n");

    fmt::print("\n");
    fmt::print("-------------------------------------------------------\n");
    fmt::print("Created an output DRAM buffer of size: {} bytes\n", output_dram_buffer->size());
    fmt::print("-------------------------------------------------------\n");

	return 0;
}