#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
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

    // Create circular buffers
    constexpr uint32_t tiles_per_circular_buffer = 2;

    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
    tt::CBIndex dst_cb_index  = tt::CBIndex::c_16;

    CircularBufferConfig c0_config = CircularBufferConfig(
        tiles_per_circular_buffer * tile_size_bytes,
        {{src0_cb_index, tt::DataFormat::Float16_b}}
    ).set_page_size(src0_cb_index, tile_size_bytes);

    CircularBufferConfig c1_config = CircularBufferConfig(
        tiles_per_circular_buffer * tile_size_bytes,
        {{src1_cb_index, tt::DataFormat::Float16_b}}
    ).set_page_size(src1_cb_index, tile_size_bytes);

    CircularBufferConfig dst_config = CircularBufferConfig(
        tiles_per_circular_buffer * tile_size_bytes,
        {{dst_cb_index, tt::DataFormat::Float16_b}}
    ).set_page_size(dst_cb_index, tile_size_bytes);

    CBHandle cb_src0 = CreateCircularBuffer(program, core, c0_config);
    CBHandle cb_src1 = CreateCircularBuffer(program, core, c1_config);
    CBHandle cb_dst  = CreateCircularBuffer(program, core, dst_config);

    // Create kernels
    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*src0_dram_buffer->get_backing_buffer()).append_to(reader_args);
    TensorAccessorArgs(*src1_dram_buffer->get_backing_buffer()).append_to(reader_args);

    auto reader = CreateKernel(
        program,
        "elementwise_addition/kernels/dataflow/read_tiles.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_args
        }
    );

    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*dst_dram_buffer->get_backing_buffer()).append_to(writer_args);

    auto writer = CreateKernel(
        program,
        "elementwise_addition/kernels/dataflow/write_tiles.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_args
        }
    );

    auto compute = CreateKernel(
        program,
        "elementwise_addition/kernels/compute/tiles_add.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4
        }
    );

    // Run the program
    SetRuntimeArgs(program, reader, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), number_of_tiles});
    SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), number_of_tiles});
    SetRuntimeArgs(program, compute, core, {number_of_tiles});

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Verify output
    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    bool pass = true;

    constexpr float eps = 1e-2f;
    TT_FATAL(result_vec.size() == src0_data.size(), "Result vector size mismatch");
    for (size_t i = 0; i < result_vec.size(); ++i) {
        //const float expected = src0_data[i].to_float() + value_to_add;
        //const float actual = result_vec[i].to_float();

        const float expected = static_cast<float>(src0_data[i]) + value_to_add;
        const float actual = static_cast<float>(result_vec[i]);

        if (std::abs(expected - actual) > eps) {
            pass = false;
            fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
        }
    }

    pass &= mesh_device->close();

    fmt::print("\n");
    fmt::print("-------------------------------------------------------\n");
    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed\n");
    }
    fmt::print("-------------------------------------------------------\n");

    return 0;
}



