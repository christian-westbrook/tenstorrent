constexpr uint32_t TILE_WIDTH  = 32;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t BYTES_PER_ELEMENT = 2;

void kernel_main() {

    // Retrieve runtime arguments
    std::uint32_t l1_buffer_addr       = get_arg_val<uint32_t>(0);
    std::uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(2);
    std::uint32_t num_tiles            = get_arg_val<uint32_t>(3);

    // Retrieve compile time arguments
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr, tile_size_bytes);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr, tile_size_bytes);

    // Calculate bytes for a single tile
    const uint32_t tile_size_bytes = TILE_WIDTH * TILE_HEIGHT * BYTES_PER_ELEMENT;

    // Transfer data
    for(uint32_t i=0; i < num_tiles; i++) {
        noc_async_read_tile(i, in0, l1_buffer_addr);
        noc_async_read_barrier();

        noc_async_write_tile(i, out0, l1_buffer_addr);
        noc_async_write_barrier();
    }
}