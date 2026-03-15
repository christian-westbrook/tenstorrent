void kernel_main() {

    // Retrieve runtime arguments
    uint32_t in0_addr        = get_arg_val<uint32_t>(0);
    uint32_t in1_addr        = get_arg_val<uint32_t>(1);
    uint32_t number_of_tiles = get_arg_val<uint32_t>(2);

    // Get handles to circular buffers
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Retrieve compile time arguments
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_size_bytes);

    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto in1 = TensorAccessor(in1_args, in1_addr, tile_size_bytes);

    // Transfer input data from DRAM to circular buffers
    for (uint32_t i = 0; i < number_of_tiles; i++) {
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);

        noc_async_read_tile(i, in0, get_write_ptr(cb_in0));
        noc_async_read_tile(i, in1, get_write_ptr(cb_in1));

        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);
    }
}