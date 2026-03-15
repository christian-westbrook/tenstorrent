#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {

    // Retrieve runtime arguments
    uint32_t number_of_tiles = get_arg_val<uint32_t>(0);

    // Get handles to circular buffers
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    constexpr uint32_t dst_reg_id = 0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    for (uint32_t i = 0; i < number_of_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();

        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg_id);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg_id, cb_out0);

        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

        tile_regs_release();
    }
}