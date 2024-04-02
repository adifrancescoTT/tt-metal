// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "debug/dprint.h"

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_dst = get_compile_time_arg_val(1);
constexpr auto cb_reshape = get_compile_time_arg_val(2);
constexpr auto cb_scanned = get_compile_time_arg_val(3);
constexpr auto cb_block = get_compile_time_arg_val(4);
constexpr auto cb_row = get_compile_time_arg_val(5);

constexpr uint32_t tiles_per_block = 8;
constexpr uint32_t blocks_per_full_reshape = 4;
constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;

namespace NAMESPACE {
void MAIN {
    uint32_t tiles_per_row = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_col = get_arg_val<uint32_t>(1);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(2);
    uint32_t total_tiles = get_arg_val<uint32_t>(3);

    untilize_init(cb_src, cb_block);

    for (uint32_t row = 0; row < tiles_per_col; ++row) {
        for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
            DPRINT_UNPACK(DPRINT << "Row " << row << " reshape " << reshape << ENDL());

            untilize_init_short(cb_src);
            for (uint32_t block = 0; block < blocks_per_full_reshape; ++block) {
                cb_wait_front(cb_src, tiles_per_block);
                cb_reserve_back(cb_block, tiles_per_block);

                DPRINT_UNPACK(DPRINT << "Untilizing " << tiles_per_block << " tiles from src to block CB" << ENDL());
                untilize_block(cb_src, tiles_per_block, cb_block);
                DPRINT_UNPACK(DPRINT << "Done untilizing block " << block << ENDL());

                cb_push_back(cb_block, tiles_per_block);
                cb_pop_front(cb_src, tiles_per_block);
            }
        }
    }
}
}  // namespace NAMESPACE
