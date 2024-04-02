// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_dst = get_compile_time_arg_val(1);
constexpr auto cb_reshape = get_compile_time_arg_val(2);
constexpr auto cb_scanned = get_compile_time_arg_val(3);
constexpr auto cb_block = get_compile_time_arg_val(4);
constexpr auto cb_row = get_compile_time_arg_val(5);

constexpr uint32_t tile_size = get_tile_size(cb_src);
constexpr uint32_t tiles_per_block = 8;
constexpr uint32_t blocks_per_full_reshape = 4;
constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;
constexpr uint32_t quarter_tile_size = tile_size / 4;
constexpr uint32_t reshape_size = tiles_per_reshape * tile_size;


void kernel_main() {
    uint32_t tiles_per_row = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_col = get_arg_val<uint32_t>(1);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(2);
    uint32_t total_tiles = get_arg_val<uint32_t>(3);

    cb_push_back(cb_src, total_tiles);  // signal to compute kernel that the src CB is ready

    for (uint32_t row = 0; row < tiles_per_col; ++row) {
        for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
            cb_reserve_back(cb_reshape, tiles_per_reshape);

            for (uint32_t block = 0; block < blocks_per_full_reshape; ++block) {
                cb_wait_front(cb_block, tiles_per_block);
                cb_pop_front(cb_block, tiles_per_block);
            }

            cb_push_back(cb_reshape, tiles_per_reshape);
        }
    }
}
