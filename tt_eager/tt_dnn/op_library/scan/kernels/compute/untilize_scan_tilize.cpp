// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_dst = get_compile_time_arg_val(1);
constexpr auto cb_reshape = get_compile_time_arg_val(2);
constexpr auto cb_scanned = get_compile_time_arg_val(3);
constexpr auto cb_block = get_compile_time_arg_val(4);
constexpr auto cb_row = get_compile_time_arg_val(5);

constexpr uint32_t tiles_per_block = 8;
constexpr uint32_t blocks_per_full_reshape = 4;
constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;

ALWI void untilize_from_src(uint32_t src_cb, uint32_t block_cb) {
    untilize_init_short(src_cb);
    for (uint32_t block = 0; block < blocks_per_full_reshape; ++block) {
        cb_wait_front(src_cb, tiles_per_block);
        cb_reserve_back(block_cb, tiles_per_block);

        untilize_block(src_cb, tiles_per_block, block_cb);

        cb_push_back(block_cb, tiles_per_block);
        cb_pop_front(src_cb, tiles_per_block);
    }
    untilize_uninit(src_cb);
}

ALWI void tilize_to_dst(uint32_t block_cb, uint32_t dst_cb) {
    tilize_init_short(block_cb, tiles_per_block);
    for (uint32_t block = 0; block < blocks_per_full_reshape; ++block) {
        cb_wait_front(block_cb, tiles_per_block);
        cb_reserve_back(dst_cb, tiles_per_block);

        tilize_block(block_cb, tiles_per_block, dst_cb);

        cb_push_back(dst_cb, tiles_per_block);
        cb_pop_front(block_cb, tiles_per_block);
    }
    tilize_uninit(block_cb);
}

namespace NAMESPACE {
void MAIN {
    uint32_t tiles_per_row = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_col = get_arg_val<uint32_t>(1);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(2);
    uint32_t total_tiles = get_arg_val<uint32_t>(3);

    binary_op_init_common(cb_reshape, cb_reshape, cb_scanned);
    mul_tiles_init();

    cb_reserve_back(cb_dst, total_tiles);

    for (uint32_t row = 0; row < tiles_per_col; ++row) {
        for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
            untilize_from_src(cb_src, cb_block);

            cb_wait_front(cb_reshape, tiles_per_reshape);
            cb_reserve_back(cb_scanned, tiles_per_reshape);

            // // copy the first tile as is
            // tile_regs_acquire();
            // copy_tile_to_dst_init_short(cb_full_reshape);
            // copy_tile(cb_full_reshape, 0, 0);
            // tile_regs_commit();
            // tile_regs_wait();
            // pack_tile(0, cb_scanned);
            // tile_regs_release();

            // // pairwise multiply tiles in the cb_full_reshape to cb_scanned
            // for (uint32_t tile = 0; tile < tiles_per_reshape - 1; ++tile) {
            //     tile_regs_acquire();
            //     mul_tiles(cb_full_reshape, cb_full_reshape, tile, tile + 1, 0);
            //     tile_regs_commit();
            //     tile_regs_wait();
            //     pack_tile(0, cb_scanned);
            //     tile_regs_release();
            // }

            cb_pop_front(cb_reshape, tiles_per_reshape);
            cb_push_back(cb_scanned, tiles_per_reshape);

            tilize_to_dst(cb_block, cb_dst);
        }
    }
}
}
