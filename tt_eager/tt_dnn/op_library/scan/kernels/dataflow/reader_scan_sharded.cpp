// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_HW = 1024;

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

#define ALWI inline __attribute__((always_inline))

ALWI void reshape_from_blocks(uint32_t block_cb, uint32_t reshape_cb) {
    cb_reserve_back(reshape_cb, tiles_per_reshape);

    for (uint32_t block = 0, offset = 0; block < blocks_per_full_reshape; ++block, offset += quarter_tile_size) {
        cb_wait_front(block_cb, tiles_per_block);

        uint64_t src_noc_addr = get_noc_addr(get_read_ptr(block_cb));
        uint32_t dst_addr = get_write_ptr(reshape_cb) + offset;
        for (uint32_t row = 0; row < TILE_HEIGHT; ++row, src_noc_addr += quarter_tile_size, dst_addr += tile_size) {
            noc_async_read(src_noc_addr, dst_addr, quarter_tile_size);
        }
        noc_async_read_barrier();

        cb_pop_front(block_cb, tiles_per_block);
    }

    cb_push_back(reshape_cb, tiles_per_reshape);
}

ALWI void undo_reshape_to_blocks(uint32_t scanned_cb, uint32_t block_cb) {
    cb_wait_front(scanned_cb, tiles_per_reshape);

    for (uint32_t block = 0, offset = 0; block < blocks_per_full_reshape; ++block, offset += quarter_tile_size) {
        cb_reserve_back(block_cb, tiles_per_block);

        uint32_t src_addr = get_read_ptr(scanned_cb) + offset;
        uint64_t dst_noc_addr = get_noc_addr(get_write_ptr(block_cb));
        for (uint32_t row = 0; row < TILE_HEIGHT; ++row, src_addr += tile_size, dst_noc_addr += quarter_tile_size) {
            noc_async_write(src_addr, dst_noc_addr, quarter_tile_size);
        }
        noc_async_write_barrier();

        cb_push_back(block_cb, tiles_per_block);
    }

    cb_pop_front(scanned_cb, tiles_per_reshape);
}

void kernel_main() {
    uint32_t tiles_per_row = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_col = get_arg_val<uint32_t>(1);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(2);
    uint32_t total_tiles = get_arg_val<uint32_t>(3);

    uint32_t dst_addr = get_write_ptr(cb_src);

    cb_push_back(cb_src, total_tiles);  // signal to compute kernel that the src CB is ready

    for (uint32_t row = 0; row < tiles_per_col; ++row) {
        for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape, dst_addr += reshape_size) {
            reshape_from_blocks(cb_block, cb_reshape); // fills cb_reshape
            undo_reshape_to_blocks(cb_scanned, cb_block); // waits for cb_scanned to be filled
        }
    }
}
