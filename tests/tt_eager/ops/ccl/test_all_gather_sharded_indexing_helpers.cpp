// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"


TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceFullTileRow_ClockWise_In3x5_NumShards3) {
    bool is_clockwise = true;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;

    { // Advance to end from start of row
        uint16_t curr_shard_tile_x = 0;
        uint16_t curr_shard_tile_y = 0;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;

        ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 1);
        ASSERT_EQ(curr_tile_index, total_num_tiles_x);
        ASSERT_EQ(curr_shard, 0);
    }
    { // Advance to end from "middle" of row
        uint16_t curr_shard_tile_x = 3;
        uint16_t curr_shard_tile_y = 0;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;

        ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 1);
        ASSERT_EQ(curr_tile_index, total_num_tiles_x);
        ASSERT_EQ(curr_shard, 0);
    }

    { // Advance to end from start of "middle" row
        uint16_t curr_shard_tile_x = 0;
        uint16_t curr_shard_tile_y = 1;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;

        ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 2);
        ASSERT_EQ(curr_tile_index, total_num_tiles_x * 2);
        ASSERT_EQ(curr_shard, 0);
    }
    { // Advance to end from "middle" of "middle" row
        uint16_t curr_shard_tile_x = 3;
        uint16_t curr_shard_tile_y = 1;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;

        ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 2);
        ASSERT_EQ(curr_tile_index, total_num_tiles_x * 2);
        ASSERT_EQ(curr_shard, 0);
    }

    { // Advance to end from "start" of row, and from last row
        uint16_t curr_shard_tile_x = 0;
        uint16_t curr_shard_tile_y = input_shard_num_tiles_y - 1;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;

        ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 0);
        ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * (num_shards_x - 1));
        ASSERT_EQ(curr_shard, num_shards_x - 1);
    }
    { // Advance to end from "middle" of row, and from last row, first shard
        uint16_t curr_shard_tile_x = 2;
        uint16_t curr_shard_tile_y = input_shard_num_tiles_y - 1;
        uint16_t curr_tile_index = (total_num_tiles_x * curr_shard_tile_y) + curr_shard_tile_x;
        uint16_t curr_shard = 0;

        ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
        ASSERT_EQ(curr_shard_tile_x, 0);
        ASSERT_EQ(curr_shard_tile_y, 0);
        ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * (num_shards_x - 1));
        ASSERT_EQ(curr_shard, num_shards_x - 1);
    }
}


TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceFullTileRow_ClockWise_In3x5_NumShards3_Sequence) {
    bool is_clockwise = true;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;
    uint16_t total_num_tiles = total_num_tiles_x * input_shard_num_tiles_y;

    uint16_t curr_shard_tile_x = 0;
    uint16_t curr_shard_tile_y = 0;
    uint16_t curr_tile_index = 0;
    uint16_t curr_shard = 0;

    // 0
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, total_num_tiles_x);
    ASSERT_EQ(curr_shard, 0);

    // 1
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, total_num_tiles_x * 2);
    ASSERT_EQ(curr_shard, 0);

    // 2
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * (num_shards_x - 1));
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 3
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * (num_shards_x - 1) + total_num_tiles_x);
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 4
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * (num_shards_x - 1) + (total_num_tiles_x * 2));
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 5
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * (num_shards_x - 2) + (total_num_tiles_x * 0));
    ASSERT_EQ(curr_shard, num_shards_x - 2);

    // 6
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * (num_shards_x - 2) + (total_num_tiles_x * 1));
    ASSERT_EQ(curr_shard, num_shards_x - 2);

    // 7
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * (num_shards_x - 2) + (total_num_tiles_x * 2));
    ASSERT_EQ(curr_shard, num_shards_x - 2);

    // 8
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * (num_shards_x - 3) + (total_num_tiles_x * 0));
    ASSERT_EQ(curr_shard, num_shards_x - 3);
}


TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceFullTileRow_CounterClockWise_In3x5_NumShards3_Sequence) {
    bool is_clockwise = false;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;
    uint16_t total_num_tiles = total_num_tiles_x * input_shard_num_tiles_y;

    uint16_t curr_shard_tile_x = 0;
    uint16_t curr_shard_tile_y = 0;
    uint16_t curr_tile_index = 0;
    uint16_t curr_shard = 0;

    // 0
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, total_num_tiles_x);
    ASSERT_EQ(curr_shard, 0);

    // 1
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, total_num_tiles_x * 2);
    ASSERT_EQ(curr_shard, 0);

    // 2
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, input_shard_num_tiles_x * 1 + (total_num_tiles_x * 0));
    ASSERT_EQ(curr_shard, 1);

    // 3
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 1));
    ASSERT_EQ(curr_shard, 1);

    // 4
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 2));
    ASSERT_EQ(curr_shard, 1);

    // 5
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 2) + (total_num_tiles_x * 0));
    ASSERT_EQ(curr_shard, 2);

    // 6
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 2) + (total_num_tiles_x * 1));
    ASSERT_EQ(curr_shard,  2);

    // 7
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 2) + (total_num_tiles_x * 2));
    ASSERT_EQ(curr_shard, 2);

    // 8
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, 0);
    ASSERT_EQ(curr_shard, 0);
}



TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceSingleTile_ClockWise_In3x5_NumShards3_Sequence) {
    bool is_clockwise = true;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;
    uint16_t total_num_tiles = total_num_tiles_x * input_shard_num_tiles_y;

    uint16_t curr_shard_tile_x = 0;
    uint16_t curr_shard_tile_y = 0;
    uint16_t curr_tile_index = 0;
    uint16_t curr_shard = 0;

    // 1
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, 1);
    ASSERT_EQ(curr_shard, 0);

    // 2
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, 2);
    ASSERT_EQ(curr_shard, 0);

    // 3
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, 3);
    ASSERT_EQ(curr_shard, 0);

    // 4
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, 4);
    ASSERT_EQ(curr_shard, 0);


    // 5
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1));
    ASSERT_EQ(curr_shard, 0);

    // 6
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1) + 1);
    ASSERT_EQ(curr_shard, 0);

    // 7
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1) + 2);
    ASSERT_EQ(curr_shard, 0);

    // 8
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1) + 3);
    ASSERT_EQ(curr_shard, 0);

    // 9
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1) + 4);
    ASSERT_EQ(curr_shard, 0);


    // 10
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 0);
    ASSERT_EQ(curr_shard, 0);

    // 11
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 1);
    ASSERT_EQ(curr_shard, 0);

    // 12
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 2);
    ASSERT_EQ(curr_shard, 0);

    // 13
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 3);
    ASSERT_EQ(curr_shard, 0);

    // 14
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 4);
    ASSERT_EQ(curr_shard, 0);


    // 15
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)));
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 16
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)) + 1);
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 17
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)) + 2);
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 18
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)) + 3);
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 19
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)) + 4);
    ASSERT_EQ(curr_shard, num_shards_x - 1);


    // 20
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)) + (total_num_tiles_x * 1) + 0);
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 21
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)) + (total_num_tiles_x * 1) + 1);
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 22
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)) + (total_num_tiles_x * 1) + 2);
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 23
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)) + (total_num_tiles_x * 1) + 3);
    ASSERT_EQ(curr_shard, num_shards_x - 1);

    // 24
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * (num_shards_x - 1)) + (total_num_tiles_x * 1) + 4);
    ASSERT_EQ(curr_shard, num_shards_x - 1);

}

TEST(AllGatherSharded_WidthShardedIndexing_FullWorkerGridVariant, AdvanceSingleTile_CounterClockWise_In3x5_NumShards3_Sequence) {
    bool is_clockwise = false;
    uint16_t const num_shards_x = 3;
    uint16_t const input_shard_num_tiles_x = 5;
    uint16_t const input_shard_num_tiles_y = 3;
    uint16_t const total_num_tiles_x = input_shard_num_tiles_x * num_shards_x;
    uint16_t total_num_tiles = total_num_tiles_x * input_shard_num_tiles_y;

    uint16_t curr_shard_tile_x = 0;
    uint16_t curr_shard_tile_y = 0;
    uint16_t curr_tile_index = 0;
    uint16_t curr_shard = 0;

    // 1
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, 1);
    ASSERT_EQ(curr_shard, 0);

    // 2
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, 2);
    ASSERT_EQ(curr_shard, 0);

    // 3
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, 3);
    ASSERT_EQ(curr_shard, 0);

    // 4
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, 4);
    ASSERT_EQ(curr_shard, 0);


    // 5
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1));
    ASSERT_EQ(curr_shard, 0);

    // 6
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1) + 1);
    ASSERT_EQ(curr_shard, 0);

    // 7
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1) + 2);
    ASSERT_EQ(curr_shard, 0);

    // 8
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1) + 3);
    ASSERT_EQ(curr_shard, 0);

    // 9
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 1) + 4);
    ASSERT_EQ(curr_shard, 0);


    // 10
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 0);
    ASSERT_EQ(curr_shard, 0);

    // 11
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 1);
    ASSERT_EQ(curr_shard, 0);

    // 12
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 2);
    ASSERT_EQ(curr_shard, 0);

    // 13
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 3);
    ASSERT_EQ(curr_shard, 0);

    // 14
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 2);
    ASSERT_EQ(curr_tile_index, (total_num_tiles_x * 2) + 4);
    ASSERT_EQ(curr_shard, 0);


    // 15
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 0) + 0);
    ASSERT_EQ(curr_shard, 1);

    // 16
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 0) + 1);
    ASSERT_EQ(curr_shard, 1);

    // 17
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 0) + 2);
    ASSERT_EQ(curr_shard, 1);

    // 18
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 0) + 3);
    ASSERT_EQ(curr_shard, 1);

    // 19
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 0);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 0) + 4);
    ASSERT_EQ(curr_shard, 1);


    // 20
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 0);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 1) + 0);
    ASSERT_EQ(curr_shard, 1);

    // 21
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 1);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 1) + 1);
    ASSERT_EQ(curr_shard, 1);

    // 22
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 2);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 1) + 2);
    ASSERT_EQ(curr_shard, 1);

    // 23
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 3);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 1) + 3);
    ASSERT_EQ(curr_shard, 1);

    // 24
    ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
        curr_shard_tile_x, curr_shard_tile_y, curr_tile_index, curr_shard, input_shard_num_tiles_x, input_shard_num_tiles_y, num_shards_x, is_clockwise);
    ASSERT_EQ(curr_shard_tile_x, 4);
    ASSERT_EQ(curr_shard_tile_y, 1);
    ASSERT_EQ(curr_tile_index, (input_shard_num_tiles_x * 1) + (total_num_tiles_x * 1) + 4);
    ASSERT_EQ(curr_shard, 1);

}
