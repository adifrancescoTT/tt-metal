// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    DPRINT << "WR: Start\n";
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));

    FullWorkerGridShardAddrGen<shard_type> input_tensor_shard_reader;
    FullWorkerGridShardAddrGen<shard_type> output_tensor_shard_reader;

    uint32_t arg_index = 0;
    uint32_t input_shard_address = get_arg_val<uint32_t>(arg_index++);
    volatile tt_l1_ptr uint32_t* eth_to_local_semaphore_address = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arg_index++));
    uint32_t const tiles_per_eth_l1_buffer = get_arg_val<uint32_t>(arg_index++);
    volatile tt_l1_ptr  uint32_t *const tiles_available_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr  uint32_t*>(get_arg_val<uint32_t>(arg_index++));
    uint32_t const eth_noc_x = get_arg_val<uint32_t>(arg_index++);
    uint32_t const eth_noc_y = get_arg_val<uint32_t>(arg_index++);
    uint32_t const eth_l1_buffer_addres = get_arg_val<uint32_t>(arg_index++);
    uint32_t const eth_semaphore_addres = get_arg_val<uint32_t>(arg_index++);
    uint32_t const num_transfers = get_arg_val<uint32_t>(arg_index++);
    FullWorkerGridShardAddrGen<shard_type>::build_with_placement_new(&output_tensor_shard_reader, arg_index);
    arg_index += output_tensor_shard_reader.get_num_args_consumed();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    uint32_t sem_idx = 0;

    uint32_t in_row_start_addr = input_shard_address;

    // Read the input shard to the CB
    uint16_t tiles_per_input_shard = output_tensor_shard_reader.input_shard_num_tiles_x * output_tensor_shard_reader.input_shard_num_tiles_y;
    DPRINT << "WR: Reading input \n";
    cb_reserve_back(cb_id_in0, tiles_per_input_shard);
    uint32_t l1_read_addr = get_read_ptr(cb_id_in0);
    noc_async_write(input_shard_address, l1_read_addr, tiles_per_input_shard);
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, tiles_per_input_shard);
    noc_semaphore_set(tiles_available_semaphore_ptr, sem_idx);

    const uint64_t eth_l1_buf_noc_address = get_noc_addr(eth_noc_x, eth_noc_y, eth_l1_buffer_addres);

    for (uint32_t i = 1; i < num_transfers; ++i) {
        DPRINT << "WR: Transfer " << i << "\n";

        uint16_t tiles_left_in_input_shard = tiles_per_input_shard;

        cb_wait_front(cb_id_in0, tiles_per_input_shard);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in0);
        for (uint32_t input_shard_tile_idx = 0; input_shard_tile_idx < tiles_per_input_shard; input_shard_tile_idx += tiles_per_eth_l1_buffer) {
            DPRINT << "WR: input_shard_tile_idx " << input_shard_tile_idx << "\n";
            uint32_t num_tiles_to_send = std::min(tiles_per_eth_l1_buffer, tiles_per_input_shard - input_shard_tile_idx);
            uint32_t read_size_in_bytes = num_tiles_to_send * output_tensor_shard_reader.get_tile_size_in_bytes();

            noc_semaphore_wait(eth_to_local_semaphore_address, 1);
            noc_semaphore_set(eth_to_local_semaphore_address, 0);
            noc_async_read(eth_l1_buf_noc_address, l1_read_addr, read_size_in_bytes);
            l1_read_addr += read_size_in_bytes;
            DPRINT << "WE: reading tiles: " << num_tiles_to_send << "\n";
            noc_async_read_barrier();
            DPRINT << "WE: done barrier\n";
            noc_semaphore_inc(eth_semaphore_addres, 1);
            cb_push_back(cb_id_in0, num_tiles_to_send);
        }
    }
    DPRINT << "WR: End\n";
}
