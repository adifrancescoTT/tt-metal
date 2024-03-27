// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);

    FullWorkerGridShardAddrGen<shard_type> input_tensor_shard_reader;
    FullWorkerGridShardAddrGen<shard_type> output_tensor_shard_reader;

    uint32_t arg_index = 0;
    volatile tt_l1_ptr uint32_t* eth_to_local_semaphore_address = get_arg_val<volatile tt_l1_ptr uint32_t*>(arg_index++);
    uint32_t const tiles_per_eth_l1_buffer = get_arg_val<uint32_t>(arg_index++);
    volatile uint32_t *const tiles_available_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_index++));
    uint32_t const eth_noc_x = get_arg_val<uint32_t>(arg_index++);
    uint32_t const eth_noc_y = get_arg_val<uint32_t>(arg_index++);
    uint32_t const eth_l1_buffer_addres = get_arg_val<uint32_t>(arg_index++);
    uint32_t const eth_semaphore_addres = get_arg_val<uint32_t>(arg_index++);
    FullWorkerGridShardAddrGen<shard_type>::build_with_placement_new(&input_tensor_shard_reader, arg_index);
    arg_index += input_tensor_shard_reader.get_num_args_consumed();
    FullWorkerGridShardAddrGen<shard_type>::build_with_placement_new(&output_tensor_shard_reader, arg_index);
    arg_index += output_tensor_shard_reader.get_num_args_consumed();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    uint32_t sem_idx = 0;

    for (uint16_t tile_row = 0; tile_row < input_tensor_shard_reader.input_shard_num_tiles_x; tile_row++) {
        uint32_t in_row_start_addr = input_tensor_shard_reader.get_next_noc_addr();
        uint32_t out_row_start_addr = input_tensor_shard_reader.get_next_noc_addr();
        uint16_t num_tiles_in_row = input_tensor_shard_reader.get_shard_tile_row_size_in_bytes();
        sem_idx += num_tiles_in_row;
        ASSERT(num_tiles_in_row == output_tensor_shard_reader.get_shard_tile_row_size_in_bytes());
        noc_async_read(in_row_start_addr, out_row_start_addr, num_tiles_in_row);
    }
    noc_async_read_barrier();
    noc_semaphore_set(tiles_available_semaphore_ptr, sem_idx);

    const uint64_t eth_l1_buf_noc_address = get_noc_addr(eth_noc_x, eth_noc_y, eth_l1_buffer_addres);

    uint16_t tiles_per_input_shard = input_tensor_shard_reader.input_shard_num_tiles_x * input_tensor_shard_reader.input_shard_num_tiles_y;
    for (uint32_t i = 1; i < num_transfers; ++i) {
        uint16_t tiles_left_in_input_shard = tiles_per_input_shard;

        for (uint32_t input_shard_tile_idx = 0; input_shard_tile_idx < tiles_per_input_shard; input_shard_tile_idx += tiles_per_eth_l1_buffer) {
            uint32_t num_tiles_to_send = std::min(tiles_per_eth_l1_buffer, tiles_per_input_shard - input_shard_tile_idx);
            noc_semaphore_wait(eth_to_local_semaphore_address, 1);
            noc_semaphore_set(eth_to_local_semaphore_address, 0);

            uint32_t tiles_left_to_read_from_eth_buffer = num_tiles_to_send;
            uint64_t source_address = eth_l1_buf_noc_address;

            while (tiles_left_to_read_from_eth_buffer > 0) {
                uint32_t num_contiguous_tiles_to_read = std::min(std::min(tiles_left_to_read_from_eth_buffer, tiles_per_eth_l1_buffer), input_tensor_shard_reader.get_tiles_left_in_row_in_shard());
                uint32_t out_row_start_addr = output_tensor_shard_reader.get_next_noc_addr();;


                ASSERT(num_tiles_in_row == output_tensor_shard_reader.get_shard_tile_row_size_in_bytes());

                uint32_t read_size_in_bytes = num_contiguous_tiles_to_read * output_tensor_shard_reader.get_tile_size_in_bytes();
                noc_async_read(source_address, out_row_start_addr, read_size_in_bytes);
                sem_idx += num_contiguous_tiles_to_read;

                output_tensor_shard_reader.advance_n_tiles(num_contiguous_tiles_to_read);

                source_address += read_size_in_bytes;
                tiles_left_to_read_from_eth_buffer -= num_contiguous_tiles_to_read;
            }
            // Advantage: single barrier for all reads from eth
            // Disadvantage: Sender can only forward after full eth_l1_buffer worth of tiles is received
            // Ideal: point the sender core to the reads complete register and use that directly instead
            //        of incrementing a semaphore
            noc_async_read_barrier();
            noc_semaphore_set(tiles_available_semaphore_ptr, sem_idx);

            noc_semaphore_inc(eth_semaphore_addres, 1);
        }
    }
}
