// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));
    FullWorkerGridShardAddrGen<shard_type> output_tensor_shard_reader;

    uint32_t arg_index = 0;
    ShardAddrGen<shard_type> addr_gen;
    const uint32_t eth_sender_l1_base_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_sender_l1_sem_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_sender_noc_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_sender_noc_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t tiles_per_eth_l1_buffer = get_arg_val<uint32_t>(arg_index++);

    // Used to wait until eth sender has space available
    volatile uint32_t *const eth_buffer_available_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arg_index++));
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_index++);
    volatile uint32_t *const tiles_available_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arg_index++));

    FullWorkerGridShardAddrGen<shard_type>::build_with_placement_new(&output_tensor_shard_reader, arg_index);
    arg_index += output_tensor_shard_reader.get_num_args_consumed();
    arg_index += addr_gen.get_num_args_consumed();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    // This is different per writer core
    const uint64_t eth_l1_sender_base_noc_addr =
        get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_base_addr);
    // Used to signal eth sender that data is available. This is different per writer core
    const uint64_t eth_l1_sender_semaphore_addr =
        get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_sem_addr);

    constexpr bool use_optimized = true;

    uint16_t const tiles_per_input_shard_row = output_tensor_shard_reader.input_shard_num_tiles_x;
    uint16_t const tiles_per_input_shard = tiles_per_input_shard_row * output_tensor_shard_reader.input_shard_num_tiles_y;
    uint16_t sem_idx = 0;
    for (uint32_t t = 0; t < num_transfers; ++t) {

        for (uint32_t i = 0; i < tiles_per_input_shard; i += tiles_per_eth_l1_buffer) {
            uint16_t pages_to_send = std::min<uint16_t>(tiles_per_eth_l1_buffer, tiles_per_input_shard - i);

            noc_semaphore_wait(eth_buffer_available_semaphore_ptr, 1);
            noc_semaphore_set(eth_buffer_available_semaphore_ptr, 0);
            uint64_t eth_l1_buffer_address = eth_l1_sender_base_noc_addr;
            while (pages_to_send > 0) {
                uint16_t num_contiguous_tiles = std::min<uint16_t>(pages_to_send, output_tensor_shard_reader.get_tiles_left_in_row_in_shard());
                sem_idx += num_contiguous_tiles;
                noc_semaphore_wait_min(tiles_available_semaphore_ptr, sem_idx);

                uint32_t source_l1_addr = output_tensor_shard_reader.get_next_l1_addr();
                uint32_t transfer_size_in_bytes = num_contiguous_tiles * output_tensor_shard_reader.get_tile_size_in_bytes();;
                noc_async_write(source_l1_addr, eth_l1_buffer_address, transfer_size_in_bytes);
                eth_l1_buffer_address += transfer_size_in_bytes;
                pages_to_send -= num_contiguous_tiles;
            }

            noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
            noc_async_write_barrier();
        }
    }
}
