// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    DPRINT << "SR: START\n";
    DPRINT << "SR: my_y=" << (uint32_t)my_y[0] << "\n";
    DPRINT << "SR: my_x=" << (uint32_t)my_x[0] << "\n";
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);

    ShardAddrGen<shard_type> input_tensor_shard_reader;
    ShardAddrGen<shard_type> output_tensor_shard_reader;

    DPRINT << "SR: 2\n";
    uint32_t arg_index = 0;
    volatile tt_l1_ptr uint32_t* local_semaphore_address = get_arg_val<volatile tt_l1_ptr uint32_t*>(arg_index++);
    uint32_t const num_shards_per_transfer = get_arg_val<uint32_t>(arg_index++);
    uint32_t const shards_per_eth_l1_buffer = get_arg_val<uint32_t>(arg_index++);
    uint32_t const half_cb_n_pages = get_arg_val<uint32_t>(arg_index++);
    ShardAddrGen<shard_type>::build_with_placement_new(&input_tensor_shard_reader, arg_index);
    DPRINT << "SR: 3\n";
    arg_index += input_tensor_shard_reader.get_num_args_consumed();
    ShardAddrGen<shard_type>::build_with_placement_new(&output_tensor_shard_reader, arg_index);
    arg_index += output_tensor_shard_reader.get_num_args_consumed();
    DPRINT << "SR: 5\n";

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    DPRINT << "SR: num_transfers " << num_transfers << "\n";
    DPRINT << "SR: input_tensor_shard_reader.get_num_dest_cores() " << input_tensor_shard_reader.get_num_dest_cores() << "\n";
    DPRINT << "SR: num_shards_per_transfer " << num_shards_per_transfer << "\n";
    DPRINT << "SR: output_tensor_shard_reader.get_num_dest_cores() " << output_tensor_shard_reader.get_num_dest_cores() << "\n";

    constexpr bool use_optimized = true;
    if constexpr (use_optimized) {

        for (uint32_t c = 0; c < num_shards_per_transfer; c += shards_per_eth_l1_buffer) {
            uint32_t num_shards_to_send = std::min(shards_per_eth_l1_buffer, num_shards_per_transfer - c);
            read_shard_from_input_tensor_sharded(cb_id_in0, input_tensor_shard_reader, num_shards_to_send);
            ASSERT(half_cb_n_pages >= num_shards_to_send);
            if (half_cb_n_pages > num_shards_to_send) {
                //
                push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - num_shards_to_send);
            }
        }
    } else {
        for (uint32_t c = 0; c < num_shards_per_transfer; ++c) {
            DPRINT << "SR: Read input tensor chunk from local chip. Shard " << c << "\n";
            read_shard_from_input_tensor_sharded(cb_id_in0, input_tensor_shard_reader, 1);
        }
        if (half_cb_n_pages > num_shards_per_transfer) {
            push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - num_shards_per_transfer);
        }
    }

    DPRINT << "SR: Finished Read input tensor chunk from local chip \n";
    uint32_t sem_idx = 1;

    for (uint32_t i = 1; i < num_transfers; ++i) {
        DPRINT << "SR: Transfer " << i << "\n";

        constexpr bool use_optimized2 = true;
        if constexpr (use_optimized2) {
            for (uint32_t c = 0; c < num_shards_per_transfer; c += shards_per_eth_l1_buffer) {
                uint32_t num_shards_to_send = std::min(shards_per_eth_l1_buffer, num_shards_per_transfer - c);
                noc_semaphore_wait_min(local_semaphore_address, sem_idx);
                sem_idx += num_shards_to_send;
                read_chunk_from_output_tensor_sharded(cb_id_in0, output_tensor_shard_reader, num_shards_to_send);  // 1 chunk == 1 page?
                if (half_cb_n_pages > num_shards_to_send) {
                    push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - num_shards_to_send);
                }
            }
        } else {
            for (uint32_t c = 0; c < num_shards_per_transfer; ++c) {
                DPRINT << "SR: Chunk " << c << "\n";
                DPRINT << "SR: Waiting for semaphore at " << (uint32_t)local_semaphore_address << "\n";
                noc_semaphore_wait_min(local_semaphore_address, sem_idx);
                DPRINT << "SR: Got semaphore\n";
                sem_idx++;
                read_chunk_from_output_tensor_sharded(cb_id_in0, output_tensor_shard_reader, 1);  // 1 chunk == 1 page?
            }
            if (half_cb_n_pages > num_shards_per_transfer) {
                push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - num_shards_per_transfer);
            }
        }
        DPRINT << "SR: DONE Transfer " << i << "\n";
    }

    DPRINT << "SR: END\n";
}
