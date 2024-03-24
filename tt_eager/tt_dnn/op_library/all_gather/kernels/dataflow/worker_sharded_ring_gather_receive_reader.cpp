// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    DPRINT << "RR: START\n";
    DPRINT << "RR: \tmy_y:" << (uint32_t)(my_y[0]) << "\n";
    DPRINT << "RR: \tmy_x:" << (uint32_t)(my_x[0]) << "\n";
    // TODO: Update the interleaver receive reader kernel invocation to just be able to use this
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));
    ShardAddrGen<shard_type> input_tensor_shard_writer;

    // Info about the eth receiver eth core (producer of this core)
    // TODO: Make this arch agnostic

    uint32_t arg_index = 0;
    const uint32_t eth_receiver_noc_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_receiver_noc_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_receiver_l1_base_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_receiver_l1_semaphore_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t receiver_read_sem_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t input_shards_per_eth_buffer = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_index++);

    ShardAddrGen<shard_type>::build_with_placement_new(&input_tensor_shard_writer, arg_index);
    arg_index += input_tensor_shard_writer.get_num_args_consumed();
    ASSERT(eth_receiver_noc_x >= 1 && eth_receiver_noc_x < 12  && (eth_receiver_noc_y == 0 || eth_receiver_noc_y == 6));


    // Eth receiver will set this semaphore when data is available
    volatile tt_l1_ptr uint32_t* receiver_read_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_read_sem_addr);

    // Address of the buffer on the eth receiver, this is different per receiver worker core
    const uint64_t eth_receiver_l1_base_noc_addr = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_base_addr);

    // Address of the semaphore on the eth receiver, this is the same per receiver worker core
    const uint64_t eth_receiver_l1_semaphore_noc_addr = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_semaphore_addr);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    // -1 because we are not receiving from our local chip (local chip has num_dest_cores_worth of tiles already handled)
    uint32_t shards_per_ring_index = input_tensor_shard_writer.get_num_dest_cores();
    DPRINT << "RR: shards_per_ring_index: " << shards_per_ring_index << "\n";
    // DPRINT << "RR: total_num_shards: " << total_num_shards << "\n";
    DPRINT << "RR: input_shards_per_eth_buffer: " << input_shards_per_eth_buffer << "\n";
    DPRINT << "RR: num_transfers: " << num_transfers << "\n";

    uint32_t const shard_size = input_tensor_shard_writer.get_shard_size_in_bytes();

    for (uint32_t t = 0; t < num_transfers; t++) {
        DPRINT << "RR: transfer " << t << "\n";
        // DPRINT << "RR: Eth buffer flush " << i << "\n";
        for (uint32_t i = 0; i < shards_per_ring_index; i += input_shards_per_eth_buffer) {
            uint32_t shards_to_send = std::min(input_shards_per_eth_buffer, shards_per_ring_index - i);
            // `shards_to_send` to CB ... need to make sure we are aware of CB wraparound
            DPRINT << "RR: Waiting for semaphore inc at " << (uint32_t)receiver_read_semaphore_addr_ptr << "\n";
            noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
            noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
            DPRINT << "RR: \tgot semaphore inc from EDM\n";
            for (uint32_t s = 0; s < shards_to_send; ++s) {
                DPRINT << "RR: Shard " << i + s << "\n";
                // DPRINT << "\tRR: got signal from EDM \n";
                // Read page by page so that writer can be kicked off instead of being blocked waiting for full
                // chunk to be read Look into perf/optimizations for this
                uint64_t source_eth_buffer_noc_addr = eth_receiver_l1_base_noc_addr + s * shard_size;
                DPRINT << "RR: \tfetching chunk from source_eth_buffer_noc_addr: " << (uint64_t)(source_eth_buffer_noc_addr & 0xFFFFFFFF) << "\n";
                fetch_chunk_sharded(
                    cb_id_in0,
                    1,
                    shard_size,
                    source_eth_buffer_noc_addr);
                // eth_receiver_l1_base_noc_addr += input_tensor_shard_writer.get_shard_size_in_bytes();
                // DPRINT << "\tRR: fetched chunk EDM \n";
            }
            noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);

        }
    }


    DPRINT << "RR: END\n";
}
