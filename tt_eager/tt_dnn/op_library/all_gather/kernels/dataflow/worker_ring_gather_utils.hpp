// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/assert.h"
#include "tt_eager/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_common.hpp"

using ccl::ShardType;

FORCE_INLINE void validate_sane_transaction_counters() {
    ASSERT (NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) != 0);
    ASSERT (NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT) != 0);
    ASSERT(noc_nonposted_writes_num_issued[noc_index] != 0);
    ASSERT(noc_nonposted_writes_acked[noc_index] != 0);
    ASSERT(noc_reads_num_issued[noc_index] != 0);
}

FORCE_INLINE void validate_sane_transaction_counters_rw() {
    ASSERT (NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) != 0);
    ASSERT (NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT) != 0);
    ASSERT(noc_nonposted_writes_num_issued[noc_index] != 0);
    ASSERT(noc_nonposted_writes_acked[noc_index] != 0);
    ASSERT(noc_reads_num_issued[noc_index] != 0);
}


template <ShardType TYPE>
struct ShardAddrGen final {
    ShardAddrGen()=default;

    FORCE_INLINE static void build_with_placement_new(ShardAddrGen* placement_new_address, const uint32_t arg_index) {
        ccl::ShardAddrGenArgs<false> input_args;

        uint32_t curr_arg_index = arg_index;
        input_args.is_clockwise = bool(get_arg_val<uint32_t>(curr_arg_index++) == 1);
        input_args.shard_size_in_bytes = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.total_chunks_per_core = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.shards_start_address = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.starting_core_index = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.starting_chunk_into_shard = get_arg_val<uint32_t>(curr_arg_index++);

        input_args.intra_core_stride_in_shards = get_arg_val<uint32_t>(curr_arg_index++);
        // input_args.contiguous_chunk_count = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.contiguous_chunks_before_stride = get_arg_val<uint32_t>(curr_arg_index++);

        input_args.num_dest_cores = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.dest_cores = reinterpret_cast<ccl::WorkerXY*>(get_arg_addr(curr_arg_index));
        curr_arg_index += input_args.num_dest_cores;

        ASSERT(input_args.shard_size_in_bytes != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE_U32);
        ASSERT(input_args.total_chunks_per_core != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.shards_start_address != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE_U32);
        ASSERT(input_args.starting_core_index != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.starting_chunk_into_shard != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.num_dest_cores != ccl::ShardAddrGenArgs<true>::UNINITIALIZED_VALUE_U16);

        ASSERT(curr_arg_index - arg_index == input_args.get_expected_num_args());

        new (placement_new_address) ShardAddrGen(
            curr_arg_index - arg_index,
            input_args);
    }

    // This addr gen will dump all tiles from an input shard contiguously, and dump the
    // next input shard contiguously after it. This approach depends on a follow up
    //
    ShardAddrGen(
        uint8_t num_args_consumed,
        ccl::ShardAddrGenArgs<false> const& input_args) :
        dest_cores(input_args.dest_cores),
        shards_start_address(input_args.shards_start_address),
        shard_size_in_bytes(input_args.shard_size_in_bytes),
        total_chunks_per_core(input_args.total_chunks_per_core),
        curr_worker_index(input_args.starting_core_index),
        curr_core_chunk_index(input_args.starting_chunk_into_shard),

        intra_core_stride_in_shards(input_args.intra_core_stride_in_shards),
        contiguous_chunk_count(1),
        contiguous_chunks_before_stride(input_args.contiguous_chunks_before_stride),
        num_dest_cores(input_args.num_dest_cores),

        num_args_consumed(num_args_consumed),
        is_clockwise(input_args.is_clockwise)
        {
            ASSERT(this->contiguous_chunks_before_stride >= 1);
            ASSERT(this->intra_core_stride_in_shards >= 1);
            ASSERT(input_args.starting_chunk_into_shard <= this->total_chunks_per_core);
        };

    static_assert(
        TYPE == ShardType::Width || TYPE == ShardType::Height || TYPE == ShardType::Block, "Invalid ShardType");

    // Clockwise vs counter clockwise only affects worker core traversal order (relative to canonical order). Since the
    // dest core list is a configurable list, we will, for now, require the host side kernel config code to produce the
    // correc order per worker
    FORCE_INLINE void advance() {
        if constexpr (TYPE == ShardType::Width or TYPE == ShardType::Height) {
            ccl::all_gather::addr_gen_advance_width_sharded(
                this->curr_core_chunk_index,
                this->curr_worker_index,
                this->contiguous_chunk_count,
                // this->current_core_chunks_visited,
                this->total_chunks_per_core,
                this->num_dest_cores,
                this->intra_core_stride_in_shards,
                this->contiguous_chunks_before_stride,
                this->is_clockwise
            );
        } else {
            // Unsupported
            ASSERT(false);
        }
    }

    [[nodiscard]] FORCE_INLINE ccl::WorkerXY get_next_noc_xy_core() const {
        ASSERT(this->curr_worker_index < this->num_dest_cores);
        return this->dest_cores[this->curr_worker_index];
    }

    [[nodiscard]] FORCE_INLINE uint64_t get_next_noc_addr() const {
        ccl::WorkerXY dest_worker = this->get_next_noc_xy_core();
        uint32_t curr_address = this->shards_start_address + this->curr_core_chunk_index * this->shard_size_in_bytes;
        ASSERT(curr_address + this->shard_size_in_bytes <= 1499136); // L1 wraparound - oops!
        ASSERT(this->shards_start_address <= curr_address);
        return get_noc_addr(dest_worker.x, dest_worker.y, curr_address);
    }

    [[nodiscard]] FORCE_INLINE uint64_t get_next_noc_addr_and_advance() {
        if constexpr (TYPE == ShardType::Width) {
            ccl::WorkerXY dest_worker = this->get_next_noc_xy_core();
            uint32_t curr_address = this->shards_start_address + this->curr_core_chunk_index * this->shard_size_in_bytes;
            ASSERT(curr_address + this->shard_size_in_bytes <= 1499136); // L1 wraparound - oops!
            ASSERT(this->shards_start_address <= curr_address);
            this->advance();
            return get_noc_addr(dest_worker.x, dest_worker.y, curr_address);
        } else {
            ASSERT(false);
            // Unsupported
            return 0;
        }
    }

    [[nodiscard]] FORCE_INLINE uint32_t get_shard_size_in_bytes() const { return this->shard_size_in_bytes; }

    [[nodiscard]] FORCE_INLINE uint32_t get_num_dest_cores() const { return this->num_dest_cores; }
    [[nodiscard]] FORCE_INLINE uint32_t get_total_chunks_per_core() const {
        return this->total_chunks_per_core;
    }
    [[nodiscard]] FORCE_INLINE uint32_t get_num_args_consumed() const { return this->num_args_consumed;}

    ccl::WorkerXY* dest_cores;
    uint32_t shards_start_address;
    // This could be shared
    uint32_t shard_size_in_bytes;
    uint16_t total_chunks_per_core;
    // uint16_t current_core_chunks_visited;
    uint16_t curr_worker_index;
    uint16_t curr_core_chunk_index;
    // new fields
    uint16_t intra_core_stride_in_shards;
    uint16_t contiguous_chunk_count;
    uint16_t contiguous_chunks_before_stride;
    ///
    uint16_t num_dest_cores;
    uint8_t num_args_consumed;
    bool is_clockwise;
};

FORCE_INLINE void push_filler_pages_to_cb(const uint32_t& cb_id, uint32_t num_pages) {
    ASSERT(num_pages < cb_interface[cb_id].fifo_num_pages);
    cb_reserve_back(cb_id, num_pages);
    cb_push_back(cb_id, num_pages);
}
FORCE_INLINE void pop_filler_pages_from_cb(const uint32_t& cb_id, uint32_t num_pages) {
    ASSERT(num_pages < cb_interface[cb_id].fifo_num_pages);
    cb_wait_front(cb_id, num_pages);
    cb_pop_front(cb_id, num_pages);
}


FORCE_INLINE void fetch_chunk(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(remote_l1_read_addr, l1_write_addr, page_size * num_pages);
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
FORCE_INLINE void fetch_chunk_sharded(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_read_addr) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    noc_async_read(remote_l1_read_addr, l1_write_addr, num_pages * page_size);
    validate_sane_transaction_counters();
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

FORCE_INLINE void send_chunk(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
FORCE_INLINE void send_chunk_sharded(
    const uint32_t& cb_id, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr, uint64_t eth_l1_sender_semaphore_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
    validate_sane_transaction_counters();
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <ShardType T>
FORCE_INLINE void write_and_send_chunk_sharded(
    const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, uint32_t const num_pages, uint64_t remote_eth_l1_write_addr, uint64_t eth_l1_sender_semaphore_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    uint32_t num_pages_remaining = num_pages;
    noc_async_write(l1_read_addr, remote_eth_l1_write_addr, num_pages * addr_gen.get_shard_size_in_bytes());
    noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
    while (num_pages_remaining > 0) {
        uint64_t dest_worker_noc_addr = addr_gen.get_next_noc_addr();
        uint32_t num_shards_to_write = std::min<uint32_t>(num_pages_remaining, addr_gen.contiguous_chunks_before_stride);
        noc_async_write(l1_read_addr, dest_worker_noc_addr, num_shards_to_write * addr_gen.get_shard_size_in_bytes());
        for (uint32_t i = 0; i < num_shards_to_write; i++) {
            addr_gen.advance();
        }
        num_pages_remaining -= num_shards_to_write;
        l1_read_addr += num_shards_to_write * addr_gen.get_shard_size_in_bytes();
    }
    validate_sane_transaction_counters();
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
template <typename AddrGen>
FORCE_INLINE void write_and_send_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t num_cols, const uint32_t num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size, uint64_t remote_l1_write_addr, uint64_t eth_l1_sender_semaphore_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
    // TODO: do eth semaphore inc here
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
#elif defined TILE_INTERLEAVED || defined SHARDED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
#endif
        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <ShardType T>
FORCE_INLINE void write_chunk_sharded(const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, const uint32_t num_pages) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    uint32_t num_pages_remaining = num_pages;
    while (num_pages_remaining > 0) {
        uint64_t dest_worker_noc_addr = addr_gen.get_next_noc_addr();
        uint32_t num_contiguous_shards = addr_gen.contiguous_chunks_before_stride;
        uint32_t num_to_send = std::min(num_pages_remaining, num_contiguous_shards);
        noc_async_write(l1_read_addr, dest_worker_noc_addr, num_to_send * addr_gen.get_shard_size_in_bytes());
        for (uint32_t i = 0; i < num_to_send; i++) {
            addr_gen.advance();
        }
        l1_read_addr += num_to_send * addr_gen.get_shard_size_in_bytes();
        num_pages_remaining -= num_to_send;
    }
    validate_sane_transaction_counters_rw();
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
template <typename AddrGen>
FORCE_INLINE void write_chunk(uint32_t& output_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& d, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
        #ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
        #endif
        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <ShardType T>
FORCE_INLINE void read_shard_from_input_tensor_sharded(
    const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, uint32_t num_shards) {
    cb_reserve_back(cb_id, num_shards);
    uint32_t local_l1_read_dest_addr = get_write_ptr(cb_id);
    for (uint32_t s = 0; s < num_shards; s++) {
        uint64_t src_noc_addr = addr_gen.get_next_noc_addr_and_advance();
        noc_async_read(src_noc_addr, local_l1_read_dest_addr, addr_gen.get_shard_size_in_bytes());
        local_l1_read_dest_addr += addr_gen.get_shard_size_in_bytes();
    }
    validate_sane_transaction_counters();
    noc_async_read_barrier();
    cb_push_back(cb_id, num_shards);
}
// read chunk from input tensor (local chip)
template <typename AddrGen>
FORCE_INLINE void read_chunk_from_input_tensor(uint32_t& input_page_idx, const uint32_t& cb_id, const AddrGen& s, const uint32_t& num_pages, const uint32_t& page_size) {
    const uint32_t end_read_idx = input_page_idx + num_pages;
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    for (; input_page_idx < end_read_idx; ++input_page_idx) {
        #ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        #elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
        #endif
        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

// Same function - just different address generators? Commonize later
template <ShardType T>
FORCE_INLINE void read_chunk_from_output_tensor_sharded(
    const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, uint32_t const num_pages) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_dest_addr = get_write_ptr(cb_id);
    uint32_t num_pages_remaining = num_pages;
    while (num_pages_remaining > 0) {
        uint64_t src_noc_addr = addr_gen.get_next_noc_addr();
        uint32_t shards_to_read = std::min<uint32_t>(num_pages_remaining, addr_gen.contiguous_chunks_before_stride);
        noc_async_read(src_noc_addr, local_l1_read_dest_addr, shards_to_read * addr_gen.get_shard_size_in_bytes());
        local_l1_read_dest_addr += shards_to_read * addr_gen.get_shard_size_in_bytes();
        for (uint32_t i = 0; i < shards_to_read; i++) {
            addr_gen.advance();
        }
        num_pages_remaining -= shards_to_read;
    }
    validate_sane_transaction_counters();
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
// read chunk from output tensor (local chip)
template <typename AddrGen>
FORCE_INLINE void read_chunk_from_output_tensor(uint32_t& input_page_idx, uint32_t& col_idx, uint32_t& row_idx, const uint32_t& cb_id, const AddrGen& s, const uint32_t& num_cols, const uint32_t& num_rows, const uint32_t& col_offset, const uint32_t& row_offset, const uint32_t& num_pages, const uint32_t& page_size) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
        #ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        input_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            input_page_idx += row_offset;
        }
        #elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
        input_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            input_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                input_page_idx += row_offset;
            }
        }
        #endif
        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
