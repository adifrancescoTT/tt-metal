// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include "dataflow_api.h"

#define DONT_STRIDE_IN_ETH_BUFFER 0

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();

        // eth_wait_for_bytes(16);
        // eth_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_done();

        // eth_send_bytes(handshake_register_address,handshake_register_address, 16);
        // eth_wait_for_receiver_done();
    }
}

template <uint32_t NUM_CHANNELS>
FORCE_INLINE uint8_t get_next_buffer_channel_pointer(uint8_t pointer) {
    if constexpr (NUM_CHANNELS % 2 == 0) {
        constexpr uint8_t CHANNEL_WRAP_MASK = NUM_CHANNELS - 1;
        return pointer = (pointer + 1) & CHANNEL_WRAP_MASK;
    } else {
        pointer = (pointer + 1);
        return pointer == NUM_CHANNELS ? 0 : pointer;
    }
}

FORCE_INLINE bool is_noc_write_in_progress(
    const uint8_t noc_writer_buffer_wrptr, const uint8_t noc_writer_buffer_ackptr) {
    return noc_writer_buffer_wrptr != noc_writer_buffer_ackptr;
}

template<uint32_t MAX_NUM_CHANNELS>
bool eth_receiver_accept_payload_sequence(
    uint8_t noc_writer_buffer_wrptr,
    uint8_t noc_writer_buffer_ackptr,
    uint8_t &eth_receiver_ptr,
    uint8_t &eth_receiver_ackptr) {
    bool did_something = false;
    bool receive_pointers_full = get_next_buffer_channel_pointer<MAX_NUM_CHANNELS>(eth_receiver_ptr) == eth_receiver_ackptr;

    if (!receive_pointers_full) {
        if (eth_bytes_are_available_on_channel(eth_receiver_ptr)) {
            eth_receiver_ptr = get_next_buffer_channel_pointer<MAX_NUM_CHANNELS>(eth_receiver_ptr);
            did_something = true;
        }
    }

    return did_something;
}

// Check if DRAM write is done -> advances ack pointer
template <uint32_t MAX_NUM_CHANNELS, bool dest_is_dram>
FORCE_INLINE bool noc_write_completion_check_sequence(
    uint8_t &noc_writer_buffer_wrptr, uint8_t &noc_writer_buffer_ackptr, const uint8_t noc_index) {
    bool did_something = false;

    bool noc_write_is_in_progress = is_noc_write_in_progress(noc_writer_buffer_wrptr, noc_writer_buffer_ackptr);
    if (noc_write_is_in_progress) {
#if EMULATE_DRAM_READ_CYCLES == 1
        bool write_finished = emulated_dram_write_cycles_finished();
#else
        bool writes_finished = ncrisc_noc_nonposted_writes_sent(noc_index);
#endif
        if (writes_finished) {
            kernel_profiler::mark_time(13);
            noc_writer_buffer_ackptr = get_next_buffer_channel_pointer<MAX_NUM_CHANNELS>(noc_writer_buffer_ackptr);

            did_something = true;
        }
    }

    return did_something;
}

// Initiate DRAM write -> advances  write pointer
template <bool dest_is_dram>
void write_chunk(
    const uint32_t eth_l1_buffer_address_base,
    const uint32_t num_pages,
    const uint32_t num_pages_per_l1_buffer,
    const uint32_t page_size,
    uint32_t &page_index,
    const InterleavedAddrGen<dest_is_dram> &dest_address_generator) {
    uint32_t local_eth_l1_curr_src_addr = eth_l1_buffer_address_base;
    uint32_t end_page_index = std::min(page_index + num_pages_per_l1_buffer, num_pages);
    for (; page_index < end_page_index; ++page_index) {
        // read source address
        uint64_t dest_noc_addr = get_noc_addr(page_index, dest_address_generator);
        noc_async_write(local_eth_l1_curr_src_addr, dest_noc_addr, page_size);
        // read dest addr
        #if DONT_STRIDE_IN_ETH_BUFFER == 0
        local_eth_l1_curr_src_addr += page_size;
        #endif
    }
}
template <uint32_t MAX_NUM_CHANNELS, bool dest_is_dram>
bool eth_initiate_noc_write_sequence(
    std::array<uint32_t, MAX_NUM_CHANNELS> &transaction_channel_receiver_buffer_addresses,
    uint8_t &noc_writer_buffer_wrptr,
    uint8_t &noc_writer_buffer_ackptr,
    const uint8_t eth_receiver_wrptr,
    const uint8_t eth_receiver_ackptr,

    const uint32_t num_pages,
    const uint32_t num_pages_per_l1_buffer,
    const uint32_t page_size,
    uint32_t &page_index,
    const InterleavedAddrGen<dest_is_dram> &dest_address_generator) {
    bool did_something = false;
    bool noc_write_is_in_progress = is_noc_write_in_progress(noc_writer_buffer_wrptr, noc_writer_buffer_ackptr);

    if (!noc_write_is_in_progress) {
        bool next_payload_received = noc_writer_buffer_wrptr != eth_receiver_wrptr;
        if (next_payload_received) {
            // Can initialize a new write if data is at this buffer location (eth num_bytes != 0)
            // and the receiver ackptr != next write pointer
            write_chunk<dest_is_dram>(
                transaction_channel_receiver_buffer_addresses[noc_writer_buffer_wrptr],
                num_pages,
                num_pages_per_l1_buffer,
                page_size,
                page_index,
                dest_address_generator);
            noc_writer_buffer_wrptr = get_next_buffer_channel_pointer<MAX_NUM_CHANNELS>(noc_writer_buffer_wrptr);
            did_something = true;
        }
    }

    return did_something;
}

// Check if eth message is received -> advances receive read pointer

template <uint32_t MAX_CONCURRENT_TRANSACTIONS>
void initialize_transaction_buffer_addresses(
    uint32_t sender_buffer_base_address,
    uint32_t receiver_buffer_base_address,
    uint32_t num_bytes_per_send,
    std::array<uint32_t, MAX_CONCURRENT_TRANSACTIONS> &transaction_channel_remote_buffer_addresses,
    std::array<uint32_t, MAX_CONCURRENT_TRANSACTIONS> &transaction_channel_local_buffer_addresses) {
    uint32_t sender_buffer_address = sender_buffer_base_address;
    uint32_t receiver_buffer_address = receiver_buffer_base_address;
    for (uint32_t i = 0; i < MAX_CONCURRENT_TRANSACTIONS; i++) {
        transaction_channel_remote_buffer_addresses[i] = sender_buffer_address;
        transaction_channel_local_buffer_addresses[i] = receiver_buffer_address;
#if ENABLE_L1_BUFFER_OVERLAP == 0
        sender_buffer_address += num_bytes_per_send;
        receiver_buffer_address += num_bytes_per_send;
#endif
    }
}

template <uint32_t MAX_NUM_CHANNELS>
FORCE_INLINE bool eth_receiver_send_ack_sequence(
    const uint8_t noc_writer_buffer_wrptr,
    const uint8_t noc_writer_buffer_ackptr,
    uint8_t &eth_receiver_rdptr,
    uint8_t &eth_receiver_ackptr,
    uint32_t &num_eth_sends_acked) {
    bool did_something = false;
    bool eth_sends_unacknowledged = eth_receiver_rdptr != eth_receiver_ackptr;
    if (eth_sends_unacknowledged) {
        // If data is done being sent out of this local l1 buffer and to the destination(s),
        // then we can safely send the ack and increment the ackptr
        bool buffer_writes_flushed = ncrisc_noc_nonposted_writes_sent(noc_index);
        // bool buffer_writes_flushed = ncrisc_noc_nonposted_writes_flushed(noc_index);
        if (buffer_writes_flushed) {
            // kernel_profiler::mark_time(15);
            eth_receiver_channel_done(eth_receiver_ackptr);
            num_eth_sends_acked++;
            eth_receiver_ackptr = get_next_buffer_channel_pointer<MAX_NUM_CHANNELS>(eth_receiver_ackptr);
            // DPRINT << "rx: Sending eth ack. ackptr incrementing to " << (uint32_t)eth_receiver_ackptr << "\n";

            did_something = true;
        }
    }

    return did_something;
}



void kernel_main() {
    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);
    constexpr std::uint32_t total_num_message_sends = get_compile_time_arg_val(2);
    constexpr std::uint32_t NUM_TRANSACTION_BUFFERS = get_compile_time_arg_val(3);
    constexpr bool dest_is_dram = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t MAX_NUM_CHANNELS = NUM_TRANSACTION_BUFFERS;
    // Handshake first before timestamping to make sure we aren't measuring any
    // dispatch/setup times for the kernels on both sides of the link.

    const std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    const std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);
    const std::uint32_t dest_addr = get_arg_val<uint32_t>(2);
    const std::uint32_t page_size = get_arg_val<uint32_t>(3);
    const std::uint32_t num_pages = get_arg_val<uint32_t>(4);
    eth_setup_handshake(remote_eth_l1_dst_addr, false);

    const InterleavedAddrGen<dest_is_dram> dest_address_generator = {
        .bank_base_address = dest_addr, .page_size = page_size};

    uint8_t noc_writer_buffer_ackptr = 0;
    uint8_t noc_writer_buffer_wrptr = 0;
    uint8_t eth_receiver_rdptr = 0;
    uint8_t eth_receiver_ackptr = 0;

    kernel_profiler::mark_time(80);

    std::array<uint32_t, MAX_NUM_CHANNELS> transaction_channel_local_buffer_addresses;
    std::array<uint32_t, MAX_NUM_CHANNELS> transaction_channel_remote_buffer_addresses;
    initialize_transaction_buffer_addresses<MAX_NUM_CHANNELS>(
        remote_eth_l1_dst_addr,
        local_eth_l1_src_addr,
        num_bytes_per_send,
        transaction_channel_remote_buffer_addresses,
        transaction_channel_local_buffer_addresses);

    constexpr uint32_t SWITCH_INTERVAL = 100000;
    uint32_t page_index = 0;
    const uint32_t num_pages_per_l1_buffer = num_bytes_per_send / page_size;

    bool write_in_flight = false;
    uint32_t num_eth_sends_acked = 0;
    uint32_t count = 0;
    while (num_eth_sends_acked < total_num_message_sends) {
        bool did_something = false;
        // kernel_profiler::mark_time(90);

        did_something = eth_receiver_accept_payload_sequence<MAX_NUM_CHANNELS>(
                            noc_writer_buffer_wrptr, noc_writer_buffer_ackptr, eth_receiver_rdptr, eth_receiver_ackptr) ||
                        did_something;

        did_something = eth_initiate_noc_write_sequence<MAX_NUM_CHANNELS, dest_is_dram>(
                            transaction_channel_local_buffer_addresses,
                            noc_writer_buffer_wrptr,
                            noc_writer_buffer_ackptr,
                            eth_receiver_rdptr,
                            eth_receiver_ackptr,

                            num_pages,
                            num_pages_per_l1_buffer,
                            page_size,
                            page_index,
                            dest_address_generator) ||
                        did_something;

        did_something = noc_write_completion_check_sequence<MAX_NUM_CHANNELS, dest_is_dram>(
                            noc_writer_buffer_wrptr, noc_writer_buffer_ackptr, noc_index) ||
                        did_something;

        did_something =
            eth_receiver_send_ack_sequence<MAX_NUM_CHANNELS>(
                noc_writer_buffer_wrptr, noc_writer_buffer_ackptr, eth_receiver_rdptr, eth_receiver_ackptr, num_eth_sends_acked) ||
            did_something;

        if (!did_something) {

            if (count++ > SWITCH_INTERVAL) {
                count = 0;
                // kernel_profiler::mark_time(15);
                run_routing();
            } else {
                count++;
            }
        }
    }

    // DPRINT << "rx: done\n";
    kernel_profiler::mark_time(81);

    kernel_profiler::mark_time(100);

    // This helps flush out the "end" timestamp
    // eth_setup_handshake(remote_eth_l1_dst_addr, false);
    // for (int i = 0; i < 30000; i++);
}
