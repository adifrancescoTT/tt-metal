// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_consumer.hpp"
#include "debug/dprint.h"

// Dispatches fast dispatch commands to worker cores. Currently only runs on remote devices
void kernel_main() {
    constexpr uint32_t cmd_base_address = get_compile_time_arg_val(0);
    constexpr uint32_t data_buffer_size = get_compile_time_arg_val(1);
    constexpr uint32_t router_cmd_base_address = get_compile_time_arg_val(2);
    constexpr uint32_t router_data_buffer_size = get_compile_time_arg_val(3);

    volatile tt_l1_ptr uint32_t* db_rx_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to 0 by host
    volatile tt_l1_ptr uint32_t* db_tx_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));  // Should be num command slots in the router

    uint64_t dispatcher_noc_encoding = uint64_t(NOC_XY_ENCODING(DISPATCHER_NOC_X, DISPATCHER_NOC_Y)) << 32;
    uint64_t signaller_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;
    uint64_t router_noc_encoding = uint64_t(NOC_XY_ENCODING(ROUTER_NOC_X, ROUTER_NOC_Y)) << 32;

    bool db_rx_buf_switch = false;
    bool db_tx_buf_switch = false;
    while (true) {
        // Wait for dispatcher to supply a command
        DPRINT << "remote signaller waiting for data at semaphore addr " << get_semaphore(0) << " value " << db_rx_semaphore_addr[0] << ENDL();
        db_acquire(db_rx_semaphore_addr, dispatcher_noc_encoding);

        // For each instruction, we need to jump to the relevant part of the device command
        uint32_t command_start_addr = get_command_slot_addr<cmd_base_address, data_buffer_size>(db_rx_buf_switch);
        uint32_t buffer_transfer_start_addr = command_start_addr + (DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t));

        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);

        uint32_t num_workers = command_ptr[DeviceCommand::num_workers_idx];  // If num_workers > 0, it means we are launching a program
        uint32_t num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];   // How many WriteBuffer commands we are running
        uint32_t is_program = command_ptr[DeviceCommand::is_program_buffer_idx];
        uint32_t page_size = command_ptr[DeviceCommand::page_size_idx];
        uint32_t consumer_cb_size = command_ptr[DeviceCommand::consumer_cb_size_idx];
        uint32_t consumer_cb_num_pages = command_ptr[DeviceCommand::consumer_cb_num_pages_idx];
        uint32_t num_pages = command_ptr[DeviceCommand::num_pages_idx];
        uint32_t producer_consumer_transfer_num_pages = command_ptr[DeviceCommand::producer_consumer_transfer_num_pages_idx];
        uint32_t sharded_buffer_num_cores = command_ptr[DeviceCommand::sharded_buffer_num_cores_idx];

        DPRINT << "remote dispatcher kernel got:"
            << " " << num_buffer_transfers
            << " " << page_size
            << " " << consumer_cb_size
            << " " << consumer_cb_num_pages
            << " " << num_pages
            << " " << producer_consumer_transfer_num_pages << ENDL();

        while (db_tx_semaphore_addr[0] == 0)
            ;  // Check that there is space in the router
        program_consumer_cb<router_cmd_base_addr, router_data_buffer_size>(db_tx_buf_switch, router_noc_encoding, consumer_cb_num_pages, page_size, consumer_cb_size);
        relay_command<router_cmd_base_addr, router_data_buffer_size>(db_tx_buf_switch, router_noc_encoding);

        // Command requests data to be transferred to host
        // Wait for data from dispatcher and forward it to router



        // notify dispatcher that it has completed transfer to ethernet router
        noc_semaphore_inc(dispatcher_noc_encoding | get_semaphore(0), 1);
        db_rx_buf_switch = not db_rx_buf_switch;
        noc_async_write_barrier(); // Barrier for now
    }
}
