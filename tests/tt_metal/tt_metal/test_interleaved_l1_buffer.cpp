// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

bool test_interleaved_l1_buffer(tt_metal::Device *device, int num_pages_one, int num_pages_two, uint32_t page_size) {
    bool pass = true;

    uint32_t buffer_size = num_pages_one * page_size;

    auto interleaved_buffer = CreateBuffer(device, buffer_size, page_size, tt_metal::BufferStorage::L1);

    std::vector<uint32_t> host_buffer = create_random_vector_of_bfloat16(
        buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    tt_metal::WriteToBuffer(interleaved_buffer, host_buffer);

    std::vector<uint32_t> readback_buffer;
    tt_metal::ReadFromBuffer(interleaved_buffer, readback_buffer);

    pass &= (host_buffer == readback_buffer);

    uint32_t second_buffer_size = num_pages_two * page_size;

    auto second_interleaved_buffer = CreateBuffer(device, second_buffer_size, page_size, tt_metal::BufferStorage::L1);

    std::vector<uint32_t> second_host_buffer = create_random_vector_of_bfloat16(
        second_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    tt_metal::WriteToBuffer(second_interleaved_buffer, second_host_buffer);

    std::vector<uint32_t> second_readback_buffer;
    tt_metal::ReadFromBuffer(second_interleaved_buffer, second_readback_buffer);

    pass &= (second_host_buffer == second_readback_buffer);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    tt::log_assert(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");
    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);

        uint32_t page_size =  2 * 1024;

        int num_bank_pages_one = 258;
        int num_bank_pages_two = 378;



        pass &= test_interleaved_l1_buffer(device, num_bank_pages_one, num_bank_pages_two, page_size);

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
