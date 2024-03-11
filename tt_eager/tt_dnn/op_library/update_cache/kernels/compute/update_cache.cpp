// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/tilize.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t cache_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_in_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t num_batched_heads = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);

    untilize_init(in_cb, untilized_in_cb);

    for (uint32_t  h = 0; h < num_batched_heads; ++h) {
        untilize_init_short(in_cb);
        cb_wait_front(in_cb, Wt);
        cb_reserve_back(untilized_in_cb, Wt);
        untilize_block(in_cb, Wt, untilized_in_cb);
        cb_push_back(untilized_in_cb, Wt);
        cb_pop_front(in_cb, Wt);
        untilize_uninit(in_cb);

        // unpack_reconfig_data_format_srca(in_cb, cache_cb);

        // for(uint32_t u = 0; u < 32; ++u) {
        //     untilize_init_short(cache_cb);
        //     cb_wait_front(cache_cb, Wt);
        //     cb_reserve_back(untilized_cache_cb, Wt);
        //     untilize_block(cache_cb, Wt, untilized_cache_cb);
        //     cb_push_back(untilized_cache_cb, Wt);
        //     cb_pop_front(cache_cb, Wt);
        //     untilize_uninit(cache_cb);

        //     unpack_reconfig_data_format_srca(cache_cb, untilized_cache2_cb);
        //     pack_reconfig_data_format(untilized_cache_cb, out_cb);

        //     tilize_init_short(untilized_cache2_cb, Wt);
        //     cb_wait_front(untilized_cache2_cb, Wt);
        //     cb_reserve_back(out_cb, Wt);
        //     tilize_block(untilized_cache2_cb, Wt, out_cb);
        //     cb_push_back(out_cb, Wt);
        //     // Untilized cache CBs share same address space
        //     // Compute pops both
        //     cb_pop_front(untilized_cache2_cb, Wt);
        //     cb_pop_front(untilized_cache_cb, Wt);
        //     tilize_uninit_with_dt(untilized_cache2_cb, cache_cb);
        //     pack_reconfig_data_format(out_cb, untilized_cache_cb);
        // }
        // unpack_reconfig_data_format_srca(cache_cb, in_cb);
    }
}
} // NAMESPACE
