// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
	constexpr uint32_t shard_cb = get_compile_time_arg_val(0);
	constexpr uint32_t config_cb = get_compile_time_arg_val(1);

	const uint32_t input_shard_addr  = get_arg_val<uint32_t>(0);
	const uint32_t num_output_pages = get_arg_val<uint32_t>(1);
	const uint32_t num_ranges = get_arg_val<uint32_t>(2);

	cb_reserve_back(shard_cb, num_output_pages);
	uint32_t config_l1_addr = get_read_ptr(config_cb);
	volatile tt_l1_ptr int* config_addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(config_l1_addr);
	uint32_t l1_write_addr = get_write_ptr(shard_cb);


	uint32_t arg_index = 0;
	for(uint32_t range_id = 0; range_id <num_ranges; range_id++) {
		uint32_t core_id_x = config_addr_ptr[arg_index++];
		uint32_t core_id_y = config_addr_ptr[arg_index++];
		uint32_t offset = config_addr_ptr[arg_index++];
		uint32_t size = config_addr_ptr[arg_index++];
		uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
				input_shard_addr + offset);
		noc_async_read(noc_address, l1_write_addr, size);
		l1_write_addr+=size;

	}
	noc_async_read_barrier();
	cb_push_back(shard_cb, num_output_pages);

}
