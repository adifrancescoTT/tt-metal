// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "tt_metal/common/core_coord.h"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "tensor/tensor_impl.hpp"
#include "tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include <sstream>
#include <type_traits>

using namespace tt::constants;

namespace tt {

namespace tt_metal {


std::tuple<CoreRangeSet,CoreRangeSet> select_worker_cores(AllGatherConfig const& all_gather_config, uint32_t num_links, uint32_t link) {
    constexpr uint32_t worker_grid_width = 8;
    const bool fit_sender_and_receiver_workers_on_same_row = (worker_grid_width / 2) >= all_gather_config.get_num_eth_buffers_per_edm();
    std::set<CoreRange> receiver_worker_cores = {};
    std::set<CoreRange> sender_worker_cores = {};
    uint32_t max_cols = 8;
    uint32_t curr_row = link * (((all_gather_config.get_num_eth_buffers_per_edm() * 2 - 1) / max_cols) + 1);
    uint32_t curr_col = 0;
    for (uint32_t r = 0; r < all_gather_config.get_num_eth_buffers_per_edm(); r++) {
        receiver_worker_cores.insert(CoreRange(CoreCoord(curr_col, curr_row)));
        curr_col ++;
        if (curr_col == max_cols) {
            curr_col = 0;
            curr_row++;
        }
    }
    for (uint32_t s = 0; s < all_gather_config.get_num_eth_buffers_per_edm(); s++) {
        sender_worker_cores.insert(CoreRange(CoreCoord(curr_col, curr_row)));
        curr_col ++;
        if (curr_col == max_cols) {
            curr_col = 0;
            curr_row++;
        }
    }
    return {CoreRangeSet(receiver_worker_cores), CoreRangeSet(sender_worker_cores)};
}

class AllGatherOpTensorConfig {
   public:
    AllGatherOpTensorConfig(Tensor const& input_tensor) :
        df(tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype()))
    {}

   protected:
    uint32_t page_size;
    DataFormat df;
};

class AllGatherOpInterleavedTensorConfig final : public AllGatherOpTensorConfig {
   public:
    AllGatherOpInterleavedTensorConfig(Tensor const& input_tensor) :
        AllGatherOpTensorConfig(input_tensor),
        shard_spec(input_tensor.shard_spec().value()) {
        if (input_tensor.get_layout() == Layout::TILE) {
            this->page_size = tt_metal::detail::TileSize(this->df);
            this->unit_size = page_size;
        } else {
            this->unit_size = shard_spec.shape[1] * input_tensor.element_size();
            this->page_size = input_tensor.get_legacy_shape()[-1] * input_tensor.element_size();
        }
    }

   private:
    uint32_t unit_size;
    ShardSpec const shard_spec;
};

class AllGatherOpShardedTensorConfig final : public AllGatherOpTensorConfig {
   public:
    AllGatherOpShardedTensorConfig(Tensor const& input_tensor) :
        AllGatherOpTensorConfig(input_tensor),
        shard_spec(input_tensor.shard_spec().value()) {
        if (input_tensor.get_layout() == Layout::TILE) {
            this->page_size = tt_metal::detail::TileSize(this->df);
            this->unit_size = page_size;
        } else {
            this->unit_size = shard_spec.shape[1] * input_tensor.element_size();
            this->page_size = input_tensor.get_legacy_shape()[-1] * input_tensor.element_size();
        }
    }

   private:
    uint32_t unit_size;
    ShardSpec const shard_spec;
};

operation::ProgramWithCallbacks all_gather_multi_core_with_workers(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t num_links, const uint32_t ring_size, const uint32_t ring_index, const std::optional<chip_id_t> receiver_device_id, const std::optional<chip_id_t> sender_device_id, all_gather_op::Topology topology) {
    TT_FATAL(!(receiver_device_id == std::nullopt && sender_device_id == std::nullopt), "At least one of receiver_device_id or sender_device_id must be specified");
    ccl::EriscDataMoverBufferSharingMode edm_buffer_sharing_mode = ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;

    tt_metal::Program program{};
    auto const& all_gather_config = AllGatherConfig(input_tensor, output_tensor, dim, ring_size, num_links, topology);

    auto const& sharding_info = ShardedAllGatherConfig(input_tensor, output_tensor, dim);
    bool enable_print = false; // ring_index == 0
    all_gather_config.print();
    if (enable_print) {
    }

    bool is_sharded = input_tensor.is_sharded();

    TT_FATAL(input_tensor.buffer()->page_size() <= all_gather_config.get_eth_buffer_size(), "Page size too large");

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    int32_t shard_size_in_bytes = is_sharded ?
        (input_buffer->page_size() * input_buffer->shard_spec().tensor2d_shape[0] * input_buffer->shard_spec().tensor2d_shape[1]) / input_tensor.shard_spec()->num_cores() :
        -1;
    uint32_t input_page_size = is_sharded ? shard_size_in_bytes : input_buffer->page_size();
    uint32_t output_page_size = is_sharded ? shard_size_in_bytes : output_buffer->page_size();
    if (is_sharded) {
        log_trace(tt::LogOp, "input_buffer->page_size: {}", input_buffer->page_size());
        log_trace(tt::LogOp, "input_buffer->shard_spec().tensor2d_shape[0]: {}", input_buffer->shard_spec().tensor2d_shape[0]);
        log_trace(tt::LogOp, "input_buffer->shard_spec().tensor2d_shape[1]: {}", input_buffer->shard_spec().tensor2d_shape[1]);
    }
    const uint32_t max_buffer_per_chunk = is_sharded ?
        round_down(all_gather_config.get_eth_buffer_size(), shard_size_in_bytes):
        round_down(all_gather_config.get_eth_buffer_size(), input_page_size);
    const uint32_t max_pages_per_chunk = is_sharded ?
        max_buffer_per_chunk / shard_size_in_bytes :
        max_buffer_per_chunk / input_page_size;
    log_trace(tt::LogOp, "shard_size_in_bytes: {}", shard_size_in_bytes);
    log_trace(tt::LogOp, "input_page_size: {}", input_page_size);
    log_trace(tt::LogOp, "max_buffer_per_chunk: {}", max_buffer_per_chunk);
    log_trace(tt::LogOp, "max_pages_per_chunk: {}", max_pages_per_chunk);
    const auto& device = input_tensor.device();
    uint32_t sender_socket_idx = 0;
    uint32_t receiver_socket_idx = 0;
    if (receiver_device_id == sender_device_id) {
        if (ring_index == 0) {
            receiver_socket_idx = 1;
        } else {
            sender_socket_idx = 1;
        }
    }

    const uint32_t num_transfers = ring_size - 1;

    bool rm = input_tensor.get_layout() == Layout::ROW_MAJOR;
    bool width = input_tensor.get_legacy_shape().rank() - 1 == dim;
    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t global_num_workers = all_gather_config.get_num_eth_buffers_per_edm() * num_links;

    std::map<string, string> worker_defines;
    if (rm) {
        worker_defines["RM_INTERLEAVED"] = "1";
    } else {
        worker_defines["TILE_INTERLEAVED"] = "1";
    }

    // number of worker cores is 2x this since there is 1 worker for the sender buffer and 1 worker for the receiver buffer
    uint32_t total_worker_core_pairs_used = num_links * all_gather_config.get_num_eth_buffers_per_edm();
    std::vector<CoreCoord> eth_sender_cores;
    eth_sender_cores.reserve(num_links);
    std::vector<CoreCoord> eth_receiver_cores;
    eth_receiver_cores.reserve(num_links);
    std::vector<KernelHandle> eth_sender_kernels;
    eth_sender_kernels.reserve(num_links);
    std::vector<KernelHandle> eth_receiver_kernels;
    eth_receiver_kernels.reserve(num_links);

    std::vector<CoreRange> worker_sender_cores;
    worker_sender_cores.reserve(num_links);
    std::vector<KernelHandle> worker_reader_sender_kernels;
    worker_reader_sender_kernels.reserve(total_worker_core_pairs_used);
    std::vector<KernelHandle> worker_writer_sender_kernels;
    worker_writer_sender_kernels.reserve(total_worker_core_pairs_used);

    std::vector<CoreRange> worker_receiver_cores;
    worker_receiver_cores.reserve(num_links);
    std::vector<KernelHandle> worker_reader_receiver_kernels;
    worker_reader_receiver_kernels.reserve(total_worker_core_pairs_used);
    std::vector<KernelHandle> worker_writer_receiver_kernels;
    worker_writer_receiver_kernels.reserve(total_worker_core_pairs_used);

    std::vector<CoreCoord> all_worker_sender_cores;
    all_worker_sender_cores.reserve(total_worker_core_pairs_used);
    std::vector<CoreCoord> all_worker_receiver_cores;
    all_worker_receiver_cores.reserve(total_worker_core_pairs_used);

    for (uint32_t l = 0; l < num_links; ++l) {
        // Get the cores for the sender and receiver worker cores
        auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id.value()).at(sender_socket_idx + l);
        eth_sender_cores.push_back(eth_sender_core);
        auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id.value()).at(receiver_socket_idx + l);
        eth_receiver_cores.push_back(eth_receiver_core);
    }

    uint32_t num_input_pages = input_tensor.buffer()->size() / input_page_size;
    uint32_t min_pages_per_link = num_input_pages / num_links;

    std::vector<uint32_t> pages_per_link(num_links, min_pages_per_link);
    for (uint32_t i = 0; i < num_input_pages % min_pages_per_link; ++i) {
        pages_per_link.at(i)++;
    }

    uint32_t num_rows = 0, num_cols = 0, row_offset = 0, col_offset = 0, num_tiles = 0;

    if (rm) {
        num_cols = input_tensor.get_legacy_shape()[-1];
        auto input_shape = input_tensor.get_legacy_shape();
        auto output_shape = output_tensor.get_legacy_shape();
        num_rows = std::accumulate(input_shape.begin()+dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>());
        row_offset = std::accumulate(output_shape.begin()+dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) - num_rows;
    } else {
        num_cols = input_tensor.get_legacy_shape()[-1] / TILE_WIDTH;
        auto input_shape = input_tensor.get_legacy_shape();
        auto output_shape = output_tensor.get_legacy_shape();
        uint32_t num_output_cols = output_tensor.get_legacy_shape()[-1] / TILE_WIDTH;
        num_rows = std::accumulate(input_shape.begin()+dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>()) / TILE_HEIGHT;
        row_offset = (std::accumulate(output_shape.begin()+dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) / TILE_HEIGHT - num_rows) * num_output_cols;
        col_offset = num_output_cols - num_cols;
        num_tiles = num_rows * num_cols;
    }



    uint32_t input_start_page_idx = 0;
    uint32_t output_addr_offset = 0;
    uint32_t col_idx = 0;
    uint32_t row_idx = 0;
    uint32_t output_page_offset = 0;

    if (rm) {
        if (width) {
            output_addr_offset = input_page_size;
        } else {
            output_page_offset = num_rows;
        }
    } else {
        if (width) {
            output_page_offset = num_cols;
        } else {
            output_page_offset = num_tiles;
        }
    }
    uint32_t output_start_page_idx = ring_index * output_page_offset;
    uint32_t output_start_addr_offset = ring_index * output_addr_offset;

    ///
    /// (counter clockwise sender) < ----- (this chip) < ----- (counter-clockwise receiver)
    ///
    /// (clockwise receiver)       ------> (this chip) ------> (clockwise sender)
    /// So clockwise sender and counter-clockwise receiver are on the same core
    //  and counter-clockwise sender and clockwise receiver are on the same corwe

    // Clockwise Direction
    std::vector<uint32_t> link_clockwise_sender_channels_offsets =
        std::vector<uint32_t>(num_links, 0);
    std::vector<uint32_t> link_clockwise_sender_num_channels =
        std::vector<uint32_t>(num_links, all_gather_config.get_num_edm_channels_in_clockwise_direction());
    std::vector<uint32_t> link_clockwise_receiver_num_channels = link_clockwise_sender_num_channels;
    // The clockwise direction's erisc's receiver offsets (i.e. for transfers coming INTO this chip)
    std::vector<uint32_t> link_clockwise_receiver_channels_offsets = link_clockwise_sender_channels_offsets;

    // Counter Clockwise Direction
    std::vector<uint32_t> link_counter_clockwise_sender_channels_offsets =
        std::vector<uint32_t>(num_links, all_gather_config.get_num_edm_channels_in_clockwise_direction());
    // Counter clock-wise buffers start after clockwise buffers in L1
    std::vector<uint32_t> link_counter_clockwise_sender_num_channels =
        std::vector<uint32_t>(num_links, all_gather_config.get_num_edm_channels_in_counter_clockwise_direction());
    std::vector<uint32_t> link_counter_clockwise_receiver_channels_offsets = link_counter_clockwise_sender_channels_offsets;
    std::vector<uint32_t> link_counter_clockwise_receiver_num_channels = link_counter_clockwise_sender_num_channels;

    std::vector<uint32_t> eth_sem_addrs;
    std::vector<uint32_t> eth_buffer_addrs;
    eth_sem_addrs.reserve(all_gather_config.get_num_eth_buffers_per_edm());
    eth_buffer_addrs.reserve(all_gather_config.get_num_eth_buffers_per_edm());

    for (uint32_t b = 0, eth_sem_addr = all_gather_config.get_eth_sems_l1_base_byte_address(), eth_buffer_addr = all_gather_config.get_eth_buffers_l1_base_byte_address(); b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
        eth_sem_addrs.push_back(eth_sem_addr);
        eth_sem_addr += all_gather_config.get_semaphore_size();
        eth_buffer_addrs.push_back(eth_buffer_addr);
        eth_buffer_addr += all_gather_config.get_eth_buffer_size();
    }

    std::vector<EriscDatamoverBuilder> clockwise_edm_builders;
    std::vector<EriscDatamoverBuilder> counter_clockwise_edm_builders;
    for (uint32_t link = 0; link < num_links; link++) {
        std::vector<uint32_t> edm_semaphore_addresses; edm_semaphore_addresses.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        std::vector<uint32_t> edm_buffer_addresses; edm_buffer_addresses.reserve(all_gather_config.get_num_eth_buffers_per_edm());

        uint32_t eth_semaphore_address = all_gather_config.get_eth_sems_l1_base_byte_address();
        uint32_t eth_buffer_address = all_gather_config.get_eth_buffers_l1_base_byte_address();
        for (uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            edm_semaphore_addresses.push_back(eth_semaphore_address);
            edm_buffer_addresses.push_back(eth_buffer_address);

            eth_semaphore_address += all_gather_config.get_semaphore_size();
            eth_buffer_address += all_gather_config.get_eth_buffer_size();
        }

        clockwise_edm_builders.emplace_back(
            all_gather_config, edm_semaphore_addresses, edm_buffer_addresses, edm_buffer_sharing_mode);
        counter_clockwise_edm_builders.emplace_back(
            all_gather_config, edm_semaphore_addresses, edm_buffer_addresses, edm_buffer_sharing_mode);
    }


    for (uint32_t i = 0; i < num_links; ++i) {
        // We can't have overlap between the mcast grid for worker cores for different links since mcasting the semaphore in receiver would corrupt other link semaphores
        // We can have overlap between a link's sender and receiver worker grids if we have the semaphores at different addresses
        auto const& [receiver_workers, sender_workers] = select_worker_cores(all_gather_config, num_links, i);
        uint32_t worker_index = 0;
        uint32_t workers_per_link = all_gather_config.get_num_workers_per_link() / all_gather_config.get_num_eth_buffers_per_edm();

        // Circular Buffer Setup
        uint32_t cb_page_size = is_sharded ? shard_size_in_bytes : input_page_size;
        log_trace(tt::LogOp, "input_page_size: {}", input_page_size);
        uint32_t cb_num_pages = 2 * max_pages_per_chunk;
        log_trace(tt::LogOp, "cb_num_pages: {}", cb_num_pages);
        uint32_t src0_cb_index = CB::c_in0;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
		.set_page_size(src0_cb_index, cb_page_size);
        CBHandle cb_src0_sender_workers = CreateCircularBuffer(program, sender_workers, cb_src0_config);
        CBHandle cb_src0_receiver_workers = CreateCircularBuffer(program, receiver_workers, cb_src0_config);

        // This semaphore is used by the receiver core to tell workers that data is available to read
        auto receiver_worker_semaphore_addr = tt_metal::CreateSemaphore(program, receiver_workers, 0);
        // This semaphore is used by the receiver core to tell the worker sender writer that sender buffer is available to write to
        auto sender_worker_writer_semaphore_addr = tt_metal::CreateSemaphore(program, sender_workers, 0);
        // This semaphore is used by the worker receiver writer to tell the worker sender reader that data has been committed to memory
        // This is currently a running counter of how many chunks were committed since the sender worker never decrements this buffer
        // Potentially avoid overflow by having it actually decrement (using noc atomic inc with value of -1)
        auto sender_worker_reader_semaphore_addr = tt_metal::CreateSemaphore(program, sender_workers, 0);

        auto sender_noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch());
        auto receiver_noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch());

        auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id.value()).at(sender_socket_idx);
        eth_sender_cores.push_back(eth_sender_core);
        auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id.value()).at(receiver_socket_idx);
        eth_receiver_cores.push_back(eth_receiver_core);

        // Rename this the _channel
        std::vector<uint32_t> pages_per_buffer;

        // number of pages that can fit in a single ethernet L1 buffer (not the number of pages sent to this channel)
        std::vector<uint32_t> pages_per_eth_l1_buffer;
        pages_per_buffer.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        // std::cout << "all_gather_config.get_eth_buffer_size()=" << all_gather_config.get_eth_buffer_size() << std::endl;
        // std::cout << "input_tensor.buffer()->page_size()=" << input_tensor.buffer()->page_size() << std::endl;
        uint32_t max_pages_per_eth_l1_sender_buffer = all_gather_config.get_eth_buffer_size() / input_page_size;
        for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            pages_per_buffer.push_back((pages_per_link.at(i) / all_gather_config.get_num_eth_buffers_per_edm()));
            pages_per_eth_l1_buffer.push_back(
                is_sharded ? std::min(pages_per_buffer.back(), max_pages_per_eth_l1_sender_buffer)
                           : max_pages_per_eth_l1_sender_buffer);
            if (b < pages_per_link.at(i) % all_gather_config.get_num_eth_buffers_per_edm()) {
                pages_per_buffer.back()++;
            }

            log_trace(tt::LogOp, "pages_per_link[{}]: {}", i, pages_per_link.at(i));
            log_trace(tt::LogOp, "pages_per_buffer[{}]: {}", b, pages_per_buffer.at(b));
            log_trace(tt::LogOp, "max_pages_per_eth_l1_sender_buffer: {}",max_pages_per_eth_l1_sender_buffer);
        }
        TT_ASSERT(std::accumulate(pages_per_buffer.begin(), pages_per_buffer.end(), 0) == pages_per_link.at(i));

        uint32_t bytes_per_chunk = 0, pages_per_chunk = 0, num_full_chunks = 0, rem_bytes = 0, rem_pages = 0;
        uint32_t link_size_bytes = pages_per_link.at(i) * input_page_size;
        if (pages_per_link.at(i) >= max_pages_per_chunk) {
            bytes_per_chunk = max_buffer_per_chunk;
            pages_per_chunk = max_pages_per_chunk;
            TT_ASSERT(max_buffer_per_chunk == max_pages_per_chunk * input_page_size);
            num_full_chunks = link_size_bytes / bytes_per_chunk;
            rem_bytes = link_size_bytes % bytes_per_chunk;
            rem_pages = pages_per_link.at(i) % max_pages_per_chunk;
        } else {
            rem_bytes = link_size_bytes;
            rem_pages = pages_per_link.at(i);
        }

        auto sender_worker_cores = corerange_to_cores(sender_workers, std::nullopt, true);
        auto receiver_worker_cores = corerange_to_cores(receiver_workers, std::nullopt, true);
        all_worker_sender_cores.insert(all_worker_sender_cores.end(), sender_worker_cores.begin(), sender_worker_cores.end());
        all_worker_receiver_cores.insert(all_worker_receiver_cores.end(), receiver_worker_cores.begin(), receiver_worker_cores.end());

        TT_ASSERT(rem_pages < pages_per_chunk || num_full_chunks == 0);
        TT_ASSERT(rem_pages <= max_pages_per_chunk);
        std::vector<uint32_t> num_full_chunks_per_worker(all_gather_config.get_num_eth_buffers_per_edm(), num_full_chunks / all_gather_config.get_num_eth_buffers_per_edm());
        std::vector<uint32_t> rem_pages_per_worker(all_gather_config.get_num_eth_buffers_per_edm(), 0);
        {
            uint32_t worker_idx = 0;
            for (worker_idx = 0; worker_idx < num_full_chunks % all_gather_config.get_num_eth_buffers_per_edm(); ++worker_idx) {
                num_full_chunks_per_worker.at(worker_idx)++;
            }
            if (rem_pages != 0) {
                rem_pages_per_worker.at(worker_idx % all_gather_config.get_num_eth_buffers_per_edm()) = rem_pages;
                TT_ASSERT(rem_pages_per_worker.at(worker_idx % all_gather_config.get_num_eth_buffers_per_edm()) * 2 <= cb_num_pages);
            }
        }

        std::vector<uint32_t> link_buffer_num_messages_to_send;
        std::vector<uint32_t> edm_semaphores_base_address;
        std::vector<uint32_t> link_buffer_sender_addresses;
        link_buffer_num_messages_to_send.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        edm_semaphores_base_address.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        link_buffer_sender_addresses.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        if (is_sharded) {
            for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
                auto input_tensor_shard_arg_generator = InputTensorShardAddrGenArgGenerator(
                                device,
                                input_tensor,
                                ring_index,
                                ring_size,
                                global_num_workers,
                                b + i * all_gather_config.get_num_workers_per_link(),
                                0,
                                0,
                                // We want the input tensor to always be read in forward shard order so we
                                // always tell it we are in counter-clockwise direction (forward read order)
                                false
                            );
                uint32_t max_shards_per_eth_buffer = std::min<uint32_t>(all_gather_config.get_eth_buffer_size() / input_tensor_shard_arg_generator.args_struct.shard_size_in_bytes, input_tensor_shard_arg_generator.args_struct.num_dest_cores);
                TT_ASSERT(max_shards_per_eth_buffer > 0, "Codepath needs further generalization to support computing multiple sends per shard");
                num_full_chunks_per_worker.at(b) = input_tensor_shard_arg_generator.args_struct.num_dest_cores < max_shards_per_eth_buffer ? 1 : input_tensor_shard_arg_generator.args_struct.num_dest_cores / max_shards_per_eth_buffer;
                rem_pages_per_worker.at(b) = max_shards_per_eth_buffer > input_tensor_shard_arg_generator.args_struct.num_dest_cores ? 0 : input_tensor_shard_arg_generator.args_struct.num_dest_cores - (num_full_chunks_per_worker.at(b) * max_shards_per_eth_buffer);
                TT_ASSERT(rem_pages_per_worker.at(b) == 0 || input_tensor_shard_arg_generator.args_struct.num_dest_cores >= num_full_chunks_per_worker.at(b) * max_shards_per_eth_buffer);
                TT_ASSERT(input_tensor_shard_arg_generator.args_struct.num_dest_cores == rem_pages_per_worker.at(b) + num_full_chunks_per_worker.at(b) * max_shards_per_eth_buffer);
            }
        }
        for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            // link num messages
            link_buffer_num_messages_to_send.push_back(
                (num_full_chunks_per_worker.at(b) + (rem_pages_per_worker.at(b) > 0 ? 1 : 0)) *
                num_transfers);
            edm_semaphores_base_address.push_back(all_gather_config.get_eth_sems_l1_base_byte_address() + b * all_gather_config.get_semaphore_size());
            link_buffer_sender_addresses.push_back(all_gather_config.get_eth_buffers_l1_base_byte_address() + b * all_gather_config.get_eth_buffer_size());
        }
        for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            log_trace(tt::LogOp, "rem_pages_per_worker[{}]: {}", b, rem_pages_per_worker.at(b));
            log_trace(tt::LogOp, "num_full_chunks_per_worker[{}]: {}", b, num_full_chunks_per_worker.at(b));
            log_trace(tt::LogOp, "link_buffer_num_messages_to_send[{}]: {}", b, link_buffer_num_messages_to_send.at(b));
        }

        std::vector<uint32_t> link_buffer_receiver_num_messages_to_send;
        std::vector<uint32_t> receiver_semaphores_base_address;
        std::vector<uint32_t> link_buffer_receiver_addresses;
        link_buffer_receiver_num_messages_to_send.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        receiver_semaphores_base_address.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        link_buffer_receiver_addresses.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            link_buffer_receiver_num_messages_to_send.push_back(
                (num_full_chunks_per_worker.at(b) + (rem_pages_per_worker.at(b) > 0 ? 1 : 0)) *
                num_transfers);
            receiver_semaphores_base_address.push_back(all_gather_config.get_eth_sems_l1_base_byte_address() + b * all_gather_config.get_semaphore_size());
            link_buffer_receiver_addresses.push_back(all_gather_config.get_eth_buffers_l1_base_byte_address() + b * all_gather_config.get_eth_buffer_size());
        }

        for (uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            uint32_t num_workers_per_eth_buffer = std::min(workers_per_link, all_gather_config.get_num_eth_buffers_per_edm() - worker_index);

            std::vector<ccl::WorkerXY> sender_worker_coords;
            std::vector<ccl::WorkerXY> receiver_worker_coords;
            for (uint32_t w = b * num_workers_per_eth_buffer; w < (b + 1) * num_workers_per_eth_buffer; ++w) {
                sender_worker_coords.push_back(
                    ccl::WorkerXY(
                        device->worker_core_from_logical_core(sender_worker_cores.at(w)).x,
                        device->worker_core_from_logical_core(sender_worker_cores.at(w)).y));
                receiver_worker_coords.push_back(
                    ccl::WorkerXY(
                        device->worker_core_from_logical_core(receiver_worker_cores.at(w)).x,
                        device->worker_core_from_logical_core(receiver_worker_cores.at(w)).y));
            }

            auto &sender_edm_builder = all_gather_config.is_buffer_in_clockwise_ring(b) ? clockwise_edm_builders.at(i) : counter_clockwise_edm_builders.at(i);
            EriscDatamoverBuilder::ChannelBufferInterface const& sender_channel_buffer_info =
                sender_edm_builder.add_sender_channel(sender_worker_writer_semaphore_addr, link_buffer_num_messages_to_send.at(b), sender_worker_coords);

            auto &receiver_edm_builder = all_gather_config.is_buffer_in_clockwise_ring(b) ? counter_clockwise_edm_builders.at(i) : clockwise_edm_builders.at(i);
            EriscDatamoverBuilder::ChannelBufferInterface const& receiver_channel_buffer_info =
                receiver_edm_builder.add_receiver_channel(receiver_worker_semaphore_addr, link_buffer_num_messages_to_send.at(b), receiver_worker_coords);
        }


        std::vector<uint32_t> const& edm_clockwise_kernel_rt_args = clockwise_edm_builders.at(i).emit_runtime_args();
        std::vector<uint32_t> const& edm_counter_clockwise_kernel_rt_args = counter_clockwise_edm_builders.at(i).emit_runtime_args();

        log_trace(tt::LogOp, "EDM CLOCKWISE KERNEL RT ARGS: ");
        clockwise_edm_builders.at(i).dump_to_log();

        log_trace(tt::LogOp, "EDM COUNTER CLOCKWISE KERNEL RT ARGS: ");
        counter_clockwise_edm_builders.at(i).dump_to_log();

        // 1 Worker per buffer
        for (uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            uint32_t global_worker_index = all_gather_config.get_num_eth_buffers_per_edm() * i + b;

            bool is_clockwise_direction = all_gather_config.is_buffer_in_clockwise_ring(b);
            TT_ASSERT(is_clockwise_direction && receiver_device_id != std::nullopt || !is_clockwise_direction && sender_device_id != std::nullopt);

            // Not fully sure about these two
            uint32_t last_output_page_offset = (num_transfers) * output_page_offset;
            uint32_t last_output_addr_offset = (num_transfers) * output_addr_offset;
            uint32_t receiver_ring_index = is_clockwise_direction ?
                (ring_index == 0 ? ring_size - 1 : ring_index - 1):
                (ring_index == ring_size - 1 ? 0 : ring_index + 1);

            uint32_t receiver_output_start_addr_offset = receiver_ring_index * output_addr_offset;

            uint32_t receiver_output_start_page_idx = output_start_page_idx;
            if (is_clockwise_direction) {
                bool is_wraparound_ring_index = ring_index == 0;
                if (is_wraparound_ring_index) {
                    receiver_output_start_page_idx += last_output_page_offset;
                } else {
                    receiver_output_start_page_idx -= output_page_offset;
                }
            } else {
                // counter clockwise direction
                bool is_wraparound_ring_index = ring_index == ring_size - 1;
                if (is_wraparound_ring_index) {
                    receiver_output_start_page_idx -= last_output_page_offset;
                } else {
                    receiver_output_start_page_idx += output_page_offset;
                }
            }

            log_trace(tt::LogOp,"Counter Clock-wise");
            log_trace(tt::LogOp,"\tlast_output_page_offset={}", last_output_page_offset);
            log_trace(tt::LogOp,"\tlast_output_addr_offset={}", last_output_addr_offset);
            log_trace(tt::LogOp,"\treceiver_ring_index={}", receiver_ring_index);
            log_trace(tt::LogOp,"\treceiver_output_start_addr_offset={}", receiver_output_start_addr_offset);
            log_trace(tt::LogOp,"\treceiver_output_start_page_idx={}", receiver_output_start_page_idx);

            // Sender Worker Kernels
            log_trace(tt::LogOp, "HOST RWS ARGS: ");
            log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
            log_trace(tt::LogOp, "\toutput_page_size: {}", output_page_size);
            log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer.at(b): {}", pages_per_eth_l1_buffer.at(b));
            log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));

            //// Send Reader
            auto build_worker_send_reader_ct_args = [&]() {
                if (is_sharded) {
                    // # Send Reader (CT)
                    // 1) Shard Type
                    // 2) num_transfers
                    std::vector<uint32_t> worker_reader_sender_ct_args = {
                        static_cast<uint32_t>(sharding_info.get_shard_type()),
                        static_cast<uint32_t>(num_transfers)
                    };
                    log_trace(tt::LogOp, "----worker_reader_sender_ct_args size={}", worker_reader_sender_ct_args.size());
                    log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());
                    log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);

                    return worker_reader_sender_ct_args;
                } else {
                    std::vector<uint32_t> worker_reader_sender_ct_args = {
                        static_cast<uint32_t>(all_gather_config.is_input_dram()),
                        static_cast<uint32_t>(all_gather_config.is_output_dram()),
                        static_cast<uint32_t>(num_transfers),
                        static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                        static_cast<uint32_t>(input_page_size),
                        static_cast<uint32_t>(output_page_size),
                        static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),
                        static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                        static_cast<uint32_t>(input_start_page_idx),
                        static_cast<uint32_t>(output_start_page_idx),
                        static_cast<uint32_t>(output_start_addr_offset),
                        static_cast<uint32_t>(row_idx),
                        static_cast<uint32_t>(col_idx),
                        static_cast<uint32_t>(row_offset),
                        static_cast<uint32_t>(col_offset),
                        static_cast<uint32_t>(num_rows),
                        static_cast<uint32_t>(num_cols),
                        static_cast<uint32_t>(last_output_page_offset),
                        static_cast<uint32_t>(output_page_offset),
                        static_cast<uint32_t>(last_output_addr_offset),
                        static_cast<uint32_t>(output_addr_offset),
                        static_cast<uint32_t>(ring_index),
                        static_cast<uint32_t>(sender_worker_reader_semaphore_addr),
                        static_cast<uint32_t>(is_clockwise_direction ? 1 : 0),
                        static_cast<uint32_t>(cb_num_pages / 2)
                    };
                    return worker_reader_sender_ct_args;
                }
            };

            std::vector<uint32_t> const& worker_send_reader_ct_args = build_worker_send_reader_ct_args();

            auto build_worker_send_reader_rt_args = [&]() {
                bool is_clockwise = all_gather_config.is_buffer_in_clockwise_ring(b);
                if (is_sharded) {
                    // # Send Reader (RT)
                    // 1) local semaphore address (same as before)
                    // 2) input tensor shard reader
                    // 3) output tensor shard reader
                    auto curr_link = i;

                    TT_ASSERT(all_gather_config.get_num_eth_buffers_per_edm() == 1 || all_gather_config.get_num_eth_buffers_per_edm() == 2 || all_gather_config.get_num_eth_buffers_per_edm() == 4 || all_gather_config.get_num_eth_buffers_per_edm() == 8);
                    TT_ASSERT(input_tensor.buffer() != nullptr);
                    auto input_tensor_shard_arg_generator =
                        InputTensorShardAddrGenArgGenerator(
                            device,
                            input_tensor,
                            ring_index,
                            ring_size,
                            global_num_workers,
                            global_worker_index,
                            0,
                            0,
                            // We want the input tensor to always be read in forward shard order so we
                            // always tell it we are in counter-clockwise direction (forward read order)
                            false
                            );
                    auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config, input_tensor, output_tensor,
                        is_clockwise ?
                            (ring_index == 0 ? ring_size - 1 : ring_index - 1) :
                            (ring_index == ring_size - 1 ? 0 : ring_index + 1),
                        global_worker_index);

                    log_trace(tt::LogOp, "SendReader {} ring_index: {}, start dest worker index: {}, starting chunk into shard: {}", global_worker_index, ring_index, starting_dest_worker_index, starting_chunk_into_shard);
                    auto output_tensor_shard_arg_generator =
                        OutputTensorShardAddrGenArgGenerator(
                            all_gather_config,
                            device,
                            input_tensor,
                            output_tensor,
                            ring_index,
                            ring_size,
                            global_num_workers,
                            global_worker_index,
                            starting_dest_worker_index,
                            starting_chunk_into_shard,
                            is_clockwise);

                    auto const& input_shard_addr_generator_args = input_tensor_shard_arg_generator.generate();
                    auto const& output_shard_addr_generator_args = output_tensor_shard_arg_generator.generate();
                    std::vector<uint32_t> worker_send_reader_rt_args;
                    worker_send_reader_rt_args.reserve(2 + input_shard_addr_generator_args.size() + output_shard_addr_generator_args.size());
                    worker_send_reader_rt_args.push_back(sender_worker_reader_semaphore_addr);
                    worker_send_reader_rt_args.push_back(pages_per_buffer.at(b));
                    worker_send_reader_rt_args.push_back(pages_per_eth_l1_buffer.at(b));
                    worker_send_reader_rt_args.push_back(cb_num_pages / 2);
                    std::copy(input_shard_addr_generator_args.begin(), input_shard_addr_generator_args.end(), std::back_inserter(worker_send_reader_rt_args));
                    std::copy(output_shard_addr_generator_args.begin(), output_shard_addr_generator_args.end(), std::back_inserter(worker_send_reader_rt_args));

                    log_trace(tt::LogOp, "---worker_send_reader_rt_args.size()={}-----", worker_send_reader_rt_args.size());
                    log_trace(tt::LogOp, "\tsender_worker_reader_semaphore_addr: {}", sender_worker_reader_semaphore_addr);
                    log_trace(tt::LogOp, "\tinput_shard_addr_generator_args:");
                    input_tensor_shard_arg_generator.dump_to_log();
                    log_trace(tt::LogOp, "\toutput_tensor_shard_arg_generator:");
                    output_tensor_shard_arg_generator.dump_to_log();

                    return worker_send_reader_rt_args;
                } else {
                    std::vector<uint32_t> args = {
                        static_cast<uint32_t>(input_buffer->address()),
                        static_cast<uint32_t>(output_buffer->address())
                    };
                    return args;
                }
            };
            std::vector<uint32_t> const& worker_send_reader_rt_args = build_worker_send_reader_rt_args();

            std::string const& send_reader_kernel_path = is_sharded ?
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_send_reader.cpp" :
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_send_reader.cpp";
            KernelHandle worker_reader_sender_kernel_id = tt_metal::CreateKernel(
                program,
                send_reader_kernel_path,
                sender_worker_cores.at(b),
                tt_metal::ReaderDataMovementConfig(worker_send_reader_ct_args, worker_defines));

            worker_reader_sender_kernels.push_back(worker_reader_sender_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_reader_sender_kernel_id,
                sender_worker_cores.at(b),
                worker_send_reader_rt_args);


            //// Send Writer
            auto build_worker_sender_writer_ct_args = [&]() {
                if (is_sharded) {
                    std::vector<uint32_t> worker_sender_writer_ct_args = {
                        static_cast<uint32_t>(sharding_info.get_shard_type())
                    };
                    log_trace(tt::LogOp, "----worker_sender_writer_ct_args size={}", worker_sender_writer_ct_args.size());
                    log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());

                    return worker_sender_writer_ct_args;
                } else {
                    CoreCoord const& worker_eth_sender_core = is_clockwise_direction ? eth_sender_cores.at(i) : eth_receiver_cores.at(i);
                    std::vector<uint32_t> worker_writer_sender_ct_args = {
                        static_cast<uint32_t>(all_gather_config.is_output_dram()),
                        static_cast<uint32_t>(num_transfers),
                        static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                        static_cast<uint32_t>(input_page_size),
                        static_cast<uint32_t>(output_page_size),
                        static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),
                        static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                        static_cast<uint32_t>(input_start_page_idx),
                        static_cast<uint32_t>(output_start_page_idx),
                        static_cast<uint32_t>(output_start_addr_offset),
                        static_cast<uint32_t>(row_idx),
                        static_cast<uint32_t>(col_idx),
                        static_cast<uint32_t>(row_offset),
                        static_cast<uint32_t>(col_offset),
                        static_cast<uint32_t>(num_rows),
                        static_cast<uint32_t>(num_cols),
                        static_cast<uint32_t>(ring_index),

                        // worker local L1 address of semaphore
                        static_cast<uint32_t>(sender_worker_writer_semaphore_addr),
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).x),
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).y),
                        static_cast<uint32_t>(cb_num_pages / 2),
                    };

                    log_trace(tt::LogOp, "HOST SWS ARGS:");
                    log_trace(tt::LogOp, "\toutput_is_dram: {}", all_gather_config.is_output_dram());
                    log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
                    log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
                    log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
                    log_trace(tt::LogOp, "\toutput_page_size: {}", output_page_size);
                    log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer.at(b): {}", pages_per_eth_l1_buffer.at(b));
                    log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));

                    return worker_writer_sender_ct_args;
                }
            };

            std::vector<uint32_t> const& worker_sender_writer_ct_args = build_worker_sender_writer_ct_args();

            auto build_worker_sender_writer_rt_args = [&]() {
                if (is_sharded) {
                    // Send Writer Args (RT)
                    // 1) eth_sender_l1_base_addr
                    // 2) eth_sender_l1_sem_addr
                    // 3) eth_sender_noc_x
                    // 4) eth_sender_noc_y
                    // 5) writer_send_sem_addr
                    // 6) num_transfers
                    // 7)

                    bool is_clockwise = all_gather_config.is_buffer_in_clockwise_ring(b);
                    auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config,
                        input_tensor,
                        output_tensor,
                        // this writes the input tensor to the first output location
                        ring_index,
                        global_worker_index);
                    auto input_tensor_shard_arg_generator = InputTensorShardAddrGenArgGenerator(
                            device,
                            input_tensor,
                            ring_index,
                            ring_size,
                            global_num_workers,
                            global_worker_index,
                            0,
                            0,
                            // We want the input tensor to always be read in forward shard order so we
                            // always tell it we are in counter-clockwise direction (forward read order)
                            false
                        );
                    auto output_tensor_shard_arg_generator =
                        OutputTensorShardAddrGenArgGenerator(
                            all_gather_config,
                            device,
                            input_tensor,
                            output_tensor,
                            ring_index,
                            ring_size,
                            global_num_workers,
                            global_worker_index,
                            starting_dest_worker_index,
                            starting_chunk_into_shard,
                            all_gather_config.is_buffer_in_clockwise_ring(b)
                        );
                    auto const& output_tensor_shard_addr_gen_args = output_tensor_shard_arg_generator.generate();

                    CoreCoord const& worker_eth_sender_core = is_clockwise_direction ? eth_sender_cores.at(i) : eth_receiver_cores.at(i);
                    std::vector<uint32_t> worker_writer_sender_rt_args = {
                        static_cast<uint32_t>(eth_buffer_addrs.at(b)), // eth_sender_l1_base_addr
                        static_cast<uint32_t>(eth_sem_addrs.at(b)), // eth_sender_l1_sem_addr
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).x), // eth_sender_noc_x
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_sender_core).y), // eth_sender_noc_y
                        static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)), //output_tensor_shard_arg_generator.args_struct.num_dest_cores),//pages_per_eth_l1_buffer.at(b)),
                        static_cast<uint32_t>(sender_worker_writer_semaphore_addr), // writer_send_sem_addr
                        static_cast<uint32_t>(num_transfers), // num_transfers
                        static_cast<uint32_t>(input_tensor_shard_arg_generator.args_struct.num_dest_cores),
                        static_cast<uint32_t>(cb_num_pages / 2),
                    };
                    std::copy(output_tensor_shard_addr_gen_args.begin(), output_tensor_shard_addr_gen_args.end(), std::back_inserter(worker_writer_sender_rt_args));

                    // Currently the kernel assumes we don't need to break up the initial local tensor send to EDM into multiple
                    // chunks

                    log_trace(tt::LogOp, "----worker_writer_sender_rt_args size={}", worker_writer_sender_rt_args.size());
                    log_trace(tt::LogOp, "\teth_sender_l1_base_addr: {}", eth_buffer_addrs.at(b));
                    log_trace(tt::LogOp, "\teth_sender_l1_sem_addr: {}", eth_sem_addrs.at(b));
                    log_trace(tt::LogOp, "\teth_sender_noc_x: {}", device->ethernet_core_from_logical_core(worker_eth_sender_core).x);
                    log_trace(tt::LogOp, "\teth_sender_noc_y: {}", device->ethernet_core_from_logical_core(worker_eth_sender_core).y);
                    log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer: {}", pages_per_eth_l1_buffer.at(b));
                    log_trace(tt::LogOp, "\twriter_send_sem_addr: {}", sender_worker_writer_semaphore_addr);
                    log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
                    output_tensor_shard_arg_generator.dump_to_log();

                    return worker_writer_sender_rt_args;
                } else {
                    std::vector<uint32_t> worker_writer_sender_rt_args = {
                        static_cast<uint32_t>(output_buffer->address()),
                        static_cast<uint32_t>(eth_buffer_addrs.at(b)),
                        static_cast<uint32_t>(eth_sem_addrs.at(b))
                    };
                    return worker_writer_sender_rt_args;
                }
            };
            std::vector<uint32_t> const& worker_sender_writer_rt_args = build_worker_sender_writer_rt_args();

            std::string const& sender_writer_kernel_path = is_sharded ?
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_send_writer.cpp" :
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_send_writer.cpp";
            KernelHandle worker_sender_writer_kernel_id = tt_metal::CreateKernel(
                program,
                sender_writer_kernel_path,
                sender_worker_cores.at(b),
                tt_metal::WriterDataMovementConfig(worker_sender_writer_ct_args, worker_defines));

            worker_writer_sender_kernels.push_back(worker_sender_writer_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_sender_writer_kernel_id,
                sender_worker_cores.at(b),
                worker_sender_writer_rt_args);

            log_trace(tt::LogOp, "HOST RWR ARGS:");
            log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
            log_trace(tt::LogOp, "\tpages_per_chunk: {}", pages_per_chunk);
            log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));

            //// Receive Reader
            auto build_worker_receiver_reader_ct_args = [&]() {
                if (is_sharded) {
                    // Receiver Reader Args (CT)
                    std::vector<uint32_t> worker_receiver_reader_ct_args = {
                        static_cast<uint32_t>(sharding_info.get_shard_type())
                    };
                    log_trace(tt::LogOp, "----worker_receiver_reader_ct_args size={}", worker_receiver_reader_ct_args.size());
                    log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());

                    return worker_receiver_reader_ct_args;
                } else {
                    CoreCoord const& worker_eth_receiver_core = is_clockwise_direction ? eth_receiver_cores.at(i) : eth_sender_cores.at(i);
                    std::vector<uint32_t> worker_receiver_reader_ct_args = {
                        static_cast<uint32_t>(num_transfers),
                        static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                        static_cast<uint32_t>(input_page_size),
                        static_cast<uint32_t>(pages_per_chunk),
                        static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).x),
                        static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).y),
                        static_cast<uint32_t>(eth_sem_addrs.at(b)),
                        static_cast<uint32_t>(receiver_worker_semaphore_addr),
                        static_cast<uint32_t>(cb_num_pages / 2)
                    };
                    return worker_receiver_reader_ct_args;
                }
            };
            std::vector<uint32_t> const& worker_receiver_reader_ct_args = build_worker_receiver_reader_ct_args();

            auto build_worker_receiver_reader_rt_args = [&]() {
                if (is_sharded) {
                    // Receiver Reader Args (RT)
                    // 1) eth_receiver_noc_x
                    // 2) eth_receiver_noc_y
                    // 3) eth_receiver_l1_base_addr
                    // 4) eth_receiver_l1_semaphore_addr
                    // 5) (local) receiver_read_sem_addr
                    // 6) output tensor shard addr gen
                    auto curr_link = i;
                    bool is_clockwise = all_gather_config.is_buffer_in_clockwise_ring(b);
                    auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config,
                        input_tensor,
                        output_tensor,
                        is_clockwise ?
                            (ring_index == 0 ? ring_size - 1 : ring_index - 1) :
                            (ring_index == ring_size - 1 ? 0 : ring_index + 1),
                        global_worker_index);
                    CoreCoord const& worker_eth_receiver_core = is_clockwise_direction ? eth_receiver_cores.at(i) : eth_sender_cores.at(i);
                    auto input_tensor_shard_arg_generator = InputTensorShardAddrGenArgGenerator(
                            device,
                            input_tensor,
                            ring_index,
                            ring_size,
                            global_num_workers,
                            global_worker_index,
                            0,
                            0,
                            // We want the input tensor to always be read in forward shard order so we
                            // always tell it we are in counter-clockwise direction (forward read order)
                            false
                        );
                    auto const& output_tensor_shard_addr_gen_args = input_tensor_shard_arg_generator.generate();
                    std::vector<uint32_t> worker_reader_receiver_rt_args;
                    worker_reader_receiver_rt_args.reserve(7 + output_tensor_shard_addr_gen_args.size());

                    worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).x)); // eth_receiver_noc_x
                    worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).y)); // eth_receiver_noc_y
                    worker_reader_receiver_rt_args.push_back(eth_buffer_addrs.at(b)); // eth_receiver_l1_base_addr
                    worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(eth_sem_addrs.at(b))); // eth_receiver_l1_semaphore_addr
                    worker_reader_receiver_rt_args.push_back(receiver_worker_semaphore_addr); // local_receiver_read_sem_addr
                    worker_reader_receiver_rt_args.push_back(pages_per_eth_l1_buffer.at(b)), //output_tensor_shard_arg_generator.args_struct.num_dest_cores), //pages_per_eth_l1_buffer.at(b)); // num_shards_per_eth_buf
                    worker_reader_receiver_rt_args.push_back(num_transfers); // local_receiver_read_sem_addr
                    worker_reader_receiver_rt_args.push_back(static_cast<uint32_t>(cb_num_pages / 2)); // local_receiver_read_sem_addr
                    std::copy(output_tensor_shard_addr_gen_args.begin(), output_tensor_shard_addr_gen_args.end(), std::back_inserter(worker_reader_receiver_rt_args));

                    log_trace(tt::LogOp, "----worker_receiver_reader_ct_args size={}", worker_receiver_reader_ct_args.size());
                    log_trace(tt::LogOp, "\teth_receiver_noc_x: {}", static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).x));
                    log_trace(tt::LogOp, "\teth_receiver_noc_y: {}", static_cast<uint32_t>(device->ethernet_core_from_logical_core(worker_eth_receiver_core).y));
                    log_trace(tt::LogOp, "\teth_receiver_l1_base_addr: {}", eth_buffer_addrs.at(b));
                    log_trace(tt::LogOp, "\teth_receiver_l1_semaphore_addr: {}", static_cast<uint32_t>(eth_sem_addrs.at(b)));
                    log_trace(tt::LogOp, "\tlocal_receiver_read_sem_addr: {}", receiver_worker_semaphore_addr);
                    log_trace(tt::LogOp, "\tnum_shards_per_eth_buf: {}", pages_per_eth_l1_buffer.at(b));

                    input_tensor_shard_arg_generator.dump_to_log();

                    return worker_reader_receiver_rt_args;
                } else {
                    std::vector<uint32_t> worker_reader_receiver_rt_args = {
                        static_cast<uint32_t>(eth_buffer_addrs.at(b))
                    };
                    return worker_reader_receiver_rt_args;
                }
            };
            std::vector<uint32_t> worker_receiver_reader_rt_args = build_worker_receiver_reader_rt_args();

            std::string const& receiver_reader_kernel_path = is_sharded ?
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_receive_reader.cpp" :
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_reader.cpp";
            KernelHandle worker_receiver_reader_kernel_id = tt_metal::CreateKernel(
                program,
                receiver_reader_kernel_path,
                receiver_worker_cores.at(b),
                tt_metal::ReaderDataMovementConfig(worker_receiver_reader_ct_args, worker_defines));

            worker_reader_receiver_kernels.push_back(worker_receiver_reader_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_receiver_reader_kernel_id,
                receiver_worker_cores.at(b),
                worker_receiver_reader_rt_args);

            log_trace(tt::LogOp, "HOST SWR ARGS: \n");
            log_trace(tt::LogOp, "\toutput_is_dram: {}", all_gather_config.is_output_dram());
            log_trace(tt::LogOp, "\tnum_transfers: {}", num_transfers);
            log_trace(tt::LogOp, "\tnum_full_chunks_per_worker.at(b): {}", num_full_chunks_per_worker.at(b));
            log_trace(tt::LogOp, "\tinput_page_size: {}", input_page_size);
            log_trace(tt::LogOp, "\toutput_page_size: {}", output_page_size);
            log_trace(tt::LogOp, "\tpages_per_eth_l1_buffer.at(b): {}", pages_per_eth_l1_buffer.at(b));
            log_trace(tt::LogOp, "\trem_pages_per_worker.at(b): {}", rem_pages_per_worker.at(b));
            log_trace(tt::LogOp, "\treceiver_output_start_page_idx: {}", receiver_output_start_page_idx);
            log_trace(tt::LogOp, "\treceiver_output_start_addr_offset: {}", receiver_output_start_addr_offset);

            //// Receive Writer
            auto build_worker_receive_writer_ct_args = [&]() {
                if (is_sharded) {
                    // # Receiver Writer (CT)
                    // 1) Shard Type
                    std::vector<uint32_t> worker_receive_writer_ct_args = {
                        static_cast<uint32_t>(sharding_info.get_shard_type())
                    };
                    log_trace(tt::LogOp, "----worker_receive_writer_ct_args size={}", worker_receive_writer_ct_args.size());
                    log_trace(tt::LogOp, "\tsharding_info.get_shard_type(): {}", sharding_info.get_shard_type());

                    return worker_receive_writer_ct_args;
                } else {
                    std::vector<uint32_t> worker_writer_receiver_ct_args = {
                        static_cast<uint32_t>(all_gather_config.is_output_dram()),
                        static_cast<uint32_t>(num_transfers),
                        static_cast<uint32_t>(num_full_chunks_per_worker.at(b)),
                        static_cast<uint32_t>(input_page_size),
                        static_cast<uint32_t>(output_page_size),
                        static_cast<uint32_t>(pages_per_eth_l1_buffer.at(b)),
                        static_cast<uint32_t>(rem_pages_per_worker.at(b)),
                        static_cast<uint32_t>(receiver_output_start_page_idx),
                        static_cast<uint32_t>(receiver_output_start_addr_offset),
                        static_cast<uint32_t>(row_idx),
                        static_cast<uint32_t>(col_idx),
                        static_cast<uint32_t>(row_offset),
                        static_cast<uint32_t>(col_offset),
                        static_cast<uint32_t>(num_rows),
                        static_cast<uint32_t>(num_cols),
                        static_cast<uint32_t>(last_output_page_offset),
                        static_cast<uint32_t>(output_page_offset),
                        static_cast<uint32_t>(last_output_addr_offset),
                        static_cast<uint32_t>(output_addr_offset),
                        static_cast<uint32_t>(receiver_ring_index),
                        static_cast<uint32_t>(sender_worker_reader_semaphore_addr),
                        static_cast<uint32_t>(is_clockwise_direction ? 1 : 0),
                        static_cast<uint32_t>(cb_num_pages / 2)
                    };
                    return worker_writer_receiver_ct_args;
                }
            };
            std::vector<uint32_t> const& worker_receive_writer_ct_args = build_worker_receive_writer_ct_args();

            auto build_worker_receive_writer_rt_args = [&]() {
                auto worker_sender_reader = device->worker_core_from_logical_core(sender_worker_cores.at(b));
                if (is_sharded) {
                    // # Receiver Writer (RT)
                    // 1) Remote sender reader semaphore address
                    // 2) Output tensor Writer shard addr gen
                    bool is_clockwise = all_gather_config.is_buffer_in_clockwise_ring(b);

                    auto const& [starting_dest_worker_index, starting_chunk_into_shard] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config,
                        input_tensor,
                        output_tensor,
                        is_clockwise ?
                            (ring_index == 0 ? ring_size - 1 : ring_index - 1) :
                            (ring_index == ring_size - 1 ? 0 : ring_index + 1),
                        global_worker_index);
                    log_trace(tt::LogOp, "ReceiverWriter {} ring_index: {}, start dest worker index: {}, starting chunk into shard: {}", global_worker_index, ring_index, starting_dest_worker_index, starting_chunk_into_shard);
                    OutputTensorShardAddrGenArgGenerator output_tensor_shard_arg_generator(
                        all_gather_config,
                        device,
                        input_tensor,
                        output_tensor,
                        ring_index,
                        ring_size,
                        global_num_workers,
                        all_gather_config.get_num_eth_buffers_per_edm() * i + b,
                        starting_dest_worker_index,
                        starting_chunk_into_shard,
                        all_gather_config.is_buffer_in_clockwise_ring(b));
                    auto const& output_shard_addr_generator_args = output_tensor_shard_arg_generator.generate();
                    std::vector<uint32_t> worker_receive_writer_rt_args;
                    worker_receive_writer_rt_args.reserve(5 + output_shard_addr_generator_args.size());
                    worker_receive_writer_rt_args.push_back(static_cast<uint32_t>(worker_sender_reader.x));
                    worker_receive_writer_rt_args.push_back(static_cast<uint32_t>(worker_sender_reader.y));
                    worker_receive_writer_rt_args.push_back(sender_worker_reader_semaphore_addr);

                    worker_receive_writer_rt_args.push_back(output_tensor_shard_arg_generator.args_struct.num_dest_cores), //pages_per_eth_l1_buffer.at(b));
                    worker_receive_writer_rt_args.push_back(num_transfers);
                    worker_receive_writer_rt_args.push_back(pages_per_buffer.at(b));
                    worker_receive_writer_rt_args.push_back(static_cast<uint32_t>(cb_num_pages / 2));


                    std::copy(output_shard_addr_generator_args.begin(), output_shard_addr_generator_args.end(), std::back_inserter(worker_receive_writer_rt_args));

                    log_trace(tt::LogOp, "----worker_receive_writer_rt_args size={}", worker_receive_writer_rt_args.size());
                    log_trace(tt::LogOp, "\tsender_worker_reader_semaphore_addr: {}", sender_worker_reader_semaphore_addr);
                    output_tensor_shard_arg_generator.dump_to_log();

                    return worker_receive_writer_rt_args;
                } else {
                    std::vector<uint32_t> worker_writer_receiver_rt_args = {
                        static_cast<uint32_t>(output_buffer->address()),
                        static_cast<uint32_t>(worker_sender_reader.x),
                        static_cast<uint32_t>(worker_sender_reader.y),
                    };
                    return worker_writer_receiver_rt_args;
                }
            };
            std::vector<uint32_t> worker_receive_writer_rt_args = build_worker_receive_writer_rt_args();

            std::string const& receiver_writer_kernel_path = is_sharded ?
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_receive_writer.cpp" :
                "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_interleaved_ring_gather_receive_writer.cpp";
            KernelHandle worker_receive_writer_kernel_id = tt_metal::CreateKernel(
                program,
                receiver_writer_kernel_path,
                receiver_worker_cores.at(b),
                tt_metal::WriterDataMovementConfig(worker_receive_writer_ct_args, worker_defines));

            worker_writer_receiver_kernels.push_back(worker_receive_writer_kernel_id);

            tt_metal::SetRuntimeArgs(
                program,
                worker_receive_writer_kernel_id,
                receiver_worker_cores.at(b),
                worker_receive_writer_rt_args);

            uint32_t pages_per_worker = num_full_chunks_per_worker.at(b) * pages_per_chunk + rem_pages_per_worker.at(b);
            if (is_sharded) {
                // nothing to do here - is handled by
            } else {
                // Only for interleaved
                if (rm) {
                    uint32_t num_rows_shifted = row_idx + pages_per_worker;
                    uint32_t num_blocks_shifted = width ? 0 : num_rows_shifted / num_rows;
                    output_start_page_idx += pages_per_worker + num_blocks_shifted * row_offset;
                    row_idx = width ? 0 : num_rows_shifted % num_rows;
                } else {
                    uint32_t num_cols_shifted = col_idx + pages_per_worker;
                    uint32_t num_rows_shifted = num_cols_shifted / num_cols;
                    uint32_t num_blocks_shifted = width ? 0 : num_rows_shifted / num_rows;
                    output_start_page_idx += pages_per_worker + num_rows_shifted * col_offset + num_blocks_shifted * row_offset;
                    col_idx = num_cols_shifted % num_cols;
                    row_idx = width ? 0 : num_rows_shifted % num_rows;
                }
                input_start_page_idx += pages_per_worker;
            }
        }

        // Ethernet Kernels
        std::vector<uint32_t> eth_sender_ct_args = clockwise_edm_builders.at(i).emit_compile_time_args();

        log_trace(tt::LogOp, "EDM sender side link_clockwise_sender_num_channels.at(i) {}", link_clockwise_sender_num_channels.at(i));
        log_trace(tt::LogOp, "EDM sender side link_counter_clockwise_receiver_num_channels.at(i) {}", link_counter_clockwise_receiver_num_channels.at(i));

        auto eth_sender_kernel = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
            eth_sender_cores.at(i),
            tt_metal::EthernetConfig{.noc=sender_noc, .compile_args=eth_sender_ct_args});


        tt_metal::SetRuntimeArgs(
            program,
            eth_sender_kernel,
            eth_sender_cores.at(i),
            edm_clockwise_kernel_rt_args);

        eth_sender_kernels.push_back(eth_sender_kernel);

        std::vector<uint32_t> eth_receiver_ct_args = counter_clockwise_edm_builders.at(i).emit_compile_time_args();

        auto eth_receiver_kernel = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
            eth_receiver_cores.at(i),
            tt_metal::EthernetConfig{.noc=receiver_noc, .compile_args=eth_receiver_ct_args});

        eth_receiver_kernels.push_back(eth_receiver_kernel);

        log_trace(tt::LogOp, "RingIndex: {}. Link {}. Clockwise EDM Core (x={},y={}), Counter-clockwise EDM Core (x={},y={})", ring_index, i, eth_sender_cores.at(i).x, eth_sender_cores.at(i).y, eth_receiver_cores.at(i).x, eth_receiver_cores.at(i).y);


        std::stringstream ss;
        ss << "HOST SENDER EDM ARGS:\n";
        for (auto const& s : edm_clockwise_kernel_rt_args) {
            ss << "\t" << s << "\n";
        }
        log_trace(tt::LogOp, "{}", ss.str());

        std::stringstream ss2;
        ss2 << "HOST RECEIVER EDM ARGS:\n";
        for (auto const& s : edm_counter_clockwise_kernel_rt_args) {
            ss2 << "\t" << s << "\n";
        }
        log_trace(tt::LogOp, "{}", ss2.str());


        tt_metal::SetRuntimeArgs(
            program,
            eth_receiver_kernel,
            eth_receiver_cores.at(i),
            edm_counter_clockwise_kernel_rt_args);

        if (receiver_device_id == sender_device_id) {
            receiver_socket_idx += 2;
            sender_socket_idx += 2;
        } else {
            receiver_socket_idx += 1;
            sender_socket_idx += 1;
        }
    }

    auto override_runtime_arguments_callback = [num_links, total_worker_core_pairs_used, worker_reader_sender_kernels, worker_writer_sender_kernels, worker_reader_receiver_kernels, worker_writer_receiver_kernels, all_worker_sender_cores, all_worker_receiver_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        for (uint32_t i = 0; i < total_worker_core_pairs_used; ++i) {
            auto &worker_reader_sender_runtime_args = GetRuntimeArgs(program, worker_reader_sender_kernels.at(i), all_worker_sender_cores.at(i));
            worker_reader_sender_runtime_args.at(0) = input.buffer()->address();
            worker_reader_sender_runtime_args.at(1) = output.buffer()->address();
            auto &worker_writer_sender_runtime_args = GetRuntimeArgs(program, worker_writer_sender_kernels.at(i), all_worker_sender_cores.at(i));
            worker_writer_sender_runtime_args.at(0) = output.buffer()->address();

            auto &worker_writer_receiver_runtime_args = GetRuntimeArgs(program, worker_writer_receiver_kernels.at(i), all_worker_receiver_cores.at(i));
            worker_writer_receiver_runtime_args.at(0) = output.buffer()->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}




////////////////////
////////////////////
////////////////////
////////////////////
////////////////////
////////////////////
////////////////////
////////////////////
////////////////////
////////////////////
////////////////////
////////////////////
////////////////////

operation::ProgramWithCallbacks all_gather_full_shard_grid(const Tensor& input_tensor, Tensor& output_tensor, const uint32_t dim, const uint32_t num_links, const uint32_t ring_size, const uint32_t ring_index, const std::optional<chip_id_t> receiver_device_id, const std::optional<chip_id_t> sender_device_id, all_gather_op::Topology topology) {
    TT_ASSERT(topology == all_gather_op::Topology::Ring, "Only ring topology is currently supported by all gather with multi-tile-high shards");

    TT_ASSERT(input_tensor.is_sharded());
    tt_metal::Program program{};
    const auto& device = input_tensor.device();

    auto const& all_gather_config = AllGatherConfig(input_tensor, output_tensor, dim, ring_size, num_links, topology);
    auto const& sharding_info = ShardedAllGatherConfig(input_tensor, output_tensor, dim);

    all_gather_config.print();

    TT_FATAL(input_tensor.buffer()->page_size() <= all_gather_config.get_eth_buffer_size(), "Page size too large");

    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();

    ///// common
    int shard_grid_x = input_tensor.shard_spec()->grid.bounding_box().grid_size().x;
    int shard_grid_y = input_tensor.shard_spec()->grid.bounding_box().grid_size().y;
    uint32_t global_num_workers = shard_grid_x * shard_grid_y;

    std::vector<KernelHandle> worker_reader_kernels;
    worker_reader_kernels.reserve(global_num_workers);
    std::vector<KernelHandle> worker_writer_kernels;
    worker_writer_kernels.reserve(global_num_workers);

    int input_shard_num_tiles_x = input_buffer->shard_spec().tensor2d_shape[1];
    int input_shard_num_tiles_y = input_buffer->shard_spec().tensor2d_shape[0];
    auto const& worker_core_range = input_tensor.shard_spec()->grid.bounding_box();
    std::vector<CoreCoord> worker_cores; worker_cores.reserve(global_num_workers);
    {
        auto const& bounding_box = input_tensor.buffer()->shard_spec().grid().bounding_box();
        CoreCoord const& start = bounding_box.start;
        CoreCoord const& end = bounding_box.end;
        for (uint32_t r = start.y; r <= end.y; r++) {
            for (uint32_t c = start.x; c <= end.x; c++) {
                worker_cores.push_back({r, c});
            }
        }
    }


    // TODO: Calculate dynamically based on number of workers and number of available EDM buffers (* links)
    ccl::EriscDataMoverBufferSharingMode edm_buffer_sharing_mode = ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN;

    const uint32_t num_transfers = ring_size - 1;
    uint32_t const worker_num_input_shard_tiles = input_shard_num_tiles_x * input_shard_num_tiles_y;
    uint32_t const worker_num_output_shard_tiles = worker_num_input_shard_tiles * shard_grid_x;
    uint32_t const worker_num_tiles_to_send_to_edm = worker_num_input_shard_tiles * num_transfers;
    uint32_t tile_size_in_bytes = input_buffer->page_size();
    uint32_t shard_size_in_bytes = (tile_size_in_bytes * input_shard_num_tiles_y * input_shard_num_tiles_x) / input_tensor.shard_spec()->num_cores();
    uint32_t input_page_size = shard_size_in_bytes;
    uint32_t output_page_size = shard_size_in_bytes;

    const uint32_t max_buffer_per_chunk = round_down(all_gather_config.get_eth_buffer_size(), shard_size_in_bytes);
    const uint32_t max_pages_per_chunk = max_buffer_per_chunk / shard_size_in_bytes;
    log_trace(tt::LogOp, "shard_size_in_bytes: {}", shard_size_in_bytes);
    log_trace(tt::LogOp, "input_page_size: {}", input_page_size);
    log_trace(tt::LogOp, "max_buffer_per_chunk: {}", max_buffer_per_chunk);
    log_trace(tt::LogOp, "max_pages_per_chunk: {}", max_pages_per_chunk);
    uint32_t sender_socket_idx = 0;
    uint32_t receiver_socket_idx = 0;
    if (receiver_device_id == sender_device_id) {
        if (ring_index == 0) {
            receiver_socket_idx = 1;
        } else {
            sender_socket_idx = 1;
        }
    }

    DataFormat df = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());


    // number of worker cores is 2x this since there is 1 worker for the sender buffer and 1 worker for the receiver buffer
    std::vector<CoreCoord> eth_sender_cores;
    eth_sender_cores.reserve(num_links);
    std::vector<CoreCoord> eth_receiver_cores;
    eth_receiver_cores.reserve(num_links);
    std::vector<KernelHandle> eth_sender_kernels;
    eth_sender_kernels.reserve(num_links);
    std::vector<KernelHandle> eth_receiver_kernels;
    eth_receiver_kernels.reserve(num_links);

    std::vector<CoreRange> worker_sender_cores;
    worker_sender_cores.reserve(num_links);
    std::vector<KernelHandle> worker_reader_sender_kernels;
    worker_reader_sender_kernels.reserve(global_num_workers);
    std::vector<KernelHandle> worker_writer_sender_kernels;
    worker_writer_sender_kernels.reserve(global_num_workers);

    std::vector<CoreRange> worker_receiver_cores;
    worker_receiver_cores.reserve(num_links);
    std::vector<KernelHandle> worker_reader_receiver_kernels;
    worker_reader_receiver_kernels.reserve(global_num_workers);
    std::vector<KernelHandle> worker_writer_receiver_kernels;
    worker_writer_receiver_kernels.reserve(global_num_workers);

    std::vector<CoreCoord> worker_reader_edm_cores;
    std::vector<CoreCoord> worker_writed_edm_cores;

    for (uint32_t l = 0; l < num_links; ++l) {
        // Get the cores for the sender and receiver worker cores
        auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id.value()).at(sender_socket_idx + l);
        eth_sender_cores.push_back(eth_sender_core);
        auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id.value()).at(receiver_socket_idx + l);
        eth_receiver_cores.push_back(eth_receiver_core);
    }

    // indexing is link, edm_buffer_index
    // Need to know:
    // - each worker's EDM core
    std::vector<CoreCoord> worker_to_edm_sender_coord_map;
    std::vector<CoreCoord> worker_to_edm_receiver_coord_map;
    std::vector<uint32_t> worker_to_edm_link;
    std::vector<uint32_t> worker_to_edm_buffer;
    worker_to_edm_sender_coord_map.reserve(global_num_workers);
    worker_to_edm_receiver_coord_map.reserve(global_num_workers);
    worker_to_edm_link.reserve(global_num_workers);
    worker_to_edm_buffer.reserve(global_num_workers);
    // - each worker's EDM buffer location (if shared buffer, will need to offset into the buffer)
    // - Each EDM's worker cores

    std::vector<std::vector<std::vector<uint32_t>>> edm_buffer_worker_map;
    edm_buffer_worker_map.resize(num_links);
    for (uint32_t l = 0; l < num_links; ++l) {
        edm_buffer_worker_map.at(l).resize(all_gather_config.get_num_eth_buffers_per_edm());
    }

    {
        uint32_t link = 0;
        uint32_t edm_buffer_index = 0;
        for (uint32_t w = 0; w < global_num_workers; ++w) {
            worker_to_edm_sender_coord_map.push_back(eth_sender_cores.at(link));
            worker_to_edm_receiver_coord_map.push_back(eth_receiver_cores.at(link));
            worker_to_edm_link.push_back(link);
            worker_to_edm_buffer.push_back(edm_buffer_index);

            edm_buffer_worker_map.at(link).at(edm_buffer_index).push_back(w);

            link++;
            if (link == num_links) {
                edm_buffer_index++;
                link = 0;
            }
        }
    }

    TT_ASSERT(worker_to_edm_sender_coord_map.size() == global_num_workers);
    TT_ASSERT(worker_to_edm_receiver_coord_map.size() == global_num_workers);

    ///
    /// (counter clockwise sender) < ----- (this chip) < ----- (counter-clockwise receiver)
    ///
    /// (clockwise receiver)       ------> (this chip) ------> (clockwise sender)
    /// So clockwise sender and counter-clockwise receiver are on the same core
    //  and counter-clockwise sender and clockwise receiver are on the same corwe

    // Clockwise Direction
    std::vector<uint32_t> link_clockwise_sender_channels_offsets =
        std::vector<uint32_t>(num_links, 0);
    std::vector<uint32_t> link_clockwise_sender_num_channels =
        std::vector<uint32_t>(num_links, all_gather_config.get_num_edm_channels_in_clockwise_direction());
    std::vector<uint32_t> link_clockwise_receiver_num_channels = link_clockwise_sender_num_channels;
    // The clockwise direction's erisc's receiver offsets (i.e. for transfers coming INTO this chip)
    std::vector<uint32_t> link_clockwise_receiver_channels_offsets = link_clockwise_sender_channels_offsets;

    // Counter Clockwise Direction
    std::vector<uint32_t> link_counter_clockwise_sender_channels_offsets =
        std::vector<uint32_t>(num_links, all_gather_config.get_num_edm_channels_in_clockwise_direction());
    // Counter clock-wise buffers start after clockwise buffers in L1
    std::vector<uint32_t> link_counter_clockwise_sender_num_channels =
        std::vector<uint32_t>(num_links, all_gather_config.get_num_edm_channels_in_counter_clockwise_direction());
    std::vector<uint32_t> link_counter_clockwise_receiver_channels_offsets = link_counter_clockwise_sender_channels_offsets;
    std::vector<uint32_t> link_counter_clockwise_receiver_num_channels = link_counter_clockwise_sender_num_channels;

    std::vector<uint32_t> eth_sem_addrs;
    std::vector<uint32_t> eth_buffer_base_addrs;
    eth_sem_addrs.reserve(all_gather_config.get_num_eth_buffers_per_edm());
    eth_buffer_base_addrs.reserve(all_gather_config.get_num_eth_buffers_per_edm());

    for (uint32_t b = 0, eth_sem_addr = all_gather_config.get_eth_sems_l1_base_byte_address(), eth_buffer_addr = all_gather_config.get_eth_buffers_l1_base_byte_address(); b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
        eth_sem_addrs.push_back(eth_sem_addr);
        eth_sem_addr += all_gather_config.get_semaphore_size();
        eth_buffer_base_addrs.push_back(eth_buffer_addr);
        eth_buffer_addr += all_gather_config.get_eth_buffer_size();
    }

    std::vector<uint32_t> max_eth_l1_buffer_tiles_per_worker;
    // This value defines the number of EDM channel buffer tiles that each worker is allowed
    // to fill (for each transfer the worker is signalled)
    max_eth_l1_buffer_tiles_per_worker.reserve(num_links);
    switch (edm_buffer_sharing_mode) {
        case ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN:
            for (uint32_t i = 0; i < num_links; ++i) {
                max_eth_l1_buffer_tiles_per_worker.push_back(all_gather_config.get_eth_buffer_size());
            }
        break;

        default:
            TT_ASSERT(false, "Unsupported EDM buffer sharing mode for sharded all-gather");
        break;
    };
    auto compute_worker_facing_edm_buffer_size = [](ccl::EriscDataMoverBufferSharingMode edm_buffer_sharing_mode, AllGatherConfig const& all_gather_config) -> uint32_t{
        switch(edm_buffer_sharing_mode) {
            case ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN:
                return all_gather_config.get_eth_buffer_size();
            break;

            default:
                TT_ASSERT(false, "Unsupported EDM buffer sharing mode for sharded all-gather");
                return 0;
            break;
        };
    };
    uint32_t worker_facing_edm_buffer_size = compute_worker_facing_edm_buffer_size(edm_buffer_sharing_mode, all_gather_config);

    std::vector<uint32_t> per_worker_edm_buffer_base_addresses;
    per_worker_edm_buffer_base_addresses.reserve(global_num_workers);
    for (uint32_t w = 0; w < global_num_workers; ++w) {
        uint32_t edm_link = worker_to_edm_link.at(w);
        uint32_t edm_buffer_index = worker_to_edm_buffer.at(w);
        uint32_t edm_buffer_base_address = eth_buffer_base_addrs.at(edm_buffer_index);
        switch (edm_buffer_sharing_mode) {
            case ccl::EriscDataMoverBufferSharingMode::ROUND_ROBIN: {
                uint32_t worker_sub_buffer_offset_into_edm = edm_buffer_base_address + worker_facing_edm_buffer_size ;
                per_worker_edm_buffer_base_addresses.push_back(worker_sub_buffer_offset_into_edm);
            } break;

            default:
                TT_ASSERT(false, "Unsupported EDM buffer sharing mode for sharded all-gather");
            break;
        };
    }
    TT_ASSERT(per_worker_edm_buffer_base_addresses.size() == global_num_workers);

    std::vector<uint32_t> worker_num_messages_to_send_to_edm;
    worker_num_messages_to_send_to_edm.reserve(global_num_workers);
    for (uint32_t w = 0; w < global_num_workers; ++w) {
        // link num messages
        worker_num_messages_to_send_to_edm.push_back(((((worker_num_input_shard_tiles * tile_size_in_bytes) - 1) / worker_facing_edm_buffer_size) + 1) * num_transfers);
    }

    for (uint32_t link = 0; link < num_links; ++link) {
    log_trace(tt::LogOp, "==============================   LINK {}   ==============================", link);
        uint32_t workers_per_link = all_gather_config.get_num_workers_per_link() / all_gather_config.get_num_eth_buffers_per_edm();

        // Circular Buffer Setup
        uint32_t cb_page_size = shard_size_in_bytes;
        log_trace(tt::LogOp, "input_page_size: {}", input_page_size);
        uint32_t cb_num_pages = 2 * worker_num_input_shard_tiles;
        // uint32_t cb_num_pages = 2 * max_pages_per_chunk;
        log_trace(tt::LogOp, "cb_num_pages: {}", cb_num_pages);
        uint32_t src0_cb_index = CB::c_in0;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb_num_pages * cb_page_size, {{src0_cb_index, df}})
		.set_page_size(src0_cb_index, cb_page_size);
        CBHandle cb_src0_sender_workers = CreateCircularBuffer(program, worker_core_range, cb_src0_config);
        // CBHandle cb_src0_receiver_workers = CreateCircularBuffer(program, worker_core_range, cb_src0_config);

        // This semaphore is used by the receiver core to tell workers that data is available to read
        auto receiver_worker_semaphore_addr = tt_metal::CreateSemaphore(program, worker_core_range, 0);
        // This semaphore is used by the receiver core to tell the worker sender writer that sender buffer is available to write to
        auto sender_worker_writer_semaphore_addr = tt_metal::CreateSemaphore(program, worker_core_range, 0);
        // This semaphore is used by the worker receiver writer to tell the worker sender reader that data has been committed to memory
        // This is currently a running counter of how many chunks were committed since the sender worker never decrements this buffer
        // Potentially avoid overflow by having it actually decrement (using noc atomic inc with value of -1)
        auto worker_reader_writer_semaphore_addr = tt_metal::CreateSemaphore(program, worker_core_range, 0);

        auto sender_noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch());
        auto receiver_noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch());

        auto eth_sender_core = device->get_ethernet_sockets(receiver_device_id.value()).at(sender_socket_idx);
        eth_sender_cores.push_back(eth_sender_core);
        auto eth_receiver_core = device->get_ethernet_sockets(sender_device_id.value()).at(receiver_socket_idx);
        eth_receiver_cores.push_back(eth_receiver_core);

        std::vector<uint32_t> edm_semaphores_base_address;
        std::vector<uint32_t> link_buffer_sender_addresses;
        edm_semaphores_base_address.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        link_buffer_sender_addresses.reserve(all_gather_config.get_num_eth_buffers_per_edm());

        for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            edm_semaphores_base_address.push_back(all_gather_config.get_eth_sems_l1_base_byte_address() + b * all_gather_config.get_semaphore_size());
            link_buffer_sender_addresses.push_back(all_gather_config.get_eth_buffers_l1_base_byte_address() + b * all_gather_config.get_eth_buffer_size());
        }

        std::vector<uint32_t> receiver_semaphores_base_address;
        std::vector<uint32_t> link_buffer_receiver_addresses;
        receiver_semaphores_base_address.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        link_buffer_receiver_addresses.reserve(all_gather_config.get_num_eth_buffers_per_edm());
        for(uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            receiver_semaphores_base_address.push_back(all_gather_config.get_eth_sems_l1_base_byte_address() + b * all_gather_config.get_semaphore_size());
            link_buffer_receiver_addresses.push_back(all_gather_config.get_eth_buffers_l1_base_byte_address() + b * all_gather_config.get_eth_buffer_size());
        }

        log_trace(tt::LogOp, "all_gather_config.get_num_eth_buffers_per_edm(): {}", all_gather_config.get_num_eth_buffers_per_edm());

        std::vector<uint32_t> edm_clockwise_kernel_rt_args = {
            static_cast<uint32_t>(all_gather_config.get_erisc_handshake_address()),
            static_cast<uint32_t>(link_clockwise_sender_channels_offsets.at(link))
        };

        std::vector<uint32_t> edm_counter_clockwise_kernel_rt_args = {
            static_cast<uint32_t>(all_gather_config.get_erisc_handshake_address()),
            static_cast<uint32_t>(link_counter_clockwise_sender_channels_offsets.at(link))
        };

        for (uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            log_trace(tt::LogOp, "\t------------------------------   EDM Buffer {} Sender Side  ------------------------------", b);
            // Setup sender direction args
            auto& edm_kernel_rt_args = all_gather_config.is_buffer_in_clockwise_ring(b) ? edm_clockwise_kernel_rt_args : edm_counter_clockwise_kernel_rt_args;
            log_trace(tt::LogOp, "\tChannel {}", b);
            log_trace(tt::LogOp, "\t\tclockwise? {}", all_gather_config.is_buffer_in_clockwise_ring(b) ? "true" : "false");
            // eth sender args
            // sender_buffer_address
            edm_kernel_rt_args.push_back(link_buffer_sender_addresses.at(b));
            log_trace(tt::LogOp, "\t\tbuffer address {}", link_buffer_sender_addresses.at(b));

            // sender_num_messages_to_send
            edm_kernel_rt_args.push_back(0);
            for (std::size_t w : edm_buffer_worker_map.at(link).at(b)) {
                edm_kernel_rt_args.back() += worker_num_messages_to_send_to_edm.at(w);
            }
            log_trace(tt::LogOp, "\t\tnum sends {}", edm_kernel_rt_args.back());

            // sender_channel_size
            edm_kernel_rt_args.push_back(all_gather_config.get_eth_buffer_size());
            log_trace(tt::LogOp, "\t\tbuffer size {}", edm_kernel_rt_args.back());

            // edm_semaphores_base_address -> erisc L1 address
            edm_kernel_rt_args.push_back(eth_sem_addrs.at(b));
            log_trace(tt::LogOp, "\t\tedm l1 semaphore address {}", edm_kernel_rt_args.back());

            // worker_semaphore_address
            edm_kernel_rt_args.push_back(sender_worker_writer_semaphore_addr);
            log_trace(tt::LogOp, "\t\tworker l1 semaphore address {}", edm_kernel_rt_args.back());

            // sender_num_workers - only 1 per channel right now
            edm_kernel_rt_args.push_back(edm_buffer_worker_map.at(link).at(b).size());
            log_trace(tt::LogOp, "\t\tnum connected workers {}:", edm_kernel_rt_args.back());

            for (std::size_t w : edm_buffer_worker_map.at(link).at(b)) {
                edm_kernel_rt_args.push_back((uint32_t)(
                    (device->worker_core_from_logical_core(worker_cores.at(w)).y << 16) |
                    (device->worker_core_from_logical_core(worker_cores.at(w)).x)
                ));
                log_trace(tt::LogOp, "\t\t\tnum connected workers (x={},y={})", worker_cores.at(w).x, worker_cores.at(w).y);
            }
        }

        // Setup receiver direction args. Clockwise receiver is same offset as sender offset for clockwise direction
        edm_clockwise_kernel_rt_args.push_back(static_cast<uint32_t>(link_counter_clockwise_receiver_channels_offsets.at(link)));

        edm_counter_clockwise_kernel_rt_args.push_back(static_cast<uint32_t>(link_clockwise_receiver_channels_offsets.at(link)));

        for (uint32_t b = 0; b < all_gather_config.get_num_eth_buffers_per_edm(); ++b) {
            log_trace(tt::LogOp, "\t------------------------------   EDM Buffer {} Receiver Side  ------------------------------", b);
            log_trace(tt::LogOp, "\tChannel {}", b);
            log_trace(tt::LogOp, "\t\tclockwise? {}", all_gather_config.is_buffer_in_clockwise_ring(b) ? "true" : "false");
            auto& edm_kernel_rt_args = all_gather_config.is_buffer_in_clockwise_ring(b) ? edm_counter_clockwise_kernel_rt_args : edm_clockwise_kernel_rt_args ;
            // eth receiver args
            // sender_buffer_address
            edm_kernel_rt_args.push_back(link_buffer_receiver_addresses.at(b));
            log_trace(tt::LogOp, "\t\tbuffer address {}", link_buffer_sender_addresses.at(b));

            // sender_num_messages_to_send
            edm_kernel_rt_args.push_back(0);
            for (std::size_t w : edm_buffer_worker_map.at(link).at(b)) {
                edm_kernel_rt_args.back() += worker_num_messages_to_send_to_edm.at(w);
            }
            log_trace(tt::LogOp, "\t\tnum receives {}", edm_kernel_rt_args.back());

            // sender_channel_size
            edm_kernel_rt_args.push_back(all_gather_config.get_eth_buffer_size());
            log_trace(tt::LogOp, "\t\tbuffer size {}", edm_kernel_rt_args.back());

            // edm_semaphores_base_address -> erisc L1 address
            edm_kernel_rt_args.push_back(eth_sem_addrs.at(b));
            log_trace(tt::LogOp, "\t\tedm l1 semaphore address {}", edm_kernel_rt_args.back());

            // worker_semaphore_address
            edm_kernel_rt_args.push_back(receiver_worker_semaphore_addr);
            log_trace(tt::LogOp, "\t\tworker l1 semaphore address {}", edm_kernel_rt_args.back());

            // sender_num_workers - only 1 per channel right now
            edm_kernel_rt_args.push_back(edm_buffer_worker_map.at(link).at(b).size());
            log_trace(tt::LogOp, "\t\tnum connected workers {}:", edm_kernel_rt_args.back());

            for (std::size_t w : edm_buffer_worker_map.at(link).at(b)) {
                edm_kernel_rt_args.push_back((uint32_t)(
                    (device->worker_core_from_logical_core(worker_cores.at(w)).y << 16) |
                    (device->worker_core_from_logical_core(worker_cores.at(w)).x)
                ));
                log_trace(tt::LogOp, "\t\t\tnum connected workers (x={},y={})", worker_cores.at(w).x, worker_cores.at(w).y);
            }
        }

        for (uint32_t b = 0; b < edm_buffer_worker_map.at(link).size(); b++) {
            log_trace(tt::LogOp, "------------------------------ Workers for EDM buffers {} ------------------------------", b);
            bool is_clockwise_direction = all_gather_config.is_buffer_in_clockwise_ring(b);
            log_trace(tt::LogOp, "\t\tis_clockwise ? {}", (is_clockwise_direction ? "true" : "false"));
            // Not fully sure about these two
            uint32_t receiver_ring_index = is_clockwise_direction ?
                (ring_index == 0 ? ring_size - 1 : ring_index - 1):
                (ring_index == ring_size - 1 ? 0 : ring_index + 1);

            for (uint32_t global_worker_index : edm_buffer_worker_map.at(link).at(b)) {
                log_trace(tt::LogOp, "\t-------- Worker: {} --------", global_worker_index);
                // for (uint32_t global_worker_index = 0; global_worker_index < global_num_workers; ++global_worker_index) {


                //// Receive Reader
                auto build_worker_writer_ct_args = [&]() {
                    std::vector<uint32_t> worker_writer_ct_args = {
                        sharding_info.get_shard_type()
                    };

                    log_trace(tt::LogOp, "\t\t--------- worker_writer_ct_args:");
                    log_trace(tt::LogOp, "\t\tshard_type: {}", sharding_info.get_shard_type());
                    return worker_writer_ct_args;
                };
                std::vector<uint32_t> const& worker_writer_ct_args = build_worker_writer_ct_args();

                auto build_worker_writer_rt_args = [&]() {
                    log_trace(tt::LogOp, "\t\t--------- build_worker_writer_rt_args:");
                    auto const& [starting_dest_worker_index, starting_shard_in_dest_core] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config,
                        input_tensor,
                        output_tensor,
                        ring_index,
                        global_worker_index
                    );
                    uint32_t starting_tile_index = starting_shard_in_dest_core * input_shard_num_tiles_x;
                    auto const addr_gen_args_gen = FullWorkerGridShardAddrGenArgGenerator(
                        all_gather_config,
                        device,
                        input_tensor,
                        output_tensor,
                        ring_index,
                        ring_size,
                        global_worker_index,
                        starting_dest_worker_index,
                        starting_tile_index,
                        is_clockwise_direction);
                    auto const& addr_gen_args = addr_gen_args_gen.generate();
                    std::vector<uint32_t> worker_writer_rt_args = {};

                    worker_writer_rt_args.push_back(per_worker_edm_buffer_base_addresses.at(global_worker_index)); // eth_sender_l1_base_addr);
                    worker_writer_rt_args.push_back(eth_sem_addrs.at(worker_to_edm_buffer.at(global_worker_index)));
                    worker_writer_rt_args.push_back(device->ethernet_core_from_logical_core(worker_to_edm_sender_coord_map.at(global_worker_index)).x);
                    worker_writer_rt_args.push_back(device->ethernet_core_from_logical_core(worker_to_edm_sender_coord_map.at(global_worker_index)).y);
                    worker_writer_rt_args.push_back(worker_facing_edm_buffer_size / tile_size_in_bytes); // tiles per edm buffer
                    worker_writer_rt_args.push_back(sender_worker_writer_semaphore_addr);
                    worker_writer_rt_args.push_back(num_transfers);
                    worker_writer_rt_args.push_back(worker_reader_writer_semaphore_addr);
                    log_trace(tt::LogOp, "\t\teth_sender_l1_base_addr: {}", per_worker_edm_buffer_base_addresses.at(global_worker_index));
                    log_trace(tt::LogOp, "\t\teth_sender_l1_sem_addr: {}", eth_sem_addrs.at(worker_to_edm_buffer.at(global_worker_index)));
                    log_trace(tt::LogOp, "\t\teth_sender_noc_x: {}", device->ethernet_core_from_logical_core(worker_to_edm_sender_coord_map.at(global_worker_index)).x);
                    log_trace(tt::LogOp, "\t\teth_sender_noc_y: {}", device->ethernet_core_from_logical_core(worker_to_edm_sender_coord_map.at(global_worker_index)).y);
                    log_trace(tt::LogOp, "\t\ttiles_per_eth_l1_buffer: {}", worker_facing_edm_buffer_size / tile_size_in_bytes);
                    log_trace(tt::LogOp, "\t\teth_buffer_available_semaphore_ptr: {}", sender_worker_writer_semaphore_addr);
                    log_trace(tt::LogOp, "\t\tnum_transfers: {}", num_transfers);
                    log_trace(tt::LogOp, "\t\ttiles_available_semaphore_ptr: {}", worker_reader_writer_semaphore_addr);
                    log_trace(tt::LogOp, "\t\t\taddr_gen_args:");
                    for (auto const& arg : addr_gen_args) {
                        log_trace(tt::LogOp, "\t\t\t\t{}", arg);
                    }

                    std::copy(addr_gen_args.begin(), addr_gen_args.end(), std::back_inserter(worker_writer_rt_args));

                    return worker_writer_rt_args;
                };
                std::vector<uint32_t> worker_writer_rt_args = build_worker_writer_rt_args();

                std::string const& worker_writer_kernel_path = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_all_shard_workers_ring_gather_writer.cpp";
                KernelHandle worker_writer_kernel_id = tt_metal::CreateKernel(
                    program,
                    worker_writer_kernel_path,
                    worker_cores.at(b),
                    tt_metal::WriterDataMovementConfig(worker_writer_ct_args));

                worker_writer_kernels.push_back(worker_writer_kernel_id);

                tt_metal::SetRuntimeArgs(
                    program,
                    worker_writer_kernel_id,
                    worker_cores.at(b),
                    worker_writer_rt_args);

                //// Receive Writer
                auto build_worker_reader_ct_args = [&]() {
                    std::vector<uint32_t> worker_writer_ct_args = {
                        sharding_info.get_shard_type()
                    };
                    log_trace(tt::LogOp, "\t\t--------- build_worker_reader_rt_args:");
                    log_trace(tt::LogOp, "\t\tshard_type: {}", sharding_info.get_shard_type());
                    return worker_writer_ct_args;
                };
                std::vector<uint32_t> const& worker_reader_ct_args = build_worker_reader_ct_args();

                auto build_worker_reader_rt_args = [&]() {
                    log_trace(tt::LogOp, "\t\t--------- build_worker_reader_rt_args:");
                    std::vector<uint32_t> args = {};

                    auto const& [starting_dest_worker_index, starting_shard_in_dest_core] = OutputTensorShardAddrGenArgGenerator::get_first_output_shard_starting_location(
                        all_gather_config,
                        input_tensor,
                        output_tensor,
                        ring_index,
                        global_worker_index
                    );
                    uint32_t starting_tile_index = starting_shard_in_dest_core * input_shard_num_tiles_x;
                    auto const output_addr_gen_args_gen = FullWorkerGridShardAddrGenArgGenerator(
                        all_gather_config,
                        device,
                        input_tensor,
                        output_tensor,
                        ring_index,
                        ring_size,
                        global_worker_index,
                        starting_dest_worker_index,
                        starting_tile_index,
                        is_clockwise_direction);
                    auto const& output_addr_gen_args = output_addr_gen_args_gen.generate();
                    args.push_back(input_buffer->address());
                    args.push_back(receiver_worker_semaphore_addr);
                    args.push_back(worker_facing_edm_buffer_size / tile_size_in_bytes);
                    args.push_back(worker_reader_writer_semaphore_addr);
                    args.push_back(device->ethernet_core_from_logical_core(worker_to_edm_receiver_coord_map.at(global_worker_index)).x);
                    args.push_back(device->ethernet_core_from_logical_core(worker_to_edm_receiver_coord_map.at(global_worker_index)).y);
                    args.push_back(per_worker_edm_buffer_base_addresses.at(global_worker_index));
                    args.push_back(eth_sem_addrs.at(worker_to_edm_buffer.at(global_worker_index)));
                    args.push_back(num_transfers);
                    log_trace(tt::LogOp, "\t\tinput_shard_address: {}", input_buffer->address());
                    log_trace(tt::LogOp, "\t\teth_to_local_semaphore_address", receiver_worker_semaphore_addr);
                    log_trace(tt::LogOp, "\t\ttiles_per_eth_l1_buffer", worker_facing_edm_buffer_size / tile_size_in_bytes);
                    log_trace(tt::LogOp, "\t\ttiles_available_semaphore_ptr", worker_reader_writer_semaphore_addr);
                    log_trace(tt::LogOp, "\t\teth_noc_x: {}", device->ethernet_core_from_logical_core(worker_to_edm_receiver_coord_map.at(global_worker_index)).x);
                    log_trace(tt::LogOp, "\t\teth_noc_y: {}", device->ethernet_core_from_logical_core(worker_to_edm_receiver_coord_map.at(global_worker_index)).y);
                    log_trace(tt::LogOp, "\t\teth_l1_buffer_addres: {}", per_worker_edm_buffer_base_addresses.at(global_worker_index));
                    log_trace(tt::LogOp, "\t\teth_semaphore_addres: {}", eth_sem_addrs.at(worker_to_edm_buffer.at(global_worker_index)));
                    log_trace(tt::LogOp, "\t\tnum_transfers: {}", num_transfers);

                    std::copy(output_addr_gen_args.begin(), output_addr_gen_args.end(), std::back_inserter(args));
                    log_trace(tt::LogOp, "\t\t\taddr_gen_args:");
                    for (auto const& arg : output_addr_gen_args) {
                        log_trace(tt::LogOp, "\t\t\t\t{}", arg);
                    }

                    return args;
                };
                std::vector<uint32_t> worker_reader_rt_args = build_worker_reader_rt_args();

                std::string const& worker_reader_kernel_path = "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_all_shard_workers_ring_gather_reader.cpp";
                KernelHandle worker_reader_kernel_id = tt_metal::CreateKernel(
                    program,
                    worker_reader_kernel_path,
                    worker_cores.at(b),
                    tt_metal::ReaderDataMovementConfig(worker_reader_ct_args));

                worker_reader_kernels.push_back(worker_reader_kernel_id);

                tt_metal::SetRuntimeArgs(
                    program,
                    worker_reader_kernel_id,
                    worker_cores.at(b),
                    worker_reader_rt_args);
            }
        }

        // Ethernet Kernels
        std::vector<uint32_t> eth_sender_ct_args = {
            static_cast<uint32_t>(all_gather_config.get_num_edm_channels_in_clockwise_direction() ? 1 : 0),
            static_cast<uint32_t>(all_gather_config.get_num_edm_channels_in_counter_clockwise_direction() ? 1 : 0),
            static_cast<uint32_t>(link_clockwise_sender_num_channels.at(link)),
            static_cast<uint32_t>(link_counter_clockwise_receiver_num_channels.at(link)),
            static_cast<uint32_t>(edm_buffer_sharing_mode)
        };


        log_trace(tt::LogOp, "EDM sender side link_clockwise_sender_num_channels.at(i) {}", link_clockwise_sender_num_channels.at(link));
        log_trace(tt::LogOp, "EDM sender side link_counter_clockwise_receiver_num_channels.at(i) {}", link_counter_clockwise_receiver_num_channels.at(link));
        log_trace(tt::LogOp, "EDM sender edm_buffer_sharing_mode {}", edm_buffer_sharing_mode);

        auto eth_sender_kernel = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
            eth_sender_cores.at(link),
            tt_metal::EthernetConfig{.noc=sender_noc, .compile_args=eth_sender_ct_args});

        tt_metal::SetRuntimeArgs(
            program,
            eth_sender_kernel,
            eth_sender_cores.at(link),
            edm_clockwise_kernel_rt_args);

        eth_sender_kernels.push_back(eth_sender_kernel);

        std::vector<uint32_t> eth_receiver_ct_args = {
            static_cast<uint32_t>(all_gather_config.get_num_edm_channels_in_counter_clockwise_direction() ? 1 : 0),
            static_cast<uint32_t>(all_gather_config.get_num_edm_channels_in_clockwise_direction() ? 1 : 0),
            static_cast<uint32_t>(link_counter_clockwise_sender_num_channels.at(link)),
            static_cast<uint32_t>(link_clockwise_receiver_num_channels.at(link)),
            static_cast<uint32_t>(edm_buffer_sharing_mode)
        };

        auto eth_receiver_kernel = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
            eth_receiver_cores.at(link),
            tt_metal::EthernetConfig{.noc=receiver_noc, .compile_args=eth_receiver_ct_args});

        eth_receiver_kernels.push_back(eth_receiver_kernel);

        log_trace(tt::LogOp, "RingIndex: {}. Link {}. Clockwise EDM Core (x={},y={}), Counter-clockwise EDM Core (x={},y={})",
            ring_index, link, eth_sender_cores.at(link).x, eth_sender_cores.at(link).y, eth_receiver_cores.at(link).x, eth_receiver_cores.at(link).y);

        std::stringstream ss;
        ss << "HOST SENDER EDM ARGS:\n";
        for (auto const& s : edm_clockwise_kernel_rt_args) {
            ss << "\t" << s << "\n";
        }
        log_trace(tt::LogOp, "{}", ss.str());

        std::stringstream ss2;
        ss2 << "HOST RECEIVER EDM ARGS:\n";
        for (auto const& s : edm_counter_clockwise_kernel_rt_args) {
            ss2 << "\t" << s << "\n";
        }
        log_trace(tt::LogOp, "{}", ss2.str());


        tt_metal::SetRuntimeArgs(
            program,
            eth_receiver_kernel,
            eth_receiver_cores.at(link),
            edm_counter_clockwise_kernel_rt_args);

        if (receiver_device_id == sender_device_id) {
            receiver_socket_idx += 2;
            sender_socket_idx += 2;
        } else {
            receiver_socket_idx += 1;
            sender_socket_idx += 1;
        }
    }

    TT_ASSERT(worker_reader_kernels.size() == global_num_workers);
    TT_ASSERT(worker_writer_kernels.size() == global_num_workers);

    ////
    ///  Callback
    ////
    auto override_runtime_arguments_callback = [num_links, global_num_workers, worker_reader_kernels, worker_writer_kernels, worker_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        for (uint32_t i = 0; i < global_num_workers; ++i) {
            auto &worker_reader_runtime_args = GetRuntimeArgs(program, worker_reader_kernels.at(i), worker_cores.at(i));
            worker_reader_runtime_args.at(0) = input.buffer()->address();
            worker_reader_runtime_args.at(1) = output.buffer()->address();
            auto &worker_writer_runtime_args = GetRuntimeArgs(program, worker_writer_kernels.at(i), worker_cores.at(i));
            worker_writer_runtime_args.at(0) = output.buffer()->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
