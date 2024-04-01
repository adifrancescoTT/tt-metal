// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include "eth_l1_address_map.h"

namespace tt {

namespace tt_metal {

AllGatherMode choose_all_gather_mode(Tensor const& input_tensor, Tensor const& output_tensor, uint32_t dim) {
    bool is_sharded = input_tensor.is_sharded();

    if (is_sharded) {
        if (input_tensor.buffer()->shard_spec().tensor2d_shape[0] > 1) {
            return AllGatherMode::FULL_WORKER_GRID_SHARDED;
        } else {
            return AllGatherMode::SINGLE_TILE_HIGH_WIDTH_SHARDED;
        }
    } else {
        return AllGatherMode::RING_INTERLEAVED;
    }
}

void AllGather::validate(const std::vector<Tensor> &input_tensors) const {
    TT_FATAL(input_tensors.size() == 1);
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % ADDRESS_ALIGNMENT == 0, "All Gather currently requires aligned pages");

    // TODO: This can be removed by passing two page sizes, actual and aligned to be used for address offsets
    // Buffer sizes also need to take this aligned page size into consideration
    // TODO: Validate ring
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr , "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0);
    TT_FATAL(this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y, "Worker cores used by links are parallelizaed over rows");
    TT_FATAL(this->receiver_device_id.has_value() || this->sender_device_id.has_value());
    if (this->receiver_device_id == this->sender_device_id) {
        TT_FATAL(input_tensor.device()->get_ethernet_sockets(this->receiver_device_id.value()).size() >= 2 * this->num_links, "2 Device all gather requires at least 2 eth connections per link");
    } else {
        TT_FATAL(input_tensor.device()->get_ethernet_sockets(this->receiver_device_id.value()).size() >= this->num_links, "All gather requires at least 1 eth connection per link between sender device {} and receiver device {}", this->sender_device_id, this->receiver_device_id);
        TT_FATAL(input_tensor.device()->get_ethernet_sockets(this->sender_device_id.value()).size() >= this->num_links, "All gather requires at least 1 eth connection per link between sender device {} and receiver device {}", this->sender_device_id, this->receiver_device_id);
    }

    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);

    // Sharding Config checks
    bool input_sharded = input_tensor.is_sharded();
    if (input_sharded) {
        // TODO(snijjar)
    }
}

std::vector<Shape> AllGather::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto shape = input_tensors[0].get_legacy_shape();
    shape[this->dim] *= this->ring_size;
    return std::vector<Shape>(input_tensors.size(), shape);
}

std::vector<Tensor> AllGather::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    if(this->output_mem_config.is_sharded()) {
        return {create_sharded_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            this->output_mem_config
            )};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks AllGather::create_program(const std::vector<Tensor> & input_tensors, std::vector<Tensor> &output_tensors) const {
    AllGatherMode all_gather_mode = tt::tt_metal::choose_all_gather_mode(input_tensors.at(0), output_tensors.at(0), dim);
    switch (all_gather_mode) {
        case AllGatherMode::RING_INTERLEAVED:
        case AllGatherMode::SINGLE_TILE_HIGH_WIDTH_SHARDED:
            return all_gather_multi_core_with_workers(input_tensors[0], output_tensors[0], this->dim, this->num_links, this->ring_size, this->ring_index, this->receiver_device_id, this->sender_device_id, this->topology);
        break;
        case AllGatherMode::FULL_WORKER_GRID_SHARDED:
            return all_gather_full_shard_grid(input_tensors.at(0), output_tensors.at(0), this->dim, this->num_links, this->ring_size, this->ring_index, this->receiver_device_id, this->sender_device_id, this->topology);
        break;
        default:
            TT_THROW("Unsupported AllGatherMode");
    };
}

tt::stl::reflection::Attributes AllGather::attributes() const {
    return {
        {"dim", this->dim},
        {"num_links", this->num_links},
        {"ring_size", this->ring_size},
        {"ring_index", this->ring_index},
        {"receiver_device_id", this->receiver_device_id},
        {"sender_device_id", this->sender_device_id},
        {"output_mem_config", this->output_mem_config},
    };
}

std::vector<Tensor> all_gather_impl(const std::vector<Tensor>& input_tensors, const uint32_t dim, const uint32_t num_links, const MemoryConfig& output_mem_config, const all_gather_op::Topology topology) {

    TT_FATAL(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr, "This op is only supported for Fast Dispatch");

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    std::vector<AllGather> ops;
    ops.reserve(input_tensors.size());
    bool is_ring = topology == all_gather_op::Topology::Ring;
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        bool is_last_chip_in_clockwise_direction = is_ring ? false : i == (input_tensors.size() - 1);
        bool is_last_chip_in_counter_clockwise_direction = is_ring ? false : i == 0;

        std::optional<chip_id_t> receiver_device_id = is_last_chip_in_clockwise_direction ?
            std::nullopt :
            std::optional<chip_id_t>(input_tensors[(i + 1) % input_tensors.size()].device()->id());
        std::optional<chip_id_t> sender_device_id = is_last_chip_in_counter_clockwise_direction ?
            std::nullopt :
            std::optional<chip_id_t>(input_tensors[i == 0 ? input_tensors.size() - 1 : i - 1].device()->id());
        ops.emplace_back(AllGather{dim, num_links, static_cast<uint32_t>(input_tensors.size()), i, receiver_device_id, sender_device_id, output_mem_config,topology});
        output_tensors.push_back(operation::run(ops[i], {input_tensors[i]}).at(0));
    }
    return output_tensors;
}

std::vector<Tensor> all_gather(const std::vector<Tensor>& input_tensors, const uint32_t dim, const uint32_t num_links, const MemoryConfig& output_mem_config) {
    return all_gather_impl(input_tensors, dim, num_links, output_mem_config, all_gather_op::Topology::Ring);
}
std::vector<Tensor> line_all_gather(const std::vector<Tensor>& input_tensors, const uint32_t dim, const uint32_t num_links, const MemoryConfig& output_mem_config) {
    return all_gather_impl(input_tensors, dim, num_links, output_mem_config, all_gather_op::Topology::Linear);
}

}  // namespace tt_metal

}  // namespace tt
