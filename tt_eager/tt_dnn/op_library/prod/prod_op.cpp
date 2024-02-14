// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/prod/prod_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include <algorithm>
#include <optional>

#include "tt_metal/common/constants.hpp"
#include <tt_eager/tt_numpy/functions.hpp>
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

void Prod_op::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to eltwise unary need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to eltwise unary need to be allocated in buffers on device!");
    TT_FATAL((input_tensor_a.layout() == Layout::TILE), "Inputs to eltwise unary must be tilized");
    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16);
}

std::vector<Shape> Prod_op::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> Prod_op::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Prod_op::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return prod_single_core(input_tensor_a, output_tensor);
}

Tensor prod(const Tensor& input, const MemoryConfig& output_mem_config ) {
    // return operation::run(Prod_op{.output_mem_config = output_mem_config}, {input}).at(0);
    Tensor result =  tiled_prod( operation::run(Prod_op{.output_mem_config = output_mem_config}, {input}).at(0), output_mem_config);
    // return result;
    return tt::numpy::prod_result_computation<bfloat16>(result, result.dtype(), result.layout(), result.device(), output_mem_config);
}

}
}
}
