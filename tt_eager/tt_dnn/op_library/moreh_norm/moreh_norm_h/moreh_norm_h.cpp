// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_norm_h_impl(const Tensor &input, float p, const Tensor &output) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.get_legacy_shape();
    const auto input_rank = input_shape.rank();

    const auto N = input_shape[0];
    const auto C = input_shape[1];
    const auto H = input_shape[2];
    const auto W = input_shape[3];

    const auto Ht = H / TILE_HEIGHT;
    const auto Wt = W / TILE_WIDTH;

    const auto origin_h = input_shape.without_padding()[input_rank - 2];

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
    auto [floored_recip_p, recip_p_decimal, recip_p_is_negative] =
        get_floored_p_and_decimal_and_p_is_negative(1.0f / p);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CoreGridDesc core_grid(device);
    const auto num_cores_y = core_grid.y_;
    CoreCoord core_grid_coord(core_grid.x_, num_cores_y);

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt_metal::split_work_to_cores(core_grid_coord, N * C * Wt);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    const uint32_t in0_t{1};  // input
    const uint32_t in1_t{1};  // one
    const uint32_t in2_t{1};  // decimal
    const uint32_t in3_t{1};  // recip_p_decimal
    const uint32_t in4_t{1};  // mask_h

    const uint32_t out0_t{1};  // output

    const uint32_t im0_t{1};  // |x|
    const uint32_t im1_t{1};  // log(|x|)
    const uint32_t im2_t{1};  // exp(log(|x|) * decimal)
    const uint32_t im3_t{1};  // |x|^p
    const uint32_t im4_t{1};  // |x|^p * exp(log(|x|) * decimal) == |x + decimal|^p
    const uint32_t im5_t{1};  // Add(|x + decimal|^p)
    const uint32_t im6_t{1};  // Sum(|x + decimal|^p)

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},    // input
            {CB::c_in1, in1_t},    // one
            {CB::c_in2, in2_t},    // decimal
            {CB::c_in3, in3_t},    // recip_p_decimal
            {CB::c_in4, in4_t},    // mask_h
            {CB::c_out0, out0_t},  // output
            {CB::c_intermed0, im0_t},
            {CB::c_intermed1, im1_t},
            {CB::c_intermed2, im2_t},
            {CB::c_intermed3, im3_t},
            {CB::c_intermed4, im4_t},
            {CB::c_intermed5, im5_t},
            {CB::c_intermed6, im6_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_h/kernels/"
        "reader_moreh_norm_h.cpp";
    const auto writer_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_h/kernels/"
        "writer_moreh_norm_h.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";

    const auto compute_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_h/kernels/"
        "moreh_norm_h_kernel.cpp";

    const auto compute_kernels_id_1 =
        CreateComputeKernel(program, compute_kernel_file, {core_group_1, num_cols_per_core_group_1}, compute_defines);

    KernelHandle compute_kernels_id_2{0};
    if (!core_group_2.ranges().empty()) {
        compute_kernels_id_2 = CreateComputeKernel(
            program, compute_kernel_file, {core_group_2, num_cols_per_core_group_2}, compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();
    const auto output_addr = output.buffer()->address();

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_cols_per_core;
        KernelHandle compute_kernel_id;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
            compute_kernel_id = compute_kernels_id_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
            compute_kernel_id = compute_kernels_id_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            *reinterpret_cast<uint32_t *>(&decimal),
            *reinterpret_cast<uint32_t *>(&recip_p_decimal),
            num_cols_per_core,
            tile_offset,
            Ht,
            Wt,
            origin_h};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            output_addr, static_cast<uint32_t>(is_dram(output)), num_cols_per_core, tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{
            num_cols_per_core,
            Ht,
            origin_h,
            floored_p,
            static_cast<uint32_t>(p_is_negative),
            floored_recip_p,
            static_cast<uint32_t>(recip_p_is_negative)};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset += num_cols_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [reader_kernels_id = reader_kernels_id,
                                           writer_kernels_id = writer_kernels_id,
                                           compute_kernels_id_1 = compute_kernels_id_1,
                                           compute_kernels_id_2 = compute_kernels_id_2,
                                           num_cores_to_be_used = num_cores_to_be_used,
                                           num_cores_y = num_cores_y,
                                           core_group_1 = core_group_1,
                                           core_group_2 = core_group_2](
                                              const void *operation,
                                              Program &program,
                                              const std::vector<Tensor> &input_tensors,
                                              const std::vector<std::optional<const Tensor>> &,
                                              const std::vector<Tensor> &) {
        const auto p = static_cast<const MorehNorm *>(operation)->p;

        auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
        auto [floored_recip_p, recip_p_decimal, recip_p_is_negative] =
            get_floored_p_and_decimal_and_p_is_negative(1.0f / p);

        auto input_buffer = input_tensors.at(0).buffer();
        auto output_buffer = input_tensors.at(1).buffer();

        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
                runtime_args[0] = input_buffer->address();
                runtime_args[2] = *reinterpret_cast<uint32_t *>(&decimal);
                runtime_args[3] = *reinterpret_cast<uint32_t *>(&recip_p_decimal);
                SetRuntimeArgs(program, reader_kernels_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
                runtime_args[0] = output_buffer->address();
                SetRuntimeArgs(program, writer_kernels_id, core, runtime_args);
            }

            {
                KernelHandle compute_kernel_id;
                if (core_group_1.core_coord_in_core_ranges(core)) {
                    compute_kernel_id = compute_kernels_id_1;
                } else if (core_group_2.core_coord_in_core_ranges(core)) {
                    compute_kernel_id = compute_kernels_id_2;
                } else {
                    TT_THROW("Core not in specified core ranges.");
                }
                auto runtime_args = GetRuntimeArgs(program, compute_kernel_id, core);
                runtime_args[3] = floored_p;
                runtime_args[4] = static_cast<uint32_t>(p_is_negative);
                runtime_args[5] = floored_recip_p;
                runtime_args[6] = static_cast<uint32_t>(recip_p_is_negative);
                SetRuntimeArgs(program, compute_kernel_id, core, runtime_args);
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
