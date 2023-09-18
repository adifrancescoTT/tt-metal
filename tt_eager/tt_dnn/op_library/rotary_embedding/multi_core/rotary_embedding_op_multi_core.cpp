// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks rotary_embedding_multi_core(const Tensor &input, const Tensor &cos, const Tensor &sin, Tensor &output, std::optional<uint32_t> token_idx) {
    Program program{};

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt_metal::detail::TileSize(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt_metal::detail::TileSize(sin_cb_data_format);

    tt::DataFormat scalar_cb_data_format = DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt_metal::detail::TileSize(scalar_cb_data_format);

    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t num_tiles = input.volume() / TILE_HW;
    uint32_t num_rows = input.volume()  / input.shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = input.shape()[-1] / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.shape()[-1] * sizeof(bfloat16);

    tt_metal::Device *device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_rows);


    uint32_t input_cb_index = 0;
    uint32_t num_input_tiles = 2 * Wt;
    auto cb_input = tt_metal::CreateCircularBuffers(
        program,
        input_cb_index,
        all_cores,
        num_input_tiles,
        num_input_tiles * input_single_tile_size,
        input_cb_data_format
    );

    uint32_t rotated_input_cb_index = 1;
    auto cb_rotated_input = tt_metal::CreateCircularBuffers(
        program,
        rotated_input_cb_index,
        all_cores,
        num_input_tiles,
        num_input_tiles * input_single_tile_size,
        input_cb_data_format
    );

    uint32_t num_cos_sin_tiles = 2 * Wt;
    uint32_t cos_cb_index = 2;
    auto cb_cos = tt_metal::CreateCircularBuffers(
        program,
        cos_cb_index,
        all_cores,
        num_cos_sin_tiles,
        num_cos_sin_tiles * cos_single_tile_size,
        cos_cb_data_format
    );

    uint32_t sin_cb_index = 3;
    auto cb_sin = tt_metal::CreateCircularBuffers(
        program,
        sin_cb_index,
        all_cores,
        num_cos_sin_tiles,
        num_cos_sin_tiles * sin_single_tile_size,
        sin_cb_data_format
    );

    // Used for bcast scalar
    uint32_t src_scalar_cb_index = 4;
    uint32_t num_scalar_tiles = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        src_scalar_cb_index,
        all_cores,
        num_scalar_tiles,
        num_scalar_tiles * scalar_single_tile_size,
        scalar_cb_data_format
    );

    uint32_t rotated_input_interm_cb_index = 24;
    auto cb_rotated_input_interm = tt_metal::CreateCircularBuffers(
        program,
        rotated_input_interm_cb_index,
        all_cores,
        num_input_tiles,
        num_input_tiles * input_single_tile_size,
        input_cb_data_format
    );

    uint32_t cos_interm_cb_index = 25;
    auto cb_cos_interm = tt_metal::CreateCircularBuffers(
        program,
        cos_interm_cb_index,
        all_cores,
        num_input_tiles,
        num_input_tiles * cos_single_tile_size,
        cos_cb_data_format
    );

    uint32_t sin_interm_cb_index = 26;
    auto cb_sin_interm = tt_metal::CreateCircularBuffers(
        program,
        sin_interm_cb_index,
        all_cores,
        num_input_tiles,
        num_input_tiles * sin_single_tile_size,
        sin_cb_data_format
    );


    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2 * Wt;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        all_cores,
        num_output_tiles,
        num_output_tiles * output_single_tile_size,
        output_cb_data_format
    );

    uint32_t untilized_cos_interm_cb_index = 27;
    uint32_t untilized_cos_sync_cb_index = 28;
    uint32_t untilized_sin_interm_cb_index = 29;
    uint32_t untilized_sin_sync_cb_index = 30;
    uint32_t retilized_cos_cb_index = 5;
    uint32_t retilized_sin_cb_index = 6;
    std::map<string, string> kernel_defines;
    if (token_idx.has_value()) {
        auto cb_cos2 = tt_metal::CreateCircularBuffers(
            program,
            retilized_cos_cb_index,
            all_cores,
            num_cos_sin_tiles,
            num_cos_sin_tiles * cos_single_tile_size,
            cos_cb_data_format
        );
        auto cb_sin2 = tt_metal::CreateCircularBuffers(
            program,
            retilized_sin_cb_index,
            all_cores,
            num_cos_sin_tiles,
            num_cos_sin_tiles * sin_single_tile_size,
            sin_cb_data_format
        );
        auto cb_untilized_cos_interm = tt_metal::CreateCircularBuffers(
            program,
            {untilized_cos_interm_cb_index, untilized_cos_sync_cb_index},
            all_cores,
            Wt,
            Wt * scalar_single_tile_size,
            scalar_cb_data_format
        );
        auto cb_untilized_sin_interm = tt_metal::CreateCircularBuffers(
            program,
            {untilized_sin_interm_cb_index, untilized_sin_sync_cb_index},
            all_cores,
            Wt,
            Wt * scalar_single_tile_size,
            scalar_cb_data_format
        );
        kernel_defines["DECODE_MODE"] = "1";
    }

    const uint16_t bfloat16_scalar = bfloat16(-1.0f).to_uint16();

    auto src_buffer = input.buffer();
    auto cos_buffer = cos.buffer();
    auto sin_buffer = sin.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool cos_is_dram = cos_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool sin_is_dram = sin_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)rotated_input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)cos_is_dram,
        (std::uint32_t)sin_is_dram,
        (std::uint32_t)bfloat16_scalar,
        (std::uint32_t)Ht,
        (std::uint32_t)Wt,
        (std::uint32_t)HtWt,
        (std::uint32_t)half_Wt,
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    if (token_idx.has_value()) {
        writer_compile_time_args.insert(writer_compile_time_args.end(), {
            untilized_cos_interm_cb_index,
            untilized_cos_sync_cb_index,
            untilized_sin_interm_cb_index,
            untilized_sin_sync_cb_index,
            Wt,
            Wbytes
        });
    }

    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_rotary_embedding_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args, .defines=kernel_defines});

    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_rotary_embedding_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args, .defines=kernel_defines});

	vector<uint32_t> compute_kernel_args = {
		(std::uint32_t)input_cb_index,
        (std::uint32_t)rotated_input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
		(std::uint32_t)num_rows_per_core_group_1,
        (std::uint32_t)Wt,
        (std::uint32_t)half_Wt
	};
    if (token_idx.has_value()) {
        compute_kernel_args.insert(compute_kernel_args.end(), {
            (std::uint32_t)untilized_cos_interm_cb_index,
            (std::uint32_t)untilized_cos_sync_cb_index,
            (std::uint32_t)untilized_sin_interm_cb_index,
            (std::uint32_t)untilized_sin_sync_cb_index,
            (std::uint32_t)retilized_cos_cb_index,
            (std::uint32_t)retilized_sin_cb_index
        });
    }

	auto rotary_embedding_kernel_group_1_id = tt_metal::CreateComputeKernel(
		program,
		"tt_metal/kernels/compute/rotary_embedding.cpp",
		core_group_1,
		tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines=kernel_defines}
	);
    if(!core_group_2.ranges().empty()) {
        compute_kernel_args[9] = num_rows_per_core_group_2;

        auto rotary_embedding_kernel_group_2_id = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/rotary_embedding.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines=kernel_defines}
        );
    }
    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    if (token_idx.has_value()) {
        cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_rows_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        if (!token_idx.has_value()) {
            cos_sin_start_id = num_tiles_written % HtWt;
        }

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                cos_buffer->address(),
                sin_buffer->address(),
                num_rows_per_core,
                num_tiles_written,
                num_tiles_written / Wt % Ht,
                cos_sin_start_id
            }
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                num_rows_per_core * Wt,
                num_tiles_written,
                cos_sin_offset,
                Wt, Wbytes
            }
        );
        num_tiles_written += num_rows_per_core * Wt;
    }

    auto override_runtime_arguments_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            num_cores,
            num_cores_y,
            core_group_1,
            num_rows_per_core_group_1,
            core_group_2,
            num_rows_per_core_group_2,
            Wbytes,
            Wt,
            HtWt
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        const auto token_idx = static_cast<const RotaryEmbedding*>(operation)->token_idx;

        auto src_buffer = input_tensors.at(0).buffer();
        auto cos_buffer = input_tensors.at(1).buffer();
        auto sin_buffer = input_tensors.at(2).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        uint32_t cos_sin_offset = 0;
        uint32_t cos_sin_start_id = 0;
        if (token_idx.has_value()) {
            cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
            cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
        }

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            uint32_t num_rows_per_core;
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_rows_per_core = num_rows_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_rows_per_core = num_rows_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }
            if (!token_idx.has_value()) {
                cos_sin_start_id = num_tiles_written % HtWt;
            }

            {
                auto runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = cos_buffer->address();
                runtime_args[2] = sin_buffer->address();
                runtime_args[6] = cos_sin_start_id;
                SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                runtime_args[3] = cos_sin_offset;
                SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
            }
            num_tiles_written += num_rows_per_core * Wt;
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
