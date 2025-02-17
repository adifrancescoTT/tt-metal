// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// Utility function
uint32_t calculate_starting_idx_h (const Tensor& tensor, uint32_t num_slices, uint32_t slice_index) {
    if (num_slices <= 1) {
        return 0;
    }

    uint32_t num_tiles_height = tensor.volume() / tensor.get_legacy_shape()[-1] / TILE_HEIGHT;
    uint32_t num_tiles_width = tensor.get_legacy_shape()[-1] / TILE_WIDTH;
    uint32_t total_num_tiles = num_tiles_height * num_tiles_width;

    uint32_t num_tiles_per_slice = total_num_tiles / num_slices;
    uint32_t starting_tile_in_slice = num_tiles_per_slice * slice_index;
    return starting_tile_in_slice;
}

operation::ProgramWithCallbacks interleaved_to_sharded_multi_core(const Tensor& input, const Tensor& output, uint32_t num_slices, uint32_t slice_index) {
    tt_metal::Program program{};

    uint32_t num_units, num_units_per_shard, input_unit_size, output_unit_size, num_units_per_shard_width,
        num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_per_shard_height_last,
        num_units_per_shard_width_last;

    tt_metal::Device* device = input.device();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    auto shard_spec = output.shard_spec().value();
    auto shard_strategy = output.memory_config().memory_layout;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    CoreCoord end_core = (*shard_spec.grid.ranges().rbegin()).end;
    if (input.get_layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        input_unit_size = tt_metal::detail::TileSize(input_cb_data_format);
        output_unit_size = tt_metal::detail::TileSize(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.get_legacy_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = input.volume() / input.get_legacy_shape()[-1] / TILE_HEIGHT / num_slices;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        num_units = (input.volume() / input.get_legacy_shape()[-1] / shard_spec.shape[0]) *
                    (input.get_legacy_shape()[-1] / shard_spec.shape[1]);
        input_unit_size = shard_spec.shape[1] * input.element_size();
        output_unit_size = shard_spec.shape[1] * output.element_size();
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.get_legacy_shape()[-1] * input.element_size();
        num_units_offset = 1;
        uint32_t num_units_height = input.volume() / input.get_legacy_shape()[-1];
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            input_unit_size - (round_up(num_units_per_row, input_unit_size) - num_units_per_row);
    }

    bool convert_df = input_cb_data_format != output_cb_data_format;

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    auto all_cores = shard_spec.grid;
    uint32_t input_cb_index = CB::c_in0;
    uint32_t scratch_cb_index = CB::c_in1;
    uint32_t out_cb_index = input_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t output_page_size = align(output_unit_size, ADDRESS_ALIGNMENT);
    if (convert_df) {
        out_cb_index = CB::c_out0;
        uint32_t input_page_size = align(input_unit_size, ADDRESS_ALIGNMENT);
        tt_metal::CircularBufferConfig input_cb_out_config =
            tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{input_cb_index, input_cb_data_format}})
                .set_page_size(input_cb_index, input_page_size);
        auto cb_input = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_out_config);
    }
    tt_metal::CircularBufferConfig output_cb_out_config =
        tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);
    if (src_is_dram && input_unit_size % DRAM_ALIGNMENT != 0) {
        uint32_t scratch_cb_page_size = align(input_unit_size, DRAM_ALIGNMENT);
        tt_metal::CircularBufferConfig scratch_cb_out_config =
            tt_metal::CircularBufferConfig(1 * scratch_cb_page_size, {{scratch_cb_index, input_cb_data_format}})
                .set_page_size(scratch_cb_index, scratch_cb_page_size);
        auto cb_scratch = tt_metal::CreateCircularBuffer(program, all_cores, scratch_cb_out_config);
    }

    tt_metal::KernelHandle unary_reader_kernel_id;
    if (input.get_layout() == Layout::TILE) {
        std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)input_cb_index, (std::uint32_t)src_is_dram};

        unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(num_units_per_row);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(num_units_per_row) : 0;
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)input_cb_index,
            (std::uint32_t)scratch_cb_index,
            (std::uint32_t)src_is_dram,
            (std::uint32_t)src_stick_size_is_power_of_two,
            (std::uint32_t)src_log2_stick_size};

        unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/"
            "reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    }

    std::vector<uint32_t> writer_compile_time_args = {out_cb_index};
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt_metal::KernelHandle compute_kernel_id = 0;
    if (convert_df) {
        compute_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/compute/eltwise_copy.cpp",
            all_cores,
            tt_metal::ComputeConfig{});
    }

    uint32_t curr_idx_h = calculate_starting_idx_h(input, num_slices, slice_index);
    uint32_t curr_idx_w = 0;

    const auto cores = corerange_to_cores(shard_spec.grid, std::nullopt, rm_orientation);
    for (const auto& core : cores) {
        uint32_t curr_num_units_per_shard = num_units_per_shard;
        if (input.get_layout() == Layout::TILE) {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = num_units_per_shard_width;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core == end_core) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core == end_core) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            curr_num_units_per_shard = shard_height * shard_width;
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {src_buffer->address(),
                 shard_height,
                 shard_width,
                 num_units_offset,
                 curr_num_units_per_shard,
                 curr_idx_h + curr_idx_w});
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w == num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = input_unit_size;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                    curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                }
            }
            uint32_t padded_shard_width = align(shard_width, ADDRESS_ALIGNMENT);
            bool aligned = src_is_dram ? curr_idx_w % DRAM_ALIGNMENT == 0 : true;
            uint32_t aligned_width_offset, aligned_shard_width, aligned_offset;
            if (!aligned) {
                aligned_width_offset = round_down(curr_idx_w, DRAM_ALIGNMENT);
                aligned_offset = curr_idx_w - aligned_width_offset;
                aligned_shard_width = aligned_offset + shard_width;
            } else {
                aligned_width_offset = curr_idx_w;
                aligned_shard_width = shard_width;
                aligned_offset = 0;
            }

            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {src_buffer->address(), num_units_per_row, shard_height, shard_width, padded_shard_width, static_cast<uint32_t>(aligned), aligned_width_offset, aligned_shard_width, aligned_offset, curr_idx_h});
            curr_idx_w += input_unit_size;
            if (curr_idx_w == num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {curr_num_units_per_shard});
        if (convert_df) {
            tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {curr_num_units_per_shard});
        }
    }

    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cb_output, cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        auto shard_spec = output_tensors.at(0).shard_spec().value();
        auto all_cores = shard_spec.grid;

        for (const auto& core : cores) {
            {
                auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }
        }
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks sharded_to_interleaved_multi_core(const Tensor& input, const Tensor& output, uint32_t num_slices, uint32_t slice_index) {
    tt_metal::Program program{};

    uint32_t num_units, num_units_per_shard, input_unit_size, output_unit_size, num_units_per_shard_width,
        num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_per_shard_height_last,
        num_units_per_shard_width_last;

    tt_metal::Device* device = input.device();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    CoreCoord end_core = (*shard_spec.grid.ranges().rbegin()).end;
    if (output.get_layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        input_unit_size = tt_metal::detail::TileSize(input_cb_data_format);
        output_unit_size = tt_metal::detail::TileSize(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.get_legacy_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = output.volume() / output.get_legacy_shape()[-1] / TILE_HEIGHT / num_slices;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        num_units = (output.volume() / output.get_legacy_shape()[-1] / shard_spec.shape[0]) *
                    (input.get_legacy_shape()[-1] / shard_spec.shape[1]);
        input_unit_size = shard_spec.shape[1] * input.element_size();
        output_unit_size = shard_spec.shape[1] * output.element_size();
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.get_legacy_shape()[-1] * output.element_size();
        num_units_offset = 1;
        uint32_t num_units_height = input.volume() / input.get_legacy_shape()[-1];
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            output_unit_size - (round_up(num_units_per_row, output_unit_size) - num_units_per_row);
    }

    bool convert_df = input_cb_data_format != output_cb_data_format;

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t out_cb_index = src0_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t input_page_size = align(input_unit_size, ADDRESS_ALIGNMENT);
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, input_page_size)
            .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);
    if (convert_df) {
        out_cb_index = CB::c_out0;
        uint32_t output_page_size = align(output_unit_size, ADDRESS_ALIGNMENT);
        tt_metal::CircularBufferConfig output_cb_out_config =
            tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
                .set_page_size(out_cb_index, output_page_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);
    }

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    tt_metal::KernelHandle unary_writer_kernel_id;
    if (input.get_layout() == Layout::TILE) {
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)out_cb_index, (std::uint32_t)dst_is_dram};

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    } else {
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(num_units_per_row);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(num_units_per_row) : 0;
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t)out_cb_index,
            (std::uint32_t)dst_is_dram,
            (std::uint32_t)dst_stick_size_is_power_of_two,
            (std::uint32_t)dst_log2_stick_size};

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/"
            "writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }
    if (convert_df) {
        vector<uint32_t> compute_kernel_args = {num_units_per_shard};

        auto eltwise_unary_kernel_group_1 = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/eltwise_copy.cpp",
            all_cores,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args});
    }

    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, {num_units_per_shard});

    uint32_t curr_idx_h = calculate_starting_idx_h(output, num_slices, slice_index);
    uint32_t curr_idx_w = 0;

    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);
    for (const auto& core : cores) {
        if (input.get_layout() == Layout::TILE) {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = num_units_per_shard_width;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {dst_buffer->address(),
                 num_units_per_shard_height,
                 num_units_per_shard_width,
                 shard_height,
                 shard_width,
                 num_units_offset,
                 num_units_per_shard,
                 curr_idx_h + curr_idx_w});
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = output_unit_size;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            uint32_t padded_shard_width = align(shard_width, ADDRESS_ALIGNMENT);
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {dst_buffer->address(), num_units_per_row, shard_height, shard_width, padded_shard_width, curr_idx_w, curr_idx_h});
            curr_idx_w += output_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
    }
    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cores, num_slices](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        Buffer* dst_buffer = nullptr;
        if (num_slices > 1) {
            // If we have num_slices > 1, it means that our op is S->I partial.
            // And currently we store output tensors there as input[1]
            dst_buffer = input_tensors.at(1).buffer();
        } else {
            dst_buffer = output_tensors.at(0).buffer();
        }

        auto shard_spec = input_tensors.at(0).shard_spec().value();
        auto all_cores = shard_spec.grid;

        for (const auto& core : cores) {
            {
                auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

std::unordered_map<CoreCoord, std::vector<PageStride>> get_core_page_ranges(
    Buffer* input_buffer, Buffer* output_buffer) {
    const auto& output_shard_to_host_mapping = output_buffer->get_dev_page_to_host_page_mapping();
    const auto& input_page_to_local_page_mapping = input_buffer->get_host_page_to_local_shard_page_mapping();
    const auto& host_page_to_input_page_mapping = input_buffer->get_host_page_to_dev_page_mapping();

    auto num_pages = std::min<uint32_t>(output_shard_to_host_mapping.size(), input_buffer->num_pages());

    // First get output_core to vector< pair<input_core, input_page> (num_pages_in_output)
    std::unordered_map<CoreCoord, std::vector<std::pair<CoreCoord, uint32_t>>> output_core_to_vector_input_core_page;

    for (uint32_t output_page_id = 0; output_page_id < num_pages; output_page_id++) {
        auto output_core = output_buffer->get_core_from_dev_page_id(output_page_id);
        auto host_page = output_shard_to_host_mapping[output_page_id];
        auto input_page = host_page_to_input_page_mapping[host_page];
        auto local_input_page = input_page_to_local_page_mapping[host_page];
        auto input_core = input_buffer->get_core_from_dev_page_id(input_page);
        if (output_core_to_vector_input_core_page.find(output_core) == output_core_to_vector_input_core_page.end()) {
            output_core_to_vector_input_core_page[output_core] = {{input_core, local_input_page}};
        } else {
            output_core_to_vector_input_core_page[output_core].push_back({input_core, local_input_page});
        }
    }

    // now compress to output_core to vector<pair<input_core, input_page_range> (num_page_ranges_in_output)
    auto output_cores = corerange_to_cores(output_buffer->shard_spec().grid());
    std::unordered_map<CoreCoord, std::vector<PageStride>> ret_map;
    ret_map.reserve(output_cores.size());

    auto output_core_host_page_indices = output_buffer->core_host_page_indices();
    auto device = input_buffer->device();
    auto full_grid = device->compute_with_storage_grid_size();
    CoreCoord end_core = (*output_buffer->shard_spec().grid().ranges().rbegin()).end;
    uint32_t output_core_id;
    for (auto output_core : output_cores) {
        ret_map.try_emplace(output_core, std::vector<PageStride>{});


        const auto& input_cores_with_pages = output_core_to_vector_input_core_page.at(output_core);
        auto it = input_cores_with_pages.begin();
        const auto end = input_cores_with_pages.end();


        while (it != end) {
            const auto start_core = it->first;
            const auto start_page = it->second;
            auto expected_next_page = start_page + 1;
            Stride stride = Stride{.core = {0,0} , .data = 0};
            if ((it + 1) == end) {
                ret_map[output_core].push_back(PageStride{.start_core = start_core, .start_data=it->second,  .stride_size=1, .stride=stride, .num_strides=1});
                it = end;
            }
            else {
                //first get a single stride, go through the number of consecutive pages in the same core
                auto consecutive_it = it+1;
                auto last_it_consec = it;
                while(consecutive_it != end) {
                    auto next_input_page = *(consecutive_it);
                    auto curr_input_page = *(last_it_consec);
                    // diff core , not consecutive
                    if(curr_input_page.first != next_input_page.first) {
                        break;
                    }
                    //not consecutive
                    else if ((curr_input_page.second + 1) != next_input_page.second) {
                        break;
                    }
                    last_it_consec = consecutive_it;
                    consecutive_it = consecutive_it+1;
                }
                uint32_t stride_size = std::distance(it, last_it_consec) + 1;
                auto stride_it = it + stride_size;
                auto last_it_stride = stride_it - 1;

                // if stride_range is within same core
                // the jump in data is end of curr - end last stride
                // if stride range is in diff core
                // jump in data is curr - beginning of last stride
                uint32_t data_stride;
                if((stride_it != end) and (stride_it != it)){
                    // data stride within core
                    if(stride_it->first == last_it_stride->first and (stride_it->second > last_it_stride->second) ) {
                        auto next_input_page = *(stride_it);
                        auto prev_input_page = *(last_it_stride);
                        data_stride = next_input_page.second - prev_input_page.second - 1;
                        stride = Stride{.core = {next_input_page.first.x - prev_input_page.first.x, next_input_page.first.y - prev_input_page.first.y},
                                        .data = data_stride};
                    }
                    // strided core but same data
                    // currently only handling increasing cores within same stride
                    // TODO : negative strides for cores
                    else if((stride_it->first != last_it_stride->first) and (stride_it->first.x >= it->first.x and stride_it->first.y >= it->first.y) and (stride_it->second == it->second)) {
                    //else {
                        auto next_input_page = *(stride_it);
                        auto prev_input_page = *it;
                        data_stride = 0;
                        stride = Stride{.core = {next_input_page.first.x - prev_input_page.first.x, next_input_page.first.y - prev_input_page.first.y},
                                        .data = data_stride};
                    }
                    // diff data and diff core, not handled yet
                    else {
                        ret_map[output_core].push_back(PageStride{.start_core = start_core, .start_data=it->second,  .stride_size=stride_size, .stride=stride, .num_strides=1});
                        it = stride_it;
                        continue;
                    }
                    //TODO add stride of data and core
                }
                // only single stride
                else {
                    data_stride = 0;
                }

                TT_ASSERT(stride.core.x < full_grid.x and stride.core.y < full_grid.y);
                TT_ASSERT(data_stride < output_buffer->num_pages());
                auto stride_start = stride_it;
                uint32_t num_strides = 1;
                while(stride_it != end) {
                    bool stride_not_complete = false;
                    auto stride_it_inner = stride_it + 1;
                    auto last_it_stride_inner = stride_it;
                    for(uint32_t i=0; i<stride_size - 1; i++) {
                        auto next_input_page = *(stride_it_inner);
                        auto curr_input_page = *(last_it_stride_inner);
                        int increment = 1;
                        if(
                            (next_input_page.first != curr_input_page.first) or
                            ((int)next_input_page.second != (int)(curr_input_page.second) + (int)increment))
                        {
                            stride_not_complete = true;
                            break;
                        }
                        last_it_stride_inner = stride_it_inner;
                        stride_it_inner = stride_it_inner+1;
                    }
                    if(stride_not_complete) {
                        break;
                    }
                    num_strides++;
                    last_it_stride = stride_it_inner - 1;
                    stride_it = stride_it_inner;
                    if(stride_it == end) {
                        break;
                    }
                    auto next_input_page = *(stride_it);
                    auto curr_input_page = *(last_it_stride);
                    bool core_stride = ((stride.core.x != 0) or (stride.core.y != 0));

                    if((next_input_page.first.x - curr_input_page.first.x != stride.core.x) or
                        (next_input_page.first.y - curr_input_page.first.y != stride.core.y) or
                        (abs((int)next_input_page.second - (int)curr_input_page.second) != (int)stride.data))
                    {
                        break;
                    }
                }
                ret_map[output_core].push_back(PageStride{.start_core = start_core, .start_data=it->second,  .stride_size=stride_size, .stride=stride, .num_strides=num_strides});
                it = stride_it;
            }
        }
    }

    return ret_map;
}

operation::ProgramWithCallbacks reshard_multi_core(const Tensor& input, Tensor& output) {
    auto device = input.device();
    auto output_core_to_page_range_pair = get_core_page_ranges(input.buffer(), output.buffer());

    tt_metal::Program program{};

    auto input_shard_spec = input.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();
    auto all_cores = output_shard_spec.grid;

    auto grid = device->compute_with_storage_grid_size();
    uint32_t dst_cb_index = 16;
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reshard_reader.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig({dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y}));

    std::vector<uint32_t> writer_compile_time_args = {dst_cb_index};
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto cores = corerange_to_cores(all_cores);

    uint32_t total_size, page_size, unit_size;
    auto output_shard_shape = output_shard_spec.shape;
    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    if (input.get_layout() == Layout::TILE) {
        page_size = tt_metal::detail::TileSize(data_format);
        unit_size = page_size;
        total_size = output_shard_spec.numel() / TILE_HW * unit_size;
    } else {
        unit_size = output_shard_spec.shape[1] * output.element_size();
        page_size = output.get_legacy_shape()[-1] * output.element_size();
        total_size = output_shard_shape[0] * unit_size;
    }

    tt_metal::CircularBufferConfig cb_dst_config =
        tt_metal::CircularBufferConfig(
            total_size, {{dst_cb_index, data_format}})
            .set_page_size(dst_cb_index, unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);


    std::vector<uint32_t> physical_core_coords;
    physical_core_coords.reserve(grid.x * grid.y);
    for(uint32_t i=0; i<grid.x; i++){
        auto physical_input_core = device->worker_core_from_logical_core(CoreCoord(i,0));
        physical_core_coords.push_back(physical_input_core.x);
    }
    for(uint32_t i=0; i<grid.y; i++){
        auto physical_input_core = device->worker_core_from_logical_core(CoreCoord(0,i));
        physical_core_coords.push_back(physical_input_core.y);
    }


    for (auto core : cores) {
        auto page_stride_vector = output_core_to_page_range_pair.at(core);
        uint32_t num_ranges = page_stride_vector.size();
        std::vector<uint32_t> runtime_args = physical_core_coords;
        runtime_args.push_back(input.buffer()->address());
        runtime_args.push_back(0);
        runtime_args.push_back(num_ranges);
        runtime_args.push_back(page_size);
        uint32_t num_output_pages = 0;
        for (const auto& [start_core, start_data, stride_size, stride, num_strides] : page_stride_vector) {
            auto physical_input_core = device->worker_core_from_logical_core(start_core);
            uint32_t core_start_stride = (start_core.x << 24) | (start_core.y << 16) | (stride.core.x << 8) | stride.core.y;
            runtime_args.push_back((uint32_t)core_start_stride); //start_x
            uint32_t stride_data_start = (stride.data << 16) | (start_data);
            runtime_args.push_back((uint32_t)stride_data_start); //stride_data
            uint32_t stride_size_num_strides = (stride_size << 16) | (num_strides);
            runtime_args.push_back((uint32_t)stride_size_num_strides);  // stride_size
            num_output_pages += stride_size * num_strides;
        }
        runtime_args[physical_core_coords.size() + 1] = num_output_pages;
        tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {num_output_pages});
    }

    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, page_size, cb_dst0](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        auto output_core_to_page_range_pair = get_core_page_ranges(src_buffer, dst_buffer);

        auto input_shard_spec = input_tensors.at(0).shard_spec().value();
        auto output_shard_spec = output_tensors.at(0).shard_spec().value();
        auto all_cores = input_shard_spec.grid.merge(output_shard_spec.grid);
        auto cores = corerange_to_cores(all_cores);
        auto device = input_tensors.at(0).device();

        for (auto core : cores) {
            auto page_stride_vector = output_core_to_page_range_pair.at(core);
            uint32_t num_ranges = page_stride_vector.size();
            std::vector<uint32_t> runtime_args = {src_buffer->address(), 0, num_ranges};
            uint32_t num_output_pages = 0;
            for (const auto& [start_core, start_data, stride_size, stride, num_strides] : page_stride_vector) {
                auto physical_input_core = device->worker_core_from_logical_core(start_core);
                runtime_args.push_back(physical_input_core.x);
                runtime_args.push_back(physical_input_core.y);
                runtime_args.push_back(stride.core.x);
                runtime_args.push_back(stride.core.y);
                runtime_args.push_back(stride.data * page_size);                // start
                runtime_args.push_back(start_data * page_size);                // start
                runtime_args.push_back(stride_size * page_size);  // stride
                runtime_args.push_back(num_strides);  // stride
                num_output_pages += stride_size * num_strides;
            }
            runtime_args[1] = num_output_pages;
            tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {num_output_pages});
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
