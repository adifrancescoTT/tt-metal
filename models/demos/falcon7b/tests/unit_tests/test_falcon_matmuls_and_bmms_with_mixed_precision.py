# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor, skip_for_wormhole_b0
import torch
import math


def run_falcon_matmul_test(
    falcon_op,
    seq_len,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    device,
):
    pcc = 0.99
    if out_dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.98

    if falcon_op == "ttl.tensor.falcon_fused_qkv_matmul":
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 4672]
        expected_output_shape = [1, 1, seq_len, 4672]
        op = ttl.tensor.falcon_fused_qkv_matmul
        in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
        in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    elif falcon_op == "ttl.tensor.falcon_selfout_matmul":
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 4544]
        expected_output_shape = [1, 1, seq_len, 4544]
        op = ttl.tensor.falcon_selfout_matmul
    elif falcon_op == "ttl.tensor.falcon_dense_4h_to_h_matmul":
        a_shape = [1, 1, seq_len, 18176]
        b_shape = [1, 1, 18176, 4544]
        expected_output_shape = [1, 1, seq_len, 4544]
        op = ttl.tensor.falcon_dense_4h_to_h_matmul

        if (seq_len == 1024 and in0_dtype == in1_dtype == out_dtype == ttl.tensor.DataType.BFLOAT16) or (
            seq_len == 2048
            and (
                in0_dtype == ttl.tensor.DataType.BFLOAT16
                or in1_dtype == ttl.tensor.DataType.BFLOAT16
                or out_dtype == ttl.tensor.DataType.BFLOAT16
            )
        ):
            logger.warning(
                f"For seq_len: {seq_len}, in0_dtype: {in0_dtype}, in1_dtype: {in1_dtype}, and out_dtype: {out_dtype}, L1 space is not enough. Running with in0, in1, and out on DRAM instead!"
            )
            in0_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            in1_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            out_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
    elif falcon_op == "ttl.tensor.falcon_dense_h_to_4h_matmul":
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 18176]
        expected_output_shape = [1, 1, seq_len, 18176]
        op = ttl.tensor.falcon_dense_h_to_4h_matmul

        in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

        if seq_len == 2048 and out_dtype == ttl.tensor.DataType.BFLOAT16:
            logger.warning(
                f"For seq_len: {seq_len}, in0_dtype: {in0_dtype}, in1_dtype: {in1_dtype}, and out_dtype: {out_dtype}, L1 space is not enough. Running with in0, in1, and out on DRAM instead!"
            )
            in0_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            in1_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            out_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
    elif falcon_op == "ttl.tensor.falcon_lm_head_matmul":
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 65024]
        expected_output_shape = [1, 1, seq_len, 65024]
        op = ttl.tensor.falcon_lm_head_matmul

        if (
            seq_len == 512
            and (
                in0_dtype == ttl.tensor.DataType.BFLOAT16
                or in1_dtype == ttl.tensor.DataType.BFLOAT16
                or out_dtype == ttl.tensor.DataType.BFLOAT16
            )
            or seq_len == 1024
            or seq_len == 2048
        ):
            logger.warning(
                f"For seq_len: {seq_len}, in0_dtype: {in0_dtype}, in1_dtype: {in1_dtype}, and out_dtype: {out_dtype}, L1 space is not enough. Running with in0, in1, and out on DRAM instead!"
            )
            in0_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            in1_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
            out_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
            )
    elif falcon_op == "post_transpose_mm":
        # attn matmul, post-transpose
        a_shape = [1, 71, seq_len, 64]
        b_shape = [1, 1, 64, seq_len]
        expected_output_shape = [1, 71, seq_len, seq_len]
        op = ttl.tensor.matmul

        in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    elif falcon_op == "post_softmax_mm":
        # attn matmul, post-softmax
        a_shape = [1, 71, seq_len, seq_len]
        b_shape = [1, 1, seq_len, 64]
        expected_output_shape = [1, 71, seq_len, 64]

        op = ttl.tensor.matmul

        in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
        out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    else:
        raise NotImplementedError(f"falcon matmul op is undefined!")

    torch.manual_seed(1234)

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = ttl.tensor.Tensor(A, in0_dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)
    b_t = ttl.tensor.Tensor(B, in1_dtype).to(ttl.tensor.Layout.TILE).to(device, in1_mem_config)
    bias_t = None

    if falcon_op == "ttl.tensor.falcon_dense_4h_to_h_matmul":
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # A = [32, 568] * [568, 142]
        # Arround 1.6 mil cycles
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            in0_block_w=8,
            per_core_M=4,
            per_core_N=18,
            out_subblock_h=1,
            out_subblock_w=6,
            transpose_mcast=False,
            fused_activation=None,
        )

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )

        # out = falcon_op(a_t, b_t, bias_t, output_mem_config=out_mem_config, output_dtype=out_dtype)
    elif falcon_op == "ttl.tensor.falcon_dense_h_to_4h_matmul":
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # # A = [32, 142] B = [142, 568]
        # # Split on 8 cores for x -> Am = [4, 142], per_core_M = 4
        # # Split on 8 cores for y -> Bn = [142, 71], per_core_N = 71
        # program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w=2,
        #     per_core_M=4,
        #     per_core_N=71,
        #     out_subblock_h=4,
        #     out_subblock_w=1,
        #     transpose_mcast=False,
        #     fused_activation=None,
        # )

        # best config ~ 2.6 mil cycles
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            in0_block_w=2,
            per_core_M=32,
            per_core_N=9,
            out_subblock_h=2,
            out_subblock_w=3,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        # This is not working
        # Theoretically:
        # A = [32, 142] B = [142, 568]
        # per_core_M = 16 => num_x_cores=2, per_core_A = [16, 142]
        # per_core_N = 18 => num_y_cores=32, per_core_B = [142, 18]
        # Total num_cores = num_x_cores * num_y_cores = 64
        # 1D doesn't assert with this, but PCC is bad
        # 2D asserts as it has constraints such that max_num_x_cores = 8 and max_num_y_cores = 8
        # Note: 568 / 18 == 17.66 => pad to 18
        # program_config =ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w = 2,
        #     per_core_M = 16,
        #     per_core_N= 18,
        #     out_subblock_h = 1,
        #     out_subblock_w = 1,
        #     fuse_batch = True,
        #     fused_activation = None,
        #     mcast_in0 = True
        # )

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    elif falcon_op == "post_transpose_mm":
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,  # try this lower
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w=8,
        #     per_core_M=4,
        #     per_core_N=18,
        #     out_subblock_h=1,
        #     out_subblock_w=6,
        #     transpose_mcast=False,
        #     fused_activation=None,
        # )
        # Dram bound
        # Dimensions:
        #         a_shape = [1, 71, seq_len, 64]
        #         b_shape = [1, 1, 64, seq_len]
        #         entire batch doesn't fit
        # Arround ~2.6 mil cycles
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            in0_block_w=1,
            per_core_M=4,
            per_core_N=4,
            out_subblock_h=1,
            out_subblock_w=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        # program_config =ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w = 2,
        #     per_core_M = 1,
        #     per_core_N= 32,
        #     out_subblock_h = 1,
        #     out_subblock_w = 1,
        #     fuse_batch = False,
        #     fused_activation = None,
        #     mcast_in0 = False
        # )

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )

        # out = ttl.tensor.matmul(
        #     a_t,
        #     b_t,
        #     output_mem_config = out_mem_config
        # )
    elif falcon_op == "post_softmax_mm":
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4,  # try this lower
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Optimum configuration for this case...
        # The matmul is DRAM bound
        # ~ 2.9 mil ns, theoretical is something like 181760ns!!!
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            in0_block_w=1,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=36,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    elif falcon_op == "ttl.tensor.falcon_selfout_matmul":
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,  # try this lower
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # A = [32, 142] B = [142, 142]
        # Best config - 475k ns
        # Theoretical: 252K
        # usage = theoretical / actual = 53%
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            in0_block_w=2,
            per_core_M=4,
            per_core_N=18,
            out_subblock_h=1,
            out_subblock_w=6,
            transpose_mcast=False,
            fused_activation=None,
        )

        # if we pad A to [32, 144] and B to [144, 142]
        # we get 395K cycles
        # Theoretical: 259k
        # Usage: 65% (first one - theoretical_first/actual = 64%)
        # program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w=16,
        #     per_core_M=4,
        #     per_core_N=18,
        #     out_subblock_h=1,
        #     out_subblock_w=6,
        #     transpose_mcast=False,
        #     fused_activation=None,
        # )

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    elif falcon_op == "ttl.tensor.falcon_fused_qkv_matmul":
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,  # try this lower
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # A = [32, 142] B = [142, 146]
        # Best config - 630K ns
        # Theoretical: 259K - 64 cores, 296k - 56 cores
        # usage = theoretical / actual = 41% (47% 56 cores)
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            in0_block_w=2,
            per_core_M=4,
            per_core_N=21,
            out_subblock_h=1,
            out_subblock_w=7,
            transpose_mcast=False,
            fused_activation=None,
        )

        # Experiment: Pad inner dim to 144 (144 * 32 = 4608), to take advantage of in0_block_w
        # This way we get 540k ns, which is (296 / 540) = 55%
        # Max in0_block_w = 18
        # program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        #     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        #     in0_block_w=18,
        #     per_core_M=4,
        #     per_core_N=21,
        #     out_subblock_h=1,
        #     out_subblock_w=7,
        #     transpose_mcast=False,
        #     fused_activation=None,
        # )

        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )

    else:
        out = op(a_t, b_t, bias_t, output_mem_config=out_mem_config, output_dtype=out_dtype)

    # Check memory and dtype of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert a_t.get_dtype() == in0_dtype
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    assert b_t.get_dtype() == in1_dtype
    assert out.memory_config().buffer_type == out_mem_config.buffer_type
    assert out.get_dtype() == out_dtype
    logger.debug(f"in0 ({a_shape}): {a_t.memory_config().buffer_type} and {a_t.get_dtype()}")
    logger.debug(f"in1 ({b_shape}): {b_t.memory_config().buffer_type} and {b_t.get_dtype()}")
    logger.debug(f"out ({expected_output_shape}): {out.memory_config().buffer_type} and {out.get_dtype()}")

    assert out.get_legacy_shape() == expected_output_shape
    pyt_got_back_rm = tt2torch_tensor(out)

    ref_bmm = torch.matmul(A, B)

    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, pcc)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "in0_mem_config, in1_mem_config, out_mem_config",
    (
        (
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ),
    ),
    ids=["weights_DRAM"],
)
@pytest.mark.parametrize(
    "out_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["out_BFLOAT8_B", "out_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in1_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in1_BFLOAT8_B", "in1_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in0_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["in0_BFLOAT8_B", "in0_BFLOAT16"],
)
@pytest.mark.parametrize(
    "falcon_op",
    (
        "ttl.tensor.falcon_fused_qkv_matmul",
        "ttl.tensor.falcon_selfout_matmul",
        "ttl.tensor.falcon_dense_4h_to_h_matmul",
        "ttl.tensor.falcon_dense_h_to_4h_matmul",
        "ttl.tensor.falcon_lm_head_matmul",
        "post_transpose_mm",
        "post_softmax_mm",
    ),
    ids=["fused_qkv", "selfout", "dense_4h_to_h", "dense_h_to_4h", "lm_head", "post_transpose_mm", "post_softmax_mm"],
)
@pytest.mark.parametrize(
    "seq_len",
    (128, 256, 512, 1024, 2048),
    ids=["seq_len_128", "seq_len_256", "seq_len_512", "seq_len_1024", "seq_len_2048"],
)
def test_falcon_matmul(
    falcon_op,
    seq_len,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    request,
    device,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    is_e75_grid_size = (compute_grid_size.x * compute_grid_size.y) == 88
    if is_e75_grid_size and (seq_len == 512) and (falcon_op == ttl.tensor.falcon_lm_head_matmul):
        pytest.skip(f"LM Head does not work on E75 grid size {compute_grid_size}")

    ttl.profiler.set_profiler_location(f"falcon_{request.node.callspec.id}")
    run_falcon_matmul_test(
        falcon_op,
        seq_len,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        device,
    )


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_wormhole_b0("non-determinstic hang, see issue #5882")
@pytest.mark.parametrize("seq_len", [1024], ids=["seq_len_1024"])
@pytest.mark.parametrize("num_slices", [4], ids=["four_slices"])
@pytest.mark.parametrize("num_cores", [64])
def test_falcon7b_attnention_sliced(
    device,
    seq_len,
    num_slices,
    num_cores,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    attention_mask_shape = [1, 71, seq_len, seq_len]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_attention_mask = torch.randn(attention_mask_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )
    attention_mask = torch2tt_tensor(
        torch_attention_mask,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttl.tensor.DataType.BFLOAT16,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    passing = True
    output = None

    # Todo: 2K seq_len
    if seq_len == 1024:
        mm_out = torch2tt_tensor(
            torch_attention_mask,
            device,
            tt_memory_config=dram_interleaved_memory_config,
            tt_dtype=ttl.tensor.DataType.BFLOAT16,
        )
        tiles_per_shard = math.ceil((((71 * seq_len) / num_cores) / num_slices) / 32)
        mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
        mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

        for i in range(num_slices):
            slice = ttl.tensor.interleaved_to_sharded_partial(
                reference_query_layer,
                grid_size,
                mm_activations_height_shard_spec,
                num_slices,  # num_slices
                i,  # slice_index
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.ShardOrientation.ROW_MAJOR,
            )

            program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid_size,
                in0_block_w=2,
                per_core_M=tiles_per_shard,
                per_core_N=32,
                out_subblock_h=1,
                out_subblock_w=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )

            # [1, 1, 71, 32, 2] * [2, 32]
            mm_slice = ttl.operations.primary.matmul(
                slice,
                reference_key_layer_transposed,
                program_config=program_config,
                output_mem_config=height_sharded_memory_config,
                output_dtype=ttl.tensor.DataType.BFLOAT16,
                compute_kernel_config=compute_kernel_config,
            )

            slice.deallocate()

            attn_mask_slice = ttl.tensor.interleaved_to_sharded_partial(
                attention_mask,
                grid_size,
                mm_output_height_shard_spec,
                num_slices,
                i,
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.ShardOrientation.ROW_MAJOR,
            )

            mm_slice = ttl.operations.primary.add(
                mm_slice,
                attn_mask_slice,
                fused_activations=None,
                output_mem_config=height_sharded_memory_config,
                output_dtype=ttl.tensor.DataType.BFLOAT16,
                in_place=True,
            )

            attn_mask_slice.deallocate()

            softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid_size,
                subblock_w=1,
                block_h=mm_output_height_shard_spec[0] // 32,
                block_w=mm_output_height_shard_spec[1] // 32,
                math_fidelity=ttl.tensor.MathFidelity.HiFi4,
                im_data_format=ttl.tensor.DataType.BFLOAT16,
            )

            mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

            ttl.tensor.sharded_to_interleaved_partial(
                mm_slice,
                mm_out,
                num_slices,
                i,
                dram_interleaved_memory_config,
            )

            mm_slice.deallocate()

        mm_out_torch = tt2torch_tensor(mm_out)

        attn_weights = ttl.tensor.matmul(
            reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
        )

        attn_weights = ttl.tensor.add(attn_weights, attention_mask, output_mem_config=dram_interleaved_memory_config)
        attn_weights = ttl.operations.primary.softmax_in_place(attn_weights)

        attn_weights_torch = tt2torch_tensor(attn_weights)
        passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing
