# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from loguru import logger


import numpy as np

import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
)
import torch
import pytest


def run_bert_large_ff2_matmul_test(device, dtype, in0_mem_config, in1_mem_config, bias_mem_config, out_mem_config):
    compute_grid_size = device.compute_with_storage_grid_size()
    if compute_grid_size.x < 12:
        pytest.skip(f"Grid size {compute_grid_size} is not supported")

    torch.manual_seed(1234)

    a_shape = [9, 1, 384, 4096]
    b_shape = [1, 1, 4096, 1024]
    bias_shape = [1, 1, 1, 1024]
    bias_pad_shape = [1, 1, 32, 1024]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95
    BIAS = torch.randint(-20, 20, bias_shape, dtype=torch.float)

    a_t = (
        ttl.tensor.Tensor(
            A.flatten().tolist(),
            a_shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in0_mem_config)
    )
    b_t = (
        ttl.tensor.Tensor(
            B.flatten().tolist(),
            b_shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, in1_mem_config)
    )

    if bias_mem_config is not None:
        bias_t = (
            ttl.tensor.Tensor(
                BIAS.flatten().tolist(),
                bias_shape,
                dtype,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .pad(bias_pad_shape, [0, 0, 0, 0], 0)
            .to(ttl.tensor.Layout.TILE)
            .to(device, bias_mem_config)
        )
    else:
        bias_t = None

    t2 = ttl.tensor.bert_large_ff2_matmul(a_t, b_t, bias_t, out_mem_config)

    # Check memory of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    if bias_mem_config is not None:
        assert bias_t.memory_config().buffer_type == bias_mem_config.buffer_type
    assert t2.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0 is on: {a_t.memory_config().buffer_type}")
    logger.debug(f"in1 is on: {b_t.memory_config().buffer_type}")
    if bias_mem_config is not None:
        logger.debug(f"bias is on: {bias_t.memory_config().buffer_type}")
    logger.debug(f"out is on: {t2.memory_config().buffer_type}")

    assert t2.get_legacy_shape() == [9, 1, 384, 1024]
    tt_host_rm = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    pyt_got_back_rm = tt_host_rm.to_torch()

    ref_bmm = torch.matmul(A, B)
    if bias_mem_config is not None:
        ref_bmm = ref_bmm + BIAS
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "bias_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        None,
    ),
    ids=["bias_DRAM", "bias_L1", "bias_None"],
)
@pytest.mark.parametrize(
    "in1_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in1_DRAM", "in1_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
def test_bert_large_ff2_matmul_test(
    device,
    dtype,
    in0_mem_config,
    in1_mem_config,
    bias_mem_config,
    out_mem_config,
    request,
):
    run_bert_large_ff2_matmul_test(device, dtype, in0_mem_config, in1_mem_config, bias_mem_config, out_mem_config)


def test_bert_large_ff2_matmul_with_program_cache(device, use_program_cache):
    dtype = ttl.tensor.DataType.BFLOAT8_B
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    for _ in range(2):
        run_bert_large_ff2_matmul_test(
            device,
            dtype,
            mem_config,
            mem_config,
            mem_config,
            mem_config,
        )
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    for _ in range(2):
        run_bert_large_ff2_matmul_test(
            device,
            dtype,
            mem_config,
            mem_config,
            mem_config,
            mem_config,
        )
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    assert device.num_program_cache_entries() == 2
