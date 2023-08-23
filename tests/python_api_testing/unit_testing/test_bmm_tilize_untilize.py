import sys
import pytest
import itertools

from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np
import torch

import tt_lib as ttl
from models.utility_functions import (
    tilize_to_list,
    untilize,
    comp_pcc,
)

TILE_HEIGHT = TILE_WIDTH = 32

## parameters
# matrix sizes as number of blocks along h and w:
a_height_nblocks = [1, 7]
a_width_nblocks = [1, 7]
b_width_nblocks = [1, 7]
# block sizes as number of tiles along h and w:
a_block_height_ntiles = [4]
a_block_width_ntiles = [4]
b_block_width_ntiles = [4]
# output sublobcking per block:
out_subblock_height_ntiles = [4]  ## == a_block_height_ntiles, <= 8
out_subblock_width_ntiles = [2]  ## == b_block_width_ntiles, <= 8
tilize_a = [True, False]
untilize_out = [True, False]

@pytest.mark.parametrize(
    "a_height_nblocks, a_width_nblocks, b_width_nblocks,\
     a_block_height_ntiles, a_block_width_ntiles, b_block_width_ntiles,\
     out_subblock_height_ntiles, out_subblock_width_ntiles,\
     tilize_a, untilize_out",
    itertools.product(
        a_height_nblocks,
        a_width_nblocks,
        b_width_nblocks,
        a_block_height_ntiles,
        a_block_width_ntiles,
        b_block_width_ntiles,
        out_subblock_height_ntiles,
        out_subblock_width_ntiles,
        tilize_a,
        untilize_out,
    ),
)
@pytest.mark.parametrize(
    "out_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["out_BFLOAT8_B", "out_BFLOAT16"],
)
@pytest.mark.parametrize(
    "b_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["b_BFLOAT8_B", "b_BFLOAT16"],
)
@pytest.mark.parametrize(
    "a_dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16),
    ids=["a_BFLOAT8_B", "a_BFLOAT16"],
)
@pytest.mark.parametrize(
    "has_bias",
    (False, True,)
)
def test_run_bmm_single_core_tilize_untilize(
    a_height_nblocks,
    a_width_nblocks,
    b_width_nblocks,
    a_block_height_ntiles,
    a_block_width_ntiles,
    b_block_width_ntiles,
    out_subblock_height_ntiles,
    out_subblock_width_ntiles,
    tilize_a,
    untilize_out,
    has_bias,
    a_dtype,
    b_dtype,
    out_dtype,
    device,
):
    if (tilize_a and a_dtype != ttl.tensor.DataType.BFLOAT16) or \
        (untilize_out and out_dtype != ttl.tensor.DataType.BFLOAT16):
        print(f"invalid case, skipping.")
        pytest.skip()

    if tilize_a and a_dtype != out_dtype:
        print(False and "Case to debug. skipping for now.")
        pytest.skip()

    if untilize_out and has_bias:
        print("Bias with untilize out is not supported.")
        pytest.skip()

    ## TODO (AS): Certain mixed-prec cases do not yet work. Skip them here (these are currently asserted out in the op.)
    if (
        not (
            a_dtype == out_dtype
            and a_dtype == b_dtype
            and a_dtype == ttl.tensor.DataType.BFLOAT16
        )
    ) and (tilize_a or untilize_out):
        print(f"TODO: Mixed-prec case to be debugged. Skipping for now.")
        pytest.skip()

    ## TODO (AS): Support multi-precision with as well. Currently bias only works for BFLOAT16
    if has_bias and (a_dtype != ttl.tensor.DataType.BFLOAT16 or b_dtype != ttl.tensor.DataType.BFLOAT16 or out_dtype != ttl.tensor.DataType.BFLOAT16):
        print(f"TODO: Support multi-precision with bias. Skipping for now.")
        pytest.skip()

    a_batch = b_batch = 1
    a_channel = b_channel = 1
    a_height = a_height_nblocks * a_block_height_ntiles * TILE_HEIGHT
    a_width = a_width_nblocks * a_block_width_ntiles * TILE_WIDTH  # == b_height
    b_width = b_width_nblocks * b_block_width_ntiles * TILE_WIDTH
    a_shape = [a_batch, a_channel, a_height, a_width]
    b_shape = [b_batch, b_channel, a_width, b_width]
    out_shape = [a_batch, a_channel, a_height, b_width]

    torch.manual_seed(0)
    a = torch.randn(a_shape, dtype=torch.bfloat16).float()
    b = torch.randn(b_shape, dtype=torch.bfloat16).float()
    # b = torch.zeros(b_shape, dtype=torch.bfloat16).float()

    if tilize_a:
        ## a in row-major
        assert (
            not (a_dtype == ttl.tensor.DataType.BFLOAT8_B)
            and "Row-major format does not support BFLOAT8_B datatype!"
        )
        a_layout = ttl.tensor.Layout.ROW_MAJOR
        a_list = a.flatten().tolist()
    else:
        ## a in tile
        a_layout = ttl.tensor.Layout.TILE
        a_list = tilize_to_list(a)
    tta = ttl.tensor.Tensor(a_list, a_shape, a_dtype, a_layout, device)

    ## tensor b, in tile format
    ttb = ttl.tensor.Tensor(
        tilize_to_list(b), b_shape, b_dtype, ttl.tensor.Layout.TILE, device
    )

    bias, ttbias = None, None
    # if has_bias:
    bias_shape = [a_batch, 1, 1, b_width]
    bias = torch.randn(bias_shape, dtype=torch.bfloat16).float()
    # bias = torch.zeros(bias_shape, dtype=torch.bfloat16).float()
    # bias = torch.ones(bias_shape, dtype=torch.bfloat16).float()
    ttbias = ttl.tensor.Tensor(torch.flatten(bias).tolist(), bias_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR)
    ttbias = ttbias.pad_to_tile(0).to(ttl.tensor.Layout.TILE).to(device)

    ## tensor out format checks
    if untilize_out:
        ## out in row-major
        assert (
            not (out_dtype == ttl.tensor.DataType.BFLOAT8_B)
            and "Row-major format does not support BFLOAT8_B datatype!"
        )
    else:
        ## out in tile
        pass

    torch.set_printoptions(
        precision=2, threshold=10000, sci_mode=False, edgeitems=80, linewidth=400
    )

    # tta_pytorch = untilize(tta.to_torch())
    # print(f'a slice: {tta_pytorch[0, 0, 0:32*a_block_height_ntiles*a_height_nblocks:32*a_block_height_ntiles, 0:32*a_width_nblocks*a_block_width_ntiles:1]}')

    # ttb_pytorch = untilize(ttb.to_torch())
    # print(f'b slice: {ttb_pytorch[0, 0, 0:32*a_block_width_ntiles*a_width_nblocks:32, 0:32*b_width_nblocks*b_block_width_ntiles:1]}')

    ## compute out
    out = ttl.tensor.bmm_tilize_untilize(
        tta,
        ttb,
        ttbias,
        out_dtype,
        a_height_nblocks,
        a_width_nblocks,
        b_width_nblocks,
        a_block_height_ntiles,
        a_block_width_ntiles,
        b_block_width_ntiles,
        out_subblock_height_ntiles,
        out_subblock_width_ntiles,
        tilize_a,
        untilize_out,
        has_bias,
    )
    out = out.cpu().to_torch()
    if not untilize_out:
        ## output is in tiled format
        out = untilize(out)

    # print(f'returned output: {out}')

    ## reference
    golden_pytorch = torch.matmul(a, b)
    if has_bias:
        golden_pytorch += bias

    # print("golden out:\n", golden_pytorch)

    ## test for equivalance
    assert out.shape == golden_pytorch.shape
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out)
    print(f"Passing PCC = {passing_pcc}")
    print(f"Output PCC = {output_pcc}")

    assert passing_pcc
