# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tt_lib as ttl
from tt_lib import tensor as tt

from models.utility_functions import skip_for_wormhole_b0, torch2tt_tensor, tt2torch_tensor


def dump_tensor_to_file(tensor, file_path):
    # Convert the tensor to numpy for easier manipulation
    tensor_np = tensor.to(torch.float32).squeeze().numpy()

    with open(file_path, "w") as f:
        for row in tensor_np:
            formatted_row = [f"{val:.4e}" for val in row]
            f.write(",".join(formatted_row) + "\n")


@skip_for_wormhole_b0()
def test_scan(device):
    torch.manual_seed(0)

    shape = (128, 1024)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    # dump_tensor_to_file(torch_input, "torch_input.csv")

    shard_grid = tt.CoreRangeSet(
        {
            tt.CoreRange(
                tt.CoreCoord(0, 0),
                tt.CoreCoord(1, 0),
            )
        }
    )

    shard_spec = tt.ShardSpec(shard_grid, [64, 1024], tt.ShardOrientation.ROW_MAJOR, False)

    tt_input = torch2tt_tensor(
        torch_input,
        device,
        tt.Layout.TILE,
        tt_memory_config=tt.MemoryConfig(tt.TensorMemoryLayout.HEIGHT_SHARDED, tt.BufferType.L1, shard_spec),
    )

    tt.scan(tt_input)

    after_scan = tt2torch_tensor(tt_input)
    # dump_tensor_to_file(after_scan, "after_scan.csv")
