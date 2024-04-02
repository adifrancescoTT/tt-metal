# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
@pytest.mark.parametrize("sizes", [[12, 1, 1, 1], [6, 1, 1, 1], [1, 24, 1, 1], [1, 3, 1, 1]])
def test_bw_repeat(input_shapes, sizes, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)

    pyt_y = in_data.repeat(sizes)

    grad_data, grad_tensor = data_gen_pt_tt(pyt_y.shape, device, True)

    tt_output_tensor_on_device = tt_lib.tensor.repeat_bw(grad_tensor, input_tensor, sizes)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
