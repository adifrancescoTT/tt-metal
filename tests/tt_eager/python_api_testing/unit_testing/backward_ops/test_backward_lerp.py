# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_lerp(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 201, device, True)
    end_data, end_tensor = data_gen_with_range(input_shapes, -199, 199, device, True)
    weight_data, weight_tensor = data_gen_with_range(input_shapes, -201, 201, device, True)

    tt_output_tensor_on_device = tt_lib.tensor.lerp_bw(grad_tensor, input_tensor, end_tensor, weight_tensor)

    in_data.retain_grad()
    end_data.retain_grad()

    pyt_y = torch.lerp(in_data, end_data, weight_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, end_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("weight", [-0.25, -25.0, 0.05, 1.0, 25.0])
def test_bw_lerp_weight_scalar(input_shapes, weight, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 201, device, True)
    end_data, end_tensor = data_gen_with_range(input_shapes, -199, 199, device, True)

    tt_output_tensor_on_device = tt_lib.tensor.lerp_bw(grad_tensor, input_tensor, end_tensor, weight)

    in_data.retain_grad()
    end_data.retain_grad()

    pyt_y = torch.lerp(in_data, end_data, weight)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, end_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
