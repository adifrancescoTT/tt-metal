# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_val,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -120, 120, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -30, 20, device)

    pyt_y = torch.nn.functional.logsigmoid(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.log_sigmoid_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid_neg_inp(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -120, -1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 1, 50, device)

    pyt_y = torch.nn.functional.logsigmoid(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.log_sigmoid_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid_pos_inp(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 1, 50, device)

    pyt_y = torch.nn.functional.logsigmoid(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.log_sigmoid_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid_zero_inp(input_shapes, device):
    in_data, input_tensor = data_gen_with_val(input_shapes, device, True, 0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -50, 50, device)

    pyt_y = torch.nn.functional.logsigmoid(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.log_sigmoid_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid_zero_grad(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -50, 50, device, True)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, True, 0)
    pyt_y = torch.nn.functional.logsigmoid(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.log_sigmoid_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
