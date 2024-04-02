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
from models.utility_functions import (
    skip_for_wormhole_b0,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "order",
    [1, 2, 3, 6, 7, 10],
)
def test_bw_polygamma(input_shapes, order, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 10, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)
    n = order

    tt_output_tensor_on_device = tt_lib.tensor.polygamma_bw(grad_tensor, input_tensor, n)

    in_data.retain_grad()

    pyt_y = torch.polygamma(n, in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "order",
    [1, 4, 7, 10],
)
def test_bw_polygamma_range_pos(input_shapes, order, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -30, 30, device)
    n = order

    tt_output_tensor_on_device = tt_lib.tensor.polygamma_bw(grad_tensor, input_tensor, n)

    in_data.retain_grad()

    pyt_y = torch.polygamma(n, in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# grad and input zero
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "order",
    [2, 5],
)
@skip_for_wormhole_b0()
def test_bw_polygamma_zero(input_shapes, order, device):
    in_data, input_tensor = data_gen_with_val(input_shapes, device, True, 0)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, True, 0)
    n = order

    tt_output_tensor_on_device = tt_lib.tensor.polygamma_bw(grad_tensor, input_tensor, n)

    in_data.retain_grad()

    pyt_y = torch.polygamma(n, in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# grad zero
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "order",
    [2, 5],
)
def test_bw_polygamma_grad_zero(input_shapes, order, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, True, 0)
    n = order

    tt_output_tensor_on_device = tt_lib.tensor.polygamma_bw(grad_tensor, input_tensor, n)

    in_data.retain_grad()

    pyt_y = torch.polygamma(n, in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# input zero
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "order",
    [1, 2, 5],
)
def test_bw_polygamma_input_zero(input_shapes, order, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    in_data, input_tensor = data_gen_with_val(input_shapes, device, True, 0)
    n = order

    tt_output_tensor_on_device = tt_lib.tensor.polygamma_bw(grad_tensor, input_tensor, n)

    in_data.retain_grad()

    pyt_y = torch.polygamma(n, in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
