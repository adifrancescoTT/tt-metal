# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
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
def test_bw_cosh(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -9, 9, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)

    tt_output_tensor_on_device = tt_lib.tensor.cosh_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y = torch.cosh(in_data)

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
def test_bw_cosh_inf(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 90, 95, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -7, 7, device)

    tt_output_tensor_on_device = tt_lib.tensor.cosh_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y = torch.cosh(in_data)

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
def test_bw_cosh_neg_inf(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -95, -89, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -7, 7, device)

    tt_output_tensor_on_device = tt_lib.tensor.cosh_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y = torch.cosh(in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@skip_for_wormhole_b0()
def test_bw_cosh_nan_test1(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 86, 89, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 35, 50, device)

    tt_output_tensor_on_device = tt_lib.tensor.cosh_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y = torch.cosh(in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_cosh_nan_test2(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 86, 89, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -50, -35, device)

    tt_output_tensor_on_device = tt_lib.tensor.cosh_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y = torch.cosh(in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
