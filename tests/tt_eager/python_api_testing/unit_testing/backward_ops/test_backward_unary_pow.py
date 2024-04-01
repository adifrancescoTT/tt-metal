# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "exponent",
    [
        -0.01,
        -1.0,
    ],
)
def test_negative_exponent(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)

    with pytest.raises(RuntimeError) as _e:
        tt_output_tensor_on_device = tt_lib.tensor.unary_pow_bw(grad_tensor, input_tensor, exponent=exponent)
    assert "exponent >= 0.0" in str(_e)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "exponent",
    [
        0,
    ],
)
def test_fw_exponent(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)

    golden_tensor = [
        torch.pow(grad_data, exponent),
    ]
    tt_output_tensor_on_device = tt_lib.tensor.pow(grad_tensor, exponent)
    status = compare_pcc([tt_output_tensor_on_device], golden_tensor)
    assert status

    # assert "exponent >= 0.0" in str(_e)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent_and_pcc",
    [
        (0.0, 0.99),
        (1.0, 0.99),
        (2.0, 0.99),
        (5.0, 0.99),
        (0.5, 0.92),
        (1.5, 0.84),
        (2.5, 0.57),
    ],
)
def test_bw_unary_pow(input_shapes, exponent_and_pcc, device):
    exponent, pcc = exponent_and_pcc
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device)

    tt_output_tensor_on_device = tt_lib.tensor.unary_pow_bw(grad_tensor, input_tensor, exponent=exponent)

    in_data.retain_grad()

    pyt_y = torch.pow(in_data, exponent)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, pcc=pcc)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_unary_pow_test_inf(input_shapes, device):
    exponent = 2
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.74e38, 1.8e38, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 1, 9, device)

    tt_output_tensor_on_device = tt_lib.tensor.unary_pow_bw(grad_tensor, input_tensor, exponent=exponent)

    in_data.retain_grad()

    pyt_y = torch.pow(in_data, exponent)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_unary_pow_test_neg_inf(input_shapes, device):
    exponent = 2
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.74e38, 1.8e38, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, -1, device)

    tt_output_tensor_on_device = tt_lib.tensor.unary_pow_bw(grad_tensor, input_tensor, exponent=exponent)

    in_data.retain_grad()

    pyt_y = torch.pow(in_data, exponent)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
