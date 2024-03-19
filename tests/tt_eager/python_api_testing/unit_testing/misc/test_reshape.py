# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax


def test_tile_major_reshape(device):
    torch.manual_seed(0)

    N = 3
    C = 5
    H = 64
    W = 96
    x = torch.randn((N, C, H, W), dtype=torch.float32).bfloat16().float()

    xtt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttl.tensor.reshape(xtt, 5, 3, 96, 64)
    assert list(xtt.get_legacy_shape()) == [5, 3, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([5, 3, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttl.tensor.reshape(xtt, 3, 5, 64, 96)
    assert list(xtt.get_legacy_shape()) == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttl.tensor.reshape(xtt, -1, 5, 96, 64)
    assert list(xtt.get_legacy_shape()) == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttl.tensor.reshape(xtt, 3, -1, 64, 96)
    assert list(xtt.get_legacy_shape()) == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttl.tensor.reshape(xtt, 3, 5, -1, 64)
    assert list(xtt.get_legacy_shape()) == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttl.tensor.reshape(xtt, 3, 5, 64, -1)
    assert list(xtt.get_legacy_shape()) == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttl.tensor.reshape(xtt, 3, 5, 32, -1)
    assert list(xtt.get_legacy_shape()) == [3, 5, 32, 96 * 2]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 32, 96 * 2])
    eq = torch.equal(x, tt_got_back)
    assert eq

    print("reshape() max absdiff=")
    print_diff_argmax(tt_got_back, x)


def test_row_major_reshape(device):
    # Power of 2 reshape
    N = 1
    C = 1
    H = 128
    W = 128
    x = torch.rand(N * C * H * W).reshape(N, C, H, W).bfloat16().float()
    xtt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(device)

    reshaped = ttl.tensor.reshape(xtt, 1, 128, 2, 64)
    reshaped = reshaped.cpu().to_torch()
    torch_reshaped = torch.Tensor(x).reshape(1, 128, 2, 64)
    eq = torch.equal(torch_reshaped, reshaped)
    assert eq
