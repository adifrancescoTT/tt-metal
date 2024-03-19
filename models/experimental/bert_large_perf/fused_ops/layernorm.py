# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch

from tt_lib import tensor, device
from tt_lib.utils import (
    pad_activation,
    pad_weight,
    tilize,
    untilize,
    tilize_to_list,
    print_diff_argmax,
    pad_weight,
    tt2torch as t2t,
    tt2torch_rm as t2trm,
    roundup32,
    float_to_bits,
)
from models.utility_functions import profiler


def create_var_scaler(H, W, layer_norm_eps, device):
    epsilon_ = tensor.Tensor(
        [layer_norm_eps] + [0.0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        tensor.DataType.BFLOAT16,
        tensor.Layout.TILE,
        device,
    )

    scaler = 1 / W
    var_scaler = tensor.fill_rm(1, 1, roundup32(H), 32, H, 1, epsilon_, scaler, 0)
    var_scaler = tensor.tilize(var_scaler)

    return var_scaler


# This ref implementation is only here for debugging
def ref_ln(x, gamma, beta=None, epsilon=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + epsilon).sqrt()
    invstd = 1.0 / std
    y1 = (x - mean) * invstd
    y = y1.clone()
    if gamma is not None:
        y *= gamma
    if beta is not None:
        y += beta
    return y, mean, var, std, invstd, y1


# TODO(AP): refactor to support any num_dims
def Layernorm(gamma: float, beta: float, epsilon: float, H, W, device, num_dims=2):
    """
    Returns a function that performs LayerNorm with parameters.

    H, W correspond to normalized_shape in pytorch Layernorm spec

    *Note*: Note that the only ``num_dims`` supported at the moment is ``2``.
    """

    num_dims_ = num_dims
    assert num_dims == 1

    # gamma, beta, epsilon should be tt::tensors of size 32*W
    # with a single populated top row
    # H, W need to be from the "true" shape (unpadded)
    assert gamma is None or gamma.get_legacy_shape() == [1, 1, 32, W]  # single H-tile
    assert beta is None or beta.get_legacy_shape() == [1, 1, 32, W]  # single H-tile

    H_ = H
    W_ = W
    # padded_h = roundup32(H)
    # if num_dims == 1:
    #     padded_h = 32
    # padded_w = roundup32(W)
    # gamma_ = tensor.Tensor(
    #     gamma,
    #     [1, 1, padded_h, padded_w],
    #     tensor.DataType.BFLOAT16,
    #     tensor.Layout.TILE,
    #     device
    # )
    gamma_ = gamma

    beta_ = None
    if beta is not None:
        # beta_ = tensor.Tensor(
        #     beta,
        #     [1, 1, padded_h, padded_w],
        #     tensor.DataType.BFLOAT16,
        #     tensor.Layout.TILE,
        #     device
        # )
        beta_ = beta

    epsilon_ = tensor.Tensor(
        [epsilon] + [0.0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        tensor.DataType.BFLOAT16,
        tensor.Layout.TILE,
        device,
    )

    if num_dims == 2:
        var_scaler_ = tensor.Tensor(
            [1 / (H * W)] + [0.0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            tensor.DataType.BFLOAT16,
            tensor.Layout.TILE,
            device,
        )
    else:
        # For num_dims==1 var_scaler_ is implemented using dynamic mask
        assert num_dims == 1

    # tensor.DataType.BFLOAT16
    RSUM = tensor.ReduceOpMath.SUM
    RW = tensor.ReduceOpDim.W
    RH = tensor.ReduceOpDim.H
    BCW = tensor.BcastOpDim.W
    BCH = tensor.BcastOpDim.H
    BCHW = tensor.BcastOpDim.HW
    BCMUL = tensor.BcastOpMath.MUL
    BCSUB = tensor.BcastOpMath.SUB
    BCADD = tensor.BcastOpMath.ADD

    # 1D variant
    # TODO(AP): merge with 2d? refactor.
    def layernorm_1d_(x, var_scaler, overrideH=None, refx=None, refgamma=None, refbeta=None):
        N = x.get_legacy_shape()[0]
        C = x.get_legacy_shape()[1]
        H = x.get_legacy_shape()[2]
        W = x.get_legacy_shape()[3]

        H_ = 1
        if overrideH is not None:
            H_ = overrideH

        # first compute the mean (m)
        means = tensor.reduce(x, RSUM, RW, 1.0 / W)  # -> NCH1
        x_minus_mean = tensor.bcast(x, means, BCSUB, BCW)  # need to blank out the H for non-multiple of 32
        if False and refx is not None:
            ry, rmean, rvar, rstd, rinvstd, ry1 = ref_ln(refx, refgamma, refbeta)

        var = tensor.mul(x_minus_mean, x_minus_mean)  # (x-m)^2
        var_redW = tensor.reduce(var, RSUM, RW, 1.0)  # sum[(x-m)^2]

        # print(f"layernorm_1d_ var_scaler shape {var_scaler.get_legacy_shape()} H {H} H_ {H_} W {W}")

        var_div_n1 = tensor.bcast(var_redW, var_scaler, BCMUL, BCW)
        var_plus_eps = tensor.bcast(var_div_n1, epsilon_, BCADD, BCHW)

        var_sqrt = tensor.sqrt(var_plus_eps)
        inv_sqrt = tensor.recip(var_sqrt)
        if False and refx is not None:
            qq = t2t(inv_sqrt)[0, 0, 0:9, 0]

        x_div_sqrt = tensor.bcast(x_minus_mean, inv_sqrt, BCMUL, BCW)

        if False and refx is not None:
            qq1 = t2t(x_div_sqrt)[0, 0, 0:9, :]

        x_gamma = tensor.bcast(x_div_sqrt, gamma_, BCMUL, BCH)
        if beta_ is not None:
            x_beta = tensor.bcast(x_gamma, beta_, BCADD, BCH)
            return x_beta
        else:
            return x_gamma

    def layernorm_2d_(x):
        N = x.get_legacy_shape()[0]
        C = x.get_legacy_shape()[1]
        H = x.get_legacy_shape()[2]
        W = x.get_legacy_shape()[3]

        # first compute the mean (m)
        redW = tensor.reduce(x, RSUM, RW, 1.0 / W)  # -> NCH1
        mean = tensor.reduce(redW, RSUM, RH, 1.0)  # -> NC11 (HW reduce doesn't behave well with small scaler)
        x_minus_mean0 = tensor.bcast(x, mean, BCSUB, BCHW)  # need to blank out the H for non-multiple of 32

        hmasku = tensor.fill_ones_rm(N, C, H, 32, 1, 1, x)  # generate a H-mask with mask[h, w] = 1.0 where h,w < 1
        hmaskt = tensor.tilize(hmasku)  # tilize the mask
        x_minus_mean = tensor.bcast(x_minus_mean0, hmaskt, BCMUL, BCW)  # zero out (x-m) for h>=H_, h<H

        print(f"layernorm_2d_ hmasku shape {hmasku.get_legacy_shape()}")

        var = tensor.mul(x_minus_mean, x_minus_mean)  # (x-m)^2
        var_redW = tensor.reduce(var, RSUM, RW, 1.0)  # sum[(x-m)^2]
        var_redHW = tensor.reduce(var_redW, RSUM, RH, 1.0)  # sum[(x-m)^2]
        var_div_n1 = tensor.bcast(var_redHW, var_scaler_, BCMUL, BCHW)  # *= 1/(everything not batch)
        var_plus_eps = tensor.bcast(var_div_n1, epsilon_, BCADD, BCHW)

        var_sqrt = tensor.sqrt(var_plus_eps)
        inv_sqrt = tensor.recip(var_sqrt)

        x_div_sqrt = tensor.bcast(x_minus_mean, inv_sqrt, BCMUL, BCHW)
        x_gamma = tensor.mul(x_div_sqrt, gamma_, BCMUL, BCH)
        if beta_ is not None:
            x_beta = tensor.add(x_gamma, beta_, BCADD, BCH)
            return x_beta
        else:
            return x_gamma

    # unbiased_var = [(x-m)^2]/(n-1)
    # m = E[x]
    # var = E[(x-m)^2]
    # result = (x - E[x])/sqrt(var+epsilon)*gamma+beta
    def layernorm_(x, var_scaler=None, overrideH=None, refx=None, refgamma=None):
        if num_dims_ == 1:
            return layernorm_1d_(x, var_scaler, overrideH, refx, refgamma)

        assert num_dims_ == 2  # Only 1d and 2d are supported at the moment
        return layernorm_2d_(x)

    return layernorm_


def ref_layernorm(x, eps, gamma, beta, H, W):
    lnorm = torch.nn.LayerNorm((W,), eps)
    lnorm.weight = torch.nn.Parameter(torch.full((W,), gamma))
    lnorm.bias = torch.nn.Parameter(torch.full((W,), beta))
    return lnorm(x)


if __name__ == "__main__":
    device = device.CreateDevice(0)

    H = 64
    W = 96
    epsf = 1e-4
    betaf = 0.345
    gammaf = 0.123
    torch.manual_seed(123)
    x = torch.randn((1, 1, H, W))
    ref_lnorm = ref_layernorm(x, epsf, gammaf, betaf, H, W)

    gamma = pad_weight(torch.full((1, 1, 1, W), gammaf))
    beta = pad_weight(torch.full((1, 1, 1, W), betaf))

    t0 = tensor.Tensor(
        tilize_to_list(x),
        [1, 1, H, W],
        tensor.DataType.BFLOAT16,
        tensor.Layout.TILE,
        device,
    )
    ttgamma = tilize_to_list(gamma)
    ttbeta = tilize_to_list(beta)
    func = Layernorm(ttgamma, ttbeta, epsf, 1, W, device, num_dims=1)

    t1 = func(t0, overrideH=H)

    tt_got_back = t1.cpu().to_torch()
    tt_got_back = untilize(tt_got_back)

    print("Layernorm max absdiff=")
    print_diff_argmax(tt_got_back, ref_lnorm)

    device.CloseDevice(device)
