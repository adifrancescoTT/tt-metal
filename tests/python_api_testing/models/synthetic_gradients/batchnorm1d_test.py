import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"

sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torch import nn
from torchvision import transforms, datasets

import libs
from libs import tt_lib as ttl

from models.utility_functions import tilize_to_list, untilize
from sweep_tests.comparison_funcs import comp_pcc, comp_allclose

epsilon = 1e-5

def batchnorm1d_inference(gamma, beta, running_mean, running_var, epsilon):

    BCHW = ttl.tensor.BcastOpDim.HW
    BCADD = ttl.tensor.BcastOpMath.ADD

    def batchnorm1d_inference_(X):
        var_plus_eps = ttl.tensor.bcast(running_var, epsilon, BCADD, BCHW)
        sqrt_var = ttl.tensor.sqrt(var_plus_eps)
        sqrt_inv = ttl.tensor.recip(sqrt_var)
        x_minus_mean = ttl.tensor.sub(X, running_mean)
        x_div_sqrt = ttl.tensor.mul(x_minus_mean, sqrt_inv)
        x_gamma = ttl.tensor.mul(x_div_sqrt, gamma)
        Y = ttl.tensor.add(x_gamma, beta)
        return Y

    return batchnorm1d_inference_


class PytorchBatchNorm1D(nn.Module):
    def __init__(self, input_dim):
        super(PytorchBatchNorm1D, self).__init__()

        self.batchnorm1d_1 = nn.BatchNorm1d(input_dim)

    def forward(self, x):

        bn1_out =  self.batchnorm1d_1(x)

        return bn1_out


def run_btchnorm_inference(device, bn_size):
    host = ttl.device.GetHost()

    inputs = torch.FloatTensor(1, bn_size).uniform_(-1., 1.).requires_grad_(True)

    # torch
    bn_torch = PytorchBatchNorm1D(bn_size)
    bn_torch.eval()
    weight_bn = torch.nn.Parameter(torch.FloatTensor(bn_size).uniform_(-1., 1.).requires_grad_(True))
    bias_bn =  torch.nn.Parameter(torch.FloatTensor(bn_size).uniform_(-1., 1.).requires_grad_(True))
    running_mean = torch.FloatTensor(bn_size).uniform_(-1., 1.).requires_grad_(False)
    running_var = torch.FloatTensor(bn_size).uniform_(0., 1.).requires_grad_(False)  #must be positive

    bn_torch.batchnorm1d_1.weight = weight_bn
    bn_torch.batchnorm1d_1.bias = bias_bn
    bn_torch.batchnorm1d_1.running_mean = running_mean
    bn_torch.batchnorm1d_1.running_var = running_var
    bn_torch.batchnorm1d_1.eps = epsilon

    # tt
    weight_bn_src = weight_bn.view(1, 1, 1, bn_size)
    weight_bn_tt = torch.zeros(1, 1, 32, bn_size)
    weight_bn_tt[:, :, :1, :] = weight_bn_src
    tilized_weight_bn_tt= tilize_to_list(weight_bn_tt)
    gamma = ttl.tensor.Tensor(tilized_weight_bn_tt, [1, 1, 32, bn_size], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    bias_bn_src = bias_bn.view(1, 1, 1, bn_size)
    bias_bn_tt = torch.zeros(1, 1, 32, bn_size)
    bias_bn_tt[:, :, :1, :] = bias_bn_src
    tilized_bias_bn_tt= tilize_to_list(bias_bn_tt)
    beta = ttl.tensor.Tensor(tilized_bias_bn_tt, [1, 1, 32, bn_size], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    running_mean_bn_src = running_mean.view(1, 1, 1, bn_size)
    running_mean_bn_tt = torch.zeros(1, 1, 32, bn_size)
    running_mean_bn_tt[:, :, :1, :] = running_mean_bn_src
    tilized_running_mean_tt= tilize_to_list(running_mean_bn_tt)
    running_mean_tt = ttl.tensor.Tensor(tilized_running_mean_tt, [1, 1, 32, bn_size], ttl.tensor.DataType.BFLOAT16,ttl.tensor.Layout.TILE, device)

    running_var_bn_src = running_var.view(1, 1, 1, bn_size)
    running_var_bn_tt = torch.zeros(1, 1, 32, bn_size)
    running_var_bn_tt[:, :, :1, :] = running_var_bn_src
    tilized_running_var_tt= tilize_to_list(running_var_bn_tt)
    running_var_tt = ttl.tensor.Tensor(tilized_running_var_tt, [1, 1, 32, bn_size], ttl.tensor.DataType.BFLOAT16,ttl.tensor.Layout.TILE, device)

    epsilon_tt = ttl.tensor.Tensor([epsilon] + [0 for _ in range(32 * 32 - 1)], [1, 1, 32, 32], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    inputs_bn_src = inputs.view(1, 1, 1, bn_size)
    inputs_bn_tt = torch.zeros(1, 1, 32, bn_size)
    inputs_bn_tt[:, :, :1, :] = inputs_bn_src
    tilized_inputs_tt = tilize_to_list(inputs_bn_tt)
    X_tt = ttl.tensor.Tensor(tilized_inputs_tt, [1, 1,  32, bn_size], ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)


    # run through models
    output_bn_torch = bn_torch(inputs)
    bn_tt =  batchnorm1d_inference(gamma, beta, running_mean_tt, running_var_tt, epsilon_tt)
    output_bn_tt = bn_tt(X_tt)

    output_bn_tt_untilized = untilize(torch.Tensor(output_bn_tt.to(host).data()).reshape(output_bn_tt.shape()))
    output_bn_tt_untilized = output_bn_tt_untilized[0, 0, 0, :]

    print('pytorch_out:', output_bn_torch[0][0:10])
    print('tt_out:', output_bn_tt_untilized[0:10])

    pcc_result = comp_pcc(output_bn_torch[0], output_bn_tt_untilized)
    print('\n\n', 'pcc:', pcc_result, '\n\n')

    allclose_result = comp_allclose(output_bn_torch[0], output_bn_tt_untilized)

    print('\n\n','atol/rtol:', allclose_result, '\n\n')

def test_batchnorm_inference():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    run_btchnorm_inference(device, 1024)
    ttl.device.CloseDevice(device)
