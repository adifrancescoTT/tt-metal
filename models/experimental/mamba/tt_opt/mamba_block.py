# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import tt_lib as ttl
from typing import Callable

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.experimental.mamba.reference.args import ModelArgs
from models.experimental.mamba.tt_opt.mamba_one_step_ssm import TtMambaSSM


class TtMambaBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args
        self.num_users = args.batch_size
        assert self.num_users == 32, "Batch size must be 32 for now"
        self.configs = configs

        in_proj_weight_name = "mixer.in_proj.weight"

        # ssm wt
        self.ssm_in_proj_weights = load_fn(
            in_proj_weight_name, lambda x: x[: self.args.d_inner, :].transpose(-1, -2), postfix="ssm"
        )

        # mlp wt
        self.mlp_proj_weights = load_fn(
            in_proj_weight_name, lambda x: x[self.args.d_inner :, :].transpose(-1, -2), postfix="mlp"
        )

        # down proj wt
        out_proj_weight_name = "mixer.out_proj.weight"
        self.out_proj_weights = load_fn(out_proj_weight_name, lambda x: x.transpose(-1, -2))

        # conv states
        conv1d_weight_name = "mixer.conv1d.weight"
        self.conv1d_weights = []
        # conv1d_weights (2E, 1, 4)->(2E, 1)->(2E, 1, 1)->(1, 1, 2E)
        for i in range(4):
            self.conv1d_weights.append(
                load_fn(
                    conv1d_weight_name,
                    lambda x: x[:, :, i].transpose(-1, -2).repeat(self.num_users, 1).unsqueeze(0).unsqueeze(0),
                    postfix=f"{i}_{args.batch_size}",
                )
            )

        conv1d_bias_name = "mixer.conv1d.bias"
        self.conv1d_bias = load_fn(
            conv1d_bias_name,
            lambda x: x.repeat(self.args.batch_size, 1),
            postfix=f"{args.batch_size}",
        )

        self.conv_states = []
        for i in range(4):
            self.conv_states.append(
                load_fn(
                    f"conv_state{i}",
                    torch_tensor=torch.zeros(1, 1, self.num_users, self.args.d_inner),
                    postfix=f"{args.batch_size}",
                )
            )

        self.tt_ssm = TtMambaSSM(self.args, self.device, configs, load_fn)

        self.compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.core_grid_row = 4
        self.core_grid_col = 8

    def forward(self, x):
        residual_connection = x  # b, e=d_model

        x = ttnn.linear(
            x,
            self.ssm_in_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=4, x=8),
        )

        # shift the states leftward
        ttnn.deallocate(self.conv_states[0])
        for i in range(3):
            self.conv_states[i] = self.conv_states[i + 1]

        # update the last state and move it back to DRAM with all the other states
        self.conv_states[3] = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # do the convolution
        conv1d_wt = ttnn.to_memory_config(self.conv1d_weights[0], memory_config=self.configs["sharded_d"])
        conv_state = ttnn.to_memory_config(self.conv_states[0], memory_config=self.configs["sharded_d"])
        x = ttnn.mul(conv_state, conv1d_wt, memory_config=self.configs["sharded_d"])
        ttnn.deallocate(conv1d_wt)
        ttnn.deallocate(conv_state)

        for i in range(1, 4):
            conv1d_wt = ttnn.to_memory_config(self.conv1d_weights[i], memory_config=self.configs["sharded_d"])
            conv_state = ttnn.to_memory_config(self.conv_states[i], memory_config=self.configs["sharded_d"])
            prod = ttnn.mul(conv_state, conv1d_wt, memory_config=self.configs["sharded_d"])
            ttnn.deallocate(conv1d_wt)
            ttnn.deallocate(conv_state)

            x = ttnn.add(x, prod, memory_config=self.configs["sharded_d"])
            ttnn.deallocate(prod)

        conv1d_bias = ttnn.to_memory_config(self.conv1d_bias, memory_config=self.configs["sharded_d"])
        x = ttnn.add(x, conv1d_bias, memory_config=self.configs["sharded_d"])
        ttnn.deallocate(conv1d_bias)

        x = ttnn.to_memory_config(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.silu(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = self.tt_ssm(x)

        residual = ttnn.linear(
            residual_connection,
            self.mlp_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=4, x=8),
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
        )
        ttnn.deallocate(residual_connection)

        residual_with_silu = ttnn.silu(residual, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(residual)

        residual_with_silu = ttnn.to_memory_config(residual_with_silu, memory_config=self.configs["sharded_d"])
        out = ttnn.mul(x, residual_with_silu, memory_config=self.configs["sharded_d"])
        ttnn.deallocate(residual_with_silu)
        ttnn.deallocate(x)

        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.linear(
            out,
            self.out_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=4, x=8),
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
        )

        return out
