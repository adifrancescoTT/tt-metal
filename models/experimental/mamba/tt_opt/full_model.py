# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from pathlib import Path
from typing import Callable

from models.experimental.mamba.tt_opt.residual_block import TtResidualBlock


class TtTensorLoader:
    def __init__(self, state_dict, device, tt_cache_path: str = ""):
        self.state_dict = state_dict
        self.tt_cache_path = tt_cache_path
        self.device = device

    def get_tensor_loader(self, layer_num):
        def load_tt_tensor(
            name: str,
            tm_fn: Callable = lambda x: x,
            postfix: str = "",
            device: ttnn.Device = self.device,
            tt_layout=ttnn.TILE_LAYOUT,
            tt_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            tt_dtype=ttnn.bfloat16,
            tensor=None,
        ):
            name = f"layers.{layer_num}.{name}"
            tensor_cache_filepath = Path(self.tt_cache_path) / (name + postfix)
            if tensor is None:
                tensor = self.state_dict[name]
            tensor = tm_fn(tensor)

            # Make all loaded tensors rank 4 because there are performance issues with certain
            # ops when using with rank 1/2 tensors in ttnn
            while tensor.size() < 4:
                tensor = tensor.unsqueeze(0)

            tt_tensor = ttnn.as_tensor(
                tensor,
                device=device,
                layout=tt_layout,
                memory_config=tt_memory_config,
                dtype=tt_dtype,
                cache_file_name=str(tensor_cache_filepath),
            )
            return tt_tensor

        return load_tt_tensor


class MambaTT(torch.nn.Module):
    def __init__(self, reference_model, device: ttnn.Device, configs, tt_cache_path: str = "", num_layers=None):
        super().__init__()
        self.args = reference_model.args
        self.device = device
        self.tt_cache_path = tt_cache_path

        if num_layers is None:
            self.num_layers = len(reference_model.layers)
        else:
            self.num_layers = num_layers
        print(f"Initalizing MambaTT with {self.num_layers} layers")

        self.embedding = reference_model.embedding

        loader = TtTensorLoader(reference_model.state_dict(), self.device, tt_cache_path=tt_cache_path)

        self.layers = [
            TtResidualBlock(self.args, device, configs, loader.get_tensor_loader(i)) for i in range(self.num_layers)
        ]

        self.norm_f = reference_model.norm_f

        self.lm_head = reference_model.lm_head

    def forward(self, x):
        x = self.embedding(x)
        x = x.squeeze(1)
        x = ttnn.from_torch(
            x,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        for layer in self.layers:
            x_old = x
            x = layer(x)
            ttnn.deallocate(x_old)

        x = ttnn.to_torch(x).to(torch.float32)
        x = x.unsqueeze(1)
        x = self.norm_f(x)
        x = self.lm_head(x)

        return x
