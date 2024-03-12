# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib

from models.utility_functions import torch2tt_tensor


class TtFalconMLP(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.hidden_size = hidden_size
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        self.dense_h_to_4h_weights = self.get_weights_cached(
            tt_cache_path,
            weight_cache_str=f"{layer_name}.mlp.dense_h_to_4h.weight",
            weight_config_str="DENSE_H_TO_4H_MM_WEIGHTS",
        )
        self.dense_4h_to_h_weights = self.get_weights_cached(
            tt_cache_path,
            weight_cache_str=f"{layer_name}.mlp.dense_4h_to_h.weight",
            weight_config_str="DENSE_4H_TO_H_MM_WEIGHTS",
        )
        if len(devices) == 1:
            self.dense_4h_to_h_weights = self.dense_4h_to_h_weights[0]
            self.dense_h_to_4h_weights = self.dense_h_to_4h_weights[0]

    def get_weights_cached(self, tt_cache_path, weight_cache_str, weight_config_str):
        """Load cached weights and duplicate per device. Store if not cached."""
        if (tt_cache_path / f"{weight_cache_str}_{self.model_config[f'{weight_config_str}_DTYPE'].name}.bin").exists():
            # Load cached weights
            weights_host = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{weight_cache_str}_{self.model_config[f'{weight_config_str}_DTYPE'].name}.bin")
            )
            # Duplicate weights on all devices
            weights = [
                weights_host.to(device, self.model_config[f"{weight_config_str}_MEMCFG"]) for device in self.devices
            ]
        else:
            weights_host = torch.transpose(
                self.state_dict[weight_cache_str],
                -2,
                -1,
            )
            # Duplicate weights on all devices
            weights = [
                torch2tt_tensor(
                    weights_host,
                    device,
                    tt_memory_config=self.model_config[f"{weight_config_str}_MEMCFG"],
                    tt_dtype=self.model_config[f"{weight_config_str}_DTYPE"],
                )
                for device in self.devices
            ]
            # Store weights (from first device)
            tt_lib.tensor.dump_tensor(
                str(tt_cache_path / f"{weight_cache_str}_{self.model_config[f'{weight_config_str}_DTYPE'].name}.bin"),
                weights[0].cpu(),
            )
        return weights

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = tt_lib.tensor.falcon_dense_h_to_4h_matmul(
            x,
            self.dense_h_to_4h_weights,
            fused_activation=[tt_lib.tensor.FusibleActivation.GELU, True],
            output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
        )
        x.deallocate()

        hidden_states = tt_lib.tensor.falcon_dense_4h_to_h_matmul(
            hidden_states,
            self.dense_4h_to_h_weights,
            output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
            packer_l1_acc=True,
        )

        # return TT Tensor
        return hidden_states
