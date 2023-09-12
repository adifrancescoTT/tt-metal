# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch import nn
from typing import Optional, Tuple

import tt_lib

from tests.models.falcon.falcon_model import TtFalconModelShared
from models.helper_funcs import Linear as TTLinear
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, dump_tensor


class TtFalconCausalLM(TtFalconModelShared):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        assert base_url == "", "base_url should be empty at the root of the model!"

        super().__init__(
            device=device,
            state_dict=state_dict,
            base_url=f"transformer",
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )
        self.model_config = model_config

        lm_head_str = f"lm_head.weight"
        if tt_cache_path is not None:
            self.lm_head_weights = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{lm_head_str}_{self.model_config['LM_HEAD_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"])
        else:
            self.lm_head_weights = torch2tt_tensor(
                torch.transpose(self.state_dict[f"lm_head.weight"], -2, -1),
                self.device,
                tt_memory_config=self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["LM_HEAD_MM_WEIGHTS_DTYPE"],
            )

    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        llm_mode: str,
        attention_mask: tt_lib.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> tt_lib.tensor.Tensor:
        hidden_states, presents = super().forward(
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )
        dump_tensor("lm_head_input", "tt", tt2torch_tensor(hidden_states))
        # dump_tensor("lm_head_weights", "tt", tt2torch_tensor(self.lm_head_weights)[0][0])
        lm_logits = tt_lib.tensor.falcon_lm_head_matmul(
            hidden_states,
            self.lm_head_weights,
            output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
        )
        # lm_logits = tt_lib.tensor.Tensor(tt2torch_tensor(hidden_states).to(torch.float32) @ tt2torch_tensor(self.lm_head_weights).to(torch.float32), tt_lib.tensor.DataType.BFLOAT16)
        dump_tensor("lm_logits", "tt", tt2torch_tensor(lm_logits))

        return lm_logits, presents
