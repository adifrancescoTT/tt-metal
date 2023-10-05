# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
import tt_lib
from loguru import logger

from transformers import AutoModelForCausalLM
from models.experimental.llama2.reference.generation import Llama

from models.experimental.llama2.tt.llama2_configuration import ModelArgs
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_pcc,
    comp_allclose,
)
from models.experimental.llama2.tt.llama2_feedforward import TtFeedForward


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_llama2_feedforward(device, pcc, model_location_generator, reset_seeds):
    llama2_path = str(model_location_generator("llama-2-7b", model_subdir="llama-2"))
    facebook_research_reference_model = Llama.build(llama2_path, llama2_path, 50, 1)

    torch_model = facebook_research_reference_model.model.layers[0].feed_forward
    state_dict = facebook_research_reference_model.model.state_dict()

    base_address = "layers.0"

    dim = 4096
    hidden_dim = 16384
    multiple_of = 256
    input = torch.rand((1, 9, 4096))

    tt_input = torch_to_tt_tensor_rm(input, device)

    Tt_model = TtFeedForward(
        ModelArgs,
        dim=dim,
        hidden_dim=hidden_dim,
        multiple_of=multiple_of,
        state_dict=state_dict,
        base_address=base_address,
        device=device,
    )

    torch_output = torch_model(input)

    tt_output = Tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output).squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("Llama2feedforward Passed!")

    assert does_pass, f"Llama2feedforward does not meet PCC requirement {pcc}."
