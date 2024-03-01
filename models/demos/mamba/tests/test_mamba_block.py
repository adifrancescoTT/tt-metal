# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.mamba.tt.full_model import TtTensorLoader
from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.tt.mamba_block import TtMambaBlock
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


class PytorchMambaBlock(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.block = hf_reference_model.layers[layer_num].mixer
        self.block.eval()

    def forward(self, x):
        result = self.block(x)
        return result


@pytest.mark.parametrize(
    "model_version, batch, pcc",
    (
        (
            "state-spaces/mamba-370m",
            1,
            0.99,
        ),
    ),
)
def test_mamba_block_inference(
    model_version: MambaPretrainedModelName,
    batch,
    pcc: float,
    device: tt_lib.device,
):
    torch.manual_seed(10)

    LAYER_NUM = 0

    reference_model = MambaDecode.from_pretrained(model_version)
    reference_model.args.batch_size = batch

    d_model = reference_model.args.d_model
    input = torch.rand(batch, 1, d_model)
    tt_input = input.clone()

    reference_output = PytorchMambaBlock(reference_model, LAYER_NUM)(input)

    residual_block = reference_model.layers[LAYER_NUM]
    assert not isinstance(residual_block, torch.Tensor), "Expected torch.Module"

    loader = TtTensorLoader(reference_model.state_dict(), device)
    model = TtMambaBlock(reference_model.args, device, loader.get_tensor_loader(LAYER_NUM))
    tt_input = tt_input.unsqueeze(1)
    tt_input = torch2tt_tensor(
        tt_input,
        device,
        tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
        tt_memory_config=tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
        ),
        tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
    )
    tt_output = model(tt_input)
    tt_output = tt2torch_tensor(tt_output).squeeze(1)
    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba SSM output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
