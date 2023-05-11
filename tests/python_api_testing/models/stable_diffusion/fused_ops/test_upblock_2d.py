from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Optional

import torch.nn as nn
import torch
from diffusers import StableDiffusionPipeline
from loguru import logger

from libs import tt_lib as ttl
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from utility_functions import comp_pcc, comp_allclose_and_pcc, torch_to_tt_tensor_rm
from upblock_2d import TtUpBlock2D


def test_run_upblock_inference():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    unet_upblock = pipe.unet.up_blocks[0]

    # synthesize the input
    base_address = 'up_blocks.0'
    in_channels = 1280
    out_channels = 1280
    prev_output_channel = 1280
    temb_channels = None
    eps = 1e-05
    resnet_groups = 32
    input_shape  = [2, 1280, 8, 8]
    hidden_state = torch.randn(input_shape, dtype=torch.float32)
    res_hidden_states_tuple = (hidden_state , hidden_state , hidden_state )
    temb_shape = [1, 1, 2, 1280]
    temb = torch.randn(temb_shape)

    # execute pytorch
    torch_output = unet_upblock(hidden_state, res_hidden_states_tuple, None, None)

    # setup tt models
    tt_upblock = TtUpBlock2D(
                            in_channels=in_channels,
                            prev_output_channel = prev_output_channel,
                            out_channels=out_channels,
                            temb_channels=temb_channels,
                            dropout= 0.0,
                            num_layers= 3,
                            resnet_eps= 1e-6,
                            resnet_time_scale_shift = "default",
                            resnet_act_fn= "silu",
                            resnet_groups=resnet_groups,
                            resnet_pre_norm= True,
                            output_scale_factor=1.0,
                            add_upsample=True,
                            state_dict=state_dict,
                            base_address = base_address
                            )

    tt_out = tt_upblock(hidden_state, res_hidden_states_tuple, None, None)
    tt_output = tt_to_torch_tensor(tt_out, host)

    print(comp_allclose_and_pcc(unet_out, tt_out))

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    ttl.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
