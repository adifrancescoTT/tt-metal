# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import json
import torch
import pytest
import numpy as np
from PIL import Image
from loguru import logger
from tqdm.auto import tqdm
from datasets import load_dataset
from scipy import integrate

from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@skip_for_grayskull()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "expected_perf",
    ((0.16),),
)
def test_stable_diffusion_perf_device(expected_perf):
    subdir = "ttnn_stable_diffusion"
    margin = 0.03
    batch = 1
    iterations = 30
    command = f"pytest models/experimental/functional_stable_diffusion/demo/demo.py::test_demo_diffusiondb"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, iterations, cols, batch)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"stable_diffusion_{batch}batch",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
