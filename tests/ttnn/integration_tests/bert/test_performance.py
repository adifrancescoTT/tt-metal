# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from loguru import logger
import torch
import transformers


import ttnn

from models.demos.bert.tt import ttnn_bert
from models.demos.bert.tt import ttnn_optimized_bert
from models.demos.bert.tt import ttnn_optimized_sharded_bert
from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask

from ttnn.model_preprocessing import preprocess_model_parameters

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report


def preprocess_inputs(
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
):
    batch_size, *_ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32)

    token_type_ids = ttnn.from_torch(token_type_ids, dtype=ttnn.uint32)

    position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32)

    if attention_mask is not None:
        attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape)
        attention_mask = attention_mask.expand((batch_size, -1, -1, -1))
        attention_mask = torch.clamp(attention_mask, min=-100000)
        attention_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    return input_ids, token_type_ids, position_ids, attention_mask


def get_expected_times(bert):
    return {
        ttnn_bert: (15, 32),
        ttnn_optimized_bert: (12, 0.08),
        ttnn_optimized_sharded_bert: (12, 0.08),
    }[bert]


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("bert", [ttnn_bert, ttnn_optimized_bert, ttnn_optimized_sharded_bert])
def test_performance(device, use_program_cache, model_name, sequence_size, bert):
    disable_persistent_kernel_cache()

    num_iterations = 10
    if bert == ttnn_bert:
        num_iterations = 2

    batch_size = 12
    if bert == ttnn_optimized_bert:
        batch_size = 8

    config = transformers.BertConfig.from_pretrained(model_name)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size) if bert != ttnn_bert else None

    if bert == ttnn_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif bert == ttnn_optimized_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    elif bert == ttnn_optimized_sharded_bert:
        config = ttnn_optimized_sharded_bert.update_model_config(config, batch_size)
        tt_model_name = f"ttnn_{model_name}_optimized_sharded"
    else:
        raise ValueError(f"Unknown bert: {bert}")

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert.custom_preprocessor,
        device=device,
    )

    durations = []
    for _ in range(num_iterations):
        ttnn_bert_inputs = preprocess_inputs(
            input_ids,
            torch_token_type_ids,
            position_ids,
            torch_attention_mask,
        )
        start = time.time()
        with ttnn.enable_fast_runtime_mode():
            ttnn_bert_inputs = [
                ttnn.to_device(tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
                if tensor is not None
                else tensor
                for tensor in ttnn_bert_inputs
            ]
            tt_output = bert.bert_for_question_answering(
                config,
                *ttnn_bert_inputs,
                parameters=parameters,
            )
            tt_output = ttnn.from_device(tt_output)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, *inference_times = durations
    average_inference_time = sum(inference_times) / len(inference_times)

    expected_compile_time, expected_inference_time = get_expected_times(bert)
    prep_perf_report(
        model_name=tt_model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=average_inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - average_inference_time}")
    logger.info(f"Inference times: {inference_times}")
    logger.info(f"Average Inference time: {average_inference_time}")
    logger.info(f"Samples per second: {1 / average_inference_time * batch_size}")
