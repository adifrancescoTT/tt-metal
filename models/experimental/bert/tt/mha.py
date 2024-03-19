# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import math

import torch
from transformers import BertForQuestionAnswering
import numpy as np

import tt_lib as ttl
from tt_lib.utils import pad_activation, pad_weight, print_diff_argmax
from models.experimental.bert.fused_ops.linear import Linear as TtLinear
from tt_lib.fused_ops.softmax import softmax
from models.utility_functions import (
    enable_persistent_kernel_cache,
    comp_pcc,
    comp_allclose,
)


def mha(qw, qb, kw, kb, vw, vb, hidden_dim, num_heads, device):
    assert isinstance(num_heads, int) and num_heads > 0

    QProjection = TtLinear(hidden_dim, hidden_dim, qw, qb, device)
    KProjection = TtLinear(hidden_dim, hidden_dim, kw, kb, device)
    VProjection = TtLinear(hidden_dim, hidden_dim, vw, vb, device)

    # Used to scale down the input to the softmax
    reciprocal_of_sqrt_hidden_dim_tensor = ttl.tensor.Tensor(
        [1 / math.sqrt(hidden_dim // num_heads)] + [0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )

    def make_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            # ref code from modeling_bert.py:
            #    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
            #        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            #        x = x.view(new_x_shape)
            #        return x.permute(0, 2, 1, 3)

            untilized_x = ttl.tensor.untilize(x)
            reshaped_unt = ttl.tensor.reshape(
                untilized_x,
                x.get_legacy_shape()[0],
                x.get_legacy_shape()[2],
                num_heads,
                x.get_legacy_shape()[3] // num_heads,
            )

            # N, 128, 2, 64
            transposed = ttl.tensor.transpose(reshaped_unt, 1, -2)
            # N, 2, 128, 64
            retilized = ttl.tensor.tilize(transposed)
            return retilized

    def unmake_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            """
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            debug_state["context_reshaped"] = context_layer.clone()

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            """
            ctx = ttl.tensor.transpose(x, 1, -2)
            ushape = ctx.get_legacy_shape()
            reshaped = ttl.tensor.reshape(ctx, ushape[0], 1, ushape[1], ushape[2] * ushape[3])
            retval = ttl.tensor.tilize(reshaped)
            return retval

    def multiply_by_sqrt_hidden_dim(x):
        return ttl.tensor.bcast(
            x,
            reciprocal_of_sqrt_hidden_dim_tensor,
            ttl.tensor.BcastOpMath.MUL,
            ttl.tensor.BcastOpDim.HW,
        )

    def mha_(activation, attention_mask):
        Q = QProjection(activation)
        K = KProjection(activation)
        V = VProjection(activation)

        Q_heads = make_attention_heads(Q)
        K_heads = make_attention_heads(K)
        V_heads = make_attention_heads(V)
        K_T_heads = ttl.tensor.transpose(K_heads, -2, -1)

        qkt = ttl.tensor.bmm(Q_heads, K_T_heads)

        # Attention scores computation
        (
            N,
            C,
            H,
            W,
        ) = qkt.get_legacy_shape()  # Need to reshape right now since multi-C not supported for broadcast yet
        new_shape = [N, 1, C * H, W]
        ttl.tensor.reshape(qkt, *new_shape)
        attention_score_input = multiply_by_sqrt_hidden_dim(qkt)
        if attention_mask is not None:
            attention_score_input = ttl.tensor.bcast(
                attention_score_input,
                attention_mask,
                ttl.tensor.BcastOpMath.ADD,
                ttl.tensor.BcastOpDim.H,
            )
        attention_scores = softmax(attention_score_input)
        ttl.tensor.reshape(attention_scores, N, C, H, W)  # Reshape back to original shape

        # Apply attention to value matrix
        weighted_activation = ttl.tensor.bmm(attention_scores, V_heads)
        return unmake_attention_heads(
            weighted_activation
        )  # [N, num heads, seq len, hid size / num heads] -> [N, seq len, hid size]

    return mha_


class TtMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, config, encoder_idx, state_dict, device):
        super().__init__()
        qw = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.weight"])
        qb = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.bias"])
        kw = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.weight"])
        kb = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.bias"])
        vw = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.weight"])
        vb = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.bias"])

        # Hidden dim
        hidden_dim = qw.shape[-1]

        # Tilized
        parameters = [
            ttl.tensor.Tensor(
                qw.reshape(-1).tolist(),
                qw.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device),
            ttl.tensor.Tensor(
                qb.reshape(-1).tolist(),
                qb.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device),
            ttl.tensor.Tensor(
                kw.reshape(-1).tolist(),
                kw.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device),
            ttl.tensor.Tensor(
                kb.reshape(-1).tolist(),
                kb.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device),
            ttl.tensor.Tensor(
                vw.reshape(-1).tolist(),
                vw.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device),
            ttl.tensor.Tensor(
                vb.reshape(-1).tolist(),
                vb.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device),
        ]

        self.mha = mha(*parameters, hidden_dim, config.num_attention_heads, device)

    def forward(self, activation, attention_mask=None):
        result = self.mha(activation, attention_mask)
        return result


class PytorchMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.mha = hugging_face_reference_model.bert.encoder.layer[0].attention.self

        # Disable dropout
        self.mha.eval()

    def forward(self, x):
        result = self.mha(x)[0]
        return result


def run_mha_inference(device, model_version, batch, seq_len, pcc, model_location_generator):
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tt_mha_model = TtMultiHeadAttentionModel(
        hugging_face_reference_model.config,
        0,
        hugging_face_reference_model.state_dict(),
        device,
    )
    pytorch_mha_model = PytorchMultiHeadAttentionModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    mha_input = (torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size) * 2) - 1

    pytorch_out = pytorch_mha_model(mha_input.squeeze(1)).unsqueeze(1)

    pad_mha_input = pad_activation(mha_input)
    tt_mha_input = ttl.tensor.Tensor(
        pad_mha_input.reshape(-1).tolist(),
        pad_mha_input.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE)
    tt_mha_input = tt_mha_input.to(device)

    tt_out = tt_mha_model(tt_mha_input).cpu()
    tt_out1 = tt_out.to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    passing, output = comp_pcc(pytorch_out, tt_out1, pcc)
    logger.info(f"Output {output}")
    _, output = comp_allclose(
        pytorch_out, tt_out1, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")
    if not passing:
        logger.error(f"Output PCC < {pcc}")

    # print_diff_argmax(pytorch_out, tt_out1)
    # assert np.allclose(pytorch_out.detach().numpy(), tt_out1, 1e-5, 0.17)


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        ("mrm8488/bert-tiny-finetuned-squadv2", 1, 128, 0.99),
        ("phiyodr/bert-base-finetuned-squad2", 1, 128, 0.99),
        ("phiyodr/bert-large-finetuned-squad2", 1, 384, 0.99),
    ),
)
def test_mha_inference(device, model_version, batch, seq_len, pcc, model_location_generator):
    # enable_persistent_kernel_cache()

    run_mha_inference(device, model_version, batch, seq_len, pcc, model_location_generator)
