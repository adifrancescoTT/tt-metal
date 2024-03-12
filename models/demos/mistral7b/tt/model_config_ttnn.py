# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from pathlib import Path


class TtModelArgs:
    """Model args for Mistral 7B as provided by the params.json config file"""

    dim = 4096
    n_layers = 32
    head_dim = 128
    hidden_dim = 14336
    n_heads = 32
    n_kv_heads = 8
    norm_eps = 1e-05
    sliding_window = 4096
    vocab_size = 32000

    # Parameters for our use
    max_batch_size = 32
    max_seq_len = 4096

    def __init__(self, model_base_path="/proj_sw/user_dev/hf_data/mistral", instruct=False):
        self.model_base_path = Path(model_base_path)
        # Some consumers like SentencePiece only accept str not Path for files
        if instruct:  # Load instruct weights and tokenizer from HF mistralai/Mistral-7B-Instruct-v0.2
            self.consolidated_weights_path = lambda i: str(
                self.model_base_path / f"mistral-7B-v0.1/pytorch_model-0000{i}-of-00003.bin"
            )
            self.tokenizer_path = str(self.model_base_path / "tokenizer_instruct.model")
        else:  # Load generative weights and tokenizer
            self.consolidated_weights_path = str(self.model_base_path / "mistral-7B-v0.1/consolidated.00.pth")
            self.tokenizer_path = str(self.model_base_path / "mistral-7B-v0.1/tokenizer.model")

    def weight_cache_path(self, dtype, instruct=False):
        # Keep the weight cache separate for generative and instruct weights
        if instruct:
            return (
                self.model_base_path
                / {ttnn.bfloat16: "tensor_cache_instruct_bf16", ttnn.bfloat8_b: "tensor_cache_instruct_bfp8"}[dtype]
            )
        else:
            return (
                self.model_base_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]
            )

    # Key mapping for the instruct weights to match the generative weights
    key_mapping = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
        "model.layers.0.self_attn.q_proj.weight": "layers.0.attention.wq.weight",
        "model.layers.0.self_attn.k_proj.weight": "layers.0.attention.wk.weight",
        "model.layers.0.self_attn.v_proj.weight": "layers.0.attention.wv.weight",
        "model.layers.0.self_attn.o_proj.weight": "layers.0.attention.wo.weight",
        "model.layers.0.mlp.gate_proj.weight": "layers.0.feed_forward.w1.weight",
        "model.layers.0.mlp.up_proj.weight": "layers.0.feed_forward.w3.weight",
        "model.layers.0.mlp.down_proj.weight": "layers.0.feed_forward.w2.weight",
        "model.layers.0.input_layernorm.weight": "layers.0.attention_norm.weight",
        "model.layers.0.post_attention_layernorm.weight": "layers.0.ffn_norm.weight",
        "model.layers.1.self_attn.q_proj.weight": "layers.1.attention.wq.weight",
        "model.layers.1.self_attn.k_proj.weight": "layers.1.attention.wk.weight",
        "model.layers.1.self_attn.v_proj.weight": "layers.1.attention.wv.weight",
        "model.layers.1.self_attn.o_proj.weight": "layers.1.attention.wo.weight",
        "model.layers.1.mlp.gate_proj.weight": "layers.1.feed_forward.w1.weight",
        "model.layers.1.mlp.up_proj.weight": "layers.1.feed_forward.w3.weight",
        "model.layers.1.mlp.down_proj.weight": "layers.1.feed_forward.w2.weight",
        "model.layers.1.input_layernorm.weight": "layers.1.attention_norm.weight",
        "model.layers.1.post_attention_layernorm.weight": "layers.1.ffn_norm.weight",
        "model.layers.2.self_attn.q_proj.weight": "layers.2.attention.wq.weight",
        "model.layers.2.self_attn.k_proj.weight": "layers.2.attention.wk.weight",
        "model.layers.2.self_attn.v_proj.weight": "layers.2.attention.wv.weight",
        "model.layers.2.self_attn.o_proj.weight": "layers.2.attention.wo.weight",
        "model.layers.2.mlp.gate_proj.weight": "layers.2.feed_forward.w1.weight",
        "model.layers.2.mlp.up_proj.weight": "layers.2.feed_forward.w3.weight",
        "model.layers.2.mlp.down_proj.weight": "layers.2.feed_forward.w2.weight",
        "model.layers.2.input_layernorm.weight": "layers.2.attention_norm.weight",
        "model.layers.2.post_attention_layernorm.weight": "layers.2.ffn_norm.weight",
        "model.layers.3.self_attn.q_proj.weight": "layers.3.attention.wq.weight",
        "model.layers.3.self_attn.k_proj.weight": "layers.3.attention.wk.weight",
        "model.layers.3.self_attn.v_proj.weight": "layers.3.attention.wv.weight",
        "model.layers.3.self_attn.o_proj.weight": "layers.3.attention.wo.weight",
        "model.layers.3.mlp.gate_proj.weight": "layers.3.feed_forward.w1.weight",
        "model.layers.3.mlp.up_proj.weight": "layers.3.feed_forward.w3.weight",
        "model.layers.3.mlp.down_proj.weight": "layers.3.feed_forward.w2.weight",
        "model.layers.3.input_layernorm.weight": "layers.3.attention_norm.weight",
        "model.layers.3.post_attention_layernorm.weight": "layers.3.ffn_norm.weight",
        "model.layers.4.self_attn.q_proj.weight": "layers.4.attention.wq.weight",
        "model.layers.4.self_attn.k_proj.weight": "layers.4.attention.wk.weight",
        "model.layers.4.self_attn.v_proj.weight": "layers.4.attention.wv.weight",
        "model.layers.4.self_attn.o_proj.weight": "layers.4.attention.wo.weight",
        "model.layers.4.mlp.gate_proj.weight": "layers.4.feed_forward.w1.weight",
        "model.layers.4.mlp.up_proj.weight": "layers.4.feed_forward.w3.weight",
        "model.layers.4.mlp.down_proj.weight": "layers.4.feed_forward.w2.weight",
        "model.layers.4.input_layernorm.weight": "layers.4.attention_norm.weight",
        "model.layers.4.post_attention_layernorm.weight": "layers.4.ffn_norm.weight",
        "model.layers.5.self_attn.q_proj.weight": "layers.5.attention.wq.weight",
        "model.layers.5.self_attn.k_proj.weight": "layers.5.attention.wk.weight",
        "model.layers.5.self_attn.v_proj.weight": "layers.5.attention.wv.weight",
        "model.layers.5.self_attn.o_proj.weight": "layers.5.attention.wo.weight",
        "model.layers.5.mlp.gate_proj.weight": "layers.5.feed_forward.w1.weight",
        "model.layers.5.mlp.up_proj.weight": "layers.5.feed_forward.w3.weight",
        "model.layers.5.mlp.down_proj.weight": "layers.5.feed_forward.w2.weight",
        "model.layers.5.input_layernorm.weight": "layers.5.attention_norm.weight",
        "model.layers.5.post_attention_layernorm.weight": "layers.5.ffn_norm.weight",
        "model.layers.6.self_attn.q_proj.weight": "layers.6.attention.wq.weight",
        "model.layers.6.self_attn.k_proj.weight": "layers.6.attention.wk.weight",
        "model.layers.6.self_attn.v_proj.weight": "layers.6.attention.wv.weight",
        "model.layers.6.self_attn.o_proj.weight": "layers.6.attention.wo.weight",
        "model.layers.6.mlp.gate_proj.weight": "layers.6.feed_forward.w1.weight",
        "model.layers.6.mlp.up_proj.weight": "layers.6.feed_forward.w3.weight",
        "model.layers.6.mlp.down_proj.weight": "layers.6.feed_forward.w2.weight",
        "model.layers.6.input_layernorm.weight": "layers.6.attention_norm.weight",
        "model.layers.6.post_attention_layernorm.weight": "layers.6.ffn_norm.weight",
        "model.layers.7.self_attn.q_proj.weight": "layers.7.attention.wq.weight",
        "model.layers.7.self_attn.k_proj.weight": "layers.7.attention.wk.weight",
        "model.layers.7.self_attn.v_proj.weight": "layers.7.attention.wv.weight",
        "model.layers.7.self_attn.o_proj.weight": "layers.7.attention.wo.weight",
        "model.layers.7.mlp.gate_proj.weight": "layers.7.feed_forward.w1.weight",
        "model.layers.7.mlp.up_proj.weight": "layers.7.feed_forward.w3.weight",
        "model.layers.7.mlp.down_proj.weight": "layers.7.feed_forward.w2.weight",
        "model.layers.7.input_layernorm.weight": "layers.7.attention_norm.weight",
        "model.layers.7.post_attention_layernorm.weight": "layers.7.ffn_norm.weight",
        "model.layers.8.self_attn.q_proj.weight": "layers.8.attention.wq.weight",
        "model.layers.8.self_attn.k_proj.weight": "layers.8.attention.wk.weight",
        "model.layers.8.self_attn.v_proj.weight": "layers.8.attention.wv.weight",
        "model.layers.8.self_attn.o_proj.weight": "layers.8.attention.wo.weight",
        "model.layers.8.mlp.gate_proj.weight": "layers.8.feed_forward.w1.weight",
        "model.layers.8.mlp.up_proj.weight": "layers.8.feed_forward.w3.weight",
        "model.layers.8.mlp.down_proj.weight": "layers.8.feed_forward.w2.weight",
        "model.layers.8.input_layernorm.weight": "layers.8.attention_norm.weight",
        "model.layers.8.post_attention_layernorm.weight": "layers.8.ffn_norm.weight",
        "model.layers.9.self_attn.q_proj.weight": "layers.9.attention.wq.weight",
        "model.layers.9.self_attn.k_proj.weight": "layers.9.attention.wk.weight",
        "model.layers.9.self_attn.v_proj.weight": "layers.9.attention.wv.weight",
        "model.layers.9.self_attn.o_proj.weight": "layers.9.attention.wo.weight",
        "model.layers.9.mlp.gate_proj.weight": "layers.9.feed_forward.w1.weight",
        "model.layers.9.mlp.up_proj.weight": "layers.9.feed_forward.w3.weight",
        "model.layers.9.mlp.down_proj.weight": "layers.9.feed_forward.w2.weight",
        "model.layers.9.input_layernorm.weight": "layers.9.attention_norm.weight",
        "model.layers.9.post_attention_layernorm.weight": "layers.9.ffn_norm.weight",
        "model.layers.10.self_attn.q_proj.weight": "layers.10.attention.wq.weight",
        "model.layers.10.self_attn.k_proj.weight": "layers.10.attention.wk.weight",
        "model.layers.10.self_attn.v_proj.weight": "layers.10.attention.wv.weight",
        "model.layers.10.self_attn.o_proj.weight": "layers.10.attention.wo.weight",
        "model.layers.10.mlp.gate_proj.weight": "layers.10.feed_forward.w1.weight",
        "model.layers.10.mlp.up_proj.weight": "layers.10.feed_forward.w3.weight",
        "model.layers.10.mlp.down_proj.weight": "layers.10.feed_forward.w2.weight",
        "model.layers.10.input_layernorm.weight": "layers.10.attention_norm.weight",
        "model.layers.10.post_attention_layernorm.weight": "layers.10.ffn_norm.weight",
        "model.layers.11.self_attn.q_proj.weight": "layers.11.attention.wq.weight",
        "model.layers.11.self_attn.k_proj.weight": "layers.11.attention.wk.weight",
        "model.layers.11.self_attn.v_proj.weight": "layers.11.attention.wv.weight",
        "model.layers.11.self_attn.o_proj.weight": "layers.11.attention.wo.weight",
        "model.layers.11.mlp.gate_proj.weight": "layers.11.feed_forward.w1.weight",
        "model.layers.11.mlp.up_proj.weight": "layers.11.feed_forward.w3.weight",
        "model.layers.11.mlp.down_proj.weight": "layers.11.feed_forward.w2.weight",
        "model.layers.11.input_layernorm.weight": "layers.11.attention_norm.weight",
        "model.layers.11.post_attention_layernorm.weight": "layers.11.ffn_norm.weight",
        "model.layers.12.self_attn.q_proj.weight": "layers.12.attention.wq.weight",
        "model.layers.12.self_attn.k_proj.weight": "layers.12.attention.wk.weight",
        "model.layers.12.self_attn.v_proj.weight": "layers.12.attention.wv.weight",
        "model.layers.12.self_attn.o_proj.weight": "layers.12.attention.wo.weight",
        "model.layers.12.mlp.gate_proj.weight": "layers.12.feed_forward.w1.weight",
        "model.layers.12.mlp.up_proj.weight": "layers.12.feed_forward.w3.weight",
        "model.layers.12.mlp.down_proj.weight": "layers.12.feed_forward.w2.weight",
        "model.layers.12.input_layernorm.weight": "layers.12.attention_norm.weight",
        "model.layers.12.post_attention_layernorm.weight": "layers.12.ffn_norm.weight",
        "model.layers.13.self_attn.q_proj.weight": "layers.13.attention.wq.weight",
        "model.layers.13.self_attn.k_proj.weight": "layers.13.attention.wk.weight",
        "model.layers.13.self_attn.v_proj.weight": "layers.13.attention.wv.weight",
        "model.layers.13.self_attn.o_proj.weight": "layers.13.attention.wo.weight",
        "model.layers.13.mlp.gate_proj.weight": "layers.13.feed_forward.w1.weight",
        "model.layers.13.mlp.up_proj.weight": "layers.13.feed_forward.w3.weight",
        "model.layers.13.mlp.down_proj.weight": "layers.13.feed_forward.w2.weight",
        "model.layers.13.input_layernorm.weight": "layers.13.attention_norm.weight",
        "model.layers.13.post_attention_layernorm.weight": "layers.13.ffn_norm.weight",
        "model.layers.14.self_attn.q_proj.weight": "layers.14.attention.wq.weight",
        "model.layers.14.self_attn.k_proj.weight": "layers.14.attention.wk.weight",
        "model.layers.14.self_attn.v_proj.weight": "layers.14.attention.wv.weight",
        "model.layers.14.self_attn.o_proj.weight": "layers.14.attention.wo.weight",
        "model.layers.14.mlp.gate_proj.weight": "layers.14.feed_forward.w1.weight",
        "model.layers.14.mlp.up_proj.weight": "layers.14.feed_forward.w3.weight",
        "model.layers.14.mlp.down_proj.weight": "layers.14.feed_forward.w2.weight",
        "model.layers.14.input_layernorm.weight": "layers.14.attention_norm.weight",
        "model.layers.14.post_attention_layernorm.weight": "layers.14.ffn_norm.weight",
        "model.layers.15.self_attn.q_proj.weight": "layers.15.attention.wq.weight",
        "model.layers.15.self_attn.k_proj.weight": "layers.15.attention.wk.weight",
        "model.layers.15.self_attn.v_proj.weight": "layers.15.attention.wv.weight",
        "model.layers.15.self_attn.o_proj.weight": "layers.15.attention.wo.weight",
        "model.layers.15.mlp.gate_proj.weight": "layers.15.feed_forward.w1.weight",
        "model.layers.15.mlp.up_proj.weight": "layers.15.feed_forward.w3.weight",
        "model.layers.15.mlp.down_proj.weight": "layers.15.feed_forward.w2.weight",
        "model.layers.15.input_layernorm.weight": "layers.15.attention_norm.weight",
        "model.layers.15.post_attention_layernorm.weight": "layers.15.ffn_norm.weight",
        "model.layers.16.self_attn.q_proj.weight": "layers.16.attention.wq.weight",
        "model.layers.16.self_attn.k_proj.weight": "layers.16.attention.wk.weight",
        "model.layers.16.self_attn.v_proj.weight": "layers.16.attention.wv.weight",
        "model.layers.16.self_attn.o_proj.weight": "layers.16.attention.wo.weight",
        "model.layers.16.mlp.gate_proj.weight": "layers.16.feed_forward.w1.weight",
        "model.layers.16.mlp.up_proj.weight": "layers.16.feed_forward.w3.weight",
        "model.layers.16.mlp.down_proj.weight": "layers.16.feed_forward.w2.weight",
        "model.layers.16.input_layernorm.weight": "layers.16.attention_norm.weight",
        "model.layers.16.post_attention_layernorm.weight": "layers.16.ffn_norm.weight",
        "model.layers.17.self_attn.q_proj.weight": "layers.17.attention.wq.weight",
        "model.layers.17.self_attn.k_proj.weight": "layers.17.attention.wk.weight",
        "model.layers.17.self_attn.v_proj.weight": "layers.17.attention.wv.weight",
        "model.layers.17.self_attn.o_proj.weight": "layers.17.attention.wo.weight",
        "model.layers.17.mlp.gate_proj.weight": "layers.17.feed_forward.w1.weight",
        "model.layers.17.mlp.up_proj.weight": "layers.17.feed_forward.w3.weight",
        "model.layers.17.mlp.down_proj.weight": "layers.17.feed_forward.w2.weight",
        "model.layers.17.input_layernorm.weight": "layers.17.attention_norm.weight",
        "model.layers.17.post_attention_layernorm.weight": "layers.17.ffn_norm.weight",
        "model.layers.18.self_attn.q_proj.weight": "layers.18.attention.wq.weight",
        "model.layers.18.self_attn.k_proj.weight": "layers.18.attention.wk.weight",
        "model.layers.18.self_attn.v_proj.weight": "layers.18.attention.wv.weight",
        "model.layers.18.self_attn.o_proj.weight": "layers.18.attention.wo.weight",
        "model.layers.18.mlp.gate_proj.weight": "layers.18.feed_forward.w1.weight",
        "model.layers.18.mlp.up_proj.weight": "layers.18.feed_forward.w3.weight",
        "model.layers.18.mlp.down_proj.weight": "layers.18.feed_forward.w2.weight",
        "model.layers.18.input_layernorm.weight": "layers.18.attention_norm.weight",
        "model.layers.18.post_attention_layernorm.weight": "layers.18.ffn_norm.weight",
        "model.layers.19.self_attn.q_proj.weight": "layers.19.attention.wq.weight",
        "model.layers.19.self_attn.k_proj.weight": "layers.19.attention.wk.weight",
        "model.layers.19.self_attn.v_proj.weight": "layers.19.attention.wv.weight",
        "model.layers.19.self_attn.o_proj.weight": "layers.19.attention.wo.weight",
        "model.layers.19.mlp.gate_proj.weight": "layers.19.feed_forward.w1.weight",
        "model.layers.19.mlp.up_proj.weight": "layers.19.feed_forward.w3.weight",
        "model.layers.19.mlp.down_proj.weight": "layers.19.feed_forward.w2.weight",
        "model.layers.19.input_layernorm.weight": "layers.19.attention_norm.weight",
        "model.layers.19.post_attention_layernorm.weight": "layers.19.ffn_norm.weight",
        "model.layers.20.self_attn.q_proj.weight": "layers.20.attention.wq.weight",
        "model.layers.20.self_attn.k_proj.weight": "layers.20.attention.wk.weight",
        "model.layers.20.self_attn.v_proj.weight": "layers.20.attention.wv.weight",
        "model.layers.20.self_attn.o_proj.weight": "layers.20.attention.wo.weight",
        "model.layers.20.mlp.gate_proj.weight": "layers.20.feed_forward.w1.weight",
        "model.layers.20.mlp.up_proj.weight": "layers.20.feed_forward.w3.weight",
        "model.layers.20.mlp.down_proj.weight": "layers.20.feed_forward.w2.weight",
        "model.layers.20.input_layernorm.weight": "layers.20.attention_norm.weight",
        "model.layers.20.post_attention_layernorm.weight": "layers.20.ffn_norm.weight",
        "model.layers.21.self_attn.q_proj.weight": "layers.21.attention.wq.weight",
        "model.layers.21.self_attn.k_proj.weight": "layers.21.attention.wk.weight",
        "model.layers.21.self_attn.v_proj.weight": "layers.21.attention.wv.weight",
        "model.layers.21.self_attn.o_proj.weight": "layers.21.attention.wo.weight",
        "model.layers.21.mlp.gate_proj.weight": "layers.21.feed_forward.w1.weight",
        "model.layers.21.mlp.up_proj.weight": "layers.21.feed_forward.w3.weight",
        "model.layers.21.mlp.down_proj.weight": "layers.21.feed_forward.w2.weight",
        "model.layers.21.input_layernorm.weight": "layers.21.attention_norm.weight",
        "model.layers.21.post_attention_layernorm.weight": "layers.21.ffn_norm.weight",
        "model.layers.22.self_attn.q_proj.weight": "layers.22.attention.wq.weight",
        "model.layers.22.self_attn.k_proj.weight": "layers.22.attention.wk.weight",
        "model.layers.22.self_attn.v_proj.weight": "layers.22.attention.wv.weight",
        "model.layers.22.self_attn.o_proj.weight": "layers.22.attention.wo.weight",
        "model.layers.22.mlp.gate_proj.weight": "layers.22.feed_forward.w1.weight",
        "model.layers.22.mlp.up_proj.weight": "layers.22.feed_forward.w3.weight",
        "model.layers.22.mlp.down_proj.weight": "layers.22.feed_forward.w2.weight",
        "model.layers.22.input_layernorm.weight": "layers.22.attention_norm.weight",
        "model.layers.22.post_attention_layernorm.weight": "layers.22.ffn_norm.weight",
        "model.layers.23.self_attn.q_proj.weight": "layers.23.attention.wq.weight",
        "model.layers.23.self_attn.k_proj.weight": "layers.23.attention.wk.weight",
        "model.layers.23.self_attn.v_proj.weight": "layers.23.attention.wv.weight",
        "model.layers.23.self_attn.o_proj.weight": "layers.23.attention.wo.weight",
        "model.layers.23.mlp.gate_proj.weight": "layers.23.feed_forward.w1.weight",
        "model.layers.23.mlp.up_proj.weight": "layers.23.feed_forward.w3.weight",
        "model.layers.23.mlp.down_proj.weight": "layers.23.feed_forward.w2.weight",
        "model.layers.23.input_layernorm.weight": "layers.23.attention_norm.weight",
        "model.layers.23.post_attention_layernorm.weight": "layers.23.ffn_norm.weight",
        "model.layers.24.self_attn.q_proj.weight": "layers.24.attention.wq.weight",
        "model.layers.24.self_attn.k_proj.weight": "layers.24.attention.wk.weight",
        "model.layers.24.self_attn.v_proj.weight": "layers.24.attention.wv.weight",
        "model.layers.24.self_attn.o_proj.weight": "layers.24.attention.wo.weight",
        "model.layers.24.mlp.gate_proj.weight": "layers.24.feed_forward.w1.weight",
        "model.layers.24.mlp.up_proj.weight": "layers.24.feed_forward.w3.weight",
        "model.layers.24.mlp.down_proj.weight": "layers.24.feed_forward.w2.weight",
        "model.layers.24.input_layernorm.weight": "layers.24.attention_norm.weight",
        "model.layers.24.post_attention_layernorm.weight": "layers.24.ffn_norm.weight",
        "model.layers.25.self_attn.q_proj.weight": "layers.25.attention.wq.weight",
        "model.layers.25.self_attn.v_proj.weight": "layers.25.attention.wv.weight",
        "model.layers.25.self_attn.k_proj.weight": "layers.25.attention.wk.weight",
        "model.layers.25.self_attn.o_proj.weight": "layers.25.attention.wo.weight",
        "model.layers.25.mlp.gate_proj.weight": "layers.25.feed_forward.w1.weight",
        "model.layers.25.mlp.up_proj.weight": "layers.25.feed_forward.w3.weight",
        "model.layers.25.mlp.down_proj.weight": "layers.25.feed_forward.w2.weight",
        "model.layers.25.input_layernorm.weight": "layers.25.attention_norm.weight",
        "model.layers.25.post_attention_layernorm.weight": "layers.25.ffn_norm.weight",
        "model.layers.26.self_attn.q_proj.weight": "layers.26.attention.wq.weight",
        "model.layers.26.self_attn.k_proj.weight": "layers.26.attention.wk.weight",
        "model.layers.26.self_attn.v_proj.weight": "layers.26.attention.wv.weight",
        "model.layers.26.self_attn.o_proj.weight": "layers.26.attention.wo.weight",
        "model.layers.26.mlp.gate_proj.weight": "layers.26.feed_forward.w1.weight",
        "model.layers.26.mlp.up_proj.weight": "layers.26.feed_forward.w3.weight",
        "model.layers.26.mlp.down_proj.weight": "layers.26.feed_forward.w2.weight",
        "model.layers.26.input_layernorm.weight": "layers.26.attention_norm.weight",
        "model.layers.26.post_attention_layernorm.weight": "layers.26.ffn_norm.weight",
        "model.layers.27.self_attn.q_proj.weight": "layers.27.attention.wq.weight",
        "model.layers.27.self_attn.k_proj.weight": "layers.27.attention.wk.weight",
        "model.layers.27.self_attn.v_proj.weight": "layers.27.attention.wv.weight",
        "model.layers.27.self_attn.o_proj.weight": "layers.27.attention.wo.weight",
        "model.layers.27.mlp.gate_proj.weight": "layers.27.feed_forward.w1.weight",
        "model.layers.27.mlp.up_proj.weight": "layers.27.feed_forward.w3.weight",
        "model.layers.27.mlp.down_proj.weight": "layers.27.feed_forward.w2.weight",
        "model.layers.27.input_layernorm.weight": "layers.27.attention_norm.weight",
        "model.layers.27.post_attention_layernorm.weight": "layers.27.ffn_norm.weight",
        "model.layers.28.self_attn.q_proj.weight": "layers.28.attention.wq.weight",
        "model.layers.28.self_attn.k_proj.weight": "layers.28.attention.wk.weight",
        "model.layers.28.self_attn.v_proj.weight": "layers.28.attention.wv.weight",
        "model.layers.28.self_attn.o_proj.weight": "layers.28.attention.wo.weight",
        "model.layers.28.mlp.gate_proj.weight": "layers.28.feed_forward.w1.weight",
        "model.layers.28.mlp.up_proj.weight": "layers.28.feed_forward.w3.weight",
        "model.layers.28.mlp.down_proj.weight": "layers.28.feed_forward.w2.weight",
        "model.layers.28.input_layernorm.weight": "layers.28.attention_norm.weight",
        "model.layers.28.post_attention_layernorm.weight": "layers.28.ffn_norm.weight",
        "model.layers.29.self_attn.q_proj.weight": "layers.29.attention.wq.weight",
        "model.layers.29.self_attn.k_proj.weight": "layers.29.attention.wk.weight",
        "model.layers.29.self_attn.v_proj.weight": "layers.29.attention.wv.weight",
        "model.layers.29.self_attn.o_proj.weight": "layers.29.attention.wo.weight",
        "model.layers.29.mlp.gate_proj.weight": "layers.29.feed_forward.w1.weight",
        "model.layers.29.mlp.up_proj.weight": "layers.29.feed_forward.w3.weight",
        "model.layers.29.mlp.down_proj.weight": "layers.29.feed_forward.w2.weight",
        "model.layers.29.input_layernorm.weight": "layers.29.attention_norm.weight",
        "model.layers.29.post_attention_layernorm.weight": "layers.29.ffn_norm.weight",
        "model.layers.30.self_attn.q_proj.weight": "layers.30.attention.wq.weight",
        "model.layers.30.self_attn.k_proj.weight": "layers.30.attention.wk.weight",
        "model.layers.30.self_attn.v_proj.weight": "layers.30.attention.wv.weight",
        "model.layers.30.self_attn.o_proj.weight": "layers.30.attention.wo.weight",
        "model.layers.30.mlp.gate_proj.weight": "layers.30.feed_forward.w1.weight",
        "model.layers.30.mlp.up_proj.weight": "layers.30.feed_forward.w3.weight",
        "model.layers.30.mlp.down_proj.weight": "layers.30.feed_forward.w2.weight",
        "model.layers.30.input_layernorm.weight": "layers.30.attention_norm.weight",
        "model.layers.30.post_attention_layernorm.weight": "layers.30.ffn_norm.weight",
        "model.layers.31.self_attn.q_proj.weight": "layers.31.attention.wq.weight",
        "model.layers.31.self_attn.k_proj.weight": "layers.31.attention.wk.weight",
        "model.layers.31.self_attn.v_proj.weight": "layers.31.attention.wv.weight",
        "model.layers.31.self_attn.o_proj.weight": "layers.31.attention.wo.weight",
        "model.layers.31.mlp.gate_proj.weight": "layers.31.feed_forward.w1.weight",
        "model.layers.31.mlp.up_proj.weight": "layers.31.feed_forward.w3.weight",
        "model.layers.31.mlp.down_proj.weight": "layers.31.feed_forward.w2.weight",
        "model.layers.31.input_layernorm.weight": "layers.31.attention_norm.weight",
        "model.layers.31.post_attention_layernorm.weight": "layers.31.ffn_norm.weight",
    }
