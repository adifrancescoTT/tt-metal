import torch
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


def get_rand_falcon_inputs(
    llm_mode, seq_len, batch, kv_cache_len, devices, global_batch, head_dim, max_position_embeddings, configuration
):
    # Generate input, attention_mask, and kv_cache --------------------------------------
    # TODO: Generate attention_mask on device
    tt_attention_input = []
    tt_attention_mask = []
    tt_layer_past = []
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert batch == 1, "For prefill, batch must be 1!"
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        attention_input = (torch.rand(global_batch, q_len, configuration.hidden_size) * 2) - 1
        attention_mask_bool = torch.ones(global_batch, 1, q_len, kv_len, dtype=bool).triu(diagonal=1)
        layer_past = None

        for i, device in enumerate(devices):
            attention_input_i = attention_input[batch * i : batch * (i + 1)]
            attention_mask_bool_i = attention_mask_bool[batch * i : batch * (i + 1)]
            tt_attention_input.append(torch2tt_tensor(attention_input_i.unsqueeze(1), device))
            tt_attention_mask.append(
                torch2tt_tensor(
                    (attention_mask_bool_i * -100000).expand(-1, configuration.num_attention_heads, -1, -1),
                    device,
                )
            )
            tt_k_cache = torch.zeros(batch, max_position_embeddings, head_dim)
            tt_v_cache = torch.zeros(batch, max_position_embeddings, head_dim)
            tt_k_cache = torch2tt_tensor(tt_k_cache.unsqueeze(1), device)
            tt_v_cache = torch2tt_tensor(tt_v_cache.unsqueeze(1), device)
            tt_layer_past.append((tt_k_cache, tt_v_cache))

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        attention_input = (torch.rand(global_batch, q_len, configuration.hidden_size) * 2) - 1
        # attention_input = (torch.rand(batch, q_len, 4544) * 2) - 1
        attention_mask_bool = torch.zeros(global_batch, 1, q_len, kv_len, dtype=bool)
        k_cache = torch.rand(global_batch, kv_cache_len, head_dim)
        v_cache = torch.rand(global_batch, kv_cache_len, head_dim)
        layer_past = (k_cache, v_cache)

        for i, device in enumerate(devices):
            tt_attention_input.append(
                torch2tt_tensor(attention_input[batch * i : batch * (i + 1)].unsqueeze(1).transpose(0, 2), device)
            )

            kv_len_padded = (kv_len + 31) // 32 * 32
            attention_mask_bool_padded = torch.cat(
                (
                    attention_mask_bool[batch * i : batch * (i + 1)],
                    torch.ones(batch, 1, q_len, kv_len_padded - kv_len, dtype=bool),
                ),
                dim=-1,
            )
            tt_attention_mask.append(
                torch2tt_tensor(
                    (attention_mask_bool_padded.transpose(0, 2) * -100000).expand(
                        -1,
                        configuration.num_attention_heads,
                        -1,
                        -1
                        # -1, 71, -1, -1
                    ),
                    device,
                )
            )
            tt_k_cache = torch.zeros(batch, max_position_embeddings, head_dim)
            tt_v_cache = torch.zeros(batch, max_position_embeddings, head_dim)
            tt_k_cache[:, :kv_cache_len, :] = k_cache[batch * i : batch * (i + 1)]
            tt_v_cache[:, :kv_cache_len, :] = v_cache[batch * i : batch * (i + 1)]
            tt_k_cache = torch2tt_tensor(tt_k_cache.unsqueeze(1), device)
            tt_v_cache = torch2tt_tensor(tt_v_cache.unsqueeze(1), device)
            tt_layer_past.append((tt_k_cache, tt_v_cache))

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    return (
        attention_input,
        attention_mask_bool,
        layer_past,
        tt_attention_input,
        tt_attention_mask,
        tt_layer_past,
        kv_len,
    )


def concat_device_outputs(num_devices, tt_out, llm_mode, tt_layer_present, kv_len):
    for i in range(num_devices):
        tt_out[i] = tt2torch_tensor(tt_out[i]).squeeze(1)
        if llm_mode == "decode":
            tt_out[i] = tt_out[i].transpose(0, 1)
        tt_layer_present[i] = (
            tt2torch_tensor(tt_layer_present[i][0]).squeeze(1),
            tt2torch_tensor(tt_layer_present[i][1]).squeeze(1),
        )
        tt_layer_present[i] = (
            tt_layer_present[i][0][:, :kv_len, :],
            tt_layer_present[i][1][:, :kv_len, :],
        )
    tt_out = torch.concat(tt_out)
    tt_layer_present = (torch.concat([x[0] for x in tt_layer_present]), torch.concat([x[1] for x in tt_layer_present]))
    return tt_out, tt_layer_present
