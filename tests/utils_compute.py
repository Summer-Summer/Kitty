import torch
import torch.nn as nn
from typing import Optional

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def create_causal_mask(batch_size: int, num_heads: int, seq_len: int, dtype=torch.float32, device='cuda'):
    """
    返回一个 causal attention mask: (B, num_heads, seq_len, seq_len)，
    下三角为 0, 上三角为 -inf.
    """
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf，下三角为 0
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
    mask = mask.expand(batch_size, num_heads, seq_len, seq_len).to(dtype)
    return mask

def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
):
    b, h_q, T, D = query.shape
    h_kv = key.shape[1]
    #
    scaling = D**-0.5
    num_key_value_groups = h_q // h_kv

    #
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    attention_mask = create_causal_mask(b, h_q, T, dtype=query.dtype, device=query.device)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def fake_quant_groupwise_lastdim(
        data: torch.Tensor,
        group_size: int,
        quantize_mask: torch.Tensor,
        quantize_bit: int = 4,
) -> torch.Tensor:
    """
    Simulate the numerical effect of group-wise quantization along the last dim.
    Input and output are both fp16 tensors, used for 'fake quantization'.
    Args:
        data: (B, nh, T, D) - input float tensor
        group_size: int - number of elements per quant group along last dim
        quantize_mask: (nh, D) - bool tensor, True → use quantize_bit, False → use fp16
        quantize_bit: int - quantization bit width (e.g. 2 or 4)
    Returns:
        dequantized_data: same shape as input, fp16 tensor with fake quantized values
    """
    assert data.dim() == 4, "Expected input shape [B, nh, T, D]"
    B, nh, T, D = data.shape
    assert T % group_size == 0, "T must be divisible by group_size"
    G = T // group_size
    #
    x = data.permute(0, 1, 3, 2).contiguous().view(B, nh, D, G, group_size)  # (B, nh, D, G, gsize)
    # min/max per group
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    eps = 1e-4
    scale = (mx - mn).clamp(min=eps) / (2 ** quantize_bit - 1)
    # quantize and dequantize
    q = ((x - mn) / scale).clamp(0, 2 ** quantize_bit - 1).round()
    dq = q * scale + mn  # fake quantized value

    # broadcast mask
    quantize_mask = quantize_mask.view(1, nh, D, 1, 1)  # (1, nh, D, 1, 1)
    quantize_mask = quantize_mask.to(x.device)
    out = torch.where(quantize_mask, dq, x)

    # reshape back to (B, nh, T, D)
    out = out.view(B, nh, D, T).permute(0, 1, 3, 2).contiguous().to(data.dtype)
    return out