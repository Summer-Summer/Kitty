import torch
from typing import Optional
import math

__all__ = [
    "build_promote_mask",
    "fake_quant_groupwise_lastdim",
]


# ToDo: Support batch size > 1, promote_mask  : (B, nh, D) bool
def build_promote_mask(key_states: torch.Tensor, promote_ratio: float, channel_selection: int) -> torch.BoolTensor:
    """
    Generating a mask to select channels based on importance scores.
    args:
        key_states : (B, nh, D, T),     key cache after RoPE but before quantization
        promote_ratio : float,          the ratio of channels to promote, in [0, 1]
        channel_selection : int,        channel selection strategy
                                        (-1) for Unspecified, raise an error;
                                        (0)  for Random Selection;
                                        (1)  Variance-based Channel Selection;
                                        (2)  Magnitude-based Channel Selection;
                                        (3)  RoPE-aware Channel Selection;
    returns:
        promote_mask  : (nh, D) bool,      promote_mask[i][j] == True -> j-th channel of i-th head is selected for promotion
    """
    assert key_states.dim() == 4
    _, nh, D, _ = key_states.shape
    assert D % 2 == 0, "RoPE-aware requires even D"
    assert 0. <= promote_ratio <= 1.0, f"promote_ratio must be in [0, 1], got {promote_ratio}"
    # corner cases
    k_chan = int(D * promote_ratio + 1e-6)  # number of channels to promote
    if k_chan % 2 != 0:  # ensure k_chan is even
        k_chan += 1
    if k_chan == 0:  # promote no channel
        return torch.zeros((nh, D), dtype=torch.bool, device=key_states.device)
    if k_chan >= D:  # promote all
        return torch.ones((nh, D), dtype=torch.bool, device=key_states.device)
    promote_mask = torch.zeros((nh, D), dtype=torch.bool, device=key_states.device)
    ##############################################channel selection strategies###############################################
    assert channel_selection in (-1, 0, 1, 2, 3), f"Invalid channel_selection strategy: {channel_selection}"
    if channel_selection == -1:                                                             # (-1) for Unspecified, raise an error;
        assert False, "channel_selection strategy is not set."
    elif channel_selection == 0:                                                            # (0)  for Random Selection;
        for i in range(nh):
            rand_idx = torch.randperm(D, device=key_states.device)[:k_chan]
            promote_mask[i, rand_idx] = True
    elif channel_selection == 1:                                                            # (1)  Variance-based Channel Selection
        diff = key_states - key_states.mean(dim=-1, keepdim=True)
        score = diff.pow(2).mean(dim=-1).mean(dim=0)  # (B, nh, D, T) → (B, nh, D) → (nh, D)
        _, top_idx = score.topk(k_chan, dim=-1)
        promote_mask.scatter_(-1, top_idx, True)   # True → promote channel
    elif channel_selection == 2:                                                            # (2)  Magnitude-based Channel Selection
        score = key_states.abs()
        score = score.mean(dim=-1).mean(dim=0)  # (B, nh, D, T) → (B, nh, D) → (nh, D)
        _, top_idx = score.topk(k_chan, dim=-1)
        promote_mask.scatter_(-1, top_idx, True)   # True → promote channel
    elif channel_selection == 3:                                                            # (3)  RoPE-aware Channel Selection
        Amplitude = (key_states[:, :, :D//2, :].pow(2) + key_states[:, :, D//2:, :].pow(2))  
        Amplitude = Amplitude.sqrt()        # (B, nh, D//2, T)
        Amplitude = Amplitude.mean(dim=-1)  # (B, nh, D//2)
        score = Amplitude.mean(dim=0)       # (nh, D//2)
        #
        k_pair = k_chan // 2                            # number of channel pairs to promote
        _, top_pair_idx = score.topk(k_pair, dim=-1)    # (nh, k_pair)
        for i in range(nh):
            idx0 = top_pair_idx[i]                      # (k_pair,)
            idx1 = idx0 + D // 2                        # shift
            promote_mask[i].scatter_(0, idx0, True)
            promote_mask[i].scatter_(0, idx1, True)
    ########################################################################################################################
    else:
        raise ValueError(f"Invalid channel_selection strategy: {channel_selection}")
    #
    return promote_mask



def fake_quant_groupwise_lastdim(
        data: torch.Tensor,
        group_size: int,
        bit: int,
        promote_mask: Optional[torch.Tensor] = None,
        promote_bit: Optional[int] = 4,
) -> torch.Tensor:
    """
    Simulate the numerical effect of group-wise quantization along the last dim.
    Input and output are both fp16 tensors, used for 'fake quantization'.
    Args:
        data: (B, nh, D, T) - input float tensor
        group_size: int - number of elements per quant group along last dim
        bit: int - quantization bit width (e.g. 2 or 4)
    Optional Args:
        promote_mask: (nh, D) - bool tensor, True → use promote_bit, False → use bit
        promote_bit:  int - bit width for channels that are promoted to (e.g. 4)
    Returns:
        dequantized_data: same shape as input, fp16 tensor with fake quantized values
    """
    assert data.dim() == 4, "Expected input shape [B, nh, D, T]"
    B, nh, D, T = data.shape
    assert T % group_size == 0, "T must be divisible by group_size"
    if bit >= 16:   # No quantization needed, return the original data
        return data
    G = T // group_size
    data = data.contiguous()  # Ensure contiguous memory layout
    x = data.view(B, nh, D, G, group_size)
    # Compute min and max per group
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    eps = 1e-4 if data.dtype in (torch.float16, torch.bfloat16) else 1e-6  # numerical stability
    
    if promote_mask is not None:
        assert promote_mask.shape == (nh, D), f"Expected mask shape (nh, D), got {promote_mask.shape}"
        scale_base = (mx - mn).clamp(min=eps) / (2 ** bit - 1)
        scale_promote = (mx - mn).clamp(min=eps) / (2 ** promote_bit - 1)
        promote_mask = promote_mask.view(1, nh, D, 1, 1)
        scale = torch.where(promote_mask, scale_promote, scale_base)
        max_val = torch.where(
            promote_mask,
            torch.full_like(scale, 2 ** promote_bit - 1),
            torch.full_like(scale, 2 ** bit - 1)
        )
    else:
        scale = (mx - mn).clamp(min=eps) / (2 ** bit - 1)
        max_val = torch.full_like(scale, 2 ** bit - 1)
    # fake quantization
    q = ((x - mn) / scale).clamp(torch.zeros_like(max_val), max_val).round()
    dq = q * scale + mn
    return dq.view(B, nh, D, T).to(data.dtype)