import torch
import torch.nn as nn
from typing import Optional
import math

from kchanboost.kvcache.utils import KVCache_Layer


import triton
import triton.language as tl


@triton.jit
def qk_kernel(
    # Query (t=1 for decoding step)
    q_ptr,                                                                      # fp16 [B, KV_GROUP, H_KV, 1, D]
    q_stride_b, q_stride_kvg, q_stride_h, q_stride_t, q_stride_d,
    # Key Cache
    sink_ptr_k,                                                                 # fp16 [B, H_KV, S, D]
    sink_stride_b_k, sink_stride_h_k, sink_stride_s_k, sink_stride_d_k,
    qbuff_ptr_k,                                                                # fp16 [B, H_KV, PAGE_SIZE, D]
    qbuff_stride_b_k, qbuff_stride_h_k, qbuff_stride_t_k, qbuff_stride_d_k,
    page_table_ptr_k,                                                           # int64 [B, MAX_PAGE]
    page_table_stride_b_k, page_table_stride_p_k,
    kcache_ptr,                                                                 # uint8 [MAX_BS * self.MAX_PAGE, BYTES_PER_PAGE_K,]
    kcache_stride_p, kcache_stride_last,
    kcache_meta_ptr,                                                            # fp16 [MAX_BS * self.MAX_PAGE, H_KV, D, 2]
    kcache_meta_stride_p, kcache_meta_stride_h, kcache_meta_stride_d, kcache_meta_stride_last,
    # Scaling, float32
    scaling,
    # Output
    output_ptr,                                                                 # fp16 [B, H_Q, t_total]
    out_stride_b, out_stride_hq, out_stride_t_total,
    # Variables
    sink_count,                                                                  # int32
    qbuff_count_k,                                                               # int32
    page_count_k,                                                                # int32
    # Other constants
    H_KV: tl.constexpr,
    H_Q: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    D_BOOST: tl.constexpr,
    S: tl.constexpr,              # Sink size
    #

):
    """
    grid = (B, H_KV)
    每个program为 (b, h) 计算完整 logits 向量：
      [Sink | Paged pages | QBuffer] 依次拼接
    """

    # Constant Strides and Offsets for K cache dequantization
    KV_GROUP:       tl.constexpr = H_Q // H_KV
    K_STRIDE_D:     tl.constexpr = PAGE_SIZE * 2 // 8
    K_STRIDE_H_KV:  tl.constexpr = K_STRIDE_D * (D + D_BOOST) + D   # including channel idx
    K_OFF_HI:       tl.constexpr = K_STRIDE_D * D
    K_OFF_IDX:      tl.constexpr = K_STRIDE_D * (D + D_BOOST)

    pid_b = tl.program_id(0)
    pid_h_kv = tl.program_id(1)

    # Offsets
    offs_kvg = tl.arange(0, KV_GROUP)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE_SIZE)
    offs_s = tl.arange(0, S)
    offs_pack = offs_t // 4
    shifts = (offs_t % 4) * 2

    # Load Query [B, H_Q, 1, D] = [B, KV_GROUP, H_KV, 1, D] → [KV_GROUP, D]
    q = tl.load(
        q_ptr
        + pid_b * q_stride_b
        + pid_h_kv * q_stride_h
        + 0 * q_stride_t
        + offs_kvg[:, None] * q_stride_kvg
        + offs_d[None, :] * q_stride_d
    )

    # Computing the Q * K_Sink
    mask = (offs_s[:, None] < sink_count) & (offs_d[None, :] >= 0)
    k_sink = tl.load(
        sink_ptr_k
        + pid_b * sink_stride_b_k
        + pid_h_kv * sink_stride_h_k
        + offs_s[:, None] * sink_stride_s_k
        + offs_d[None, :] * sink_stride_d_k,
        mask=mask,
    )
    #
    logits_sink = tl.dot(q, tl.trans(k_sink))  # [KV_GROUP, S]
    logits_sink = logits_sink * scaling
    logits_sink = logits_sink.to(tl.float16)
    #
    mask = (offs_s[None, :] < sink_count) & (offs_kvg[:, None] >= 0)
    tl.store(
        output_ptr
        + pid_b * out_stride_b
        + pid_h_kv * KV_GROUP * out_stride_hq
        + offs_kvg[:, None] * out_stride_hq
        + offs_s[None, :] * out_stride_t_total,
        logits_sink,
        mask=mask,
    )

    # Computing the Q * K_Paged
    i = 0
    while i < page_count_k:
        ######################## load dequantized K_page ########################
        page_id = tl.load(
            page_table_ptr_k
            + pid_b * page_table_stride_b_k
            + i * page_table_stride_p_k
        )
        ######################## load scale & zero_point ########################
        meta_base = (
            kcache_meta_ptr
            + page_id * kcache_meta_stride_p
            + pid_h_kv * kcache_meta_stride_h
        )
        scale = tl.load(
            meta_base
            + offs_d * kcache_meta_stride_d
            + 0 * kcache_meta_stride_last)  # [D]
        zero_point = tl.load(
            meta_base
            + offs_d * kcache_meta_stride_d
            + 1 * kcache_meta_stride_last)  # [D]
        ######################## load quantized K_page ########################
        cache_base = kcache_ptr + page_id * kcache_stride_p + pid_h_kv * K_STRIDE_H_KV
        # Loading dense to sparse index
        boost_idx = tl.load(cache_base + K_OFF_IDX + offs_d)  # [D], uint8
        boost_mask = boost_idx < D_BOOST    # [D], bool
        # Loading low 2-bits INT2
        x_uint8_low = tl.load(
            cache_base
            + offs_d[:, None] * K_STRIDE_D
            + offs_pack[None, :] * 1,
        )  # [D, PAGE_SIZE] uint8
        x_uint8_low = (x_uint8_low >> shifts[None, :]) & 0x3  # [D, PAGE_SIZE] uint8
        # Loading high 2-bits INT2 (Only boosted channels)
        mask = boost_mask[:, None] & (offs_pack >= 0)  # [D, PAGE_SIZE], bool
        x_uint8_high = tl.load(
            cache_base + K_OFF_HI
            + boost_idx[:, None] * K_STRIDE_D
            + offs_pack[None, :] * 1,
            mask=mask,
            padding_option="zero"
        )  # [D_BOOST, PAGE_SIZE] uint8
        x_uint8_high = (x_uint8_high >> shifts[None, :]) & 0x3  # [D_BOOST, PAGE_SIZE] uint8
        ######################## dequantizing the K_page ########################
        # combine high and low bits
        x_uint8 = x_uint8_low | (x_uint8_high << 2)  # [D, PAGE_SIZE] uint8
        # dequantize
        x_fp16 = x_uint8.to(tl.float16)  # [D, PAGE_SIZE]
        x_fp16 = x_fp16 * scale[:, None] + zero_point[:, None]  # [D, PAGE_SIZE]
        # Computing the Q * K_page
        logits_page = tl.dot(q, x_fp16)  # [KV_GROUP, PAGE_SIZE]
        logits_page = logits_page * scaling
        logits_page = logits_page.to(tl.float16)
        ######################## store logits_page ########################
        tl.store(output_ptr
            + pid_b * out_stride_b
            + pid_h_kv * KV_GROUP * out_stride_hq
            + offs_kvg[:, None] * out_stride_hq
            + (sink_count + i * PAGE_SIZE + offs_t)[None, :] * out_stride_t_total,
            logits_page
        )
        ######################## iterate next page ########################
        i += 1

    # Computing the Q * K_QBuffer
    mask = (offs_t[:, None] < qbuff_count_k) & (offs_d[None, :] >= 0)
    k_qbuff = tl.load(
        qbuff_ptr_k
        + pid_b * qbuff_stride_b_k
        + pid_h_kv * qbuff_stride_h_k
        + offs_t[:, None] * qbuff_stride_t_k
        + offs_d[None, :] * qbuff_stride_d_k,
        mask=mask,
    )
    #
    logits_qbuff = tl.dot(q, tl.trans(k_qbuff))  # [KV_GROUP, PAGE_SIZE]
    logits_qbuff = logits_qbuff * scaling
    logits_qbuff = logits_qbuff.to(tl.float16)
    #
    mask = (offs_t[None, :] < qbuff_count_k) & (offs_kvg[:, None] >= 0)
    tl.store(
        output_ptr
        + pid_b * out_stride_b
        + pid_h_kv * KV_GROUP * out_stride_hq
        + offs_kvg[:, None] * out_stride_hq
        + (sink_count + page_count_k * PAGE_SIZE + offs_t)[None, :] * out_stride_t_total,
        logits_qbuff,
        mask=mask,
    )


@triton.jit
def sv_kernel(
    # softmax logits (Input)
    s_ptr,                                                                      # fp16 [B, H_Q, t_total]
    s_stride_b, s_stride_hq, s_stride_t_total,
    # Value Cache (Input)
    sink_ptr_v,                                                                 # fp16 [B, H_KV, S, D]
    sink_stride_b_v, sink_stride_h_v, sink_stride_s_v, sink_stride_d_v,
    qbuff_ptr_v,                                                                # fp16 [B, H_KV, PAGE_SIZE, D]
    qbuff_stride_b_v, qbuff_stride_h_v, qbuff_stride_p_v, qbuff_stride_d_v,
    local_ptr,                                                                  # fp16 [B, H_KV, PAGE_SIZE, D]
    local_stride_b, local_stride_h, local_stride_p, local_stride_d,
    page_table_ptr_v,                                                           # int64 [B, MAX_PAGE]
    page_table_stride_b_v, page_table_stride_p_v,
    vcache_ptr,                                                                 # uint8 [MAX_BS * self.MAX_PAGE, BYTES_PER_PAGE_V,]
    vcache_stride_p, vcache_stride_last,
    vcache_meta_ptr,                                                            # fp16 [MAX_BS * self.MAX_PAGE, H_KV, D, 2]
    vcache_meta_stride_p, vcache_meta_stride_h, vcache_meta_stride_d, vcache_meta_stride_last,
    # Output
    output_ptr,                                                                 # fp16 [B, H_Q, 1, D]
    output_stride_b, output_stride_h, output_stride_t, output_stride_d,
    # Variables
    sink_count,                                                                  # int32
    qbuff_count_v,                                                               # int32
    local_count_v,                                                               # int32
    local_offset_v,                                                              # int32
    page_count_v,                                                                # int32 
    # Constants
    BYTES_PER_PAGE_V: tl.constexpr,
    V_STRIDE_T: tl.constexpr,     # bytes, D*2//8
    # Other constants
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    D_BOOST: tl.constexpr,
):
    """
    grid = (B, H_KV)
    每个program为 (b, h) 计算完整 attn_output 向量：
      [Sink | Paged pages | QBuffer | Local Buffer] 依次拼接
    """
    


def kchanboost_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    kv_cache: KVCache_Layer,
    scaling: float,
):
    assert query.is_contiguous(), "Query tensor must be contiguous."
    B, H_Q, t_query, D = query.size()
    assert t_query == 1, "Only decoding step with t_query=1 is supported."

    # Reshape query to [B, KV_GROUP, H_KV, 1, D]
    query = query.view(B, module.num_key_value_groups, H_Q // module.num_key_value_groups, 1, D)

    logits = torch.empty(
        (B, H_Q, kv_cache.get_seq_length()),
        dtype=query.dtype,
        device=query.device,
    )

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights