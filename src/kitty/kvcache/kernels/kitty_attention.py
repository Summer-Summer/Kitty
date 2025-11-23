# MIT License
# Copyright (c) 2025 Haojun Xia
# See the LICENSE file in the project root for more information.

# src/kitty/kvcache/kernels/kitty_attention.py

import torch
import torch.nn as nn

from kitty.kvcache.utils_kv_per_layer import KVCache_Layer

import triton
import triton.language as tl


@triton.jit
def qk_kernel(
    # Query (t=1 for decoding step)
    q_ptr,                                                                      # fp16 [B, H_KV, KV_GROUP, 1, D]
    q_stride_b, q_stride_h, q_stride_kvg, q_stride_t, q_stride_d,
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
    S: tl.constexpr,              # Sink size
    D_BOOST: tl.constexpr,
    # Tiling constants
    MAX_KV_GROUP: tl.constexpr
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
    offs_max_kvg = tl.arange(0, MAX_KV_GROUP)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE_SIZE)
    offs_s = tl.arange(0, S)
    offs_pack = offs_t // 4
    shifts = (offs_t % 4) * 2

    # Load Query [B, KV_GROUP, H_KV, 1, D] → [MAX_KV_GROUP, D]
    mask_load_q = (offs_max_kvg[:, None] < KV_GROUP) & (offs_d[None, :] >= 0)
    q = tl.load(
        q_ptr
        + pid_b * q_stride_b
        + pid_h_kv * q_stride_h
        + 0 * q_stride_t
        + offs_max_kvg[:, None] * q_stride_kvg
        + offs_d[None, :] * q_stride_d,
        mask=mask_load_q,
        other=0.0
    )

    # Computing the Q * K_Sink
    mask_load_k_sink = (offs_s[:, None] < sink_count) & (offs_d[None, :] >= 0)
    k_sink = tl.load(
        sink_ptr_k
        + pid_b * sink_stride_b_k
        + pid_h_kv * sink_stride_h_k
        + offs_s[:, None] * sink_stride_s_k
        + offs_d[None, :] * sink_stride_d_k,
        mask=mask_load_k_sink,
        other=0.0
    )
    #
    logits_sink = tl.dot(q, tl.trans(k_sink))  # [MAX_KV_GROUP, S]
    logits_sink = logits_sink * scaling
    logits_sink = logits_sink.to(tl.float16)
    #
    mask_store_score_sink = (offs_s[None, :] < sink_count) & (offs_max_kvg[:, None] < KV_GROUP)
    tl.store(
        output_ptr
        + pid_b * out_stride_b
        + pid_h_kv * KV_GROUP * out_stride_hq
        + offs_max_kvg[:, None] * out_stride_hq
        + offs_s[None, :] * out_stride_t_total,
        logits_sink,
        mask=mask_store_score_sink,
    )

    # Computing the Q * K_Paged
    i = 0
    while i < page_count_k:
        ######################## load page ID ########################
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
        mask_load_k_page = boost_mask[:, None] & (offs_pack >= 0)  # [D, PAGE_SIZE], bool
        x_uint8_high = tl.load(
            cache_base + K_OFF_HI
            + boost_idx[:, None] * K_STRIDE_D
            + offs_pack[None, :] * 1,
            mask=mask_load_k_page,
            other=0.0
        )  # [D_BOOST, PAGE_SIZE] uint8
        x_uint8_high = (x_uint8_high >> shifts[None, :]) & 0x3  # [D_BOOST, PAGE_SIZE] uint8
        ######################## dequantizing the K_page ########################
        # combine high and low bits
        x_uint8 = x_uint8_low | (x_uint8_high << 2)  # [D, PAGE_SIZE] uint8
        # dequantize
        x_fp16 = x_uint8.to(tl.float16)  # [D, PAGE_SIZE]
        x_fp16 = x_fp16 * scale[:, None] + zero_point[:, None]  # [D, PAGE_SIZE]
        ######################## Computing the Q * K_page ########################
        logits_page = tl.dot(q, x_fp16)  # [MAX_KV_GROUP, PAGE_SIZE]
        logits_page = logits_page * scaling
        logits_page = logits_page.to(tl.float16)
        ######################## store logits_page ########################
        mask_store_score_page = (offs_max_kvg[:, None] < KV_GROUP) & (offs_t[None, :] >= 0)
        tl.store(output_ptr
            + pid_b * out_stride_b
            + pid_h_kv * KV_GROUP * out_stride_hq
            + offs_max_kvg[:, None] * out_stride_hq
            + (sink_count + i * PAGE_SIZE + offs_t)[None, :] * out_stride_t_total,
            logits_page,
            mask=mask_store_score_page,
        )
        ######################## iterate next page ########################
        i += 1

    # Computing the Q * K_QBuffer
    mask_load_k_qbuff = (offs_t[:, None] < qbuff_count_k) & (offs_d[None, :] >= 0)
    k_qbuff = tl.load(
        qbuff_ptr_k
        + pid_b * qbuff_stride_b_k
        + pid_h_kv * qbuff_stride_h_k
        + offs_t[:, None] * qbuff_stride_t_k
        + offs_d[None, :] * qbuff_stride_d_k,
        mask=mask_load_k_qbuff,
        other=0.0
    )
    #
    logits_qbuff = tl.dot(q, tl.trans(k_qbuff))  # [MAX_KV_GROUP, PAGE_SIZE]
    logits_qbuff = logits_qbuff * scaling
    logits_qbuff = logits_qbuff.to(tl.float16)
    #
    mask_store_score_qbuff = (offs_t[None, :] < qbuff_count_k) & (offs_max_kvg[:, None] < KV_GROUP)
    tl.store(
        output_ptr
        + pid_b * out_stride_b
        + (pid_h_kv * KV_GROUP + offs_max_kvg[:, None]) * out_stride_hq
        + (sink_count + page_count_k * PAGE_SIZE + offs_t)[None, :] * out_stride_t_total,
        logits_qbuff,
        mask=mask_store_score_qbuff,
    )


@triton.jit
def sv_kernel(
    # softmax logits (Input)
    attn_score_ptr,                                                             # fp16 [B, H_Q, t_total]
    attn_score_stride_b, attn_score_stride_hq, attn_score_stride_t_total,
    # Value Cache (Input)
    sink_ptr_v,                                                                 # fp16 [B, H_KV, S, D]
    sink_stride_b_v, sink_stride_h_v, sink_stride_s_v, sink_stride_d_v,
    qbuff_ptr_v,                                                                # fp16 [B, H_KV, PAGE_SIZE, D]
    qbuff_stride_b_v, qbuff_stride_h_v, qbuff_stride_t_v, qbuff_stride_d_v,
    local_ptr,                                                                  # fp16 [B, H_KV, PAGE_SIZE, D]
    local_stride_b, local_stride_h, local_stride_t, local_stride_d,
    page_table_ptr_v,                                                           # int64 [B, MAX_PAGE]
    page_table_stride_b_v, page_table_stride_p_v,
    vcache_ptr,                                                                 # uint8 [MAX_BS * self.MAX_PAGE, BYTES_PER_PAGE_V,]
    vcache_stride_p, vcache_stride_last,
    vcache_meta_ptr,                                                            # fp16 [MAX_BS * self.MAX_PAGE, H_KV, PAGE_SIZE, 2]
    vcache_meta_stride_p, vcache_meta_stride_h, vcache_meta_stride_t, vcache_meta_stride_last,
    # Output in shape [B, T=1, H_Q, D] to be compatible with huggingface transformers.
    output_ptr,                                                                 # fp16 [B, H_Q, 1, D]
    output_stride_b, output_stride_t, output_stride_hq, output_stride_d,
    # Variables
    sink_count,                                                                  # int32
    qbuff_count_v,                                                               # int32
    local_count_v,                                                               # int32
    local_offset_v,                                                              # int32
    page_count_v,                                                                # int32 
    # Other constants
    H_KV: tl.constexpr,
    H_Q: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,              # Sink size
    # Tiling constants
    MAX_KV_GROUP: tl.constexpr
):
    """
    grid = (B, H_KV)
    每个program为 (b, h) 计算完整 attn_output 向量：
      [Sink | Paged pages | QBuffer | Local Buffer] 依次拼接
    """
    
    # Constant Strides and Offsets for V cache dequantization
    V_STRIDE_T:     tl.constexpr = D * 2 // 8
    V_STRIDE_H_KV:  tl.constexpr = V_STRIDE_T * PAGE_SIZE
    KV_GROUP:       tl.constexpr = H_Q // H_KV

    #
    pid_b = tl.program_id(0)
    pid_h_kv = tl.program_id(1)

    # Offsets
    offs_s = tl.arange(0, S)
    offs_max_kvg = tl.arange(0, MAX_KV_GROUP)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE_SIZE)
    offs_pack = offs_d // 4
    shifts = (offs_d % 4) * 2

    # Computing the Sink part
    mask_load_score_sink = (offs_s[None, :] < sink_count) & (offs_max_kvg[:, None] < KV_GROUP)
    attn_score_sink = tl.load(
        attn_score_ptr
        + pid_b * attn_score_stride_b
        + (pid_h_kv * KV_GROUP + offs_max_kvg[:, None]) * attn_score_stride_hq
        + (0 + offs_s[None, :]) * attn_score_stride_t_total,
        mask=mask_load_score_sink,
        other=0.0
    )  # [MAX_KV_GROUP, S]
    mask_load_v_sink = (offs_s[:, None] < sink_count) & (offs_d[None, :] >= 0)
    v_sink = tl.load(
        sink_ptr_v
        + pid_b * sink_stride_b_v
        + pid_h_kv * sink_stride_h_v
        + offs_s[:, None] * sink_stride_s_v
        + offs_d[None, :] * sink_stride_d_v,
        mask=mask_load_v_sink,
        other=0.0
    )  # [S, D]
    attn_output_acc = tl.dot(attn_score_sink, v_sink)  # [MAX_KV_GROUP, D]

    # Computing the Paged part
    i = 0
    while i < page_count_v:
        ######################## load page ID ########################
        page_id = tl.load(
            page_table_ptr_v
            + pid_b * page_table_stride_b_v
            + i * page_table_stride_p_v
        )
        ######################## load scale & zero_point ########################
        meta_base = vcache_meta_ptr + page_id * vcache_meta_stride_p + pid_h_kv * vcache_meta_stride_h
        scale = tl.load(
            meta_base 
            + offs_t * vcache_meta_stride_t
            + 0 * vcache_meta_stride_last)  # [PAGE_SIZE]
        zero_point = tl.load(
            meta_base 
            + offs_t * vcache_meta_stride_t
            + 1 * vcache_meta_stride_last)  # [PAGE_SIZE]
        ######################## load quantized V_page ########################
        cache_base = vcache_ptr + page_id * vcache_stride_p + pid_h_kv * V_STRIDE_H_KV
        x_uint8 = tl.load(
            cache_base
            + offs_t[:, None] * V_STRIDE_T
            + offs_pack[None, :] * 1,
        )  # [PAGE_SIZE, D] uint8
        x_uint8 = (x_uint8 >> shifts[None, :]) & 0x3  # [PAGE_SIZE, D] uint8
        ######################## dequantizing the V_page ########################
        x_fp16 = x_uint8.to(tl.float16)  # [PAGE_SIZE, D]
        x_fp16 = x_fp16 * scale[:, None] + zero_point[:, None]  # [PAGE_SIZE, D]
        ######################## load a page of attention score ########################
        mask_load_score_page = (offs_max_kvg[:, None] < KV_GROUP) & (offs_t[None, :] >= 0)
        attn_score_page = tl.load(
            attn_score_ptr
            + pid_b * attn_score_stride_b
            + (pid_h_kv * KV_GROUP + offs_max_kvg[:, None]) * attn_score_stride_hq
            + (S + i * PAGE_SIZE + offs_t[None, :]) * attn_score_stride_t_total,
            mask=mask_load_score_page,
            other=0.0
        )  # [MAX_KV_GROUP, PAGE_SIZE]
        ######################## compute attn_output_page ########################
        attn_output_acc = tl.dot(attn_score_page, x_fp16, attn_output_acc)  # [MAX_KV_GROUP, D]
        ######################## iterate next page ########################
        i += 1

    # Computing the QBuffer part
    #
    mask_load_score_qbuff = (offs_t[None, :] < qbuff_count_v) & (offs_max_kvg[:, None] < KV_GROUP)
    attn_score_qbuff = tl.load(
        attn_score_ptr
        + pid_b * attn_score_stride_b
        + (pid_h_kv * KV_GROUP + offs_max_kvg[:, None]) * attn_score_stride_hq
        + (S + page_count_v * PAGE_SIZE + offs_t[None, :]) * attn_score_stride_t_total,
        mask=mask_load_score_qbuff,
        other=0.0
    )  # [KV_GROUP, PAGE_SIZE]
    #
    mask_load_v_qbuff = (offs_t[:, None] < qbuff_count_v) & (offs_d[None, :] >= 0)
    v_qbuff = tl.load(
        qbuff_ptr_v
        + pid_b * qbuff_stride_b_v
        + pid_h_kv * qbuff_stride_h_v
        + offs_t[:, None] * qbuff_stride_t_v
        + offs_d[None, :] * qbuff_stride_d_v,
        mask=mask_load_v_qbuff,
        other=0.0
    )  # [PAGE_SIZE, D]
    #
    attn_output_acc = tl.dot(attn_score_qbuff, v_qbuff, attn_output_acc)  # [MAX_KV_GROUP, D]

    # Computing the Local Buffer part
    #
    mask_load_score_local = (offs_t[None, :] < local_count_v) & (offs_max_kvg[:, None] < KV_GROUP)
    attn_score_local = tl.load(
        attn_score_ptr
        + pid_b * attn_score_stride_b
        + (pid_h_kv * KV_GROUP + offs_max_kvg[:, None]) * attn_score_stride_hq
        + (S + page_count_v * PAGE_SIZE + qbuff_count_v + offs_t[None, :]) * attn_score_stride_t_total,
        mask=mask_load_score_local,
        other=0.0
    )  # [MAX_KV_GROUP, PAGE_SIZE]
    #
    offs_local = tl.arange(0, PAGE_SIZE) + local_offset_v
    offs_local = offs_local % PAGE_SIZE
    mask_load_v_local = (offs_t[:, None] < local_count_v) & (offs_d[None, :] >= 0)
    v_local = tl.load(
        local_ptr
        + pid_b * local_stride_b
        + pid_h_kv * local_stride_h
        + offs_local[:, None] * local_stride_t
        + offs_d[None, :] * local_stride_d,
        mask=mask_load_v_local,
        other=0.0
    )  # [PAGE_SIZE, D]
    #
    attn_output_acc = tl.dot(attn_score_local, v_local, attn_output_acc)  # [MAX_KV_GROUP, D]

    # Store attn_output
    attn_output_acc = attn_output_acc.to(tl.float16)
    mask_store_attn_output = (offs_max_kvg[:, None] < KV_GROUP) & (offs_d[None, :] >= 0)
    tl.store(
        output_ptr
        + output_stride_b * pid_b
        + output_stride_t * 0
        + output_stride_hq * (pid_h_kv * KV_GROUP + offs_max_kvg[:, None])
        + output_stride_d * offs_d[None, :],
        attn_output_acc,
        mask=mask_store_attn_output,
    )



def kitty_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    kv_cache: KVCache_Layer,
    scaling: float,
):
    assert query.is_contiguous(), "Query tensor must be contiguous."
    B, H_Q, t_query, D = query.size()
    assert t_query == 1, "Only decoding step with t_query=1 is supported."

    assert H_Q == module.num_attention_heads, "H_Q must match num_attention_heads."
    H_KV = module.num_key_value_heads
    KV_GROUP = H_Q // H_KV

    #
    MAX_KV_GROUP = 16   # Tiling config for tl.dot()
    assert KV_GROUP <= MAX_KV_GROUP, f"KV_GROUP ({KV_GROUP}) exceeds MAX_KV_GROUP ({MAX_KV_GROUP})."
    
    # Reshape query to [B, H_KV, KV_GROUP, 1, D]
    query = query.view(B, H_KV, KV_GROUP, 1, D)

    # Prepare output tensor for attention scores
    t_total_kvcache = kv_cache.get_total_length()
    attn_score = torch.empty(
        (B, H_Q, t_total_kvcache),
        dtype=query.dtype,
        device=query.device,
    )

    # Launch QK kernel
    grid = (B, H_KV)
    qk_kernel[grid](
        # Query
        query,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3), query.stride(4),
        # Key Cache
        kv_cache.Sink_Buffer_K,
        kv_cache.Sink_Buffer_K.stride(0), kv_cache.Sink_Buffer_K.stride(1),
        kv_cache.Sink_Buffer_K.stride(2), kv_cache.Sink_Buffer_K.stride(3),
        kv_cache.Q_Buffer_K,
        kv_cache.Q_Buffer_K.stride(0), kv_cache.Q_Buffer_K.stride(1),
        kv_cache.Q_Buffer_K.stride(2), kv_cache.Q_Buffer_K.stride(3),
        kv_cache.PageTable_K,
        kv_cache.PageTable_K.stride(0), kv_cache.PageTable_K.stride(1),
        kv_cache.KeyCache,
        kv_cache.KeyCache.stride(0), kv_cache.KeyCache.stride(1),
        kv_cache.KeyCache_metadata,
        kv_cache.KeyCache_metadata.stride(0), kv_cache.KeyCache_metadata.stride(1),
        kv_cache.KeyCache_metadata.stride(2), kv_cache.KeyCache_metadata.stride(3),
        # Scaling
        scaling,
        # Output
        attn_score,
        attn_score.stride(0), attn_score.stride(1), attn_score.stride(2),
        # Variables
        kv_cache.Sink_Count,
        kv_cache.Q_Buffer_Count_K,
        kv_cache.PageCount_K,
        # Other constants
        H_KV,
        H_Q,
        kv_cache.PAGE_SIZE,
        D,
        kv_cache.S,
        kv_cache.D_BOOSTED,
        #
        MAX_KV_GROUP
    )

    # Apply softmax to attention scores
    attn_score = nn.functional.softmax(attn_score, dim=-1, dtype=torch.float32).to(query.dtype)

    # Prepare output tensor for attention output
    # Output in shape [B, T=1, H_Q, D] to be compatible with huggingface transformers.  
    attn_output = torch.empty(
        (B, t_query, H_Q, D),
        dtype=query.dtype,
        device=query.device,
    )

    # Launch SV kernel
    grid = (B, H_KV)
    sv_kernel[grid](
        # softmax logits (Input)
        attn_score,
        attn_score.stride(0), attn_score.stride(1), attn_score.stride(2),
        # Value Cache (Input)
        kv_cache.Sink_Buffer_V,
        kv_cache.Sink_Buffer_V.stride(0), kv_cache.Sink_Buffer_V.stride(1),
        kv_cache.Sink_Buffer_V.stride(2), kv_cache.Sink_Buffer_V.stride(3),
        kv_cache.Q_Buffer_V,
        kv_cache.Q_Buffer_V.stride(0), kv_cache.Q_Buffer_V.stride(1),
        kv_cache.Q_Buffer_V.stride(2), kv_cache.Q_Buffer_V.stride(3),
        kv_cache.Local_Buffer_V,
        kv_cache.Local_Buffer_V.stride(0), kv_cache.Local_Buffer_V.stride(1),
        kv_cache.Local_Buffer_V.stride(2), kv_cache.Local_Buffer_V.stride(3),
        kv_cache.PageTable_V,
        kv_cache.PageTable_V.stride(0), kv_cache.PageTable_V.stride(1),
        kv_cache.ValueCache,
        kv_cache.ValueCache.stride(0), kv_cache.ValueCache.stride(1),
        kv_cache.ValueCache_metadata,
        kv_cache.ValueCache_metadata.stride(0), kv_cache.ValueCache_metadata.stride(1),
        kv_cache.ValueCache_metadata.stride(2), kv_cache.ValueCache_metadata.stride(3),
        # Output
        attn_output,
        attn_output.stride(0), attn_output.stride(1), attn_output.stride(2), attn_output.stride(3),
        # Variables
        kv_cache.Sink_Count,
        kv_cache.Q_Buffer_Count_V,
        kv_cache.Local_Count_V,
        kv_cache.Write_Offset_Local_V,
        kv_cache.PageCount_V,
        # Other constants
        H_KV,
        H_Q,
        kv_cache.PAGE_SIZE,
        D,
        kv_cache.S,
        #
        MAX_KV_GROUP
    )

    # do not return attention scores
    return attn_output, None