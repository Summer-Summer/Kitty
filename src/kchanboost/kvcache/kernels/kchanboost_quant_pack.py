# src/kchanboost/kvcache/kernels/kchanboost_quant_pack.py

import torch
import triton
import triton.language as tl



@triton.jit
def quantize_pack_k_kernel(
    ############################ Data source ############################
    key_ptr,                                   # fp16 [B, H, T, D]
    key_stride_b, key_stride_h, key_stride_t, key_stride_d,
    #
    dense_sparse_idx_ptr,                      # uint8 [B, H, page_count, D]
    idx_stride_b, idx_stride_h, idx_stride_page, idx_stride_d,
    ########################## Data destination ##########################
    # Page Table
    page_table_ptr,                              # int64 [B, MAX_PAGE]
    page_table_stride_b, page_table_stride_last,
    page_offset,                              # int
    # KV Cache
    cache_ptr,                                   # uint8 [B*MAX_PAGE, bytes_per_page_K]
    cache_stride_bp, cache_stride_last,
    cache_meta_ptr,                              # fp16 [B*MAX_PAGE, H, D, 2]
    cache_meta_stride_bp, cache_meta_stride_h, cache_meta_stride_d, cache_meta_stride_last,
    # Constants
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    D_BOOST: tl.constexpr,              # number of channels that are boosted.
):
    """
    The tokens processed could be multiple pages (prefill) or single page (decode).
    Each Triton program handles 1 page of (PAGE_SIZE, D).
    """

    # Constants
    K_STRIDE_D:     tl.constexpr = PAGE_SIZE * 2 // 8
    K_STRIDE_H_KV:  tl.constexpr = K_STRIDE_D * (D + D_BOOST) + D   # including channel idx
    K_OFF_HI:       tl.constexpr = K_STRIDE_D * D
    K_OFF_IDX:      tl.constexpr = K_STRIDE_D * (D + D_BOOST)

    pid_b = tl.program_id(0)                     # batch
    pid_page = tl.program_id(1)                  # page index (within batch)
    pid_h = tl.program_id(2)                     # head
    
    #
    offs_t = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, D)
    offs_packed = tl.arange(0, PAGE_SIZE // 4)

    # Reading the original fp16 value states, [PAGE_SIZE, D]
    src_ptr = key_ptr + pid_b * key_stride_b + pid_h * key_stride_h
    x = tl.load(
        src_ptr 
        + (offs_t[:, None] + pid_page * PAGE_SIZE) * key_stride_t
        + offs_d[None, :] * key_stride_d
    )   # [PAGE_SIZE, D]

    # reading the dense to sparse channel index
    dense_sparse_idx = tl.load(
        dense_sparse_idx_ptr
        + pid_b * idx_stride_b
        + pid_h * idx_stride_h
        + pid_page * idx_stride_page
        + offs_d * idx_stride_d)   # [D]
    boost_mask = dense_sparse_idx < D_BOOST   # uint8 to bool, [D], if true → boost to int4

    # Computing the quantization parameters
    # Asynmetric group-wise quantization along D (per token), storing the fp16 scale & x_min
    x_max = tl.max(x, axis=0)
    x_min = tl.min(x, axis=0)                           # [D]
    scale = tl.where(
        boost_mask,
        (x_max - x_min) / 15.0, # for int4
        (x_max - x_min) / 3.0)  # for int2
    # avoid divide by zero
    scale = tl.where(scale < 1e-6, 1e-6, scale)         # [D]

    # page id
    page_id = tl.load(
        page_table_ptr
        + pid_b * page_table_stride_b
        + (page_offset + pid_page) * page_table_stride_last
    )

    # store metadata
    meta_base = cache_meta_ptr + page_id * cache_meta_stride_bp + pid_h * cache_meta_stride_h
    tl.store(
        meta_base
        + offs_d * cache_meta_stride_d
        + 0 * cache_meta_stride_last,
        scale.to(tl.float16))
    tl.store(
        meta_base
        + offs_d * cache_meta_stride_d
        + 1 * cache_meta_stride_last,
        x_min.to(tl.float16))


    # quantize to int4 & int2, 
    x_f = (x - x_min[None, :]) / scale[None, :]
    x_q = tl.floor(x_f + 0.5)
    x_q = tl.where(
        boost_mask[None, :],
        tl.clamp(x_q, 0.0, 15.0),
        tl.clamp(x_q, 0.0, 3.0))
    x_q = x_q.to(tl.uint8) # [PAGE_SIZE, D]

    # packing Ithe low 2bits of INT2 and INT4, the elements of each channel are continuously packed.
    tl.static_assert(PAGE_SIZE % 4 == 0, "PAGE_SIZE must be multiple of 4 for int2 packing.")
    shifts = (offs_t % 4) * 2                                   # 0. shift offsets: [0, 2, 4, 6, 0, 2, 4, 6, ...]     
    x_shifted = (x_q & 0x3) << shifts[:, None]                  # 1. shift [PAGE_SIZE, D] elements in parallel
    x_grouped = tl.reshape(x_shifted, (PAGE_SIZE // 4, 4, D))   # 2. reshape to [PAGE_SIZE // 4, 4, D]
    packed = tl.sum(x_grouped, axis=1).to(tl.uint8)             # 3. Sum == Bitwise OR, because the bits do not overlap

    # packing the high 2bits of INT4
    x_q = (x_q >> 2) & 0x3                                      # keep only the high 2 bits
    x_shifted2 = x_q << shifts[:, None]                         # 1. shift [PAGE_SIZE, D] elements in parallel
    x_grouped2 = tl.reshape(x_shifted2, (PAGE_SIZE // 4, 4, D)) # 2. reshape to [PAGE_SIZE // 4, 4, D]
    packed2 = tl.sum(x_grouped2, axis=1).to(tl.uint8)           # 3. Sum == Bitwise OR, because the bits do not overlap

    # store packed data (INT2)
    cache_base = cache_ptr + page_id * cache_stride_bp + pid_h * K_STRIDE_H_KV
    tl.store(
        cache_base
        + offs_packed[:, None]
        + offs_d[None, :] * K_STRIDE_D,
        packed
    )

    # store packed data (high 2bits of INT4)
    mask = boost_mask[None, :] & (offs_packed[:, None] >= 0)
    tl.store(
        cache_base + K_OFF_HI
        + offs_packed[:, None]
        + dense_sparse_idx[None, :] * K_STRIDE_D,
        packed2,
        mask=mask
    )

    # store dense to sparse index
    tl.store(
        cache_base
        + K_OFF_IDX
        + offs_d,
        dense_sparse_idx
    )


@triton.jit
def quantize_pack_v_kernel(
    ############################ Data source ############################
    value_ptr,                                   # fp16 [B, H, T, D]
    value_stride_b, value_stride_h, value_stride_t, value_stride_d,
    ########################## Data destination ##########################
    # Page Table
    page_table_ptr,                              # int64 [B, MAX_PAGE]
    page_table_stride_b, page_table_stride_last,
    page_offset,                                   # int
    # KV Cache
    cache_ptr,                                   # uint8 [B*MAX_PAGE, bytes_per_page_V]
    cache_stride_bp, cache_stride_last,
    cache_meta_ptr,                              # fp16 [B*MAX_PAGE, H, PAGE_SIZE, 2]
    cache_meta_stride_bp, cache_meta_stride_h, cache_meta_stride_t, cache_meta_stride_last,
    # Constants
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    """
    The tokens processed could be multiple pages (prefill) or single page (decode).
    Each Triton program handles 1 page of (PAGE_SIZE, D).
    """

    # Constant Strides and Offsets for V cache dequantization
    V_STRIDE_T:     tl.constexpr = D * 2 // 8
    V_STRIDE_H_KV:  tl.constexpr = V_STRIDE_T * PAGE_SIZE

    pid_b = tl.program_id(0)                     # batch
    pid_page = tl.program_id(1)                  # page index (within batch)
    pid_h = tl.program_id(2)                     # kv head

    # tl.arange()
    offs_t = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, D)

    # Reading the original fp16 value states
    src_ptr = value_ptr + pid_b * value_stride_b + pid_h * value_stride_h + (pid_page * PAGE_SIZE) * value_stride_t
    x = tl.load(
        src_ptr
        + offs_d[:, None] * value_stride_d
        + offs_t[None, :] * value_stride_t
    )   # [D, PAGE_SIZE]

    # Computing the quantization parameters
    # Asynmetric group-wise quantization along D (per token), storing the fp16 scale & x_min
    x_max = tl.max(x, axis=0)
    x_min = tl.min(x, axis=0)                           # [PAGE_SIZE]
    scale = (x_max - x_min) / 3.0                       # 3.0 for int2
    scale = tl.where(scale < 1e-6, 1e-6, scale)         # avoid divide by zero

    # page id
    page_id = tl.load(
        page_table_ptr
        + pid_b * page_table_stride_b
        + (page_offset + pid_page) * page_table_stride_last
    )
    # store metadata
    meta_base = cache_meta_ptr + page_id * cache_meta_stride_bp + pid_h * cache_meta_stride_h
    tl.store(
        meta_base
        + offs_t * cache_meta_stride_t
        + 0 * cache_meta_stride_last,
        scale.to(tl.float16)
    )
    tl.store(
        meta_base
        + offs_t * cache_meta_stride_t
        + 1 * cache_meta_stride_last,
        x_min.to(tl.float16))

    # quantize to int2
    x_f = (x - x_min[None, :]) / scale[None, :]
    x_q = tl.floor(x_f + 0.5)
    x_q = tl.clamp(x_q, 0.0, 3.0)
    x_q = x_q.to(tl.uint8)                          # [D, PAGE_SIZE]

    # packing, the D channels of each token is continuously packed.
    tl.static_assert(D % 4 == 0, "D must be multiple of 4 for int2 quantization.")
    shifts = (offs_d % 4) * 2                                   # 1. shift offsets: [0, 2, 4, 6, 0, 2, 4, 6, ...]
    x_shifted = (x_q & 0x3) << shifts[:, None]                  # 2. parallel shifting
    x_grouped = tl.reshape(x_shifted, (D//4, 4, PAGE_SIZE))     # 3. reshape to [D//4, 4, PAGE_SIZE]
    packed = tl.sum(x_grouped, axis=1).to(tl.uint8)             # 4. Sum == Bitwise OR，because bits do not overlap.

    # store packed data
    # packed: [D // 4, PAGE_SIZE], each element contains 4 quantized values and these 4 values are from continuous 4 channels of a token.
    # we need to transpose it to [PAGE_SIZE, D // 4] so that each token's data are continuous in memory.
    # However, we choose to swap the strides of the destination tensor instead of transposing the data for efficiency.
    cache_base = cache_ptr + page_id * cache_stride_bp + pid_h * V_STRIDE_H_KV
    tl.store(
        cache_base
        + offs_t[None, :] * V_STRIDE_T
        + tl.arange(0, D//4)[:, None],
        packed
    )



def quantize_pack_k(
    # data source
    key_states: torch.Tensor,           # (B, H_KV, T, D), the complete prefill (sink+...) or the Q-Buffer (1 page)
    key_states_t_offset: int,           # quantize from which token (t offset)
    key_states_page_count: int,         # how many pages to quantize
    # data destination
    page_table_k: torch.Tensor,
    page_offset_k: int,
    cache: torch.Tensor,                # [MAX_BS * MAX_PAGE, bytes_per_page_K], uint8
    cache_metadata: torch.Tensor,       # [MAX_BS * MAX_PAGE, H_K, D, 2], float16
    #
    page_size: int,
    d_boost: int,                       # number of channels that are boosted.
):
    """
    Quantize and pack the key states into the paged key cache.
    """
    assert key_states.dim() == 4
    B, H_KV, _, D = key_states.shape
    assert 0 <= d_boost <= D, f"d_boost must be in [0, {D}], got {d_boost}"

    # Slicing the key states to pages
    key_to_quantize = key_states[:, :, key_states_t_offset : key_states_t_offset + page_size * key_states_page_count, :].contiguous() # (B, H_KV, page_count*PAGE_SIZE, D)
    key_to_quantize = key_to_quantize.view(B, H_KV, key_states_page_count, page_size, D)    # (B, H_KV, page_count, PAGE_SIZE, D)

    ##################################################################################################################
    # Computing the channel score, the formular of channel score can be modified here. We use maganitude here.
    score = key_to_quantize.abs().mean(dim=-2)       # (B, H_KV, PAGE_COUNT, PAGE_SIZE, D) → (B, H_KV, PAGE_COUNT, D)
    ##################################################################################################################

    # Generating the boost mask
    _, top_idx = score.topk(d_boost, dim=-1)     # (B, H_KV, PAGE_COUNT, d_boost)

    # Note: idx=d_boost means this channel is not boosted.
    dense_to_sparse_idx = torch.full((B, H_KV, key_states_page_count, D), d_boost, dtype=torch.uint8, device=key_states.device)
    dense_to_sparse_idx.scatter_(-1, top_idx, torch.arange(d_boost, device=key_states.device, dtype=torch.uint8).view(1, 1, 1, -1).expand_as(top_idx))

    # Launch Triton kernel
    grid = (B, key_states_page_count, H_KV)
    quantize_pack_k_kernel[grid](
        # Data source
        key_to_quantize,
        key_to_quantize.stride(0), key_to_quantize.stride(1), key_to_quantize.stride(3), key_to_quantize.stride(4),
        # dense to sparse index
        dense_to_sparse_idx,
        dense_to_sparse_idx.stride(0), dense_to_sparse_idx.stride(1), dense_to_sparse_idx.stride(2), dense_to_sparse_idx.stride(3),
        #
        page_table_k,
        page_table_k.stride(0), page_table_k.stride(1),
        page_offset_k,
        # KV Cache
        cache,
        cache.stride(0), cache.stride(1),
        cache_metadata,
        cache_metadata.stride(0), cache_metadata.stride(1), cache_metadata.stride(2), cache_metadata.stride(3),
        # Constants
        page_size,
        D,
        d_boost,
    )


def quantize_pack_v(
    # data source
    value_states: torch.Tensor,                 # (B, H_KV, T, D), the complete prefill (sink+...) or the Q-Buffer (1 page)
    value_states_t_offset: int,                 # quantize from which token (t offset)
    value_states_page_count: int,               # how many pages to quantize
    # data destination
    page_table_v: torch.Tensor,                 # (MAX_BS, MAX_PAGE), int64
    page_offset_v: int,                         # current number of pages already used
    cache: torch.Tensor,                        # (MAX_BS * MAX_PAGE, bytes_per_page_V), uint8
    cache_metadata: torch.Tensor,               # (MAX_BS * MAX_PAGE, H_KV, PAGE_SIZE, 2), float16
    #
    page_size: int,                            # page size
):
    """
    Quantize and pack the value states into the paged value cache.
    """
    assert value_states.dim() == 4
    B, H_KV, _, D = value_states.shape

    # Slicing the value states to pages
    value_to_quantize = value_states[:, :, value_states_t_offset : value_states_t_offset + page_size * value_states_page_count, :].contiguous() # (B, H_KV, page_count*PAGE_SIZE, D)
    value_to_quantize = value_to_quantize.view(B, H_KV, value_states_page_count, page_size, D)    # (B, H_KV, page_count, PAGE_SIZE, D) 

    # Launch Triton kernel
    grid = (B, value_states_page_count, H_KV)
    quantize_pack_v_kernel[grid](
        # Data source
        value_to_quantize,
        value_to_quantize.stride(0), value_to_quantize.stride(1), value_to_quantize.stride(3), value_to_quantize.stride(4),
        #
        page_table_v,
        page_table_v.stride(0), page_table_v.stride(1),
        page_offset_v,
        # KV Cache
        cache,
        cache.stride(0), cache.stride(1),
        cache_metadata,
        cache_metadata.stride(0), cache_metadata.stride(1), cache_metadata.stride(2), cache_metadata.stride(3),
        # Constants
        page_size,
        D,
    )




