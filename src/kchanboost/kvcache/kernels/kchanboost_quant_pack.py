import torch
import triton
import triton.language as tl


@triton.jit
def quantize_pack_v_kernel(
    # Data source
    value_ptr,                                   # fp16 [B, H, T, D]
    value_stride_b, value_stride_h, value_stride_t, value_stride_d,
    token_offset: int,                           # 从哪个token开始量化, excluding sink tokens for prefill
    # Data destination
    cache_ptr,                                   # uint8 [B*MAX_PAGE, bytes_per_page_V]
    cache_stride_bp, cache_stride_last,
    cache_meta_ptr,                              # fp16 [B*MAX_PAGE, H, PAGE_SIZE, 2]
    cache_meta_stride_bp, cache_meta_stride_h, cache_meta_stride_t, cache_meta_stride_last,
    #
    page_table_ptr,                              # int64 [B, MAX_PAGE]
    page_table_stride_b, page_table_stride_last,
    page_count,                              # int
    #
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    """
    The tokens processed could be multiple pages (prefill) or single page (decode).
    Each Triton program handles 1 page of (PAGE_SIZE, D).
    """
    pid_b = tl.program_id(0)                     # batch
    pid_page = tl.program_id(1)                  # page index (within batch)
    pid_h = tl.program_id(2)                     # head

    # Reading the original fp16 value states, [PAGE_SIZE, D]
    src_ptr = value_ptr + pid_b * value_stride_b + pid_h * value_stride_h + (pid_page * PAGE_SIZE + token_offset) * value_stride_t
    offs_t = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, D)
    x = tl.load(src_ptr + offs_t[:, None] * value_stride_t + offs_d[None, :] * value_stride_d)

    # Computing the quantization parameters
    # Asynmetric group-wise quantization along D (per token), storing the fp16 scale & x_min
    x_max = tl.max(x, axis=1)
    x_min = tl.min(x, axis=1)                           # [PAGE_SIZE]
    scale = (x_max - x_min) / 3.0  # for int2
    # avoid divide by zero
    scale = tl.where(scale < 1e-6, 1e-6, scale)         # [PAGE_SIZE]

    # page id
    page_id = tl.load(page_table_ptr + pid_b * page_table_stride_b + (page_count+ pid_page))
    # store metadata
    meta_base = cache_meta_ptr + page_id * cache_meta_stride_bp + pid_h * cache_meta_stride_h
    tl.store(meta_base + offs_t * cache_meta_stride_t + 0, scale.to(tl.float16))
    tl.store(meta_base + offs_t * cache_meta_stride_t + 1, x_min.to(tl.float16))


    # quantize to int2
    x_f = (x - x_min[:, None]) / scale[:, None]
    x_q = tl.floor(x_f + 0.5)
    x_q = tl.clamp(x_q, 0.0, 3.0)
    x_q = x_q.to(tl.uint8)                          # [PAGE_SIZE, D]

    # packing, the D channels of each token is continuously packed.
    tl.static_assert(D % 4 == 0, "D must be multiple of 4 for int2 quantization.")
    NUM_PACKED = D // 4
    # 1. 创建位移量 [0, 2, 4, 6, 0, 2, 4, 6, ...]
    shifts = (offs_d % 4) * 2
    # 2. 并行地对 [PAGE_SIZE, D] 中的所有元素进行位移
    x_shifted = (x_q & 0x3) << shifts[None, :]
    # 3. 重塑为 [PAGE_SIZE, NUM_PACKED, 4]
    x_grouped = tl.reshape(x_shifted, (PAGE_SIZE, NUM_PACKED, 4))
    # 4. 沿最后一个维度求和 (Sum == Bitwise OR，因为位不重叠)
    packed = tl.sum(x_grouped, axis=2).to(tl.uint8)     # [PAGE_SIZE, D // 4]

    # store packed data
    cache_base = cache_ptr + page_id * cache_stride_bp + pid_h * PAGE_SIZE * D * 2 // 8
    tl.store(cache_base + offs_t[:, None] * NUM_PACKED + tl.arange(0, NUM_PACKED)[None, :], packed)



@triton.jit
def quantize_pack_k_kernel(
    # Data source
    key_ptr,                                   # fp16 [B, H, T, D]
    key_stride_b, key_stride_h, key_stride_t, key_stride_d,
    dense_sparse_idx_ptr,                      # uint8 [B, H, page_count, D]
    idx_stride_b, idx_stride_h, idx_stride_page, idx_stride_d,
    # Data destination
    cache_ptr,                                   # uint8 [B*MAX_PAGE, bytes_per_page_K]
    cache_stride_bp, cache_stride_last,
    cache_meta_ptr,                              # fp16 [B*MAX_PAGE, H, D, 2]
    cache_meta_stride_bp, cache_meta_stride_h, cache_meta_stride_d, cache_meta_stride_last,
    #
    page_table_ptr,                              # int64 [B, MAX_PAGE]
    page_table_stride_b, page_table_stride_last,
    page_count,                              # int
    #
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    #
    D_BOOST: tl.constexpr,              # number of channels that are boosted.
    #
    BOOST_BLOCK_OFFSET: tl.constexpr,
    BOOST_IDX_OFFSET: tl.constexpr,
):
    """
    The tokens processed could be multiple pages (prefill) or single page (decode).
    Each Triton program handles 1 page of (PAGE_SIZE, D).
    """
    pid_b = tl.program_id(0)                     # batch
    pid_page = tl.program_id(1)                  # page index (within batch)
    pid_h = tl.program_id(2)                     # head
    
    #
    offs_t = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, D)

    # Reading the original fp16 value states, [PAGE_SIZE, D]
    src_ptr = key_ptr + pid_b * key_stride_b + pid_h * key_stride_h + (pid_page * PAGE_SIZE) * key_stride_t
    x = tl.load(src_ptr + offs_t[:, None] * key_stride_t + offs_d[None, :] * key_stride_d)

    # reading the dense to sparse channel index
    dense_sparse_idx = tl.load(
        dense_sparse_idx_ptr + pid_b * idx_stride_b + pid_h * idx_stride_h + pid_page * idx_stride_page + offs_d[None, :] * idx_stride_d)
    boost_mask = dense_sparse_idx < D_BOOST   # uint8 to bool

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
    page_id = tl.load(page_table_ptr + pid_b * page_table_stride_b + (page_count+ pid_page))
    # store metadata
    meta_base = cache_meta_ptr + page_id * cache_meta_stride_bp + pid_h * cache_meta_stride_h
    tl.store(meta_base + offs_d * cache_meta_stride_d + 0, scale.to(tl.float16))
    tl.store(meta_base + offs_d * cache_meta_stride_d + 1, x_min.to(tl.float16))


    # quantize to int4 & int2, 
    x_f = (x - x_min[None, :]) / scale[None, :]
    x_q = tl.floor(x_f + 0.5)
    x_q = tl.where(
        boost_mask[None, :],
        tl.clamp(x_q, 0.0, 15.0),
        tl.clamp(x_q, 0.0, 3.0))
    x_q = x_q.to(tl.uint8) # [PAGE_SIZE, D]

    # packing INT2 and the low 2bits of INT4, each channel of PAGE_SIZE tokens are continuously packed.
    tl.static_assert(PAGE_SIZE % 4 == 0, "PAGE_SIZE must be multiple of 4 for int2 packing.")
    NUM_PACKED = PAGE_SIZE // 4
    # 1. 创建位移量 [0, 2, 4, 6, 0, 2, 4, 6, ...]
    shifts = (offs_t % 4) * 2
    # 2. 并行地对 [PAGE_SIZE, D] 中的所有元素进行位移
    x_shifted = (x_q & 0x3) << shifts[:, None]
    # 3. 重塑为 [NUM_PACKED, 4, D]
    x_grouped = tl.reshape(x_shifted, (NUM_PACKED, 4, D))
    # 4. 沿最后一个维度求和 (Sum == Bitwise OR，因为位不重叠)
    packed = tl.sum(x_grouped, axis=1).to(tl.uint8)     # [PAGE_SIZE // 4, D]

    # packing the high 2bits of INT4
    x_q = x_q >> 2                                      # keep only the high 2 bits
    x_shifted2 = (x_q & 0x3) << shifts[:, None]
    x_grouped2 = tl.reshape(x_shifted2, (NUM_PACKED, 4, D))
    packed2 = tl.sum(x_grouped2, axis=1).to(tl.uint8)     # [PAGE_SIZE // 4, D]

    # store packed data (INT2)
    cache_base = cache_ptr + page_id * cache_stride_bp + pid_h * PAGE_SIZE * D * 2 // 8
    tl.store(cache_base + offs_d[None, :] * NUM_PACKED + tl.arange(0, NUM_PACKED)[:, None], packed)

    # store packed data (high 2bits of INT4)
    cache_base2 = cache_base + BOOST_BLOCK_OFFSET
    tl.store(cache_base2 + dense_sparse_idx * NUM_PACKED + tl.arange(0, NUM_PACKED)[:, None], packed2, mask=boost_mask[None, :])

    # store dense to sparse index
    idx_base = cache_base + BOOST_IDX_OFFSET + pid_h * D
    tl.store(idx_base + offs_d, dense_sparse_idx)




def quantize_pack_k(
    # data source
    key_states: torch.Tensor,           # [B, H_KV, T, D]
    key_states_t_offset: int,
    key_states_page_count: int,
    page_size: int,
    # data destination
    page_table_k: torch.Tensor,
    page_table_k_metadata: torch.Tensor,
    page_count_k: int,
    #
    boost_ratio: float,
):
    """
    Quantize and pack the key states into the paged key cache.
    """
    assert key_states.dim() == 4
    B, H_KV, _, D = key_states.shape
    assert 0. <= boost_ratio <= 1.0, f"boost_ratio must be in [0, 1], got {boost_ratio}"
    k_chan = int(D * boost_ratio + 1e-6)  # number of channels to promote
    assert k_chan >= 0 and k_chan <= D, f"Invalid boost_ratio resulting in k_chan={k_chan} for D={D}"

    # Slicing the key states to pages
    key_to_quantize = key_states[:, :, key_states_t_offset : key_states_t_offset + page_size * key_states_page_count, :].contiguous()
    key_to_quantize = key_to_quantize.view(B, H_KV, key_states_page_count, page_size, D)    # (B, H_KV, page_count, PAGE_SIZE, D)

    ##################################################################################################################
    # Computing the channel score, the formular of channel score can be modified here. We use maganitude here.
    score = key_to_quantize.abs().mean(dim=-2)       # (B, H_KV, PAGE_COUNT, PAGE_SIZE, D) → (B, H_KV, PAGE_COUNT, D)
    ##################################################################################################################

    # Generating the boost mask
    _, top_idx = score.topk(k_chan, dim=-1)     # (B, H_KV, PAGE_COUNT, k_chan)

    # Note: idx=k_chan means this channel is not boosted.
    dense_to_sparse_idx = torch.full((B, H_KV, key_states_page_count, D), k_chan, dtype=torch.uint8, device=key_states.device)
    dense_to_sparse_idx.scatter_(-1, top_idx, torch.arange(k_chan, device=key_states.device, dtype=torch.uint8).view(1, 1, 1, -1))

    # Launch Triton kernel
    


def quantize_pack_v(
    # data source
    value_states: torch.Tensor,                 # (B, H_KV, T, D)
    value_states_t_offset: int,                 # quantize from which token (t offset)
    value_states_page_count: int,               # how many pages to quantize
    pages_size: int,                            # page size
    # data destination
    value_cache: torch.Tensor,                  # (MAX_BS * MAX_PAGE, bytes_per_page_V), uint8
    value_cache_metadata: torch.Tensor,         # (MAX_BS * MAX_PAGE, H_KV, PAGE_SIZE, 2), float16
    page_table_v: torch.Tensor,                 # (MAX_BS, MAX_PAGE), int64
    page_count_v: int,                          # current number of pages used in the batch
    #
    low_bit: int,
    bytes_per_page_V: int,
):
    """
    Quantize and pack the value states into the paged value cache.
    """
    #
    BITS_PER_BYTES = 8 
    #
    BS, H_KV, T, D = value_states.shape
    assert T == pages_size * value_states_page_count, "T must equal to page_size * page_count."
    assert low_bit == 2, "bit_width must be 2."

    grid = (BS, value_states_page_count, H_KV)
    quantize_pack_v_kernel[grid](
        value_states,
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        #
        value_cache,
        value_cache_metadata,
        page_table_v,
        page_count_v,
        bytes_per_page_V,
        pages_size,
        D,
        low_bit,
        num_warps=4,
        num_stages=2,
    )




