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
    BYTES_PER_PAGE: tl.constexpr,                # 每页字节数
    PAGE_SIZE: tl.constexpr,
    MAX_PAGE: tl.constexpr,
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

    # packing
    tl.static_assert(D % 4 == 0, "D must be multiple of 4 for int2 quantization.")
    NUM_PACKED = D // 4
    # 1. 创建位移量 [0, 2, 4, 6, 0, 2, 4, 6, ...]
    shifts = (offs_d % 4) * 2
    # 2. 并行地对 [PAGE_SIZE, D] 中的所有元素进行位移
    x_shifted = (x_q & 0x3) << shifts[None, :]
    # 3. 重塑为 [PAGE_SIZE, NUM_PACKED, 4]
    x_grouped = tl.reshape(x_shifted, (PAGE_SIZE, NUM_PACKED, 4))
    # 4. 沿最后一个维度求和 (Sum == Bitwise OR，因为位不重叠)
    packed = tl.sum(x_grouped, axis=2).to(tl.uint8)     # [PAGE_SIZE, NUM_PACKED]

    # store packed data
    cache_base = cache_ptr + page_id * cache_stride_bp + pid_h * PAGE_SIZE * D * 2 // 8
    tl.store(cache_base + offs_t[:, None] * NUM_PACKED + tl.arange(0, NUM_PACKED)[None, :], packed)



@triton.jit
def quantize_pack_k_kernel(
    # Data source
    key_ptr,                                   # fp16 [B, H, T, D]
    key_stride_b, key_stride_h, key_stride_t, key_stride_d,
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
    BYTES_PER_PAGE: tl.constexpr,                # 每页字节数
    PAGE_SIZE: tl.constexpr,
    MAX_PAGE: tl.constexpr,
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

    # packing
    tl.static_assert(D % 4 == 0, "D must be multiple of 4 for int2 quantization.")
    NUM_PACKED = D // 4
    # 1. 创建位移量 [0, 2, 4, 6, 0, 2, 4, 6, ...]
    shifts = (offs_d % 4) * 2
    # 2. 并行地对 [PAGE_SIZE, D] 中的所有元素进行位移
    x_shifted = (x_q & 0x3) << shifts[None, :]
    # 3. 重塑为 [PAGE_SIZE, NUM_PACKED, 4]
    x_grouped = tl.reshape(x_shifted, (PAGE_SIZE, NUM_PACKED, 4))
    # 4. 沿最后一个维度求和 (Sum == Bitwise OR，因为位不重叠)
    packed = tl.sum(x_grouped, axis=2).to(tl.uint8)     # [PAGE_SIZE, NUM_PACKED]

    # store packed data
    cache_base = cache_ptr + page_id * cache_stride_bp + pid_h * PAGE_SIZE * D * 2 // 8
    tl.store(cache_base + offs_t[:, None] * NUM_PACKED + tl.arange(0, NUM_PACKED)[None, :], packed)





def quantize_pack_k(
    # data source
    key_states: torch.Tensor,
    key_states_t_offset: int,
    key_states_page_count: int,
    pages_size: int,
    # data destination
    page_table_k: torch.Tensor,
    page_table_k_metadata: torch.Tensor,
    page_count_k: int,
):
    pass



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




