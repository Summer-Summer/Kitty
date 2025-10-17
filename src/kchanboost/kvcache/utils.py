import math
from typing import Optional, Callable
import torch
import torch.nn as nn

class KVCache_Layer:
    """
    KV Cache for a single layer.
    The pages are statically allocated in our current version.
    To Do: Multi-GPU Inference.
    """
    def __init__(self, MAX_BS: int, MAX_LEN: int, H_KV: int, D: int, D_BOOSTED: int, LOW_BIT: int, HIGH_BIT: int, PAGE_SIZE: int, S: int):
        ######################################### Quantized Pages #########################################
        assert PAGE_SIZE % 128 == 0, "PAGE_SIZE must be a multiple of 128."
        self.BITS_PER_BYTE = 8
        self.HIGH_BIT = HIGH_BIT
        self.LOW_BIT = LOW_BIT
        #
        self.MAX_BS = MAX_BS
        self.MAX_LEN = MAX_LEN
        self.H_KV = H_KV
        self.D = D
        self.D_BOOSTED = D_BOOSTED
        self.PAGE_SIZE = PAGE_SIZE
        self.S = S
        # Initialize Key Cache
        self.MAX_PAGE = math.ceil(MAX_LEN / PAGE_SIZE)
        self.D_hi = D_BOOSTED
        self.D_lo = D - D_BOOSTED
        self.bytes_per_page_K = (H_KV * PAGE_SIZE * self.D_hi * self.HIGH_BIT // self.BITS_PER_BYTE    # INT4
                                + H_KV * PAGE_SIZE * self.D_lo * self.LOW_BIT // self.BITS_PER_BYTE    # INT2    
                                + H_KV * D)                                                            # ch_idx for channel reordering
        self.KeyCache = torch.zeros(
            (MAX_BS, self.MAX_PAGE, self.bytes_per_page_K), dtype=torch.uint8, device='cuda')
        self.KeyCache_metadata = torch.zeros( # scale & zero_point
            (MAX_BS, self.MAX_PAGE, H_KV, D, 2), dtype=torch.half, device='cuda')
        # Initialize Value Cache
        self.bytes_per_page_V = H_KV * PAGE_SIZE * D * self.LOW_BIT // self.BITS_PER_BYTE              # INT2
        self.ValueCache = torch.zeros(
            (MAX_BS, self.MAX_PAGE, self.bytes_per_page_V), dtype=torch.uint8, device='cuda')
        self.ValueCache_metadata = torch.zeros(         # scale & zero_point
            (MAX_BS, self.MAX_PAGE, H_KV, PAGE_SIZE, 2), dtype=torch.half, device='cuda')
        # Initialize Page Table
        self.PageTable_K = torch.zeros(
            (MAX_BS, self.MAX_PAGE), dtype=torch.int64, device='cuda')
        self.PageTable_K_metadata = torch.zeros(
            (MAX_BS, self.MAX_PAGE), dtype=torch.int64, device='cuda')
        self.PageTable_V = torch.zeros(
            (MAX_BS, self.MAX_PAGE), dtype=torch.int64, device='cuda')
        self.PageTable_V_metadata = torch.zeros(
            (MAX_BS, self.MAX_PAGE), dtype=torch.int64, device='cuda')
        # The number of pages used in each batch
        self.PageCount_K = 0
        self.PageCount_V = 0
        # Filling the PTR to the page table
        for b in range(MAX_BS):
            for p in range(self.MAX_PAGE):
                self.PageTable_K[b, p] = self.KeyCache[b, p].data_ptr()
                self.PageTable_V[b, p] = self.ValueCache[b, p].data_ptr()
                self.PageTable_K_metadata[b, p] = self.KeyCache_metadata[b, p].data_ptr()
                self.PageTable_V_metadata[b, p] = self.ValueCache_metadata[b, p].data_ptr()
        ############################################## Sink ##############################################
        self.Sink_Buffer_K = torch.zeros(
            (MAX_BS, H_KV, S, D), dtype=torch.float16, device='cuda')
        self.Sink_Buffer_V = torch.zeros(
            (MAX_BS, H_KV, S, D), dtype=torch.float16, device='cuda')
        self.Sink_Count = 0
        ############################################ Q-Buffer ############################################
        self.Q_Buffer_K = torch.zeros(
            (MAX_BS, H_KV, PAGE_SIZE, D), dtype=torch.float16, device='cuda')
        self.Q_Buffer_V = torch.zeros(
            (MAX_BS, H_KV, PAGE_SIZE, D), dtype=torch.float16, device='cuda')
        self.Q_Buffer_Count_K = 0
        self.Q_Buffer_Count_V = 0
        ####################################### Local (Value Cache) ######################################
        self.Local_Buffer_V = torch.zeros(
            (MAX_BS, H_KV, PAGE_SIZE, D),
            dtype=torch.float16,
            device='cuda'
        )
        self.Local_Count_V = 0
        self.Write_Offset_Local_V = 0
        # Legacy for compatibility of prefills
        self.key_states: Optional[torch.Tensor] = None
        self.value_states: Optional[torch.Tensor] = None