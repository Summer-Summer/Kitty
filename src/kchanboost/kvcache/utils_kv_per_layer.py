# MIT License
# Copyright (c) 2025 Haojun Xia
# See the LICENSE file in the project root for more information.

# src/kchanboost/kvcache/utils_kv_per_layer.py

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
        self.bytes_per_page_K = (H_KV * self.D         * PAGE_SIZE * self.LOW_BIT // self.BITS_PER_BYTE     # Low INT2
                               + H_KV * self.D_BOOSTED * PAGE_SIZE * self.LOW_BIT // self.BITS_PER_BYTE     # High INT2
                               + H_KV * D)                                                                  # ch_idx for channel reordering
        self.KeyCache = torch.zeros(
            (MAX_BS * self.MAX_PAGE, self.bytes_per_page_K), dtype=torch.uint8, device='cuda')
        self.KeyCache_metadata = torch.zeros( # scale & zero_point
            (MAX_BS * self.MAX_PAGE, H_KV, D, 2), dtype=torch.half, device='cuda')
        # Initialize Value Cache
        self.bytes_per_page_V = H_KV * PAGE_SIZE * D * self.LOW_BIT // self.BITS_PER_BYTE              # INT2
        self.ValueCache = torch.zeros(
            (MAX_BS * self.MAX_PAGE, self.bytes_per_page_V), dtype=torch.uint8, device='cuda')
        self.ValueCache_metadata = torch.zeros(         # scale & zero_point
            (MAX_BS * self.MAX_PAGE, H_KV, PAGE_SIZE, 2), dtype=torch.half, device='cuda')
        # Initialize Page Table
        self.PageTable_K = torch.zeros(
            (MAX_BS, self.MAX_PAGE), dtype=torch.int64, device='cuda')
        self.PageTable_V = torch.zeros(
            (MAX_BS, self.MAX_PAGE), dtype=torch.int64, device='cuda')
        # The number of pages used in each batch
        self.PageCount_K = 0
        self.PageCount_V = 0
        # Filling the page ID to the page table
        for b in range(MAX_BS):
            for p in range(self.MAX_PAGE):
                self.PageTable_K[b, p] = p + b * self.MAX_PAGE
                self.PageTable_V[b, p] = p + b * self.MAX_PAGE
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
        self.Write_Offset_Local_V = 0       # To track the write offset in the local buffer, circular buffer.
        # Legacy for compatibility of prefills
        self.key_states: Optional[torch.Tensor] = None
        self.value_states: Optional[torch.Tensor] = None
    
    def get_total_length(self) -> int:
        """Returns the total length of the cached states."""
        total_length = (self.PageCount_K * self.PAGE_SIZE
                        + self.Sink_Count
                        + self.Q_Buffer_Count_K)
        seqlen_V = (self.PageCount_V * self.PAGE_SIZE
                    + self.Sink_Count
                    + self.Q_Buffer_Count_V
                    + self.Local_Count_V)
        assert total_length == seqlen_V, f"seqlen_K: {total_length}, seqlen_V: {seqlen_V}"
        return total_length