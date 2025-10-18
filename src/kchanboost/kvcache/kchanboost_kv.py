# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py.
# Author: Haojun Xia (xhjustc@gmail.com)

from typing import Optional, Any
from dataclasses import dataclass
import argparse
import math

import torch
from transformers.cache_utils import CacheConfig, Cache
from transformers.configuration_utils import PretrainedConfig

#
from .kernels.kchanboost_quant_pack import quantize_pack_k, quantize_pack_v
from .utils import KVCache_Layer
        

class KChanBoostCache(Cache):
    """
    Class for KChanBoost KV caches.
    """
    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_length: int,
    ) -> None:
        super().__init__()
        self.kv_cache: list[KVCache_Layer] = []
        # Reading the model configurations
        self.num_hidden_layers = config.num_hidden_layers
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        ######################## KChanBoost Specific Configurations ########################
        self.page_size = 128                    # PAGE_SIZE=128
        self.d_boosted = self.head_dim // 4     # 25% channels are boosted to INT4
        self.sink_length = 32                   # SINK_LENGTH=32
        self.low_bit = 2                        # LOW_BIT=2
        self.high_bit = 4                       # HIGH_BIT=4
        ###################################################################################
        # Initialize KV Cache for each layer
        for _ in range(self.num_hidden_layers):
            self.kv_cache.append(KVCache_Layer(
                MAX_BS = max_batch_size,
                MAX_LEN = max_length,
                H_KV = self.num_key_value_heads,
                D = self.head_dim,
                D_BOOSTED = self.d_boosted,
                LOW_BIT = self.low_bit,
                HIGH_BIT = self.high_bit,
                PAGE_SIZE = self.page_size,
                S = self.sink_length
            ))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            Bool: True if in Prefill phase, False if in Decode phase.
        """

        kvcache = self.kv_cache[layer_idx]
        Is_Prefill = (kvcache.Sink_Count == 0)
        # Prefill
        if Is_Prefill:
            # Update the KV Cache
            kvcache.key_states = key_states.contiguous()
            kvcache.value_states = value_states.contiguous()
            return True
        # Decode
        assert key_states.shape[-2] == 1 and value_states.shape[-2] == 1, \
            "Decode: key_states and value_states should have sequence length of 1."
        #  
        # Update the KV Cache
        with torch.no_grad():
            if kvcache.Sink_Count < kvcache.S:
                idx = kvcache.Sink_Count
                kvcache.Sink_Buffer_K[:,:,idx,:] = key_states[:, :, 0, :].contiguous()
                kvcache.Sink_Buffer_V[:,:,idx,:] = value_states[:, :, 0, :].contiguous()
                kvcache.Sink_Count += 1
            else:
                # Key Cache with Q-Buffer
                assert kvcache.Q_Buffer_Count_K < kvcache.Q_Buffer_K.size(2)
                kvcache.Q_Buffer_K[:, :, kvcache.Q_Buffer_Count_K, :] = key_states[:, :, 0, :].contiguous()
                kvcache.Q_Buffer_Count_K += 1
                # Value Cache with Local Buffer + Q-Buffer
                assert kvcache.Local_Count_V <= kvcache.PAGE_SIZE 
                if kvcache.Local_Count_V < kvcache.PAGE_SIZE:
                    kvcache.Local_Buffer_V[:,:,kvcache.Local_Count_V,:] = value_states[:, :, 0, :].contiguous()
                    kvcache.Local_Count_V += 1
                else:   # Local Buffer is full, Move the evicted token to Q-Buffer
                    assert kvcache.Q_Buffer_Count_V < kvcache.Q_Buffer_V.size(2)
                    kvcache.Q_Buffer_V[:, :, kvcache.Q_Buffer_Count_V, :] = kvcache.Local_Buffer_V[:, :, kvcache.Write_Offset_Local_V, :].contiguous()
                    kvcache.Q_Buffer_Count_V += 1
                    # Write the new token to Local Buffer & update the write offset
                    kvcache.Local_Buffer_V[:,:,kvcache.Write_Offset_Local_V,:] = value_states[:, :, 0, :].contiguous()
                    kvcache.Write_Offset_Local_V = (kvcache.Write_Offset_Local_V + 1) % kvcache.PAGE_SIZE
        return False

    def quantize_prefill(self, layer_idx: int = 0) -> None:
        kvcache = self.kv_cache[layer_idx]
        #
        if kvcache.key_states is None or kvcache.value_states is None:
            raise ValueError("No key_states or value_states to quantize. Please call update() first.")
        len_prefill = kvcache.key_states.shape[-2]
        len_sink = min(kvcache.S, len_prefill)
        assert len_sink >= 0, "Sink length must be greater than or equal to 0."
        # K Cache
        len_qbuf_k = max(0, len_prefill - len_sink) % kvcache.PAGE_SIZE
        pages_k = (len_prefill - len_sink - len_qbuf_k) // kvcache.PAGE_SIZE
        kvcache.Sink_Buffer_K[:,:,:len_sink,:].copy_(kvcache.key_states[:, :, :len_sink, :].contiguous())
        kvcache.Sink_Count = len_sink
        if len_qbuf_k > 0:
            kvcache.Q_Buffer_K[:,:,:len_qbuf_k,:].copy_(kvcache.key_states[:, :, -len_qbuf_k:, :].contiguous())
            kvcache.Q_Buffer_Count_K = len_qbuf_k
        if pages_k > 0:
            quantize_pack_k(
                # data source
                key_states = kvcache.key_states,
                key_states_t_offset = len_sink,
                key_states_page_count = pages_k,
                pages_size = kvcache.PAGE_SIZE,
                # data destination
                page_table_k = kvcache.PageTable_K,
                page_table_k_metadata = kvcache.PageTable_K_metadata,
                page_count_k = kvcache.PageCount_K,
            )
        # V Cache
        len_local_v = max(0, min(kvcache.PAGE_SIZE, len_prefill - len_sink))
        len_qbuf_v = max(0, len_prefill - len_sink - len_local_v) % kvcache.PAGE_SIZE
        pages_v = (len_prefill - len_sink - len_local_v - len_qbuf_v) // kvcache.PAGE_SIZE
        kvcache.Sink_Buffer_V[:,:,:len_sink,:].copy_(kvcache.value_states[:, :, :len_sink, :].contiguous())
        assert kvcache.Sink_Count == len_sink
        if len_local_v > 0:
            kvcache.Local_Buffer_V[:,:,:len_local_v,:].copy_(kvcache.value_states[:, :, -len_local_v:, :].contiguous())
            kvcache.Local_Count_V = len_local_v
        if len_qbuf_v > 0:
            assert len_local_v > 0, "Local buffer must be filled before using Q-buffer."
            kvcache.Q_Buffer_V[:,:,:len_qbuf_v,:].copy_(kvcache.value_states[:, :, -len_local_v-len_qbuf_v:-len_local_v, :].contiguous())
            kvcache.Q_Buffer_Count_V = len_qbuf_v
        if pages_v > 0:
            assert len_prefill - len_sink - len_local_v - len_qbuf_v == pages_v * kvcache.PAGE_SIZE
            quantize_pack_v(
                # data source
                value_states = kvcache.value_states,
                value_states_t_offset = len_sink,
                value_states_page_count = pages_v,
                pages_size = kvcache.PAGE_SIZE,
                # data destination
                page_table_v = kvcache.PageTable_V,
                page_table_v_metadata = kvcache.PageTable_V_metadata,
                page_count_v = kvcache.PageCount_V,
            )
            kvcache.PageCount_V += pages_v
        # Clear the legacy states
        kvcache.key_states = None
        kvcache.value_states = None
        return

    def quantize_decode(self, layer_idx: int = 0) -> None:
        """
        Only detect if the Q-Buffer is full, then quantize it to the KV Cache.
        Note that the Q-Buffer of Key Cache and Value Cache may not be full at the same time.
        """
        kvcache = self.kv_cache[layer_idx]
        # Key Cache
        if kvcache.Q_Buffer_Count_K == kvcache.PAGE_SIZE:
            quantize_pack_k(
                # data source
                key_states = kvcache.Q_Buffer_K,
                key_states_t_offset = 0,
                key_states_page_count = 1,
                pages_size = kvcache.PAGE_SIZE,
                # data destination
                page_table_k = kvcache.PageTable_K,
                page_table_k_metadata = kvcache.PageTable_K_metadata,
                page_count_k = kvcache.PageCount_K,
            )
            kvcache.PageCount_K += 1
            kvcache.Q_Buffer_Count_K = 0
        # Value Cache
        if kvcache.Q_Buffer_Count_V == kvcache.PAGE_SIZE:
            quantize_pack_v(
                # data source
                value_states = kvcache.Q_Buffer_V,
                value_states_t_offset = 0,
                value_states_page_count = 1,
                pages_size = kvcache.PAGE_SIZE,
                # data destination
                page_table_v = kvcache.PageTable_V,
                page_table_v_metadata = kvcache.PageTable_V_metadata,
                page_count_v = kvcache.PageCount_V,
            )
            kvcache.PageCount_V += 1
            kvcache.Q_Buffer_Count_V = 0

        return

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        KVCache_Layer = self.kv_cache[layer_idx]
        seqlen_K = KVCache_Layer.PageCount_K * KVCache_Layer.PAGE_SIZE + KVCache_Layer.Sink_Count + KVCache_Layer.Q_Buffer_Count_K
        # For debug
        seqlen_V = KVCache_Layer.PageCount_V * KVCache_Layer.PAGE_SIZE + KVCache_Layer.Sink_Count + KVCache_Layer.Q_Buffer_Count_V + KVCache_Layer.Local_Count_V
        assert seqlen_K == seqlen_V, f"seqlen_K: {seqlen_K}, seqlen_V: {seqlen_V}"
        #
        return seqlen_K


def get_kvcache_kchanboost(
        config: PretrainedConfig,
        max_batch_size: int,
        max_length: int,) -> KChanBoostCache:
    """
    Get the KChanBoostCache object.
    Returns:
        KChanBoostCache: The KChanBoostCache object.
    """
    #
    return KChanBoostCache(
        config=config,
        max_batch_size=max_batch_size,
        max_length=max_length,
    )

