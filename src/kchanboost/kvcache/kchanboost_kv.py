# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py.
# Author: Haojun Xia (xhjustc@gmail.com)

from typing import Optional, Any
from dataclasses import dataclass
import argparse
import math

import torch
from transformers.cache_utils import CacheConfig, Cache
from transformers.configuration_utils import PretrainedConfig


@dataclass
class KChanBoostCacheConfig(CacheConfig):
    """
    Configuration class for KChanBoost KV cache settings.
    """

    def __init__(self,) -> None:
        super().__init__("kchanboost_kv")
        #
        self.validate()

    def validate(self):
        """Validates if the arguments passed are correct"""
        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Add validation checks for any parameters if needed
        pass
        """
        if self.channel_selection not in [0,1,2,3]:
        raise ValueError(
            incorrect_arg_msg.format(
                key="channel_selection",
                correct_value="0 or 1 or 2 or 3",
                found_value=self.channel_selection,
            ),
        )
        """


class KVCache_Layer:
    """
    KV Cache for a single layer.
    The pages are statically allocated in our current version.
    To Do: Multi-GPU Inference.
    """
    def __init__(self, MAX_BS: int, MAX_LEN: int, H_KV: int, D: int, D_BOOSTED: int, PAGE_SIZE: int, S: int):
        ######################################### Quantized Pages #########################################
        assert PAGE_SIZE % 128 == 0, "PAGE_SIZE must be a multiple of 128."
        self.BITS_PER_BYTE = 8
        self.HIGH_BIT = 4
        self.LOW_BIT = 2
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
        self.key_states = None
        self.value_states = None

        

class KChanBoostCache(Cache):
    """
    Class for KChanBoost KV caches.
    """
    def __init__(
        self,
        config: PretrainedConfig,
        kchanboost_config: KChanBoostCacheConfig,
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
        # Initialize KV Cache for each layer
        for _ in range(self.num_hidden_layers):
            self.kv_cache.append(KVCache_Layer(
                MAX_BS = max_batch_size,
                MAX_LEN = max_length,
                H_KV = self.num_key_value_heads,
                D = self.head_dim,
                D_BOOSTED = self.d_boosted,
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
            kvcache.key_states = key_states
            kvcache.value_states = value_states
            return True
        # Decode
        assert key_states.shape[-2] == 1 and value_states.shape[-2] == 1, \
            "Decode: key_states and value_states should have sequence length of 1."
        #  
        # Update the KV Cache
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
        kvcache.key_states = None
        kvcache.value_states = None
        return

    def quantize_decode(self, layer_idx: int = 0) -> None:
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







        
class RoCKKVCache(DynamicCache):
    """
    A quantizer cache that supports RoCK-KV quantization.
    [batch_size, num_heads, seq_len, head_dim]
    """

    def __init__(self, cache_config: RoCKKVCacheConfig) -> None:
        super().__init__()
        # Initialize RoCK-KV specific configurations
        self.sink_length = cache_config.sink_length
        self.buffer_length = cache_config.buffer_length
        self.group_size = cache_config.group_size
        self.kbits = cache_config.kbits
        self.vbits = cache_config.vbits
        self.promote_ratio = cache_config.promote_ratio
        self.promote_bit = cache_config.promote_bit
        self.channel_selection = cache_config.channel_selection
        self.VCache_BitDecoding = cache_config.VCache_BitDecoding
        self.cache_implementation = cache_config.cache_implementation

        # Used only for QuantCache where the seq-length can't be inferred easily from cache contents
        #self._seen_tokens = 0
        #self._quantized_key_cache: list[torch.Tensor] = []
        #self._quantized_value_cache: list[torch.Tensor] = []

    # To Do: support prefill length smaller than sink_length
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        #if layer_idx == 0:
        #    self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")
        ################################################## Prefill Phase ##################################################
        elif len(self.key_cache) == layer_idx:
            #self._quantized_key_cache.append(self._quantize(key_states.contiguous(), axis=self.axis_key))
            #self._quantized_value_cache.append(self._quantize(value_states.contiguous(), axis=self.axis_value))
            keys_to_return, values_to_return = key_states, value_states

            # Initialize the key and value caches for the layer
            self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))

            # Pre-processing Sink Tokens
            assert key_states.shape[-2] > self.sink_length, "RoCK-KV: sequence length must be greater than sink_length, currently."
            key_states_sink   = key_states[:, :, :self.sink_length, :].contiguous()
            value_states_sink = value_states[:, :, :self.sink_length, :].contiguous()
            key_states        = key_states[:, :, self.sink_length:, :].contiguous()
            value_states      = value_states[:, :, self.sink_length:, :].contiguous()
            #
            if key_states.shape[-2] <= self.buffer_length:
                key_states_full = key_states
                value_states_full = value_states
            else:
                # key_states.shape[-2] > self.buffer_length
                num_tokens = key_states.shape[-2]
                num_token_to_buffer = num_tokens % self.buffer_length
                num_token_to_quantize = num_tokens - num_token_to_buffer
                # Quantize Key Cache
                key_states_quant   = key_states[:, :, :num_token_to_quantize, :].contiguous()
                key_states_full    = key_states[:, :, num_token_to_quantize:, :].contiguous()
                #
                num_to_quant = key_states_quant.shape[-2]
                assert num_to_quant > 0 and num_to_quant % self.buffer_length == 0, f"num_to_quant: {num_to_quant}, buffer_length: {self.buffer_length}"
                for i in range(0, num_to_quant, self.buffer_length):
                    key_states_quant_slice = key_states_quant[:, :, i:i+self.buffer_length, :].transpose(2, 3).contiguous()
                    promote_mask = build_promote_mask(key_states_quant_slice, self.promote_ratio, self.channel_selection)
                    key_states_quant_slice = fake_quant_groupwise_lastdim(key_states_quant_slice, self.group_size, self.kbits, promote_mask, self.promote_bit).transpose(2, 3).contiguous()
                    key_states_quant[:, :, i:i+self.buffer_length, :] = key_states_quant_slice
                key_states_full  = torch.cat([key_states_quant, key_states_full],   dim=2)
                # Quantize Value Cache
                if not self.VCache_BitDecoding:             # KIVI Style Value Cache
                    num_token_to_quantize = num_tokens - self.buffer_length
                value_states_quant = value_states[:, :, :num_token_to_quantize, :].contiguous()
                value_states_full  = value_states[:, :, num_token_to_quantize:, :].contiguous()
                value_states_quant = fake_quant_groupwise_lastdim(value_states_quant, self.group_size, self.vbits)
                value_states_full = torch.cat([value_states_quant, value_states_full], dim=2)

            # Updating KV Cache
            self.key_cache[layer_idx]   = torch.cat([key_states_sink,   key_states_full],   dim=2)
            self.value_cache[layer_idx] = torch.cat([value_states_sink, value_states_full], dim=2)
        ################################################## Decoding Phase ##################################################
        else:
            #dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            #dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            #keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            #values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

            # Update key and value caches
            keys_to_return = [self.key_cache[layer_idx], key_states]
            values_to_return = [self.value_cache[layer_idx], value_states]
            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)

            #self.key_cache[layer_idx] = keys_to_return
            #self.value_cache[layer_idx] = values_to_return
            #return keys_to_return, values_to_return

            # quantize
            num_tokens_kv = keys_to_return.shape[-2]
            assert num_tokens_kv == keys_to_return.shape[-2]
            num_tokens_kv_to_quantize = num_tokens_kv - self.sink_length - self.buffer_length
            if num_tokens_kv_to_quantize > 0 and (num_tokens_kv_to_quantize % self.buffer_length == 1):  # need to quantize
                # Quantize Key Cache
                promote_mask = build_promote_mask(keys_to_return[:, :, -self.buffer_length-1:-1, :].transpose(2, 3).contiguous(), self.promote_ratio, self.channel_selection)
                newly_quantized_key = fake_quant_groupwise_lastdim(
                    keys_to_return[:, :, -self.buffer_length-1:-1, :].transpose(2, 3).contiguous(),
                    self.group_size, self.kbits,
                    promote_mask, self.promote_bit).transpose(2, 3).contiguous()
                #self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx][:,:,:-self.buffer_length,:], newly_quantized_key, key_states], dim=-2)
                self.key_cache[layer_idx] = torch.cat([keys_to_return[:,:,:-self.buffer_length-1,:], newly_quantized_key, key_states], dim=-2)
                # Quantize Value Cache (BitDecoding)
                if self.VCache_BitDecoding:
                    newly_quantized_value = fake_quant_groupwise_lastdim(values_to_return[:, :, -self.buffer_length-1:-1, :], self.group_size, self.vbits)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx][:,:,:-self.buffer_length,:], newly_quantized_value, value_states], dim=-2) # -self.buffer_length
            else:
                self.key_cache[layer_idx] = keys_to_return
                if self.VCache_BitDecoding:
                    self.value_cache[layer_idx] = values_to_return
            # Quantize Value Cache (KIVI Style Value Cache, quantizing a Token each Decoding Step)   
            if not self.VCache_BitDecoding:
                if num_tokens_kv_to_quantize > 0:
                    newly_quantized_value = fake_quant_groupwise_lastdim(values_to_return[:,:, -self.buffer_length-1:-self.buffer_length, :], self.group_size, self.vbits)
                    #self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx][:,:,:-self.buffer_length,:], newly_quantized_value, value_states], dim=-2) # -self.buffer_length
                    self.value_cache[layer_idx] = torch.cat([values_to_return[:,:,:-self.buffer_length-1,:], newly_quantized_value, values_to_return[:,:,-self.buffer_length:,:]], dim=-2) # -self.buffer_length
                else:
                    self.value_cache[layer_idx] = values_to_return
        ####################################################################################################################
        return keys_to_return, values_to_return


def get_kvcache_rock_kv(args: argparse.Namespace) -> RoCKKVCache:
    """
    Get the RoCKKVCache object.
    Returns:
        RoCKKVCache: The RoCKKVCache object.
    """
    #
    cache_config = RoCKKVCacheConfig(
        sink_length         = args.sink_length,
        buffer_length       = args.buffer_length,
        group_size          = args.group_size,
        kbits               = args.kbits,
        vbits               = args.vbits,
        promote_ratio       = args.promote_ratio,
        promote_bit         = args.promote_bit,
        channel_selection   = args.channel_selection,
        VCache_BitDecoding  = False,  # Using KIVI Style V Cache
    )
    #
    return RoCKKVCache(cache_config=cache_config)