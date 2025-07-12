import torch
from typing import Optional
from dataclasses import dataclass

from transformers.cache_utils import CacheConfig, DynamicCache

from .utils_quant import build_promote_mask, fake_quant_groupwise_lastdim


@dataclass
class RoCKKVCacheConfig(CacheConfig):
    """
    Configuration class for RoCK-KV cache settings.

    Attributes:
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
    """
    def __init__(
        self,
        sink_length: int = 32,
        buffer_length: int = 128,
        group_size: int = 128,
        k_bits: int = 2,
        v_bits: int = 2,
        promote_ratio: float = 0.1,
        promote_bit: int = 4,
        channel_selection: int = 3,               # -1: Unspecified, 0: Random, 1: Variance-based, 2: Magnitude-based, 3: RoPE-aware
        VCache_BitDecoding: bool = False,         # The behavior of Value Cache, set to True means BitDecoding, otherwise KIVI Style Value Cache
    ):
        super().__init__()
        self.sink_length = sink_length
        self.buffer_length = buffer_length
        self.group_size = group_size
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.promote_ratio = promote_ratio
        self.promote_bit = promote_bit
        self.channel_selection = channel_selection
        self.VCache_BitDecoding = VCache_BitDecoding
        #
        self.validate()

    def validate(self):
        """Validates if the arguments passed are correct"""
        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.channel_selection not in [0,1,2,3]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="channel_selection",
                    correct_value="0 or 1 or 2 or 3",
                    found_value=self.channel_selection,
                ),
            )
        if self.buffer_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="buffer_length",
                    correct_value="larger than 0",
                    found_value=self.buffer_length,
                ),
            )
        
class RoCKKVCache(DynamicCache):
    """
    A quantizer cache that supports RoCK-KV quantization.
    [batch_size, num_heads, seq_len, head_dim]
    """

    def __init__(self, cache_config: RoCKKVConfig) -> None:
        super().__init__()
        # Initialize RoCK-KV specific configurations
        self.sink_length = cache_config.sink_length
        self.buffer_length = cache_config.buffer_length
        self.group_size = cache_config.group_size
        self.kbits = cache_config.k_bits
        self.vbits = cache_config.v_bits
        self.promote_ratio = cache_config.promote_ratio
        self.promote_bit = cache_config.promote_bit
        self.channel_selection = cache_config.channel_selection
        self.VCache_BitDecoding = cache_config.VCache_BitDecoding
        # Used only for QuantCache where the seq-length can't be inferred easily from cache contents
        #self._seen_tokens = 0
        #self._quantized_key_cache: list[torch.Tensor] = []
        #self._quantized_value_cache: list[torch.Tensor] = []

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
            assert self.sink_length >= 0
            assert key_states.shape[-2] > self.sink_length, "RoCK-KV: sequence length must be greater than sink_length, currently."
            key_states_sink   = key_states[:, :, :self.sink_length, :].contiguous()
            value_states_sink = value_states[:, :, :self.sink_length, :].contiguous()
            key_states        = key_states[:, :, self.sink_length:, :].contiguous()
            value_states      = value_states[:, :, self.sink_length:, :].contiguous()
            #
            assert self.buffer_length % self.group_size == 0    # buffer length must be divisible by quantization group size
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
                    key_states_quant_slice = fake_quant_groupwise_lastdim(key_states_quant_slice, self.group_size, self.k_bits, promote_mask, self.promote_bit).transpose(2, 3).contiguous()
                    key_states_quant[:, :, i:i+self.buffer_length, :] = key_states_quant_slice
                key_states_full  = torch.cat([key_states_quant, key_states_full],   dim=2)
                # Quantize Value Cache
                if not self.VCache_BitDecoding:             # KIVI Style Value Cache
                    num_token_to_quantize = num_tokens - self.buffer_length
                value_states_quant = value_states[:, :, :num_token_to_quantize, :].contiguous()
                value_states_full  = value_states[:, :, num_token_to_quantize:, :].contiguous()
                value_states_quant = fake_quant_groupwise_lastdim(value_states_quant, self.group_size, self.v_bits)
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

            # quantize
            num_tokens_kv = keys_to_return.shape[-2]
            assert num_tokens_kv == keys_to_return.shape[-2]
            num_tokens_kv_to_quantize = num_tokens_kv - self.sink_length - self.buffer_length
            if num_tokens_kv_to_quantize > 0 and (num_tokens_kv_to_quantize % self.buffer_length == 1):  # need to quantize
                start_idx = - self.buffer_length - 1
                # Quantize Key Cache
                promote_mask = build_promote_mask(keys_to_return[:, :, -self.buffer_length-1:-1, :].transpose(2, 3).contiguous(), self.promote_ratio, self.channel_selection)
                newly_quantized_key = fake_quant_groupwise_lastdim(
                    keys_to_return[:, :, -self.buffer_length-1:-1, :].transpose(2, 3).contiguous(),
                    self.group_size, self.k_bits,
                    promote_mask, self.promote_bit).transpose(2, 3).contiguous()
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx][:,:,:-self.buffer_length,:], newly_quantized_key, key_states], dim=-2)
                # Quantize Value Cache (BitDecoding)
                if self.VCache_BitDecoding:
                    newly_quantized_value = fake_quant_groupwise_lastdim(values_to_return[:, :, -self.buffer_length-1:-1, :], self.group_size, self.v_bits)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx][:,:,:-self.buffer_length,:], newly_quantized_value, value_states], dim=-2) # -self.buffer_length
            # Quantize Value Cache (KIVI Style Value Cache, quantizing a Token each Decoding Step)   
            if not self.VCache_BitDecoding:
                if num_tokens_kv_to_quantize > 0:
                    newly_quantized_value = fake_quant_groupwise_lastdim(values_to_return[:,:, -self.buffer_length-1:-self.buffer_length, :], self.group_size, self.v_bits)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx][:,:,:-self.buffer_length,:], newly_quantized_value, value_states], dim=-2) # -self.buffer_length
        ####################################################################################################################
        return keys_to_return, values_to_return

    '''
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")
    '''