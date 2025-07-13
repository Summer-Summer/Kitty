import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Import from transformers
from transformers import LlamaConfig, AutoTokenizer, PreTrainedModel, AutoModelForCausalLM

# Import RoCK-KV specific modules
from .utils_quant import build_promote_mask, fake_quant_groupwise_lastdim
from .rock_kv_config import RoCKKVConfig



def rock_kv_patch_llama(hf_model: PreTrainedModel, config: RoCKKVConfig) -> PreTrainedModel:
    """
    Monkey patch the official transformers PreTrainedModel to use RoCK-KV attention.
    """
    def rock_kv_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        


    for layer in hf_model.model.layers:
            # Add RoCK-KV specific configs to the layer
            layer.self_attn.k_bits = config.k_bits
            layer.self_attn.v_bits = config.v_bits
            layer.self_attn.group_size = config.group_size
            layer.self_attn.buffer_length = config.buffer_length
            layer.self_attn.sink_length = config.sink_length
            layer.self_attn.promote_ratio = config.promote_ratio
            layer.self_attn.promote_bit = config.promote_bit
            layer.self_attn.channel_selection = config.channel_selection
            layer.self_attn.VCache_BitDecoding = config.VCache_BitDecoding
            #
            original_attention_forward = layer.self_attn.forward
            
            def patched_attn_forward(*args, **kwargs):
                return 0
            
            layer.self_attn.forward = patched_attn_forward




    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
       NotImplementedError("This method should be implemented by subclasses")


class LlamaFlashAttention_RoCKKV(LlamaAttention_RoCKKV):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return self.forward_RoCKKV(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)

    def forward_RoCKKV(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        ###################################################### Decoding Phase ######################################################
        if past_key_value is not None:                              
            key_states_fake = past_key_value[0]
            value_states_fake = past_key_value[1]
            assert (key_states_fake is not None) and (value_states_fake is not None)
            # update kv cache
            key_states_fake = torch.cat([key_states_fake, key_states], dim=2)
            value_states_fake = torch.cat([value_states_fake, value_states], dim=2)

            attn_weights = torch.matmul(query_states, repeat_kv(key_states_fake, self.num_key_value_groups).transpose(2, 3))
            attn_weights = attn_weights / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device, dtype=attn_weights.dtype)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, repeat_kv(value_states_fake, self.num_key_value_groups))
            attn_output = attn_output.transpose(1, 2).contiguous()

            # quantize
            num_tokens_kv = key_states_fake.shape[-2]
            assert num_tokens_kv == value_states_fake.shape[-2]
            num_tokens_kv_to_quantize = num_tokens_kv - self.sink_length - self.buffer_length
            if num_tokens_kv_to_quantize > 0 and (num_tokens_kv_to_quantize % self.buffer_length == 1):  # need to quantize
                start_idx = - self.buffer_length - 1
                # Quantize Key Cache
                promote_mask = build_promote_mask(key_states_fake[:,:,   start_idx:-1, :].transpose(2, 3).contiguous(), self.promote_ratio, self.channel_selection)
                key_states_fake[:,:,   start_idx:-1, :] = fake_quant_groupwise_lastdim(
                    key_states_fake[:,:,   start_idx:-1, :].transpose(2, 3).contiguous(),
                    self.group_size, self.k_bits,
                    promote_mask, self.promote_bit).transpose(2, 3).contiguous()
                # Quantize Value Cache
                if self.VCache_BitDecoding:
                    value_states_fake[:,:, start_idx:-1, :] = fake_quant_groupwise_lastdim(
                        value_states_fake[:,:, start_idx:-1, :], 
                        self.group_size, self.v_bits)
                    
            if not self.VCache_BitDecoding:  # KIVI Style Value Cache, quantizing a Token each Decoding Step
                if num_tokens_kv_to_quantize > 0:
                    value_states_fake[:,:, -self.buffer_length-1:-self.buffer_length, :] = fake_quant_groupwise_lastdim(value_states_fake[:,:, -self.buffer_length-1:-self.buffer_length, :], self.group_size, self.v_bits)
        ###################################################### Prefill Phase ######################################################
        else:
            input_dtype = query_states.dtype
            assert input_dtype in (torch.float16, torch.bfloat16), f"Unsupported input dtype: {input_dtype}"
            #if input_dtype == torch.float32:
            #    # Handle the case where the model is quantized
            #    if hasattr(self.config, "_pre_quantization_dtype"):
            #        target_dtype = self.config._pre_quantization_dtype
            #    else:
            #        target_dtype = self.q_proj.weight.dtype
            #
            #    logger.warning_once(
            #        f"The input hidden states seems to be silently casted in float32, this might be related to"
            #        f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            #        f" {target_dtype}."
            #    )
            #
            #    query_states = query_states.to(target_dtype)
            #    key_states = key_states.to(target_dtype)
            #    value_states = value_states.to(target_dtype)
            attn_output = self._flash_attention_forward(
                query_states.transpose(1, 2), key_states.transpose(1, 2), 
                value_states.transpose(1, 2), None, q_len, dropout=0.0
            )

            # quantize
            assert self.sink_length >= 0
            assert key_states.shape[-2] > self.sink_length
            key_states_sink   = key_states[:, :, :self.sink_length, :].contiguous()
            value_states_sink = value_states[:, :, :self.sink_length, :].contiguous()
            key_states        = key_states[:, :, self.sink_length:, :].contiguous()
            value_states      = value_states[:, :, self.sink_length:, :].contiguous()

            #
            assert self.buffer_length % self.group_size == 0    # buffer length must be divisible by quantization group size
            # Quantizing Key Cache
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

            # Final Concat
            key_states_fake   = torch.cat([key_states_sink,   key_states_full],   dim=2)
            value_states_fake = torch.cat([value_states_sink, value_states_full], dim=2)
            
        #
        past_key_value = (key_states_fake, value_states_fake, kv_seq_len) if use_cache else None

        #
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        attn_weights = None
        return attn_output, attn_weights, past_key_value


    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output


    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )