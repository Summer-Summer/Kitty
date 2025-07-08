# rock_kv/__init__.py
"""
RoCK-KV: RoPE-aware Channel KV Quantization
"""

from .models.llama_rock_kv import LlamaForCausalLM_RoCKKV

__all__ = ["LlamaForCausalLM_RoCKKV"]