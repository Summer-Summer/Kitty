# Latency Benchmarking

This directory contains scripts used to measure end-to-end decoding latency of Kitty and baseline KV-cache implementations.

## Supported cache implementations
- `0` — Kitty KV cache
- `1` — HuggingFace FP16 static cache
- `2` — HuggingFace FP16 dynamic cache
- `3` — HF quantized cache (quanto backend)

## Benchmark Kitty

Choosing GPU to be used:
```bash
export CUDA_VISIBLE_DEVICES=0
```

Running Kitty-Pro (K2V2 + 25% key-cache channels promoted to INT4):
```bash
python3 benchmark_kitty.py --max_seq_len 256 --batch_size 1 --warmup_runs 1 --repeat_runs 1 --cache_implementation 0
```

Running original Huggingface FP16 KV cache:
```bash
python3 benchmark_kitty.py --max_seq_len 256 --batch_size 1 --warmup_runs 1 --repeat_runs 1 --cache_implementation 2
```

singularity exec --nv --bind /data:/data --overlay /data/jisenli2/Kitty/build/kitty_v2.img:ro /data/jisenli2/Kitty/build/kitty_v2.sif env HF_HOME=/data/jisenli2/hf_cache_local python3 /data/jisenli2/Kitty/latency_benchmarking/benchmark_kitty.py --max_seq_len 8192 --batch_size 1 --warmup_runs 1 --repeat_runs 3 --cache_implementation 0

singularity exec --nv --bind /data:/data --overlay /data/jisenli2/Kitty/build/kitty_v2.img:ro /data/jisenli2/Kitty/build/kitty_v2.sif env HF_HOME=/data/jisenli2/hf_cache_local python3 /data/jisenli2/Kitty/latency_benchmarking/run_benchmark.py --max_seq_len 2048 --batch_size 1 --warmup_runs 1 --repeat_runs 1 --cache_implementation 3