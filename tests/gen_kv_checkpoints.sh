#!/bin/bash

# 获取 GPU 编号
GPU=$1
if [ -z "$GPU" ]; then
  echo "Usage: bash $0 <GPU_ID>"
  exit 1
fi

export TORCH_CUDA_ARCH_LIST="9.0"

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-32B"
)

for MODEL in "${MODELS[@]}"; do
  for PROMPT in 1 2; do
    echo "Generating for ${MODEL} on GPU ${GPU} with RoCK-KV Config, prompt ${PROMPT}."
    CUDA_VISIBLE_DEVICES=${GPU} \
    gen_rock_kv \
      --model ${MODEL} \
      --gen_rock_kv \
      --max_token_new 2048 \
      --batch_size 1 \
      --kbits 16 \
      --vbits 16 \
      --prompt_choice ${PROMPT} \
      --visualize_kv
  done
done