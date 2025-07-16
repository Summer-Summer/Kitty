#!/bin/bash    

export TORCH_CUDA_ARCH_LIST="9.0"

GPU=6
MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL=meta-llama/Llama-3.3-70B-Instruct
MODEL=Qwen/Qwen3-8B

echo "Generating for ${MODEL} on GPU ${GPU} with Original Huggingface KV Config."
CUDA_VISIBLE_DEVICES=${GPU} \
gen_rock_kv \
--model ${MODEL} \
--max_token_new 500 \
--batch_size 1 \
--sink_length 32 \
--buffer_length 128 \
--group_size 128 \
--kbits 16 \
--vbits 16 \
--promote_ratio 0.0 \
--promote_bit 4 \
--channel_selection 3

echo "Generating for ${MODEL} on GPU ${GPU} with RoCK-KV Config."
CUDA_VISIBLE_DEVICES=${GPU} \
gen_rock_kv \
--model ${MODEL} \
--gen_rock_kv \
--max_token_new 500 \
--batch_size 1 \
--sink_length 32 \
--buffer_length 128 \
--group_size 128 \
--kbits 2 \
--vbits 2 \
--promote_ratio 0.0 \
--promote_bit 4 \
--channel_selection 3