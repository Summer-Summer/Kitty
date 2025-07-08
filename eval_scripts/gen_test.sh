#!/bin/bash    

export TORCH_CUDA_ARCH_LIST="9.0"

GPU=0,3
HF_HOME=/home/xhjustc/HF_HOME
MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL=meta-llama/Llama-3.3-70B-Instruct

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
--channel_selection -1