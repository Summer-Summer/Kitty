#!/bin/bash    

export TORCH_CUDA_ARCH_LIST="9.0"

GPU=4,5,6,7
HF_HOME=/home/xhjustc/HF_HOME
#MODEL=${HF_HOME}/llama2/llama2-7b
#MODEL=${HF_HOME}/llama2/llama2-70b
#MODEL=${HF_HOME}/llama3.1/llama3.1-8b
#MODEL=${HF_HOME}/llama3.1/llama3.1-70b
MODEL=${HF_HOME}/DeepSeek_R1/llama-8b
#MODEL=${HF_HOME}/DeepSeek_R1/llama-70b


CUDA_VISIBLE_DEVICES=${GPU} \
python gen_test.py --model ${MODEL} \
--gen_kivi --gen_flashkv --gen_hf \
--torch_compile


# Visualize the kv cache
#CUDA_VISIBLE_DEVICES=${GPU} \
#python gen_test.py --model ${MODEL} \
#--gen_hf --visualize_kv