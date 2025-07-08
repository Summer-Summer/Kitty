#!/bin/bash

# 参数检查
if [ "$#" -ne 3 ]; then
  echo 'Usage:   ./XXXXX.sh <MODEL> <TASK_NAME> <GPUs>'
  echo '<MODEL>: "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct"'
  echo '<TASK_NAME>: "gsm8k_haojun" "minerva_math_algebra" "humaneval_haojun2" "gpqa_diamond_haojun" "mmlu_flan_cot_fewshot"'
  echo '<GPUs>: "6,7" "0"'
  exit 1
fi

MODEL=$1
TASK_NAME=$2
GPUs=$3

# 固定设置
export TORCH_CUDA_ARCH_LIST="9.0"
HF_HOME=/home/xhjustc/HF_HOME
GROUP_SIZE=128
BUFFER_LENGTH=128
PROMOTE_RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#PROMOTE_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# 通用函数
run_experiment () {
  local sink=$1
  local channel=$2
  local label=$3
  local kbits=$4
  local vbits=$5
  local promote_bit=$6

  for PROMOTE_RATIO in "${PROMOTE_RATIOS[@]}"; do
    echo "Launching $TASK_NAME ($label promote_ratio=$PROMOTE_RATIO, promote_bit=$promote_bit, k=$kbits, v=$vbits) on GPUs $GPUs"
    CUDA_VISIBLE_DEVICES=$GPUs TOKENIZERS_PARALLELISM=false \
    python eval_channelkv.py $MODEL \
      --task_list $TASK_NAME \
      --eval_channelkv \
      --sink_length $sink \
      --kbits ${kbits} --vbits ${vbits} \
      --channel_selection $channel \
      --group_size ${GROUP_SIZE} --buffer_length ${BUFFER_LENGTH} \
      --promote_ratio $PROMOTE_RATIO \
      --promote_bit $promote_bit \
      > logs/${GPUs//,/}.log 2>&1 &
    wait
  done
}

# 通用函数
run_baseline () {
  local sink=$1
  local channel=$2
  local label=$3
  local kbits=$4
  local vbits=$5

  echo "Launching $TASK_NAME ($label promote_ratio=0.0, sink=$sink, k=$kbits, v=$vbits, channel=$channel) on GPUs $GPUs"
  CUDA_VISIBLE_DEVICES=$GPUs TOKENIZERS_PARALLELISM=false \
    python eval_channelkv.py $MODEL \
      --task_list $TASK_NAME \
      --eval_channelkv \
      --sink_length $sink \
      --kbits ${kbits} --vbits ${vbits} \
      --channel_selection $channel \
      --group_size ${GROUP_SIZE} --buffer_length ${BUFFER_LENGTH} \
      --promote_ratio 0.0 \
      > logs/${GPUs//,/}.log 2>&1 &
  wait
}