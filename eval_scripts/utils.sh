#!/bin/bash
GROUP_SIZE=128
BUFFER_LENGTH=128
PROMOTE_RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# 通用函数
run_experiment () {
  local label=$1
  local sink=$2
  local channel=$3
  local kbits=$4
  local vbits=$5
  local promote_bit=$6
  mkdir -p eval_logs

  for PROMOTE_RATIO in "${PROMOTE_RATIOS[@]}"; do
    echo "Launching $TASK_NAME on GPUs $GPUs"
    echo "label=$label, sink=$sink, channel_sel=$channel, k=$kbits, v=$vbits, promote_bit=$promote_bit, promote_ratio=$PROMOTE_RATIO"
    CUDA_VISIBLE_DEVICES=$GPUs TOKENIZERS_PARALLELISM=false \
    eval_rock_kv $MODEL \
      --task_list $TASK_NAME \
      --eval_rock_kv \
      --sink_length $sink \
      --buffer_length ${BUFFER_LENGTH} \
      --group_size ${GROUP_SIZE}  \
      --kbits ${kbits} \
      --vbits ${vbits} \
      --promote_ratio $PROMOTE_RATIO \
      --promote_bit $promote_bit \
      --channel_selection $channel \
      --debug \
      > eval_logs/${GPUs//,/}.log 2>&1 &
    wait
  done
}

# 通用函数
run_baseline () {
  local label=$1
  local sink=$2
  local channel=$3
  local kbits=$4
  local vbits=$5
  local promote_bit=$6
  local promote_ratio=$7
  echo "Launching $TASK_NAME on GPUs $GPUs"
  echo "label=$label, sink=$sink, channel_sel=$channel, k=$kbits, v=$vbits, promote_bit=$promote_bit, promote_ratio=0$promote_ratio"
  mkdir -p eval_logs

  CUDA_VISIBLE_DEVICES=$GPUs TOKENIZERS_PARALLELISM=false \
    eval_rock_kv $MODEL \
      --task_list $TASK_NAME \
      --eval_rock_kv \
      --sink_length $sink \
      --buffer_length ${BUFFER_LENGTH} \
      --group_size ${GROUP_SIZE} \
      --kbits ${kbits} \
      --vbits ${vbits} \
      --promote_ratio $promote_ratio \
      --promote_bit $promote_bit \
      --channel_selection $channel \
      --debug \
      > eval_logs/${GPUs//,/}.log 2>&1 &
  wait
}