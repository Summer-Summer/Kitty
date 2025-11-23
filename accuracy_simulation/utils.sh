#!/bin/bash
GROUP_SIZE=128
BUFFER_LENGTH=128
PROMOTE_RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Helper function to get node name from hostname
get_node_name() {
  local hostname=$(hostname)
  # Extract node number from format: research-external-05.cloud.together.ai
  local node_num=$(echo "$hostname" | grep -oP 'external-\K\d+' | head -1)
  
  if [ -n "$node_num" ]; then
    echo "node_${node_num}"
  else
    # Fallback: use full hostname
    echo "${hostname%%.*}"
  fi
}

# Set log directory with node subfolder
LOG_BASE_DIR="eval_logs/$(get_node_name)"

# Helper function to generate model shortname
get_model_shortname() {
  local model=$1
  # Extract model name: "Qwen/Qwen3-4B" -> "Qwen3_4B"
  echo "$model" | sed 's|.*/||' | sed 's/-/_/g'
}


# run accuracy evaluation with simulated kitty.
run_single_exp () {
  local label=$1
  local sink=$2
  local channel=$3
  local kbits=$4
  local vbits=$5
  local promote_bit=$6
  local promote_ratio=$7
  
  # Use NUM_REPEATS from environment, default to 1
  local repeats=${NUM_REPEATS:-1}
  local debug_flag=""
  if [ "${DEBUG}" = "1" ] || [ "${DEBUG}" = "true" ]; then
    repeats=1
    debug_flag="--debug"
  fi
  
  echo "Launching $TASK_NAME on GPUs $GPUs"
  echo "label=$label, sink=$sink, channel_sel=$channel, k=$kbits, v=$vbits, promote_bit=$promote_bit, promote_ratio=$promote_ratio, num_repeats=${repeats}"
  mkdir -p ${LOG_BASE_DIR}

  local model_short=$(get_model_shortname "$MODEL")
  local promote_ratio_str=$(echo "$promote_ratio" | sed 's/\./_/g')
  local log_name="${GPUs//,/}_${model_short}_${TASK_NAME}_s${sink}_k${kbits}v${vbits}_pro${promote_ratio_str}.log"

  nohup sh -c "
    CUDA_VISIBLE_DEVICES=$GPUs TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_TRUST_REMOTE_CODE=1 \
      eval_kitty $MODEL \
        --task $TASK_NAME \
        --eval_kitty \
        --sink_length $sink \
        --buffer_length ${BUFFER_LENGTH} \
        --group_size ${GROUP_SIZE} \
        --kbits ${kbits} \
        --vbits ${vbits} \
        --promote_ratio $promote_ratio \
        --promote_bit $promote_bit \
        --channel_selection $channel \
        --num_repeats ${repeats} \
        --batch_size ${BATCH_SIZE:-1} \
        ${debug_flag}
  " > ${LOG_BASE_DIR}/${log_name} 2>&1 &
  wait
}


# run accuracy evaluation with standard huggingface.
run_hf_baseline () {
  # Use NUM_REPEATS from environment, default to 1
  local repeats=${NUM_REPEATS:-1}
  
  # If DEBUG is set, override to 1 repeat
  local debug_flag=""
  if [ "${DEBUG}" = "1" ] || [ "${DEBUG}" = "true" ]; then
    repeats=1
    debug_flag="--debug"
    echo "DEBUG mode enabled: forcing num_repeats=1, limit=8"
  fi
  
  echo "Launching $TASK_NAME on GPUs $GPUs"
  echo "Huggingface baseline, HF standard implementation, k=16, v=16, num_repeats=${repeats}"
  mkdir -p ${LOG_BASE_DIR}

  local model_short=$(get_model_shortname "$MODEL")
  local log_name="${GPUs//,/}_${model_short}_${TASK_NAME}_fp16.log"

  nohup sh -c "
    CUDA_VISIBLE_DEVICES=$GPUs TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_TRUST_REMOTE_CODE=1 \
      eval_kitty $MODEL \
        --task $TASK_NAME \
        --num_repeats ${repeats} \
        --batch_size ${BATCH_SIZE:-1} \
        ${debug_flag}
  " > ${LOG_BASE_DIR}/${log_name} 2>&1 &
  wait
}


# run accuracy evaluation with HuggingFace's Quantized KV Cache
run_hf_quantized_exp () {
  local backend=$1        # "HQQ" or "quanto"
  local nbits=$2          # 2, 4, 8
  local axis_key=$3       # 0 or 1 (HQQ: 1, quanto: 0)
  local axis_value=$4     # 0 or 1 (HQQ: 1, quanto: 0)
  
  # Use NUM_REPEATS from environment, default to 1
  local repeats=${NUM_REPEATS:-1}
  local debug_flag=""
  if [ "${DEBUG}" = "1" ] || [ "${DEBUG}" = "true" ]; then
    repeats=1
    debug_flag="--debug"
  fi
  
  echo "Launching $TASK_NAME on GPUs $GPUs"
  echo "HF Quantized KV Cache: backend=$backend, nbits=$nbits, axis_key=$axis_key, axis_value=$axis_value, num_repeats=${repeats}"
  mkdir -p ${LOG_BASE_DIR}

  local model_short=$(get_model_shortname "$MODEL")
  local log_name="${GPUs//,/}_${model_short}_${TASK_NAME}_hf_${backend,,}_int${nbits}_axis${axis_key}.log"

  nohup sh -c "
    CUDA_VISIBLE_DEVICES=$GPUs TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_TRUST_REMOTE_CODE=1 \
      eval_hf_kv $MODEL \
        --task $TASK_NAME \
        --hf_cache_backend $backend \
        --hf_cache_nbits $nbits \
        --hf_axis_key $axis_key \
        --hf_axis_value $axis_value \
        --num_repeats ${repeats} \
        --batch_size ${BATCH_SIZE:-1} \
        --max_new_tokens ${MAX_NEW_TOKENS:-4096} \
        --results_dir ${RESULTS_DIR:-./eval_results} \
        ${debug_flag}
  " > ${LOG_BASE_DIR}/${log_name} 2>&1 &
  wait
}