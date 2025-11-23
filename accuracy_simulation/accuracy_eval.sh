#!/bin/bash
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
  echo 'Usage:   ./accuracy_eval5.sh <MODEL> <TASK_NAME> <GPUs> [NUM_REPEATS] [BATCH_SIZE]'
  echo '<MODEL>: "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B" "Qwen/Qwen3-4B"'
  echo '<TASK_NAME>: "gsm8k_cot_llama" "minerva_math_algebra" "humaneval_instruct" "gpqa_diamond_cot_n_shot" "mmlu_flan_cot_fewshot" "aime24" "aime25"'
  echo '<GPUs>: "6,7" "0"'
  echo '[NUM_REPEATS]: Number of evaluation repeats (default: 1)'
  echo '[BATCH_SIZE]: Batch size for inference (default: 1)'
  echo ''
  echo 'Examples:'
  echo '  ./accuracy_eval5.sh "Qwen/Qwen3-8B" aime24 0 10 2    # 10 repeats, batch_size=2'
  echo '  ./accuracy_eval5.sh "Qwen/Qwen3-8B" aime24 0 10      # 10 repeats, batch_size=1 (default)'
  echo '  ./accuracy_eval5.sh "Qwen/Qwen3-8B" aime24 0         # 1 repeat, batch_size=1 (default)'
  exit 1
fi
MODEL=$1
TASK_NAME=$2
GPUs=$3
NUM_REPEATS=${4:-1}  # Default to 1 if not provided
BATCH_SIZE=${5:-1}   # Default to 1 if not provided

#
export TORCH_CUDA_ARCH_LIST="9.0"


echo "===================================="
echo "Evaluating MODEL: $MODEL"
echo "Running TASK: $TASK_NAME"
echo "GPUs: $GPUs"
echo "NUM_REPEATS: $NUM_REPEATS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "===================================="

source ./utils.sh


#####################################################################################################################
# Note:
# Please comment the experiments that you do not want to run.
# channel_selection: (0)  for Random Selection; (1)  Magnitude-based Channel Selection;

# Evaluating Kitty algorithm
#               label           sink  channel_sel  kbits  vbits  promote_bit    promote_ratio
run_single_exp  "Kitty-Pro"     32    1            2      2      4              0.2

# Evaluating K16V16 baseline
run_hf_baseline

# More evaluations
#               label      sink   channel_sel  kbits  vbits  promote_bit    promote_ratio
run_single_exp  "KIVI-2"    0     0            2      2      4              0.0             # KIVI-2
run_single_exp  "KIVI*-2"  32     0            2      2      4              0.0             # KIVI*-2

echo "Accuracy Evaluations for $TASK_NAME on $MODEL completed."
#####################################################################################################################