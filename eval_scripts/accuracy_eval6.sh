#!/bin/bash
if [ "$#" -ne 3 ]; then
  echo 'Usage:   ./accuracy_eval.sh <MODEL> <TASK_NAME> <GPUs>'
  echo '<MODEL>: "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B"'
  echo '<TASK_NAME>: "gsm8k_cot_llama" "minerva_math_algebra" "humaneval_instruct" "gpqa_diamond_cot_n_shot" "mmlu_flan_cot_fewshot"'
  echo '<GPUs>: "6,7" "0"'
  exit 1
fi
MODEL=$1
TASK_NAME=$2
GPUs=$3

# 固定设置
export TORCH_CUDA_ARCH_LIST="9.0"





echo "===================================="
echo "Evaluating MODEL: $MODEL"
echo "Running TASK: $TASK_NAME"
echo "GPUs: $GPUs"
echo "===================================="

source ./utils.sh

# channel_selection:
# (0)  for Random Selection;
# (1)  Variance-based Channel Selection;
# (2)  Magnitude-based Channel Selection;
# (3)  RoPE-aware Channel Selection;
#               label                     sink  channel_sel  kbits  vbits  promote_bit    promote_ratio
run_single_exp  "Accuracy_Across_Ratios"  8     0            2      2      8              0.0
run_single_exp  "Accuracy_Across_Ratios"  32     0            2      2      8              0.0
run_single_exp  "Accuracy_Across_Ratios"  128     0            2      2      8              0.0


echo "Accuracy Evaluations for $TASK_NAME on $MODEL completed."