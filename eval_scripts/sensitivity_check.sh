#!/bin/bash
if [ "$#" -ne 3 ]; then
  echo 'Usage:   ./sensitivity_check.sh <MODEL> <TASK_NAME> <GPUs>'
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
HF_HOME=/home/xhjustc/HF_HOME




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
#               label                   sink  channel_sel  kbits  vbits     promote_bit    $promote_ratio
#run_baseline    "Sensitivity Check"     32    0            4      4          8              0.1
#run_baseline    "Sensitivity Check"     32    1            4      4          8              0.1
#run_baseline    "Sensitivity Check"     32    2            4      4          8              0.1
#run_baseline    "Sensitivity Check"     32    3            4      4          8              0.1

Ks=(2 3 4 8 16)
Vs=(2 3 4 5 6 8 16)
Ks=(16)

for k in "${Ks[@]}"; do
  for v in "${Vs[@]}"; do
    echo "Running sensitivity check with k=$k, v=$v"
    run_baseline   "Sensitivity Check"  32  0     $k  $v  16   0.0
  done
done

echo "Sensitivity checks for $TASK_NAME on $MODEL completed."