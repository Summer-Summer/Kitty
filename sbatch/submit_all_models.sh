#!/bin/bash

# Master script to submit all model evaluation jobs
# 3 models × 8 methods × 1 task (AIME24) = 24 experiments total
# Total nodes needed: 1 (14B) + 2 (32B) + 4 (70B) = 7 nodes

echo "=========================================="
echo "Submitting Model Evaluation Jobs"
echo "=========================================="
echo "Total: 3 models × 8 methods × 1 task = 24 experiments"
echo "Nodes required: 7 total (1 for 14B, 2 for 32B, 4 for 70B)"
echo ""

# Check available idle nodes
IDLE_COUNT=$(sinfo -N -h -t idle -o "%N" | grep research-secure | wc -l)
echo "Available idle nodes: $IDLE_COUNT"
echo ""

if [ $IDLE_COUNT -lt 7 ]; then
    echo "⚠️  WARNING: Need 7 idle nodes, only $IDLE_COUNT available"
    echo "Some jobs may queue. Continue? (Ctrl-C to cancel)"
    sleep 3
fi

# ============================================================================
# Missing Repeat Experiments - KChanBoost Configurations
# ============================================================================
# Configuration: sink=32, k=2, v=2, promote_bit=4, channel_sel=2
# K2.25V2 = promote_ratio 0.125 (12.5%) -> EXP_START=0, EXP_END=0
# K2.5V2  = promote_ratio 0.25  (25%)  -> EXP_START=1, EXP_END=1
# Assuming NUM_REPEATS=3 (repeat_idx: 0, 1, 2)
# ============================================================================

# ----------------------------------------------------------------------------
# 1. Qwen3-14B (NUM_GPUS=1)
# ----------------------------------------------------------------------------

# 1.1 Qwen3-14B - K2.25V2 - gsm8k_cot_llama - Missing repeat_idx=2
echo "[Qwen3-14B] Submitting: K2.25V2 - gsm8k_cot_llama (repeat 2)"
sbatch --job-name=qwen14b_k2.25v2_gsm8k_r2 \
    --nodes=1 \
    --export=ALL,MODEL="Qwen/Qwen3-14B",TASK_NAME="gsm8k_cot_llama",NUM_GPUS=1,BATCH_SIZE=16,\
NUM_REPEATS=3,REPEAT_START=2,REPEAT_COUNT=1,EXP_START=0,EXP_END=0 \
    /home/shirley/RoCK-KV/sbatch/run_eval.sh
sleep 1

# ----------------------------------------------------------------------------
# 2. Qwen3-32B (NUM_GPUS=2)
# ----------------------------------------------------------------------------

# 2.1 Qwen3-32B - K2.25V2 - minerva_math_algebra - Missing repeat_idx=1,2
echo "[Qwen3-32B] Submitting: K2.25V2 - minerva_math_algebra (repeats 1-2)"
sbatch --job-name=qwen32b_k2.25v2_math_r12 \
    --nodes=1 \
    --export=ALL,MODEL="Qwen/Qwen3-32B",TASK_NAME="minerva_math_algebra",NUM_GPUS=2,BATCH_SIZE=16,\
NUM_REPEATS=3,REPEAT_START=1,REPEAT_COUNT=2,EXP_START=0,EXP_END=0 \
    /home/shirley/RoCK-KV/sbatch/run_eval.sh
sleep 1

# 2.2 Qwen3-32B - K2.5V2 - gsm8k_cot_llama - Missing repeat_idx=2
echo "[Qwen3-32B] Submitting: K2.5V2 - gsm8k_cot_llama (repeat 2)"
sbatch --job-name=qwen32b_k2.5v2_gsm8k_r2 \
    --nodes=1 \
    --export=ALL,MODEL="Qwen/Qwen3-32B",TASK_NAME="gsm8k_cot_llama",NUM_GPUS=2,BATCH_SIZE=16,\
NUM_REPEATS=3,REPEAT_START=2,REPEAT_COUNT=1,EXP_START=1,EXP_END=1 \
    /home/shirley/RoCK-KV/sbatch/run_eval.sh
sleep 1

# 2.3 Qwen3-32B - K2.5V2 - minerva_math_algebra - Missing repeat_idx=1,2
echo "[Qwen3-32B] Submitting: K2.5V2 - minerva_math_algebra (repeats 1-2)"
sbatch --job-name=qwen32b_k2.5v2_math_r12 \
    --nodes=1 \
    --export=ALL,MODEL="Qwen/Qwen3-32B",TASK_NAME="minerva_math_algebra",NUM_GPUS=2,BATCH_SIZE=16,\
NUM_REPEATS=3,REPEAT_START=1,REPEAT_COUNT=2,EXP_START=1,EXP_END=1 \
    /home/shirley/RoCK-KV/sbatch/run_eval.sh
sleep 1

# ----------------------------------------------------------------------------
# 3. LLaMA-3.3-70B-Instruct (NUM_GPUS=4) [OPTIONAL]
# ----------------------------------------------------------------------------

# 3.1 LLaMA-3.3-70B - K2.25V2 - minerva_math_algebra - Missing repeat_idx=1,2
echo "[LLaMA-3.3-70B] Submitting: K2.25V2 - minerva_math_algebra (repeats 1-2)"
sbatch --job-name=llama70b_k2.25v2_math_r12 \
    --nodes=1 \
    --export=ALL,MODEL="meta-llama/Llama-3.3-70B-Instruct",TASK_NAME="minerva_math_algebra",NUM_GPUS=4,BATCH_SIZE=16,\
NUM_REPEATS=3,REPEAT_START=1,REPEAT_COUNT=2,EXP_START=0,EXP_END=0 \
    /home/shirley/RoCK-KV/sbatch/run_eval.sh
sleep 1

# 3.2 LLaMA-3.3-70B - K2.5V2 - minerva_math_algebra - Missing repeat_idx=1,2
echo "[LLaMA-3.3-70B] Submitting: K2.5V2 - minerva_math_algebra (repeats 1-2)"
sbatch --job-name=llama70b_k2.5v2_math_r12 \
    --nodes=1 \
    --export=ALL,MODEL="meta-llama/Llama-3.3-70B-Instruct",TASK_NAME="minerva_math_algebra",NUM_GPUS=4,BATCH_SIZE=16,\
NUM_REPEATS=3,REPEAT_START=1,REPEAT_COUNT=2,EXP_START=1,EXP_END=1 \
    /home/shirley/RoCK-KV/sbatch/run_eval.sh
sleep 1

# ============================================================================
# End of Missing Repeat Experiments
# ============================================================================




# # Job 1: Qwen3-14B - 1 node, 8 experiments (1 GPU each)
# echo "[1/7] Submitting: Qwen/Qwen3-14B (1 node, 8 parallel experiments)"
# sbatch --job-name=qwen14b_gpqa \
#     --nodes=1 \
#     --export=ALL,MODEL="Qwen/Qwen3-14B",NUM_GPUS=1,BATCH_SIZE=16,EXP_START=0,EXP_END=7 \
#     /home/shirley/RoCK-KV/sbatch/run_eval.sh
# #BATCH_SIZE=24,
# sleep 1

# # Jobs 2-3: Qwen3-32B - 2 nodes, 8 experiments (2 GPUs each, 4 per node)
# echo "[2/7] Submitting: Qwen/Qwen3-32B node 1/2 (experiments 0-3)"
# sbatch --job-name=qwen32b_n1_gpqa \
#     --nodes=1 \
#     --export=ALL,MODEL="Qwen/Qwen3-32B",NUM_GPUS=2,BATCH_SIZE=16,EXP_START=0,EXP_END=3 \
#     /home/shirley/RoCK-KV/sbatch/run_eval.sh
# #BATCH_SIZE=16,
# sleep 1

# echo "[3/7] Submitting: Qwen/Qwen3-32B node 2/2 (experiments 4-7)"
# sbatch --job-name=qwen32b_n2_gpqa \
#     --nodes=1 \
#     --export=ALL,MODEL="Qwen/Qwen3-32B",NUM_GPUS=2,BATCH_SIZE=16,EXP_START=4,EXP_END=7 \
#     /home/shirley/RoCK-KV/sbatch/run_eval.sh
# #BATCH_SIZE=16,
# sleep 1

# Jobs 4-7: Llama-3.3-70B - 4 nodes, 8 experiments (4 GPUs each, 2 per node)
# 🆕 现在需要指定 REPEAT_START 和 REPEAT_COUNT（必选参数）
# 示例：8 个 repeat，分成 4 组并行运行（每组 2 个 repeat）

# echo "[4/7] Submitting: meta-llama/Llama-3.3-70B-Instruct node 1/4 (experiments 0-1, repeats 0-1)"
# sbatch --job-name=llama70b_n1_exp01_r01 \
#     --nodes=1 \
#     --export=ALL,MODEL="meta-llama/Llama-3.3-70B-Instruct",NUM_GPUS=4,BATCH_SIZE=16,\
# EXP_START=0,EXP_END=1,NUM_REPEATS=8,REPEAT_START=0,REPEAT_COUNT=2 \
#     /home/shirley/RoCK-KV/sbatch/run_eval.sh

# sleep 1

# 🆕 提交更多并行任务（每组运行不同的 repeat）
# echo "[5/7] Submitting: meta-llama/Llama-3.3-70B-Instruct node 2/4 (experiments 0-1, repeats 2-3)"
# sbatch --job-name=llama70b_n2_exp01_r23 \
#     --nodes=1 \
#     --export=ALL,MODEL="meta-llama/Llama-3.3-70B-Instruct",NUM_GPUS=4,BATCH_SIZE=16,\
# EXP_START=0,EXP_END=1,NUM_REPEATS=8,REPEAT_START=2,REPEAT_COUNT=2 \
#     /home/shirley/RoCK-KV/sbatch/run_eval.sh

# sleep 1

# echo "[6/7] Submitting: meta-llama/Llama-3.3-70B-Instruct node 3/4 (experiments 0-1, repeats 4-5)"
# sbatch --job-name=llama70b_n3_exp01_r45 \
#     --nodes=1 \
#     --export=ALL,MODEL="meta-llama/Llama-3.3-70B-Instruct",NUM_GPUS=4,BATCH_SIZE=16,\
# EXP_START=0,EXP_END=1,NUM_REPEATS=8,REPEAT_START=4,REPEAT_COUNT=2 \
#     /home/shirley/RoCK-KV/sbatch/run_eval.sh

# sleep 1

# echo "[7/7] Submitting: meta-llama/Llama-3.3-70B-Instruct node 4/4 (experiments 0-1, repeats 6-7)"
# sbatch --job-name=llama70b_n4_exp01_r67 \
#     --nodes=1 \
#     --export=ALL,MODEL="meta-llama/Llama-3.3-70B-Instruct",NUM_GPUS=4,BATCH_SIZE=16,\
# EXP_START=0,EXP_END=1,NUM_REPEATS=8,REPEAT_START=6,REPEAT_COUNT=2 \
#     /home/shirley/RoCK-KV/sbatch/run_eval.sh

echo ""
echo "=========================================="
echo "All 7 jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u $USER"
echo "Logs: ~/RoCK-KV/log/"
echo ""
