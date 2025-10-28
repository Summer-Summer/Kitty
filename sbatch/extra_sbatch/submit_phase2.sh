#!/bin/bash

# Phase 2: Submit remaining 14B and 32B jobs
# Run this after Phase 1 completes

echo "=========================================="
echo "PHASE 2: Remaining 14B and 32B jobs"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$HOME/RoCK-KV/sbatch/extra_sbatch"
NUM_REPEATS=3

# Check available idle nodes
IDLE_COUNT=$(sinfo -N -h -t idle -o "%N" | grep research-secure | wc -l)
echo "Available idle nodes: $IDLE_COUNT"
echo ""

if [ $IDLE_COUNT -lt 3 ]; then
    echo "⚠️  WARNING: Need 3 idle nodes for Phase 2, only $IDLE_COUNT available"
    echo "Some jobs may queue. Continue? (Ctrl-C to cancel)"
    sleep 3
fi

echo "Submitting consolidated jobs for full node utilization..."
echo ""

echo "[1/3] Submitting: Job 1 - All 14B tasks + 32B aime24 (8 GPUs)"
echo "  → 14B minerva (exp 0-1): 2 GPUs"
echo "  → 14B gpqa (exp 2-3): 2 GPUs"
echo "  → 14B gsm8k (exp 4-5): 2 GPUs"
echo "  → 32B aime24 baseline (exp 6): 2 GPUs"
# sbatch --job-name=phase2_node1 \
#     --nodes=1 \
#     --gres=gpu:8 \
#     --export=ALL,MODEL="Qwen/Qwen3-14B",TASK_NAME="minerva_math_algebra",NUM_GPUS=1,BATCH_SIZE=16,NUM_REPEATS=$NUM_REPEATS,EXP_START=0,EXP_END=6 \
#     $SCRIPT_DIR/run_eval.sh
# sleep 1

# echo "[2/3] Submitting: Job 2 - 32B gsm8k + minerva (8 GPUs)"
# echo "  → 32B gsm8k (exp 7-8): 4 GPUs"
# echo "  → 32B minerva (exp 9-10): 4 GPUs"
# sbatch --job-name=phase2_node2 \
#     --nodes=1 \
#     --gres=gpu:8 \
#     --export=ALL,MODEL="Qwen/Qwen3-32B",TASK_NAME="gsm8k_cot_llama",NUM_GPUS=2,BATCH_SIZE=16,NUM_REPEATS=$NUM_REPEATS,EXP_START=7,EXP_END=10 \
#     $SCRIPT_DIR/run_eval.sh
# sleep 1

echo "[3/3] Submitting: Job 3 - 32B gpqa (4 GPUs, 50% utilization)"
echo "  → 32B gpqa (exp 11-12): 4 GPUs"
sbatch --job-name=phase2_node3 \
    --nodes=1 \
    --gres=gpu:8 \
    --export=ALL,MODEL="Qwen/Qwen3-32B",TASK_NAME="gpqa_diamond_cot_n_shot",NUM_GPUS=2,BATCH_SIZE=8,NUM_REPEATS=$NUM_REPEATS,EXP_START=11,EXP_END=12 \
    $SCRIPT_DIR/run_eval.sh
sleep 1

echo ""
echo "=========================================="
echo "Phase 2 submitted (3 jobs, 3 nodes, 20 GPUs)"
echo "Node utilization:"
echo "  Node 1: 8/8 GPUs (100%) - All 14B + aime24"
echo "  Node 2: 8/8 GPUs (100%) - 32B gsm8k + minerva"
echo "  Node 3: 4/8 GPUs (50%)  - 32B gpqa"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u $USER"
echo "Logs: ~/RoCK-KV/log/"
echo ""
