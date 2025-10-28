#!/bin/bash

# Master script to submit all model evaluation jobs
# 3 models × 2 experiments × 3 tasks = 18 total experiments
# Strategy: Prioritize 70B models, use 2-phase execution

echo "=========================================="
echo "Submitting Model Evaluation Jobs"
echo "=========================================="
echo "Total: 3 models × 2 experiments × 3 tasks = 18 experiments"
echo "Tasks: gsm8k_cot_llama, minerva_math_algebra, gpqa_diamond_cot_n_shot"
echo "Strategy: 70B priority (Phase 1: 4 nodes, Phase 2: 2 nodes)"
echo ""

# Check available idle nodes
IDLE_COUNT=$(sinfo -N -h -t idle -o "%N" | grep research-secure | wc -l)
echo "Available idle nodes: $IDLE_COUNT"
echo ""

if [ $IDLE_COUNT -lt 4 ]; then
    echo "⚠️  WARNING: Need 4 idle nodes for Phase 1, only $IDLE_COUNT available"
    echo "Some jobs may queue. Continue? (Ctrl-C to cancel)"
    sleep 3
fi

# Configuration
SCRIPT_DIR="$HOME/RoCK-KV/sbatch/extra_sbatch"
export NUM_REPEATS=3

echo "=========================================="
echo "PHASE 1: 70B Priority + 14B (4 nodes)"
echo "=========================================="
echo ""

# Phase 1: All 70B tasks (3 nodes) + 14B gsm8k (1 node)

echo "[1/4] Submitting: Llama-3.3-70B-Instruct - gsm8k_cot_llama"
sbatch --job-name=llama70b_gsm8k \
    --nodes=1 \
    --export=ALL,MODEL="meta-llama/Llama-3.3-70B-Instruct",TASK_NAME="gsm8k_cot_llama",NUM_GPUS=4,BATCH_SIZE=32,NUM_REPEATS=$NUM_REPEATS,EXP_START=0,EXP_END=1 \
    $SCRIPT_DIR/run_eval.sh
sleep 1

echo "[2/4] Submitting: Llama-3.3-70B-Instruct - minerva_math_algebra"
sbatch --job-name=llama70b_minerva \
    --nodes=1 \
    --export=ALL,MODEL="meta-llama/Llama-3.3-70B-Instruct",TASK_NAME="minerva_math_algebra",NUM_GPUS=4,BATCH_SIZE=32,NUM_REPEATS=$NUM_REPEATS,EXP_START=0,EXP_END=1 \
    $SCRIPT_DIR/run_eval.sh
sleep 1

echo "[3/4] Submitting: Llama-3.3-70B-Instruct - gpqa_diamond_cot_n_shot"
sbatch --job-name=llama70b_gpqa \
    --nodes=1 \
    --export=ALL,MODEL="meta-llama/Llama-3.3-70B-Instruct",TASK_NAME="gpqa_diamond_cot_n_shot",NUM_GPUS=4,BATCH_SIZE=24,NUM_REPEATS=$NUM_REPEATS,EXP_START=0,EXP_END=1 \
    $SCRIPT_DIR/run_eval.sh
sleep 1

# echo "[4/4] Submitting: Qwen3-14B - gsm8k_cot_llama"
# sbatch --job-name=qwen14b_gsm8k \
#     --nodes=1 \
#     --export=ALL,MODEL="Qwen/Qwen3-14B",TASK_NAME="gsm8k_cot_llama",NUM_GPUS=1,BATCH_SIZE=16,NUM_REPEATS=$NUM_REPEATS,EXP_START=0,EXP_END=1 \
#     $SCRIPT_DIR/run_eval.sh
# sleep 1

echo ""
echo "=========================================="
echo "Phase 1 submitted (4 jobs, 4 nodes)"
echo "=========================================="
echo ""
echo "Waiting for Phase 1 to complete before starting Phase 2..."
echo "You can monitor progress with: squeue -u $USER"
echo ""
echo "After Phase 1 completes, run Phase 2 manually:"
echo "  ./submit_phase2.sh"
echo ""
