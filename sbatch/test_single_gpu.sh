#!/bin/bash
#SBATCH --job-name=test_rock_kv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --output=log/test_%j.out
#SBATCH --error=log/test_%j.err

echo "=========================================="
echo "Test Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="
echo ""

# Environment variables
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME=/data/huggingface
export HF_TOKEN="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"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Apptainer paths
APPTAINER_SIF="$HOME/RoCK-KV/build/kchanboost.sif"
APPTAINER_IMG="$HOME/RoCK-KV/build/kchanboost.img"

# Test with a small model using debug mode (only 8 samples)
MODEL="Qwen/Qwen3-14B"
# MODEL="meta-llama/Llama-3.3-70B-Instruct"
TASK="gsm8k_cot_llama"

echo "Testing with model: $MODEL"
echo "Task: $TASK"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Run test inside Apptainer
apptainer exec --nv \
    --bind $HOME:/workspace \
    --bind /data:/data \
    --overlay "$APPTAINER_IMG":ro \
    "$APPTAINER_SIF" \
    bash -c "
        cd /workspace/RoCK-KV

        export CUDA_VISIBLE_DEVICES=0
        export HF_HOME=/data/huggingface
        export HF_TOKEN='$HF_TOKEN'
        export TOKENIZERS_PARALLELISM=false
        export HF_DATASETS_TRUST_REMOTE_CODE=1

        echo '=== Testing eval_rock_kv command ==='
        echo 'Command: eval_rock_kv'
        which eval_rock_kv
        echo ''

        echo '=== Running baseline test (FP16) ==='
        eval_rock_kv $MODEL \
            --task $TASK \
            --debug \
            --num_repeats 1 \
            --batch_size 1 \
            --kbits 2 \
            --vbits 2 \
            --group_size  128 \
            --eval_rock_kv \
            --buffer_length 128 \
            --promote_ratio 0.1 \
            --promote_bit 4 \
            --channel_selection 2 \
            --sink_length 32 \



        echo ''
        echo '=== Test completed ==='
    "

echo ""
echo "=========================================="
echo "Test Job Completed!"
echo "End: $(date)"
echo "=========================================="
