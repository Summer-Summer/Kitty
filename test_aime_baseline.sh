#!/bin/bash

# AIME Task Baseline Evaluation Script (Full Node - FP16 Original Model)
# Test AIME mathematical reasoning tasks using HuggingFace default dynamic cache

sbatch --gres=gpu:1 --time=1:00:00 --cpus-per-task=4 --job-name=aime_baseline --output=log/baseline/aime_baseline_%j.out --wrap="
cd /home/jisenli2/RoCK-KV && 
singularity exec --nv \
    --bind /home/jisenli2/RoCK-KV:/workspace \
    --bind /home/jisenli2/lm-evaluation-harness:/lm-eval \
    --bind /home/jisenli2/transformers:/transformers \
    --bind /data:/data \
    rock_kv_cuda121.sif \
    bash -c '
        cd /workspace &&
        echo \"=== Environment Check ===\" &&
        python --version &&
        python -c \"import torch; print(f\\\"PyTorch version: {torch.__version__}\\\"); print(f\\\"CUDA available: {torch.cuda.is_available()}\\\"); print(f\\\"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\\\")\" &&
        python -c \"import transformers; print(f\\\"Transformers version: {transformers.__version__}\\\")\" &&
        python -c \"import lm_eval; print(f\\\"LM-eval version: {lm_eval.__version__}\\\")\" &&
        echo \"\" &&
        echo \"=== Installing RoCK-KV Project ===\" &&
        pip install -e . --force-reinstall --no-deps &&
        echo \"\" &&
        echo \"=== Setting Up Environment ===\" &&
        export PATH=\"/home/$USER/.local/bin:$PATH\" &&
        export HF_HOME=\"/workspace/.cache/huggingface\" &&
        mkdir -p /workspace/.cache/huggingface &&
        export TOKENIZERS_PARALLELISM=false &&
        export HF_DATASETS_TRUST_REMOTE_CODE=1 &&
        echo \"PATH is now: $PATH\" &&
        echo \"HF_HOME: $HF_HOME\" &&
        echo \"Available GPUs: $(nvidia-smi -L | wc -l)\" &&
        nvidia-smi &&
        echo \"\" &&
        echo \"=== Starting AIME Baseline Evaluation (FP16 Original Model) ===\" &&
        echo \"Model: Qwen/Qwen3-8B\" &&
        echo \"Task: aime (American Invitational Mathematics Examination)\" &&
        echo \"Mode: Debug (limited samples)\" &&
        echo \"Cache: HuggingFace Default Dynamic Cache (FP16)\" &&
        echo \"GPUs: Full Node\" &&
        echo \"Note: This is the baseline comparison for RoCK-KV\" &&
        echo \"\" &&
        /home/$USER/.local/bin/eval_rock_kv Qwen/Qwen3-8B \
            --task aime \
            --debug
    '
"
