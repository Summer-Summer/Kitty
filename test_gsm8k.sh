#!/bin/bash

sbatch --gres=gpu:1 --time=1:00:00 --cpus-per-task=4 --job-name=rock_kv_custom_test --output=rock_kv_gsm8k_%j.out --wrap="
cd /home/jisenli2/RoCK-KV && 
singularity exec --nv \
    --bind /home/jisenli2/RoCK-KV:/workspace \
    --bind /home/jisenli2/lm-evaluation-harness:/lm-eval \
    --bind /home/jisenli2/transformers:/transformers \
    --bind /data:/data \
    rock_kv_cuda121.sif \
    bash -c '
        cd /workspace &&
        echo \"Checking container environment...\" &&
        python --version &&
        python -c \"import torch; print(f\\\"PyTorch version: {torch.__version__}\\\"); print(f\\\"CUDA available: {torch.cuda.is_available()}\\\"); print(f\\\"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\\\")\" &&
        python -c \"import transformers; print(f\\\"Transformers version: {transformers.__version__}\\\")\" &&
        python -c \"import lm_eval; print(f\\\"LM-eval version: {lm_eval.__version__}\\\")\" &&
        echo \"Installing RoCK-KV project...\" &&
        pip install -e . &&
        echo \"Adding .local/bin to PATH...\" &&
        export PATH=\"/home/$USER/.local/bin:$PATH\" &&
        echo \"PATH is now: $PATH\" &&
        echo \"Setting environment variables...\" &&
        export HF_HOME=\"/workspace/.cache/huggingface\" &&
        mkdir -p /workspace/.cache/huggingface &&
        export TOKENIZERS_PARALLELISM=false &&
        export HF_DATASETS_TRUST_REMOTE_CODE=1 &&
        echo \"=== Starting RoCK-KV evaluation ===\" &&
        /home/$USER/.local/bin/eval_rock_kv Qwen/Qwen3-8B --task gsm8k_cot_llama --eval_rock_kv --channel_selection 3 --debug
    '
"
