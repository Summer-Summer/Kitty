#!/bin/bash
#SBATCH --job-name=kitty_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=150:00:00
#SBATCH --partition=batch
#SBATCH --output=log/slurm_%j.out
#SBATCH --error=log/slurm_%j.err

# ============================================================================
# 配置区域
# ============================================================================

export MODEL="Qwen/Qwen3-8B"
export NUM_REPEATS=5
export BATCH_SIZE=32
export MAX_NEW_TOKENS=4096  # aime24/25建议32768, 其他4096

export RESULTS_DIR="./eval_results_rerun"
export LOGS_DIR="eval_logs_rerun"

export DEBUG=0

export BUFFER_LENGTH=128
export GROUP_SIZE=128

export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME=/data/shared/huggingface

APPTAINER_SIF="/data/jisenli2/Kitty/build/kitty_v2.sif"
APPTAINER_IMG="/data/jisenli2/Kitty/build/kitty_v3.img" # have the math lib installed

# ============================================================================
# 实验配置
# ============================================================================
# Rerun: Variance-based Channel Selection (channel_selection=2)
# 每行一个实验, task 在第3列, 可以混合不同 task
#
# GPU | 函数           | task_name              | label              | sink | ch_sel | kbits | vbits | promo_bit | promo_ratio
declare -a EXPERIMENTS=(
    # --- gsm8k_cot_llama ---
    # "0  | run_single_exp | gsm8k_cot_llama       | sinkKIVI-K2V2      | 32   | 2      | 2     | 2     | 4         | 0.0"
    # "1  | run_single_exp | gsm8k_cot_llama       | kChanBoost-12.5    | 32   | 2      | 2     | 2     | 4         | 0.125"
    # "2  | run_single_exp | gsm8k_cot_llama       | kChanBoost-25      | 32   | 2      | 2     | 2     | 4         | 0.25"
    # "3  | run_single_exp | gsm8k_cot_llama       | kChanBoost-37.5    | 32   | 2      | 2     | 2     | 4         | 0.375"
    # "4  | run_single_exp | gsm8k_cot_llama       | kChanBoost-50      | 32   | 2      | 2     | 2     | 4         | 0.5"
    # "5  | run_single_exp | gsm8k_cot_llama       | kChanBoost-62.5    | 32   | 2      | 2     | 2     | 4         | 0.625"
    # "6  | run_single_exp | gsm8k_cot_llama       | kChanBoost-75      | 32   | 2      | 2     | 2     | 4         | 0.75"
    # "7  | run_single_exp | gsm8k_cot_llama       | kChanBoost-87.5    | 32   | 2      | 2     | 2     | 4         | 0.875"
    # --- minerva_math_algebra (uncomment when ready) ---
    "0  | run_single_exp | minerva_math_algebra  | sinkKIVI-K2V2      | 32   | 1      | 2     | 2     | 4         | 0.0"
    "1  | run_single_exp | minerva_math_algebra  | kChanBoost-12.5    | 32   | 1      | 2     | 2     | 4         | 0.125"
    "2  | run_single_exp | minerva_math_algebra  | kChanBoost-25      | 32   | 1      | 2     | 2     | 4         | 0.25"
    "3  | run_single_exp | minerva_math_algebra  | kChanBoost-37.5    | 32   | 1      | 2     | 2     | 4         | 0.375"
    "4  | run_single_exp | minerva_math_algebra  | kChanBoost-50      | 32   | 1      | 2     | 2     | 4         | 0.5"
    "5  | run_single_exp | minerva_math_algebra  | kChanBoost-62.5    | 32   | 1      | 2     | 2     | 4         | 0.625"
    "6  | run_single_exp | minerva_math_algebra  | kChanBoost-75      | 32   | 1      | 2     | 2     | 4         | 0.75"
    "7  | run_single_exp | minerva_math_algebra  | kChanBoost-87.5    | 32   | 1      | 2     | 2     | 4         | 0.875"
)

# ============================================================================
# 主程序
# ============================================================================

echo "=========================================="
echo "Kitty Evaluation (SLURM Mode)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Model: $MODEL"
echo "Num Repeats: $NUM_REPEATS"
echo "Batch Size: $BATCH_SIZE"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Results Dir: $RESULTS_DIR"
echo "Logs Dir: $LOGS_DIR"
echo "=========================================="
echo ""

mkdir -p "/data/jisenli2/Kitty/log"

declare -a PIDS=()

echo "Starting parallel experiments in Apptainer..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -ra PARTS <<< "$exp"
    TRIMMED_PARTS=()
    for part in "${PARTS[@]}"; do
        TRIMMED_PARTS+=($(echo "$part" | xargs))
    done

    GPU_ID="${TRIMMED_PARTS[0]}"
    FUNC_NAME="${TRIMMED_PARTS[1]}"
    TASK_NAME="${TRIMMED_PARTS[2]}"

    if [ "$FUNC_NAME" = "run_single_exp" ]; then
        LABEL="${TRIMMED_PARTS[3]}"
        SINK="${TRIMMED_PARTS[4]}"
        CHANNEL="${TRIMMED_PARTS[5]}"
        KBITS="${TRIMMED_PARTS[6]}"
        VBITS="${TRIMMED_PARTS[7]}"
        PROMOTE_BIT="${TRIMMED_PARTS[8]}"
        PROMOTE_RATIO="${TRIMMED_PARTS[9]}"
    fi

    echo "GPU ${GPU_ID}: [${TASK_NAME}] ${LABEL:-baseline} (sel=${CHANNEL:-n/a}, promo=${PROMOTE_RATIO:-n/a})"

    apptainer exec --nv \
        --bind $HOME:/workspace \
        --bind /data:/data \
        --overlay "$APPTAINER_IMG":ro \
        "$APPTAINER_SIF" \
        bash -c "
            cd /data/jisenli2/Kitty/accuracy_simulation

            export MODEL='$MODEL'
            export TASK_NAME='$TASK_NAME'
            export NUM_REPEATS=$NUM_REPEATS
            export BATCH_SIZE=$BATCH_SIZE
            export MAX_NEW_TOKENS=$MAX_NEW_TOKENS
            export RESULTS_DIR='$RESULTS_DIR'
            export DEBUG='$DEBUG'
            export GPUs='${GPU_ID}'
            export CUDA_VISIBLE_DEVICES='${GPU_ID}'
            export TORCH_CUDA_ARCH_LIST='9.0'
            export HF_HOME=/data/shared/huggingface
            export PYTHONUNBUFFERED=1
            export TOKENIZERS_PARALLELISM=false
            export HF_DATASETS_TRUST_REMOTE_CODE=1
            export PYTHONPATH='/data/jisenli2/Kitty/third_party/transformers/src:/data/jisenli2/Kitty/third_party/lm-evaluation-harness'

            source ./utils.sh

            MODEL_SHORT=\$(get_model_shortname \"\$MODEL\")
            export LOG_BASE_DIR=\"/data/jisenli2/Kitty/\${LOGS_DIR}/\${MODEL_SHORT}/\${TASK_NAME}\"
            mkdir -p \${LOG_BASE_DIR}

            if [ '${FUNC_NAME}' = 'run_hf_baseline' ]; then
                local_run() {
                    local repeats=\${NUM_REPEATS:-1}
                    local debug_flag=''
                    if [ \"\${DEBUG}\" = '1' ] || [ \"\${DEBUG}\" = 'true' ]; then
                        repeats=1
                        debug_flag='--debug'
                    fi
                    local model_short=\$(get_model_shortname \"\$MODEL\")
                    local log_name=\"\${GPUs//,/}_\${model_short}_\${TASK_NAME}_fp16.log\"

                    CUDA_VISIBLE_DEVICES=\$GPUs TOKENIZERS_PARALLELISM=false \
                    HF_DATASETS_TRUST_REMOTE_CODE=1 \
                        eval_kitty \$MODEL \
                            --task \$TASK_NAME \
                            --num_repeats \${repeats} \
                            --batch_size \${BATCH_SIZE:-1} \
                            --max_new_tokens \${MAX_NEW_TOKENS:-4096} \
                            --results_dir \${RESULTS_DIR:-./eval_results} \
                            \${debug_flag} > \${LOG_BASE_DIR}/\${log_name} 2>&1
                }
                local_run
            else
                local_run() {
                    local label=\$1 sink=\$2 channel=\$3
                    local kbits=\$4 vbits=\$5 promote_bit=\$6 promote_ratio=\$7

                    local repeats=\${NUM_REPEATS:-1}
                    local debug_flag=''
                    if [ \"\${DEBUG}\" = '1' ] || [ \"\${DEBUG}\" = 'true' ]; then
                        repeats=1
                        debug_flag='--debug'
                    fi

                    echo \"Launching \$TASK_NAME on GPUs \$GPUs\"
                    echo \"label=\$label, sink=\$sink, channel_sel=\$channel, k=\$kbits, v=\$vbits, promote_bit=\$promote_bit, promote_ratio=\$promote_ratio, num_repeats=\${repeats}\"

                    local model_short=\$(get_model_shortname \"\$MODEL\")
                    local promote_ratio_str=\$(echo \"\$promote_ratio\" | sed 's/\\./_/g')
                    local log_name=\"\${GPUs//,/}_\${model_short}_\${TASK_NAME}_s\${sink}_sel\${channel}_k\${kbits}v\${vbits}_pro\${promote_ratio_str}.log\"

                    CUDA_VISIBLE_DEVICES=\$GPUs TOKENIZERS_PARALLELISM=false \
                    HF_DATASETS_TRUST_REMOTE_CODE=1 \
                        eval_kitty \$MODEL \
                            --task \$TASK_NAME \
                            --eval_kitty \
                            --sink_length \$sink \
                            --buffer_length \${BUFFER_LENGTH} \
                            --group_size \${GROUP_SIZE} \
                            --kbits \${kbits} \
                            --vbits \${vbits} \
                            --promote_ratio \$promote_ratio \
                            --promote_bit \$promote_bit \
                            --channel_selection \$channel \
                            --num_repeats \${repeats} \
                            --batch_size \${BATCH_SIZE:-1} \
                            --max_new_tokens \${MAX_NEW_TOKENS:-4096} \
                            --results_dir \${RESULTS_DIR:-./eval_results} \
                            \${debug_flag} > \${LOG_BASE_DIR}/\${log_name} 2>&1
                }
                local_run '${LABEL}' ${SINK} ${CHANNEL} ${KBITS} ${VBITS} ${PROMOTE_BIT} ${PROMOTE_RATIO}
            fi
        " &

    PIDS+=($!)
    sleep 2
done

echo ""
echo "All tasks launched. PIDs: ${PIDS[@]}"
echo ""

echo "Waiting for all experiments to complete..."
wait

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End: $(date)"
echo "=========================================="
echo ""

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | sed 's/-/_/g')

echo "Results Summary:"
echo "----------------------------------------"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -ra PARTS <<< "$exp"
    TRIMMED_PARTS=()
    for part in "${PARTS[@]}"; do
        TRIMMED_PARTS+=($(echo "$part" | xargs))
    done
    GPU_ID="${TRIMMED_PARTS[0]}"
    TASK="${TRIMMED_PARTS[2]}"

    LOG_DIR="/data/jisenli2/Kitty/${LOGS_DIR}/${MODEL_SHORT}/${TASK}"
    LOG_FILES=$(ls ${LOG_DIR}/${GPU_ID}_*.log 2>/dev/null)

    if [ -n "$LOG_FILES" ]; then
        for LOG_FILE in $LOG_FILES; do
            COMPLETED=$(grep -c "All samples evaluated" "$LOG_FILE" 2>/dev/null || echo "0")
            FILENAME=$(basename "$LOG_FILE")
            echo "GPU ${GPU_ID}: [${TASK}] ${FILENAME} - ${COMPLETED}/${NUM_REPEATS} repeats completed"
        done
    else
        echo "GPU ${GPU_ID}: [${TASK}] No log file found"
    fi
done

echo ""
echo "Results: ${RESULTS_DIR}/${MODEL##*/}/"
echo "Logs: ${LOGS_DIR}/${MODEL_SHORT}/"
