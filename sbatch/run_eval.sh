#!/bin/bash
#SBATCH --job-name=rock_kv_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=200:00:00
#SBATCH --partition=batch
#SBATCH --output=log/slurm_%j.out
#SBATCH --error=log/slurm_%j.err
#SBATCH --exclude=research-secure-02
##SBATCH --nodelist=research-secure-04  # Commented: let SLURM auto-assign from idle nodes

# ============================================================================
# 配置区域 - 在这里修改参数
# ============================================================================

# 模型和任务配置
# <MODEL>: "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B" "Qwen/Qwen3-4B"
# <TASK_NAME>: "gsm8k_cot_llama" "minerva_math_algebra" "humaneval_instruct" "gpqa_diamond_cot_n_shot" "mmlu_flan_cot_fewshot" "aime24" "aime25"
# These can be overridden by sbatch --export
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export TASK_NAME="${TASK_NAME:-minerva_math_algebra}"
export NUM_REPEATS="${NUM_REPEATS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-6}"
export NUM_GPUS="${NUM_GPUS:-1}"
export EXP_START="${EXP_START:-0}"
export EXP_END="${EXP_END:-7}"

# 🆕 Repeat 并行运行参数（必选）
export REPEAT_START="${REPEAT_START}"
export REPEAT_COUNT="${REPEAT_COUNT}"

# DEBUG模式 (设置为1或true则只运行1个repeat且只运行前3题，用于快速测试)
export DEBUG=0

# 环境变量
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN="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"

# Apptainer路径
APPTAINER_SIF="$HOME/RoCK-KV/build/kchanboost.sif"
APPTAINER_IMG="$HOME/RoCK-KV/build/kchanboost.img"

# ============================================================================
# 实验配置 - 8个GPU上运行的8个不同配置
# ============================================================================
# 参考 accuracy_eval5.sh 中的调用方式:
#   run_hf_baseline
#   run_single_exp  "label"  sink  channel_sel  kbits  vbits  promote_bit  promote_ratio
#
# channel_selection: (0) Random  (1) Variance  (2) Magnitude  (3) RoPE-aware
# 当前所有实验统一使用 channel_sel=2 (Magnitude-based)
#
# 注意：
# 1. GPU可以是单个(0)或多个(0,1)，多GPU时会使用tensor parallel
# 2. 如果只运行部分GPU，注释掉不需要的行即可
# 3. 可以在同一节点多次提交，只要GPU_ID不重复
# ============================================================================
#3x8 
## 14B==> 1 nodes
## 32B ==> 2nodes
## 70==>4 nodes
# GPU     |  函数             |  label                   |  sink  |  channel_sel  |  kbits  |  vbits  |  promote_bit  |  promote_ratio
declare -a EXPERIMENTS=(
    # Baseline
    # "0    |  run_hf_baseline"
    # # KIVI K4V4
    # "1    |  run_single_exp  |  Accuracy_Across_Ratios  |   0    |      2        |    4    |    4    |       4       |      0.0"
    # # KIVI K2V2
    # "2    |  run_single_exp  |  Accuracy_Across_Ratios  |   0    |      2        |    2    |    2    |       4       |      0.0"
    # # sinkKIVI K4V2
    # "3    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    4    |    2    |       4       |      0.0"
    # # sinkKIVI K2V4
    # "4    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    2    |    4    |       4       |      0.0"
    # # sinkKIVI K2V2
    # "5    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    2    |    2    |       4       |      0.0"
    # sinkKIVI K2.2V2
    "0    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    2    |    2    |       4       |      0.125"
    # sinkKIVI K2.4V2
    "1    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    2    |    2    |       4       |      0.25"
)

# ============================================================================
# 主程序 - 通常不需要修改
# ============================================================================

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Model: $MODEL"
echo "Task: $TASK_NAME"
echo "Num Repeats: $NUM_REPEATS"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

mkdir -p "$HOME/RoCK-KV/log"

# 存储后台进程PID
declare -a PIDS=()

echo "Starting 8 parallel experiments in Apptainer..."
echo ""

# 启动GPU任务 (配置根据NUM_GPUS和EXP_START/EXP_END调整)
# Process only experiments in range [EXP_START, EXP_END]
# Map them to available GPUs 0-7 on this node
EXP_IDX=0
GPU_OFFSET=0

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -ra PARTS <<< "$exp"
    # Trim所有参数的空格
    TRIMMED_PARTS=()
    for part in "${PARTS[@]}"; do
        TRIMMED_PARTS+=($(echo "$part" | xargs))
    done

    EXP_ID="${TRIMMED_PARTS[0]}"
    FUNC_NAME="${TRIMMED_PARTS[1]}"

    # Skip experiments outside our assigned range
    if [ "$EXP_ID" -lt "$EXP_START" ] || [ "$EXP_ID" -gt "$EXP_END" ]; then
        continue
    fi

    # Configure GPU string based on NUM_GPUS for tensor parallelism
    # Map to local GPU indices starting from 0
    if [ "$NUM_GPUS" -eq 1 ]; then
        # 14B: 1 GPU per exp, experiments map to GPUs 0-7
        GPU_STRING="${GPU_OFFSET}"
        ((GPU_OFFSET++))
    elif [ "$NUM_GPUS" -eq 2 ]; then
        # 32B: 2 GPUs per exp, experiments map to GPU pairs (0-1, 2-3, 4-5, 6-7)
        GPU_START=$((GPU_OFFSET))
        GPU_END=$((GPU_OFFSET + 1))
        GPU_STRING="${GPU_START},${GPU_END}"
        GPU_OFFSET=$((GPU_OFFSET + 2))
    elif [ "$NUM_GPUS" -eq 4 ]; then
        # 70B: 4 GPUs per exp, experiments map to GPU quads (0-3, 4-7)
        GPU_START=$((GPU_OFFSET))
        GPU_END=$((GPU_OFFSET + 3))
        GPU_STRING=$(seq -s, ${GPU_START} ${GPU_END})
        GPU_OFFSET=$((GPU_OFFSET + 4))
    fi

    echo "Experiment ${EXP_ID} on GPU ${GPU_STRING}: ${FUNC_NAME} ${TRIMMED_PARTS[@]:2}"
    
    # 提取run_single_exp的参数（如果有）
    if [ "$FUNC_NAME" = "run_single_exp" ]; then
        LABEL="${TRIMMED_PARTS[2]}"
        SINK="${TRIMMED_PARTS[3]}"
        CHANNEL="${TRIMMED_PARTS[4]}"
        KBITS="${TRIMMED_PARTS[5]}"
        VBITS="${TRIMMED_PARTS[6]}"
        PROMOTE_BIT="${TRIMMED_PARTS[7]}"
        PROMOTE_RATIO="${TRIMMED_PARTS[8]}"
    fi
    
    # 在Apptainer中运行（后台）
    # 进入eval_scripts目录，source utils.sh，然后调用相应函数
    apptainer exec --nv \
        --bind $HOME:/workspace \
        --bind /data:/data \
        --overlay "$APPTAINER_IMG":ro \
        "$APPTAINER_SIF" \
        bash -c "
            cd /workspace/RoCK-KV/eval_scripts
            
            # 导出环境变量
            export MODEL='$MODEL'
            export TASK_NAME='$TASK_NAME'
            export NUM_REPEATS=$NUM_REPEATS
            export BATCH_SIZE=$BATCH_SIZE
            export REPEAT_START=$REPEAT_START     # 🆕
            export REPEAT_COUNT=$REPEAT_COUNT     # 🆕
            export DEBUG='$DEBUG'
            export GPUs='${GPU_STRING}'
            export CUDA_VISIBLE_DEVICES='${GPU_STRING}'
            export TORCH_CUDA_ARCH_LIST='9.0'
            export HF_HOME=/data/huggingface
            export HF_TOKEN='$HF_TOKEN'
            export TOKENIZERS_PARALLELISM=false
            export HF_DATASETS_TRUST_REMOTE_CODE=1
            
            # Source utils.sh
            source ./utils.sh
            
            # 调用函数（移除wait以真正后台运行）
            if [ '${FUNC_NAME}' = 'run_hf_baseline' ]; then
                # 修改函数，移除wait
                run_hf_baseline() {
                  local repeats=\${NUM_REPEATS:-1}
                  local debug_flag=''
                  if [ \"\${DEBUG}\" = '1' ] || [ \"\${DEBUG}\" = 'true' ]; then
                    repeats=1
                    debug_flag='--debug'
                  fi
                  
                  echo \"Launching \$TASK_NAME on GPUs \$GPUs\"
                  echo \"Huggingface baseline, k=16, v=16, num_repeats=\${repeats}\"
                  mkdir -p \${LOG_BASE_DIR}

                  local model_short=\$(get_model_shortname \"\$MODEL\")
                  local log_name=\"\${GPUs//,/}_\${model_short}_\${TASK_NAME}_fp16.log\"

                  CUDA_VISIBLE_DEVICES=\$GPUs TOKENIZERS_PARALLELISM=false \
                  HF_DATASETS_TRUST_REMOTE_CODE=1 \
                    eval_rock_kv \$MODEL \
                      --task \$TASK_NAME \
                      --num_repeats \${repeats} \
                      --batch_size \${BATCH_SIZE:-1} \
                      --repeat_start \${REPEAT_START} \
                      --repeat_count \${REPEAT_COUNT} \
                      \${debug_flag} > \${LOG_BASE_DIR}/\${log_name} 2>&1
                }
                run_hf_baseline
            else
                # run_single_exp
                run_single_exp() {
                  local label=\$1
                  local sink=\$2
                  local channel=\$3
                  local kbits=\$4
                  local vbits=\$5
                  local promote_bit=\$6
                  local promote_ratio=\$7
                  
                  local repeats=\${NUM_REPEATS:-1}
                  local debug_flag=''
                  if [ \"\${DEBUG}\" = '1' ] || [ \"\${DEBUG}\" = 'true' ]; then
                    repeats=1
                    debug_flag='--debug'
                  fi
                  
                  echo \"Launching \$TASK_NAME on GPUs \$GPUs\"
                  echo \"label=\$label, sink=\$sink, channel_sel=\$channel, k=\$kbits, v=\$vbits, promote_bit=\$promote_bit, promote_ratio=\$promote_ratio, num_repeats=\${repeats}\"
                  mkdir -p \${LOG_BASE_DIR}

                  local model_short=\$(get_model_shortname \"\$MODEL\")
                  local promote_ratio_str=\$(echo \"\$promote_ratio\" | sed 's/\\./_/g')
                  local log_name=\"\${GPUs//,/}_\${model_short}_\${TASK_NAME}_s\${sink}_k\${kbits}v\${vbits}_pro\${promote_ratio_str}.log\"

                  CUDA_VISIBLE_DEVICES=\$GPUs TOKENIZERS_PARALLELISM=false \
                  HF_DATASETS_TRUST_REMOTE_CODE=1 \
                    eval_rock_kv \$MODEL \
                      --task \$TASK_NAME \
                      --eval_rock_kv \
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
                      --repeat_start \${REPEAT_START} \
                      --repeat_count \${REPEAT_COUNT} \
                      \${debug_flag} > \${LOG_BASE_DIR}/\${log_name} 2>&1
                }
                run_single_exp '${LABEL}' ${SINK} ${CHANNEL} ${KBITS} ${VBITS} ${PROMOTE_BIT} ${PROMOTE_RATIO}
            fi
        " &
    
    PIDS+=($!)
    sleep 2
done

echo ""
echo "All tasks launched. PIDs: ${PIDS[@]}"
echo ""

# 等待所有任务完成
echo "Waiting for all experiments to complete..."
wait

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End: $(date)"
echo "=========================================="
echo ""

# 获取节点名称
NODE_NAME=$(hostname | grep -oP 'external-\K\d+' | head -1)
if [ -n "$NODE_NAME" ]; then
    NODE_NAME="node_${NODE_NAME}"
else
    NODE_NAME=$(hostname | cut -d. -f1)
fi

LOG_DIR="$HOME/RoCK-KV/eval_scripts/eval_logs/${NODE_NAME}"

# 检查结果
echo "Results Summary:"
echo "----------------------------------------"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -ra PARTS <<< "$exp"
    # Trim空格
    GPU_ID=$(echo "${PARTS[0]}" | xargs)
    FUNC_NAME=$(echo "${PARTS[1]}" | xargs)
    
    # 查找日志文件
    LOG_FILES=$(ls ${LOG_DIR}/${GPU_ID}_*.log 2>/dev/null)
    
    if [ -n "$LOG_FILES" ]; then
        for LOG_FILE in $LOG_FILES; do
            COMPLETED=$(grep -c "All samples evaluated" "$LOG_FILE" 2>/dev/null || echo "0")
            FILENAME=$(basename "$LOG_FILE")
            echo "GPU ${GPU_ID}: ${FILENAME} - ${COMPLETED}/${NUM_REPEATS} repeats completed"
        done
    else
        echo "GPU ${GPU_ID}: No log file found"
    fi
done

echo ""
echo "Results: eval_scripts/eval_results/${MODEL##*/}/${TASK_NAME}/"
echo "Logs: ${LOG_DIR}/"

