#!/bin/bash
#SBATCH --job-name=rock_kv_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=48:00:00
#SBATCH --partition=batch
#SBATCH --output=log/slurm_%j.out
#SBATCH --error=log/slurm_%j.err
#SBATCH --nodelist=research-external-12  # 检查节点是否可用，不可用会一直在queue里等待，不会自动切换节点

# ============================================================================
# 配置区域 - 在这里修改参数
# ============================================================================

# 模型和任务配置
# <MODEL>: "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B" "Qwen/Qwen3-4B"
# <TASK_NAME>: "gsm8k_cot_llama" "minerva_math_algebra" "humaneval_instruct" "gpqa_diamond_cot_n_shot" "mmlu_flan_cot_fewshot" "aime24" "aime25"
export MODEL="Qwen/Qwen3-8B"
export TASK_NAME="aime24"
export NUM_REPEATS=10
export BATCH_SIZE=6

# DEBUG模式 (设置为1或true则只运行1个repeat且只运行前3题，用于快速测试)
export DEBUG=0

# 环境变量
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME=/workspace/.cache/huggingface

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

# GPU     |  函数             |  label                   |  sink  |  channel_sel  |  kbits  |  vbits  |  promote_bit  |  promote_ratio
declare -a EXPERIMENTS=(
    # AIME24 Baseline
    "0    |  run_hf_baseline"
    # AIME24 KIVI K4V4
    "1    |  run_single_exp  |  Accuracy_Across_Ratios  |   0    |      2        |    4    |    4    |       4       |      0.0"
    # AIME24 KIVI K2V2
    "2    |  run_single_exp  |  Accuracy_Across_Ratios  |   0    |      2        |    2    |    2    |       4       |      0.0"
    # AIME24 sinkKIVI K4V2
    "3    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    4    |    2    |       4       |      0.0"
    # AIME24 sinkKIVI K2V4
    "4    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    2    |    4    |       4       |      0.0"
    # AIME24 sinkKIVI K2V2
    "5    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    2    |    2    |       4       |      0.0"
    # AIME24 sinkKIVI K2.2V2
    "6    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    2    |    2    |       4       |      0.1"
    # AIME24 sinkKIVI K2.4V2
    "7    |  run_single_exp  |  Accuracy_Across_Ratios  |  32    |      2        |    2    |    2    |       4       |      0.2"
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

# 启动8个GPU任务
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -ra PARTS <<< "$exp"
    # Trim所有参数的空格
    TRIMMED_PARTS=()
    for part in "${PARTS[@]}"; do
        TRIMMED_PARTS+=($(echo "$part" | xargs))
    done
    
    GPU_ID="${TRIMMED_PARTS[0]}"
    FUNC_NAME="${TRIMMED_PARTS[1]}"
    
    echo "GPU ${GPU_ID}: ${FUNC_NAME} ${TRIMMED_PARTS[@]:2}"
    
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
            export DEBUG='$DEBUG'
            export GPUs='${GPU_ID}'
            export CUDA_VISIBLE_DEVICES='${GPU_ID}'
            export TORCH_CUDA_ARCH_LIST='9.0'
            export HF_HOME=/workspace/.cache/huggingface
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

