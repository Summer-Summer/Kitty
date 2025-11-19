#!/bin/bash
#SBATCH --job-name=rock_kv_eval_multi_task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=200:00:00
#SBATCH --partition=batch
#SBATCH --output=log/slurm_%j.out
#SBATCH --error=log/slurm_%j.err
#SBATCH --nodelist=research-external-05  # 检查节点是否可用，不可用会一直在queue里等待，不会自动切换节点

# ============================================================================
# 配置区域 - 在这里修改参数
# ============================================================================

# 模型配置
# <MODEL>: "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B" "Qwen/Qwen3-4B"
export MODEL="meta-llama/Llama-3.1-8B-Instruct"

# 默认参数（可以在EXPERIMENTS中为每个任务覆盖）
export DEFAULT_NUM_REPEATS=3
export DEFAULT_BATCH_SIZE=32
export DEFAULT_MAX_NEW_TOKENS=4096  # 最大生成token数 (aime24/25建议32768, 其他4096)

# 输出路径配置
export RESULTS_DIR="./eval_results_10_19"  # 评估结果保存目录
export LOGS_DIR="eval_logs_10_19"          # 日志保存目录

# DEBUG模式 (设置为1或true则只运行1个repeat且只运行前3题，用于快速测试)
export DEBUG=0

# 环境变量
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME=/workspace/.cache/huggingface

# Apptainer路径
APPTAINER_SIF="$HOME/RoCK-KV/build/kchanboost.sif"
APPTAINER_IMG="$HOME/RoCK-KV/build/kchanboost.img"

# ============================================================================
# 实验配置 - 支持不同GPU运行不同Task
# ============================================================================
# 
# 可用的TASK_NAME: 
#   - "gsm8k_cot_llama"
#   - "minerva_math_algebra"
#   - "humaneval_instruct"
#   - "gpqa_diamond_cot_n_shot"
#   - "mmlu_flan_cot_fewshot"
#   - "aime24"
#   - "aime25"
#
# 格式: GPU | 函数 | task_name | label | sink | channel_sel | kbits | vbits | promote_bit | promote_ratio | [num_repeats] | [batch_size] | [max_new_tokens]
# 注意: num_repeats, batch_size, max_new_tokens 是可选的，如果不指定则使用DEFAULT值
#
declare -a EXPERIMENTS=(
    # GPU 0,1: gsm8k_cot_llama (sink 0 和 sink 32)
    "0    |  run_single_exp  |  gsm8k_cot_llama  |  Accuracy_Across_Ratios  |   0    |      2        |    3    |    2    |       4       |      0.0   |  3  |  32  |  4096"
    "1    |  run_single_exp  |  gsm8k_cot_llama  |  Accuracy_Across_Ratios  |   32   |      2        |    3    |    2    |       4       |      0.0   |  3  |  32  |  4096"
    # GPU 2,3: gpqa_diamond_cot_n_shot (sink 0 和 sink 32, 5 repeats)
    "2    |  run_single_exp  |  gpqa_diamond_cot_n_shot  |  Accuracy_Across_Ratios  |   0    |      2        |    3    |    2    |       4       |      0.0   |  5  |  32  |  4096"
    "3    |  run_single_exp  |  gpqa_diamond_cot_n_shot  |  Accuracy_Across_Ratios  |   32   |      2        |    3    |    2    |       4       |      0.0   |  5  |  32  |  4096"
    # GPU 4,5: humaneval_instruct (sink 0 和 sink 32, 10 repeats)
    "4    |  run_single_exp  |  humaneval_instruct  |  Accuracy_Across_Ratios  |   0    |      2        |    3    |    2    |       4       |      0.0   |  10  |  32  |  4096"
    "5    |  run_single_exp  |  humaneval_instruct  |  Accuracy_Across_Ratios  |   32   |      2        |    3    |    2    |       4       |      0.0   |  10  |  32  |  4096"
    # GPU 6,7: minerva_math_algebra (sink 0 和 sink 32, 3 repeats)
    "6    |  run_single_exp  |  minerva_math_algebra  |  Accuracy_Across_Ratios  |   0    |      2        |    3    |    2    |       4       |      0.0   |  3  |  32  |  4096"
    "7    |  run_single_exp  |  minerva_math_algebra  |  Accuracy_Across_Ratios  |   32   |      2        |    3    |    2    |       4       |      0.0   |  3  |  32  |  4096"
)

# ============================================================================
# 主程序
# ============================================================================

echo "=========================================="
echo "RoCK-KV Evaluation - Multi-Task Mode"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Model: $MODEL"
echo "Results Dir: $RESULTS_DIR"
echo "Logs Dir: $LOGS_DIR"
echo "=========================================="
echo ""

mkdir -p "$HOME/RoCK-KV/log"

# 存储后台进程PID
declare -a PIDS=()

echo "Starting parallel experiments in Apptainer..."
echo ""

# 启动任务
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -ra PARTS <<< "$exp"
    # Trim所有参数的空格
    TRIMMED_PARTS=()
    for part in "${PARTS[@]}"; do
        TRIMMED_PARTS+=($(echo "$part" | xargs))
    done
    
    GPU_ID="${TRIMMED_PARTS[0]}"
    FUNC_NAME="${TRIMMED_PARTS[1]}"
    TASK_NAME="${TRIMMED_PARTS[2]}"
    
    echo "GPU ${GPU_ID}: ${FUNC_NAME} - Task: ${TASK_NAME}"
    
    # 提取run_single_exp的参数
    if [ "$FUNC_NAME" = "run_single_exp" ]; then
        LABEL="${TRIMMED_PARTS[3]}"
        SINK="${TRIMMED_PARTS[4]}"
        CHANNEL="${TRIMMED_PARTS[5]}"
        KBITS="${TRIMMED_PARTS[6]}"
        VBITS="${TRIMMED_PARTS[7]}"
        PROMOTE_BIT="${TRIMMED_PARTS[8]}"
        PROMOTE_RATIO="${TRIMMED_PARTS[9]}"
        
        # 可选参数，如果没有则使用默认值
        NUM_REPEATS="${TRIMMED_PARTS[10]:-$DEFAULT_NUM_REPEATS}"
        BATCH_SIZE="${TRIMMED_PARTS[11]:-$DEFAULT_BATCH_SIZE}"
        MAX_NEW_TOKENS="${TRIMMED_PARTS[12]:-$DEFAULT_MAX_NEW_TOKENS}"
        
        echo "  - Config: sink=$SINK, channel=$CHANNEL, k=$KBITS, v=$VBITS, promote_bit=$PROMOTE_BIT, promote_ratio=$PROMOTE_RATIO"
        echo "  - Params: repeats=$NUM_REPEATS, batch_size=$BATCH_SIZE, max_tokens=$MAX_NEW_TOKENS"
    fi
    
    # 在Apptainer中运行（后台）
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
            export MAX_NEW_TOKENS=$MAX_NEW_TOKENS
            export RESULTS_DIR='$RESULTS_DIR'
            export DEBUG='$DEBUG'
            export GPUs='${GPU_ID}'
            export CUDA_VISIBLE_DEVICES='${GPU_ID}'
            export TORCH_CUDA_ARCH_LIST='9.0'
            export HF_HOME=/workspace/.cache/huggingface
            export TOKENIZERS_PARALLELISM=false
            export HF_DATASETS_TRUST_REMOTE_CODE=1
            
            # Source utils.sh
            source ./utils.sh
            
            # 设置带模型和任务的日志目录
            NODE_NAME_TMP=\$(hostname | grep -oP 'external-\K\d+' | head -1)
            if [ -n \"\$NODE_NAME_TMP\" ]; then
                NODE_NAME_TMP=\"node_\${NODE_NAME_TMP}\"
            else
                NODE_NAME_TMP=\$(hostname | cut -d. -f1)
            fi
            MODEL_SHORT=\$(get_model_shortname \"\$MODEL\")
            export LOG_BASE_DIR=\"/workspace/RoCK-KV/eval_scripts/\${LOGS_DIR}/\${NODE_NAME_TMP}/\${MODEL_SHORT}/\${TASK_NAME}\"
            mkdir -p \${LOG_BASE_DIR}
            
            # 调用函数
            if [ '${FUNC_NAME}' = 'run_hf_baseline' ]; then
                run_hf_baseline() {
                  local repeats=\${NUM_REPEATS:-1}
                  local debug_flag=''
                  if [ \"\${DEBUG}\" = '1' ] || [ \"\${DEBUG}\" = 'true' ]; then
                    repeats=1
                    debug_flag='--debug'
                  fi
                  
                  echo \"Launching \$TASK_NAME on GPUs \$GPUs\"
                  echo \"Huggingface baseline, k=16, v=16, num_repeats=\${repeats}\"

                  local model_short=\$(get_model_shortname \"\$MODEL\")
                  local log_name=\"\${GPUs//,/}_\${model_short}_\${TASK_NAME}_fp16.log\"

                  CUDA_VISIBLE_DEVICES=\$GPUs TOKENIZERS_PARALLELISM=false \
                  HF_DATASETS_TRUST_REMOTE_CODE=1 \
                    eval_rock_kv \$MODEL \
                      --task \$TASK_NAME \
                      --num_repeats \${repeats} \
                      --batch_size \${BATCH_SIZE:-1} \
                      --max_new_tokens \${MAX_NEW_TOKENS:-4096} \
                      --results_dir \${RESULTS_DIR:-./eval_results} \
                      \${debug_flag} > \${LOG_BASE_DIR}/\${log_name} 2>&1
                }
                run_hf_baseline
            else
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

                  local model_short=\$(get_model_shortname \"\$MODEL\")
                  local promote_ratio_str=\$(echo \"\$promote_ratio\" | sed 's/\\./_/g')
                  local log_name=\"\${GPUs//,/}_\${model_short}_\${TASK_NAME}_s\${sink}_sel\${channel}_k\${kbits}v\${vbits}_pro\${promote_ratio_str}.log\"

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
                      --max_new_tokens \${MAX_NEW_TOKENS:-4096} \
                      --results_dir \${RESULTS_DIR:-./eval_results} \
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

# 获取模型简称
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | sed 's/-/_/g')

# 检查结果（按task分组）
echo "Results Summary:"
echo "----------------------------------------"

# 收集所有不同的task
declare -A TASKS_MAP
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -ra PARTS <<< "$exp"
    GPU_ID=$(echo "${PARTS[0]}" | xargs)
    TASK=$(echo "${PARTS[2]}" | xargs)
    TASKS_MAP["$TASK"]=1
done

# 对每个task显示结果
for TASK in "${!TASKS_MAP[@]}"; do
    echo ""
    echo "Task: $TASK"
    echo "  Log dir: ${LOGS_DIR}/${NODE_NAME}/${MODEL_SHORT}/${TASK}/"
    
    LOG_DIR="$HOME/RoCK-KV/eval_scripts/${LOGS_DIR}/${NODE_NAME}/${MODEL_SHORT}/${TASK}"
    
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -ra PARTS <<< "$exp"
        GPU_ID=$(echo "${PARTS[0]}" | xargs)
        EXP_TASK=$(echo "${PARTS[2]}" | xargs)
        
        # 只显示匹配的task
        if [ "$EXP_TASK" = "$TASK" ]; then
            # 提取num_repeats（第11个参数，如果存在）
            TRIMMED_PARTS=()
            for part in "${PARTS[@]}"; do
                TRIMMED_PARTS+=($(echo "$part" | xargs))
            done
            EXP_REPEATS="${TRIMMED_PARTS[10]:-$DEFAULT_NUM_REPEATS}"
            
            # 查找日志文件
            LOG_FILES=$(ls ${LOG_DIR}/${GPU_ID}_*.log 2>/dev/null)
            
            if [ -n "$LOG_FILES" ]; then
                for LOG_FILE in $LOG_FILES; do
                    COMPLETED=$(grep -c "All samples evaluated" "$LOG_FILE" 2>/dev/null || echo "0")
                    FILENAME=$(basename "$LOG_FILE")
                    echo "  GPU ${GPU_ID}: ${FILENAME} - ${COMPLETED}/${EXP_REPEATS} repeats completed"
                done
            else
                echo "  GPU ${GPU_ID}: No log file found"
            fi
        fi
    done
done

echo ""
echo "Results base dir: eval_scripts/${RESULTS_DIR}/${MODEL##*/}/"
echo "Logs base dir: ${LOGS_DIR}/${NODE_NAME}/${MODEL_SHORT}/"
echo ""


