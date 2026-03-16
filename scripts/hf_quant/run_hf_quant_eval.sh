#!/bin/bash
#SBATCH --job-name=hf_quant_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=200:00:00
#SBATCH --partition=batch
#SBATCH --output=log/slurm_%j.out
#SBATCH --error=log/slurm_%j.err
#SBATCH --nodelist=research-external-03  # 检查节点是否可用，不可用会一直在queue里等待，不会自动切换节点

# ============================================================================
# HuggingFace Quantized KV Cache 评估脚本 - Multi-Task 版本
# ============================================================================
# 
# 用途: 评估 HuggingFace 原生的量化 KV Cache (HQQ/Quanto backend)
# 支持在不同GPU上同时运行不同任务
#
# ============================================================================

# ============================================================================
# 配置区域 - 在这里修改参数
# ============================================================================

# 模型配置
# <MODEL>: "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B" "Qwen/Qwen3-4B"
export MODEL="Qwen/Qwen3-8B"

# 默认参数（可以在EXPERIMENTS中为每个任务覆盖）
export DEFAULT_NUM_REPEATS=3
export DEFAULT_BATCH_SIZE=32
export DEFAULT_MAX_NEW_TOKENS=4096  # 最大生成token数 (aime24/25建议32768, 其他4096)

# 输出路径配置
export RESULTS_DIR="./eval_results_hf_quant"  # 评估结果保存目录
export LOGS_DIR="eval_logs_hf_quant"          # 日志保存目录

# DEBUG模式 (设置为1或true则只运行1个repeat且只运行前8题，用于快速测试)
export DEBUG=0

# 环境变量
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 GPU (Ampere架构)
export HF_HOME=/workspace/.cache/huggingface

# Apptainer路径
APPTAINER_SIF="$HOME/RoCK-KV/build/kchanboost.sif"
APPTAINER_IMG="$HOME/RoCK-KV/build/hf_kv.img"  # 使用 hf_kv.img (已安装 optimum-quanto)

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
# 格式: GPU | 函数 | task_name | backend | nbits | axis_key | axis_value | [num_repeats] | [batch_size] | [max_new_tokens]
# 注意: num_repeats, batch_size, max_new_tokens 是可选的，如果不指定则使用DEFAULT值
#
# backend 选项:
#   - "HQQ": 使用 HQQ backend (推荐 axis_key=1, axis_value=1)
#   - "quanto": 使用 Quanto backend (需要安装 optimum-quanto, 推荐 axis_key=0, axis_value=0)
#
# nbits 选项: 2, 4
#
declare -a EXPERIMENTS=(
    # GPU 1: gsm8k_cot_llama - Quanto INT4 (3 repeats, batch_size=32, axis=0)
    "0    |  run_hf_quantized_exp  |  gsm8k_cot_llama  |  quanto  |  4  |  0  |  0  |  3  |  32  |  4096"
    
    # GPU 2: gpqa_diamond_cot_n_shot - Quanto INT4 (5 repeats, batch_size=32, axis=0)
    "1    |  run_hf_quantized_exp  |  gpqa_diamond_cot_n_shot  |  quanto  |  4  |  0  |  0  |  5  |  32  |  4096"
    
    # GPU 3: humaneval_instruct - Quanto INT4 (10 repeats, batch_size=32, axis=0)
    "2    |  run_hf_quantized_exp  |  humaneval_instruct  |  quanto  |  4  |  0  |  0  |  10  |  32  |  4096"
    
    # GPU 4: minerva_math_algebra - Quanto INT4 (3 repeats, batch_size=32, axis=0)
    "3    |  run_hf_quantized_exp  |  minerva_math_algebra  |  quanto  |  4  |  0  |  0  |  3  |  32  |  4096"
)

# ============================================================================
# 主程序
# ============================================================================

echo "=========================================="
echo "HF Quantized KV Cache Evaluation - Multi-Task Mode"
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
    
    # 提取参数（针对不同的函数）
    if [ "$FUNC_NAME" = "run_hf_baseline" ]; then
        # run_hf_baseline 的可选参数
        NUM_REPEATS="${TRIMMED_PARTS[3]:-$DEFAULT_NUM_REPEATS}"
        BATCH_SIZE="${TRIMMED_PARTS[4]:-$DEFAULT_BATCH_SIZE}"
        MAX_NEW_TOKENS="${TRIMMED_PARTS[5]:-$DEFAULT_MAX_NEW_TOKENS}"
        
        echo "  - FP16 Baseline"
        echo "  - Params: repeats=$NUM_REPEATS, batch_size=$BATCH_SIZE, max_tokens=$MAX_NEW_TOKENS"
    elif [ "$FUNC_NAME" = "run_hf_quantized_exp" ]; then
        BACKEND="${TRIMMED_PARTS[3]}"
        NBITS="${TRIMMED_PARTS[4]}"
        AXIS_KEY="${TRIMMED_PARTS[5]}"
        AXIS_VALUE="${TRIMMED_PARTS[6]}"
        
        # 可选参数，如果没有则使用默认值
        NUM_REPEATS="${TRIMMED_PARTS[7]:-$DEFAULT_NUM_REPEATS}"
        BATCH_SIZE="${TRIMMED_PARTS[8]:-$DEFAULT_BATCH_SIZE}"
        MAX_NEW_TOKENS="${TRIMMED_PARTS[9]:-$DEFAULT_MAX_NEW_TOKENS}"
        
        echo "  - Config: backend=$BACKEND, nbits=$NBITS, axis_key=$AXIS_KEY, axis_value=$AXIS_VALUE"
        echo "  - Params: repeats=$NUM_REPEATS, batch_size=$BATCH_SIZE, max_tokens=$MAX_NEW_TOKENS"
    fi
    
    # 根据 GPU_ID 选择对应的 image 副本（每个 GPU 有独立的可写 image）
    GPU_IMG="$HOME/RoCK-KV/build/hf_kv_${GPU_ID}.img"
    echo "  - Using image: hf_kv_${GPU_ID}.img"
    
    # 在Apptainer中运行（后台）
    apptainer exec --nv \
        --bind $HOME:/workspace \
        --bind /data:/data \
        --overlay "$GPU_IMG":rw \
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
            export TORCH_CUDA_ARCH_LIST='8.0'
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
                  echo \"Huggingface FP16 baseline (for comparison), num_repeats=\${repeats}\"

                  local model_short=\$(get_model_shortname \"\$MODEL\")
                  local log_name=\"\${GPUs//,/}_\${model_short}_\${TASK_NAME}_fp16.log\"

                  CUDA_VISIBLE_DEVICES=\$GPUs TOKENIZERS_PARALLELISM=false \
                  HF_DATASETS_TRUST_REMOTE_CODE=1 \
                    \$HOME/.local/bin/eval_rock_kv \$MODEL \
                      --task \$TASK_NAME \
                      --num_repeats \${repeats} \
                      --batch_size \${BATCH_SIZE:-1} \
                      --max_new_tokens \${MAX_NEW_TOKENS:-4096} \
                      --results_dir \${RESULTS_DIR:-./eval_results} \
                      \${debug_flag} > \${LOG_BASE_DIR}/\${log_name} 2>&1
                }
                run_hf_baseline
            elif [ '${FUNC_NAME}' = 'run_hf_quantized_exp' ]; then
                run_hf_quantized_exp() {
                  local backend=\$1
                  local nbits=\$2
                  local axis_key=\$3
                  local axis_value=\$4
                  
                  local repeats=\${NUM_REPEATS:-1}
                  local debug_flag=''
                  if [ \"\${DEBUG}\" = '1' ] || [ \"\${DEBUG}\" = 'true' ]; then
                    repeats=1
                    debug_flag='--debug'
                  fi
                  
                  echo \"Launching \$TASK_NAME on GPUs \$GPUs\"
                  echo \"HF Quantized KV Cache: backend=\$backend, nbits=\$nbits, axis_key=\$axis_key, axis_value=\$axis_value, num_repeats=\${repeats}\"

                  local model_short=\$(get_model_shortname \"\$MODEL\")
                  local log_name=\"\${GPUs//,/}_\${model_short}_\${TASK_NAME}_hf_\${backend,,}_int\${nbits}_axis\${axis_key}.log\"

                  CUDA_VISIBLE_DEVICES=\$GPUs TOKENIZERS_PARALLELISM=false \
                  HF_DATASETS_TRUST_REMOTE_CODE=1 \
                    \$HOME/.local/bin/eval_hf_kv \$MODEL \
                      --task \$TASK_NAME \
                      --hf_cache_backend \$backend \
                      --hf_cache_nbits \$nbits \
                      --hf_axis_key \$axis_key \
                      --hf_axis_value \$axis_value \
                      --num_repeats \${repeats} \
                      --batch_size \${BATCH_SIZE:-1} \
                      --max_new_tokens \${MAX_NEW_TOKENS:-4096} \
                      --results_dir \${RESULTS_DIR:-./eval_results} \
                      \${debug_flag} > \${LOG_BASE_DIR}/\${log_name} 2>&1
                }
                run_hf_quantized_exp '${BACKEND}' ${NBITS} ${AXIS_KEY} ${AXIS_VALUE}
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
        FUNC_NAME=$(echo "${PARTS[1]}" | xargs)
        EXP_TASK=$(echo "${PARTS[2]}" | xargs)
        
        # 只显示匹配的task
        if [ "$EXP_TASK" = "$TASK" ]; then
            # 提取num_repeats（根据函数类型确定位置）
            TRIMMED_PARTS=()
            for part in "${PARTS[@]}"; do
                TRIMMED_PARTS+=($(echo "$part" | xargs))
            done
            
            if [ "$FUNC_NAME" = "run_hf_baseline" ]; then
                EXP_REPEATS="${TRIMMED_PARTS[3]:-$DEFAULT_NUM_REPEATS}"
            else
                # run_hf_quantized_exp: num_repeats 在第8个位置（索引7）
                EXP_REPEATS="${TRIMMED_PARTS[7]:-$DEFAULT_NUM_REPEATS}"
            fi
            
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
echo "=========================================="
echo "HF Quantized KV Cache Evaluation Complete"
echo "=========================================="
