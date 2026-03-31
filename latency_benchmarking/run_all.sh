#!/usr/bin/env bash
# Dynamic GPU scheduling: each job waits for its GPU to be free, then launches.
# All jobs run as background processes; results merged into results.csv at the end.

# =============================================================================
# Benchmark Configs
# =============================================================================
# Format: "method|bs|repeat|max_seq_len|gpu"
#
#   method      : cache implementation
#                   0 = Kitty KV cache
#                   1 = HuggingFace FP16 static cache
#                   2 = HuggingFace FP16 dynamic cache
#                   3 = HF INT4 quantized cache (quanto backend)
#   bs          : batch size
#   repeat      : number of timed runs (warmup is always 1)
#   max_seq_len : maximum sequence length (prompt + generation)
#   gpu         : GPU index to use (CUDA_VISIBLE_DEVICES)
#
BENCHMARK_CONFIGS=(
    # Kitty-Pro, BS = 8 16 32 64 96 128 192 256  (GPUs 0,1,2,3,6,7; avoid 4,5)
    "0|8  |3|8192|0"
    "0|16 |3|8192|1"
    "0|32 |3|8192|2"
    "0|64 |3|8192|3"
    "0|96 |3|8192|6"
    "0|128|3|8192|7"
    "0|192|3|8192|0"
    "0|256|3|8192|1"

    # HF_Static_FP16, BS = 8 16 32 64 96 128 192 256
    "1|8  |3|8192|0"
    "1|16 |3|8192|1"
    "1|32 |3|8192|2"
    "1|64 |3|8192|3"
    "1|96 |3|8192|6"
    "1|128|3|8192|7"
    "1|192|3|8192|0"
    "1|256|3|8192|1"

    # HF_Dynamic_FP16, BS = 8 16 32 64 96 128 192 256
    "2|8  |3|8192|0"
    "2|16 |3|8192|1"
    "2|32 |3|8192|2"
    "2|64 |3|8192|3"
    "2|96 |3|8192|6"
    "2|128|3|8192|7"
    "2|192|3|8192|0"
    "2|256|3|8192|1"

    # HF_KIVI_INT4, BS = 8 16 32 64 96 128 192 256
    "3|8  |3|8192|0"
    "3|16 |3|8192|1"
    "3|32 |3|8192|2"
    "3|64 |3|8192|3"
    "3|96 |3|8192|6"
    "3|128|3|8192|7"
    "3|192|3|8192|0"
    "3|256|3|8192|1"
)

# =============================================================================
# Environment
# =============================================================================
export TORCH_CUDA_ARCH_LIST="9.0"  # H100
export TRITON_CACHE_DIR="/data/jisenli2/.triton_cache"

SINGULARITY_IMG=/data/jisenli2/Kitty/build/kitty_v2.img
SINGULARITY_SIF=/data/jisenli2/Kitty/build/kitty_v2.sif
HF_CACHE=/data/shared/huggingface
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_kitty.py"
CSV_PATH="${SCRIPT_DIR}/results.csv"
LOG_DIR="${SCRIPT_DIR}/logs"

# GPU is considered free when used memory is below this threshold (MB)
GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-500}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-30}"

mkdir -p "${LOG_DIR}"

# =============================================================================
# GPU helpers (adapted from eval_quant.sh)
# =============================================================================
gpu_is_free() {
    local gpu_id="$1"
    local used_mb
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d '[:space:]')
    if [[ -z "$used_mb" ]]; then
        echo "[$(date '+%H:%M:%S')] WARNING: could not query GPU ${gpu_id}, assuming busy"
        return 1
    fi
    [[ "$used_mb" -lt "$GPU_FREE_MEM_MB" ]]
}

wait_for_gpu_free() {
    local gpu_id="$1" label="$2"
    local waited=0
    while ! gpu_is_free "$gpu_id"; do
        if [[ $((waited % 300)) -eq 0 ]]; then
            echo "[$(date '+%H:%M:%S')] Waiting for GPU ${gpu_id} to be free... ($((waited / 60))min) [$label]"
        fi
        sleep "$GPU_POLL_INTERVAL"
        waited=$((waited + GPU_POLL_INTERVAL))
    done
    [[ $waited -gt 0 ]] && echo "[$(date '+%H:%M:%S')] GPU ${gpu_id} now free after $((waited / 60))min"
}

# How long to sleep after launching a job before the next one:
#   - GPU overlap : wait for current job to finish, then COOLDOWN_OVERLAP_S
#   - No overlap  : COOLDOWN_NO_OVERLAP_S (let model load and register on GPU)
COOLDOWN_OVERLAP_S="${COOLDOWN_OVERLAP_S:-60}"
COOLDOWN_NO_OVERLAP_S="${COOLDOWN_NO_OVERLAP_S:-60}"

# =============================================================================
# Launch all jobs dynamically
# =============================================================================
all_pids=()
all_infos=()
all_job_csvs=()
declare -A exit_codes
any_error=0
N=${#BENCHMARK_CONFIGS[@]}

echo "Scheduling ${N} job(s). Results -> ${CSV_PATH}"
echo "========================================================"

for i in "${!BENCHMARK_CONFIGS[@]}"; do
    cfg="${BENCHMARK_CONFIGS[$i]}"
    IFS='|' read -r method bs repeat max_seq_len gpu <<< "$cfg"
    method="${method// /}"; bs="${bs// /}"
    repeat="${repeat// /}"; max_seq_len="${max_seq_len// /}"; gpu="${gpu// /}"

    label="method=${method} bs=${bs} GPU=${gpu}"
    log_file="${LOG_DIR}/job_${i}_gpu${gpu}_method${method}_bs${bs}.log"
    job_csv="${SCRIPT_DIR}/results_job${i}_gpu${gpu}_method${method}_bs${bs}.csv"

    entry_script="${BENCHMARK_SCRIPT}"
    [[ "$method" == "3" ]] && entry_script="${SCRIPT_DIR}/run_benchmark.py"

    # Wait until the target GPU is free before launching
    echo "[$(date '+%H:%M:%S')] [$((i+1))/${N}] Waiting for GPU ${gpu}: ${label}"
    wait_for_gpu_free "$gpu" "$label"

    echo "[$(date '+%H:%M:%S')] Launching job ${i}: ${label} -> $(basename "${log_file}")"

    CUDA_VISIBLE_DEVICES="${gpu}" singularity exec \
        --nv \
        --bind /data:/data \
        --overlay "${SINGULARITY_IMG}:ro" \
        "${SINGULARITY_SIF}" \
        env HF_HOME="${HF_CACHE}" CUDA_VISIBLE_DEVICES="${gpu}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
        python3 "${entry_script}" \
            --cache_implementation "${method}" \
            --batch_size            "${bs}" \
            --warmup_runs           1 \
            --repeat_runs           "${repeat}" \
            --max_seq_len           "${max_seq_len}" \
            --csv                   "${job_csv}" \
        > "${log_file}" 2>&1 &

    all_pids+=($!)
    exit_codes[$i]=-1
    all_infos+=("$label")
    all_job_csvs+=("$job_csv")

    # Check GPU overlap with next job
    next=$((i + 1))
    if [[ $next -lt $N ]]; then
        next_cfg="${BENCHMARK_CONFIGS[$next]}"
        next_gpu=$(echo "$next_cfg" | cut -d'|' -f5 | tr -d ' ')
        if [[ "$next_gpu" == "$gpu" ]]; then
            echo "[$(date '+%H:%M:%S')] Next job shares GPU ${gpu}, waiting for current job to finish..."
            wait "${all_pids[$i]}"
            exit_codes[$i]=$?
            echo "[$(date '+%H:%M:%S')] Cooling down ${COOLDOWN_OVERLAP_S}s for GPU memory to release..."
            sleep "${COOLDOWN_OVERLAP_S}"
        else
            echo "[$(date '+%H:%M:%S')] No GPU overlap, sleeping ${COOLDOWN_NO_OVERLAP_S}s for model load..."
            sleep "${COOLDOWN_NO_OVERLAP_S}"
        fi
    fi
done

# =============================================================================
# Wait for all jobs and collect exit codes
# =============================================================================
echo ""
echo "All jobs launched. Waiting for completion..."
echo "========================================================"

for i in "${!all_pids[@]}"; do
    if [[ "${exit_codes[$i]}" -eq -1 ]]; then
        wait "${all_pids[$i]}"
        exit_codes[$i]=$?
    fi
    if [[ "${exit_codes[$i]}" -ne 0 ]]; then
        echo "  ERROR: ${all_infos[$i]} exited with code ${exit_codes[$i]}"
        any_error=1
    else
        echo "  OK:    ${all_infos[$i]} done"
    fi
done

# =============================================================================
# Merge per-job CSVs into results.csv
# =============================================================================
echo ""
echo "Merging results into ${CSV_PATH} ..."

for job_csv in "${all_job_csvs[@]}"; do
    [[ ! -f "${job_csv}" ]] && continue
    if [[ ! -s "${CSV_PATH}" ]]; then
        cat "${job_csv}" >> "${CSV_PATH}"
    else
        tail -n +2 "${job_csv}" >> "${CSV_PATH}"
    fi
    rm "${job_csv}"
done

echo "========================================================"
if [[ $any_error -eq 0 ]]; then
    echo "All jobs completed successfully. Results -> ${CSV_PATH}"
else
    echo "Some jobs failed. Check logs in ${LOG_DIR}/"
fi
