# Accuracy Simulation

This directory contains scripts for running large-scale **accuracy evaluations**
of Kitty and baseline KV-cache quantization schemes.  
All evaluations rely on HuggingFace `model.generate()` and our customized  
LM-Evaluation-Harness (`kitty` branch).

---

## Supported Models

- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-14B`
- `Qwen/Qwen3-32B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`

**Note:** Llama models require HuggingFace authentication. Login with:
```bash
huggingface-cli login
```
Or visit https://huggingface.co/settings/tokens to create an access token.

---

## Supported Tasks

| Task Name                     | Description                           |
|------------------------------|---------------------------------------|
| `gsm8k_cot_llama`            | Grade-school math reasoning           |
| `minerva_math_algebra`       | Competition-level algebra             |
| `humaneval_instruct`         | Code generation (Pass@1)              |
| `gpqa_diamond_cot_n_shot`    | Graduate-level science QA             |
| `aime24`                     | AIME 2024 Math (long-context)         |
| `aime25`                     | AIME 2025 Math (long-context)         |

---

## KV-Cache Variants Evaluated

| Variant      | Description                                                         |
|-------------|---------------------------------------------------------------------|
| **Kitty-Pro** | 2-bit KV + 4-bit channel-wise boost (K2V2 + promoted channels)     |
| **FP16 Baseline** | Standard HuggingFace FP16 static KV cache                     |
| **KIVI-2**   | Original KIVI (2-bit KV)                                            |
| **KIVI*-2**  | KIVI + Sink (keep first 32 tokens unquantized)                      |

---

## Running Evaluations

All evaluations are driven by:

- `accuracy_simulation/accuracy_eval.sh`
- helper functions in `accuracy_simulation/utils.sh` and related scripts.

Before running, make sure:

- You have installed `Kitty` and its dependencies as in the top-level README.
- You are inside the correct directory:
```
cd accuracy_simulation
```
---

### View Script Usage

To see the usage and argument description:

    ./accuracy_eval.sh

The script expects:

- `<MODEL>` — HF model name (e.g., `"Qwen/Qwen3-8B"`)
- `<TASK_NAME>` — benchmark name (e.g., `"aime24"`)
- `<GPUs>` — GPU list, e.g. `"0"` or `"0,1"`
- `[NUM_REPEATS]` — optional, default `1`
- `[BATCH_SIZE]` — optional, default `1`

If arguments are missing or invalid, the script prints usage help and exits.

---

### Example Commands

#### 1. Run with 10 repeats, batch_size = 2

    ./accuracy_eval.sh "Qwen/Qwen3-8B" "aime24" "0" "10" "2"

#### 2. Run with defaults (1 repeat, batch_size = 1)

    ./accuracy_eval.sh "Qwen/Qwen3-8B" "aime24" "0"

#### 3. Multi-GPU Evaluation

    ./accuracy_eval.sh "Qwen/Qwen3-32B" "aime25" "0,1" "10" "1"

