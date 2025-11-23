"""Utilities to run downstream evaluation for Kitty-KV models."""
import torch
from transformers import PreTrainedModel, AutoTokenizer
import lm_eval
#
import os
import gc
import time
import json
import matplotlib.pyplot as plt
from typing import Optional
import shutil


#
from kitty_sim import KittyKVCache
from .eval_helper import (
    load_completed_repeats,
    print_checkpoint_status,
    run_evaluation_repeats,
    generate_summary_statistics
)

########################################### Shared utility functions ###########################################
def print_gpu_memory():
    """Print GPU memory usage for all visible devices."""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def release_model_memory(model: PreTrainedModel):
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()

def visualize_kv_cache(kv, save_dir="kv_visualizations"):
    def visualize_kv_tensor(key, value, suffix="", cmap='viridis', save_dir=save_dir):
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot K
        im0 = axes[0].imshow(key, cmap=cmap, aspect='auto')
        axes[0].set_title(f'Layer {layer_} Head {head_} - K')
        axes[0].set_xlabel('Head Dimension')
        axes[0].set_ylabel('Sequence Length')
        fig.colorbar(im0, ax=axes[0])
        # Plot V
        im1 = axes[1].imshow(value, cmap=cmap, aspect='auto')
        axes[1].set_title(f'Layer {layer_} Head {head_} - V')
        axes[1].set_xlabel('Head Dimension')
        axes[1].set_ylabel('Sequence Length')
        fig.colorbar(im1, ax=axes[1])
        #
        plt.tight_layout()
        plt.savefig(f'{save_dir}/layer{layer_}_head{head_}_kvcache_{suffix}.png')
        plt.close()
    #
    for layer_ in range(len(kv)):
        print("Visualizing KV Cache: Layer_{}".format(layer_))
        key = kv[layer_][0].cpu().numpy()
        value = kv[layer_][1].cpu().numpy()
        assert key.shape == value.shape
        b, h, s, d = key.shape
        assert b == 1
        for head_ in range(h):
            visualize_kv_tensor(key[0, head_], value[0, head_], "magnitute")

########################################### Used by cli/eval_kitty.py ###########################################
@torch.no_grad()
def eval_model_downstream(model: PreTrainedModel, task: str, ModelName, fileName, DEBUG=False, kv_cache: Optional[KittyKVCache] = None, num_repeats: int = None, batch_size: int = 8, max_new_tokens: int = 4096, results_dir: str = "./eval_results"):
    """
    Evaluate model on downstream tasks with support for multiple repeats and checkpoint resumption.
    
    Args:
        num_repeats: Number of times to repeat the evaluation (for sampling-based tasks).
                     If None, will try to read from task's YAML config. If not found, defaults to 1.
        batch_size: Batch size for inference (default: 8).
    """
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    
    lm = HFLM(model, batch_size=batch_size)
    
    # If num_repeats is not specified, try to read from task config
    if num_repeats is None:
        try:
            task_manager = TaskManager()
            task_dict = task_manager.load_task_or_group([task])
            task_obj = task_dict[task][0] if isinstance(task_dict[task], list) else task_dict[task]
            num_repeats = getattr(task_obj.config, 'repeats', 1)
            print(f"[Info] Using repeats={num_repeats} from task YAML config")
        except Exception as e:
            print(f"[Warning] Could not read repeats from task config: {e}. Using default repeats=1")
            num_repeats = 1

    model_configs = {}
    model_configs["ModelArch"] = model.__class__.__name__
    model_configs["ModelPath"] = model.config._name_or_path
    if kv_cache is not None:
        model_configs["cache_implementation"] = kv_cache.cache_implementation
        model_configs["kbits"] = kv_cache.kbits
        model_configs["vbits"] = kv_cache.vbits
        model_configs["group_size"] = kv_cache.group_size
        model_configs["sink_length"] = kv_cache.sink_length
        model_configs["buffer_length"] = kv_cache.buffer_length
        model_configs["promote_ratio"] = kv_cache.promote_ratio
        model_configs["promote_bit"] = kv_cache.promote_bit
        model_configs["channel_selection"] = kv_cache.channel_selection
        model_configs["VCache_BitDecoding"] = kv_cache.VCache_BitDecoding

    # Set the number of shots for different tasks
    few_shot_dict = {"mmlu": 4, "gsm8k": 8, "gpqa": 5, "math": 4, "bbh": 3}
    for key, value in few_shot_dict.items():
        if key in task:
            num_fewshot = value
            break
    else:
        num_fewshot = 0

    # Model-specific stop words
    model_name = model.config._name_or_path.lower()
    
    if "qwen" in model_name:
        # Qwen models
        stop_words = [
            "<|endoftext|>",       # Qwen-3  (151643)   real EOS
            "<|im_end|>",          # Qwen-3  (151645)   message ended
        ]
    elif "llama" in model_name:
        # LLaMA models
        stop_words = [
            "<|end_of_text|>",     # Llama-3 (128001)   real EOS
            "<|eot_id|>",          # Llama-3 (128009)   turn ended
            "<|end_header_id|>",   # Llama-3 (128008)   head ended
        ]
    else:
        # Fallback: include all common stop tokens
        stop_words = [
            "<|end_of_text|>",     # Llama-3
            "<|eot_id|>",          # Llama-3
            "<|end_header_id|>",   # Llama-3
            "<|endoftext|>",       # Qwen-3
            "<|im_end|>",          # Qwen-3
        ]
    # Set the stop words for different tasks
    stop_words_dict = {
        "gsm8k":     ["Given the following problem"],
        "math":      ["Problem:"],
        "gpqa":      ["Question:"],
        "humaneval": ["\n```"],
        "aime":      [
            "Given the following problem",
            # "</s>"
        ],
    }
    for key, value in stop_words_dict.items():
        if key in task:
            stop_words.extend(value)
            break
    #
    # For thinking mode (Qwen3-4B), use sampling parameters from YAML
    # Otherwise use greedy decoding for deterministic evaluation
    
    # max_new_tokens is now passed as a parameter from command line
    # Default: 4096, but can be overridden via --max_new_tokens argument
    
    gen_kwargs = {
            "past_key_values": kv_cache,  # Use KittyKV cache if provided
            "max_new_tokens": max_new_tokens,
            "max_length": None,
            # Commenting out these to allow YAML generation_kwargs to take effect
            # "do_sample": False,    # Disable sampling for deterministic evaluation
            # "temperature": None,
            # "top_p": None,
            # "top_k": None,         # Disable top-k sampling
            "until": stop_words,  # Stop words for different tasks
        }

    # Enable code evaluation for humaneval task
    if "humaneval" in task:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    # For DEBUG mode, limit the number of samples
    limit = None
    if DEBUG:
        limit = 8

    print("=" * 80)
    print("Evaluating model accuracy on downstream tasks...")
    print("ModelName: ", model_configs["ModelPath"])
    print("Task: ", task)
    print("Eval_Configs: ", fileName)
    print("Num_Fewshot: ", num_fewshot)
    print("Num_Repeats: ", num_repeats)
    print("Batch_Size: ", batch_size)
    print("GPU Memory:")
    print_gpu_memory()
    print("=" * 80)

    # Prepare output directory (results_dir can be configured via command line)
    FileDir = "{}/{}/{}".format(results_dir, ModelName, task)
    if not os.path.exists(FileDir):
        os.makedirs(FileDir, exist_ok=True)
    
    # Check which repeats are already completed (for checkpoint resumption)
    completed_repeats, all_results = load_completed_repeats(FileDir, fileName, num_repeats)
    all_completed = print_checkpoint_status(completed_repeats, num_repeats)
    
    # Run evaluations for remaining repeats (skip if all completed)
    if not all_completed:
        all_results = run_evaluation_repeats(
            lm=lm,
            model=model,
            task=task,
            num_fewshot=num_fewshot,
            limit=limit,
            gen_kwargs=gen_kwargs,
            model_configs=model_configs,
            kv_cache=kv_cache,
            file_dir=FileDir,
            file_name=fileName,
            num_repeats=num_repeats,
            completed_repeats=completed_repeats,
            all_results=all_results,
            batch_size=batch_size
        )
    
    # Generate summary statistics across all repeats
    print(f"\n{'='*80}")
    print("Generating summary statistics...")
    print(f"{'='*80}\n")
    
    summary = generate_summary_statistics(all_results, task, model_configs, num_repeats)
    
    # Create subdirectory for this configuration and save summary file
    config_dir = "{}/{}".format(FileDir, fileName)
    os.makedirs(config_dir, exist_ok=True)
    summary_file = "{}/{}_summary.json".format(config_dir, fileName)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[Saved] Summary statistics: {summary_file}")


########################################### Used by cli/eval_hf_kv.py ###########################################
@torch.no_grad()
def eval_model_downstream_hf(
    model: PreTrainedModel, 
    task: str, 
    ModelName: str,
    fileName: str, 
    DEBUG: bool = False, 
    hf_cache_config: Optional[dict] = None,
    num_repeats: int = None, 
    batch_size: int = 8, 
    max_new_tokens: int = 4096, 
    results_dir: str = "./eval_results"
):
    """
    Evaluate model on downstream tasks using HuggingFace native quantized KV Cache.
    
    Args:
        model: PreTrainedModel to evaluate
        task: Task name (e.g., "gsm8k_cot_llama")
        ModelName: Model short name for file paths
        fileName: Configuration identifier for file naming
        DEBUG: Debug mode (limit=8)
        hf_cache_config: HF cache configuration dict with structure:
            {
                "cache_implementation": "quantized",
                "cache_config": {
                    "backend": "HQQ",
                    "nbits": 4,
                    "axis_key": 1,
                    "axis_value": 1
                }
            }
        num_repeats: Number of evaluation repeats
        batch_size: Batch size for inference
        max_new_tokens: Maximum number of new tokens to generate
        results_dir: Directory to save results
    """
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    
    lm = HFLM(model, batch_size=batch_size)
    
    # If num_repeats is not specified, try to read from task config
    if num_repeats is None:
        try:
            task_manager = TaskManager()
            task_dict = task_manager.load_task_or_group([task])
            task_obj = task_dict[task][0] if isinstance(task_dict[task], list) else task_dict[task]
            num_repeats = getattr(task_obj.config, 'repeats', 1)
            print(f"[Info] Using repeats={num_repeats} from task YAML config")
        except Exception as e:
            print(f"[Warning] Could not read repeats from task config: {e}. Using default repeats=1")
            num_repeats = 1

    # Build model_configs
    model_configs = {}
    model_configs["ModelArch"] = model.__class__.__name__
    model_configs["ModelPath"] = model.config._name_or_path
    model_configs["cache_type"] = "hf_quantized"
    
    if hf_cache_config is not None:
        model_configs["cache_implementation"] = hf_cache_config["cache_implementation"]
        model_configs["cache_backend"] = hf_cache_config["cache_config"]["backend"]
        model_configs["cache_nbits"] = hf_cache_config["cache_config"]["nbits"]
        model_configs["cache_axis_key"] = hf_cache_config["cache_config"]["axis_key"]
        model_configs["cache_axis_value"] = hf_cache_config["cache_config"]["axis_value"]

    # Set the number of shots for different tasks
    few_shot_dict = {"mmlu": 4, "gsm8k": 8, "gpqa": 5, "math": 4, "bbh": 3}
    for key, value in few_shot_dict.items():
        if key in task:
            num_fewshot = value
            break
    else:
        num_fewshot = 0

    # Model-specific stop words
    model_name = model.config._name_or_path.lower()
    
    if "qwen" in model_name:
        # Qwen models
        stop_words = [
            "<|endoftext|>",       # Qwen-3  (151643)   real EOS
            "<|im_end|>",          # Qwen-3  (151645)   message ended
        ]
    elif "llama" in model_name:
        # LLaMA models
        stop_words = [
            "<|end_of_text|>",     # Llama-3 (128001)   real EOS
            "<|eot_id|>",          # Llama-3 (128009)   turn ended
            "<|end_header_id|>",   # Llama-3 (128008)   head ended
        ]
    else:
        # Fallback: include all common stop tokens
        stop_words = [
            "<|end_of_text|>",     # Llama-3
            "<|eot_id|>",          # Llama-3
            "<|end_header_id|>",   # Llama-3
            "<|endoftext|>",       # Qwen-3
            "<|im_end|>",          # Qwen-3
        ]
    
    # Set the stop words for different tasks
    stop_words_dict = {
        "gsm8k":     ["Given the following problem"],
        "math":      ["Problem:"],
        "gpqa":      ["Question:"],
        "humaneval": ["\n```"],
        "aime":      ["Given the following problem"],
    }
    for key, value in stop_words_dict.items():
        if key in task:
            stop_words.extend(value)
            break
    
    # Build gen_kwargs with HF cache configuration
    gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "max_length": None,
            "until": stop_words,
        }
    
    # Add HF cache configuration
    if hf_cache_config is not None:
        gen_kwargs.update(hf_cache_config)
        # This adds:
        # gen_kwargs["cache_implementation"] = "quantized"
        # gen_kwargs["cache_config"] = {"backend": "HQQ", "nbits": 4, ...}

    # Enable code evaluation for humaneval task
    if "humaneval" in task:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    # For DEBUG mode, limit the number of samples
    limit = None
    if DEBUG:
        limit = 8

    print("=" * 80)
    print("Evaluating model accuracy on downstream tasks...")
    print("ModelName: ", model_configs["ModelPath"])
    print("Task: ", task)
    print("Eval_Configs: ", fileName)
    print("Cache_Type: HF Quantized KV Cache")
    if hf_cache_config:
        print(f"  Backend: {hf_cache_config['cache_config']['backend']}")
        print(f"  Quantization: {hf_cache_config['cache_config']['nbits']}-bit")
        print(f"  Axis: key={hf_cache_config['cache_config']['axis_key']}, value={hf_cache_config['cache_config']['axis_value']}")
    print("Num_Fewshot: ", num_fewshot)
    print("Num_Repeats: ", num_repeats)
    print("Batch_Size: ", batch_size)
    print("GPU Memory:")
    print_gpu_memory()
    print("=" * 80)

    # Prepare output directory
    FileDir = "{}/{}/{}".format(results_dir, ModelName, task)
    if not os.path.exists(FileDir):
        os.makedirs(FileDir, exist_ok=True)
    
    # Check which repeats are already completed (for checkpoint resumption)
    completed_repeats, all_results = load_completed_repeats(FileDir, fileName, num_repeats)
    all_completed = print_checkpoint_status(completed_repeats, num_repeats)
    
    # Run evaluations for remaining repeats (skip if all completed)
    if not all_completed:
        all_results = run_evaluation_repeats(
            lm=lm,
            model=model,
            task=task,
            num_fewshot=num_fewshot,
            limit=limit,
            gen_kwargs=gen_kwargs,
            model_configs=model_configs,
            kv_cache=None,  # For HF cache, we pass config via gen_kwargs, not past_key_values
            file_dir=FileDir,
            file_name=fileName,
            num_repeats=num_repeats,
            completed_repeats=completed_repeats,
            all_results=all_results,
            batch_size=batch_size
        )
    
    # Generate summary statistics across all repeats
    print(f"\n{'='*80}")
    print("Generating summary statistics...")
    print(f"{'='*80}\n")
    
    summary = generate_summary_statistics(all_results, task, model_configs, num_repeats)
    
    # Create subdirectory for this configuration and save summary file
    config_dir = "{}/{}".format(FileDir, fileName)
    os.makedirs(config_dir, exist_ok=True)
    summary_file = "{}/{}_summary.json".format(config_dir, fileName)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[Saved] Summary statistics: {summary_file}")