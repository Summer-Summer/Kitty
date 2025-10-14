"""Helper functions for downstream evaluation."""

import os
import gc
import json
import copy
import torch
import numpy as np
import lm_eval
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any, Optional
from transformers import PreTrainedModel


########################################### Checkpoint Management ###########################################

def load_completed_repeats(file_dir: str, file_name: str, num_repeats: int) -> Tuple[List[int], List[Dict]]:
    """
    Check which repeats are already completed and load their results.
    Only loads minimal data (tasks and repeat_idx) to save memory.
    
    Args:
        file_dir: Directory containing repeat result files
        file_name: Base filename for repeat results
        num_repeats: Total number of repeats expected
    
    Returns:
        Tuple of (list of completed repeat indices, list of minimal results for summary)
    """
    completed_repeats = []
    all_results = []
    
    for repeat_idx in range(num_repeats):
        repeat_file = f"{file_dir}/{file_name}_repeat_{repeat_idx}.json"
        if os.path.exists(repeat_file):
            print(f"[Checkpoint] Found existing result for repeat {repeat_idx}: {repeat_file}")
            completed_repeats.append(repeat_idx)
            # Only load minimal data needed for summary (not samples)
            with open(repeat_file, "r") as f:
                full_result = json.load(f)
                minimal_result = {
                    "repeat_idx": full_result["repeat_idx"],
                    "tasks": full_result["tasks"]
                }
                all_results.append(minimal_result)
                del full_result  # Free memory immediately
    
    return completed_repeats, all_results


def print_checkpoint_status(completed_repeats: List[int], num_repeats: int) -> bool:
    """
    Print checkpoint status and return whether all repeats are completed.
    
    Args:
        completed_repeats: List of completed repeat indices
        num_repeats: Total number of repeats expected
    
    Returns:
        True if all repeats are completed, False otherwise
    """
    if len(completed_repeats) == num_repeats:
        print(f"[Checkpoint] All {num_repeats} repeats already completed. Skipping evaluation.")
        return True
    else:
        print(f"[Checkpoint] {len(completed_repeats)}/{num_repeats} repeats completed. "
              f"Continuing from repeat {len(completed_repeats)}.")
        return False


########################################### Output JSON Builder ###########################################

def build_eval_output(
    results: Dict,
    task: str,
    repeat_idx: int,
    model: PreTrainedModel,
    model_configs: Dict
) -> Dict:
    """
    Build output dictionary in lm_eval format, following the reference code style.
    
    Args:
        results: Results dictionary from lm_eval.simple_evaluate
        task: Task name
        repeat_idx: Current repeat index
        model: The model being evaluated
        model_configs: Model configuration dictionary
    
    Returns:
        Dictionary in lm_eval output format
    """
    cdt_timezone = timezone(timedelta(hours=-5))  # CDT: UTC-5
    
    output = {}
    output["This file is generated_at"] = datetime.now(cdt_timezone).isoformat()
    output["repeat_idx"] = repeat_idx
    output["tasks"] = results['results']
    output["model_configs"] = model_configs
    
    # Only deep copy eval_configs to avoid modifying the original (important for multiple repeats)
    # This is much lighter than copying the entire results dict
    output["eval_configs"] = copy.deepcopy(results["config"])
    
    # Dealing with corner cases, where model configs can not be serialized.
    output["eval_configs"]["model_dtype"] = str(model.dtype)
    if output["eval_configs"]["gen_kwargs"]["past_key_values"] is not None:
        output["eval_configs"]["gen_kwargs"]["past_key_values"] = output["eval_configs"]["gen_kwargs"]["past_key_values"].cache_implementation
    
    if "samples" in results:
        # For samples, we need to modify the arguments structure, but we can do this more efficiently
        # by only copying what we need to modify
        output["samples"] = {}
        
        for task_name, task_samples in results["samples"].items():
            output["samples"][task_name] = []
            
            # Extract shared gen_kwargs from the first sample (only once)
            if task_samples and "arguments_from_samples" not in output["eval_configs"]:
                first_sample = task_samples[0]
                arguments_from_samples = copy.deepcopy(first_sample["arguments"][0][1])
                if arguments_from_samples["past_key_values"] is not None:
                    arguments_from_samples["past_key_values"] = arguments_from_samples["past_key_values"].cache_implementation
                output["eval_configs"]["arguments_from_samples"] = arguments_from_samples
            
            # Process each sample - only copy the fields we need
            for sample in task_samples:
                sample_copy = {
                    "doc_id": sample.get("doc_id"),
                    "doc": sample.get("doc"),
                    "target": sample.get("target"),
                    "resps": sample.get("resps"),
                    "filtered_resps": sample.get("filtered_resps"),
                    "arguments": sample["arguments"][0][0],  # Only keep the first part (without gen_kwargs)
                }
                # Add other fields if they exist
                for key in ["doc_hash", "prompt", "metrics"]:
                    if key in sample:
                        sample_copy[key] = sample[key]
                
                output["samples"][task_name].append(sample_copy)
    
    return output


########################################### Evaluation Loop ###########################################

def run_evaluation_repeats(
    lm,
    model: PreTrainedModel,
    task: str,
    num_fewshot: int,
    limit: Optional[int],
    gen_kwargs: Dict,
    model_configs: Dict,
    kv_cache: Optional[Any],
    file_dir: str,
    file_name: str,
    num_repeats: int,
    completed_repeats: List[int],
    all_results: List[Dict],
    batch_size: int = 8,
    base_random_seed: int = 0,
    base_numpy_seed: int = 1234,
    base_torch_seed: int = 1234,
    base_fewshot_seed: int = 1234
) -> List[Dict]:
    """
    Run evaluation for remaining repeats and save results.
    
    Args:
        lm: Language model instance (lm_eval model)
        model: The PreTrainedModel being evaluated
        task: Task name (e.g., "aime24")
        num_fewshot: Number of few-shot examples
        limit: Number of samples to evaluate (None for all)
        gen_kwargs: Generation kwargs (includes past_key_values for RoCK-KV)
        model_configs: Model configuration dictionary
        kv_cache: Optional RoCK-KV cache object
        file_dir: Directory to save results
        file_name: Base filename for results
        num_repeats: Total number of repeats to run
        completed_repeats: List of already completed repeat indices
        all_results: List to accumulate results (will be modified in-place)
        batch_size: Batch size for inference (default: 8)
        base_random_seed: Base random seed for Python's random module (default: 0)
        base_numpy_seed: Base random seed for numpy (default: 1234)
        base_torch_seed: Base random seed for torch (default: 1234)
        base_fewshot_seed: Base random seed for fewshot sampler (default: 1234)
    
    Returns:
        Updated list of all results (same as all_results parameter)
    """
    # Task-specific configuration (same for all repeats)
    task_lower = task.lower()
    if "aime" in task_lower:
        # AIME: apply chat template, but not fewshot_as_multiturn
        apply_chat = True
        fewshot_multiturn = False
        print(f"Task config: {task} (apply_chat_template=True, fewshot_as_multiturn=False)")
    else:
        # All other tasks: disable both
        apply_chat = False
        fewshot_multiturn = False
        print(f"Task config: {task} (apply_chat_template=False, fewshot_as_multiturn=False)")
    
    # Check if task requires code execution (unsafe tasks like humaneval, mbpp)
    unsafe_task_keywords = ["humaneval"]
    needs_unsafe_code = any(keyword in task_lower for keyword in unsafe_task_keywords)
    if needs_unsafe_code:
        print(f"Task requires code execution, setting confirm_run_unsafe_code=True")
    
    for repeat_idx in range(num_repeats):
        if repeat_idx in completed_repeats:
            continue
        
        # Calculate unique seeds for this repeat
        current_random_seed = base_random_seed + repeat_idx
        current_numpy_seed = base_numpy_seed + repeat_idx
        current_torch_seed = base_torch_seed + repeat_idx
        current_fewshot_seed = base_fewshot_seed + repeat_idx
        
        print(f"\n{'='*80}")
        print(f"Running repeat {repeat_idx + 1}/{num_repeats}")
        print(f"Seeds: random={current_random_seed}, numpy={current_numpy_seed}, "
              f"torch={current_torch_seed}, fewshot={current_fewshot_seed}")
        print(f"{'='*80}\n")
        
        # Run evaluation with unique seeds for this repeat
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            gen_kwargs=gen_kwargs,  # Includes past_key_values (RoCK-KV cache)
            log_samples=True,
            apply_chat_template=apply_chat,
            fewshot_as_multiturn=fewshot_multiturn,
            # Different seeds for each repeat
            random_seed=current_random_seed,
            numpy_random_seed=current_numpy_seed,
            torch_random_seed=current_torch_seed,
            fewshot_random_seed=current_fewshot_seed,
            # Only allow unsafe code execution for specific tasks
            confirm_run_unsafe_code=needs_unsafe_code,
        )
        
        print(f"\n[Completed] All samples evaluated for repeat {repeat_idx}")
        
        # Build output in lm_eval format
        output = build_eval_output(
            results=results,
            task=task,
            repeat_idx=repeat_idx,
            model=model,
            model_configs=model_configs
        )
        
        # Save individual repeat result (with full samples) to file
        repeat_file = f"{file_dir}/{file_name}_repeat_{repeat_idx}.json"
        with open(repeat_file, "w") as f:
            json.dump(output, f, indent=4)
        print(f"[Saved] Repeat {repeat_idx} result: {repeat_file}")
        
        # Only keep minimal data in memory for summary statistics (drop samples to save memory)
        # This is critical to prevent OOM when running many repeats
        minimal_result = {
            "repeat_idx": output["repeat_idx"],
            "tasks": output["tasks"]
        }
        all_results.append(minimal_result)
        
        # Clean up memory after each repeat to prevent OOM
        del results  # Delete the large results dict from lm_eval
        del output   # Delete the full output dict (already saved to file)
        del minimal_result  # Just the name, object is in all_results
        gc.collect()  # Run garbage collection
        torch.cuda.empty_cache()  # Clear CUDA cache
        print(f"[Memory] Cleaned up after repeat {repeat_idx}")
    
    return all_results


########################################### Summary Statistics ###########################################

def generate_summary_statistics(
    all_results: List[Dict], 
    task: str, 
    model_configs: Dict[str, Any], 
    num_repeats: int
) -> Dict:
    """
    Generate summary statistics from multiple repeat results.
    
    Args:
        all_results: List of result dictionaries from each repeat
        task: Task name (e.g., "aime24")
        model_configs: Model configuration dictionary
        num_repeats: Total number of repeats
    
    Returns:
        Dictionary containing summary statistics (mean, std, variance, min, max, median)
    """
    cdt_timezone = timezone(timedelta(hours=-5))  # CDT: UTC-5
    
    summary = {
        "generated_at": datetime.now(cdt_timezone).isoformat(),
        "num_repeats": num_repeats,
        "model_configs": model_configs,
        "task": task,
        "statistics": {}
    }
    
    # Extract metrics from all repeats
    # Assuming task results structure: results["tasks"][task_name][metric_name]
    task_name = list(all_results[0]["tasks"].keys())[0]
    metrics = all_results[0]["tasks"][task_name].keys()
    
    for metric in metrics:
        # Skip stderr metrics (they contain "_stderr" in the name)
        if "_stderr" in metric:
            continue
            
        values = []
        for result in all_results:
            value = result["tasks"][task_name].get(metric)
            if isinstance(value, (int, float)):
                values.append(value)
        
        if values:
            values_array = np.array(values)
            summary["statistics"][metric] = {
                "values": values,
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "variance": float(np.var(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "median": float(np.median(values_array)),
            }
    
    return summary

