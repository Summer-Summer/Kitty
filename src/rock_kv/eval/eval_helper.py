"""Helper functions for downstream evaluation."""

import os
import json
import numpy as np
import lm_eval
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any, Optional
from transformers import PreTrainedModel


########################################### Checkpoint Management ###########################################

def load_completed_repeats(file_dir: str, file_name: str, num_repeats: int) -> Tuple[List[int], List[Dict]]:
    """
    Check which repeats are already completed and load their results.
    
    Args:
        file_dir: Directory containing repeat result files
        file_name: Base filename for repeat results
        num_repeats: Total number of repeats expected
    
    Returns:
        Tuple of (list of completed repeat indices, list of loaded results)
    """
    completed_repeats = []
    all_results = []
    
    for repeat_idx in range(num_repeats):
        repeat_file = f"{file_dir}/{file_name}_repeat_{repeat_idx}.json"
        if os.path.exists(repeat_file):
            print(f"[Checkpoint] Found existing result for repeat {repeat_idx}: {repeat_file}")
            completed_repeats.append(repeat_idx)
            with open(repeat_file, "r") as f:
                all_results.append(json.load(f))
    
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
    task: str,
    repeat_idx: int,
    all_samples: List[Dict],
    aggregated_metrics: Dict,
    model: PreTrainedModel,
    model_configs: Dict,
    gen_kwargs: Dict,
    limit: Optional[int],
    kv_cache: Optional[Any] = None,
    batch_size: int = 8,
    random_seed: int = 0,
    numpy_seed: int = 1234,
    torch_seed: int = 1234,
    fewshot_seed: int = 1234
) -> Dict:
    """
    Build output dictionary in lm_eval format.
    
    Args:
        task: Task name
        repeat_idx: Current repeat index
        all_samples: List of evaluated samples
        aggregated_metrics: Aggregated metrics from evaluation
        model: The model being evaluated
        model_configs: Model configuration dictionary
        gen_kwargs: Generation kwargs (may include past_key_values)
        limit: Number of samples to evaluate (None for all)
        kv_cache: Optional RoCK-KV cache object
        batch_size: Batch size for inference (default: 8)
        random_seed: Random seed used for Python's random module
        numpy_seed: Random seed used for numpy
        torch_seed: Random seed used for torch
        fewshot_seed: Random seed used for fewshot sampler
    
    Returns:
        Dictionary in lm_eval output format
    """
    cdt_timezone = timezone(timedelta(hours=-5))  # CDT: UTC-5
    
    output = {
        "This file is generated_at": datetime.now(cdt_timezone).isoformat(),
        "repeat_idx": repeat_idx,
        "tasks": {
            task: aggregated_metrics
        },
        "model_configs": model_configs,
        "eval_configs": {
            "model": model.config._name_or_path,
            "model_args": None,
            "model_num_parameters": sum(p.numel() for p in model.parameters()),
            "model_dtype": str(model.dtype),
            "model_revision": "main",
            "model_sha": "",
            "batch_size": batch_size,
            "batch_sizes": [],
            "device": str(model.device) if hasattr(model, 'device') else None,
            "use_cache": None,
            "limit": limit,
            "bootstrap_iters": 100000,
            "gen_kwargs": {k: v for k, v in gen_kwargs.items() if k != "past_key_values"},
            "random_seed": random_seed,
            "numpy_seed": numpy_seed,
            "torch_seed": torch_seed,
            "fewshot_seed": fewshot_seed,
        },
        "samples": {
            task: all_samples
        }
    }
    
    # Handle KV cache serialization
    if gen_kwargs.get("past_key_values") is not None:
        output["eval_configs"]["gen_kwargs"]["past_key_values"] = kv_cache.cache_implementation if kv_cache else None
    
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
            apply_chat_template=True,
            fewshot_as_multiturn=False,
            # Different seeds for each repeat
            random_seed=current_random_seed,
            numpy_random_seed=current_numpy_seed,
            torch_random_seed=current_torch_seed,
            fewshot_random_seed=current_fewshot_seed,
        )
        
        # Get samples and metrics from results
        all_samples = results.get("samples", {}).get(task, [])
        aggregated_metrics = results.get("results", {}).get(task, {})
        
        print(f"\n[Completed] All samples evaluated for repeat {repeat_idx}")
        
        # Build output in lm_eval format
        output = build_eval_output(
            task=task,
            repeat_idx=repeat_idx,
            all_samples=all_samples,
            aggregated_metrics=aggregated_metrics,
            model=model,
            model_configs=model_configs,
            gen_kwargs=gen_kwargs,
            limit=limit,
            kv_cache=kv_cache,
            batch_size=batch_size,
            random_seed=current_random_seed,
            numpy_seed=current_numpy_seed,
            torch_seed=current_torch_seed,
            fewshot_seed=current_fewshot_seed
        )
        
        # Save individual repeat result
        repeat_file = f"{file_dir}/{file_name}_repeat_{repeat_idx}.json"
        with open(repeat_file, "w") as f:
            json.dump(output, f, indent=4)
        print(f"[Saved] Repeat {repeat_idx} result: {repeat_file}")
        
        all_results.append(output)
    
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

