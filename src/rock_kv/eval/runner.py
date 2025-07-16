"""Utilities to run downstream evaluation for RoCK-KV models."""
import torch
from transformers import PreTrainedModel, AutoTokenizer
import lm_eval
#
from datetime import datetime, timezone, timedelta
import os
import gc
import time
import json
import matplotlib.pyplot as plt
from typing import Optional
import shutil


#
from rock_kv import RoCKKVCache

########################################### Shared utility functions ###########################################
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

########################################### Used by cli/eval_rock_kv.py ###########################################
@torch.no_grad()
def eval_model_downstream(model: PreTrainedModel, task: str, ModelName, fileName, DEBUG=False, kv_cache: Optional[RoCKKVCache] = None):

    from lm_eval.models.huggingface import HFLM
    lm = HFLM(model)

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
    few_shot_dict = {"mmlu": 4, "gsm8k": 8, "gpqa": 5, "math": 4, "bbh": 3,}
    for key, value in few_shot_dict.items():
        if key in task:
            num_fewshot = value
            break
    else:
        num_fewshot = 0

    # Enable code evaluation for humaneval task
    if "humaneval" in task:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    # For DEBUG mode, limit the number of samples
    limit = None
    if DEBUG:
        limit = 8

    print("Evaluating model accuracy on downstream tasks...")
    print("ModelName: ", model_configs["ModelPath"])
    print("Task: ", task)
    print("Eval_Configs: ", fileName)
    print("Num_Fewshot: ", num_fewshot)

    # Evaluating...
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=num_fewshot,
        limit=limit,                    # for debugging
        confirm_run_unsafe_code=True,   # 允许运行 unsafe 代码
        log_samples=True,
        verbosity="ERROR",
        #batch_size='auto',
        batch_size=1,
        cache_requests=False,
        gen_kwargs={
            "past_key_values": kv_cache,  # Use RoCKKV cache if provided
            "max_new_tokens": 1024,  # Maximum number of new tokens to generate
            "max_length": None,
            "do_sample": False,    # Disable sampling for deterministic evaluation
            "temperature": None,
            "top_p": None,
            "top_k": None,         # Disable top-k sampling
        }
    )

    output = {}
    #
    utc_plus_10 = timezone(timedelta(hours=10))
    output["This file is generated_at"] = datetime.now(utc_plus_10).isoformat()
    #
    output["tasks"] = results['results']
    output["model_configs"] = model_configs
    output["eval_configs"] = results["config"]

    # Dealing with corner cases, where model configs can not be serialized.
    output["eval_configs"]["model_dtype"] = str(model.dtype)
    if output["eval_configs"]["gen_kwargs"]["past_key_values"] is not None:
        output["eval_configs"]["gen_kwargs"]["past_key_values"] = output["eval_configs"]["gen_kwargs"]["past_key_values"].cache_implementation
    if "samples" in results:
        # Extract shared gen_kargs from the first sample
        _, first_task_samples = next(iter(results["samples"].items()))
        first_task_sample = first_task_samples[0]
        arguments_from_samples = first_task_sample["arguments"][0][1]
        if arguments_from_samples["past_key_values"] is not None:
            arguments_from_samples["past_key_values"] = arguments_from_samples["past_key_values"].cache_implementation
        output["eval_configs"]["arguments_from_samples"] = arguments_from_samples
        # Exclude the gen_kwargs from the output of each sample
        for task_name, task_samples in results["samples"].items():
            for sample in task_samples:
                sample["arguments"] = sample["arguments"][0][0]
        output["samples"] = results["samples"]

    # Save results to file
    FileDir = "./eval_results/{}/{}".format(ModelName, task)
    if not os.path.exists(FileDir):
        os.makedirs(FileDir, exist_ok=True)
    FilePATH = "{}/{}.json".format(FileDir, fileName)
    with open(FilePATH, "w") as f:
        json.dump(output, f, indent=4)
    print("Results saved to:", FilePATH)


########################################### Used by cli/gen_rock_kv.py ###########################################
def test_model_generate(model: PreTrainedModel, tokenizer: AutoTokenizer, inputs: dict, max_new_tokens=200, kv_cache: Optional[RoCKKVCache] = None):
    # Time consumption
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    # Execution
    outputs = model.generate(
        input_ids=inputs.input_ids.cuda(),
        attention_mask=inputs.attention_mask.cuda() if "attention_mask" in inputs else None,
        max_new_tokens=max_new_tokens, # max_new_tokens is the maximum number of new tokens to generate.
        return_dict_in_generate=True,
        past_key_values=kv_cache,
        do_sample=False,
        max_length=None,  # No maximum length, let the model decide based on max_new_tokens
        temperature=None,  # No temperature for deterministic generation
        top_p=None,       # No top-p sampling
        top_k=None,       # No top-k sampling
    )
    # Time consumption
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Model.generate() execution time: {elapsed_time:.4f} 秒")
    #
    output_text = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True)
    #
    width = shutil.get_terminal_size().columns
    print('-' * width)
    print(output_text)
    print('-' * width)
    #
    return outputs