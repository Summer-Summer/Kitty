"""Utilities to run downstream evaluation for RoCK-KV models."""
import torch
from transformers import PreTrainedModel, AutoTokenizer
import lm_eval
#
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import gc
import time
import json
import matplotlib.pyplot as plt
#
from typing import Optional
from rock_kv import RoCKKVCache


#
def GetRoCKKVCache():
    """
    Get the RoCKKVCache class.
    """
    from rock_kv.llama_rock_kv import RoCKKVCache
    return RoCKKVCache

########################################### Shared utility functions ###########################################
def release_model_memory(model: PreTrainedModel):
    from accelerate.hooks import remove_hook_from_submodules
    remove_hook_from_submodules(model)
    model.to("cpu")
    #
    del model
    gc.collect()
    torch.cuda.empty_cache()

########################################### Used by cli/eval_rock_kv.py ###########################################
@torch.no_grad()
def eval_model_downstream(model: PreTrainedModel, downstream_tasks_list: list[str], ModelName, fileName, DEBUG=False, kv_cache: Optional[RoCKKVCache] = None):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    from lm_eval.models.huggingface import HFLM
    lm = HFLM(model)

    model_configs = {}
    model_configs["ModelArch"] = model.__class__.__name__
    model_configs["ModelPath"] = model.config._name_or_path
    if hasattr(model.config, "k_bits"):
        model_configs["k_bits"] = model.config.k_bits
        model_configs["v_bits"] = model.config.v_bits
        model_configs["group_size"] = model.config.group_size
        model_configs["sink_length"] = model.config.sink_length
        model_configs["buffer_length"] = model.config.buffer_length
        model_configs["promote_ratio"] = model.config.promote_ratio
        model_configs["promote_bit"] = model.config.promote_bit
        model_configs["channel_selection"] = model.config.channel_selection

    for task in downstream_tasks_list:
        if "mmlu" in task:
            num_fewshot = 4
        elif "gsm8k" in task:
            num_fewshot = 8
        elif "gpqa" in task:
            num_fewshot = 5
        elif "math" in task:
            num_fewshot = 4
        elif "bbh" in task:
            num_fewshot = 3
        else:
            num_fewshot = 0
        
        if DEBUG:
            limit = 8
        else:
            limit = None

        print("Evaluating model accuracy on downstream tasks...")
        print("ModelName: ", model_configs["ModelPath"])
        print("Task: ", task)
        print("Eval_Configs: ", fileName)
        print("Num_Fewshot: ", num_fewshot)
        #
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
                "max_new_tokens": 1024,  # Maximum number of new tokens to generate
                "do_sample": False,    # Disable sampling for deterministic evaluation
                "past_key_values": kv_cache,  # Use RoCKKV cache if provided
            }
        )

        output = {}
        au_tz = ZoneInfo("Australia/Sydney")
        output["This file is generated_at"] = datetime.now(au_tz).isoformat()
        output["tasks"] = results['results']
        output["model_configs"] = model_configs
        output["eval_configs"] = results["config"]
        # 
        output["eval_configs"]["model_dtype"] = str(model.dtype)
        if output["eval_configs"]["gen_kwargs"]["past_key_values"] is not None:
            output["eval_configs"]["gen_kwargs"]["past_key_values"] = output["eval_configs"]["gen_kwargs"]["past_key_values"].cache_implementation
        #
        if "samples" in results:
            _, first_task_samples = next(iter(results["samples"].items()))
            first_task_sample = first_task_samples[0]
            arguments_from_samples = first_task_sample["arguments"][0][1]
            if arguments_from_samples["past_key_values"] is not None:
                arguments_from_samples["past_key_values"] = arguments_from_samples["past_key_values"].cache_implementation
            output["eval_configs"]["arguments_from_samples"] = arguments_from_samples
            #
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

def test_model_generate(model: PreTrainedModel, tokenizer: AutoTokenizer, inputs: dict, KV_Type: str, max_new_tokens=200, visualize_kv: bool= False, kv_cache: Optional[RoCKKVCache] = None):
    # Time consumption
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    # Execution
    outputs = model.generate(
        input_ids=inputs.input_ids.cuda(),
        max_new_tokens=max_new_tokens, # max_new_tokens is the maximum number of new tokens to generate.
        return_dict_in_generate=True,
        past_key_values=kv_cache,
        do_sample=False, # llama3 enables sampling by default, but we need to disable it to evaluate the deterministic generation performance.
    )
    # Time consumption
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    #
    output_text = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True)
    #
    print("#"*200)
    print(f"Model.generate() execution time: {elapsed_time:.4f} 秒")
    print("Outputs_{}:\n".format(KV_Type), output_text)
    print("#"*200)
    #
    if visualize_kv:
        assert KV_Type == "HF"
        #torch.save(outputs.past_key_values, "past_key_values.pt")
        visualize_kv_cache(outputs.past_key_values, save_dir="kv_visualizations")
    #
    return outputs