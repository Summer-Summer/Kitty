import torch
from transformers import AutoModelForCausalLM
#
from rock_kv import get_kvcache_rock_kv
from rock_kv.eval.runner import eval_model_downstream, release_model_memory

from .utils_cli import update_parser

import argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RoCK-KV models on downstream tasks.")
    parser.add_argument('model',          type=str, default="Qwen/Qwen3-8B",    help='llama model to load')
    parser.add_argument("--task",         type=str, default="gsm8k_cot_llama",  help="Downstream task to evaluate")
    parser.add_argument("--eval_rock_kv", action="store_true",                  help="Evaluate RoCK-KV model")
    parser.add_argument("--debug",        action="store_true",                  help="Debug mode, limit=8")
    parser.add_argument("--num_repeats",  type=int, default=None,                help="Number of times to repeat evaluation. If not specified, will read from task's YAML config (default: 1)")
    parser.add_argument("--batch_size",   type=int, default=1,                   help="Batch size for inference (default: 1)")
    parser.add_argument("--max_new_tokens", type=int, default=4096,              help="Maximum number of new tokens to generate (default: 4096)")
    parser.add_argument("--results_dir",  type=str, default="./eval_results",    help="Directory to save evaluation results (default: ./eval_results)")
    parser = update_parser(parser)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    ModelName = args.model.split("/")[-1]
    #
    if args.eval_rock_kv:
        print("Using RoCK-KV Cache")
        rock_kv_cache = get_kvcache_rock_kv(args)
        FileName = "rock_kv_g{}_b{}_s{}_sel{}_k{}_v{}_pb{}_pr{}".format(args.group_size, args.buffer_length, args.sink_length, args.channel_selection, args.kbits, args.vbits, args.promote_bit, args.promote_ratio)
    else:
        print("Using HF Default Dynamic Cache")
        rock_kv_cache = None
        FileName = "{}_fp16_hf_default_{}".format(ModelName.lower().replace("-", "_"), args.task)
    #
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
    eval_model_downstream(model, args.task, ModelName, FileName, args.debug, rock_kv_cache, args.num_repeats, args.batch_size, args.max_new_tokens, args.results_dir)
    release_model_memory(model)

if __name__ == "__main__":
    main()