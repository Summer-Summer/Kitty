import torch
from transformers import LlamaConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
#
from rock_kv import RoCKKVCacheConfig, RoCKKVCache
from rock_kv.eval.runner import eval_model_downstream, release_model_memory
from .utils_cli import update_parser



import argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RoCK-KV models on downstream tasks.")
    parser.add_argument('model',                type=str,                   help='llama model to load')
    parser.add_argument("--task_list",          nargs="+",                  help="List of downstream tasks to evaluate")
    parser.add_argument("--eval_rock_kv",       action="store_true",        help="Evaluate RoCK-KV model")
    parser.add_argument("--debug",              action="store_true",        help="Debug mode, limit=8")
    parser = update_parser(parser)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    ModelName = args.model.split("/")[-1]
    #
    if args.eval_rock_kv:
        print("Using RoCK-KV Cache")
        cache_config = RoCKKVCacheConfig(
            sink_length=args.sink_length,
            buffer_length=args.buffer_length,
            group_size=args.group_size,
            kbits=args.kbits,
            vbits=args.vbits,
            promote_ratio=args.promote_ratio,
            promote_bit=args.promote_bit,
            channel_selection=args.channel_selection,
            VCache_BitDecoding=False,  # Using KIVI Style V Cache
        )
        #
        rock_kv_cache = RoCKKVCache(cache_config=cache_config)        
        FileName = "rock_kv_g{}_b{}_s{}_sel{}_k{}_v{}_pb{}_pr{}".format(args.group_size, args.buffer_length, args.sink_length, args.channel_selection, args.kbits, args.vbits, args.promote_bit, args.promote_ratio)
    else:
        print("Using HF Default Dynamic Cache")
        #
        rock_kv_cache = None
        FileName = "hf_default"
    #
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
    eval_model_downstream(model, args.task_list, ModelName, FileName, args.debug, rock_kv_cache)
    release_model_memory(model)

if __name__ == "__main__":
    main()