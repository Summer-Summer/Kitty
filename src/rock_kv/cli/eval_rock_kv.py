import torch
#
from transformers import LlamaConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
#
from rock_kv import LlamaForCausalLM_RoCKKV
from rock_kv.eval.runner import eval_model_downstream, release_model_memory
from .utils_cli import update_parser
#
import argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RoCK-KV models on downstream tasks.")
    parser.add_argument('model',                type=str,                   help='llama model to load')
    parser.add_argument("--task_list",          nargs="+",                  help="List of downstream tasks to evaluate")
    parser.add_argument("--eval_hf",            action="store_true",        help="Evaluate original HF model")
    parser.add_argument("--eval_rock_kv",       action="store_true",        help="Evaluate RoCK-KV model")
    parser.add_argument("--debug",              action="store_true",        help="Debug mode, limit=8")
    parser = update_parser(parser)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    ModelName = args.model.split("/")[-1]

    #
    if args.eval_hf:
        model_hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
        eval_model_downstream(model_hf, args.task_list, ModelName, "hf", args.debug)
        release_model_memory(model_hf)
    #
    elif args.eval_rock_kv:
        config = LlamaConfig.from_pretrained(args.model)
        config.sink_length = args.sink_length
        config.buffer_length = args.buffer_length
        config.group_size = args.group_size
        config.k_bits = args.kbits
        config.v_bits = args.vbits
        config.promote_ratio = args.promote_ratio
        config.promote_bit = args.promote_bit
        config.channel_selection = args.channel_selection
        #
        model = LlamaForCausalLM_RoCKKV.from_pretrained(args.model, config=config, torch_dtype=torch.float16)
        device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer_RoCKKV"], max_memory={0: "75GB", 1: "78GB"})
        model = dispatch_model(model, device_map=device_map)
        FileNameSuffix = "rock_kv_g{}_b{}_s{}_pb{}_sel{}_k{}_v{}_pr{}".format(config.group_size, config.buffer_length, config.sink_length, config.promote_bit, config.channel_selection, config.k_bits, config.v_bits, config.promote_ratio)
        eval_model_downstream(model, args.task_list, ModelName, FileNameSuffix, args.debug)
        release_model_memory(model)
    else:
        raise ValueError("Please specify either --eval_hf or --eval_rock_kv to evaluate the model.")

if __name__ == "__main__":
    main()