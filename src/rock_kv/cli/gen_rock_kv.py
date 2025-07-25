import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
#
from rock_kv import get_kvcache_rock_kv
from rock_kv.eval.runner import release_model_memory, test_model_generate, visualize_kv_cache
from .utils_cli import update_parser, get_prompt


#
import argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='llama model to load')
    parser.add_argument("--gen_rock_kv",     action="store_true",                        help="Evaluate RoCKKV model")
    parser.add_argument("--max_token_new",   type=int, default=200,                      help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size",      type=int, default=1,                        help="Batch size for generation, repeat the prompt for each batch")
    parser.add_argument("--visualize_kv",    action="store_true",                        help="Visualize KV Cache")
    parser.add_argument("--prompt_choice",   type=int, default=0,                        help="Choice of prompt to use")
    parser = update_parser(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("Model: ",args.model)
    #
    prompt_choice = args.prompt_choice
    task_name, prompt = get_prompt(prompt_choice)
    print(f"Task: {task_name}, Prompt: {prompt}")
    #
    prompt = [prompt for _ in range(args.batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer(text=prompt, return_tensors="pt")
    #
    if args.gen_rock_kv:
        print("Using RoCK-KV Cache")
        rock_kv_cache = get_kvcache_rock_kv(args)
    else:
        print("Using HF Default Dynamic Cache")
        rock_kv_cache = None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
    outputs = test_model_generate(model, tokenizer, inputs, args.max_token_new, rock_kv_cache)
    release_model_memory(model)

    #
    if args.visualize_kv:
        assert rock_kv_cache is None, "visualization is only supported for HF default dynamic cache, not RoCK-KV cache."
        model_name = args.model.split("/")[-1]
        torch.save(outputs.past_key_values, f"past_kv_{model_name}_{task_name}.pt")
        #visualize_kv_cache(outputs.past_key_values, save_dir="kv_visualizations")

if __name__ == "__main__":
    main()