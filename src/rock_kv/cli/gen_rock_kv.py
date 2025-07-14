import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
#
from rock_kv import get_kvcache_rock_kv
from rock_kv.eval.runner import release_model_memory, test_model_generate, visualize_kv_cache
from .utils_cli import update_parser


#
import argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='llama model to load')
    parser.add_argument("--gen_rock_kv",     action="store_true",                        help="Evaluate RoCKKV model")
    parser.add_argument("--max_token_new",   type=int, default=200,                      help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size",      type=int, default=1,                        help="Batch size for generation, repeat the prompt for each batch")
    parser.add_argument("--visualize_kv",    action="store_true",                        help="Visualize KV Cache")
    parser = update_parser(parser)
    return parser


prompt1 = """Q: There are 15 trees in the grove. Grove workers will plant trees in the
grove today. After they are done, there will be 21 trees. How many trees did
the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there were 21 trees 
after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
"""
prompt2 = "现在父亲的年龄是儿子的 3 倍。再过 15 年后，父亲的年龄会是儿子的 2 倍。请回答以下两个问题： 1.现在父亲几岁？ 2.现在儿子几岁？"

def main() -> None:
    args = build_parser().parse_args()
    print("Model: ",args.model)
    #
    prompt = prompt2
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
        #torch.save(outputs.past_key_values, "past_key_values.pt")
        visualize_kv_cache(outputs.past_key_values, save_dir="kv_visualizations")

if __name__ == "__main__":
    main()