import torch
from transformers import LlamaConfig, AutoTokenizer, PreTrainedModel, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
#
from models.llama_rock_kv import LlamaForCausalLM_RoCKKV
from rock_kv.eval.runner import release_model_memory, test_model_generate
from .utils_cli import update_parser

#
import argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='llama model to load')
    parser.add_argument("--gen_hf",          action="store_true",                        help="Evaluate original HF model")
    parser.add_argument("--gen_rock_kv",     action="store_true",                        help="Evaluate RoCKKV model")
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
prompt2 = "现在父亲的年龄是儿子的 3 倍。再过 15 年后，父亲的年龄会是儿子的 2 倍。请问： 1.现在父亲几岁？ 2.现在儿子几岁？"

def main() -> None:
    args = build_parser().parse_args()
    print("Model:\n",args.model)
    #
    prompt = prompt2
    prompt = [prompt for _ in range(args.batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer(text=prompt, return_tensors="pt")
    #
    if args.gen_hf:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
        outputs_hf = test_model_generate(model, tokenizer, inputs, "HF", args.visualize_kv)
        release_model_memory(model)
    #
    if args.gen_flashkv:
        # Configs
        config = LlamaConfig.from_pretrained(args.model)
        config.k_bits = 2 # current support 2/4 bit for KV Cache
        config.v_bits = 2 # current support 2/4 bit for KV Cache
        config.group_size = 32
        config.residual_length = 128 # the number of recent fp16 tokens
        config.use_flash = True
        #
        config.k_bits = 2 # current support 2/4 bit for KV Cache
        config.v_bits = 2 # current support 2/4 bit for KV Cache
        #
        config.haojun_version = "FlashKV_v2"
        config.sink_length = 32
        #
        model_flashkv = LlamaForCausalLM_FlashKV.from_pretrained(args.model, config=config, torch_dtype=torch.float16)
        device_map = infer_auto_device_map(model_flashkv, no_split_module_classes=["LlamaDecoderLayer_FlashKV"], max_memory={i: "40GB" for i in range(4)})
        model_flashkv = dispatch_model(model_flashkv, device_map=device_map)
        outputs_flashkv = Test_Generate(model_flashkv, tokenizer, inputs, "FlashKV", args)
        release_model_memory(model)_flashkv)
    #
    if not args.gen_flashkv and not args.gen_hf:
        raise ValueError("Please specify either --gen_hf or --gen_flashkv to evaluate the model.")

if __name__ == "__main__":
    main()