import torch
from transformers import AutoTokenizer
from transformers.models.qwen3 import Qwen3Config
#from transformers.models.qwen3 import Qwen3ForCausalLM
from kchanboost.models.qwen3 import Qwen3ForCausalLM
#
from rock_kv.eval.runner import release_model_memory, test_model_generate
from rock_kv.cli.utils_cli import update_parser, get_prompt


#
import argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-32B", help='Qwen3 model to load')
    parser.add_argument("--max_token_new",   type=int, default=200,    help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size",      type=int, default=1,      help="Batch size for generation, repeat the prompt for each batch")
    parser.add_argument("--prompt_choice",   type=int, default=0,      help="Choice of prompt to use")
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
    #
    APPLY_CHAT_TEMPLATE = True
    if APPLY_CHAT_TEMPLATE:
        messages = [[{"role": "user", "content": p}] for p in prompt]
        texts = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = tokenizer(text=texts, return_tensors="pt")
    else:
        inputs = tokenizer(text=prompt, return_tensors="pt")
    #
    config = Qwen3Config.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        config = config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map='auto'
    )
    model.eval()
    #
    outputs = test_model_generate(model, tokenizer, inputs, args.max_token_new, None)
    release_model_memory(model)


if __name__ == "__main__":
    main()