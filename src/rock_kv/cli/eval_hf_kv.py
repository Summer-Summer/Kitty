import torch
from transformers import AutoModelForCausalLM
from rock_kv.eval.runner import eval_model_downstream_hf, release_model_memory

import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace native quantized KV Cache on downstream tasks.")
    parser.add_argument('model',          type=str, help='HuggingFace model path (e.g., Qwen/Qwen3-8B)')
    parser.add_argument("--task",         type=str, required=True, help="Downstream task to evaluate (e.g., gsm8k_cot_llama)")
    
    # HF Quantized Cache 参数
    parser.add_argument("--hf_cache_backend", type=str, default="HQQ", 
                       choices=["HQQ", "quanto"], 
                       help="Quantization backend: HQQ (recommended) or quanto")
    parser.add_argument("--hf_cache_nbits", type=int, default=4,
                       choices=[2, 4, 8], 
                       help="Number of quantization bits (default: 4)")
    parser.add_argument("--hf_axis_key", type=int, default=1,
                       help="Axis for key quantization. HQQ: 1 (default), quanto: 0")
    parser.add_argument("--hf_axis_value", type=int, default=1,
                       help="Axis for value quantization. HQQ: 1 (default), quanto: 0")
    
    # 评估参数
    parser.add_argument("--debug",        action="store_true", help="Debug mode, limit=8")
    parser.add_argument("--num_repeats",  type=int, default=None, 
                       help="Number of times to repeat evaluation. If not specified, will read from task's YAML config (default: 1)")
    parser.add_argument("--batch_size",   type=int, default=1, 
                       help="Batch size for inference (default: 1)")
    parser.add_argument("--max_new_tokens", type=int, default=4096, 
                       help="Maximum number of new tokens to generate (default: 4096)")
    parser.add_argument("--results_dir",  type=str, default="./eval_results", 
                       help="Directory to save evaluation results (default: ./eval_results)")
    
    return parser

def main() -> None:
    args = build_parser().parse_args()
    ModelName = args.model.split("/")[-1]
    
    print("=" * 80)
    print("HuggingFace Native Quantized KV Cache Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Backend: {args.hf_cache_backend}")
    print(f"Quantization: {args.hf_cache_nbits}-bit")
    print(f"Axis: key={args.hf_axis_key}, value={args.hf_axis_value}")
    print("=" * 80)
    
    # 构建 HF cache 配置
    hf_cache_config = {
        "cache_implementation": "quantized",
        "cache_config": {
            "backend": args.hf_cache_backend,
            "nbits": args.hf_cache_nbits,
            "axis_key": args.hf_axis_key,
            "axis_value": args.hf_axis_value
        }
    }
    
    # 生成文件名
    FileName = f"hf_{args.hf_cache_backend.lower()}_int{args.hf_cache_nbits}_axis{args.hf_axis_key}"
    
    # 加载模型
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16, 
        device_map='auto'
    )
    print("✓ Model loaded successfully\n")
    
    # 评估
    eval_model_downstream_hf(
        model, 
        args.task, 
        ModelName, 
        FileName, 
        args.debug, 
        hf_cache_config, 
        args.num_repeats, 
        args.batch_size, 
        args.max_new_tokens, 
        args.results_dir
    )
    
    # 清理内存
    release_model_memory(model)

if __name__ == "__main__":
    main()

