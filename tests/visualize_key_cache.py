import torch
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set to the GPU you want to use

from utils_visualization import visualize_tensor_3d

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Key-Value Cache")
    parser.add_argument('--file', type=str, required=True, help='Path to the KV cache checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./key_cache_visualizations', help='Output directory for visualization')
    return parser

parser = build_parser()
args = parser.parse_args()
fileName = args.file
output_dir = args.output_dir + "/" + fileName.replace('.pt', '').replace('past_kv_', '')
os.makedirs(output_dir, exist_ok=True)

data = torch.load(fileName)
q=data.query_cache
k=data.key_cache
v=data.value_cache


num_layers = len(k)
print(f"Number of layers in KV cache: {num_layers}")
for layer in range(0, num_layers, 10):
    query = q[layer]        # [B, H, T, D]
    key = k[layer]
    value = v[layer]
    assert query.dim() == 4, "Query, Key, and Value must be 4D tensors"
    assert query.shape[0] == 1, "Batch size must be 1 for visualization"
    for head in range(key.shape[1]):
        print(f"Visualizing KV Cache: Layer {layer}, Head {head}")
        T = min(key.shape[2],2048)  # Limit to 2048 for visualization

        key_head = key[0, head, :T, :]
        value_head = value[0, head, :, :T]
        # Visualize query, key, and value tensors
        ModelName = fileName.split('_')[2]
        Title = f"{ModelName} Layer {layer}, Head {head}"
        visualize_tensor_3d(key_head, FILE_PATH=f"{output_dir}/layer{layer}_head{head}_key.png", Title=Title + " - KeyCache")
        visualize_tensor_3d(value_head, FILE_PATH=f"{output_dir}/layer{layer}_head{head}_value.png", Title=Title + " - ValueCache")





