import torch
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set to the GPU you want to use

from utils_visualization import visualize_tensor_3d, visualize_mse_lineplot
from utils_compute import fake_quant_groupwise_lastdim, eager_attention_forward

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the effect of quantizing a channel in a Key cache.")
    parser.add_argument('--file', type=str, required=True, help='Path to the KV cache checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./quantize_a_channel', help='Output directory for visualization')
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
for layer in range(0, num_layers, 1):
    print(f"Processing Layer {layer}")
    query = q[layer]        # [B, H, T, D]
    key = k[layer]
    value = v[layer]
    assert query.dim() == 4, "Query, Key, and Value must be 4D tensors"
    assert query.shape[0] == 1, "Batch size must be 1 for visualization"
    #
    query = query[:, :, :1024+32, :]
    key   = key[:, :, :1024+32, :]
    value = value[:, :, :1024+32, :]
    #
    B, h_q, T, D = query.shape
    h_kv = key.shape[1]
    G = h_q // h_kv
    # baseline Attention
    attn_output, attn_weights = eager_attention_forward(query, key, value)
    score = attn_weights[0]

    #
    mse_all_group = []
    for g in range(G):
        MSE = []
        base_mask = torch.zeros((h_kv, D), dtype=torch.bool)
        for c in range(0, D):
            quantize_mask = base_mask.clone()
            quantize_mask[:, c] = True  # Promote this channel
            #
            key_quantized = fake_quant_groupwise_lastdim(key[:,:,32:1024+32,:], group_size=128, quantize_mask=quantize_mask, quantize_bit=2)
            key_quantized = torch.cat([key[:,:,:32,:], key_quantized], dim=2)
            attn_output_quantized, attn_weights_quantized = eager_attention_forward(query, key_quantized, value)
            score_h0_quantized = attn_weights_quantized[0]
            #
            diff = score[g] - score_h0_quantized[g]
            # MSE
            mse = torch.mean(diff.pow(2))
            MSE.append(mse)
        mse_tensor = torch.stack(MSE)  # Shape: (D,)
        mse_all_group.append(mse_tensor)
    # Visualize the MSE for each channel
    mse_matrix = torch.stack(mse_all_group, dim=0)  # Shape: (G, D)
    save_path = f"{output_dir}/layer{layer}_mse.png"
    visualize_mse_lineplot(mse_matrix, save_path, title=f"Layer {layer} - MSE")