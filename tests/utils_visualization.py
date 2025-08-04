import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import seaborn as sns

def visualize_tensor_3d(tensor, FILE_PATH='3d_tensor.png', Title='3D Tensor Visualization'):
    if len(tensor.shape) != 2:
        raise ValueError("Input tensor must be a 2D tensor.")

    # 将 Tensor 转为 NumPy 数组
    data = tensor.clone().detach().cpu().numpy()
    data = np.abs(data)  # 使用绝对值显示数据

    z_max = np.percentile(data, 99.99)
    data = np.clip(data, 0, z_max)

    # 获取 X 和 Y 坐标
    tokens, channels = data.shape
    X, Y = np.meshgrid(np.arange(channels), np.arange(tokens))

    # 创建图形
    fig = plt.figure(figsize=(4.5, 3.5))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 图
    ax.plot_surface(X, Y, data, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel("Channel", fontsize=12, labelpad=5)
    ax.set_ylabel("Token", fontsize=12, labelpad=5)
    #ax.set_zlabel("Absolute Value", fontsize=14, labelpad=10)
    #ax.zaxis.label.set_rotation(90)

    # 显示网格线
    ax.grid(True)
    ax.view_init(elev=20, azim=-60)
    ax.tick_params(axis='both', which='major', labelsize=8)
    #
    plt.savefig(FILE_PATH, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()



import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_mse_lineplot(mse_tensor: torch.Tensor, save_path: str, title: str = "Channel-wise Attention Score MSE"):
    """
    Visualize one or more MSE curves.
    
    Args:
        mse_tensor: 
            - shape (D,) for a single line
            - shape (G, D) for multiple Q heads over same KV head
        save_path: Path to save the image.
        title: Title of the plot.
    """
    mse_np = mse_tensor.detach().cpu().numpy()
    
    plt.figure(figsize=(4, 2))

    if mse_np.ndim == 1:
        plt.plot(mse_np, linewidth=1.5)
    elif mse_np.ndim == 2:
        G = mse_np.shape[0]
        for g in range(G):
            plt.plot(mse_np[g], label=f"Q_head {g}", linewidth=1.2)
        plt.legend(fontsize=8, loc='upper right', ncol=2, frameon=False)
    else:
        raise ValueError("mse_tensor must be 1D or 2D (G x D).")

    #plt.title(title, fontsize=12)
    plt.xlabel("Channel Index", fontsize=10)
    plt.ylabel("MSE", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_tensor_1d(tensor, FILE_PATH='1d_tensor.png'):
    """
    Visualize a 1D PyTorch tensor as a line plot.
    
    Args:
        tensor (torch.Tensor): Input 1D tensor.
        FILE_PATH (str): Path to save the output image.
    """
    if len(tensor.shape) != 1:
        raise ValueError("Input tensor must be a 1D tensor.")

    # Convert tensor to NumPy array
    data = tensor.clone().detach().cpu().numpy()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(data, color='blue', label="Tensor Values")
    plt.title("1D Tensor Visualization (Line Plot)", fontsize=16, weight='bold')
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(FILE_PATH, dpi=300)
    plt.close()


def visualize_tensor_1d_histogram(tensor, output_file='1d_tensor_hist.png', bins=100):
    """
    Compute the histogram of a 1D PyTorch tensor.
    """
    if len(tensor.shape) != 1:
        raise ValueError("Input tensor must be 1D.")
    
    # Convert tensor to NumPy array
    data = tensor.clone().detach().cpu().numpy()

    # Compute histogram
    hist, bin_edges = np.histogram(data, bins=bins)

    plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Tensor")

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.close()

def visualize_attention_matrix(tensor, output_file="attention_matrix.png"):
    """
    Visualizes a 2D attention weight matrix.

    Args:
        attention_weights (numpy.ndarray): 2D matrix of attention weights (shape: [seq_len, seq_len]).
        output_file (str): Path to save the visualization.
    """
    # Ensure the input is a numpy array
    attention_weights = tensor.clone().detach().cpu().numpy()
    seq_len = attention_weights.shape[0]

    # Mask the upper triangle to make it similar to the example
    #mask = np.triu(np.ones_like(attention_weights, dtype=bool), k=1)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        #mask=mask,  # Apply the mask to hide the upper triangle
        cmap="coolwarm",  # Use a diverging colormap
        annot=False,
        cbar=True,
        xticklabels=False, #np.arange(1, seq_len + 1),  # Set tokens as x-axis labels
        yticklabels=False, #np.arange(1, seq_len + 1),  # Set tokens as y-axis labels
    )
    
    # Customize the visualization
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.title("Attention Matrix", fontsize=16)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.close()

"""
def visualize_kv_cache(kv, save_dir="kv_visualizations"):
    def visualize_kv_tensor(key, value, suffix="", cmap='viridis', save_dir=save_dir):
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot K
        im0 = axes[0].imshow(key, cmap=cmap, aspect='auto')
        axes[0].set_title(f'Layer {layer_} Head {head_} - K')
        axes[0].set_xlabel('Head Dimension')
        axes[0].set_ylabel('Sequence Length')
        fig.colorbar(im0, ax=axes[0])
        # Plot V
        im1 = axes[1].imshow(value, cmap=cmap, aspect='auto')
        axes[1].set_title(f'Layer {layer_} Head {head_} - V')
        axes[1].set_xlabel('Head Dimension')
        axes[1].set_ylabel('Sequence Length')
        fig.colorbar(im1, ax=axes[1])
        #
        plt.tight_layout()
        plt.savefig(f'{save_dir}/layer{layer_}_head{head_}_kvcache_{suffix}.png')
        plt.close()
    #
    for layer_ in range(len(kv)):
        print("Visualizing KV Cache: Layer_{}".format(layer_))
        key = kv[layer_][0].cpu().numpy()
        value = kv[layer_][1].cpu().numpy()
        assert key.shape == value.shape
        b, h, s, d = key.shape
        assert b == 1
        for head_ in range(h):
            visualize_kv_tensor(key[0, head_], value[0, head_], "magnitute")
"""