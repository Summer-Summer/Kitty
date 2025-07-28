import torch


fileName = "past_kv_Qwen3-32B_gpqa.pt"
fileName = "past_kv_Qwen3-32B_gsm8k.pt"
fileName = "past_kv_Qwen3-8B_gpqa.pt"

data = torch.load(fileName)  # 或者 'checkpoint.pt' 等
q=data.query_cache
k=data.key_cache
v=data.value_cache

torch.set_printoptions(
    threshold=float('inf'),    # 打印全部元素
    linewidth=150,             # 每行显示多少字符
    precision=4,               # 小数位数
    sci_mode=False             # 禁止科学计数法
)

x=k[0].abs().mean(-2)[:,0,:]
y,idx=torch.sort(x)

def Amplitude(q):
    x = (q[:, :, :, :q.shape[3]//2].pow(2) + q[:, :, :, q.shape[3]//2:].pow(2)).sqrt()
    return x

def print_matrix(matrix):
    m = matrix.reshape(-1, 8)  # 展平为二维
    for row in m:
        print(" ".join(f"{val:4.2f}\t" for val in row))