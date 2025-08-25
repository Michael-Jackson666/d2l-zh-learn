import torch
from torch import nn
import time
import platform # 引入platform库来检测操作系统

# =============================================================================
# Part 1: 检测和指定 Apple Silicon (MPS) 设备
# -----------------------------------------------------------------------------
# 为macOS环境编写一个健壮的设备检测函数。
# =============================================================================
print("--- 1. 检测和指定计算设备 ---")

def get_apple_gpu_device():
    """
    智能检测并返回可用的最佳设备：优先MPS，否则返回CPU。
    """
    # 检查PyTorch版本是否支持MPS，并且当前系统是Darwin (macOS)
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        print("检测到Apple Metal (MPS) 后端可用，将使用MPS设备。")
        return torch.device('mps')
    print("MPS不可用，将使用CPU。")
    return torch.device('cpu')

# --- 演示设备选择 ---
# 在你的M1/M2/M3 Mac上，此函数会返回 device(type='mps')
device = get_apple_gpu_device()
print("当前选择的设备:", device)
print("-" * 50)


# =============================================================================
# Part 2: 张量与MPS设备
# -----------------------------------------------------------------------------
# 数据和模型必须在同一个设备上才能进行计算。
# =============================================================================
print("--- 2. 张量与MPS设备 ---")

# --- 查询张量所在的设备 ---
x_cpu = torch.tensor([1, 2, 3])
print("默认情况下，张量创建在:", x_cpu.device)

# --- 将张量移动到MPS设备 ---
# 使用 .to(device) 是最通用的方法
X_mps = x_cpu.to(device)
print(f"\n将张量移动到 {device} 后:\n", X_mps)
print(f"确认张量所在设备: {X_mps.device}")

# 也可以在创建时直接指定设备
Y_mps = torch.ones(2, 3, device=device)
print("\n直接在MPS上创建的张量:\n", Y_mps)
print("-" * 50)


# =============================================================================
# Part 3: 神经网络与MPS设备
# -----------------------------------------------------------------------------
# 模型的所有参数也需要移动到MPS设备上。
# =============================================================================
print("--- 3. 神经网络与MPS设备 ---")
# --- 将模型移动到MPS设备 ---
# .to(device) 方法会递归地将模型的所有参数和缓冲区移动到指定设备
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=device)
print(f"模型已移动到: {next(net.parameters()).device}")

# --- 在MPS设备上进行计算 ---
# 当输入张量 Y_mps 和模型 net 都在同一个设备上时，
# 前向传播就会在MPS设备上自动执行。
output = net(Y_mps)
print("\n在MPS上计算的输出:\n", output)
print("确认输出张量所在的设备:", output.device)
print("-" * 50)