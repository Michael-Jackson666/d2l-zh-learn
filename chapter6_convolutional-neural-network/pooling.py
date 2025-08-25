import torch
from torch import nn

# =============================================================================
# Part 1: 从零实现汇聚层
# -----------------------------------------------------------------------------
# 汇聚层没有可学习的参数，它对输入窗口内的值执行一个固定的计算。
# =============================================================================
print("--- 1. 从零实现汇聚层 ---")

def pool2d(X, pool_size, mode='max'):
    """手动实现二维汇聚层"""
    p_h, p_w = pool_size
    # 初始化输出张量
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    # 遍历输出的每个像素位置
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 提取输入窗口
            window = X[i: i + p_h, j: j + p_w]
            if mode == 'max':
                # 计算窗口内的最大值
                Y[i, j] = window.max()
            elif mode == 'avg':
                # 计算窗口内的平均值
                Y[i, j] = window.mean()
    return Y

# --- 测试我们实现的汇聚层 ---
X_pool = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

# 验证最大汇聚
max_pool_output = pool2d(X_pool, (2, 2), mode='max')
print("输入 X:\n", X_pool)
print("手动实现的最大汇聚输出:\n", max_pool_output)

# 验证平均汇聚
avg_pool_output = pool2d(X_pool, (2, 2), mode='avg')
print("手动实现的平均汇聚输出:\n", avg_pool_output)
print("-" * 50)


# =============================================================================
# Part 2: 使用内置层演示填充和步幅
# -----------------------------------------------------------------------------
# PyTorch 的 nn.MaxPool2d 和 nn.AvgPool2d 提供了高效的实现。
# =============================================================================
print("--- 2. 填充和步幅 ---")
X_torch = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print("4D 输入张量 X:\n", X_torch)

# --- 默认步幅 ---
# 默认情况下，步幅与汇聚窗口大小相同
# 输入 4x4, 窗口 3x3, 步幅 3x3 -> 输出 1x1
pool_default = nn.MaxPool2d(3)
print("\n使用 3x3 窗口，默认步幅 (3,3) 的输出:\n", pool_default(X_torch))

# --- 自定义步幅和填充 ---
# 输入 4x4, 窗口 3x3, 填充 1, 步幅 2
# 输出尺寸: floor((4 + 2*1 - 3)/2) + 1 = floor(3.5) + 1 = 3 -> (PyTorch公式)
# (4 + 2*1 - 3 + 2 -1)/2 = 4 -> (d2l公式)
# Pytorch: H_out = floor((H_in + 2*padding - kernel_size)/stride) + 1
# H_out = floor((4 + 2*1 - 3)/2) + 1 = floor(1.5)+1 = 2
output_h = ((4 + 2*1 - 3) // 2) + 1 # (4+2-3)//2+1 = 1+1 = 2
pool_custom = nn.MaxPool2d(3, padding=1, stride=2)
print("使用 3x3 窗口, padding=1, stride=2 的输出:\n", pool_custom(X_torch))

# --- 非对称窗口、填充和步幅 ---
pool_asymmetric = nn.MaxPool2d(kernel_size=(2, 3), padding=(0, 1), stride=(2, 3))
print("使用非对称窗口、填充和步幅的输出:\n", pool_asymmetric(X_torch))
print("-" * 50)


# =============================================================================
# Part 3: 多通道汇聚
# -----------------------------------------------------------------------------
# 汇聚层在每个输入通道上独立运算，因此输出通道数等于输入通道数。
# =============================================================================
print("--- 3. 多通道汇聚 ---")
# 构建一个2通道的输入张量
X_multi_channel = torch.cat((X_torch, X_torch + 1), 1)
print("2通道输入张量 X (形状 {}):\n".format(X_multi_channel.shape), X_multi_channel)

# 应用汇聚层
pool_multi = nn.MaxPool2d(3, padding=1, stride=2)
output_multi = pool_multi(X_multi_channel)
print("对2通道输入进行汇聚后的输出 (形状 {}):\n".format(output_multi.shape), output_multi)
print("-" * 50)