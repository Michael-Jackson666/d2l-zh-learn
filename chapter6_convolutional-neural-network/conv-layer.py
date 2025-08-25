import torch
from torch import nn

# =============================================================================
# Part 1: 二维互相关运算 (Cross-Correlation)
# -----------------------------------------------------------------------------
# 这是卷积神经网络中“卷积”操作的数学核心。
# =============================================================================
print("--- 1. 二维互相关运算 ---")

def corr2d(X, K):
    """手动实现二维互相关运算"""
    # 获取卷积核的高和宽
    h, w = K.shape
    # 初始化输出张量，其大小由输入和卷积核大小决定
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 遍历输出的每个像素位置
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 提取输入中对应的窗口，与卷积核进行按元素乘法并求和
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# --- 测试 corr2d ---
X_corr = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K_corr = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
output_corr = corr2d(X_corr, K_corr)
print("输入 X:\n", X_corr)
print("卷积核 K:\n", K_corr)
print("互相关输出 Y:\n", output_corr)
print("-" * 50)


# =============================================================================
# Part 2: 自定义二维卷积层
# =============================================================================
print("--- 2. 自定义二维卷积层 ---")
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # 将权重和偏置注册为可学习参数
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 在前向传播中调用我们实现的 corr2d 函数
        return corr2d(x, self.weight) + self.bias

# =============================================================================
# Part 3: 边缘检测示例
# -----------------------------------------------------------------------------
# 演示一个固定的卷积核如何从图像中提取特定特征（如垂直边缘）。
# =============================================================================
print("--- 3. 边缘检测示例 ---")
# 构造一个 6x8 的黑白图像 (0=黑, 1=白)
X_edge = torch.ones((6, 8))
X_edge[:, 2:6] = 0
print("原始图像 X:\n", X_edge)

# 构造一个用于检测垂直边缘的 1x2 卷积核
K_edge = torch.tensor([[1.0, -1.0]])
Y_edge = corr2d(X_edge, K_edge)
print("应用垂直边缘检测核后的输出 Y:\n", Y_edge)
# 输出中的 1 表示从白到黑的边缘，-1 表示从黑到白的边缘

# --- 验证该核无法检测水平边缘 ---
# 将图像转置后，垂直边缘变为水平边缘
Y_edge_transposed = corr2d(X_edge.t(), K_edge)
print("对转置后的图像应用该核，输出 Y.t():\n", Y_edge_transposed)
# 输出全为0，说明无法检测水平边缘
print("-" * 50)


# =============================================================================
# Part 4: 通过学习得到卷积核
# -----------------------------------------------------------------------------
# 演示卷积核的参数是可以通过梯度下降从数据中学习得到的。
# =============================================================================
print("--- 4. 学习卷积核 ---")
# 目标：从输入 X_edge 和输出 Y_edge 中，学习出接近 K_edge 的卷积核

# 构造一个内置的二维卷积层
# 输入通道=1, 输出通道=1, 卷积核大小=(1, 2), 无偏置
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# PyTorch的卷积层需要4D输入: (批量大小, 通道数, 高, 宽)
X_learn = X_edge.reshape((1, 1, 6, 8))
Y_learn = Y_edge.reshape((1, 1, 6, 7))

lr = 3e-2  # 学习率
for i in range(10):
    Y_hat = conv2d(X_learn)
    l = (Y_hat - Y_learn) ** 2 # 平方损失
    conv2d.zero_grad()
    l.sum().backward()
    # 手动更新权重 (等同于 optimizer.step())
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'Epoch {i+1}, loss {l.sum():.3f}')

print("\n学习到的卷积核权重:")
print(conv2d.weight.data.reshape((1, 2)))
print("原始的卷积核权重:")
print(K_edge)
print("-" * 50)