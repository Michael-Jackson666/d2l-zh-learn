import torch
from torch import nn

# =============================================================================
# Part 1: 填充 (Padding)
# -----------------------------------------------------------------------------
# 填充用于控制卷积后的输出尺寸，最常见的用途是保持输入输出尺寸不变。
# =============================================================================
print("--- 1. 填充 (Padding) ---")

# --- 辅助函数，用于简化卷积计算和形状查看 ---
def comp_conv2d(conv2d_layer, X):
    """
    对输入X执行卷积，并返回输出的形状。
    会自动处理批量和通道维度的添加与移除。
    """
    # PyTorch的卷积层需要4D输入: (批量大小, 通道数, 高, 宽)
    # 这里我们假设批量大小和通道数都为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d_layer(X)
    # 返回输出的空间维度 (高, 宽)
    return Y.reshape(Y.shape[2:])

# --- 示例1: 保持尺寸不变的填充 ("Same" Padding) ---
# 输入形状: 8x8
# 卷积核形状: 3x3
# 步幅: 1 (默认)
# 根据公式 (n_h - k_h + p_h + 1)，为了使输出也是8，需要 8 - 3 + p_h + 1 = 8 => p_h = 2。
# PyTorch的 `padding` 参数指定的是单边的填充量，所以 padding=1 意味着上下左右各填充1行/列，总共增加2行2列。
conv2d_same = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))

output_shape_same = comp_conv2d(conv2d_same, X).shape
print(f"输入形状: {X.shape}")
print(f"使用 kernel_size=3, padding=1 后，输出形状: {output_shape_same}")

# --- 示例2: 非对称填充 ---
# 使用一个非正方形的卷积核，并分别指定高度和宽度的填充
# 卷积核: 5x3, 填充: (2, 1) -> (p_h=2, p_w=1)
# 输出高度: 8 - 5 + (2*2) + 1 = 8
# 输出宽度: 8 - 3 + (1*2) + 1 = 8
conv2d_asymmetric = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
output_shape_asymmetric = comp_conv2d(conv2d_asymmetric, X).shape
print(f"使用 kernel_size=(5,3), padding=(2,1) 后，输出形状: {output_shape_asymmetric}")
print("-" * 50)


# =============================================================================
# Part 2: 步幅 (Stride)
# -----------------------------------------------------------------------------
# 步幅用于对特征图进行下采样 (downsampling)，从而减小其空间维度。
# =============================================================================
print("--- 2. 步幅 (Stride) ---")

# --- 示例1: 步幅为2 ---
# 这将使输出尺寸大约减半。
# 输出高度: floor((8 - 3 + 2*1 + 2) / 2) = floor(9 / 2) = 4
# 输出宽度: floor((8 - 3 + 2*1 + 2) / 2) = floor(9 / 2) = 4
conv2d_stride2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
output_shape_stride2 = comp_conv2d(conv2d_stride2, X).shape
print(f"输入形状: {X.shape}")
print(f"使用 kernel_size=3, padding=1, stride=2 后，输出形状: {output_shape_stride2}")

# --- 示例2: 复杂的填充和步幅组合 ---
# 卷积核: 3x5, 填充: (0, 1), 步幅: (3, 4)
# 输出高度: floor((8 - 3 + 0*2 + 3) / 3) = floor(8 / 3) = 2
# 输出宽度: floor((8 - 5 + 1*2 + 4) / 4) = floor(9 / 4) = 2
conv2d_complex = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
output_shape_complex = comp_conv2d(conv2d_complex, X).shape
print(f"使用复杂组合后，输出形状: {output_shape_complex}")
print("-" * 50)