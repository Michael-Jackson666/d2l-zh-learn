import torch
import torch.nn.functional as F
from torch import nn

# =============================================================================
# Part 1: 不带参数的自定义层
# -----------------------------------------------------------------------------
# 这种层通常用于实现一些固定的数学运算。
# =============================================================================
print("--- 1. 不带参数的自定义层 ---")
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # 功能：从输入张量中减去其均值，实现中心化
        return X - X.mean()

# --- 测试 CenteredLayer ---
layer = CenteredLayer()
output_centered = layer(torch.FloatTensor([1, 2, 3, 4, 5]))
print("CenteredLayer的输出:", output_centered)
# 验证均值是否接近于0
print("输出的均值:", output_centered.mean())

# --- 将自定义层集成到模型中 ---
net_with_centered = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net_with_centered(torch.rand(4, 8))
# 验证输出的均值是否接近于0
print("集成到Sequential后，输出的均值:", Y.mean())
print("-" * 50)


# =============================================================================
# Part 2: 带参数的自定义层
# -----------------------------------------------------------------------------
# 这是构建新颖模型结构的核心，我们需要手动定义和管理层的参数。
# =============================================================================
print("--- 2. 带参数的自定义层 ---")
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # nn.Parameter 将一个张量注册为模型的参数。
        # 这意味着它会被 net.parameters() 识别，并参与梯度计算和优化。
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        # (修正!) 直接使用 self.weight 和 self.bias 进行计算，
        # 以确保计算图能够建立，从而支持反向传播。
        linear = torch.matmul(X, self.weight) + self.bias
        return F.relu(linear)

# --- 测试 MyLinear 层 ---
linear = MyLinear(5, 3)
print("自定义MyLinear层的权重:", linear.weight)
output_mylinear = linear(torch.rand(2, 5))
print("MyLinear层的输出:\n", output_mylinear)

# --- 将带参数的自定义层集成到模型中 ---
net_with_mylinear = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
output_integrated = net_with_mylinear(torch.rand(2, 64))
print("集成MyLinear到Sequential后的输出:\n", output_integrated)
print("-" * 50)


# =============================================================================
# Part 3: 练习题实现
# =============================================================================

# --- 练习 1: 实现一个二次变换层 ---
class QuadraticLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 定义可学习的参数 W，形状为 (in_features, in_features, out_features)
        self.weights = nn.Parameter(
            torch.randn(in_features, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # x 的形状是 (batch_size, in_features)
        # 使用 einsum 实现高效计算 y_k = x^T W_k x + b_k
        # 'bi,ijk,bj->bk' 的含义是:
        # b: batch 维度
        # i, j: 输入特征的维度
        # k: 输出特征的维度
        quadratic_term = torch.einsum('bi,ijk,bj->bk', x, self.weights, x)
        return quadratic_term + self.bias

print("--- 练习1: QuadraticLayer ---")
quad_layer = QuadraticLayer(in_features=5, out_features=3)
input_tensor_q = torch.rand(2, 5) # batch_size=2, in_features=5
output_tensor_q = quad_layer(input_tensor_q)

print(f"输入形状: {input_tensor_q.shape}")
print(f"二次层输出形状: {output_tensor_q.shape}") # 预期输出 (2, 3)
print("-" * 50)


# --- 练习 2: 实现一个傅立叶系数层 ---
class FourierLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x 的形状是 (batch_size, signal_length)
        # 对最后一个维度（信号长度）进行实数傅立叶变换
        fft_coeffs = torch.fft.rfft(x, dim=-1)
        # rfft 已经只返回了非冗余的前半部分系数
        return fft_coeffs

print("--- 练习2: FourierLayer ---")
fourier_layer = FourierLayer()
# 创建一个 batch_size=2, 信号长度=100 的输入
input_signal = torch.randn(2, 100) 
output_coeffs = fourier_layer(input_signal)

print(f"输入信号形状: {input_signal.shape}")
# 预期输出形状 (2, 100 // 2 + 1) = (2, 51)
print(f"傅立叶层输出形状: {output_coeffs.shape}")
print(f"输出的数据类型: {output_coeffs.dtype}") # 预期输出 torch.complex64
print("-" * 50)