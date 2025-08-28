import torch

# --- 验证RNN中隐状态计算的等价性 ---

# 假设我们有一个小批量数据，批量大小为3，输入特征维度为1
# X_t 的形状是 (批量大小, 输入维度) -> (3, 1)
X = torch.normal(0, 1, (3, 1))

# 隐藏层的大小(隐藏单元数量)为4
# W_xh 是输入到隐藏层的权重矩阵
# 它的形状是 (输入维度, 隐藏单元数) -> (1, 4)
W_xh = torch.normal(0, 1, (1, 4))

# H_{t-1} 是上一个时间步的隐状态
# 它的形状是 (批量大小, 隐藏单元数) -> (3, 4)
H = torch.normal(0, 1, (3, 4))

# W_hh 是从上一个隐状态到当前隐状态的权重矩阵
# 它的形状是 (隐藏单元数, 隐藏单元数) -> (4, 4)
W_hh = torch.normal(0, 1, (4, 4))

# --- 方法一：按照原始公式分别计算再相加 ---
# H_t = X_t @ W_xh + H_{t-1} @ W_hh
# (3, 1) @ (1, 4) -> (3, 4)
# (3, 4) @ (4, 4) -> (3, 4)
# 最终结果形状为 (3, 4)
output1 = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
print("--- 方法一：分别计算后相加 ---")
print(output1)
print("形状:", output1.shape)


# --- 方法二：先拼接输入和权重，再进行一次矩阵乘法 ---
# 1. 拼接输入 X_t 和 H_{t-1}
# 沿着维度1（特征维度）拼接
# (3, 1) 和 (3, 4) 拼接后得到 (3, 5)
input_concat = torch.cat((X, H), dim=1)

# 2. 拼接权重 W_xh 和 W_hh
# 沿着维度0拼接，因为它们分别对应于 X_t 和 H_{t-1} 的输入
# (1, 4) 和 (4, 4) 拼接后得到 (5, 4)
weight_concat = torch.cat((W_xh, W_hh), dim=0)

# 3. 将拼接后的矩阵相乘
# (3, 5) @ (5, 4) -> (3, 4)
output2 = torch.matmul(input_concat, weight_concat)
print("\n--- 方法二：拼接后一次计算 ---")
print(output2)
print("形状:", output2.shape)


# --- 验证两种方法的结果是否相等 ---
# 使用 torch.allclose 来比较两个浮点数张量是否在数值上足够接近
are_equal = torch.allclose(output1, output2)
print(f"\n两种计算方法的结果是否相等: {are_equal}")