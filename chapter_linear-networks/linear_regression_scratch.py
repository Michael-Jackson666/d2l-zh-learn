import torch
import random

# =============================================================================
# 步骤 1: 定义数据生成函数
# 我们将创建一个人工数据集，这样我们就能知道真实答案，以便验证模型。
# =============================================================================
def synthetic_data(w, b, num_examples):
    """
    生成人工数据集 y = Xw + b + 噪声。

    参数:
    w (torch.Tensor): 真实的权重向量。
    b (float): 真实的偏置。
    num_examples (int): 样本数量。

    返回:
    X (torch.Tensor): 特征矩阵, 形状为 (num_examples, len(w))。
    y (torch.Tensor): 标签向量, 形状为 (num_examples, 1)。
    """
    # 生成服从标准正态分布的特征 X
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 根据真实 w 和 b 计算 y
    y = torch.matmul(X, w) + b
    # 加入一些随机噪声，模拟真实世界的数据不完美性
    y += torch.normal(0, 0.01, y.shape)
    # 返回特征和 reshape 成列向量的标签
    return X, y.reshape((-1, 1))

# =============================================================================
# 步骤 2: 定义数据迭代器 (小批量加载)
# 在训练时，我们不是一次处理所有数据，而是一小批一小批地处理。
# 这个函数负责打乱数据并按批次返回。
# =============================================================================
def data_iter(batch_size, features, labels):
    """
    将数据集随机打乱，并按指定批量大小返回数据。

    参数:
    batch_size (int): 每个小批量的大小。
    features (torch.Tensor): 完整的特征矩阵。
    labels (torch.Tensor): 完整的标签向量。

    返回:
    一个生成器(generator)，每次迭代返回一小批特征和标签。
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机打乱样本的索引
    random.shuffle(indices)
    # 从 0 开始，每次前进一个步长 batch_size
    for i in range(0, num_examples, batch_size):
        # 获取当前批次的索引列表
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        # 'yield' 关键字让这个函数成为一个生成器，节省内存
        yield features[batch_indices], labels[batch_indices]

# =============================================================================
# 步骤 3: 定义模型、损失函数和优化器
# 这三部分是机器学习模型的核心组件。
# =============================================================================
def linreg(X, w, b):
    """线性回归模型。"""
    # 实现公式: y = Xw + b
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方损失函数。"""
    # 实现公式: (预测值 - 真实值)^2 / 2
    # reshape 是为了确保 y_hat 和 y 的形状一致
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降 (SGD) 优化器。"""
    # `torch.no_grad()`: 接下来的操作不应被 PyTorch 追踪梯度，因为我们是手动更新参数
    with torch.no_grad():
        for param in params:
            # 更新参数: param = param - 学习率 * 梯度 / 批量大小
            # 除以 batch_size 是为了让学习率对批量大小不那么敏感
            param -= lr * param.grad / batch_size
            # 清空梯度，为下一次计算做准备，这是至关重要的一步
            param.grad.zero_()

# =============================================================================
# 步骤 4: 训练过程
# 这是所有组件协同工作的地方。
# =============================================================================

# --- 4.1 设置超参数 ---
lr = 0.03            # 学习率 (Learning Rate): 每次更新参数的步长
num_epochs = 10      # 迭代周期数 (Epochs): 完整遍历数据集的次数
batch_size = 10      # 批量大小 (Batch Size): 每次更新参数所用的样本数
net = linreg         # 指定使用的模型
loss = squared_loss  # 指定使用的损失函数

# --- 4.2 生成数据集 ---
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# --- 4.3 初始化模型参数 ---
# 随机初始化权重 w，初始化偏置 b 为 0
# `requires_grad=True`: 告诉 PyTorch 需要计算这些张量的梯度
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# --- 4.4 主训练循环 ---
print("开始训练...")
for epoch in range(num_epochs):
    # 对于每一个 epoch，遍历所有数据
    for X, y in data_iter(batch_size, features, labels):
        # 1. 前向传播：计算小批量的损失
        # net(X, w, b) 是模型预测值 y_hat
        # loss(...) 计算 y_hat 和真实值 y 之间的差距
        l = loss(net(X, w, b), y)
        
        # 2. 反向传播：计算梯度
        # l 是一个形状为 (batch_size, 1) 的向量，需要先求和变成标量才能调用 backward()
        l.sum().backward()
        
        # 3. 更新参数：使用 SGD 优化器更新 w 和 b
        sgd([w, b], lr, batch_size)
    
    # 在每个 epoch 结束后，在整个数据集上评估损失，监控训练进展
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'Epoch {epoch + 1}, Loss {float(train_l.mean()):f}')

# --- 4.5 验证结果 ---
print("\n训练完成！")
print(f'我们学到的权重 w 的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'我们学到的偏置 b 的估计误差: {true_b - b}')