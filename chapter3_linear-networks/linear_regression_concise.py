import torch
from torch.utils import data
from torch import nn

# =============================================================================
# 步骤 1: 生成数据集
# 这一步与“从零实现”版本完全相同。我们依然需要一个数据集。
# =============================================================================
def synthetic_data(w, b, num_examples):
    """生成人工数据集 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# --- 生成数据集 ---
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# =============================================================================
# 步骤 2: 使用 PyTorch API 读取数据
# 对比“从零实现”：我们不再需要手动编写 data_iter 函数。
# PyTorch 提供了强大的 DataLoader，可以帮我们自动完成批量加载和随机打乱。
# =============================================================================
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    # 将特征和标签张量打包成一个 TensorDataset
    dataset = data.TensorDataset(*data_arrays)
    # 使用 DataLoader 来创建数据迭代器
    # shuffle=True 会在每个 epoch 开始时自动打乱数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# --- 创建数据迭代器 ---
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# =============================================================================
# 步骤 3: 使用 PyTorch API 定义模型、损失函数和优化器
# 对比“从零实现”：这是最能体现框架优势的地方。
# 我们不再需要手动定义函数，而是直接实例化框架提供的类。
# =============================================================================

# --- 3.1 定义模型 ---
# nn.Sequential 是一个容器，可以按顺序将多个层组合在一起。
# nn.Linear(2, 1) 定义了一个全连接层，输入特征维度为 2，输出维度为 1。
# 这个层自动包含了权重 w 和偏置 b。
net = nn.Sequential(nn.Linear(2, 1))

# --- 3.2 初始化模型参数 ---
# 通过 net[0] 访问模型的第一层 (也就是我们唯一的 Linear 层)
# 使用 .weight.data 和 .bias.data 直接访问参数的数据
# .normal_(0, 0.01) 和 .fill_(0) 是原地操作，直接修改参数值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# --- 3.3 定义损失函数 ---
# nn.MSELoss() 是 PyTorch 内置的均方误差损失函数。
# 它会自动计算批量损失的平均值。
loss = nn.MSELoss()

# --- 3.4 定义优化器 ---
# torch.optim.SGD 是 PyTorch 内置的随机梯度下降优化器。
# 我们需要将模型的参数 (通过 net.parameters() 获取) 和学习率传入。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# =============================================================================
# 步骤 4: 训练过程
# 对比“从零实现”：训练循环的逻辑变得更清晰、更标准化。
# =============================================================================

# --- 设置超参数 ---
num_epochs = 3

# --- 主训练循环 ---
print("开始训练...")
for epoch in range(num_epochs):
    # DataLoader 非常方便，可以直接在 for 循环中迭代
    for X, y in data_iter:
        # 1. 前向传播：计算损失
        # net(X) 直接调用模型得到预测值
        # loss(...) 计算预测值和真实值之间的损失
        l = loss(net(X), y)
        
        # 2. 反向传播：计算梯度
        # 在计算梯度前，先清空旧的梯度，这是 PyTorch 的标准流程
        trainer.zero_grad()
        l.backward()
        
        # 3. 更新参数
        # 调用优化器的 step() 方法，它会自动根据梯度更新所有参数
        trainer.step()
    
    # 在每个 epoch 结束后，评估在整个数据集上的损失
    l = loss(net(features), labels)
    print(f'Epoch {epoch + 1}, Loss {l:f}')

# --- 验证结果 ---
print("\n训练完成！")
# 从模型中提取学到的权重和偏置
w = net[0].weight.data
b = net[0].bias.data
print(f'我们学到的权重 w 的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'我们学到的偏置 b 的估计误差: {true_b - b}')