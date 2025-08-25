import torch
from torch import nn
from torch.nn import functional as F

# =============================================================================
# Part 1: 使用 nn.Sequential (回顾)
# nn.Sequential 是一个方便的容器，用于按顺序堆叠层。
# =============================================================================
print("--- 1. 使用 nn.Sequential ---")
# 定义一个包含两个线性层和一个ReLU激活函数的序列块
net_sequential = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

# 创建一个随机输入张量
X = torch.rand(2, 20)
# 通过网络进行前向传播
output_sequential = net_sequential(X)
print("nn.Sequential的输出:\n", output_sequential)
print("-" * 50)


# =============================================================================
# Part 2: 自定义块 (继承 nn.Module)
# 这是构建复杂模型的标准方式。
# =============================================================================
print("--- 2. 自定义MLP块 ---")
class MLP(nn.Module):
    # __init__ 方法用于定义模型中需要用到的层和参数
    def __init__(self):
        # 必须调用父类的构造函数
        super().__init__()
        # 定义一个隐藏层 (全连接层)
        self.hidden = nn.Linear(20, 256)
        # 定义一个输出层 (全连接层)
        self.out = nn.Linear(256, 10)

    # forward 方法定义了数据如何流过网络（前向传播逻辑）
    def forward(self, X):
        # 隐藏层的计算：先线性变换，然后通过ReLU激活函数
        hidden_output = F.relu(self.hidden(X))
        # 输出层的计算
        return self.out(hidden_output)

# 实例化我们自定义的MLP
net_custom_mlp = MLP()
output_custom_mlp = net_custom_mlp(X)
print("自定义MLP的输出:\n", output_custom_mlp)
print("-" * 50)


# =============================================================================
# Part 3: 自定义 Sequential 块
# 通过自己实现一个Sequential，来理解其内部工作原理。
# =============================================================================
print("--- 3. 自定义MySequential块 ---")
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # 遍历传入的所有模块 (层)
        for idx, module in enumerate(args):
            # _modules 是 nn.Module 的一个特殊属性 (有序字典)
            # PyTorch会自动注册添加到 _modules 中的模块，以便追踪其参数
            self._modules[str(idx)] = module

    def forward(self, X):
        # 按照添加的顺序，依次将输入X通过每一个模块
        for block in self._modules.values():
            X = block(X)
        return X

# 使用我们自己的MySequential来构建网络
net_my_sequential = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
output_my_sequential = net_my_sequential(X)
print("MySequential的输出:\n", output_my_sequential)
print("-" * 50)


# =============================================================================
# Part 4: 在 forward 方法中实现更灵活的逻辑
# 这展示了自定义块相比Sequential的巨大优势。
# =============================================================================
print("--- 4. 具有灵活逻辑的自定义块 ---")
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个常量权重，它不是模型的参数，不会被训练更新
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        # 定义一个可训练的线性层
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        # 第一次线性变换
        X = self.linear(X)
        # 与常量权重进行矩阵乘法，并加上偏置1，然后通过ReLU
        # 这里展示了任意数学运算的集成
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用同一个线性层，这实现了参数共享
        X = self.linear(X)
        # 加入Python的控制流 (while循环)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net_flexible = FixedHiddenMLP()
output_flexible = net_flexible(X)
print("FixedHiddenMLP的输出:\n", output_flexible)
print("-" * 50)


# =============================================================================
# Part 5: 块的嵌套
# 块可以像积木一样任意组合和嵌套。
# =============================================================================
print("--- 5. 块的嵌套 ---")
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net 本身就是一个Sequential块
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

# 构建一个“嵌合体”模型，将各种自定义块和标准层嵌套在一起
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
output_chimera = chimera(X)
print("嵌套块 (Chimera) 的输出:\n", output_chimera)
print("-" * 50)