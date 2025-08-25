import torch
from torch import nn
from torch.nn import functional as F

# =============================================================================
# Part 1: 加载和保存单个或多个张量
# -----------------------------------------------------------------------------
# 这是最基础的文件 IO 操作，适用于保存中间计算结果或简单数据。
# =============================================================================
print("--- 1. 加载和保存张量 ---")

# --- 保存和加载单个张量 ---
x = torch.arange(4)
torch.save(x, 'x-file') # 保存张量 x 到文件 'x-file'

x2 = torch.load('x-file') # 从文件加载
print("加载的单个张量:", x2)

# --- 保存和加载张量列表 ---
y = torch.zeros(4)
torch.save([x, y], 'x-files') # 保存一个包含 x 和 y 的列表
x2, y2 = torch.load('x-files') # 加载后解包
print("加载的张量列表:", (x2, y2))

# --- 保存和加载张量字典 ---
# 这种方式更常用，因为键可以提供描述性信息
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict') # 保存字典
mydict2 = torch.load('mydict') # 加载字典
print("加载的张量字典:", mydict2)
print("-" * 50)


# =============================================================================
# Part 2: 加载和保存模型参数 (State Dictionary)
# -----------------------------------------------------------------------------
# 这是保存和加载模型最常用、最推荐的方法。
# 它只保存模型的可学习参数(权重和偏置)，不保存模型结构。
# =============================================================================
print("--- 2. 加载和保存模型参数 ---")

# --- 首先，定义一个模型架构 ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# --- 保存模型参数 ---
# net.state_dict() 返回一个包含模型所有参数的有序字典
# 字典的键是参数的名称 (例如 'hidden.weight')，值是参数的张量
torch.save(net.state_dict(), 'mlp.params')
print("已将模型参数保存到 'mlp.params'")

# --- 加载模型参数 ---
# 1. 必须先创建一个相同结构的模型实例
clone = MLP()
# 2. 调用 .load_state_dict() 方法加载参数
clone.load_state_dict(torch.load('mlp.params'))
# 3. 将模型设置为评估模式 (这会关闭Dropout和BatchNorm等)
clone.eval()
print("已创建新模型实例并加载参数。")

# --- 验证加载是否成功 ---
# 使用相同的输入，比较原始模型和加载后模型的输出是否一致
Y_clone = clone(X)
are_equal = torch.equal(Y, Y_clone)
print("原始模型输出和加载后模型输出是否完全一致:", are_equal)
print("-" * 50)