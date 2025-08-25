import torch
from torch import nn
from torch.nn import functional as F

# =============================================================================
# Part 1: 参数访问
# -----------------------------------------------------------------------------
# 我们将学习如何访问模型中特定层、特定名称的参数。
# =============================================================================
print("--- 1. 参数访问 ---")
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

# --- 访问特定层的参数 ---
# nn.Sequential中的层可以通过索引访问
print("输出层 (net[2]) 的 state_dict:")
print(net[2].state_dict())

# --- 访问参数的具体数值和梯度 ---
# 参数本身是 torch.nn.parameter.Parameter 类的实例
print("\n输出层偏置的类型:", type(net[2].bias))
print("输出层偏置对象:", net[2].bias)
# 使用 .data 访问参数的数值 (张量)，不带梯度信息
print("输出层偏置的数值 (.data):", net[2].bias.data)
# 使用 .grad 访问参数的梯度 (此时为None，因为还未反向传播)
print("输出层权重的梯度 (.grad):", net[2].weight.grad)

# --- 一次性访问所有参数 ---
# 使用 .named_parameters() 遍历网络中所有被注册的参数
print("\n网络中所有参数 (名称, 形状):")
for name, param in net.named_parameters():
    print(f"  ({name}, {param.shape})")

# --- 通过名称访问特定参数 ---
# state_dict() 返回一个包含所有参数名称和数值的有序字典
print("\n通过名称访问输出层偏置:", net.state_dict()['2.bias'].data)
print("-" * 50)


# =============================================================================
# Part 2: 访问嵌套块中的参数
# -----------------------------------------------------------------------------
# 演示在更复杂的嵌套结构中，参数是如何被命名的。
# =============================================================================
print("--- 2. 访问嵌套块的参数 ---")
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 使用 add_module 添加带名字的子块
        net.add_module(f'block_{i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print("嵌套网络结构:")
print(rgnet)

# --- 通过层级索引访问深层参数 ---
# rgnet[0] -> block2()
# rgnet[0][1] -> 'block_1'
# rgnet[0][1][0] -> 第一个 nn.Linear
# rgnet[0][1][0].bias -> 偏置参数
print("\n访问 rgnet[0][1][0].bias.data:")
print(rgnet[0][1][0].bias.data)
print("-" * 50)


# =============================================================================
# Part 3: 参数初始化
# -----------------------------------------------------------------------------
# PyTorch提供了多种内置初始化方法，也可以自定义。
# =============================================================================
print("--- 3. 参数初始化 ---")

# --- 3.1 内置初始化 ---
def init_normal(m):
    # isinstance 检查一个对象是否是某个类的实例
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01) # 正态分布初始化
        nn.init.zeros_(m.bias) # 零初始化

net.apply(init_normal) # .apply() 会递归地将函数应用到网络所有子模块
print("正态分布初始化后，net[0].weight.data[0]:\n", net[0].weight.data[0])
print("正态分布初始化后，net[0].bias.data[0]:\n", net[0].bias.data[0])

def init_constant(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1) # 常数初始化
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print("常数初始化后，net[0].weight.data[0]:\n", net[0].weight.data[0])
print("常数初始化后，net[0].bias.data[0]:\n", net[0].bias.data[0])

# --- 3.2 对不同层使用不同初始化 ---
def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight) # Xavier初始化

def init_42(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 42) # 另一个常数初始化

net[0].apply(init_xavier)
net[2].apply(init_42)
print("对不同层应用不同初始化:")
print("  net[0].weight.data[0] (Xavier):\n", net[0].weight.data[0])
print("  net[2].weight.data[0] (Constant 42):\n", net[2].weight.data[0])

# --- 3.3 自定义初始化 ---
def my_init(m):
    if isinstance(m, nn.Linear):
        print("  正在初始化:", m)
        # 均匀分布初始化
        nn.init.uniform_(m.weight, -10, 10)
        # 根据条件将部分权重置零
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print("自定义初始化后，net[0].weight.data[:2]:\n", net[0].weight.data[:2])
net[0].weight.data[:] += 1
net[0].weight.data[0,0] = 42
net[0].weight.data[0]
print("-" * 50)


# =============================================================================
# Part 4: 参数共享 (绑定)
# -----------------------------------------------------------------------------
# 让网络中的多个层使用完全相同的参数。
# =============================================================================
print("--- 4. 参数共享 ---")
# 首先创建一个要共享的层
shared = nn.Linear(8, 8)
# 在Sequential中多次使用同一个实例
net_shared = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                           shared, nn.ReLU(),
                           shared, nn.ReLU(),
                           nn.Linear(8, 1))

net_shared(X)
print("共享前，net_shared[2] 和 net_shared[4] 的权重是否相同:",
      torch.equal(net_shared[2].weight.data, net_shared[4].weight.data))

# 修改其中一个层的权重
net_shared[2].weight.data[0, 0] = 100
# 另一个层的权重也随之改变，因为它们是同一个对象
print("修改后，net_shared[2] 和 net_shared[4] 的权重是否相同:",
      torch.equal(net_shared[2].weight.data, net_shared[4].weight.data))
print("-" * 50)