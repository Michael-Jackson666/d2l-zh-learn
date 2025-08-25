import torch
from torch import nn
from torch.nn import functional as F

# =============================================================================
# Part 1: 加载和保存单个或多个张量
# -----------------------------------------------------------------------------
print("--- 1. 加载和保存张量 ---")

# --- 保存和加载单个张量 ---
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print("加载的单个张量:", x2)

# --- 保存和加载张量列表 ---
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print("加载的张量列表:", (x2, y2))

# --- 保存和加载张量字典 ---
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print("加载的张量字典:", mydict2)
print("-" * 50)


# =============================================================================
# Part 2: 加载和保存模型参数 (State Dictionary)
# -----------------------------------------------------------------------------
print("--- 2. 加载和保存模型参数 ---")

class MLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, output_dim=10):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')
print("已将模型参数保存到 'mlp.params'")

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
print("已创建新模型实例并加载参数。")

Y_clone = clone(X)
are_equal = torch.equal(Y, Y_clone)
print("原始模型输出和加载后模型输出是否完全一致:", are_equal)
print("-" * 50)

# =============================================================================
# Part 3: 练习题实现
# =============================================================================
print("--- 3. 练习题实现 ---")

# --- 练习 2: 复用网络的一部分 ---
print("--- 练习 2: 复用网络的一部分 ---")
# 假设 net 是我们已经训练好的预训练模型
# 我们想复用它的隐藏层 (self.hidden)

class NewModel(nn.Module):
    def __init__(self, pretrained_hidden_layer):
        super().__init__()
        # 直接使用传入的、已经训练好的隐藏层
        self.hidden_layer = pretrained_hidden_layer
        # 定义一个新的输出层，用于新任务 (例如，一个5分类任务)
        self.new_output_layer = nn.Linear(256, 5)

    def forward(self, x):
        # 通过复用的隐藏层提取特征
        features = F.relu(self.hidden_layer(x))
        # 将提取的特征送入新的输出层
        return self.new_output_layer(features)

# 创建新模型，并将 net 的隐藏层传入
new_net = NewModel(net.hidden)

# (可选) 冻结复用层的参数，使其在后续训练中不被更新
for param in new_net.hidden_layer.parameters():
    param.requires_grad = False

print("创建了一个新模型，并复用了原模型的隐藏层。")
print("新模型结构:\n", new_net)

# 我们可以检查新模型隐藏层的权重是否与旧模型一致
are_hidden_weights_equal = torch.equal(net.hidden.weight, new_net.hidden_layer.weight)
print("新旧模型的隐藏层权重是否一致:", are_hidden_weights_equal)
print("-" * 50)


# --- 练习 3: 同时保存架构和参数 ---
print("--- 练习 3: 同时保存架构和参数 ---")
# 这种方法虽然方便，但不推荐，因为它使文件依赖于具体的代码结构

# 保存整个模型对象
torch.save(net, 'full_model.pth')
print("已将完整的模型对象保存到 'full_model.pth'")

# 加载整个模型对象
# 注意：在加载前，定义MLP的class必须是可用的
loaded_full_model = torch.load('full_model.pth')
loaded_full_model.eval()

print("已从文件完整加载模型对象。")

# 验证
Y_full_loaded = loaded_full_model(X)
are_outputs_equal_full = torch.equal(Y, Y_full_loaded)
print("原始模型输出和完整加载后模型的输出是否完全一致:", are_outputs_equal_full)
print("-" * 50)