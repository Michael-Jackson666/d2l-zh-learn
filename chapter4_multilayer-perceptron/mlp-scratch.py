import torch
from torch import nn
from d2l import torch as d2l

# =============================================================================
# 步骤 1: 加载数据集
# =============================================================================
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# =============================================================================
# 步骤 2: 初始化模型参数
# 这是本节的核心：我们手动创建并管理每一层的权重和偏置。
# =============================================================================
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# --- 隐藏层参数 ---
# W1: (输入层 -> 隐藏层) 的权重
# b1: (输入层 -> 隐藏层) 的偏置
# 使用 nn.Parameter 将这些张量包装起来，这样PyTorch的自动求导机制
# 就能追踪它们，并且优化器也能找到它们。
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

# --- 输出层参数 ---
# W2: (隐藏层 -> 输出层) 的权重
# b2: (隐藏层 -> 输出层) 的偏置
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# 将所有参数放入一个列表中，方便后续传入优化器
params = [W1, b1, W2, b2]

# =============================================================================
# 步骤 3: 定义激活函数和模型
# =============================================================================

def relu(X):
    """手动实现ReLU激活函数"""
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    """定义单隐藏层的MLP模型"""
    # 将输入展平为 (batch_size, 784)
    X = X.reshape((-1, num_inputs))
    # 计算隐藏层的输出: H = ReLU(X @ W1 + b1)
    # "@" 是PyTorch中矩阵乘法的简洁写法
    H = relu(X @ W1 + b1)
    # 计算输出层的输出 (logits)
    return (H @ W2 + b2)

# =============================================================================
# 步骤 4: 定义损失函数和优化器
# =============================================================================

# 使用PyTorch内置的交叉熵损失函数，它包含了Softmax并且数值稳定
loss = nn.CrossEntropyLoss()

# 使用PyTorch内置的SGD优化器，并将我们手动创建的参数列表传入
lr = 0.1
updater = torch.optim.SGD(params, lr=lr)


# =============================================================================
# (新增!) 定义我们自己的辅助工具，不再依赖d2l库中易变动的函数
# =============================================================================

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    # 这里的net是一个函数，而不是nn.Module，所以不需要net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# =============================================================================
# 步骤 5: 训练
# =============================================================================
num_epochs = 10
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                      legend=['train loss', 'train acc', 'test acc'])

print("开始训练...")
for epoch in range(num_epochs):
    train_metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()
        with torch.no_grad():
            train_metric.add(l * len(y), accuracy(y_hat, y), y.numel())
    
    train_loss = train_metric[0] / train_metric[2]
    train_acc = train_metric[1] / train_metric[2]
    test_acc = evaluate_accuracy(net, test_iter)
    animator.add(epoch + 1, (train_loss, train_acc, test_acc))

print(f'\nFinal Train Loss: {train_loss:.4f}')
print(f'Final Train Acc:  {train_acc:.4f}')
print(f'Final Test Acc:   {test_acc:.4f}')

d2l.plt.show()

# =============================================================================
# 步骤 6: 预测
# =============================================================================
def predict_ch3_new(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()

print("\n进行预测...")
predict_ch3_new(net, test_iter)