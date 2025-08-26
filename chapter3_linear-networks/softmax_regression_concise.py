import torch
from torch import nn
from d2l import torch as d2l

# =============================================================================
# 步骤 1: 加载数据集
# =============================================================================
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# =============================================================================
# 步骤 2: 定义并初始化模型
# =============================================================================
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# =============================================================================
# 步骤 3: 定义损失函数和优化器
# =============================================================================
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# =============================================================================
# (新增!) 定义我们自己的辅助工具，不再依赖d2l库中易变动的函数
# =============================================================================

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

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
    net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # (正确预测数, 总样本数)
    with torch.no_grad():
        for X, y in data_iter:
            # 调用我们自己定义的 accuracy 函数
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# =============================================================================
# 步骤 4: 训练 (使用我们自己的辅助函数)
# =============================================================================
num_epochs = 10
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                      legend=['train loss', 'train acc', 'test acc'])

print("开始训练...")
for epoch in range(num_epochs):
    net.train() # 将模型设置为训练模式
    train_metric = Accumulator(3) # (总损失, 总正确数, 总样本数)
    
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        with torch.no_grad():
            # 调用我们自己定义的 accuracy 函数
            train_metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
    
    train_loss = train_metric[0] / train_metric[2]
    train_acc = train_metric[1] / train_metric[2]

    # 调用我们自己定义的评估函数
    test_acc = evaluate_accuracy(net, test_iter)

    animator.add(epoch + 1, (train_loss, train_acc, test_acc))

# --- 打印最终结果 ---
final_train_loss, final_train_acc = train_loss, train_acc
print(f'\nFinal Train Loss: {final_train_loss:.4f}')
print(f'Final Train Acc:  {final_train_acc:.4f}')
print(f'Final Test Acc:   {test_acc:.4f}')

# 在训练结束后，显示最终的训练曲线图
d2l.plt.show()

# =============================================================================
# 步骤 5: 预测
# =============================================================================
def predict_ch3_new(net, test_iter, n=6):
    """预测函数"""
    for X, y in test_iter:
        break # 只取第一个批次
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    
    # 在生成预测图像后，显示它们
    d2l.plt.show()

print("\n进行预测...")
predict_ch3_new(net, test_iter)