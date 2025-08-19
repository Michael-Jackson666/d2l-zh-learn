import torch
from torch import nn
from d2l import torch as d2l

# =============================================================================
# 步骤 1: 定义模型 (简洁实现)
# 直接在 nn.Sequential 中添加 nn.Dropout 层。
# =============================================================================
dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    # 在第一个全连接层之后添加一个dropout层
                    nn.Dropout(dropout1),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    # 在第二个全连接层之后添加一个dropout层
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# =============================================================================
# 定义我们自己的辅助工具和训练循环 (自包含，不依赖d2l易变函数)
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
    net.eval()  # 关键: 将模型设置为评估模式，这将自动禁用Dropout
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_loop(net, train_iter, test_iter, loss, trainer, num_epochs):
    """一个包含动画的完整训练循环"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                          legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        net.train() # 关键: 将模型设置为训练模式，这将自动启用Dropout
        train_metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            with torch.no_grad():
                train_metric.add(l.sum(), accuracy(y_hat, y), y.numel())
        
        train_loss = train_metric[0] / train_metric[2]
        train_acc = train_metric[1] / train_metric[2]
        
        # 评估时会自动调用 net.eval() 禁用 dropout
        test_acc = evaluate_accuracy(net, test_iter)
        
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
    
    print(f'\nFinal Train Loss: {train_loss:.4f}')
    print(f'Final Train Acc:  {train_acc:.4f}')
    print(f'Final Test Acc:   {test_acc:.4f}')
    d2l.plt.show()

# =============================================================================
# 步骤 2: 主执行块
# =============================================================================
if __name__ == '__main__':
    # --- 初始化 ---
    net.apply(init_weights)

    # --- 数据、损失和优化器 ---
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # --- 开始训练 ---
    print("开始训练...")
    train_loop(net, train_iter, test_iter, loss, trainer, num_epochs)