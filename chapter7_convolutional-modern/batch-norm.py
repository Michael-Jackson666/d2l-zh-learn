import torch
from torch import nn
from d2l import torch as d2l
import platform
import time

# =============================================================================
# Part 1: 从零开始实现批量规范化
# =============================================================================
print("--- 1. 从零开始实现批量规范化 ---")

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    手动实现批量规范化。
    X: 输入张量 (2D for FC, 4D for Conv)
    gamma, beta: 可学习的拉伸和偏移参数
    moving_mean, moving_var: 用于预测的全局均值和方差
    eps: 防止除以零的小常数
    momentum: 更新全局统计量的动量
    """
    # 判断当前是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 预测模式：使用全局统计量
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 全连接层：在特征维度(dim=1)上计算均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 卷积层：在批量、高、宽维度上计算，保留通道维度
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        
        # 训练模式：使用当前小批量的统计量进行规范化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新全局移动平均统计量
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
        
    # 应用可学习的拉伸和偏移
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        shape = (1, num_features) if num_dims == 2 else (1, num_features, 1, 1)
        # 定义可学习的gamma和beta
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 定义不参与训练的全局统计量 (moving average)
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 确保全局统计量和输入在同一个设备上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

# --- 将自定义BatchNorm应用于LeNet ---
net_scratch = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))


# =============================================================================
# Part 3: 训练与评估
# =============================================================================
# --- 通用函数 ---
def get_device():
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device('mps')
    print("MPS not available, falling back to CPU.")
    return torch.device('cpu')

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('Training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    d2l.plt.show()


# =============================================================================
# 主执行块
# =============================================================================
if __name__ == '__main__':
    device = get_device()
    # BN允许使用更大的学习率
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    
    # 训练从零实现的版本
    print("\n--- 训练从零实现的BatchNorm ---")
    train_ch6(net_scratch, train_iter, test_iter, num_epochs, lr, device)
    
    # 查看学到的gamma和beta
    print("\n从零实现的BN层学到的gamma和beta:")
    print(net_scratch[1].gamma.reshape((-1,)))
    print(net_scratch[1].beta.reshape((-1,)))