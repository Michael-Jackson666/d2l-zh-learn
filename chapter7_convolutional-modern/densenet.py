import torch
from torch import nn
from d2l import torch as d2l
import platform
import time

# =============================================================================
# Part 1: 定义 DenseNet 的核心组件
# -----------------------------------------------------------------------------
# DenseNet 主要由 稠密块(DenseBlock) 和 过渡层(transition_block) 组成。
# =============================================================================
print("--- 1. 定义 DenseNet 核心组件 ---")

def conv_block(input_channels, num_channels):
    """
    DenseNet中使用的基础卷积块: BN -> ReLU -> Conv
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )

class DenseBlock(nn.Module):
    """稠密块"""
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        # 这里的 num_channels 是增长率 (growth_rate)
        for i in range(num_convs):
            # 每个卷积块的输入通道数 = 初始输入通道数 + 之前所有卷积块的输出通道数之和
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 在通道维度上将输入X和输出Y进行拼接
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):
    """
    过渡层，用于连接两个稠密块。
    它通过 1x1 卷积减少通道数，并通过平均汇聚层减半高和宽。
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

# =============================================================================
# Part 2: 定义 DenseNet 整体架构
# =============================================================================
print("--- 2. 定义 DenseNet 架构 ---")
# 初始卷积块 (stem)
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# --- 构建稠密块和过渡层 ---
# growth_rate (增长率) 是每个卷积块产生的通道数
num_channels, growth_rate = 64, 32
# 每个稠密块中卷积层的数量
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 更新当前通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加过渡层，将通道数减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# --- 组装成完整的DenseNet ---
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))


# =============================================================================
# Part 3: 训练模型
# =============================================================================
print("--- 3. 训练 DenseNet 模型 ---")

# --- 训练函数 (与上一章通用) ---
def get_device(i=0):
    if torch.cuda.is_available() and torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device('mps')
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
# Part 4: 主执行块
# =============================================================================
if __name__ == '__main__':
    # --- 超参数 ---
    lr, num_epochs, batch_size = 0.1, 10, 256
    
    # --- 获取设备 ---
    device = get_device()
    
    # --- 加载数据并调整大小 ---
    # 为了在Fashion-MNIST上快速训练，将图像缩小到 96x96
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

    # --- 开始训练 ---
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)