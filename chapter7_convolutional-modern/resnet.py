import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import platform
import time

# =============================================================================
# Part 1: 定义残差块 (Residual Block)
# -----------------------------------------------------------------------------
# ResNet的核心思想。它包含一个“快捷连接”(shortcut)，允许梯度直接流过，
# 从而缓解梯度消失问题，也使得学习恒等映射变得容易。
# =============================================================================
print("--- 1. 定义残差块 ---")
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 主路径包含两个卷积层
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 快捷连接路径
        if use_1x1conv:
            # 当输入输出通道数或尺寸不匹配时，使用1x1卷积来调整X的形状
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # 主路径计算
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # 快捷连接路径计算
        if self.conv3:
            X = self.conv3(X)
        # 将主路径输出和快捷连接路径输出相加
        Y += X
        # 最后通过ReLU激活
        return F.relu(Y)

# =============================================================================
# Part 2: 定义 ResNet-18 整体架构
# -----------------------------------------------------------------------------
# 整个ResNet网络由一个初始卷积块和四个残差模块串联而成。
# =============================================================================
print("--- 2. 定义 ResNet-18 架构 ---")
# 初始卷积块 (stem)
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 用于创建残差模块的辅助函数
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        # 第一个残差块需要调整通道数和尺寸
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else: # 后续的残差块保持尺寸和通道数不变
            blk.append(Residual(num_channels, num_channels))
    return blk

# 构建四个残差模块
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)) # 模块2
b3 = nn.Sequential(*resnet_block(64, 128, 2))                  # 模块3
b4 = nn.Sequential(*resnet_block(128, 256, 2))                 # 模块4
b5 = nn.Sequential(*resnet_block(256, 512, 2))                  # 模块5

# 将所有部分串联起来，最后接全局平均汇聚和分类器
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))

# --- 检查每个块的输出形状 ---
X = torch.rand(size=(1, 1, 224, 224))
print("逐块检查ResNet-18输出形状:")
for blk in net:
    X = blk(X)
    print(f"{blk.__class__.__name__:<18} output shape:\t{X.shape}")
print("-" * 50)

# =============================================================================
# Part 3: 训练模型
# =============================================================================
print("--- 3. 训练 ResNet-18 模型 ---")

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
    lr, num_epochs, batch_size = 0.05, 10, 256
    device = get_device()
    # ResNet通常用于较大尺寸的图像
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)