import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import platform
import time

# =============================================================================
# Part 1: 定义 Inception 块 (Block)
# -----------------------------------------------------------------------------
# Inception 块是GoogLeNet的核心。它通过并行使用不同尺寸的卷积核
# 和汇聚层，来捕捉不同尺度的特征，然后将它们拼接起来。
# =============================================================================
print("--- 1. 定义 Inception 块 ---")
class Inception(nn.Module):
    # c1-c4 是四条并行路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        # 路径1: 1x1 卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        
        # 路径2: 1x1 卷积 -> 3x3 卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        
        # 路径3: 1x1 卷积 -> 5x5 卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        
        # 路径4: 3x3 最大汇聚 -> 1x1 卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上拼接四条路径的输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# =============================================================================
# Part 2: 定义 GoogLeNet 整体架构
# -----------------------------------------------------------------------------
# GoogLeNet 由多个阶段性的块 (b1-b5) 串联而成，其中b3, b4, b5内部堆叠了Inception块。
# =============================================================================
print("--- 2. 定义 GoogLeNet 架构 ---")
# 阶段1：一个大的卷积层，快速降低分辨率
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 阶段2：两个卷积层，增加通道数
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 阶段3：两个Inception块
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 阶段4：五个Inception块
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 阶段5：两个Inception块 + 全局平均汇聚
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten())

# 将所有块串联起来，最后接一个线性层作为分类器
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# --- 检查每个块的输出形状 (使用96x96的输入) ---
X = torch.rand(size=(1, 1, 96, 96))
print("逐块检查GoogLeNet输出形状:")
for blk in net:
    X = blk(X)
    print(f"{blk.__class__.__name__:<12} output shape:\t{X.shape}")
print("-" * 50)


# =============================================================================
# Part 3: 训练模型
# =============================================================================
print("--- 3. 训练 GoogLeNet 模型 ---")

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
    lr, num_epochs, batch_size = 0.1, 10, 128
    
    # --- 获取设备 ---
    device = get_device()
    
    # --- 加载数据并调整大小 ---
    # 为了在Fashion-MNIST上快速训练，将图像缩小到 96x96
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

    # --- 开始训练 ---
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)