import torch
from torch import nn
from d2l import torch as d2l
import platform
import time

# =============================================================================
# Part 1: 定义NiN块 (Block)
# -----------------------------------------------------------------------------
# NiN的核心思想是用一个迷你的多层感知机 (MLP) 来替代传统的线性卷积核。
# 这个“逐像素MLP”是通过一个常规卷积层后接多个1x1卷积层来实现的。
# =============================================================================
print("--- 1. 定义 NiN 块 ---")
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """
    定义一个NiN块。
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        # 1x1卷积层，用于增加非线性，可以看作在通道维度上的全连接层
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        # 再一个1x1卷积层，进一步增强非线性表达能力
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )

# =============================================================================
# Part 2: 定义 NiN 整体架构
# -----------------------------------------------------------------------------
# NiN的整体架构类似于AlexNet，但用NiN块替换了卷积层，并用全局平均汇聚
# 替换了最后的全连接层。
# =============================================================================
print("--- 2. 定义 NiN 架构 ---")
net = nn.Sequential(
    # 块1
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    # 块2
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    # 块3
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    
    nn.Dropout(0.5),
    
    # 块4 (输出块)
    # 输出通道数等于类别数
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    
    # 全局平均汇聚层
    # 将每个通道的空间维度(H, W)平均为一个单一的值
    # 输入: (B, 10, 5, 5) -> 输出: (B, 10, 1, 1)
    nn.AdaptiveAvgPool2d((1, 1)),
    
    # 将四维张量展平为二维张量 (B, 10) 以匹配损失函数的输入
    nn.Flatten()
)

# --- 检查每一层的输出形状 ---
X = torch.rand(size=(1, 1, 224, 224))
print("逐层检查NiN输出形状:")
for layer in net:
    X = layer(X)
    print(f"{layer.__class__.__name__:<18} output shape:\t{X.shape}")
print("-" * 50)

# =============================================================================
# Part 3: 训练模型
# =============================================================================
print("--- 3. 训练 NiN 模型 ---")

# --- 训练函数 (与上一章通用) ---
def get_device(i=0):
    """优先使用 CUDA 设备，无则回退到 mps 或 cpu"""
    if torch.cuda.is_available() and torch.cuda.device_count() >= i + 1:
        print("检测到 CUDA GPU，将使用 CUDA 设备。")
        return torch.device(f'cuda:{i}')
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        print("检测到 Apple Metal (MPS) 后端可用，将使用 MPS 设备。")
        return torch.device('mps')
    print("未检测到 CUDA/MPS，使用 CPU。")
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
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224, num_workers=0)

    # --- 开始训练 ---
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)