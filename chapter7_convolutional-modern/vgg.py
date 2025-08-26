import torch
from torch import nn
from d2l import torch as d2l
import platform
import time

# =============================================================================
# Part 1: 定义VGG块 (Block)
# -----------------------------------------------------------------------------
# VGG的核心思想是使用重复的块结构。每个块包含若干个卷积层和一个汇聚层。
# =============================================================================
print("--- 1. 定义 VGG 块 ---")
def vgg_block(num_convs, in_channels, out_channels):
    """
    定义一个VGG块。
    
    num_convs: 块中卷积层的数量。
    in_channels: 输入通道数。
    out_channels: 输出通道数。
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        # 下一个卷积层的输入通道数等于当前层的输出通道数
        in_channels = out_channels
    # 每个块的最后是一个最大汇聚层，用于将高宽减半
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# =============================================================================
# Part 2: 定义 VGG 整体架构
# -----------------------------------------------------------------------------
# 整个VGG网络就是将多个VGG块和最后的全连接层串联起来。
# =============================================================================
print("--- 2. 定义 VGG-11 架构 ---")

# VGG-11的架构配置: (块中的卷积层数, 输出通道数)
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1 # 初始输入通道数 (灰度图)
    
    # --- 构建卷积部分 ---
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    # --- 构建全连接部分 ---
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 输入维度计算: out_channels * (image_size / 2**num_blocks)**2
        # 对于 224x224 输入, 经过5个汇聚层后是 7x7
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

# 实例化一个VGG-11网络
net = vgg(conv_arch)


# --- 检查每一层的输出形状 ---
X = torch.randn(size=(1, 1, 224, 224))
print("逐层检查VGG-11输出形状:")
for blk in net:
    X = blk(X)
    print(f"{blk.__class__.__name__:<12} output shape:\t{X.shape}")
print("-" * 50)


# =============================================================================
# Part 3: 训练模型
# -----------------------------------------------------------------------------
# 我们将使用一个通道数较少的版本来在Fashion-MNIST上进行演示。
# =============================================================================
print("--- 3. 训练一个精简版的 VGG ---")

# --- 定义一个通道数较少的VGG变体 ---
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net_small = vgg(small_conv_arch)

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
    lr, num_epochs, batch_size = 0.05, 10, 128
    
    # --- 获取设备 ---
    device = get_device()
    
    # --- 加载数据并调整大小 ---
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    # --- 开始训练 ---
    train_ch6(net_small, train_iter, test_iter, num_epochs, lr, device)