import torch
from torch import nn
from d2l import torch as d2l
import platform
import time

# =============================================================================
# Part 1: 定义 AlexNet 模型
# -----------------------------------------------------------------------------
# AlexNet 的核心是更深、更宽的卷积网络，并首次成功应用了ReLU和Dropout。
# =============================================================================
print("--- 1. 定义 AlexNet 模型 ---")
net = nn.Sequential(
    # 卷积块 1
    # 输入: (B, 1, 224, 224)
    # 使用11x11的大卷积核捕捉大尺寸特征
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    
    # 卷积块 2
    # 减小卷积核，增加通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    
    # 卷积块 3 (连续卷积)
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    
    nn.Flatten(),
    
    # 全连接块
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5), # 使用Dropout正则化
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10) # 输出层，对应Fashion-MNIST的10个类别
)

# --- 检查每一层的输出形状 ---
# 创建一个符合输入尺寸的随机张量 (224x224)
X = torch.randn(1, 1, 224, 224)
print("逐层检查AlexNet输出形状:")
temp_X = X
for layer in net:
    temp_X = layer(temp_X)
    print(f"{layer.__class__.__name__:<12} output shape:\t{temp_X.shape}")
print("-" * 50)


# =============================================================================
# Part 2: 定义训练和评估函数 (自包含)
# =============================================================================
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

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if device is None and isinstance(net, nn.Module):
        device = next(net.parameters()).device
    net.eval()
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

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
        
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    d2l.plt.show()

# =============================================================================
# Part 3: 主执行块
# =============================================================================
if __name__ == '__main__':
    # --- 超参数 ---
    # AlexNet更复杂，通常需要更小的学习率和更长的训练时间
    lr, num_epochs, batch_size = 0.01, 10, 128
    
    # --- 获取设备 ---
    device = get_device()
    
    # --- 加载数据并调整大小 ---
    # AlexNet的原始输入是224x224，所以我们将Fashion-MNIST图像放大
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    # --- 开始训练 ---
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)