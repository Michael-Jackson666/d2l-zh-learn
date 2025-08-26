import torch
from torch import nn
from d2l import torch as d2l
import platform

# =============================================================================
# Part 1: 定义LeNet模型
# -----------------------------------------------------------------------------
# LeNet 是一个经典的CNN架构，由卷积块和全连接块组成。
# =============================================================================
print("--- 1. 定义LeNet模型 ---")
# PyTorch的 nn.Sequential 使得构建LeNet非常直观
net = nn.Sequential(
    # 卷积块 1
    # 输入: (B, 1, 28, 28)
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), # -> (B, 6, 28, 28)
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), # -> (B, 6, 14, 14)
    
    # 卷积块 2
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), # -> (B, 16, 10, 10)
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), # -> (B, 16, 5, 5)
    
    # 从卷积块到全连接块的过渡
    nn.Flatten(), # -> (B, 16 * 5 * 5) = (B, 400)
    
    # 全连接块
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

# --- 检查每一层的输出形状 ---
# 创建一个符合输入尺寸的随机张量
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# 逐层打印输出形状，以验证架构是否正确
print("逐层检查模型输出形状:")
temp_X = X
for layer in net:
    temp_X = layer(temp_X)
    print(f"{layer.__class__.__name__:<12} output shape: \t{temp_X.shape}")
print("-" * 50)


# =============================================================================
# Part 2: 定义训练和评估函数 (自包含)
# -----------------------------------------------------------------------------
# 我们使用一个通用的训练函数，它能智能选择CPU或GPU。
# =============================================================================
def get_device(i=0):
    """智能检测并返回可用的最佳设备"""
    if torch.cuda.is_available() and torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device('mps')
    return torch.device('cpu')

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """在指定设备上计算模型精度"""
    if device is None and isinstance(net, nn.Module):
        device = next(net.parameters()).device # 如果没指定设备，就用模型所在的设备
    net.eval()
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """通用训练函数(第6章版本)"""
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
    lr, num_epochs, batch_size = 0.9, 10, 256
    
    # --- 加载数据 ---
    # 禁用多进程以保证在macOS上运行稳定
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    # --- 开始训练 ---
    train_ch6(net, train_iter, test_iter, num_epochs, lr, get_device())