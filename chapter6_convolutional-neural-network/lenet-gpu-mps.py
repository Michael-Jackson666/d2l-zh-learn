import torch
from torch import nn
from d2l import torch as d2l
import platform
import time

# =============================================================================
# Part 1: 智能设备检测 (适用于macOS)
# =============================================================================
def get_device():
    """
    智能检测并返回Apple Silicon (MPS) GPU设备，如果不可用则回退到CPU。
    """
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        print("检测到Apple Metal (MPS) 后端可用，将使用 MPS 设备。")
        return torch.device('mps')
    print("MPS 不可用，将使用 CPU。")
    return torch.device('cpu')

# =============================================================================
# Part 2: 定义LeNet模型
# =============================================================================
net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

# =============================================================================
# Part 3: 定义一个通用的、与设备无关的训练函数
# =============================================================================
def train(net, train_iter, test_iter, num_epochs, lr, device):
    """
    在指定设备上训练模型(CUDA或MPS兼容)。
    """
    # 1. 初始化权重 (Xavier)
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    
    # 2. 将模型移动到目标设备
    print('Training on', device)
    net.to(device)
    
    # 3. 定义优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    # 4. 训练主循环（去除动画，直接打印日志，避免mps崩溃）
    timer, num_batches = d2l.Timer(), len(train_iter)
    def evaluate_accuracy_gpu_fixed(net, data_iter, device):
        net.eval()
        metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                X, y = X.to(device), y.to(device)
                metric.add(d2l.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            # mps: loss.backward()后建议同步，避免异步导致崩溃
            if device.type == 'mps':
                torch.mps.synchronize()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
        # mps: epoch结束后建议清理缓存
        if device.type == 'mps':
            torch.mps.empty_cache()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu_fixed(net, test_iter, device)
        print(f'Epoch {epoch+1}: loss {train_l:.4f}, train acc {train_acc:.4f}, test acc {test_acc:.4f}')
    print(f'Final: loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

# =============================================================================
# Part 4: 主执行块
# =============================================================================
if __name__ == '__main__':
    # --- 超参数 ---
    lr, num_epochs, batch_size = 0.9, 10, 256
    
    # --- 获取设备 ---
    # 这一步会自动为你的Mac选择 'mps'
    device = get_device()
    
    # --- 加载数据 ---
    # 禁用多进程以避免在macOS上出现RuntimeError
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    # --- 开始训练 ---
    train(net, train_iter, test_iter, num_epochs, lr, device)