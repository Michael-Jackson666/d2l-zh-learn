import torch
from torch import nn
from d2l import torch as d2l

# --- 1. 设备配置 (适配Mac的MPS, CUDA或CPU) ---
def get_device():
    """获取可用的设备 (MPS, CUDA 或 CPU)"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

device = get_device()
print(f'Using device: {device}')

# --- 2. 数据生成 ---
# 使用正弦函数和一些可加性噪声来生成序列数据
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# d2l.plt.show() # 取消注释以显示图像

# --- 3. 特征和标签准备 ---
# 基于嵌入维度 tau，我们将数据映射为数据对
# y_t = x_t 和 x_t = [x_{t-tau}, ..., x_{t-1}]
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

# --- 4. 数据加载器 ---
batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# --- 5. 模型定义 ---
# 一个简单的多层感知机 (MLP)
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    return net

# 平方损失
loss = nn.MSELoss()

# --- 6. 训练函数 ---
def train(net, train_iter, loss_fn, epochs, lr, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr)
    print("--- Starting Training ---")
    for epoch in range(epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            l = loss_fn(net(X), y)
            l.backward()
            optimizer.step()
        
        # 评估损失
        net.eval()
        with torch.no_grad():
            train_loss_sum = 0
            num_batches = 0
            for X, y in train_iter:
                X, y = X.to(device), y.to(device)
                train_loss_sum += loss_fn(net(X), y).item()
                num_batches += 1
            avg_loss = train_loss_sum / num_batches
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:f}')
    print("--- Finished Training ---")

# --- 7. 开始训练 ---
net = get_net()
train(net, train_iter, loss, 5, 0.01, device)

# --- 8. 预测与评估 ---
# 将模型和数据都移动到CPU上进行绘图
net.to('cpu')
features_cpu = features.to('cpu')
x_cpu = x.to('cpu')

# 8.1 单步预测
onestep_preds = net(features_cpu)
d2l.plot([time, time[tau:]],
         [x_cpu.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.show()

# 8.2 多步预测 (使用自己的预测结果作为下一步的输入)
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x_cpu[: n_train + tau]
for i in range(n_train + tau, T):
    # 将输入数据整形为 (1, tau) 的批次
    input_seq = multistep_preds[i - tau:i].reshape((1, -1))
    multistep_preds[i] = net(input_seq)

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x_cpu.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# 8.3 k步预测的可视化
max_steps = 64
features_k = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
for i in range(tau):
    features_k[:, i] = x_cpu[i: i + T - tau - max_steps + 1]

for i in range(tau, tau + max_steps):
    features_k[:, i] = net(features_k[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features_k[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.show()