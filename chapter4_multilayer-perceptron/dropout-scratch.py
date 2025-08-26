import torch
from torch import nn
from d2l import torch as d2l

# =============================================================================
# Part 1: 从零开始实现 Dropout
# =============================================================================

def dropout_layer(X, dropout):
    """
    手动实现 Dropout 层。
    dropout: 丢弃神经元的概率。
    """
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    # 生成一个随机的掩码(mask)，值在[0, 1]之间
    # 将大于dropout概率的位置设为1，否则为0
    mask = (torch.rand(X.shape) > dropout).float()
    # 应用掩码，并除以(1.0 - dropout)来拉伸(rescale)数值，
    # 从而保持激活值的期望不变 (Inverted Dropout)
    return mask * X / (1.0 - dropout)

def train_scratch():
    """使用从零实现的Dropout层进行训练"""
    print("\n--- 从零实现 Dropout ---")
    
    # --- 模型定义 ---
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5

    class Net(nn.Module):
        def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
            super(Net, self).__init__()
            self.training = is_training
            self.lin1 = nn.Linear(num_inputs, num_hiddens1)
            self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
            self.lin3 = nn.Linear(num_hiddens2, num_outputs)
            self.relu = nn.ReLU()

        def forward(self, X):
            H1 = self.relu(self.lin1(X.reshape((-1, num_inputs))))
            # 只在训练模式下应用Dropout
            if self.training:
                H1 = dropout_layer(H1, dropout1)
            H2 = self.relu(self.lin2(H1))
            if self.training:
                H2 = dropout_layer(H2, dropout2)
            out = self.lin3(H2)
            return out

    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    
    # --- 训练 ---
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, num_workers=0)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    
    # 使用 d2l 库中预置的通用训练函数
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()

# =============================================================================
# Part 2: 简洁实现 Dropout
# =============================================================================

def train_concise():
    """使用PyTorch内置的Dropout层进行训练"""
    print("\n--- 简洁实现 Dropout ---")

    dropout1, dropout2 = 0.2, 0.5
    
    # --- 模型定义 ---
    # 直接在 nn.Sequential 中添加 nn.Dropout 层
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout1), # 在第一个隐藏层后添加Dropout
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout2), # 在第二个隐藏层后添加Dropout
                        nn.Linear(256, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    # --- 训练 ---
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, num_workers=0)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()

# =============================================================================
# 主执行块
# =============================================================================
if __name__ == '__main__':
    train_scratch()
    train_concise()