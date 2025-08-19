import torch
from torch import nn
from d2l import torch as d2l

# =============================================================================
# Part 1: 从零开始实现权重衰减
# =============================================================================

def train_scratch(lambd):
    """从零开始实现权重衰减的训练函数"""
    print(f"\n--- 从零实现: lambda = {lambd} ---")
    
    # --- 数据生成 ---
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size) 
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)

    # --- 初始化参数 ---
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    params = [w, b]
    
    # --- 定义L2惩罚项 ---
    def l2_penalty(w):
        return torch.sum(w.pow(2)) / 2

    # --- 训练 ---
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 损失函数中加入了L2惩罚项
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd(params, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            # 评估时不需要惩罚项
            train_loss = d2l.evaluate_loss(net, train_iter, loss)
            test_loss = d2l.evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))
            
    print('w的L2范数是:', torch.norm(w).item())
    d2l.plt.show()

# =============================================================================
# Part 2: 简洁实现权重衰减
# =============================================================================

def train_concise(wd):
    """使用高级API简洁实现权重衰减的训练函数"""
    print(f"\n--- 简洁实现: weight_decay = {wd} ---")

    # --- 数据生成 ---
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)

    # --- 模型和初始化 ---
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
        
    # --- 损失和优化器 ---
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    
    # 关键：在优化器中直接设置 weight_decay 参数
    # PyTorch会自动将这个惩罚应用到指定的参数上
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd}, # 只对权重进行衰减
        {"params": net[0].bias} # 不对偏置进行衰减
    ], lr=lr)

    # --- 训练 ---
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            train_loss = d2l.evaluate_loss(net, train_iter, loss)
            test_loss = d2l.evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))
            
    print('w的L2范数：', net[0].weight.norm().item())
    d2l.plt.show()

# =============================================================================
# 主执行块
# =============================================================================
if __name__ == '__main__':
    # 运行从零实现版本
    train_scratch(lambd=0)   # 无权重衰减，观察过拟合
    train_scratch(lambd=3)    # 有权重衰减，观察效果

    # 运行简洁实现版本
    train_concise(wd=0)      # 无权重衰减
    train_concise(wd=3)       # 有权重衰减