import torch
from torch import nn
from d2l import torch as d2l

# =============================================================================
# 步骤 1: 定义模型和初始化策略
# =============================================================================
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# =============================================================================
# 定义完整的训练和评估循环
# =============================================================================
def train_and_evaluate(net, train_iter, test_iter, loss, trainer, num_epochs):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                          legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        net.train()
        train_loss_total, train_acc_total, num_samples = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            with torch.no_grad():
                train_acc_total += (y_hat.argmax(dim=1) == y).sum().item()
                train_loss_total += l * y.shape[0]
                num_samples += y.shape[0]

        train_loss = train_loss_total / num_samples
        train_acc = train_acc_total / num_samples

        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_iter:
                y_hat = net(X)
                correct += (y_hat.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
    
    print(f'\nFinal Train Loss: {train_loss:.4f}')
    print(f'Final Train Acc:  {train_acc:.4f}')
    print(f'Final Test Acc:   {test_acc:.4f}')
    d2l.plt.show()

def predict_and_show(net, test_iter, n=6):
    net.eval()
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()

# =============================================================================
# 步骤 2: 主执行块
# =============================================================================
if __name__ == '__main__':
    # --- 初始化 ---
    net.apply(init_weights)

    # --- 数据、损失和优化器 ---
    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    
    # -------------------------------------------------------------------------
    # 加载Fashion-MNIST数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # -------------------------------------------------------------------------

    # --- 开始训练和预测 ---
    print("开始训练...")
    train_and_evaluate(net, train_iter, test_iter, loss, trainer, num_epochs)
    
    print("\n进行预测...")
    predict_and_show(net, test_iter)