import torch
from d2l import torch as d2l

# =============================================================================
# 步骤 1: 加载数据集
# 我们使用Fashion-MNIST数据集，这是一个比MNIST稍复杂的图像分类任务。
# DataLoader会自动帮我们完成数据的下载、读取、打乱和批量化。
# =============================================================================
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# =============================================================================
# 步骤 2: 初始化模型参数
# Fashion-MNIST的每张图片是 28x28 像素，我们将其展平为 784 维的向量。
# 数据集有10个类别，所以模型的输出维度是 10。
# =============================================================================
num_inputs = 784   # 输入维度 (28 * 28)
num_outputs = 10   # 输出维度 (10个类别)

# 权重 W: 形状为 (输入维度, 输出维度)
# 偏置 b: 形状为 (输出维度)
# requires_grad=True: 告诉PyTorch我们需要计算这些参数的梯度
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# =============================================================================
# 步骤 3: 定义核心函数 (模型、损失、评估)
# 我们将手动实现Softmax函数、交叉熵损失函数和精度计算函数。
# =============================================================================

def softmax(X):
    """
    手动实现Softmax函数。
    注意：这个实现没有做数值稳定性处理，在logits很大时可能会上溢。
    """
    # 1. 对每个元素求指数
    X_exp = torch.exp(X)
    # 2. 对每一行（每个样本）求和，得到规范化常数
    partition = X_exp.sum(1, keepdim=True)
    # 3. 将每一行的元素除以其规范化常数
    return X_exp / partition  # 利用了广播机制

def net(X):
    """
    定义Softmax回归模型。
    """
    # 1. 将输入的图片展平为向量
    # 2. 执行线性变换: XW + b
    # 3. 应用Softmax函数得到概率分布
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    """
    手动实现交叉熵损失函数。
    注意：这个实现没有做数值稳定性处理，当y_hat中某个概率为0时会出错。
    """
    # y_hat是模型的预测概率矩阵，y是真实标签向量
    # 通过高级索引，我们能一次性从y_hat中挑选出所有样本的正确类别的预测概率
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """计算预测正确的样本数量。"""
    # y_hat.shape: (batch_size, num_outputs)
    # y_hat.argmax(axis=1) 会返回每行最大值的索引，即预测的类别
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # 将预测类别与真实标签进行比较，得到一个布尔张量
    cmp = y_hat.type(y.dtype) == y
    # 求和得到正确的数量
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """在指定数据集上评估模型的精度。"""
    # 这是一个辅助函数，用于在测试集上计算模型的整体精度
    if isinstance(net, torch.nn.Module):
        net.eval()  # 如果是nn.Module，设置为评估模式
    
    metric = d2l.Accumulator(2)  # (正确预测数, 总样本数)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# =============================================================================
# 步骤 4: 定义训练过程
# 我们将定义单轮训练和完整训练过程的函数。
# =============================================================================

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期。"""
    if isinstance(net, torch.nn.Module):
        net.train() # 如果是nn.Module，设置为训练模式
    
    # (总损失, 总正确数, 总样本数)
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        # 前向传播，计算损失
        y_hat = net(X)
        l = loss(y_hat, y)
        
        # 反向传播，计算梯度
        l.sum().backward()
        
        # 更新参数
        updater(batch_size)
        
        # 累加指标
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    # 返回平均训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """完整的训练函数，包含可视化。"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                          legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # 训练一轮
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 在测试集上评估
        test_acc = evaluate_accuracy(net, test_iter)
        # 更新动画
        animator.add(epoch + 1, train_metrics + (test_acc,))
    
    train_loss, train_acc = train_metrics
    print(f'\nFinal Train Loss: {train_loss:.4f}')
    print(f'Final Train Acc:  {train_acc:.4f}')
    print(f'Final Test Acc:   {test_acc:.4f}')

# =============================================================================
# 步骤 5: 开始训练
# =============================================================================

# --- 设置超参数 ---
lr = 0.1
num_epochs = 10

# --- 定义优化器 ---
# 这里我们使用 d2l 包中预先定义好的 sgd 函数
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

# --- 调用训练函数 ---
print("开始训练...")
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


# =============================================================================
# 步骤 6: 进行预测 (使用新版 d2l 函数)
# =============================================================================
def predict_ch3_new(net, test_iter, n=6):
    """一个兼容新版d2l的预测函数"""
    for X, y in test_iter:
        break # 只取第一个批次
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

print("\n进行预测...")
predict_ch3_new(net, test_iter)