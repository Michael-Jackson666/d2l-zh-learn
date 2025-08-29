import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib中文支持和图像显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)

# --- d2l 辅助函数 (简化版，用于独立运行) ---
def use_svg_display():
    """在Jupyter等环境中用svg格式显示绘图，脚本中可忽略。"""
    pass

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小。"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点。"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim: axes.set_xlim(xlim)
    if ylim: axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()

class Animator:
    """简化的动画绘制类，用于训练过程可视化。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.X, self.Y = [], []

    def add(self, x, y):
        """添加数据点"""
        if not hasattr(y, "__len__"):
            y = [y]
        if not hasattr(x, "__len__"):
            x = [x]
        
        self.X.extend(x)
        self.Y.extend(y)
        
        # 简单打印进度，不实时绘图
        print(f"  {self.xlabel}: {x[0]}, {self.ylabel}: {y[0]:.6f}")

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(10, 6),
                  cmap='Reds', save_path=None):
    """显示矩阵热图，颜色条放在图片外侧"""
    use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    
    # 调整图形大小，为颜色条留出空间
    adjusted_figsize = (figsize[0] + 1.5, figsize[1])
    fig, axes = plt.subplots(num_rows, num_cols, figsize=adjusted_figsize,
                                 sharex=True, sharey=True, squeeze=False)
    
    # 如果只有一个子图，axes需要特殊处理
    if num_rows == 1 and num_cols == 1:
        axes = axes.reshape(1, 1)
    
    pcm = None  # 保存最后一个plot对象用于colorbar
    
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            # 确保矩阵是numpy格式
            if torch.is_tensor(matrix):
                matrix_np = matrix.detach().numpy()
            else:
                matrix_np = matrix
            
            pcm = ax.imshow(matrix_np, cmap=cmap, aspect='auto')
            if i == num_rows - 1:
                ax.set_xlabel(xlabel, fontsize=12)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            if titles and j < len(titles):
                ax.set_title(titles[j], fontsize=14)
    
    # 调整子图布局，为颜色条留出空间
    plt.subplots_adjust(right=0.85)
    
    # 添加颜色条，放在图片右侧
    if pcm is not None:
        cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(pcm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('注意力权重', rotation=270, labelpad=15, fontsize=12)
    
    # 保存图片（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图片已保存为: {save_path}")
    
    plt.show()
    return fig

# --- 课程代码开始 ---

# 1. 生成数据集
print("--- 1. 生成数据集 ---")
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)
print(f"生成了 {n_train} 个训练样本和 {n_test} 个测试样本。")

def plot_kernel_reg(y_hat, title=""):
    """绘制核回归结果对比图"""
    plt.figure(figsize=(10, 6))
    plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['真实值', '预测值'],
             xlim=[0, 5], ylim=[-1, 5])
    plt.plot(x_train.numpy(), y_train.numpy(), 'o', alpha=0.5, label='训练数据')
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 2. 平均汇聚 (Baseline)
print("\n--- 2. 平均汇聚 (效果很差的基线模型) ---")
y_hat_avg = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat_avg, "平均汇聚回归 (基线模型)")
print(f"平均汇聚预测值: 所有预测都是 {y_train.mean():.3f}")

print("\n" + "="*60)

# 3. 非参数注意力汇聚 (Nadaraya-Watson)
print("\n--- 3. 非参数注意力汇聚 (Nadaraya-Watson) ---")
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
y_hat_nw = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat_nw, "Nadaraya-Watson 核回归 (非参数)")
print("非参数模型预测完成。")

# 可视化非参数注意力权重
print("\n可视化非参数注意力权重模式...")
show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
              xlabel='训练输入 (已排序)',
              ylabel='测试输入 (已排序)',
              titles=['非参数注意力权重'],
              save_path='nadaraya_watson_attention.png')

print("\n" + "="*60)

# 4. 带参数注意力汇聚
print("\n--- 4. 带参数注意力汇聚 ---")

# 批量矩阵乘法示例
X_bmm = torch.ones((2, 1, 4))
Y_bmm = torch.ones((2, 4, 6))
print(f"批量矩阵乘法示例: X(2,1,4) bmm Y(2,4,6) -> shape: {torch.bmm(X_bmm, Y_bmm).shape}")

# 定义模型
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

# 准备训练数据 (leave-one-out)
X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

# 训练模型
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)

print("开始训练带参数的模型...")
losses = []
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    current_loss = float(l.sum())
    losses.append(current_loss)
    print(f'epoch {epoch + 1}, loss {current_loss:.6f}')

# 显示训练过程
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), losses, 'b-o', linewidth=2, markersize=8)
plt.xlabel('训练轮数', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.title('带参数注意力模型训练过程', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"训练完成！学习到的核参数 w = {net.w.item():.4f}")

# 预测和可视化
print("\n带参数模型预测完成。")
keys_test = x_train.repeat((n_test, 1))
values_test = y_train.repeat((n_test, 1))
y_hat_para = net(x_test, keys_test, values_test).unsqueeze(1).detach()
plot_kernel_reg(y_hat_para, "带参数的注意力回归 (可学习核)")

# 可视化带参数的注意力权重
print("\n可视化带参数注意力权重模式...")
show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
              xlabel='训练输入 (已排序)',
              ylabel='测试输入 (已排序)',
              titles=['带参数注意力权重'],
              save_path='parametric_attention.png')

print("\n" + "="*60)
print("=== 实验总结 ===")
print("生成的图片文件:")
print("1. nadaraya_watson_attention.png - 非参数注意力权重模式")
print("2. parametric_attention.png - 带参数注意力权重模式")
print()
print("三种方法对比:")
print("1. 平均汇聚: 预测值为常数，效果最差")
print("2. 非参数Nadaraya-Watson: 根据距离自动调整注意力")
print("3. 带参数模型: 通过学习优化注意力核参数")
print(f"   学习到的最优核参数: w = {net.w.item():.4f}")
print()
print("注意力权重热图说明:")
print("- 颜色越深表示注意力权重越大")
print("- 对角线模式表示局部注意力")
print("- 带参数模型学习到更集中的注意力模式")