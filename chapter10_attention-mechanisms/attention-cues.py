import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib中文支持和图像显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 8)

# --- d2l aite
# 为了代码能够独立运行，我们在这里定义一个简化版的 d2l 辅助函数
# 李沐老师课程中的 d2l 包包含了许多这样的教学辅助函数

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图。"""
    try:
        # 这个设置在标准的Python脚本中不起作用，但在Jupyter环境中可以提高图像质量
        from IPython import display
        display.set_matplotlib_formats('svg')
        print("Using SVG display for plots.")
    except ImportError:
        print("IPython display not found. Will use default plot display.")

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小。"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def save_figure(filename):
    """保存图片到文件"""
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图片已保存为: {filename}")

# --- 课程代码开始 ---

#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(8, 6),
                  cmap='Reds', save_path=None):
    """
    显示矩阵热图。
    matrices 的形状是 (要显示的行数, 要显示的列数, 查询的数目, 键的数目)。
    """
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
            
            # 创建热图
            pcm = ax.imshow(matrix_np, cmap=cmap, aspect='auto')
            
            # 设置标签
            if i == num_rows - 1:
                ax.set_xlabel(xlabel, fontsize=12)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            if titles and j < len(titles):
                ax.set_title(titles[j], fontsize=14)
            
            # 添加数值标注（对于小矩阵）
            if matrix_np.shape[0] <= 10 and matrix_np.shape[1] <= 10:
                for x in range(matrix_np.shape[1]):
                    for y in range(matrix_np.shape[0]):
                        text = ax.text(x, y, f'{matrix_np[y, x]:.2f}',
                                     ha="center", va="center", 
                                     color="white" if matrix_np[y, x] > 0.5 else "black",
                                     fontsize=8)
    
    # 调整子图布局，为颜色条留出空间
    plt.subplots_adjust(right=0.85)
    
    # 添加颜色条，放在图片右侧
    if pcm is not None:
        cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(pcm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('注意力权重', rotation=270, labelpad=15, fontsize=12)
    
    # 保存图片（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图片已保存为: {save_path}")
    
    # 显示图像
    plt.show()
    
    return fig

# --- 主程序 ---

# 示例1：可视化一个预设的注意力权重矩阵
print("=== 示例1: 可视化预设的注意力权重（单位矩阵） ===")
# torch.eye(10) 创建一个10x10的单位矩阵（对角线为1，其余为0）
# reshape 成 (1, 1, 10, 10) 是为了匹配 show_heatmaps 函数的输入格式
attention_weights_preset = torch.eye(10).reshape((1, 1, 10, 10))
print("单位矩阵形状:", attention_weights_preset.shape)
print("这个矩阵表示每个查询只注意自己对应的键")

# 显示热图并保存
show_heatmaps(attention_weights_preset, 
              xlabel='键 (Keys)', 
              ylabel='查询 (Queries)',
              titles=['单位矩阵注意力权重'],
              save_path='attention_identity_matrix.png')

print("\n" + "="*50)


# --- 练习题2的代码实现 ---
print("=== 示例2: 可视化随机生成并经过softmax的注意力权重 ===")
# 1. 随机生成一个 10x10 的矩阵
random_matrix = torch.randn(10, 10)
print("原始随机矩阵形状:", random_matrix.shape)
print("原始随机矩阵第一行:", random_matrix[0])

# 2. 使用 softmax 运算确保每行都是有效的概率分布
# dim=1 表示对每一行进行softmax操作
attention_weights_random = torch.softmax(random_matrix, dim=1)

# 3. 检查一下第一行的和是否为1，以验证softmax的有效性
print(f"\nSoftmax后第一行权重: {attention_weights_random[0].numpy()}")
print(f"第一行权重之和: {torch.sum(attention_weights_random[0]):.6f}")
print("所有行权重之和:", torch.sum(attention_weights_random, dim=1))

# 4. 可视化输出的注意力权重
# 同样，需要 reshape 以匹配函数输入
print("\n正在生成随机注意力权重热图...")
show_heatmaps(attention_weights_random.reshape((1, 1, 10, 10)),
              xlabel='键 (Keys)', 
              ylabel='查询 (Queries)',
              titles=['随机注意力权重 (Softmax)'],
              save_path='attention_random_weights.png')

print("\n" + "="*50)

# --- 额外示例：不同的注意力模式 ---
print("=== 示例3: 不同类型的注意力模式 ===")

# 1. 均匀注意力（所有位置权重相等）
uniform_attention = torch.ones(10, 10) / 10
print("均匀注意力 - 每个查询对所有键给予相等注意力")

# 2. 局部注意力（只注意附近的位置）
local_attention = torch.zeros(10, 10)
for i in range(10):
    for j in range(max(0, i-2), min(10, i+3)):  # 注意前后2个位置
        local_attention[i, j] = 1.0
# 归一化使每行和为1
local_attention = local_attention / local_attention.sum(dim=1, keepdim=True)
print("局部注意力 - 每个查询只注意附近的键")

# 3. 递减注意力（距离越远注意力越小）
decay_attention = torch.zeros(10, 10)
for i in range(10):
    for j in range(10):
        decay_attention[i, j] = torch.exp(-torch.abs(torch.tensor(i - j, dtype=torch.float)) / 2)
# 归一化
decay_attention = decay_attention / decay_attention.sum(dim=1, keepdim=True)
print("递减注意力 - 距离越远注意力越小")

# 将三种模式组合成一个2x2的网格
attention_patterns = torch.stack([
    torch.stack([uniform_attention, local_attention]),
    torch.stack([decay_attention, attention_weights_random])
])

print(f"\n组合注意力模式形状: {attention_patterns.shape}")

# 显示四种不同的注意力模式
titles = ['均匀注意力', '局部注意力', '递减注意力', '随机注意力']
show_heatmaps(attention_patterns,
              xlabel='键 (Keys)', 
              ylabel='查询 (Queries)',
              titles=titles,
              figsize=(12, 10),
              save_path='attention_different_patterns.png')

print("\n" + "="*50)
print("=== 运行完成！ ===")
print("生成的图片文件:")
print("1. attention_identity_matrix.png - 单位矩阵注意力")
print("2. attention_random_weights.png - 随机注意力权重")  
print("3. attention_different_patterns.png - 四种不同注意力模式对比")
print("\n这些图片展示了不同类型的注意力机制模式，帮助理解注意力权重的可视化")