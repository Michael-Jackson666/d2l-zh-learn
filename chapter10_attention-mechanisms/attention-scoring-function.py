import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)

# --- d2l 辅助函数 (简化版，用于独立运行) ---
def use_svg_display():
    pass

def set_figsize(figsize=(3.5, 2.5)):
    pass

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

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

# --- 课程代码开始 ---

# 1. 掩蔽softmax操作
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作
    
    参数:
        X: 3D张量，形状为 (batch_size, num_queries, num_keys)
        valid_lens: 1D或2D张量，指定每个序列的有效长度
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 如果valid_lens是1D，为每个查询复制相同的有效长度
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 如果valid_lens是2D，展平它
            valid_lens = valid_lens.reshape(-1)
        
        # 将3D张量reshape为2D，应用mask，然后恢复形状
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

print("--- 1. 演示掩蔽softmax操作 ---")
# 示例1: valid_lens是1D张量
print("示例1 (valid_lens是1D):")
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))

# 示例2: valid_lens是2D张量
print("\n示例2 (valid_lens是2D):")
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

# 2. 加性注意力 (Bahdanau注意力)
class AdditiveAttention(nn.Module):
    """加性注意力/Bahdanau注意力
    
    公式: score(q,k) = v^T * tanh(W_q*q + W_k*k)
    其中 q是查询，k是键，v、W_q、W_k是可学习参数
    """
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """
        参数:
            queries: (batch_size, num_queries, query_size)
            keys: (batch_size, num_keys, key_size)  
            values: (batch_size, num_keys, value_size)
            valid_lens: (batch_size,) 或 (batch_size, num_queries)
        """
        # 投影到隐藏维度
        queries, keys = self.W_q(queries), self.W_k(keys)
        
        # 广播加法: (batch_size, num_queries, 1, num_hiddens) + (batch_size, 1, num_keys, num_hiddens)
        # 结果: (batch_size, num_queries, num_keys, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        
        # 计算注意力分数: (batch_size, num_queries, num_keys)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        # 加权求和: (batch_size, num_queries, value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)

print("\n--- 2. 演示加性注意力 ---")
# 测试数据准备
queries_add = torch.normal(0, 1, (2, 1, 20))  # 2个样本，1个查询，20维
keys_add = torch.ones((2, 10, 2))  # 2个样本，10个键，2维
values_add = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)  # 2个样本，10个值，4维
valid_lens_add = torch.tensor([2, 6])  # 第一个样本有效长度2，第二个样本有效长度6

print(f"查询形状: {queries_add.shape}")
print(f"键形状: {keys_add.shape}")  
print(f"值形状: {values_add.shape}")
print(f"有效长度: {valid_lens_add}")

attention_add = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention_add.eval()
output_add = attention_add(queries_add, keys_add, values_add, valid_lens_add)
print(f"加性注意力输出形状: {output_add.shape}")
print(f"注意力权重形状: {attention_add.attention_weights.shape}")

# 可视化加性注意力的权重
print("\n可视化加性注意力权重...")
show_heatmaps(attention_add.attention_weights.reshape((1, 1, 2, 10)),
              xlabel='键 (Keys)', ylabel='查询 (Queries)',
              titles=['加性注意力权重'],
              save_path='additive_attention_weights.png')

print("\n" + "="*60)

# 3. 缩放点积注意力 (Transformer注意力)
class DotProductAttention(nn.Module):
    """缩放点积注意力/Transformer注意力
    
    公式: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    其中 d_k 是键的维度，用于缩放防止梯度消失
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        参数:
            queries: (batch_size, num_queries, d_k)
            keys: (batch_size, num_keys, d_k)
            values: (batch_size, num_keys, d_v)
            valid_lens: (batch_size,) 或 (batch_size, num_queries)
        """
        d = queries.shape[-1]  # 键的维度
        # 计算注意力分数: (batch_size, num_queries, num_keys)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 加权求和: (batch_size, num_queries, d_v)
        return torch.bmm(self.dropout(self.attention_weights), values)

print("\n--- 3. 演示缩放点积注意力 ---")
# 为了匹配维度，查询和键的最后一维必须相同
queries_dot = torch.normal(0, 1, (2, 1, 2))  # 2个样本，1个查询，2维
keys_dot = torch.ones((2, 10, 2))  # 2个样本，10个键，2维  
values_dot = values_add  # 复用之前的值：2个样本，10个值，4维
valid_lens_dot = valid_lens_add  # 复用之前的有效长度

print(f"查询形状: {queries_dot.shape}")
print(f"键形状: {keys_dot.shape}")
print(f"值形状: {values_dot.shape}")
print(f"有效长度: {valid_lens_dot}")

attention_dot = DotProductAttention(dropout=0.5)
attention_dot.eval()
output_dot = attention_dot(queries_dot, keys_dot, values_dot, valid_lens_dot)
print(f"缩放点积注意力输出形状: {output_dot.shape}")
print(f"注意力权重形状: {attention_dot.attention_weights.shape}")

# 可视化缩放点积注意力的权重
print("\n可视化缩放点积注意力权重...")
show_heatmaps(attention_dot.attention_weights.reshape((1, 1, 2, 10)),
              xlabel='键 (Keys)', ylabel='查询 (Queries)',
              titles=['缩放点积注意力权重'],
              save_path='dot_product_attention_weights.png')

print("\n" + "="*60)
print("=== 两种注意力机制对比 ===")
print("1. 加性注意力 (Bahdanau):")
print("   - 可以处理不同维度的查询和键")
print("   - 使用可学习的参数进行特征变换")  
print("   - 计算复杂度较高")
print(f"   - 输出: {output_add.shape}")

print("\n2. 缩放点积注意力 (Transformer):")
print("   - 要求查询和键的维度相同")
print("   - 直接计算点积，简单高效")
print("   - 使用缩放因子防止梯度消失")
print(f"   - 输出: {output_dot.shape}")

print("\n生成的图片文件:")
print("1. additive_attention_weights.png - 加性注意力权重模式")
print("2. dot_product_attention_weights.png - 缩放点积注意力权重模式")
print("\n注意力权重热图说明:")
print("- 每一行代表一个查询对所有键的注意力分布")
print("- 颜色越深表示注意力权重越大")
print("- 由于mask的作用，超出有效长度的位置权重为0（深蓝色）")