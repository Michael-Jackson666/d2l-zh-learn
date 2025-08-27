import random
import torch
import collections
import re
from d2l import torch as d2l
'''语言模型与数据集'''

# --- 第 8.2 节中的辅助函数 (为了让脚本可以独立运行) ---
# 这部分代码是上一节内容，但本节会用到，所以我们将其包含进来。

# 定义数据集的下载路径和哈希值
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将《时光机器》数据集加载到文本行的列表中"""
    # 下载文件
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 使用正则表达式将所有非字母字符替换为空格，然后去除首尾空格并转为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元(token)"""
    if token == 'word':
        # 按空格分割，将每行文本变成一个单词列表
        return [line.split() for line in lines]
    elif token == 'char':
        # 将每行文本变成一个字符列表
        return [list(line) for line in lines]
    else:
        raise ValueError(f'错误：未知词元类型：{token}')

def count_corpus(tokens):
    """统计词元的频率"""
    # 如果输入是二维列表（行的列表），则先将其展平为一维列表
    if len(tokens) > 0 and isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    # 使用collections.Counter快速统计每个词元的出现次数
    return collections.Counter(tokens)

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        # 统计词频并按频率降序排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 构建索引到词元(idx_to_token)和词元到索引(token_to_idx)的映射
        # <unk>是未知词元，索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """查找单个或多个词元的索引"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """根据索引查找词元"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """未知词元的索引"""
        return 0

    @property
    def token_freqs(self):
        """返回词频列表"""
        return self._token_freqs
    
# --- 第 8.3 节的核心代码 ---

# 1. 自然语言统计分析
print("--- 1. 自然语言统计分析 ---")
# 按单词进行词元化
tokens = tokenize(read_time_machine())
# 将所有行的词元合并成一个长列表，称为语料库(corpus)
corpus = [token for line in tokens for token in line]
vocab = Vocab(corpus)
print("最常见的10个单词 (一元语法):", vocab.token_freqs[:10])

# 绘制一元语法的词频图 (齐普夫定律)
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x (rank)', ylabel='frequency: n(x)',
         xscale='log', yscale='log', figsize=(3.5, 2.5))
# d2l.plt.show()

# 分析二元语法和三元语法
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = Vocab(bigram_tokens)
print("\n最常见的10个词对 (二元语法):", bigram_vocab.token_freqs[:10])

trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)
print("\n最常见的10个词组 (三元语法):", trigram_vocab.token_freqs[:10])

# 在同一张图上绘制三者的频率分布
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x (rank)',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'], figsize=(4.5, 3))
# d2l.plt.show()


# 2. 读取长序列数据的策略
def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从一个随机的偏移量开始，以增加数据的多样性
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 计算子序列的总数（向下取整）
    num_subseqs = (len(corpus) - 1) // num_steps
    # 生成所有子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 随机打乱这些起始索引，这是“随机采样”的关键
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始，长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    # 计算总共可以生成的批次数
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 每次抽取batch_size个随机的起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        # X是特征，Y是标签。Y是X向右移动一个时间步的结果
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 同样使用随机偏移量来增加多样性
    offset = random.randint(0, num_steps - 1)
    # 计算可以构造成规整批次的总词元数
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    # 提取有效部分的特征(Xs)和标签(Ys)
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    # 将长序列重塑成(batch_size, num_subsequences_per_batch)的形状
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    # 计算每个批次有多少个时间步（num_steps）
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        # 在重塑后的张量上，按列切片，得到小批量
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

# 3. 将数据加载器封装成类
class SeqDataLoader:
    """一个加载序列数据的迭代器类"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        # 加载并处理语料库
        lines = read_time_machine()
        tokens = tokenize(lines, 'char') # 后续模型将使用字符级
        self.vocab = Vocab(tokens)
        self.corpus = [self.vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            self.corpus = self.corpus[:max_tokens]
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        """返回数据迭代器"""
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """最终的加载函数，返回数据迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# --- 演示两种采样方法 ---
if __name__ == '__main__':
    my_seq = list(range(35))
    print("\n--- 随机采样演示 ---")
    # 随机采样的批次之间在原始序列中是不连续的
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

    print("\n--- 顺序分区演示 ---")
    # 顺序分区的批次之间在原始序列中是连续的
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)