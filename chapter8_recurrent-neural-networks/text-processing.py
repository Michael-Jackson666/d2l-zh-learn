import collections
import re
from d2l import torch as d2l
'''文本预处理'''

# --- 1. 数据读取 ---
# 定义数据源并提供下载功能
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 使用正则表达式将非字母字符替换为空格，并转为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# --- 2. 词元化 ---
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError(f'错误：未知词元类型：{token}')

# --- 3. 词表构建 ---
def count_corpus(tokens):
    """统计词元的频率"""
    # 如果tokens是2D列表，则将其展平
    if len(tokens) > 0 and isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
    
class Vocab:
    """文本词表类，负责将文本词元映射到数字索引"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        构造函数
        :param tokens: 语料库，可以是词元的一维或二维列表
        :param min_freq: 最小词频，低于此频率的词将被丢弃
        :param reserved_tokens: 保留的特殊词元列表，如<pad>, <bos>, <eos>
        """
        if tokens is None: tokens = []
        if reserved_tokens is None: reserved_tokens = []
        
        # 统计词频
        counter = count_corpus(tokens)
        # 按词频从高到低排序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 构建从索引到词元(idx_to_token)和从词元到索引(token_to_idx)的映射
        # '<unk>' 代表未知词元，它的索引固定为0。然后添加其他保留词元。
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        # 遍历排序后的词频列表，将满足条件的词元添加到词表中
        for token, freq in self._token_freqs:
            if freq < min_freq:
                # 后面的词频更低，直接跳出循环
                break
            if token not in self.token_to_idx:
                # 将新词元添加到映射中
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """返回词表的大小"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """查找单个或多个词元的索引"""
        if not isinstance(tokens, (list, tuple)):
            # 如果是单个词元，返回其索引。如果不在词表中，返回未知词元的索引(0)
            return self.token_to_idx.get(tokens, self.unk)
        # 如果是词元列表，递归调用自身
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """根据索引列表返回对应的词元列表"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """未知词元的索引始终为0"""
        return 0

    @property
    def token_freqs(self):
        """返回按频率排序的词元列表"""
        return self._token_freqs


# --- 4. 整合函数：加载并处理时光机器数据集 ---
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    # 使用字符进行词元化
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 将所有文本行展平到一个单一的索引列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# --- 5. 运行与演示 ---
if __name__ == '__main__':
    # 演示读取和词元化（按单词）
    print("--- 演示读取与单词词元化 ---")
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print("第一行:", lines[0])
    print("第十一行:", lines[10])
    
    tokens = tokenize(lines, 'word')
    print("\n前5行的词元:")
    for i in range(5):
        print(f"行 {i}: {tokens[i]}")

    # 演示构建词表和转换
    print("\n--- 演示构建词表与转换 ---")
    vocab = Vocab(tokens)
    print("词表示例（前10个）:", list(vocab.token_to_idx.items())[:10])
    
    print("\n文本行转换为索引:")
    sample_text = tokens[0]
    sample_indices = vocab[sample_text]
    print(f"文本: {sample_text}")
    print(f"索引: {sample_indices}")
    print(f"索引转回文本: {vocab.to_tokens(sample_indices)}")

    # 运行最终的整合函数（按字符）
    print("\n--- 运行最终的字符级处理函数 ---")
    corpus, vocab = load_corpus_time_machine()
    print(f"语料库总长度 (字符数): {len(corpus)}")
    print(f"词表大小 (独立字符数): {len(vocab)}")
    print("语料库前20个字符的索引:", corpus[:20])