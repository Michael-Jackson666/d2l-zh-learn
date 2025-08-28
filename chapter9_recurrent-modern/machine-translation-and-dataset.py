import os
import torch
from d2l import torch as d2l
import collections
import re

# --- 1. 下载和读取数据 ---
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()


# --- 2. 文本预处理 ---
#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        # 判断当前字符是否是标点符号，并且前一个字符不是空格
        return char in set(',.!?') and prev_char != ' '

    # 用空格替换不间断空格，并转换为小写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


# --- 3. 词元化 ---
#@save
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        # 如果设置了num_examples，则只处理指定数量的样本
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            # 按空格分割成单词列表
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# --- 4. 词表构建与序列处理 ---
# 自定义Vocab类以确保兼容性
class Vocab:
    """词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频
        counter = collections.Counter()
        for token_list in tokens:
            counter.update(token_list)
        
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 构建词汇表：保留词元 + 高频词元
        self.idx_to_token = list(reserved_tokens)
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
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):  # 未知词元的索引为0
        return 0
    
    @property 
    def token_freqs(self):
        return self._token_freqs

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)
#@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

#@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    # 1. 将文本词元列表转换为数字索引列表
    lines = [vocab[l] for l in lines]
    # 2. 在每个序列末尾添加<eos>词元
    lines = [l + [vocab['<eos>']] for l in lines]
    # 3. 对每个序列进行截断或填充
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    # 4. 计算每个序列的有效长度（不包括填充部分）
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# --- 5. 整合的数据加载器 ---
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    # 完整流程：读取 -> 预处理 -> 词元化
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    
    # 分别为源语言和目标语言构建词表
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
                          
    # 将词元化后的文本转换为固定长度的数字张量
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    
    # 封装为PyTorch的TensorDataset和DataLoader
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

# --- 6. 演示 ---
if __name__ == '__main__':
    # 原始文本预览
    raw_text = read_data_nmt()
    print("--- 原始文本 ---")
    print(raw_text[:75])
    
    # 预处理后文本预览
    text = preprocess_nmt(raw_text)
    print("\n--- 预处理后文本 ---")
    print(text[:80])
    
    # 词元化后预览
    source, target = tokenize_nmt(text)
    print("\n--- 词元化后 ---")
    print("源语言:", source[:6])
    print("目标语言:", target[:6])
    
    # 加载数据并查看第一个批次
    print("\n--- 加载数据迭代器并查看第一个批次 ---")
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break