# 导入所有必要的库
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import collections
import re
import random

# --- 1. 设备配置 (适配Mac的MPS, CUDA或CPU) ---
def get_device():
    """获取可用的设备 (MPS, CUDA 或 CPU)"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

device = get_device()
print(f'将使用设备: {device}')


# --- 2. 数据加载与预处理 (依赖前面章节的函数) ---
# 为了让脚本可以独立运行，我们将之前章节的所有相关函数都包含进来。

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError(f'错误：未知词元类型：{token}')

def count_corpus(tokens):
    if len(tokens) > 0 and isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None: tokens = []
        if reserved_tokens is None: reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq: break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    def __len__(self): return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    @property
    def unk(self): return 0
    @property
    def token_freqs(self): return self._token_freqs

def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)
    def data(pos):
        return corpus[pos: pos + num_steps]
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps - 1)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# --- 3. 从零实现RNN模型 ---

def get_params(vocab_size, num_hiddens, device):
    """初始化模型参数"""
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # 使用正态分布初始化权重，乘以0.01以避免初始值过大
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    # W_xh: 输入 -> 隐藏层 的权重
    W_xh = normal((num_inputs, num_hiddens))
    # W_hh: 上一时刻的隐藏状态 -> 当前隐藏状态 的权重
    W_hh = normal((num_hiddens, num_hiddens))
    # b_h: 隐藏层的偏置
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    # W_hq: 隐藏层 -> 输出层 的权重
    W_hq = normal((num_hiddens, num_outputs))
    # b_q: 输出层的偏置
    b_q = torch.zeros(num_outputs, device=device)
    
    # 附加梯度，以便进行反向传播
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    """初始化隐状态"""
    # 返回一个元组，第一个元素是全零张量
    # 形状为 (批量大小, 隐藏单元数)
    # 使用元组是为了将来兼容LSTM等具有多个状态的模型
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    """
    定义一个时间步内的RNN计算逻辑
    :param inputs: 形状为 (时间步数, 批量大小, 词表大小) 的输入数据
    :param state:  上一时间步的隐状态
    :param params: 模型参数列表
    :return: 输出张量和更新后的隐状态
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # `inputs` 的形状是 (时间步数, 批量大小, 词表大小)
    # `X` 在循环中是每个时间步的输入，形状为 (批量大小, 词表大小)
    for X in inputs:
        # 核心计算公式：H_t = tanh(X_t @ W_xh + H_{t-1} @ W_hh + b_h)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 输出层计算：Y_t = H_t @ W_hq + b_q
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    
    # 将所有时间步的输出拼接起来
    # 最终形状为 (时间步数 * 批量大小, 词表大小)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    """从零开始实现的RNN模型类"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """模型的前向传播"""
        # 1. 将输入的索引(X)转换为独热编码
        # X 原始形状: (批量大小, 时间步数)
        # X.T 形状: (时间步数, 批量大小)
        # one_hot后形状: (时间步数, 批量大小, 词表大小)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32).to(device)
        # 2. 调用核心的rnn函数
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """返回初始隐状态"""
        return self.init_state(batch_size, self.num_hiddens, device)

# --- 4. 预测函数 ---
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在`prefix`后面生成新字符"""
    # 初始化隐状态，批量大小为1，因为我们一次只预测一个序列
    state = net.begin_state(batch_size=1, device=device)
    # 将prefix的第一个字符作为初始输出
    outputs = [vocab[prefix[0]]]
    # 定义一个函数，获取当前最后一个字符作为下一时间步的输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # "预热"阶段：用prefix中的字符来更新隐状态，但不生成预测
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # 预测阶段：生成num_preds个新字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # 选择概率最高的字符作为预测结果
        outputs.append(int(y.argmax(dim=1).reshape(1)))
        
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# --- 5. 梯度裁剪与训练 ---
def grad_clipping(net, theta):
    """裁剪梯度以防止梯度爆炸"""
    # 兼容我们自己实现的模型和PyTorch的nn.Module
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 计算所有参数梯度的L2范数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        # 如果范数超过阈值theta，则按比例缩小梯度
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练一个迭代周期"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 用于累加训练损失和词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 第一次迭代或使用随机采样时，初始化隐状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 使用顺序分区时，需要分离隐状态，避免梯度传遍整个数据集
            for s in state:
                s.detach_()
        
        # Y的形状是(批量大小, 时间步数)，转置后展平为一维向量
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        # 前向传播
        y_hat, state = net(X, state)
        # 计算损失
        l = loss(y_hat, y.long()).mean()

        # 反向传播和更新
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1) # 裁剪梯度
            updater.step()
        else: # 我们自定义的updater
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        
        metric.add(l * y.numel(), y.numel())
    # 返回困惑度和速度
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """高级API，用于训练和评估模型"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    
    # 定义优化器
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # 自定义的更新函数
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    
    # 训练循环
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

# --- 6. 执行训练 ---
if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    
    num_hiddens = 512
    num_epochs, lr = 500, 1
    
    # 实例化从零开始的模型
    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                          init_rnn_state, rnn)
                          
    # 使用顺序分区进行训练
    print("\n--- 开始使用顺序分区进行训练 ---")
    train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False)
    
    # 使用随机抽样进行训练
    print("\n--- 开始使用随机抽样进行训练 ---")
    train_iter_random, _ = load_data_time_machine(batch_size, num_steps, use_random_iter=True)
    net_random = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                                 init_rnn_state, rnn)
    train_ch8(net_random, train_iter_random, vocab, lr, num_epochs, device, use_random_iter=True)