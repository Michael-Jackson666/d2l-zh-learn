import collections
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
import time

# --- 0. 基础编码器-解码器架构 ---
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# --- 数据处理函数 ---
def read_data_nmt():
    """载入"英语－法语"数据集"""
    # 创建示例数据，实际使用时可以替换为真实数据文件
    data_dir = os.path.join('..', 'data', 'fra-eng')
    file_path = os.path.join(data_dir, 'fra.txt')
    
    # 如果文件不存在，创建一些示例数据
    if not os.path.exists(file_path):
        os.makedirs(data_dir, exist_ok=True)
        sample_data = [
            "Go.\tVa !\n",
            "Hi.\tSalut !\n", 
            "Run!\tCours !\n",
            "Run!\tCourez !\n",
            "Who?\tQui ?\n",
            "Wow!\tCa alors !\n",
            "Fire!\tAu feu !\n",
            "Help!\tA l'aide !\n",
            "Jump.\tSaute.\n",
            "Stop!\tCa suffit !\n",
            "Wait.\tAttends.\n",
            "Come.\tViens.\n"
        ]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(sample_data * 50)  # 复制以增加数据量
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    """预处理"英语－法语"数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 用空格替换不间断空格，并转换为小写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """词元化"英语－法语"数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

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

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

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
    dataset = TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = DataLoader(dataset, batch_size, shuffle=True)
    return data_iter, src_vocab, tgt_vocab

# --- 工具类 ---
class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    """简化的动画绘制类（用于训练进度显示）"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data = []

    def add(self, x, y):
        """添加数据点并显示当前训练进度"""
        if not hasattr(y, "__len__"):
            y = [y]
        self.data.append((x, y[0]))
        print(f"Epoch {x}: {self.ylabel} = {y[0]:.6f}")

def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    # elif torch.backends.mps.is_available():
    #     return torch.device('mps')
    return torch.device('cpu')

# --- 1. Encoder (编码器) ---
#@save
class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层，将输入的词元索引转换为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 使用GRU作为核心循环层
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输入X的形状: (批量大小, 时间步数)
        # 1. 经过嵌入层
        # 输出X的形状: (批量大小, 时间步数, 嵌入维度)
        X = self.embedding(X)
        # 2. PyTorch的RNN层期望输入的第一维是时间步，所以进行转置
        # X的形状变为: (时间步数, 批量大小, 嵌入维度)
        X = X.permute(1, 0, 2)
        # 3. 通过GRU层
        # output的形状: (时间步数, 批量大小, 隐藏单元数)
        # state的形状: (层数, 批量大小, 隐藏单元数)
        output, state = self.rnn(X)
        return output, state

# --- 2. Decoder (解码器) ---
class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 解码器的GRU输入维度是“嵌入维度 + 隐藏单元数”
        # 因为每个时间步的输入都拼接了上下文向量
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        # 输出层，将隐藏状态映射到词表
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # 直接使用编码器最后一个时间步的隐状态来初始化解码器的隐状态
        return enc_outputs[1]

    def forward(self, X, state):
        # 输入X的形状: (批量大小, 时间步数)
        # 1. 经过嵌入层并转置
        # X的形状变为: (时间步数, 批量大小, 嵌入维度)
        X = self.embedding(X).permute(1, 0, 2)
        
        # 2. 准备上下文向量 (context)
        # state[-1] 是编码器最后一层的最后一个时间步的隐状态
        # 形状是 (批量大小, 隐藏单元数)
        # 将其复制 num_steps 次，使其形状变为 (时间步数, 批量大小, 隐藏单元数)
        context = state[-1].repeat(X.shape[0], 1, 1)
        
        # 3. 将输入和上下文向量在特征维度上拼接
        X_and_context = torch.cat((X, context), 2)
        
        # 4. 通过GRU层
        output, state = self.rnn(X_and_context, state)
        
        # 5. 通过输出层
        output = self.dense(output).permute(1, 0, 2)
        # output的最终形状: (批量大小, 时间步数, 词表大小)
        # state的最终形状: (层数, 批量大小, 隐藏单元数)
        return output, state

# --- 3. 损失函数 (带遮蔽) ---
#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# --- 4. 训练函数 ---
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss',
                        xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 准备解码器的输入：在目标序列Y前添加<bos>
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            # 前向传播
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

# --- 5. 预测函数 ---
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    
    # 编码
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    
    # 解码
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 将上一时刻的预测作为下一时刻的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), None

#@save
def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# --- 6. 执行训练和评估 ---
if __name__ == '__main__':
    print("正在初始化 seq2seq 模型...")
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 50, try_gpu()  # 减少训练轮数用于测试

    print("正在加载数据...")
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    print("正在构建模型...")
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    
    print(f"开始训练，设备: {device}")
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    print("开始评估...")
    # 评估
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, _ = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')