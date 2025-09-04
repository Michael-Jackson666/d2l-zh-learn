#!/usr/bin/env python3
"""
最简洁的Transformer模型调用代码
只需要这个文件和模型文件(.pth)即可进行翻译
"""

import torch
from torch import nn
import math
import os

# === 最小化的模型架构定义 ===
class MinimalTransformer:
    """最简化的Transformer翻译器"""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.net, self.src_vocab, self.tgt_vocab = self._load_model()
    
    def _sequence_mask(self, X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                          device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    def _masked_softmax(self, X, valid_lens):
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = self._sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    
    def _truncate_pad(self, line, num_steps, padding_token):
        if len(line) > num_steps:
            return line[:num_steps]
        return line + [padding_token] * (num_steps - len(line))
    
    def _load_model(self):
        """加载模型的最简化版本"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 直接加载整个模型对象
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 如果是完整的checkpoint格式
        if 'model_state_dict' in checkpoint:
            # 动态重建模型架构
            from transformer_inference import TransformerEncoder, TransformerDecoder, EncoderDecoder
            
            src_vocab = checkpoint['src_vocab']
            tgt_vocab = checkpoint['tgt_vocab']
            config = checkpoint['model_config']
            
            encoder = TransformerEncoder(
                config['src_vocab_size'], config['key_size'], config['query_size'], 
                config['value_size'], config['num_hiddens'], config['norm_shape'], 
                config['ffn_num_input'], config['ffn_num_hiddens'], config['num_heads'],
                config['num_layers'], config['dropout'])
            decoder = TransformerDecoder(
                config['tgt_vocab_size'], config['key_size'], config['query_size'], 
                config['value_size'], config['num_hiddens'], config['norm_shape'], 
                config['ffn_num_input'], config['ffn_num_hiddens'], config['num_heads'],
                config['num_layers'], config['dropout'])
            net = EncoderDecoder(encoder, decoder)
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 直接加载模型对象
            net = checkpoint
            src_vocab = None
            tgt_vocab = None
        
        net.to(self.device)
        net.eval()
        print(f"模型已加载到 {self.device}")
        return net, src_vocab, tgt_vocab
    
    def translate(self, sentence):
        """翻译句子"""
        if not sentence.strip().endswith('.'):
            sentence = sentence.strip() + ' .'
        
        # 预处理
        src_tokens = self.src_vocab[sentence.lower().split(' ')] + [self.src_vocab['<eos>']]
        enc_valid_len = torch.tensor([len(src_tokens)], device=self.device)
        num_steps = 15
        src_tokens = self._truncate_pad(src_tokens, num_steps, self.src_vocab['<pad>'])
        enc_X = torch.unsqueeze(
            torch.tensor(src_tokens, dtype=torch.long, device=self.device), dim=0)
        
        # 编码
        enc_outputs = self.net.encoder(enc_X, enc_valid_len)
        dec_state = self.net.decoder.init_state(enc_outputs, enc_valid_len)
        
        # 解码
        dec_X = torch.unsqueeze(torch.tensor(
            [self.tgt_vocab['<bos>']], dtype=torch.long, device=self.device), dim=0)
        output_seq = []
        
        for _ in range(num_steps):
            Y, dec_state = self.net.decoder(dec_X, dec_state)
            dec_X = Y.argmax(dim=2)
            pred = dec_X.squeeze(dim=0).type(torch.int32).item()
            if pred == self.tgt_vocab['<eos>']:
                break
            output_seq.append(pred)
        
        return ' '.join(self.tgt_vocab.to_tokens(output_seq))

# === 最简单的使用方式 ===
def translate_sentence(model_path, sentence):
    """
    最简单的翻译函数
    
    Args:
        model_path: 模型文件路径
        sentence: 要翻译的英语句子
    
    Returns:
        翻译后的法语句子
    """
    translator = MinimalTransformer(model_path)
    return translator.translate(sentence)

# === 示例用法 ===
if __name__ == '__main__':
    # 使用示例
    model_file = 'transformer_fra_eng.pth'
    
    if not os.path.exists(model_file):
        print(f"错误: 模型文件 {model_file} 不存在!")
        print("请先运行训练脚本生成模型文件")
    else:
        # 创建翻译器
        translator = MinimalTransformer(model_file)
        
        # 测试翻译
        test_sentences = [
            "hello .",
            "good morning .",
            "how are you ?",
            "i love you ."
        ]
        
        print("=== 翻译测试 ===")
        for sentence in test_sentences:
            try:
                translation = translator.translate(sentence)
                print(f"{sentence} => {translation}")
            except Exception as e:
                print(f"翻译 '{sentence}' 时出错: {e}")
        
        # 交互式翻译
        print("\n=== 交互式翻译 ===")
        print("输入英语句子进行翻译 (输入 'quit' 退出):")
        
        while True:
            try:
                user_input = input("英语: ").strip()
                if user_input.lower() in ['quit', 'exit', '']:
                    break
                
                translation = translator.translate(user_input)
                print(f"法语: {translation}")
                
            except KeyboardInterrupt:
                print("\n程序退出")
                break
            except Exception as e:
                print(f"翻译错误: {e}")
