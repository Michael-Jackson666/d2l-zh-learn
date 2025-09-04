#!/usr/bin/env python3
"""
批量翻译示例脚本
演示如何使用保存的Transformer模型进行批量英语到法语翻译
"""

import sys
import os
import importlib.util

# 导入transformer模块
spec = importlib.util.spec_from_file_location("transformer_d2l", "transformer-d2l.py")
transformer_d2l = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_d2l)

def main():
    model_path = 'transformer_fra_eng.pth'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在！")
        print("请先运行 transformer-d2l.py 进行训练。")
        return
    
    # 示例英语句子
    test_sentences = [
        'hello .',
        'how are you ?',
        'i am fine .',
        'good morning .',
        'see you later .',
        'i love you .',
        'thank you .',
        'what is your name ?',
        'i am learning french .',
        'this is a beautiful day .'
    ]
    
    print("=== 批量翻译示例 ===")
    print("使用训练好的Transformer模型进行英语到法语翻译")
    print("=" * 50)
    
    try:
        # 进行批量翻译
        device = transformer_d2l.try_gpu()
        results = transformer_d2l.translate_with_saved_model(
            model_path, test_sentences, num_steps=15, device=device
        )
        
        print("\n翻译结果对比:")
        print("-" * 50)
        for eng, fra in zip(test_sentences, results):
            print(f"英语: {eng}")
            print(f"法语: {fra}")
            print("-" * 30)
            
    except Exception as e:
        print(f"翻译时出错: {e}")

if __name__ == '__main__':
    main()
