#!/usr/bin/env python3
"""
独立的翻译脚本，使用训练好的Transformer模型进行英语到法语的翻译
"""

import sys
import os

# 添加当前目录到Python路径，以便导入transformer模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模块时使用实际的文件名
import importlib.util
spec = importlib.util.spec_from_file_location("transformer_d2l", "transformer-d2l.py")
transformer_d2l = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_d2l)

# 导入需要的函数
translate_with_saved_model = transformer_d2l.translate_with_saved_model
try_gpu = transformer_d2l.try_gpu

def main():
    model_path = 'transformer_fra_eng.pth'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在！")
        print("请先运行 transformer-d2l.py 进行训练。")
        return
    
    print("=== Transformer 英语到法语翻译器 ===")
    print("输入英语句子，程序将翻译成法语")
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 40)
    
    device = try_gpu()
    
    while True:
        try:
            sentence = input("\n请输入英语句子: ").strip()
            
            if sentence.lower() in ['quit', 'exit', '']:
                print("再见！")
                break
            
            # 确保句子以点号结尾
            if not sentence.endswith('.'):
                sentence += ' .'
            
            # 进行翻译
            results = translate_with_saved_model(
                model_path, [sentence], num_steps=10, device=device
            )
            
            print(f"法语翻译: {results[0]}")
            
        except KeyboardInterrupt:
            print("\n\n程序被中断，再见！")
            break
        except Exception as e:
            print(f"翻译时出错: {e}")

if __name__ == '__main__':
    main()
