#!/usr/bin/env python3
"""
使用训练好的Transformer模型进行翻译的简单示例
只需要transformer_inference.py和模型文件即可运行
"""

from transformer_inference import TransformerTranslator, quick_translate, quick_translate_batch

def main():
    # 模型文件路径
    model_path = 'transformer_fra_eng.pth'
    
    print("=== Transformer 英法翻译器 ===")
    
    # 方法1：使用便捷函数进行单句翻译
    print("\n方法1：快速单句翻译")
    sentence = "hello world ."
    try:
        translation = quick_translate(model_path, sentence)
        print(f"英语: {sentence}")
        print(f"法语: {translation}")
    except FileNotFoundError:
        print(f"错误: 模型文件 {model_path} 不存在!")
        print("请先运行 transformer-d2l.py 进行训练")
        return
    
    # 方法2：使用便捷函数进行批量翻译
    print("\n方法2：快速批量翻译")
    sentences = [
        "good morning .",
        "how are you ?",
        "i love you .",
        "thank you very much ."
    ]
    
    translations = quick_translate_batch(model_path, sentences)
    for eng, fra in zip(sentences, translations):
        print(f"英语: {eng} => 法语: {fra}")
    
    # 方法3：使用翻译器类 (推荐用于多次翻译)
    print("\n方法3：使用翻译器类")
    translator = TransformerTranslator(model_path)
    
    # 交互式翻译
    print("\n开始交互式翻译 (输入 'quit' 退出):")
    while True:
        try:
            user_input = input("请输入英语句子: ").strip()
            if user_input.lower() in ['quit', 'exit', '']:
                break
            
            translation = translator.translate(user_input)
            print(f"法语翻译: {translation}")
            
        except KeyboardInterrupt:
            print("\n退出程序")
            break
        except Exception as e:
            print(f"翻译错误: {e}")

if __name__ == '__main__':
    main()
