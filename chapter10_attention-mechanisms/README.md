# Transformer 英语到法语翻译模型

这个项目实现了一个完整的Transformer模型用于英语到法语的机器翻译。

## 文件说明

### 训练相关
- `transformer-d2l.py`: 主要的Transformer模型实现和训练脚本

### 模型调用相关
- `transformer_inference.py`: 完整的推理模块，包含所有必要的类和函数
- `use_model.py`: 使用示例，展示多种调用方式
- `simple_translator.py`: 最简化的翻译器，只需要这个文件和模型即可
- `translate.py`: 交互式翻译脚本 (依赖主训练文件)
- `batch_translate.py`: 批量翻译示例脚本 (依赖主训练文件)

### 生成的文件
- `transformer_fra_eng.pth`: 训练好的模型文件（训练后生成）

## 使用方法

### 1. 训练模型

首次运行时，需要训练模型：

```bash
python transformer-d2l.py
```

这将：
- 下载并处理英语-法语数据集
- 训练Transformer模型
- 自动保存训练好的模型到 `transformer_fra_eng.pth`
- 进行测试翻译并显示BLEU分数

### 2. 使用训练好的模型

训练完成后，有多种方式使用模型：

#### 方式1：使用完整推理模块（推荐）

```python
from transformer_inference import TransformerTranslator

# 创建翻译器
translator = TransformerTranslator('transformer_fra_eng.pth')

# 翻译单个句子
translation = translator.translate("hello world")
print(translation)

# 批量翻译
sentences = ["hello .", "good morning ."]
translations = translator.translate_batch(sentences)
```

#### 方式2：使用便捷函数

```python
from transformer_inference import quick_translate, quick_translate_batch

# 快速翻译
result = quick_translate('transformer_fra_eng.pth', "hello world")

# 快速批量翻译
results = quick_translate_batch('transformer_fra_eng.pth', ["hello .", "goodbye ."])
```

#### 方式3：使用最简化翻译器

```python
from simple_translator import MinimalTransformer

translator = MinimalTransformer('transformer_fra_eng.pth')
translation = translator.translate("hello world")
```

#### 方式4：运行示例脚本

```bash
# 运行使用示例
python use_model.py

# 运行简化翻译器
python simple_translator.py

# 交互式翻译
python translate.py

# 批量翻译示例
python batch_translate.py
```

### 3. 部署建议

如果只需要进行翻译（不需要训练），只需要以下文件：

**最简部署（推荐）：**
- `transformer_inference.py`：完整推理模块
- `transformer_fra_eng.pth`：训练好的模型
- `use_model.py`：使用示例

**超简部署：**
- `simple_translator.py`：最简化翻译器
- `transformer_fra_eng.pth`：训练好的模型

## 模型特点

- **完全基于注意力机制**: 无RNN或CNN结构
- **编码器-解码器架构**: 2层编码器和解码器
- **多头注意力**: 4个注意力头
- **位置编码**: 保持序列位置信息
- **残差连接和层归一化**: 提高训练稳定性

## 模型参数

- 隐藏层维度: 32
- 层数: 2
- 注意力头数: 4
- Dropout: 0.1
- 前馈网络隐藏层: 64

## API说明

### TransformerTranslator类

```python
# 初始化
translator = TransformerTranslator(model_path, device=None)

# 翻译单句
translation = translator.translate(sentence, num_steps=15)

# 批量翻译
translations = translator.translate_batch(sentences, num_steps=15)
```

### 便捷函数

```python
# 快速单句翻译
translation = quick_translate(model_path, sentence, device=None)

# 快速批量翻译
translations = quick_translate_batch(model_path, sentences, device=None)
```

## 依赖库

- torch
- d2l (仅训练时需要)
- collections
- math
- os

## 示例输出

```
英语: i love you .
法语: je t'aime .

英语: good morning .
法语: bonjour .

英语: how are you ?
法语: comment allez-vous ?
```

## 注意事项

1. 首次运行需要下载数据集，请确保网络连接正常
2. 模型文件大约几MB，训练完成后会自动保存
3. 推理时只需要PyTorch，不需要d2l库
4. 可以通过修改配置参数来调整模型性能
