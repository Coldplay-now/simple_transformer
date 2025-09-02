# Simple Transformer Implementation

一个简单的Transformer模型实现，用于学习和理解Transformer架构的核心概念。

## 项目概述

本项目实现了一个简化版的Transformer模型，包含以下核心组件：
- Multi-Head Self-Attention机制
- Position Encoding（位置编码）
- Feed Forward Network（前馈网络）
- Layer Normalization
- 完整的训练和推理流程

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── PRD.md                    # 产品需求文档
├── requirements.txt          # 依赖包列表
├── config/                   # 配置文件目录
│   └── model_config.yaml    # 模型配置
├── data/                     # 数据目录
│   ├── sample_text.txt      # 示例训练数据
│   └── tokenizer.json       # 分词器词汇表
├── src/                      # 源代码目录
│   ├── model/               # 模型实现
│   │   ├── __init__.py
│   │   ├── transformer.py   # Transformer主模型
│   │   ├── attention.py     # 注意力机制
│   │   ├── layers.py        # 基础层实现
│   │   └── embedding.py     # 嵌入层
│   ├── data/                # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py       # 数据集类
│   │   └── tokenizer.py     # 分词器
│   ├── training/            # 训练相关
│   │   ├── __init__.py
│   │   └── trainer.py       # 训练器
│   └── inference/           # 推理相关
│       ├── __init__.py
│       └── generator.py     # 文本生成器
├── examples/                # 示例代码
│   ├── train_example.py     # 训练示例
│   └── inference_example.py # 推理示例
├── checkpoints/             # 模型检查点
└── outputs/                 # 输出文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
python examples/train_example.py
```

训练过程会：
- 加载示例数据并创建分词器
- 初始化Transformer模型
- 进行10个epoch的训练
- 保存模型检查点到 `checkpoints/` 目录
- 在训练过程中生成示例文本

### 2. 推理示例

```bash
python examples/inference_example.py
```

推理示例会：
- 加载训练好的模型
- 展示基本文本生成
- 演示不同温度参数的效果
- 进行批量生成
- 保存生成结果到 `outputs/` 目录

### 3. 交互式生成

```bash
python -m src.inference.generator --checkpoint checkpoints/checkpoint_epoch_10_best.pt --interactive
```

## 模型配置

模型配置文件位于 `config/model_config.yaml`，包含以下参数：

```yaml
model:
  vocab_size: 1000        # 词汇表大小
  d_model: 512           # 模型维度
  n_heads: 8             # 注意力头数
  n_layers: 6            # Transformer层数
  d_ff: 2048             # 前馈网络维度
  max_seq_len: 512       # 最大序列长度
  dropout: 0.1           # Dropout率

training:
  batch_size: 32         # 批次大小
  learning_rate: 0.0001  # 学习率
  num_epochs: 10         # 训练轮数
  warmup_steps: 1000     # 预热步数
  save_every: 2          # 保存间隔
```

## 核心组件说明

### 1. Multi-Head Attention
- 实现了标准的缩放点积注意力
- 支持因果掩码（用于语言建模）
- 可配置的注意力头数

### 2. Position Encoding
- 使用正弦和余弦函数的位置编码
- 支持任意长度的序列

### 3. Transformer Block
- 包含自注意力层和前馈网络
- 使用残差连接和层归一化
- 支持dropout正则化

### 4. 文本生成
- 支持多种采样策略（温度采样、top-k、top-p）
- 可配置的重复惩罚
- 批量生成支持

## 训练数据

项目包含一个示例训练文件 `data/sample_text.txt`，包含中文技术文本。你可以：

1. 替换为自己的训练数据
2. 修改 `src/data/dataset.py` 中的数据加载逻辑
3. 调整分词器以适应不同的语言或领域

## 性能优化建议

1. **GPU加速**：确保安装了CUDA版本的PyTorch
2. **批次大小**：根据GPU内存调整batch_size
3. **序列长度**：较短的序列可以提高训练速度
4. **模型大小**：减少层数或维度可以加快训练

## 扩展功能

本实现可以作为基础，扩展以下功能：

1. **更复杂的分词器**：使用BPE或SentencePiece
2. **预训练模型**：实现BERT风格的预训练
3. **多任务学习**：添加分类、问答等任务
4. **模型压缩**：实现知识蒸馏或量化
5. **分布式训练**：支持多GPU训练

## 常见问题

### Q: 生成的文本质量不高怎么办？
A: 这是正常的，因为：
- 训练数据量较小
- 模型相对简单
- 训练时间较短
可以通过增加训练数据、调整模型参数、延长训练时间来改善。

### Q: 如何使用自己的数据？
A: 
1. 将文本数据放入 `data/` 目录
2. 修改 `examples/train_example.py` 中的数据路径
3. 根据需要调整分词器配置

### Q: 如何调整模型大小？
A: 修改 `config/model_config.yaml` 中的参数：
- `d_model`: 模型维度
- `n_layers`: 层数
- `n_heads`: 注意力头数
- `d_ff`: 前馈网络维度

## 许可证

本项目仅用于学习和研究目的。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)