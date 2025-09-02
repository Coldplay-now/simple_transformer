# 快速开始指南

本指南将帮助你在5分钟内运行Simple LLM Transformer项目。

## 🚀 一键运行

### 步骤1: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤2: 运行训练示例

```bash
python examples/train_example.py
```

这将：
- 自动准备示例数据
- 创建并训练一个小型Transformer模型
- 生成示例文本

### 步骤3: 测试文本生成

```bash
python examples/inference_example.py
```

## 📋 系统要求

- Python 3.8+
- PyTorch 1.9+
- 4GB+ RAM
- （可选）NVIDIA GPU 或 Apple Silicon Mac

## 🎯 预期结果

训练完成后，你将看到：

```
训练完成！

生成示例文本:
输入: 人工智能
输出: 人工智能是计算机科学的一个分支，它企图了解智能的实质...

输入: 深度学习
输出: 深度学习是机器学习的一个子领域，它基于人工神经网络...
```

## 🔧 自定义配置

编辑 `config/model_config.yaml` 来调整模型参数：

```yaml
model:
  d_model: 128        # 减小模型以加快训练
  n_layers: 4         # 减少层数
  
training:
  max_epochs: 5       # 减少训练轮数
  batch_size: 8       # 适应内存大小
```

## 🎮 交互式体验

```bash
# 交互式文本生成
python examples/inference_example.py --interactive
```

然后输入你想要的提示词，模型会实时生成文本！

## 📊 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir logs

# 在浏览器中打开
open http://localhost:6006
```

## ❓ 遇到问题？

### 内存不足
```bash
# 使用更小的配置
python examples/train_example.py  # 已经使用了较小的默认配置
```

### 训练太慢
```bash
# 检查是否使用了GPU加速
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

### 生成质量不好
- 增加训练时间（修改 `max_epochs`）
- 使用更多训练数据
- 调整生成参数（temperature, top_k）

## 🎉 下一步

1. 阅读完整的 [README.md](README.md)
2. 查看 [PRD.md](PRD.md) 了解项目设计
3. 探索 `src/` 目录下的代码实现
4. 尝试修改模型架构
5. 使用自己的数据训练模型

祝你学习愉快！🎓