"""数据预处理工具"""

import os
import re
import requests
from typing import List, Optional
from .tokenizer import SimpleTokenizer, create_sample_tokenizer
from .dataset import split_texts, save_texts_to_file


def clean_text(text: str) -> str:
    """清理文本
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留基本标点）
    text = re.sub(r'[^\w\s.!?,:;"\'-]', '', text)
    
    # 移除首尾空白
    text = text.strip()
    
    return text


def prepare_sample_data() -> List[str]:
    """准备示例数据
    
    Returns:
        示例文本列表
    """
    sample_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "深度学习是机器学习的一个子领域，它基于人工神经网络的研究，特别是利用多层次的神经网络来进行学习和模式识别。",
        "Transformer是一种深度学习模型，主要用于自然语言处理任务。它完全基于注意力机制，摒弃了循环神经网络和卷积神经网络。",
        "自然语言处理是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言。",
        "机器学习是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。",
        "神经网络是一种模仿生物神经网络的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。",
        "计算机视觉是一门研究如何使机器看的科学，更进一步的说，就是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉。",
        "强化学习是机器学习中的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。",
        "大数据是指无法在一定时间范围内用常规软件工具进行捕捉、管理和处理的数据集合。",
        "云计算是基于互联网的相关服务的增加、使用和交付模式，通常涉及通过互联网来提供动态易扩展且经常是虚拟化的资源。",
        "区块链是一个分布式数据库，即一串使用密码学方法相关联产生的数据块。",
        "物联网是互联网、传统电信网等信息承载体，让所有能行使独立功能的普通物体实现互联互通的网络。",
        "量子计算是一种遵循量子力学规律调控量子信息单元进行计算的新型计算模式。",
        "边缘计算是指在靠近物或数据源头的一侧，采用网络、计算、存储、应用核心能力为一体的开放平台。",
        "5G是第五代移动通信技术，是具有高速率、低时延和大连接特点的新一代宽带移动通信技术。",
        "虚拟现实技术是一种可以创建和体验虚拟世界的计算机仿真技术。",
        "增强现实技术是一种将虚拟信息与真实世界巧妙融合的技术。",
        "数字孪生是充分利用物理模型、传感器更新、运行历史等数据，集成多学科、多物理量、多尺度、多概率的仿真过程。",
        "人机交互是研究人和计算机之间的信息交换的学科。",
        "软件工程是一门研究用工程化方法构建和维护有效的、实用的和高质量的软件的学科。"
    ]
    
    return sample_texts


def download_sample_text(url: str, save_path: str) -> bool:
    """下载示例文本文件
    
    Args:
        url: 下载链接
        save_path: 保存路径
        
    Returns:
        是否下载成功
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        print(f"文本文件已下载到: {save_path}")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def prepare_training_data(
    data_dir: str = "data",
    vocab_size: int = 1000,
    train_ratio: float = 0.8
) -> tuple:
    """准备训练数据
    
    Args:
        data_dir: 数据目录
        vocab_size: 词汇表大小
        train_ratio: 训练集比例
        
    Returns:
        (tokenizer, train_texts, val_texts)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # 获取示例数据
    sample_texts = prepare_sample_data()
    
    # 清理文本
    cleaned_texts = [clean_text(text) for text in sample_texts if clean_text(text)]
    
    # 分割训练和验证集
    train_texts, val_texts = split_texts(cleaned_texts, train_ratio)
    
    # 保存到文件
    train_file = os.path.join(data_dir, "train.txt")
    val_file = os.path.join(data_dir, "val.txt")
    
    save_texts_to_file(train_texts, train_file)
    save_texts_to_file(val_texts, val_file)
    
    # 创建分词器
    print("创建分词器...")
    tokenizer = create_sample_tokenizer(train_texts, vocab_size)
    
    # 保存分词器
    tokenizer_file = os.path.join(data_dir, "tokenizer.json")
    tokenizer.save(tokenizer_file)
    
    print(f"数据准备完成:")
    print(f"  训练样本: {len(train_texts)}")
    print(f"  验证样本: {len(val_texts)}")
    print(f"  词汇表大小: {tokenizer.get_vocab_size()}")
    
    return tokenizer, train_texts, val_texts


def load_prepared_data(data_dir: str = "data") -> tuple:
    """加载已准备的数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        (tokenizer, train_texts, val_texts)
    """
    tokenizer_file = os.path.join(data_dir, "tokenizer.json")
    train_file = os.path.join(data_dir, "train.txt")
    val_file = os.path.join(data_dir, "val.txt")
    
    # 检查文件是否存在
    if not all(os.path.exists(f) for f in [tokenizer_file, train_file, val_file]):
        raise FileNotFoundError("数据文件不完整，请先运行 prepare_training_data()")
    
    # 加载分词器
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_file)
    
    # 加载文本
    with open(train_file, 'r', encoding='utf-8') as f:
        train_texts = [line.strip() for line in f if line.strip()]
        
    with open(val_file, 'r', encoding='utf-8') as f:
        val_texts = [line.strip() for line in f if line.strip()]
    
    print(f"数据加载完成:")
    print(f"  训练样本: {len(train_texts)}")
    print(f"  验证样本: {len(val_texts)}")
    print(f"  词汇表大小: {tokenizer.get_vocab_size()}")
    
    return tokenizer, train_texts, val_texts


def create_sample_text_file(file_path: str):
    """创建示例文本文件
    
    Args:
        file_path: 文件保存路径
    """
    sample_texts = prepare_sample_data()
    
    # 扩展数据（重复几次以增加训练数据量）
    extended_texts = sample_texts * 5
    
    # 添加一些变化
    import random
    random.shuffle(extended_texts)
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for text in extended_texts:
            f.write(text + '\n')
            
    print(f"示例文本文件已创建: {file_path}")
    print(f"包含 {len(extended_texts)} 行文本")


if __name__ == "__main__":
    # 示例用法
    print("准备训练数据...")
    tokenizer, train_texts, val_texts = prepare_training_data()
    
    # 测试分词器
    test_text = "这是一个测试文本。"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\n测试分词器:")
    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")