"""数据集和数据加载器实现"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import os
import random

from .tokenizer import SimpleTokenizer


class TextDataset(Dataset):
    """文本数据集
    
    用于语言模型训练的数据集
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: SimpleTokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        """
        Args:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 最大序列长度
            stride: 滑动窗口步长
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # 预处理所有文本
        self.examples = self._prepare_examples(texts)
        
    def _prepare_examples(self, texts: List[str]) -> List[List[int]]:
        """准备训练样本
        
        Args:
            texts: 原始文本列表
            
        Returns:
            token序列列表
        """
        examples = []
        
        for text in texts:
            # 编码文本
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            
            # 如果文本太短，直接添加
            if len(token_ids) <= self.max_length:
                # 填充到max_length
                padded_ids = token_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(token_ids))
                examples.append(padded_ids)
            else:
                # 使用滑动窗口切分长文本
                for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
                    chunk = token_ids[i:i + self.max_length]
                    examples.append(chunk)
                    
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含input_ids、attention_mask和labels的字典
        """
        token_ids = self.examples[idx]
        
        # 创建attention mask (1表示真实token，0表示padding)
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in token_ids]
        
        # 转换为tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # 对于语言模型，labels就是input_ids
        }


class TextFileDataset(Dataset):
    """从文件加载的文本数据集"""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: SimpleTokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        """
        Args:
            file_path: 文本文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            stride: 滑动窗口步长
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
            
        # 初始化基础数据集
        self.base_dataset = TextDataset(texts, tokenizer, max_length, stride)
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.base_dataset[idx]


def create_data_loaders(
    train_texts: List[str],
    val_texts: Optional[List[str]],
    tokenizer: SimpleTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    stride: int = 256,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """创建训练和验证数据加载器
    
    Args:
        train_texts: 训练文本列表
        val_texts: 验证文本列表
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        stride: 滑动窗口步长
        num_workers: 数据加载进程数
        
    Returns:
        训练和验证数据加载器
    """
    # 创建训练数据集
    train_dataset = TextDataset(train_texts, tokenizer, max_length, stride)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # 创建验证数据集
    val_loader = None
    if val_texts:
        val_dataset = TextDataset(val_texts, tokenizer, max_length, stride)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
    return train_loader, val_loader


def create_data_loaders_from_files(
    train_file: str,
    val_file: Optional[str],
    tokenizer: SimpleTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    stride: int = 256,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """从文件创建数据加载器
    
    Args:
        train_file: 训练文件路径
        val_file: 验证文件路径
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        stride: 滑动窗口步长
        num_workers: 数据加载进程数
        
    Returns:
        训练和验证数据加载器
    """
    # 创建训练数据集
    train_dataset = TextFileDataset(train_file, tokenizer, max_length, stride)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # 创建验证数据集
    val_loader = None
    if val_file and os.path.exists(val_file):
        val_dataset = TextFileDataset(val_file, tokenizer, max_length, stride)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
    return train_loader, val_loader


def split_texts(texts: List[str], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[str], List[str]]:
    """分割文本为训练集和验证集
    
    Args:
        texts: 文本列表
        train_ratio: 训练集比例
        seed: 随机种子
        
    Returns:
        训练文本和验证文本
    """
    random.seed(seed)
    shuffled_texts = texts.copy()
    random.shuffle(shuffled_texts)
    
    split_idx = int(len(shuffled_texts) * train_ratio)
    train_texts = shuffled_texts[:split_idx]
    val_texts = shuffled_texts[split_idx:]
    
    return train_texts, val_texts


def load_text_file(file_path: str) -> List[str]:
    """加载文本文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        文本行列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
        
    return texts


def save_texts_to_file(texts: List[str], file_path: str):
    """保存文本到文件
    
    Args:
        texts: 文本列表
        file_path: 保存路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
            
    print(f"已保存 {len(texts)} 行文本到 {file_path}")