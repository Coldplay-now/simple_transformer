"""Embedding层实现"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Token嵌入层
    
    将token ID转换为dense vector表示
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 嵌入维度
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token IDs, shape: (batch_size, seq_len)
            
        Returns:
            嵌入向量, shape: (batch_size, seq_len, d_model)
        """
        # 按照Transformer论文，嵌入需要乘以sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """位置编码层
    
    为序列中的每个位置添加位置信息
    使用sin/cos函数生成位置编码
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        Args:
            d_model: 嵌入维度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # 计算div_term用于sin/cos函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度并注册为buffer（不参与梯度更新）
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入嵌入, shape: (batch_size, seq_len, d_model)
            
        Returns:
            添加位置编码后的嵌入, shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 添加位置编码
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x


class TransformerEmbedding(nn.Module):
    """完整的Transformer嵌入层
    
    结合Token嵌入和位置编码
    """
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 嵌入维度
            max_seq_len: 最大序列长度
            dropout: Dropout率
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token IDs, shape: (batch_size, seq_len)
            
        Returns:
            完整嵌入, shape: (batch_size, seq_len, d_model)
        """
        # Token嵌入
        token_emb = self.token_embedding(x)
        
        # 添加位置编码
        pos_emb = self.positional_encoding(token_emb)
        
        # 应用dropout
        return self.dropout(pos_emb)