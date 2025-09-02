"""Transformer基础层实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """前馈神经网络
    
    Transformer中的Position-wise Feed-Forward Network
    结构：Linear -> ReLU -> Dropout -> Linear
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model)
            
        Returns:
            前馈网络输出, shape: (batch_size, seq_len, d_model)
        """
        # 第一个线性层 + ReLU激活
        x = F.relu(self.linear1(x))
        
        # Dropout
        x = self.dropout(x)
        
        # 第二个线性层
        x = self.linear2(x)
        
        return x


class LayerNorm(nn.Module):
    """层归一化
    
    对最后一个维度进行归一化
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: 模型维度
            eps: 数值稳定性参数
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量, shape: (..., d_model)
            
        Returns:
            归一化后的张量, shape: (..., d_model)
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualConnection(nn.Module):
    """残差连接
    
    实现残差连接和层归一化：LayerNorm(x + Sublayer(x))
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            dropout: Dropout率
        """
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Args:
            x: 输入张量
            sublayer: 子层（如注意力层或前馈网络）
            
        Returns:
            残差连接输出
        """
        # Pre-norm: LayerNorm -> Sublayer -> Dropout -> Residual
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """Transformer块
    
    包含自注意力层和前馈网络，以及相应的残差连接
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            max_seq_len: 最大序列长度
            dropout: Dropout率
        """
        super().__init__()
        
        # 导入注意力模块
        from .attention import CausalSelfAttention
        
        self.self_attention = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 残差连接
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len), 可选
            
        Returns:
            Transformer块输出, shape: (batch_size, seq_len, d_model)
        """
        # 自注意力 + 残差连接
        x = self.residual1(x, lambda x: self.self_attention(x, attention_mask))
        
        # 前馈网络 + 残差连接
        x = self.residual2(x, self.feed_forward)
        
        return x


class OutputProjection(nn.Module):
    """输出投影层
    
    将模型输出投影到词汇表大小的logits
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model: 模型维度
            vocab_size: 词汇表大小
        """
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量, shape: (batch_size, seq_len, d_model)
            
        Returns:
            logits, shape: (batch_size, seq_len, vocab_size)
        """
        return self.linear(x)