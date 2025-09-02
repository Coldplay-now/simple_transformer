"""注意力机制实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制
    
    实现Transformer中的Multi-Head Attention
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 线性变换层：Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: torch.Tensor = None) -> tuple:
        """缩放点积注意力
        
        Args:
            Q: Query矩阵, shape: (batch_size, n_heads, seq_len, d_k)
            K: Key矩阵, shape: (batch_size, n_heads, seq_len, d_k)
            V: Value矩阵, shape: (batch_size, n_heads, seq_len, d_k)
            mask: 注意力掩码, shape: (batch_size, 1, seq_len, seq_len)
            
        Returns:
            注意力输出和注意力权重
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（用于防止看到未来信息）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算注意力输出
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: Query输入, shape: (batch_size, seq_len, d_model)
            key: Key输入, shape: (batch_size, seq_len, d_model)
            value: Value输入, shape: (batch_size, seq_len, d_model)
            mask: 注意力掩码, shape: (batch_size, 1, seq_len, seq_len)
            
        Returns:
            多头注意力输出, shape: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. 线性变换得到Q, K, V
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)    # (batch_size, seq_len, d_model)
        V = self.w_v(value)  # (batch_size, seq_len, d_model)
        
        # 2. 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # 现在形状为: (batch_size, n_heads, seq_len, d_k)
        
        # 3. 计算缩放点积注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 连接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 5. 最终线性变换
        output = self.w_o(attention_output)
        
        return output


class CausalSelfAttention(MultiHeadAttention):
    """因果自注意力
    
    用于语言模型的自注意力，包含因果掩码防止看到未来信息
    """
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            max_seq_len: 最大序列长度
            dropout: Dropout率
        """
        super().__init__(d_model, n_heads, dropout)
        self.max_seq_len = max_seq_len
        
        # 注册因果掩码
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            )
        )
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入序列, shape: (batch_size, seq_len, d_model)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len), 可选
            
        Returns:
            自注意力输出, shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # 获取当前序列长度的因果掩码
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # 如果提供了attention_mask，将其与因果掩码结合
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 结合因果掩码和padding掩码
            mask = mask * attention_mask
        
        # 自注意力：query, key, value都是同一个输入
        return super().forward(x, x, x, mask)