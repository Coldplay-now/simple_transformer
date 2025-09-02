"""完整的Transformer模型实现"""

import torch
import torch.nn as nn
from typing import Optional

from .embedding import TransformerEmbedding
from .layers import TransformerBlock, OutputProjection


class SimpleTransformer(nn.Module):
    """简单的Transformer语言模型
    
    仅包含Decoder部分，用于文本生成任务
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: Transformer层数
            d_ff: 前馈网络隐藏层维度
            max_seq_len: 最大序列长度
            dropout: Dropout率
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # 嵌入层
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Transformer块堆叠
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # 最终层归一化
        from .layers import LayerNorm
        self.final_norm = LayerNorm(d_model)
        
        # 输出投影层
        self.output_projection = OutputProjection(d_model, vocab_size)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len), 可选
            
        Returns:
            logits: 输出logits, shape: (batch_size, seq_len, vocab_size)
        """
        # 检查序列长度
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_seq_len}")
            
        # 嵌入层
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # 通过所有Transformer块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
            
        # 最终层归一化
        x = self.final_norm(x)
        
        # 输出投影
        logits = self.output_projection(x)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        文本生成
        
        Args:
            input_ids: 输入token IDs, shape: (batch_size, seq_len)
            max_length: 生成的最大长度
            temperature: 采样温度
            top_k: Top-k采样
            top_p: Top-p采样
            pad_token_id: 填充token ID
            
        Returns:
            生成的token IDs, shape: (batch_size, max_length)
        """
        self.eval()
        
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        
        # 如果输入已经达到最大长度，直接返回
        if current_length >= max_length:
            return input_ids[:, :max_length]
            
        # 生成序列
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # 获取当前序列的logits
                logits = self.forward(generated)  # (batch_size, seq_len, vocab_size)
                
                # 只取最后一个位置的logits
                next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
                
                # 应用温度
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                    
                # Top-k采样
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                # Top-p采样
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 找到累积概率超过top_p的位置
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # 将不需要的logits设为负无穷
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                
                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
                
                # 检查是否超过最大序列长度
                if generated.size(1) >= self.max_seq_len:
                    break
                    
        return generated
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'max_seq_len': self.max_seq_len,
            'num_params': self.get_num_params()
        }


def create_model_from_config(config: dict) -> SimpleTransformer:
    """从配置创建模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        SimpleTransformer模型实例
    """
    model_config = config.get('model', {})
    
    return SimpleTransformer(
        vocab_size=model_config.get('vocab_size', 10000),
        d_model=model_config.get('d_model', 256),
        n_heads=model_config.get('n_heads', 8),
        n_layers=model_config.get('n_layers', 6),
        d_ff=model_config.get('d_ff', 1024),
        max_seq_len=model_config.get('max_seq_len', 512),
        dropout=model_config.get('dropout', 0.1)
    )