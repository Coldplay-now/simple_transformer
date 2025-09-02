"""简单分词器实现"""

import json
import re
from typing import List, Dict, Optional
from collections import Counter


class SimpleTokenizer:
    """简单的字符级分词器
    
    用于将文本转换为token序列，适合学习使用
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        Args:
            vocab_size: 词汇表大小
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        
        # 特殊token
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'  # Begin of sequence
        self.eos_token = '<EOS>'  # End of sequence
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # 初始化特殊token
        self._init_special_tokens()
        
    def _init_special_tokens(self):
        """初始化特殊token"""
        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id
        }
        
        self.inverse_vocab = {
            self.pad_token_id: self.pad_token,
            self.unk_token_id: self.unk_token,
            self.bos_token_id: self.bos_token,
            self.eos_token_id: self.eos_token
        }
        
    def _preprocess_text(self, text: str) -> str:
        """预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 转换为小写
        text = text.lower()
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 在标点符号前后添加空格
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        
        # 移除首尾空白
        text = text.strip()
        
        return text
    
    def build_vocab(self, texts: List[str]):
        """构建词汇表
        
        Args:
            texts: 训练文本列表
        """
        print("构建词汇表...")
        
        # 收集所有字符
        char_counter = Counter()
        
        for text in texts:
            processed_text = self._preprocess_text(text)
            # 按字符分割
            chars = list(processed_text)
            char_counter.update(chars)
            
        # 获取最常见的字符
        most_common_chars = char_counter.most_common(self.vocab_size - len(self.vocab))
        
        # 添加到词汇表
        for char, count in most_common_chars:
            if char not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[char] = token_id
                self.inverse_vocab[token_id] = char
                
        print(f"词汇表构建完成，共 {len(self.vocab)} 个token")
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本为token ID序列
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊token
            
        Returns:
            token ID列表
        """
        processed_text = self._preprocess_text(text)
        
        # 字符级编码
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
            
        for char in processed_text:
            token_id = self.vocab.get(char, self.unk_token_id)
            token_ids.append(token_id)
            
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
            
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码token ID序列为文本
        
        Args:
            token_ids: token ID列表
            skip_special_tokens: 是否跳过特殊token
            
        Returns:
            解码后的文本
        """
        chars = []
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
                
            char = self.inverse_vocab.get(token_id, self.unk_token)
            if not (skip_special_tokens and char == self.unk_token):
                chars.append(char)
                
        return ''.join(chars)
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None, 
                    padding: bool = True, truncation: bool = True) -> Dict[str, List[List[int]]]:
        """批量编码文本
        
        Args:
            texts: 文本列表
            max_length: 最大长度
            padding: 是否填充
            truncation: 是否截断
            
        Returns:
            包含input_ids和attention_mask的字典
        """
        encoded_texts = []
        
        for text in texts:
            token_ids = self.encode(text)
            
            # 截断
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                
            encoded_texts.append(token_ids)
            
        # 填充
        if padding and max_length:
            for i, token_ids in enumerate(encoded_texts):
                if len(token_ids) < max_length:
                    pad_length = max_length - len(token_ids)
                    encoded_texts[i] = token_ids + [self.pad_token_id] * pad_length
                    
        # 创建attention mask
        attention_masks = []
        for token_ids in encoded_texts:
            mask = [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]
            attention_masks.append(mask)
            
        return {
            'input_ids': encoded_texts,
            'attention_mask': attention_masks
        }
    
    def save(self, filepath: str):
        """保存分词器
        
        Args:
            filepath: 保存路径
        """
        tokenizer_data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            
        print(f"分词器已保存到 {filepath}")
        
    def load(self, filepath: str):
        """加载分词器
        
        Args:
            filepath: 加载路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            
        self.vocab = tokenizer_data['vocab']
        self.vocab_size = tokenizer_data['vocab_size']
        
        # 重建inverse_vocab
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        
        # 加载特殊token
        special_tokens = tokenizer_data['special_tokens']
        self.pad_token = special_tokens['pad_token']
        self.unk_token = special_tokens['unk_token']
        self.bos_token = special_tokens['bos_token']
        self.eos_token = special_tokens['eos_token']
        
        # 更新特殊token ID
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        
        print(f"分词器已从 {filepath} 加载")
        
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocab)
    
    def __len__(self) -> int:
        """返回词汇表大小"""
        return len(self.vocab)


def create_sample_tokenizer(sample_texts: List[str], vocab_size: int = 1000) -> SimpleTokenizer:
    """创建示例分词器
    
    Args:
        sample_texts: 示例文本列表
        vocab_size: 词汇表大小
        
    Returns:
        训练好的分词器
    """
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(sample_texts)
    return tokenizer