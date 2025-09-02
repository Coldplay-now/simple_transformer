"""训练器模块"""

import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from ..model.transformer import SimpleTransformer
from ..data.tokenizer import SimpleTokenizer
from ..inference.generator import TextGenerator


class Trainer:
    """Transformer训练器"""
    
    def __init__(
        self,
        model: SimpleTransformer,
        tokenizer: SimpleTokenizer,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        初始化训练器
        
        Args:
            model: Transformer模型
            tokenizer: 分词器
            config: 配置字典
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['max_epochs'],
            eta_min=config['training']['learning_rate'] * 0.1
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 日志记录
        self.writer = None
        if config['logging']['use_tensorboard']:
            log_dir = config['logging']['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.config['training']['max_epochs']}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 创建标签（下一个token预测）
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()
            
            # 前向传播
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            
            # 计算损失
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip_norm']
            )
            
            # 优化器步骤
            self.optimizer.step()
            
            # 统计
            batch_loss = loss.item()
            batch_tokens = (labels != self.tokenizer.pad_token_id).sum().item()
            
            total_loss += batch_loss
            total_tokens += batch_tokens
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'ppl': f'{torch.exp(loss):.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录日志
            if self.writer and self.global_step % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', batch_loss, self.global_step)
                self.writer.add_scalar('train/perplexity', torch.exp(loss).item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        avg_loss = total_loss / len(train_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens
        }
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                # 数据移到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 创建标签
                labels = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                attention_mask = attention_mask[:, :-1].contiguous()
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask)
                
                # 计算损失
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
                
                # 统计
                batch_tokens = (labels != self.tokenizer.pad_token_id).sum().item()
                total_loss += loss.item()
                total_tokens += batch_tokens
        
        avg_loss = total_loss / len(val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens
        }
    
    def save_checkpoint(
        self,
        filepath: str,
        is_best: bool = False
    ):
        """保存检查点
        
        Args:
            filepath: 保存路径
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'tokenizer_vocab': self.tokenizer.vocab,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")
    
    def load_checkpoint(
        self,
        filepath: str
    ) -> Dict:
        """加载检查点
        
        Args:
            filepath: 检查点路径
            
        Returns:
            检查点信息
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"检查点已加载: {filepath}")
        print(f"Epoch: {self.current_epoch}, Step: {self.global_step}")
        
        return checkpoint
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            resume_from: 恢复训练的检查点路径
        """
        # 恢复训练
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        print(f"开始训练...")
        print(f"设备: {self.device}")
        print(f"模型参数量: {self.model.get_num_params():,}")
        
        for epoch in range(start_epoch, self.config['training']['max_epochs']):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step()
            
            # 计算时间
            epoch_time = time.time() - start_time
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{self.config['training']['max_epochs']}:")
            print(f"  训练损失: {train_metrics['loss']:.4f}, 困惑度: {train_metrics['perplexity']:.2f}")
            if val_metrics:
                print(f"  验证损失: {val_metrics['loss']:.4f}, 困惑度: {val_metrics['perplexity']:.2f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  时间: {epoch_time:.2f}s")
            
            # 记录到tensorboard
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/train_perplexity', train_metrics['perplexity'], epoch)
                if val_metrics:
                    self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('epoch/val_perplexity', val_metrics['perplexity'], epoch)
            
            # 保存检查点
            if (epoch + 1) % self.config['checkpointing']['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.config['checkpointing']['checkpoint_dir'],
                    f"checkpoint_epoch_{epoch+1}.pt"
                )
                
                is_best = False
                if val_metrics and val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    is_best = True
                
                self.save_checkpoint(checkpoint_path, is_best)
        
        print("\n训练完成！")
        
        # 关闭tensorboard writer
        if self.writer:
            self.writer.close()
    
    def generate_sample(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> str:
        """生成文本样本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: top-k采样
            
        Returns:
            生成的文本
        """
        # 创建文本生成器
        generator = TextGenerator(self.model, self.tokenizer, self.device)
        
        # 生成文本
        generated_texts = generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            num_return_sequences=1
        )
        
        return generated_texts[0] if generated_texts else prompt


def create_trainer_from_config(
    config_path: str,
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer
) -> Trainer:
    """从配置文件创建训练器
    
    Args:
        config_path: 配置文件路径
        model: 模型
        tokenizer: 分词器
        
    Returns:
        训练器实例
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return Trainer(model, tokenizer, config)