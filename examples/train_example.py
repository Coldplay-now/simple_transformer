"""训练示例脚本"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import prepare_training_data
from src.data.dataset import create_data_loaders_from_files
from src.model.transformer import create_model_from_config
from src.training.trainer import Trainer
import yaml
import torch


def main():
    """训练示例"""
    print("Simple Transformer 训练示例")
    print("=" * 40)
    
    # 配置
    config = {
        'model': {
            'vocab_size': 1000,  # 将在数据准备后更新
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1024,
            'max_seq_length': 128,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'max_epochs': 10,
            'grad_clip_norm': 1.0,
            'num_workers': 2
        },
        'logging': {
            'use_tensorboard': True,
            'log_dir': 'logs',
            'log_interval': 10
        },
        'checkpointing': {
            'checkpoint_dir': 'checkpoints',
            'save_interval': 2
        }
    }
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("\n1. 准备训练数据...")
    data_dir = "data"
    tokenizer, train_texts, val_texts = prepare_training_data(
        data_dir=data_dir,
        vocab_size=config['model']['vocab_size'],
        train_ratio=0.8
    )
    
    # 更新词汇表大小
    config['model']['vocab_size'] = tokenizer.get_vocab_size()
    print(f"实际词汇表大小: {config['model']['vocab_size']}")
    
    # 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader, val_loader = create_data_loaders_from_files(
        train_file=os.path.join(data_dir, 'train.txt'),
        val_file=os.path.join(data_dir, 'val.txt'),
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_seq_length'],
        num_workers=config['training']['num_workers']
    )
    
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    
    # 创建模型
    print("\n3. 创建模型...")
    model = create_model_from_config(config['model'])
    print(f"模型参数量: {model.get_num_params():,}")
    
    # 创建训练器
    print("\n4. 创建训练器...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
    
    # 开始训练
    print("\n5. 开始训练...")
    print("=" * 40)
    
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        print("\n训练完成！")
        
        # 生成示例文本
        print("\n6. 生成示例文本...")
        test_prompts = ["人工智能", "深度学习", "机器学习"]
        
        for prompt in test_prompts:
            generated = trainer.generate_sample(
                prompt=prompt,
                max_length=50,
                temperature=0.8
            )
            print(f"输入: {prompt}")
            print(f"输出: {generated}")
            print()
        
    except KeyboardInterrupt:
        print("\n训练被中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()