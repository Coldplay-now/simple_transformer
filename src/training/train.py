"""训练脚本"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.model.transformer import create_model_from_config
from src.data.preprocessing import prepare_training_data, load_prepared_data
from src.data.dataset import create_data_loaders_from_files
from src.training.trainer import Trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Simple Transformer模型')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='数据目录'
    )
    
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='是否重新准备数据'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='训练设备'
    )
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """获取训练设备"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = get_device(args.device)
    print(f"使用设备: {device}")
    
    # 准备数据
    data_dir = Path(args.data_dir)
    
    if args.prepare_data or not (data_dir / 'tokenizer.json').exists():
        print("准备训练数据...")
        tokenizer, train_texts, val_texts = prepare_training_data(
            data_dir=str(data_dir),
            vocab_size=config['model']['vocab_size'],
            train_ratio=0.8
        )
    else:
        print("加载已准备的数据...")
        tokenizer, train_texts, val_texts = load_prepared_data(str(data_dir))
    
    # 更新配置中的词汇表大小
    config['model']['vocab_size'] = tokenizer.get_vocab_size()
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader = create_data_loaders_from_files(
        train_file=str(data_dir / 'train.txt'),
        val_file=str(data_dir / 'val.txt'),
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_seq_length'],
        num_workers=config['training']['num_workers']
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # 创建模型
    print("创建模型...")
    model = create_model_from_config(config['model'])
    model.to(device)
    
    print(f"模型参数量: {model.get_num_parameters():,}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 创建训练器
    print("创建训练器...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
    
    # 开始训练
    print("\n" + "="*50)
    print("开始训练")
    print("="*50)
    
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from=args.resume
        )
        
        print("\n" + "="*50)
        print("训练完成")
        print("="*50)
        
        # 生成一些示例文本
        print("\n生成示例文本:")
        print("-" * 30)
        
        test_prompts = [
            "人工智能",
            "深度学习",
            "Transformer",
            "机器学习"
        ]
        
        for prompt in test_prompts:
            generated = trainer.generate_sample(
                prompt=prompt,
                max_length=50,
                temperature=0.8,
                top_k=50
            )
            print(f"输入: {prompt}")
            print(f"输出: {generated}")
            print()
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        
        # 保存当前状态
        checkpoint_dir = Path(config['checkpointing']['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        interrupted_path = checkpoint_dir / "interrupted_checkpoint.pt"
        trainer.save_checkpoint(str(interrupted_path))
        print(f"当前状态已保存到: {interrupted_path}")
        
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()