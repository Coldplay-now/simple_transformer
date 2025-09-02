"""文本生成器"""

import os
import yaml
import torch
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..model.transformer import SimpleTransformer, create_model_from_config
from ..data.tokenizer import SimpleTokenizer


class TextGenerator:
    """文本生成器"""
    
    def __init__(
        self,
        model: SimpleTransformer,
        tokenizer: SimpleTokenizer,
        device: Optional[torch.device] = None
    ):
        """
        初始化文本生成器
        
        Args:
            model: 训练好的Transformer模型
            tokenizer: 分词器
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数，控制随机性
            top_k: top-k采样
            top_p: nucleus采样
            repetition_penalty: 重复惩罚
            do_sample: 是否使用采样
            num_return_sequences: 返回序列数量
            
        Returns:
            生成的文本列表
        """
        results = []
        
        with torch.no_grad():
            for _ in range(num_return_sequences):
                # 将prompt转换为input_ids
                input_ids = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([input_ids], device=self.device)
                
                # 生成token序列
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # 将生成的token转换为文本
                generated_text = self.tokenizer.decode(generated_ids[0].tolist())
                results.append(generated_text)
        
        return results
    
    def interactive_generate(self):
        """交互式文本生成"""
        print("=" * 50)
        print("Simple Transformer 文本生成器")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 50)
        
        while True:
            try:
                # 获取用户输入
                prompt = input("\n请输入提示文本: ").strip()
                
                if prompt.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if not prompt:
                    print("请输入有效的提示文本")
                    continue
                
                # 获取生成参数
                try:
                    max_length = int(input("最大长度 (默认100): ") or "100")
                    temperature = float(input("温度参数 (默认0.8): ") or "0.8")
                    top_k = int(input("Top-k (默认50): ") or "50")
                except ValueError:
                    print("参数格式错误，使用默认值")
                    max_length = 100
                    temperature = 0.8
                    top_k = 50
                
                # 生成文本
                print("\n生成中...")
                generated_texts = self.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    num_return_sequences=1
                )
                
                # 显示结果
                print("\n" + "-" * 30)
                print(f"输入: {prompt}")
                print(f"输出: {generated_texts[0]}")
                print("-" * 30)
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"生成过程中发生错误: {e}")
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[List[str]]:
        """批量生成文本
        
        Args:
            prompts: 提示文本列表
            **kwargs: 生成参数
            
        Returns:
            生成结果列表
        """
        results = []
        
        for prompt in prompts:
            generated = self.generate(prompt, **kwargs)
            results.append(generated)
        
        return results
    
    def save_generated_text(
        self,
        generated_texts: List[str],
        output_file: str,
        prompts: Optional[List[str]] = None
    ):
        """保存生成的文本
        
        Args:
            generated_texts: 生成的文本列表
            output_file: 输出文件路径
            prompts: 对应的提示文本列表
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, text in enumerate(generated_texts):
                if prompts and i < len(prompts):
                    f.write(f"提示: {prompts[i]}\n")
                f.write(f"生成: {text}\n")
                f.write("-" * 50 + "\n")
        
        print(f"生成的文本已保存到: {output_file}")


def load_model_and_tokenizer(
    checkpoint_path: str,
    config_path: Optional[str] = None
) -> tuple[SimpleTransformer, SimpleTokenizer]:
    """加载模型和分词器
    
    Args:
        checkpoint_path: 模型检查点路径
        config_path: 配置文件路径
        
    Returns:
        (model, tokenizer)
    """
    # 加载检查点
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint['config']
    
    # 创建模型
    model = create_model_from_config(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建分词器
    tokenizer = SimpleTokenizer()
    if 'tokenizer_vocab' in checkpoint:
        tokenizer.vocab = checkpoint['tokenizer_vocab']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    else:
        # 如果检查点中没有分词器，尝试从数据目录加载
        tokenizer_path = Path(checkpoint_path).parent.parent / 'data' / 'tokenizer.json'
        if tokenizer_path.exists():
            tokenizer.load(str(tokenizer_path))
        else:
            raise FileNotFoundError("无法找到分词器文件")
    
    print(f"模型已加载: {checkpoint_path}")
    print(f"模型参数量: {model.get_num_params():,}")
    print(f"词汇表大小: {tokenizer.get_vocab_size()}")
    
    return model, tokenizer


def create_generator_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> TextGenerator:
    """从检查点创建文本生成器
    
    Args:
        checkpoint_path: 检查点路径
        config_path: 配置文件路径
        device: 设备
        
    Returns:
        文本生成器
    """
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, config_path)
    return TextGenerator(model, tokenizer, device)


def demo_generation(
    checkpoint_path: str,
    config_path: Optional[str] = None
):
    """演示文本生成
    
    Args:
        checkpoint_path: 检查点路径
        config_path: 配置文件路径
    """
    try:
        # 创建生成器
        generator = create_generator_from_checkpoint(checkpoint_path, config_path)
        
        # 示例提示
        demo_prompts = [
            "人工智能",
            "深度学习",
            "Transformer",
            "机器学习是",
            "自然语言处理"
        ]
        
        print("\n" + "="*50)
        print("文本生成演示")
        print("="*50)
        
        for prompt in demo_prompts:
            print(f"\n提示: {prompt}")
            
            # 生成多个版本
            generated_texts = generator.generate(
                prompt=prompt,
                max_length=80,
                temperature=0.8,
                top_k=50,
                num_return_sequences=2
            )
            
            for i, text in enumerate(generated_texts, 1):
                print(f"生成{i}: {text}")
        
        print("\n" + "="*50)
        print("演示完成")
        print("="*50)
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='文本生成器')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--interactive', action='store_true', help='交互式生成')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_generation(args.checkpoint, args.config)
    elif args.interactive:
        generator = create_generator_from_checkpoint(args.checkpoint, args.config)
        generator.interactive_generate()
    else:
        print("请指定 --interactive 或 --demo 参数")