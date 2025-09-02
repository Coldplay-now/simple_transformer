"""推理示例脚本"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.generator import create_generator_from_checkpoint, demo_generation


def main():
    """推理示例"""
    print("Simple Transformer 推理示例")
    print("=" * 40)
    
    # 检查点路径（需要先训练模型）
    checkpoint_dir = "checkpoints"
    
    # 查找最新的检查点
    checkpoint_files = []
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pt'):
                checkpoint_files.append(os.path.join(checkpoint_dir, file))
    
    if not checkpoint_files:
        print("错误: 未找到训练好的模型检查点")
        print("请先运行训练脚本生成模型")
        print("示例: python examples/train_example.py")
        return
    
    # 使用最新的检查点
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"使用检查点: {latest_checkpoint}")
    
    try:
        # 创建生成器
        print("\n1. 加载模型...")
        generator = create_generator_from_checkpoint(latest_checkpoint)
        
        # 示例1: 基本文本生成
        print("\n2. 基本文本生成示例...")
        print("-" * 30)
        
        prompts = [
            "人工智能",
            "深度学习",
            "Transformer模型",
            "机器学习算法"
        ]
        
        for prompt in prompts:
            generated = generator.generate(
                prompt=prompt,
                max_length=60,
                temperature=0.8,
                top_k=50,
                num_return_sequences=1
            )
            print(f"输入: {prompt}")
            print(f"输出: {generated[0]}")
            print()
        
        # 示例2: 不同温度参数的效果
        print("\n3. 不同温度参数的效果...")
        print("-" * 30)
        
        test_prompt = "自然语言处理"
        temperatures = [0.5, 0.8, 1.0, 1.2]
        
        for temp in temperatures:
            generated = generator.generate(
                prompt=test_prompt,
                max_length=50,
                temperature=temp,
                top_k=50,
                num_return_sequences=1
            )
            print(f"温度 {temp}: {generated[0]}")
        
        # 示例3: 批量生成
        print("\n4. 批量生成示例...")
        print("-" * 30)
        
        batch_prompts = ["计算机视觉", "强化学习", "神经网络"]
        batch_results = generator.batch_generate(
            prompts=batch_prompts,
            max_length=50,
            temperature=0.8,
            num_return_sequences=2
        )
        
        for i, (prompt, results) in enumerate(zip(batch_prompts, batch_results)):
            print(f"提示 {i+1}: {prompt}")
            for j, result in enumerate(results, 1):
                print(f"  生成 {j}: {result}")
            print()
        
        # 示例4: 保存生成结果
        print("\n5. 保存生成结果...")
        print("-" * 30)
        
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成更多文本
        extended_prompts = [
            "人工智能的发展",
            "深度学习的应用",
            "Transformer架构",
            "机器学习的未来",
            "自然语言处理技术"
        ]
        
        generated_texts = []
        for prompt in extended_prompts:
            result = generator.generate(
                prompt=prompt,
                max_length=80,
                temperature=0.8,
                num_return_sequences=1
            )
            generated_texts.append(result[0])
        
        # 保存到文件
        output_file = os.path.join(output_dir, "generated_samples.txt")
        generator.save_generated_text(
            generated_texts=generated_texts,
            output_file=output_file,
            prompts=extended_prompts
        )
        
        print("\n" + "="*40)
        print("推理示例完成！")
        print("="*40)
        
        # 提示交互式模式
        print("\n提示: 你可以运行以下命令进入交互式生成模式:")
        print(f"python -m src.inference.generator --checkpoint {latest_checkpoint} --interactive")
        
    except Exception as e:
        print(f"\n推理过程中发生错误: {e}")
        raise


def interactive_demo():
    """交互式演示"""
    checkpoint_dir = "checkpoints"
    
    # 查找检查点
    checkpoint_files = []
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pt'):
                checkpoint_files.append(os.path.join(checkpoint_dir, file))
    
    if not checkpoint_files:
        print("错误: 未找到训练好的模型检查点")
        return
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    
    try:
        generator = create_generator_from_checkpoint(latest_checkpoint)
        generator.interactive_generate()
    except Exception as e:
        print(f"交互式演示中发生错误: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='推理示例')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    else:
        main()