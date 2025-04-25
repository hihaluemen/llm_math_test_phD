#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用生成的数据集评估大模型API的基础加法能力
支持测试GPT-4、Claude、Gemini等模型
"""

import json
import os
import argparse
import time
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np

# 导入API库
try:
    from openai import OpenAI
except ImportError:
    print("警告: OpenAI库未安装，无法使用API模型。可以通过 pip install openai 安装")

class ChatLLM(object):
    """统一的LLM API客户端封装，支持OpenAI接口格式"""
    def __init__(self, config):
        self.client = OpenAI(
            api_key=config.get('api_key', ''),
            base_url=config.get('base_url', '')
        )
        self.model = config.get('model', 'gpt-4o')
        self.temperature = config.get('temperature', 0.0)

    def generate(self, prompt):
        messages = list()
        message = {"role": "user", "content": prompt}
        messages.append(message)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        answer = response.choices[0].message.content
        return answer


class LLMTester:
    """大模型加法能力测试器"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str = None, temperature: float = 0.0):
        """
        初始化测试器
        
        Args:
            model_name: 模型名称 (如 'gpt-4o', 'claude-3-opus', 'gemini-pro')
            api_key: API密钥
            base_url: API基础URL（可选，用于自定义端点）
            temperature: 采样温度 (默认为0，确定性输出)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.client = self._setup_client()
        
    def _setup_client(self):
        """设置LLM客户端"""
        config = {
            'api_key': self.api_key,
            'model': self.model_name,
            'base_url': self.base_url,
            'temperature': self.temperature
        }
        return ChatLLM(config)
    
    def create_prompt(self, test_case: Dict[str, Any]) -> str:
        """
        根据测试用例创建提示词
        
        Args:
            test_case: 测试用例
            
        Returns:
            格式化的提示词
        """
        if test_case["type"] == "numerical":
            # 数值任务的简单提示词
            return f"问题: {test_case['operand1']} + {test_case['operand2']} = \\boxed{{?}}\n\n请直接给出答案，不要有任何解释。"
        else:
            # 符号任务需要提供映射定义
            mapping_def = eval(test_case["symbolic_mapping_definition"])
            mapping_str = "\n".join([f"'{k}' 映射为 '{v}'" for k, v in mapping_def.items()])
            
            prompt = f"""我们定义了一种符号系统，其中:
{mapping_str}

在这个符号系统中，加法遵循与十进制数字相同的规则。例如，如果 '1' 映射为 'd'，'2' 映射为 'a'，那么 'd + a' 就等于 '3' 对应的符号。

请计算: {test_case['operand1']} + {test_case['operand2']} = \\boxed{{?}}

请直接给出答案，不要有任何解释。"""
            return prompt
    
    def call_api(self, prompt: str) -> str:
        """
        调用LLM API
        
        Args:
            prompt: 提示词
            
        Returns:
            模型的响应
        """
        try:
            return self.client.generate(prompt)
        except Exception as e:
            print(f"API调用错误: {e}")
            return f"ERROR: {str(e)}"
    
    def extract_answer(self, response: str) -> str:
        """
        从模型响应中提取答案
        
        Args:
            response: 模型的响应
            
        Returns:
            提取的答案
        """
        # 尝试从\boxed{}中提取
        import re
        boxed_match = re.search(r'\\boxed{(.*?)}', response)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # 如果没有\boxed{}，尝试直接提取数字或符号序列
        # 首先查找数字序列
        number_match = re.search(r'\b\d+\b', response)
        if number_match:
            return number_match.group(0)
        
        # 如果没有找到数字，尝试提取任何非空白字符序列作为可能的符号答案
        # 这是一个简化的方法，可能需要根据实际响应格式进行调整
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('问题:', '计算:')):
                # 返回第一个非空且不是问题陈述的行
                return line
        
        # 如果都失败了，返回整个响应
        return response.strip()
    
    def evaluate_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个测试用例
        
        Args:
            test_case: 测试用例
            
        Returns:
            评估结果
        """
        prompt = self.create_prompt(test_case)
        response = self.call_api(prompt)
        extracted_answer = self.extract_answer(response)
        
        is_correct = extracted_answer == test_case["correct_answer"]
        
        result = {
            "case_id": test_case["case_id"],
            "pair_id": test_case["pair_id"],
            "type": test_case["type"],
            "order": test_case["order"],
            "digit_length_A": test_case["digit_length_A"],
            "digit_length_B": test_case["digit_length_B"],
            "symbolic_mapping_id": test_case["symbolic_mapping_id"],
            "model_answer": extracted_answer,
            "correct_answer": test_case["correct_answer"],
            "is_correct": is_correct,
            "full_response": response
        }
        
        return result

def run_evaluation(
        dataset_path: str,
        model_name: str,
        api_key: str,
        output_dir: str,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
        sample_size: Optional[int] = None,
        save_interval: int = 10,
        start_count: int = 0,
        previous_results: Optional[List[Dict[str, Any]]] = None
    ) -> None:
    """
    运行评估
    
    Args:
        dataset_path: 数据集路径
        model_name: 模型名称
        api_key: API密钥
        output_dir: 输出目录
        temperature: 温度参数
        base_url: API基础URL
        sample_size: 样本大小 (如果指定，将随机抽样)
        save_interval: 保存中间结果的间隔
        start_count: 起始计数
        previous_results: 之前的测试结果，用于恢复测试
    """
    # 加载数据集
    test_cases = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_cases.append(json.loads(line))
    
    # 如果指定了样本大小，随机抽样
    if sample_size and sample_size < len(test_cases):
        import random
        random.seed(42)  # 设置随机种子以确保可重复性
        test_cases = random.sample(test_cases, sample_size)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化测试器
    tester = LLMTester(model_name, api_key, base_url, temperature)
    
    # 初始化结果列表，如果有之前的结果则使用
    results = previous_results or []
    total_count = start_count  # 初始化总计数为开始计数
    
    for test_case in tqdm(test_cases, desc=f"评估 {model_name}"):
        result = tester.evaluate_test_case(test_case)
        results.append(result)
        total_count += 1  # 每测试一个用例，总计数加1
        
        # 按指定间隔保存中间结果，防止API错误导致全部丢失
        if len(results) % save_interval == 0:
            with open(f"{output_dir}/intermediate_results_{total_count}.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n中间结果已保存: {output_dir}/intermediate_results_{total_count}.json")
        
        # 添加短暂延迟，避免API速率限制
        time.sleep(0.5)
    
    # 保存完整结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/{model_name}_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"评估完成，结果已保存到 {results_file}")
    
    # 分析结果
    analyze_results(results, output_dir, model_name, timestamp)

def analyze_results(results: List[Dict[str, Any]], output_dir: str, model_name: str, timestamp: str) -> None:
    """
    分析评估结果
    
    Args:
        results: 评估结果列表
        output_dir: 输出目录
        model_name: 模型名称
        timestamp: 时间戳
    """
    # 转换为DataFrame便于分析
    df = pd.DataFrame(results)
    
    # 计算总体准确率
    overall_accuracy = df['is_correct'].mean()
    
    # 按类型分组计算准确率
    type_accuracy = df.groupby('type')['is_correct'].mean()
    
    # 按顺序分组计算准确率
    order_accuracy = df.groupby('order')['is_correct'].mean()
    
    # 按位数分组计算准确率
    # 创建位数范围分类
    df['max_digits'] = df[['digit_length_A', 'digit_length_B']].max(axis=1)
    df['digit_range'] = pd.cut(df['max_digits'], 
                              bins=[0, 3, 10, 25, float('inf')],
                              labels=['1-3位', '4-10位', '11-25位', '25位以上'])
    digit_range_accuracy = df.groupby('digit_range')['is_correct'].mean()
    
    # 按具体位数计算准确率
    digit_accuracy = df.groupby('max_digits')['is_correct'].mean().reset_index()
    
    # 按符号映射分组计算准确率（仅符号类型）
    symbol_df = df[df['type'] == 'symbolic']
    if not symbol_df.empty:
        symbol_mapping_accuracy = symbol_df.groupby('symbolic_mapping_id')['is_correct'].mean()
    else:
        symbol_mapping_accuracy = pd.Series()
    
    # 检查交换律一致性
    # 对于每个数对，检查A+B和B+A是否都正确或都错误
    pair_consistency = df.groupby(['pair_id', 'type']).agg({
        'is_correct': ['all', 'any', 'mean']
    })
    pair_consistency.columns = ['all_correct', 'any_correct', 'accuracy']
    
    # 计算非对称失败率（一个方向对，另一个方向错）
    asymmetric_failure_rate = (pair_consistency['any_correct'] & ~pair_consistency['all_correct']).mean()
    
    # 保存分析结果
    analysis = {
        "model_name": model_name,
        "timestamp": timestamp,
        "total_test_cases": len(df),
        "overall_accuracy": float(overall_accuracy),
        "type_accuracy": type_accuracy.to_dict(),
        "order_accuracy": order_accuracy.to_dict(),
        "digit_range_accuracy": digit_range_accuracy.to_dict(),
        "asymmetric_failure_rate": float(asymmetric_failure_rate),
        "symbol_mapping_accuracy": symbol_mapping_accuracy.to_dict() if not symbol_mapping_accuracy.empty else {}
    }
    
    analysis_file = f"{output_dir}/{model_name}_analysis_{timestamp}.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    # 生成可视化
    plt.figure(figsize=(12, 8))
    
    # 类型准确率条形图
    plt.subplot(2, 2, 1)
    sns.barplot(x=type_accuracy.index, y=type_accuracy.values)
    plt.title('不同类型的准确率')
    plt.ylim(0, 1)
    
    # 顺序准确率条形图
    plt.subplot(2, 2, 2)
    sns.barplot(x=order_accuracy.index, y=order_accuracy.values)
    plt.title('不同顺序的准确率')
    plt.ylim(0, 1)
    
    # 位数范围准确率条形图
    plt.subplot(2, 2, 3)
    sns.barplot(x=digit_range_accuracy.index, y=digit_range_accuracy.values)
    plt.title('不同位数范围的准确率')
    plt.ylim(0, 1)
    
    # 位数与准确率关系曲线图
    plt.subplot(2, 2, 4)
    sns.lineplot(data=digit_accuracy, x='max_digits', y='is_correct')
    plt.title('位数与准确率的关系')
    plt.xlabel('最大位数')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_analysis_{timestamp}.png", dpi=300)
    
    print(f"分析结果已保存到 {analysis_file}")
    print(f"可视化图表已保存到 {output_dir}/{model_name}_analysis_{timestamp}.png")
    
    # 打印主要分析结果
    print("\n===== 主要分析结果 =====")
    print(f"总体准确率: {overall_accuracy:.4f}")
    print("\n类型准确率:")
    for type_name, acc in type_accuracy.items():
        print(f"  {type_name}: {acc:.4f}")
    
    print("\n顺序准确率:")
    for order_name, acc in order_accuracy.items():
        print(f"  {order_name}: {acc:.4f}")
    
    print("\n位数范围准确率:")
    for range_name, acc in digit_range_accuracy.items():
        print(f"  {range_name}: {acc:.4f}")
    
    print(f"\n非对称失败率 (一个方向对，另一个方向错): {asymmetric_failure_rate:.4f}")
    
    if not symbol_mapping_accuracy.empty:
        print("\n符号映射准确率:")
        for mapping_id, acc in symbol_mapping_accuracy.items():
            print(f"  {mapping_id}: {acc:.4f}")

def main():
    parser = argparse.ArgumentParser(description='评估大模型的加法能力')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--model', type=str, required=True,
                        help='模型名称')
    parser.add_argument('--api_key', type=str, required=True,
                        help='API密钥')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='温度参数')
    parser.add_argument('--base_url', type=str, default=None,
                        help='API基础URL')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='样本大小 (如果指定，将随机抽样)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存中间结果的间隔')
    parser.add_argument('--start_count', type=int, default=0,
                        help='起始计数，用于恢复测试')
    parser.add_argument('--previous_results', type=str, default=None,
                        help='之前的测试结果文件路径，用于恢复测试')
    
    args = parser.parse_args()
    
    # 如果指定了之前的结果文件，加载它
    previous_results = None
    if args.previous_results and os.path.exists(args.previous_results):
        try:
            with open(args.previous_results, 'r', encoding='utf-8') as f:
                previous_results = json.load(f)
            print(f"加载了 {len(previous_results)} 条之前的测试结果")
        except Exception as e:
            print(f"警告: 加载之前的结果文件时出错: {e}")
    
    run_evaluation(
        dataset_path=args.dataset,
        model_name=args.model,
        api_key=args.api_key,
        output_dir=args.output_dir,
        temperature=args.temperature,
        base_url=args.base_url,
        sample_size=args.sample_size,
        save_interval=args.save_interval,
        start_count=args.start_count,
        previous_results=previous_results
    )

if __name__ == "__main__":
    main()
