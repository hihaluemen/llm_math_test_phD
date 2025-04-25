#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试单个加法测试用例，查看生成的提示词和结果
"""

import json
import argparse
import os
from config import MODELS
from openai import OpenAI

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

def create_prompt(test_case):
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

在这个符号系统中，加法遵循与十进制数字相同的规则。例如，如果 '1' 映射为 '#'，'2' 映射为 '$'，那么 '# + $' 就等于 '3' 对应的符号。

请计算: {test_case['operand1']} + {test_case['operand2']} = \\boxed{{?}}

请直接给出答案，不要有任何解释。"""
        return prompt

def extract_answer(response):
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
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('问题:', '计算:')):
            # 返回第一个非空且不是问题陈述的行
            return line
    
    # 如果都失败了，返回整个响应
    return response.strip()

def test_single_case(test_case, model_config):
    """
    测试单个测试用例
    
    Args:
        test_case: 测试用例
        model_config: 模型配置
        
    Returns:
        测试结果
    """
    # 创建LLM客户端
    llm = ChatLLM(model_config)
    
    # 创建提示词
    prompt = create_prompt(test_case)
    
    # 调用API
    print(f"\n===== 提示词 =====\n{prompt}\n")
    
    response = llm.generate(prompt)
    print(f"\n===== 模型响应 =====\n{response}\n")
    
    # 提取答案
    extracted_answer = extract_answer(response)
    print(f"\n===== 提取的答案 =====\n{extracted_answer}\n")
    
    # 检查答案是否正确
    is_correct = extracted_answer == test_case["correct_answer"]
    print(f"\n===== 正确答案 =====\n{test_case['correct_answer']}\n")
    print(f"\n===== 结果 =====\n{'✓ 正确' if is_correct else '✗ 错误'}\n")
    
    return {
        "case_id": test_case["case_id"],
        "prompt": prompt,
        "response": response,
        "extracted_answer": extracted_answer,
        "correct_answer": test_case["correct_answer"],
        "is_correct": is_correct
    }

def main():
    parser = argparse.ArgumentParser(description='测试单个加法测试用例')
    parser.add_argument('--case_id', type=str, required=True,
                        help='测试用例ID')
    parser.add_argument('--dataset', type=str, default='/Users/lhh/Desktop/llm/learn/test_llm/0424/addition_dataset.jsonl',
                        help='数据集文件路径')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='模型名称')
    parser.add_argument('--output_dir', type=str, default='/Users/lhh/Desktop/llm/learn/test_llm/0424/results/single_tests',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 查找测试用例
    test_case = None
    with open(args.dataset, 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line)
            if case["case_id"] == args.case_id:
                test_case = case
                break
    
    if not test_case:
        print(f"错误: 未找到ID为 {args.case_id} 的测试用例")
        return
    
    # 查找模型配置
    model_config = None
    for model in MODELS:
        if model["name"] == args.model:
            model_config = model
            break
    
    if not model_config:
        print(f"错误: 未找到名称为 {args.model} 的模型配置")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 测试用例
    result = test_single_case(test_case, model_config)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, f"{args.case_id}_{args.model}_result.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
