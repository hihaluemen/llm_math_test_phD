#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成用于测试大模型基础加法能力的数据集
基于论文方法，测试模型是依赖规则学习还是模式记忆
重点考察交换律和对符号表示的泛化能力
"""

import json
import random
import os
from typing import Dict, List, Tuple, Any, Optional
import argparse
from tqdm import tqdm

# 定义符号映射
SYMBOL_MAPPINGS = {
    "letter_map": {
        '0': 'Z', '1': 'Y', '2': 'X', '3': 'W', '4': 'V', 
        '5': 'U', '6': 'T', '7': 'S', '8': 'R', '9': 'Q'
    },
    "special_char_map": {
        '0': '@', '1': '#', '2': '$', '3': '%', '4': '^', 
        '5': '&', '6': '*', '7': '(', '8': ')', '9': '!'
    },
    "paper_example_map": {
        '0': 'u', '1': 'd', '2': 'a', '3': 'i', '4': 'h', 
        '5': 'v', '6': 'e', '7': 'y', '8': 'r', '9': 'c'
    }
}

def generate_number_pair(min_digits: int, max_digits: int) -> Tuple[str, str]:
    """
    生成一对具有指定位数范围的随机数
    
    Args:
        min_digits: 最小位数
        max_digits: 最大位数
        
    Returns:
        两个随机数的元组 (A, B)，均为字符串格式
    """
    # 随机选择位数
    digits_a = random.randint(min_digits, max_digits)
    digits_b = random.randint(min_digits, max_digits)
    
    # 生成随机数
    if digits_a == 1:
        a = str(random.randint(1, 9))  # 避免前导零
    else:
        a = str(random.randint(10**(digits_a-1), 10**digits_a - 1))
        
    if digits_b == 1:
        b = str(random.randint(1, 9))  # 避免前导零
    else:
        b = str(random.randint(10**(digits_b-1), 10**digits_b - 1))
    
    return a, b

def apply_symbol_mapping(number: str, mapping: Dict[str, str]) -> str:
    """
    将数字转换为符号表示
    
    Args:
        number: 要转换的数字（字符串格式）
        mapping: 数字到符号的映射字典
        
    Returns:
        转换后的符号表示
    """
    return ''.join(mapping[digit] for digit in number)

def create_test_case(pair_id: str, a: str, b: str, 
                    mapping_id: Optional[str] = None, 
                    mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    为一对数字创建四个测试用例：数值A+B、数值B+A、符号A+B、符号B+A
    
    Args:
        pair_id: 数对的唯一ID
        a: 第一个数字（字符串格式）
        b: 第二个数字（字符串格式）
        mapping_id: 符号映射的ID
        mapping: 符号映射字典
        
    Returns:
        包含四个测试用例的列表
    """
    # 计算正确答案
    correct_sum = str(int(a) + int(b))
    
    # 创建数值测试用例
    cases = [
        {
            "case_id": f"{pair_id}_num_AB",
            "pair_id": pair_id,
            "type": "numerical",
            "order": "A+B",
            "operand1": a,
            "operand2": b,
            "question": f"{a} + {b} = ?",
            "correct_answer": correct_sum,
            "digit_length_A": len(a),
            "digit_length_B": len(b),
            "symbolic_mapping_id": None,
            "symbolic_mapping_definition": None
        },
        {
            "case_id": f"{pair_id}_num_BA",
            "pair_id": pair_id,
            "type": "numerical",
            "order": "B+A",
            "operand1": b,
            "operand2": a,
            "question": f"{b} + {a} = ?",
            "correct_answer": correct_sum,
            "digit_length_A": len(a),
            "digit_length_B": len(b),
            "symbolic_mapping_id": None,
            "symbolic_mapping_definition": None
        }
    ]
    
    # 如果提供了符号映射，创建符号测试用例
    if mapping and mapping_id:
        a_sym = apply_symbol_mapping(a, mapping)
        b_sym = apply_symbol_mapping(b, mapping)
        sum_sym = apply_symbol_mapping(correct_sum, mapping)
        
        mapping_str = str(mapping)
        
        cases.extend([
            {
                "case_id": f"{pair_id}_sym_AB_{mapping_id}",
                "pair_id": pair_id,
                "type": "symbolic",
                "order": "A+B",
                "operand1": a_sym,
                "operand2": b_sym,
                "question": f"{a_sym} + {b_sym} = ?",
                "correct_answer": sum_sym,
                "digit_length_A": len(a),
                "digit_length_B": len(b),
                "symbolic_mapping_id": mapping_id,
                "symbolic_mapping_definition": mapping_str
            },
            {
                "case_id": f"{pair_id}_sym_BA_{mapping_id}",
                "pair_id": pair_id,
                "type": "symbolic",
                "order": "B+A",
                "operand1": b_sym,
                "operand2": a_sym,
                "question": f"{b_sym} + {a_sym} = ?",
                "correct_answer": sum_sym,
                "digit_length_A": len(a),
                "digit_length_B": len(b),
                "symbolic_mapping_id": mapping_id,
                "symbolic_mapping_definition": mapping_str
            }
        ])
    
    return cases

def generate_dataset(total_pairs: int, output_file: str, 
                     small_range: Tuple[int, int] = (1, 3),
                     medium_range: Tuple[int, int] = (4, 10),
                     large_range: Tuple[int, int] = (11, 25)) -> None:
    """
    生成完整的数据集
    
    Args:
        total_pairs: 要生成的唯一数对总数
        output_file: 输出文件路径
        small_range: 小数字的位数范围
        medium_range: 中等数字的位数范围
        large_range: 大数字的位数范围
    """
    # 确定每个范围的数对数量
    small_count = int(total_pairs * 0.3)  # 30%小数字
    medium_count = int(total_pairs * 0.4)  # 40%中等数字
    large_count = total_pairs - small_count - medium_count  # 剩余为大数字
    
    all_cases = []
    
    # 生成小数字数对
    print(f"生成小数字数对 ({small_count}对)...")
    for i in tqdm(range(small_count)):
        pair_id = f"pair{i+1:04d}"
        a, b = generate_number_pair(*small_range)
        
        # 随机选择一个符号映射
        mapping_id = random.choice(list(SYMBOL_MAPPINGS.keys()))
        mapping = SYMBOL_MAPPINGS[mapping_id]
        
        # 创建测试用例
        cases = create_test_case(pair_id, a, b, mapping_id, mapping)
        all_cases.extend(cases)
    
    # 生成中等数字数对
    print(f"生成中等数字数对 ({medium_count}对)...")
    for i in tqdm(range(medium_count)):
        pair_id = f"pair{i+small_count+1:04d}"
        a, b = generate_number_pair(*medium_range)
        
        # 随机选择一个符号映射
        mapping_id = random.choice(list(SYMBOL_MAPPINGS.keys()))
        mapping = SYMBOL_MAPPINGS[mapping_id]
        
        # 创建测试用例
        cases = create_test_case(pair_id, a, b, mapping_id, mapping)
        all_cases.extend(cases)
    
    # 生成大数字数对
    print(f"生成大数字数对 ({large_count}对)...")
    for i in tqdm(range(large_count)):
        pair_id = f"pair{i+small_count+medium_count+1:04d}"
        a, b = generate_number_pair(*large_range)
        
        # 随机选择一个符号映射
        mapping_id = random.choice(list(SYMBOL_MAPPINGS.keys()))
        mapping = SYMBOL_MAPPINGS[mapping_id]
        
        # 创建测试用例
        cases = create_test_case(pair_id, a, b, mapping_id, mapping)
        all_cases.extend(cases)
    
    # 保存数据集
    print(f"保存数据集到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    
    print(f"数据集生成完成！共 {len(all_cases)} 个测试用例，来自 {total_pairs} 对唯一数字。")
    
    # 生成数据集统计信息
    generate_dataset_stats(all_cases, output_file.replace('.jsonl', '_stats.json'))

def generate_dataset_stats(cases: List[Dict[str, Any]], output_file: str) -> None:
    """
    生成数据集的统计信息
    
    Args:
        cases: 所有测试用例
        output_file: 输出文件路径
    """
    stats = {
        "total_cases": len(cases),
        "total_unique_pairs": len(set(case["pair_id"] for case in cases)),
        "type_distribution": {
            "numerical": sum(1 for case in cases if case["type"] == "numerical"),
            "symbolic": sum(1 for case in cases if case["type"] == "symbolic")
        },
        "order_distribution": {
            "A+B": sum(1 for case in cases if case["order"] == "A+B"),
            "B+A": sum(1 for case in cases if case["order"] == "B+A")
        },
        "digit_length_distribution": {},
        "symbolic_mapping_distribution": {}
    }
    
    # 统计位数分布
    for case in cases:
        if case["type"] == "numerical":  # 只统计数值测试用例，避免重复计数
            a_len = case["digit_length_A"]
            b_len = case["digit_length_B"]
            
            # 记录A的位数
            if a_len not in stats["digit_length_distribution"]:
                stats["digit_length_distribution"][a_len] = 0
            stats["digit_length_distribution"][a_len] += 1
            
            # 记录B的位数
            if b_len not in stats["digit_length_distribution"]:
                stats["digit_length_distribution"][b_len] = 0
            stats["digit_length_distribution"][b_len] += 1
    
    # 统计符号映射分布
    for case in cases:
        if case["type"] == "symbolic":
            mapping_id = case["symbolic_mapping_id"]
            if mapping_id not in stats["symbolic_mapping_distribution"]:
                stats["symbolic_mapping_distribution"][mapping_id] = 0
            stats["symbolic_mapping_distribution"][mapping_id] += 1
    
    # 将位数分布转换为有序字典
    stats["digit_length_distribution"] = {
        str(k): v for k, v in sorted(stats["digit_length_distribution"].items())
    }
    
    # 保存统计信息
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"数据集统计信息已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='生成用于测试大模型基础加法能力的数据集')
    parser.add_argument('--output', type=str, default='addition_dataset.jsonl',
                        help='输出文件路径 (默认: addition_dataset.jsonl)')
    parser.add_argument('--total_pairs', type=int, default=1000,
                        help='要生成的唯一数对总数 (默认: 1000)')
    parser.add_argument('--small_min', type=int, default=1,
                        help='小数字的最小位数 (默认: 1)')
    parser.add_argument('--small_max', type=int, default=3,
                        help='小数字的最大位数 (默认: 3)')
    parser.add_argument('--medium_min', type=int, default=4,
                        help='中等数字的最小位数 (默认: 4)')
    parser.add_argument('--medium_max', type=int, default=10,
                        help='中等数字的最大位数 (默认: 10)')
    parser.add_argument('--large_min', type=int, default=11,
                        help='大数字的最小位数 (默认: 11)')
    parser.add_argument('--large_max', type=int, default=25,
                        help='大数字的最大位数 (默认: 25)')
    
    args = parser.parse_args()
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成数据集
    generate_dataset(
        total_pairs=args.total_pairs,
        output_file=args.output,
        small_range=(args.small_min, args.small_max),
        medium_range=(args.medium_min, args.medium_max),
        large_range=(args.large_min, args.large_max)
    )

if __name__ == "__main__":
    main()
