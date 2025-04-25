#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析LLM加法测试结果，生成详细的统计报告
包括总体准确率、不同类型的准确率、位数范围准确率等
同时显示满足条件的样本数和总样本数
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

def load_results(result_file: str) -> List[Dict[str, Any]]:
    """
    加载测试结果
    
    Args:
        result_file: 结果文件路径
        
    Returns:
        测试结果列表
    """
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析测试结果
    
    Args:
        results: 测试结果列表
        
    Returns:
        分析结果
    """
    if not results:
        return {"error": "没有可用的结果数据"}
    
    # 总体准确率
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    total_count = len(results)
    overall_accuracy = correct_count / total_count if total_count > 0 else 0
    
    # 按类型分类
    numerical_results = [r for r in results if r.get("type") == "numerical"]
    symbolic_results = [r for r in results if r.get("type") == "symbolic"]
    
    numerical_correct = sum(1 for r in numerical_results if r.get("is_correct", False))
    symbolic_correct = sum(1 for r in symbolic_results if r.get("is_correct", False))
    
    numerical_accuracy = numerical_correct / len(numerical_results) if len(numerical_results) > 0 else 0
    symbolic_accuracy = symbolic_correct / len(symbolic_results) if len(symbolic_results) > 0 else 0
    
    # 按顺序分类
    ab_results = [r for r in results if r.get("order") == "A+B"]
    ba_results = [r for r in results if r.get("order") == "B+A"]
    
    ab_correct = sum(1 for r in ab_results if r.get("is_correct", False))
    ba_correct = sum(1 for r in ba_results if r.get("is_correct", False))
    
    ab_accuracy = ab_correct / len(ab_results) if len(ab_results) > 0 else 0
    ba_accuracy = ba_correct / len(ba_results) if len(ba_results) > 0 else 0
    
    # 按位数范围分类
    digit_ranges = {
        "1-3位": (1, 3),
        "4-10位": (4, 10),
        "11-25位": (11, 25),
        "25位以上": (26, float('inf'))
    }
    
    digit_range_results = {}
    for range_name, (min_digit, max_digit) in digit_ranges.items():
        range_results = [r for r in results if min_digit <= max(r.get("digit_length_A", 0), r.get("digit_length_B", 0)) <= max_digit]
        correct_in_range = sum(1 for r in range_results if r.get("is_correct", False))
        if range_results:
            digit_range_results[range_name] = {
                "accuracy": correct_in_range / len(range_results),
                "correct": correct_in_range,
                "total": len(range_results)
            }
        else:
            digit_range_results[range_name] = {
                "accuracy": float('nan'),
                "correct": 0,
                "total": 0
            }
    
    # 非对称失败率（一个方向对，另一个方向错）
    pair_ids = set(r.get("pair_id") for r in results)
    asymmetric_failure_count = 0
    total_pairs = 0
    
    for pair_id in pair_ids:
        pair_results = [r for r in results if r.get("pair_id") == pair_id]
        if len(pair_results) == 2:  # 确保有A+B和B+A两个方向
            total_pairs += 1
            results_correct = [r.get("is_correct", False) for r in pair_results]
            if results_correct.count(True) == 1:  # 只有一个方向正确
                asymmetric_failure_count += 1
    
    asymmetric_failure_rate = asymmetric_failure_count / total_pairs if total_pairs > 0 else 0
    
    # 按符号映射分类
    symbolic_mapping_ids = set(r.get("symbolic_mapping_id") for r in symbolic_results if r.get("symbolic_mapping_id"))
    symbolic_mapping_results = {}
    
    for mapping_id in symbolic_mapping_ids:
        mapping_results = [r for r in symbolic_results if r.get("symbolic_mapping_id") == mapping_id]
        correct_with_mapping = sum(1 for r in mapping_results if r.get("is_correct", False))
        symbolic_mapping_results[mapping_id] = {
            "accuracy": correct_with_mapping / len(mapping_results) if len(mapping_results) > 0 else 0,
            "correct": correct_with_mapping,
            "total": len(mapping_results)
        }
    
    # 汇总分析结果
    analysis = {
        "overall_accuracy": {
            "accuracy": overall_accuracy,
            "correct": correct_count,
            "total": total_count
        },
        "type_accuracy": {
            "numerical": {
                "accuracy": numerical_accuracy,
                "correct": numerical_correct,
                "total": len(numerical_results)
            },
            "symbolic": {
                "accuracy": symbolic_accuracy,
                "correct": symbolic_correct,
                "total": len(symbolic_results)
            }
        },
        "order_accuracy": {
            "A+B": {
                "accuracy": ab_accuracy,
                "correct": ab_correct,
                "total": len(ab_results)
            },
            "B+A": {
                "accuracy": ba_accuracy,
                "correct": ba_correct,
                "total": len(ba_results)
            }
        },
        "digit_range_accuracy": digit_range_results,
        "asymmetric_failure": {
            "rate": asymmetric_failure_rate,
            "count": asymmetric_failure_count,
            "total": total_pairs
        },
        "symbolic_mapping_accuracy": symbolic_mapping_results
    }
    
    return analysis

def generate_visualizations(analysis: Dict[str, Any], output_file: str) -> None:
    """
    生成可视化图表
    
    Args:
        analysis: 分析结果
        output_file: 输出文件路径
    """
    if "error" in analysis:
        print(f"错误: {analysis['error']}")
        return
    
    plt.figure(figsize=(15, 15))
    
    # 总体准确率
    plt.subplot(3, 2, 1)
    overall = analysis["overall_accuracy"]
    plt.bar(["总体准确率"], [overall["accuracy"]], color='blue')
    plt.title(f'总体准确率: {overall["accuracy"]:.4f} ({overall["correct"]}/{overall["total"]})')
    plt.ylim(0, 1)
    
    # 类型准确率
    plt.subplot(3, 2, 2)
    type_acc = analysis["type_accuracy"]
    types = list(type_acc.keys())
    accuracies = [type_acc[t]["accuracy"] for t in types]
    labels = [f"{t}\n({type_acc[t]['correct']}/{type_acc[t]['total']})" for t in types]
    plt.bar(labels, accuracies, color=['green', 'red'])
    plt.title('不同类型的准确率')
    plt.ylim(0, 1)
    
    # 顺序准确率
    plt.subplot(3, 2, 3)
    order_acc = analysis["order_accuracy"]
    orders = list(order_acc.keys())
    accuracies = [order_acc[o]["accuracy"] for o in orders]
    labels = [f"{o}\n({order_acc[o]['correct']}/{order_acc[o]['total']})" for o in orders]
    plt.bar(labels, accuracies, color=['orange', 'purple'])
    plt.title('不同顺序的准确率')
    plt.ylim(0, 1)
    
    # 位数范围准确率
    plt.subplot(3, 2, 4)
    digit_range_acc = analysis["digit_range_accuracy"]
    ranges = list(digit_range_acc.keys())
    accuracies = [digit_range_acc[r]["accuracy"] for r in ranges]
    labels = [f"{r}\n({digit_range_acc[r]['correct']}/{digit_range_acc[r]['total']})" for r in ranges]
    plt.bar(labels, accuracies, color=['cyan', 'magenta', 'yellow', 'black'])
    plt.title('不同位数范围的准确率')
    plt.ylim(0, 1)
    
    # 非对称失败率
    plt.subplot(3, 2, 5)
    asymm = analysis["asymmetric_failure"]
    plt.bar(["非对称失败率"], [asymm["rate"]], color='brown')
    plt.title(f'非对称失败率: {asymm["rate"]:.4f} ({asymm["count"]}/{asymm["total"]})')
    plt.ylim(0, 1)
    
    # 符号映射准确率
    plt.subplot(3, 2, 6)
    sym_map_acc = analysis["symbolic_mapping_accuracy"]
    if sym_map_acc:
        mappings = list(sym_map_acc.keys())
        accuracies = [sym_map_acc[m]["accuracy"] for m in mappings]
        labels = [f"{m}\n({sym_map_acc[m]['correct']}/{sym_map_acc[m]['total']})" for m in mappings]
        plt.bar(labels, accuracies, color=['lightgreen', 'lightblue', 'pink'])
        plt.title('不同符号映射的准确率')
        plt.ylim(0, 1)
    else:
        plt.text(0.5, 0.5, "没有符号映射数据", horizontalalignment='center', verticalalignment='center')
        plt.title('不同符号映射的准确率')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"可视化图表已保存到: {output_file}")

def print_analysis_summary(analysis: Dict[str, Any]) -> None:
    """
    打印分析摘要
    
    Args:
        analysis: 分析结果
    """
    if "error" in analysis:
        print(f"错误: {analysis['error']}")
        return
    
    print("\n===== 详细分析结果 =====")
    
    # 总体准确率
    overall = analysis["overall_accuracy"]
    print(f"总体准确率: {overall['accuracy']:.4f} ({overall['correct']}/{overall['total']})")
    
    # 类型准确率
    print("\n类型准确率:")
    for type_name, data in analysis["type_accuracy"].items():
        print(f"  {type_name}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
    
    # 顺序准确率
    print("\n顺序准确率:")
    for order, data in analysis["order_accuracy"].items():
        print(f"  {order}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
    
    # 位数范围准确率
    print("\n位数范围准确率:")
    for range_name, data in analysis["digit_range_accuracy"].items():
        if data["total"] > 0:
            print(f"  {range_name}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
        else:
            print(f"  {range_name}: 无数据")
    
    # 非对称失败率
    asymm = analysis["asymmetric_failure"]
    print(f"\n非对称失败率 (一个方向对，另一个方向错): {asymm['rate']:.4f} ({asymm['count']}/{asymm['total']})")
    
    # 符号映射准确率
    print("\n符号映射准确率:")
    if analysis["symbolic_mapping_accuracy"]:
        for mapping_id, data in analysis["symbolic_mapping_accuracy"].items():
            print(f"  {mapping_id}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")
    else:
        print("  没有符号映射数据")

def main():
    parser = argparse.ArgumentParser(description='分析LLM加法测试结果')
    parser.add_argument('--result_file', type=str, required=True,
                        help='结果文件路径，可以是中间结果文件或最终结果文件')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认使用结果文件所在目录')
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.result_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载结果
    results = load_results(args.result_file)
    print(f"加载了 {len(results)} 条测试结果")
    
    # 分析结果
    analysis = analyze_results(results)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(args.result_file).split('.')[0]
    output_base = f"{base_name}_detailed_analysis_{timestamp}"
    
    # 保存分析结果
    analysis_file = os.path.join(output_dir, f"{output_base}.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"详细分析结果已保存到 {analysis_file}")
    
    # 生成可视化
    chart_file = os.path.join(output_dir, f"{output_base}.png")
    generate_visualizations(analysis, chart_file)
    
    # 打印摘要
    print_analysis_summary(analysis)

if __name__ == "__main__":
    main()
