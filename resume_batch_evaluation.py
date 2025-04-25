#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量测试多个大模型的加法能力，支持从中断处恢复测试
检查目标路径下是否已存在目标模型的文件夹，如果存在，读取其中最新的中间结果，然后继续运行
"""

import os
import json
import time
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from config import DATASET_CONFIG, MODELS, OUTPUT_CONFIG
from typing import Dict, List, Any, Optional, Set, Tuple

def find_latest_intermediate_results(model_output_dir: str) -> Tuple[Optional[str], Set[str], int]:
    """
    查找最新的中间结果文件
    
    Args:
        model_output_dir: 模型输出目录
        
    Returns:
        (最新的中间结果文件路径, 已测试的用例 ID集合, 已测试的用例数量)
    """
    if not os.path.exists(model_output_dir):
        return None, set(), 0
    
    # 查找所有中间结果文件
    intermediate_files = [f for f in os.listdir(model_output_dir) if f.startswith("intermediate_results_")]
    if not intermediate_files:
        return None, set(), 0
    
    # 提取文件中的数字（已测试的用例数量）
    file_numbers = []
    for filename in intermediate_files:
        match = re.search(r'intermediate_results_(\d+)\.json', filename)
        if match:
            file_numbers.append((int(match.group(1)), filename))
    
    if not file_numbers:
        return None, set(), 0
    
    # 按数字排序，获取最大的文件（包含最多测试用例的文件）
    file_numbers.sort(reverse=True)
    latest_count, latest_file = file_numbers[0]
    latest_file_path = os.path.join(model_output_dir, latest_file)
    
    # 读取文件，获取已测试的用例 ID
    tested_case_ids = set()
    try:
        with open(latest_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            for result in results:
                tested_case_ids.add(result["case_id"])
    except Exception as e:
        print(f"警告: 读取中间结果文件 {latest_file_path} 时出错: {e}")
        return None, set(), 0
    
    # 始终使用文件中的实际结果数量作为已测试用例数量
    # 文件名中的数字可能不准确，但我们仍然使用它作为起始计数点
    # 这样可以确保新的中间结果文件名不会重复
    actual_count = len(tested_case_ids)
    print(f"文件名中的数字: {latest_count}, 实际结果数量: {actual_count}")
    return latest_file_path, tested_case_ids, actual_count

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    加载数据集
    
    Args:
        dataset_path: 数据集文件路径
        
    Returns:
        测试用例列表
    """
    test_cases = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_cases.append(json.loads(line))
    return test_cases

def run_single_evaluation_with_resume(model_config: Dict[str, Any], 
                                     dataset_path: str, 
                                     output_dir: str,
                                     sample_size: Optional[int] = None) -> str:
    """
    运行单个模型的评估，支持从中断处恢复
    
    Args:
        model_config: 模型配置
        dataset_path: 数据集路径
        output_dir: 输出目录
        sample_size: 样本大小
        
    Returns:
        结果文件路径
    """
    # 创建模型专用输出目录
    model_output_dir = os.path.join(output_dir, model_config["name"])
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 查找最新的中间结果
    latest_file, tested_case_ids, tested_count = find_latest_intermediate_results(model_output_dir)
    
    if latest_file:
        print(f"发现模型 {model_config['name']} 的中间结果: {latest_file}")
        print(f"已测试 {tested_count} 个用例，将从中断处继续测试")
    else:
        print(f"未找到模型 {model_config['name']} 的中间结果，将从头开始测试")
    
    # 加载数据集
    all_test_cases = load_dataset(dataset_path)
    
    # 如果指定了样本大小，随机抽样
    if sample_size and sample_size < len(all_test_cases):
        import random
        random.seed(42)  # 设置随机种子以确保可重复性
        all_test_cases = random.sample(all_test_cases, sample_size)
    
    # 筛选出未测试的用例
    untested_cases = [case for case in all_test_cases if case["case_id"] not in tested_case_ids]
    
    # 打印总用例数量信息，用于调试
    print(f"数据集总用例数量: {len(all_test_cases)}")
    
    # 已经使用实际的已测试用例数量，不需要再次检查
    
    print(f"已测试用例数量: {tested_count}")
    print(f"未测试用例数量: {len(untested_cases)}")
    
    if not untested_cases:
        print(f"模型 {model_config['name']} 的所有用例已测试完成")
        
        # 查找最新的分析结果文件
        analysis_files = [f for f in os.listdir(model_output_dir) if f.endswith("_analysis.json")]
        if analysis_files:
            analysis_files.sort(reverse=True)
            latest_analysis = os.path.join(model_output_dir, analysis_files[0])
            print(f"最新的分析结果文件: {latest_analysis}")
            return latest_analysis
        
        return None
    
    print(f"模型 {model_config['name']} 还有 {len(untested_cases)} 个用例需要测试")
    
    # 创建临时数据集文件，只包含未测试的用例
    temp_dataset_path = os.path.join(model_output_dir, "untested_cases.jsonl")
    with open(temp_dataset_path, 'w', encoding='utf-8') as f:
        for case in untested_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    
    # 构建命令
    cmd = [
        "python", "evaluate_llm_addition.py",
        "--dataset", temp_dataset_path,
        "--model", model_config["name"],
        "--api_key", model_config["api_key"],
        "--output_dir", model_output_dir,
        "--temperature", str(model_config["temperature"])
    ]
    
    # 添加可选参数
    if model_config.get("base_url"):
        cmd.extend(["--base_url", model_config["base_url"]])
    
    # 添加保存中间结果的参数
    # 在evaluate_llm_addition.py中添加参数，指定保存中间结果的间隔
    cmd.extend(["--save_interval", "10"])
    
    # 添加开始计数参数，使用文件名中的数字作为起始计数点
    # 这样可以确保新的中间结果文件名不会重复
    latest_count = 0
    
    # 如果有最新的中间结果文件，使用它的文件名中的数字作为起始计数点
    # 并添加之前的结果文件参数，确保合并之前的结果
    if latest_file:
        match = re.search(r'intermediate_results_(\d+)\.json', os.path.basename(latest_file))
        if match:
            latest_count = int(match.group(1))
        cmd.extend(["--start_count", str(latest_count)])
        cmd.extend(["--previous_results", latest_file])
    
    # 运行评估
    print(f"\n开始评估模型: {model_config['name']} (继续测试)")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        # 直接使用subprocess.run，并将输出重定向到终端
        subprocess.run(cmd, check=True)
        
        # 合并结果
        if latest_file:
            # 读取之前的结果
            with open(latest_file, 'r', encoding='utf-8') as f:
                previous_results = json.load(f)
            
            # 读取新的结果
            result_files = [f for f in os.listdir(model_output_dir) if f.endswith("_results.json")]
            if result_files:
                result_files.sort(reverse=True)
                latest_result_file = os.path.join(model_output_dir, result_files[0])
                with open(latest_result_file, 'r', encoding='utf-8') as f:
                    new_results = json.load(f)
                
                # 合并结果
                merged_results = previous_results + new_results
                
                # 保存合并后的结果
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                merged_file = os.path.join(model_output_dir, f"{model_config['name']}_results_{timestamp}.json")
                with open(merged_file, 'w', encoding='utf-8') as f:
                    json.dump(merged_results, f, ensure_ascii=False, indent=2)
                
                print(f"已合并之前的 {len(previous_results)} 个结果和新的 {len(new_results)} 个结果")
                print(f"合并后的结果保存在: {merged_file}")
                
                # 重新生成分析结果
                from evaluate_llm_addition import analyze_results
                analyze_results(merged_results, model_output_dir, model_config["name"], timestamp)
                
                # 返回最新的分析结果文件
                analysis_file = os.path.join(model_output_dir, f"{model_config['name']}_analysis_{timestamp}.json")
                return analysis_file
        
        # 如果没有之前的结果，或者合并失败，查找最新的分析结果文件
        analysis_files = [f for f in os.listdir(model_output_dir) if f.endswith("_analysis.json")]
        if analysis_files:
            analysis_files.sort(reverse=True)
            latest_analysis = os.path.join(model_output_dir, analysis_files[0])
            print(f"最新的分析结果文件: {latest_analysis}")
            return latest_analysis
        
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"错误: 评估模型 {model_config['name']} 时出错: {e}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)

def generate_summary_report(result_files: List[str], output_file: str, chart_file: str) -> None:
    """
    生成汇总报告
    
    Args:
        result_files: 结果文件列表
        output_file: 输出文件路径
        chart_file: 图表文件路径
    """
    if not result_files:
        print("错误: 没有可用的结果文件")
        return
    
    # 加载所有结果
    results = []
    for file_path in result_files:
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    results.append(data)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析结果文件 {file_path}")
    
    if not results:
        print("错误: 没有可用的结果数据")
        return
    
    # 提取关键指标
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [],
        "overall_accuracy": {},
        "numerical_accuracy": {},
        "symbolic_accuracy": {},
        "asymmetric_failure_rate": {},
        "digit_range_accuracy": {}
    }
    
    for result in results:
        model_name = result["model_name"]
        summary["models"].append(model_name)
        summary["overall_accuracy"][model_name] = result["overall_accuracy"]
        summary["numerical_accuracy"][model_name] = result["type_accuracy"].get("numerical", 0)
        summary["symbolic_accuracy"][model_name] = result["type_accuracy"].get("symbolic", 0)
        summary["asymmetric_failure_rate"][model_name] = result["asymmetric_failure_rate"]
        
        # 位数范围准确率
        for range_name, acc in result["digit_range_accuracy"].items():
            if range_name not in summary["digit_range_accuracy"]:
                summary["digit_range_accuracy"][range_name] = {}
            summary["digit_range_accuracy"][range_name][model_name] = acc
    
    # 保存汇总报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"汇总报告已保存到: {output_file}")
    
    # 生成比较图表
    generate_comparison_charts(summary, chart_file)

def generate_comparison_charts(summary: Dict[str, Any], chart_file: str) -> None:
    """
    生成模型比较图表
    
    Args:
        summary: 汇总数据
        chart_file: 图表文件路径
    """
    models = summary["models"]
    
    # 准备数据
    overall_acc = [summary["overall_accuracy"][model] for model in models]
    numerical_acc = [summary["numerical_accuracy"][model] for model in models]
    symbolic_acc = [summary["symbolic_accuracy"][model] for model in models]
    asymmetric_rate = [summary["asymmetric_failure_rate"][model] for model in models]
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 总体准确率
    plt.subplot(2, 2, 1)
    sns.barplot(x=models, y=overall_acc)
    plt.title('各模型总体准确率')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # 数值vs符号准确率
    plt.subplot(2, 2, 2)
    df = pd.DataFrame({
        'Model': models + models,
        'Type': ['数值'] * len(models) + ['符号'] * len(models),
        'Accuracy': numerical_acc + symbolic_acc
    })
    sns.barplot(x='Model', y='Accuracy', hue='Type', data=df)
    plt.title('数值vs符号准确率')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(title='类型')
    
    # 非对称失败率
    plt.subplot(2, 2, 3)
    sns.barplot(x=models, y=asymmetric_rate)
    plt.title('非对称失败率 (一个方向对，另一个方向错)')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # 位数范围准确率
    plt.subplot(2, 2, 4)
    digit_ranges = list(summary["digit_range_accuracy"].keys())
    data = []
    for model in models:
        for digit_range in digit_ranges:
            data.append({
                'Model': model,
                'Digit Range': digit_range,
                'Accuracy': summary["digit_range_accuracy"][digit_range].get(model, 0)
            })
    
    df = pd.DataFrame(data)
    sns.barplot(x='Digit Range', y='Accuracy', hue='Model', data=df)
    plt.title('不同位数范围的准确率')
    plt.ylim(0, 1)
    plt.legend(title='模型')
    
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300)
    
    print(f"比较图表已保存到: {chart_file}")
    
    # 打印主要结果
    print("\n===== 模型比较结果 =====")
    print("总体准确率:")
    for i, model in enumerate(models):
        print(f"  {model}: {overall_acc[i]:.4f}")
    
    print("\n数值准确率:")
    for i, model in enumerate(models):
        print(f"  {model}: {numerical_acc[i]:.4f}")
    
    print("\n符号准确率:")
    for i, model in enumerate(models):
        print(f"  {model}: {symbolic_acc[i]:.4f}")
    
    print("\n非对称失败率:")
    for i, model in enumerate(models):
        print(f"  {model}: {asymmetric_rate[i]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='批量测试多个大模型的加法能力，支持从中断处恢复')
    parser.add_argument('--models', type=str, nargs='+',
                        help='要测试的模型名称，不指定则测试配置中的所有模型')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(OUTPUT_CONFIG["dir"], exist_ok=True)
    
    # 确定要测试的模型
    models_to_test = []
    if args.models:
        # 只测试指定的模型
        for model_name in args.models:
            for model in MODELS:
                if model["name"] == model_name:
                    models_to_test.append(model)
                    break
            else:
                print(f"警告: 配置中未找到模型 {model_name}")
    else:
        # 测试所有配置的模型
        models_to_test = MODELS
    
    if not models_to_test:
        print("错误: 没有可测试的模型")
        return
    
    # 依次测试每个模型
    result_files = []
    for model in models_to_test:
        result_file = run_single_evaluation_with_resume(
            model_config=model,
            dataset_path=DATASET_CONFIG["path"],
            output_dir=OUTPUT_CONFIG["dir"],
            sample_size=DATASET_CONFIG["sample_size"]
        )
        result_files.append(result_file)
    
    # 生成汇总报告
    summary_file = os.path.join(OUTPUT_CONFIG["dir"], OUTPUT_CONFIG["summary_file"])
    chart_file = os.path.join(OUTPUT_CONFIG["dir"], OUTPUT_CONFIG["summary_chart"])
    generate_summary_report(result_files, summary_file, chart_file)

if __name__ == "__main__":
    main()
