#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量测试多个大模型的加法能力
读取config.py中的配置信息，依次测试每个模型
"""

import os
import json
import time
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from config import DATASET_CONFIG, MODELS, OUTPUT_CONFIG
from typing import Dict, List, Any

def run_single_evaluation(model_config: Dict[str, Any], 
                         dataset_path: str, 
                         output_dir: str,
                         sample_size: int = None) -> str:
    """
    运行单个模型的评估
    
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
    
    # 构建命令
    cmd = [
        "python", "0424/evaluate_llm_addition.py",
        "--dataset", dataset_path,
        "--model", model_config["name"],
        "--api_key", model_config["api_key"],
        "--output_dir", model_output_dir,
        "--temperature", str(model_config["temperature"])
    ]
    
    # 添加可选参数
    if model_config.get("base_url"):
        cmd.extend(["--base_url", model_config["base_url"]])
    
    if sample_size:
        cmd.extend(["--sample_size", str(sample_size)])
    
    # 运行评估
    print(f"\n开始评估模型: {model_config['name']}")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        
        # 找到最新的结果文件
        result_files = [f for f in os.listdir(model_output_dir) if f.endswith("_analysis.json")]
        if not result_files:
            print(f"警告: 未找到模型 {model_config['name']} 的分析结果文件")
            return None
        
        # 按时间戳排序，获取最新的结果文件
        result_files.sort(reverse=True)
        latest_result = os.path.join(model_output_dir, result_files[0])
        print(f"模型 {model_config['name']} 评估完成，结果保存在: {latest_result}")
        return latest_result
        
    except subprocess.CalledProcessError as e:
        print(f"错误: 评估模型 {model_config['name']} 时出错: {e}")
        return None

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
    parser = argparse.ArgumentParser(description='批量测试多个大模型的加法能力')
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
        result_file = run_single_evaluation(
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
