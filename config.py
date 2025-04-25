#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大模型加法能力测试配置文件
"""

# 数据集配置
DATASET_CONFIG = {
    "path": "addition_dataset.jsonl",
    "sample_size": 500  # 设置为None则使用全部数据集，设置具体数字则随机抽样
}

# 模型配置
MODELS = [
    {
        "name": "gpt-4o",
        "api_key": "sk-",
        "base_url": "https://",  
        "temperature": 0.0
    },
    {
        "name": "gpt-4.1-2025-04-14",
        "api_key": "sk-",
        "base_url": "https://",  
        "temperature": 0.0
    },
    {
        "name": "claude-3-7-sonnet-20250219",
        "api_key": "sk-",
        "base_url": "https://",  
        "temperature": 0.0
    },
    {
        "name": "gemini-2.5-pro-exp-03-25",
        "api_key": "sk-",
        "base_url": "https://",  
        "temperature": 0.0
    }
]

# 输出配置
OUTPUT_CONFIG = {
    "dir": "results",
    "summary_file": "summary_report.json",
    "summary_chart": "model_comparison.png"
}
