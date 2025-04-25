# 大模型基础加法能力测试框架

> 论文《Do PhD-level LLMs Truly Grasp Elementary Addition? Probing Rule Learning vs. Memorization in Large Language Models》内容的尝试复现
> 
> 注：相关代码和数据集由大模型辅助完成

## 项目概述

本项目是一个用于测试和评估大型语言模型（LLM）基础加法能力的框架，尝试复现论文《Do PhD-level LLMs Truly Grasp Elementary Addition? Probing Rule Learning vs. Memorization in Large Language Models》中的实验。通过设计特定的测试数据集，可以系统地评估不同模型在数值计算和符号表示下的加法能力，探究它们是依赖规则学习还是模式记忆，并重点考察交换律和对符号表示的泛化能力。

## 功能特点

- **多模型支持**：支持测试OpenAI的GPT系列、Anthropic的Claude系列、Google的Gemini系列等多种大模型
- **全面测试**：包含数值和符号表示的加法测试，覆盖1-25位数字范围
- **交换律测试**：通过A+B和B+A两种顺序测试模型对加法交换律的理解
- **符号映射**：使用多种符号映射规则测试模型对符号系统的理解和泛化能力
- **批量评估**：支持批量测试多个模型并生成比较报告
- **断点续测**：支持从中断处恢复测试，避免API调用失败导致的重复测试
- **可视化分析**：自动生成详细的分析报告和可视化图表

## 项目结构

```
.
├── README.md                        # 项目说明文档
├── config.py                        # 配置文件（模型API信息、数据集配置等）
├── generate_addition_dataset.py     # 数据集生成脚本
├── addition_dataset.jsonl           # 生成的测试数据集
├── addition_dataset_stats.json      # 数据集统计信息
├── dataset_description.md           # 数据集详细说明
├── evaluate_llm_addition.py         # 单模型评估脚本
├── run_batch_evaluation.py          # 批量评估脚本
├── resume_batch_evaluation.py       # 支持断点续测的批量评估脚本
├── analyze_results.py               # 结果分析脚本
├── test_single_case.py              # 单个测试用例测试脚本
└── results/                         # 测试结果输出目录
    ├── <model_name>/                # 各模型结果目录
    ├── summary_report.json          # 汇总报告
    └── model_comparison.png         # 模型比较图表
```

## 数据集说明

测试数据集包含8,000个测试用例，由2,000对唯一数字对构成，每对数字生成4个测试用例（数值A+B、数值B+A、符号A+B、符号B+A）。

- **数值类型**：直接使用数字进行加法测试
- **符号类型**：使用三种不同的符号映射规则（字母映射、特殊字符映射、论文示例映射）
- **数字范围**：包含小数字(1-3位，30%)、中等数字(4-10位，40%)和大数字(11-25位，30%)

详细信息请参考 `dataset_description.md`。

## 使用方法

### 1. 配置模型API信息

编辑 `config.py` 文件，设置要测试的模型API信息：

```python
MODELS = [
    {
        "name": "gpt-4o",
        "api_key": "your-api-key",
        "base_url": "https://api.openai.com/v1",  
        "temperature": 0.0
    },
    # 添加更多模型...
]
```

### 2. 单模型评估

```bash
python evaluate_llm_addition.py --dataset addition_dataset.jsonl --model gpt-4o --api_key your-api-key --output_dir results
```

### 3. 批量评估所有模型

```bash
python run_batch_evaluation.py
```

### 4. 支持断点续测的批量评估

```bash
python resume_batch_evaluation.py
```

### 5. 测试特定模型

```bash
python run_batch_evaluation.py --models gpt-4o claude-3-opus
```

### 6. 测试单个用例

```bash
python test_single_case.py --model gpt-4o --api_key your-api-key --case_id pair0001_num_AB
```

## 评估指标

- **总体准确率**：所有测试用例的准确率
- **类型准确率**：数值类型和符号类型的准确率
- **顺序准确率**：A+B和B+A顺序的准确率
- **位数范围准确率**：小数字、中等数字和大数字的准确率
- **非对称失败率**：一个方向正确但另一个方向错误的比率
- **符号映射准确率**：不同符号映射规则下的准确率

## 依赖库

- openai
- pandas
- matplotlib
- seaborn
- numpy
- tqdm

## 安装依赖

```bash
pip install openai pandas matplotlib seaborn numpy tqdm
```

## 注意事项

1. 使用前请确保已设置正确的API密钥和基础URL
2. 大规模测试可能会消耗大量API调用次数，请注意控制成本
3. 对于大数据集，建议使用支持断点续测的脚本，以防中断
4. 测试结果将保存在results目录下，包括中间结果和最终分析
