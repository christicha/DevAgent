import os
from datasets import load_dataset
import json
from config import DATA_CONFIG


class DataLoader:
    @staticmethod
    def load_human_eval():
        """加载HumanEval代码基准数据集"""
        dataset = load_dataset("openai_humaneval")
        return dataset["test"]

    @staticmethod
    def load_swe_bench():
        """加载SWE-Bench评估数据集"""
        if os.path.exists(DATA_CONFIG["swe_bench_path"]):
            with open(DATA_CONFIG["swe_bench_path"], "r") as f:
                return [json.loads(line) for line in f]
        # 若本地无数据，可从HuggingFace加载
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite")
        return dataset["test"]

    @staticmethod
    def get_sample_task(task_type="code_generation"):
        """获取示例任务（用于CLI演示）"""
        if task_type == "code_generation":
            return {
                "instruction": "编写一个Python函数，实现快速排序算法，要求处理整数列表，包含边界条件检查",
                "type": "code_generation"
            }
        elif task_type == "bug_fix":
            return {
                "instruction": "修复以下Python代码的Bug：\n```python\ndef calculate_average(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total / len(numbers)\n```\n问题：当输入空列表时会抛出ZeroDivisionError",
                "type": "bug_fix"
            }
        elif task_type == "test_writing":
            return {
                "instruction": "为以下Python函数编写单元测试：\n```python\ndef is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```\n要求覆盖边界条件（如n=0、1、2、质数、非质数）",
                "type": "test_writing"
            }