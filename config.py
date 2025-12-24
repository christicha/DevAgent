import os

import torch

# 模型配置
MODEL_CONFIG = {
    "model_name": "./deepseek-coder-7b-instruct-v1.5",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_new_tokens": 2048,
    "temperature": 0.2,
    "top_p": 0.95
}

# 数据路径
DATA_CONFIG = {
    "human_eval_path": "./data/human_eval.jsonl",
    "swe_bench_path": "./data/swe_bench.jsonl"
}

# CLI配置
CLI_CONFIG = {
    "prompt_template_path": "./core/prompt_templates/"
}

# 确保目录存在
os.makedirs(CLI_CONFIG["prompt_template_path"], exist_ok=True)