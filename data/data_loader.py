# data/data_loader.py 完整修复版
import os
import json
from datasets import load_dataset

# 数据集配置（适配SWE-bench_Lite）
SWE_BENCH_CONFIG = {
    "dataset_name": "princeton-nlp/SWE-bench_Lite",
    "split": "test",
    "cache_dir": "./data/cache"
}


class DataLoader:
    @staticmethod
    def load_swe_bench():
        """加载SWE-bench_Lite数据集，兼容缺失字段"""
        try:
            # 创建缓存目录
            os.makedirs(SWE_BENCH_CONFIG["cache_dir"], exist_ok=True)

            # 从HF Hub加载数据集
            dataset = load_dataset(
                SWE_BENCH_CONFIG["dataset_name"],
                split=SWE_BENCH_CONFIG["split"],
                cache_dir=SWE_BENCH_CONFIG["cache_dir"]
            )

            # 第一步：先打印数据集的实际字段（帮助排查字段问题）
            if len(dataset) > 0:
                print("=== SWE-bench_Lite 实际字段列表 ===")
                print(dataset[0].keys())  # 打印第一个实例的所有字段

            # 转换为列表（兼容缺失字段，用get方法设置默认值）
            swe_bench_data = []
            for idx, instance in enumerate(dataset):
                # 核心字段（SWE-bench_Lite必含）
                core_fields = {
                    "instance_id": instance.get("instance_id", f"unknown_{idx}"),
                    "repo": instance.get("repo", "unknown/repo"),
                    "problem_statement": instance.get("problem_statement", ""),
                    "patch": instance.get("patch", ""),
                    "FAIL_TO_PASS": instance.get("FAIL_TO_PASS", ""),
                    "PASS_TO_PASS": instance.get("PASS_TO_PASS", "")
                }

                # 可选字段（SWE-bench_Lite可能缺失，设置默认值）
                optional_fields = {
                    "base_commit": instance.get("base_commit", "unknown_commit"),
                    "issue_url": instance.get("issue_url", ""),  # 缺失则为空字符串
                    "pr_url": instance.get("pr_url", ""),
                    "hints_text": instance.get("hints_text", "")
                }

                # 合并字段，避免KeyError
                processed_instance = {**core_fields, **optional_fields}
                swe_bench_data.append(processed_instance)

            print(f"\n成功加载{len(swe_bench_data)}个有效SWE-bench_Lite实例")
            return swe_bench_data

        except Exception as e:
            raise RuntimeError(f"加载SWE-bench失败：{str(e)}")

    @staticmethod
    def get_sample_task(task_type):
        """获取示例任务（仅使用核心字段，避免缺失字段）"""
        if task_type == "bug_fix":
            swe_bench_data = DataLoader.load_swe_bench()
            if not swe_bench_data:
                raise ValueError("无有效SWE-bench_Lite实例")

            # 取第一个实例作为示例（仅用核心字段）
            sample = swe_bench_data[0]
            return {
                "instruction": f"""修复以下代码仓库的Bug：
仓库：{sample['repo']}
问题描述：{sample['problem_statement']}
要求：
1. 修复后需让FAIL_TO_PASS中的测试用例全部通过；
2. 确保PASS_TO_PASS中的测试用例仍保持通过（不引入新Bug）；
3. 仅返回修复的代码补丁，无需多余内容。""",
                "task_type": "bug_fix",
                "original_instance": sample  # 携带处理后的实例（无缺失字段）
            }
        else:
            # 保留原有code_generation/test_writing逻辑（若有）
            raise NotImplementedError(f"暂不支持任务类型：{task_type}")