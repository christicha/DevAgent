import time
from data.data_loader import DataLoader


class SWEBenchEvaluator:
    def __init__(self, agent):
        self.agent = agent
        self.swe_bench_data = DataLoader.load_swe_bench()
        # 简化评估：仅验证代码可运行性和功能正确性
        self.success_count = 0
        self.total_count = 0

    def _validate_code(self, code, test_case):
        """验证代码正确性（简化版）"""
        try:
            # 执行代码并运行测试用例
            local_namespace = {}
            exec(code, local_namespace)
            # 运行测试用例（SWE-Bench包含预设测试）
            test_result = eval(test_case, local_namespace)
            return test_result is True
        except Exception as e:
            print(f"代码验证失败：{e}")
            return False

    def evaluate(self):
        """执行SWE-Bench评估"""
        # 采样前10个任务（避免耗时过长）
        test_tasks = self.swe_bench_data[:10]
        self.total_count = len(test_tasks)

        for task in test_tasks:
            self.total_count += 1
            try:
                # 执行任务
                result = self.agent.execute_task(task["instruction"])
                # 验证结果
                if self._validate_code(result["result"], task["test_case"]):
                    self.success_count += 1
                time.sleep(1)  # 避免模型过载
            except Exception as e:
                print(f"处理任务失败：{e}")
                continue

        success_rate = self.success_count / self.total_count if self.total_count > 0 else 0
        return success_rate, {
            "success": self.success_count,
            "total": self.total_count,
            "failed": self.total_count - self.success_count
        }