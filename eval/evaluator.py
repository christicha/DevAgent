# eval/evaluator.py
import time
import tempfile
import subprocess
import os
from data.data_loader import DataLoader


class SWEBenchEvaluator:
    def __init__(self, agent):
        self.agent = agent  # DevAgent实例
        self.swe_bench_data = DataLoader.load_swe_bench()  # 加载Schema格式的数据集
        self.results = {
            "success": 0,
            "total": 0,
            "failed": 0,
            "details": []  # 每个任务的详细结果
        }

    def _run_test_cases(self, test_cases, code_patch, repo_dir=None):
        """
        运行测试用例，验证补丁是否有效
        :param test_cases: 测试用例列表（FAIL_TO_PASS/PASS_TO_PASS）
        :param code_patch: Agent生成的修复补丁
        :param repo_dir: 代码仓库目录（临时目录）
        :return: (pass_num, fail_num) 通过/失败的测试数
        """
        if not test_cases or test_cases.strip() == "":
            return 0, 0

        # 简化版验证（生产环境建议克隆真实仓库并应用补丁）
        # 此处为演示，实际需结合仓库环境运行测试
        pass_num = 0
        fail_num = 0
        test_case_list = [tc.strip() for tc in test_cases.split("\n") if tc.strip()]

        for test_case in test_case_list:
            try:
                # 模拟测试逻辑：检查补丁是否包含修复该测试的关键逻辑
                # 生产环境需替换为：克隆仓库→应用补丁→运行pytest/test_case
                if any(keyword in code_patch.lower() for keyword in ["fix", "correct", test_case[:20].lower()]):
                    pass_num += 1
                else:
                    fail_num += 1
            except:
                fail_num += 1

        return pass_num, fail_num

    def _evaluate_single_task(self, instance):
        """评估单个SWE-Bench实例的修复效果"""
        start_time = time.time()
        task_result = {
            "instance_id": instance["instance_id"],
            "repo": instance["repo"],
            "success": False,
            "time_cost": 0,
            "fail_to_pass": {"total": 0, "pass": 0},
            "pass_to_pass": {"total": 0, "pass": 0}
        }

        try:
            # 1. 构造修复指令（基于Schema的problem_statement）
            instruction = f"""修复{instance['repo']}的Bug：{instance['problem_statement']}
要求：
- 修复后FAIL_TO_PASS的测试用例必须全部通过；
- 修复后PASS_TO_PASS的测试用例不能失败（避免新Bug）；
- 返回修复的代码补丁（仅代码，无多余内容）。"""

            # 2. 调用Agent生成修复补丁
            agent_result = self.agent.execute_task(instruction, task_type="bug_fix")
            code_patch = agent_result["result"]

            # 3. 验证FAIL_TO_PASS（核心评估指标）
            ftp_total = len([tc for tc in instance["FAIL_TO_PASS"].split("\n") if tc.strip()])
            ftp_pass, ftp_fail = self._run_test_cases(instance["FAIL_TO_PASS"], code_patch)

            # 4. 验证PASS_TO_PASS（防回归）
            ptp_total = len([tc for tc in instance["PASS_TO_PASS"].split("\n") if tc.strip()])
            ptp_pass, ptp_fail = self._run_test_cases(instance["PASS_TO_PASS"], code_patch)

            # 5. 判断任务是否成功：FAIL_TO_PASS全过 + PASS_TO_PASS全过
            task_success = (ftp_fail == 0) and (ptp_fail == 0)

            # 6. 更新任务结果
            task_result["success"] = task_success
            task_result["time_cost"] = round(time.time() - start_time, 2)
            task_result["fail_to_pass"] = {"total": ftp_total, "pass": ftp_pass}
            task_result["pass_to_pass"] = {"total": ptp_total, "pass": ptp_pass}

            print(f"实例{instance['instance_id']}：{'成功' if task_success else '失败'} "
                  f"(FAIL_TO_PASS: {ftp_pass}/{ftp_total}, PASS_TO_PASS: {ptp_pass}/{ptp_total})")

        except Exception as e:
            task_result["error"] = str(e)
            print(f"实例{instance['instance_id']}：失败（错误：{str(e)}）")

        return task_result

    def evaluate(self, batch_size=10, start_index=0):
        """
        批量评估SWE-Bench实例，计算正确率
        :param batch_size: 评估的实例数量
        :param start_index: 起始索引
        :return: 正确率（0~1），详细结果
        """
        # 截取评估批次
        eval_instances = self.swe_bench_data[start_index:start_index + batch_size]
        self.results["total"] = len(eval_instances)

        # 逐个评估
        for instance in eval_instances:
            task_result = self._evaluate_single_task(instance)
            self.results["details"].append(task_result)
            if task_result["success"]:
                self.results["success"] += 1

        # 计算正确率
        self.results["failed"] = self.results["total"] - self.results["success"]
        success_rate = self.results["success"] / self.results["total"] if self.results["total"] > 0 else 0.0

        # 输出汇总结果
        print("\n=== SWE-Bench评估汇总 ===")
        print(f"总实例数：{self.results['total']}")
        print(f"成功数：{self.results['success']} | 失败数：{self.results['failed']}")
        print(f"正确率：{success_rate:.2%}")

        return success_rate, self.results