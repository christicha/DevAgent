# 获取当前脚本（main.py）的目录
import os
import sys

CLI_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（dev_agent/，即cli/的上级目录）
PROJECT_ROOT = os.path.dirname(CLI_DIR)
# 将项目根目录添加到Python搜索路径
sys.path.append(PROJECT_ROOT)

import click
from core.agent import DevAgent
from data.data_loader import DataLoader
from eval.evaluator import SWEBenchEvaluator

# 初始化Agent（全局单例）
agent = None

@click.group()
def cli():
    """AI驱动的软件开发助手（DevAgent）"""
    global agent
    click.echo("初始化DevAgent...")
    agent = DevAgent()
    click.echo("DevAgent初始化完成！")

@cli.command()
@click.option("--task-type", type=click.Choice(["code_generation", "test_writing", "bug_fix"]),
              default="code_generation", help="任务类型")
def demo(task_type):
    """运行示例任务"""
    click.echo(f"运行{task_type}示例任务...")
    sample_task = DataLoader.get_sample_task(task_type)
    result = agent.execute_task(sample_task["instruction"])
    click.echo(f"\n任务类型：{result['task_type']}")
    click.echo(f"执行结果：\n{result['result']}")

@cli.command()
@click.option("--instruction", prompt="请输入开发需求", help="自定义开发需求")
def run(instruction):
    """运行自定义任务"""
    click.echo("正在处理您的需求...")
    result = agent.execute_task(instruction)
    click.echo(f"\n任务类型：{result['task_type']}")
    click.echo(f"执行结果：\n{result['result']}")

@cli.command()
def evaluate():
    """在SWE-Bench上评估DevAgent性能"""
    click.echo("加载SWE-Bench数据集...")
    evaluator = SWEBenchEvaluator(agent)
    success_rate, results = evaluator.evaluate()
    click.echo(f"\n评估完成！")
    click.echo(f"SWE-Bench成功率：{success_rate:.2%}")
    click.echo(f"详细结果：{results}")

if __name__ == "__main__":
    cli()