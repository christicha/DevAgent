import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_CONFIG
from core.code_generator import CodeGenerator
from core.test_writer import TestWriter
from core.bug_fixer import BugFixer


class DevAgent:
    def __init__(self):
        # 初始化模型和Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["model_name"],
            torch_dtype=torch.float16 if MODEL_CONFIG["device"] == "cuda" else torch.float32,
            device_map="auto"
        )
        # 初始化功能模块
        self.code_generator = CodeGenerator(self.model, self.tokenizer, MODEL_CONFIG)
        self.test_writer = TestWriter(self.model, self.tokenizer, MODEL_CONFIG)
        self.bug_fixer = BugFixer(self.model, self.tokenizer, MODEL_CONFIG)

    def understand_requirement(self, instruction):
        """需求理解：解析用户输入，判断任务类型"""
        prompt = f"""
        分析以下软件开发需求，判断任务类型（仅返回code_generation/test_writing/bug_fix中的一个）：
        需求：{instruction}
        任务类型：
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(MODEL_CONFIG["device"])
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.1,
            top_p=0.9,
            do_sample=False
        )
        task_type = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # 兜底处理
        if task_type not in ["code_generation", "test_writing", "bug_fix"]:
            task_type = "code_generation"
        return task_type

    def execute_task(self, instruction):
        """执行任务：根据需求理解结果调度对应模块"""
        task_type = self.understand_requirement(instruction)
        print(f"[DevAgent] 识别任务类型：{task_type}")

        if task_type == "code_generation":
            result = self.code_generator.generate_code(instruction)
        elif task_type == "test_writing":
            result = self.test_writer.write_tests(instruction)
        elif task_type == "bug_fix":
            result = self.bug_fixer.fix_bug(instruction)
        else:
            result = "无法识别的任务类型"

        return {
            "task_type": task_type,
            "result": result
        }