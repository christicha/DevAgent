class TestWriter:
    def __init__(self, model, tokenizer, model_config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config

    def write_tests(self, instruction):
        """生成单元测试：基于目标代码编写pytest风格的单元测试"""
        prompt = f"""
        你是专业的测试工程师，根据以下需求编写pytest风格的Python单元测试：
        需求：{instruction}
        要求：
        1. 导入必要的模块（如pytest）
        2. 覆盖所有核心功能和边界条件
        3. 测试用例命名规范，包含清晰的注释
        4. 输出完整的可运行测试代码，无需额外解释
        测试代码：
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model_config["device"])
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.model_config["max_new_tokens"],
            temperature=self.model_config["temperature"],
            top_p=self.model_config["top_p"],
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        test_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("测试代码：")[-1].strip()
        # 清理格式
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0].strip()
        return test_code