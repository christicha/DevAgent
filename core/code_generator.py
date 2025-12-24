class CodeGenerator:
    def __init__(self, model, tokenizer, model_config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config

    def generate_code(self, instruction):
        """生成代码：基于需求生成符合规范的代码"""
        prompt = f"""
        你是专业的Python开发工程师，根据以下需求编写高质量、可运行、鲁棒的代码：
        需求：{instruction}
        要求：
        1. 代码格式规范，包含必要的注释
        2. 处理边界条件和异常
        3. 输出完整的可运行代码，无需额外解释
        代码：
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
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("代码：")[-1].strip()
        # 清理代码格式（提取```python包裹的内容）
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        return code