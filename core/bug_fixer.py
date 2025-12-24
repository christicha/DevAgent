class BugFixer:
    def __init__(self, model, tokenizer, model_config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config

    def fix_bug(self, instruction):
        """修复Bug：定位代码问题并生成修复后的代码"""
        prompt = f"""
        你是专业的软件调试工程师，根据以下需求修复代码中的Bug：
        需求：{instruction}
        要求：
        1. 明确指出问题所在
        2. 生成修复后的完整可运行代码
        3. 添加注释说明修复思路
        4. 输出修复后的代码，无需额外解释
        修复后的代码：
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
        fixed_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("修复后的代码：")[-1].strip()
        # 清理格式
        if "```python" in fixed_code:
            fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
        return fixed_code