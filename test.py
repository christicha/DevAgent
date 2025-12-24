import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_CONFIG

# 测试模型加载
try:
    print("开始加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
    print("Tokenizer加载成功！")

    print("开始加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_name"],
        torch_dtype=torch.float16 if MODEL_CONFIG["device"] == "cuda" else torch.float32,
        device_map="auto"
    )
    print("模型加载成功！")

    # 测试简单生成
    prompt = "编写一个Python函数，计算1到n的和"
    inputs = tokenizer(prompt, return_tensors="pt").to(MODEL_CONFIG["device"])
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("测试生成结果：")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
except Exception as e:
    print(f"加载失败！错误：{e}")