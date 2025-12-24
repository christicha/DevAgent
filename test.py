import torch
# 检查PyTorch版本
print(torch.__version__)
# 检查CUDA是否可用
print(torch.cuda.is_available())
# 检查CUDA版本（应显示11.7）
print(torch.version.cuda)
# 检查显卡是否被识别
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # 输出显卡名称，如NVIDIA GeForce RTX 3060