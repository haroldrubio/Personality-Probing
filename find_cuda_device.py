import torch

print('-'*20)
print(f"cuda available: {torch.cuda.is_available()}")

print(f"cuda address: {torch.cuda.device(0)}")

print(f"cuda device count: {torch.cuda.device_count()}")

print(f"cuda device name: {torch.cuda.get_device_name(0)}")
print('-'*20)