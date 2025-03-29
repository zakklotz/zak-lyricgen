import torch

def print_vram_usage(label=""):
    used = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"[🧠 VRAM] {label} | Used: {used:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB")
