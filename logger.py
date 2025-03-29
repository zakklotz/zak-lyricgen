import os
import json
from datetime import datetime
from .config import Config
import torch

class ChatLogger:
    def __init__(self, config: Config):
        self.path = config.LOG_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log_interaction(self, user_input: str, model_output: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "output": model_output
        }
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        logs.append(entry)
        with open(self.path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"[📄] Logged to: {self.path}")

def print_vram_usage(context=""):
    if not torch.cuda.is_available():
        print(f"[🧠 CPU Mode] No GPU available for VRAM usage logging.")
        return

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[🧠 VRAM] {context} | Used: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB")
