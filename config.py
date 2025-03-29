import os
from dataclasses import dataclass

@dataclass
class Config:
    # Existing fields...
    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    MAX_NEW_TOKENS = 16384
    MAX_CONTEXT_LENGTH = 32000
    LOG_PATH = "logs/chat_session.json"
    OFFLOAD_FOLDER = "offload_qwen14b"
    ATTENTION_IMPL = "flash_attention_2"
    TEMPERATURE = 0.6
    TOP_P = 0.95
    SAFE_PATHS = ["./", "/media/zak/"]

    # 🧠 Recursive Generation Config
    MAX_RECURSION_DEPTH = 3
    RECURSION_SIMILARITY_THRESHOLD = 0.75
    ENABLE_RECURSIVE_REFINEMENT = True

    # Latent extraction
    LATENT_LAYER_INDEX = -1
    PADDING_SIDE = "left"
