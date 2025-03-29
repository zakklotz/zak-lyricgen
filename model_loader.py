import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .config import Config

def load_components(config: Config):
    try:
        print("[+] Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_ID, trust_remote_code=True, use_fast=True
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            offload_folder=config.OFFLOAD_FOLDER,
            attn_implementation=config.ATTENTION_IMPL
        )
        model.eval()
        return tokenizer, model

    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        raise
