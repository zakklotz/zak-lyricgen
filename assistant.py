import os
import sys
import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

class CodeAssistAgent:
    def __init__(self, model_id=MODEL_ID):
        try:
            print("[+] Loading model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
                offload_folder="offload_qwen14b",
                attn_implementation="flash_attention_2"
            )
            self.model.eval()
        except Exception as e:
            print(f"[!] Failed to load model: {e}")
            sys.exit(1)
        self.chat_history = []

    def _generate(self, prompt: str, max_new_tokens=1024) -> str:
        try:
            full_prompt = "\n".join(self.chat_history + [f"User: {prompt}", "Assistant:"])
            inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("Assistant:")[-1].strip()
            return assistant_response
        except Exception as e:
            print(f"[!] Inference failed: {e}")
            return ""

    def log_response(self, user_input: str, model_output: str, log_path: str):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "output": model_output
        }
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"[📄] Logged interaction to: {log_path}")

    def chat(self):
        print("\n[🤖] Interactive Code Assistant Chat. Type 'exit' to quit.")
        default_log_path = "logs/chat_session.json"
        while True:
            raw = input("\nYou > ").strip()
            if raw.lower() == "exit":
                break
            prompt = raw
            log_path = default_log_path

            if raw.startswith("file="):
                path = raw.split("file=", 1)[1].strip()
                if os.path.exists(path):
                    with open(path, "r") as f:
                        content = f.read()
                    self.chat_history.append(f"User opened file: {path}")
                    self.chat_history.append(f"Assistant: Here is the file content:{content}")
                    print(f"[📂] Loaded file '{path}' into memory.")
                else:
                    print("[!] File not found.")
                continue

            elif raw.startswith("logfile="):
                log_path = raw.split("logfile=", 1)[1].strip()
                print(f"[→] Log file set to: {log_path}")
                continue

            output = self._generate(prompt)
            if raw.startswith("savefile="):
                path = raw.split("savefile=", 1)[1].strip()
                try:
                    with open(path, "w") as f:
                        f.write(output)
                    print(f"[💾] Output saved to {path}")
                except Exception as e:
                    print(f"[!] Failed to save file: {e}")
            print(f"\nAssistant >\n{output}\n")
            self.chat_history.append(f"User: {prompt}")
            self.chat_history.append(f"Assistant: {output}")
            self.log_response(prompt, output, log_path)


def main():
    agent = CodeAssistAgent()
    agent.chat()


if __name__ == "__main__":
    main()
