import os
import re
from typing import Optional, Tuple
from .config import Config

class FileHandler:
    def __init__(self, config: Config):
        self.config = config
        self.safe_paths = config.SAFE_PATHS
        
    def validate_path(self, path: str) -> bool:
        normalized = os.path.normpath(path)
        return any(normalized.startswith(p) for p in self.safe_paths)
    
    def read_file(self, path: str) -> Optional[str]:
        try:
            if not self.validate_path(path):
                return None
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            print(f"[!] File read error: {e}")
            return None

    def save_output(self, output: str, base_path: str) -> str:
        try:
            base, ext = os.path.splitext(base_path)
            version = 1
            while True:
                new_path = f"{base}_v{version}{ext}"
                if not os.path.exists(new_path):
                    with open(new_path, "w") as f:
                        f.write(output)
                    return new_path
                version += 1
        except Exception as e:
            print(f"[!] File save error: {e}")
            return ""
    
    def load_files(self, path: str) -> dict:
        if not os.path.exists(path):
            return {"error": "Path not found."}
        if os.path.isfile(path):
            content = self.read_file(path)
            return {path: content} if content else {}
        elif os.path.isdir(path):
            results = {}
            for root, _, files in os.walk(path):
                for name in files:
                    full_path = os.path.join(root, name)
                    content = self.read_file(full_path)
                    if content:
                        results[full_path] = content
            return results
        return {"error": "Unsupported path type."}
