from typing import List, Optional, Dict
from .model_loader import load_components
from .config import Config
from .logger import ChatLogger
from .file_handler import FileHandler
from .generator import ResponseGenerator

class CodeAssistAgent:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.chat_history: List[str] = []
        
        # Initialize components
        self.tokenizer, self.model = load_components(config)
        self.logger = ChatLogger(config)
        self.file_handler = FileHandler(config)
        self.generator = ResponseGenerator(
            self.tokenizer, 
            self.model, 
            config
        )

    def process_command(self, raw_input: str) -> Optional[Dict[str, str]]:
        """Handle file/directory loading commands"""
        if raw_input.startswith(("file=", "dir=")):
            target = raw_input.split("=", 1)[1].strip()
            return self._handle_file_or_dir_input(target)
        return None

    def _handle_file_or_dir_input(self, target: str) -> Dict[str, str]:
        """Process both single files and directories"""
        loaded_files = self.file_handler.load_files(target)
        
        if "error" in loaded_files:
            return {"error": loaded_files["error"]}
            
        if not loaded_files:
            return {"error": "No valid files found"}
            
        # Add files to context
        for path, content in loaded_files.items():
            self.chat_history.extend([
                f"User loaded file: {path}",
                f"File content:\n{content}"
            ])
            
        return {
            "success": f"Loaded {len(loaded_files)} files",
            "files": list(loaded_files.keys())
        }
