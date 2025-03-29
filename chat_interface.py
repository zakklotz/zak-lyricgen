import os
import json
from datetime import datetime

from .chat_agent import CodeAssistAgent
from .config import Config

class ChatInterface:
    def __init__(self, agent: CodeAssistAgent):
        self.agent = agent
        self.chat_history = []
        self.log_path = agent.config.LOG_PATH

    def save_json(self, tag_type, prompt, result_text):
        from datetime import datetime
        import os, json
    
        os.makedirs("output_dumps", exist_ok=True)
        dump = {
            "type": tag_type,
            "prompt": prompt,
            "result": result_text,
            "timestamp": datetime.now().isoformat()
        }
        fname = tag_type + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        path = os.path.join("output_dumps", fname)
        with open(path, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"[💾] Saved to: {path}")

    def start_chat(self):
        print("\n[🤖] Interactive Code Assistant Chat. Type 'exit' to quit.")
        while True:
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                break

            if user_input.startswith("latent="):
                prompt = user_input.split("=", 1)[1].strip()
                result = self.agent.generator.get_latent_representation(prompt)
                print("[LATENT] Decoded Text:", result["decoded_text"])
                print("[LATENT] Input IDs:", result["input_ids"].shape)
                print("[LATENT] Hidden State:", result["hidden_state"].shape)
                # Save to file
                dump = {
                    "type": "latent",
                    "prompt": prompt,
                    "decoded_text": result["decoded_text"],
                    "input_ids": result["input_ids"].tolist(),
                    "hidden_state_shape": list(result["hidden_state"].shape),
                    "timestamp": datetime.now().isoformat()
                }
                os.makedirs("output_dumps", exist_ok=True)
                fname = f"latent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                path = os.path.join("output_dumps", fname)
                with open(path, "w") as f:
                    json.dump(dump, f, indent=2)
                print(f"[💾] Latent dumped to: {path}")
                continue

            if user_input.startswith("lyrics="):
                raw_lyrics = user_input.split("=", 1)[1].strip()

                full_prompt = f"""
You are a music analysis assistant. Analyze the following lyrics and return:
1. The lyrics as structured lines
2. Emotional tags
3. Thematic tags
4. Any useful metadata (genre cues, imagery, tempo suggestion, etc.)

Lyrics:
{raw_lyrics}
"""
                output = self._generate_response(full_prompt)
                print(f"\n[🎰 Lyrics Analysis + Tags]\n{output}")
                self.save_json("lyrics_analysis", raw_lyrics, output)
                continue

            if user_input.startswith("emotion="):
                prompt = user_input.split("=", 1)[1].strip()
                full_prompt = f"What are the primary emotional and thematic tags of this prompt?\n{prompt}"
                output = self._generate_response(full_prompt)
                print(f"\n[🎭 Emotion Tags]\n{output}")
                self.save_json("emotion", prompt, output)
                continue

            if user_input.startswith("recursive="):
                prompt = user_input.split("=", 1)[1].strip()
                print(f"[🌀] Starting recursive refinement on:\n{prompt}\n")
            
                output = self.agent.generator.recursive_generate(prompt)
            
                print(f"\n[🧠 Recursive Output]\n{output}")
                self.save_json("recursive", prompt, output)
                continue

            if user_input.startswith("lyrics_from_themes="):
                theme_text = user_input.split("=", 1)[1].strip()
            
                print(f"[🎤] Starting recursive lyrics generation from themes:\n{theme_text}")
                lyrics_output = self.agent.generator.generate_lyrics_from_themes(theme_text)
                embedding = self.agent.generator.embed_theme_block(theme_text)
            
                # Save full output
                from datetime import datetime
                import os, json
                os.makedirs("output_dumps", exist_ok=True)
                payload = {
                    "input_theme": theme_text,
                    "lyrics_output": lyrics_output,
                    "theme_embedding": embedding,
                    "timestamp": datetime.now().isoformat()
                }
                fname = f"lyrics_from_themes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                path = os.path.join("output_dumps", fname)
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"\n[🎼 Recursive Lyrics Generated]\n{lyrics_output}")
                print(f"[💾] Saved to: {path}")
                continue

            command_result = self.agent.process_command(user_input)
            if command_result:
                continue

            output = self._generate_response(user_input)
            self._handle_output(user_input, output)

    def _generate_response(self, user_input: str) -> str:
        return self.agent.generator.generate(
            user_input, 
            self.agent.chat_history
        )

    def _handle_output(self, user_input: str, output: str):
        print(f"\nAssistant >\n{output}\n")
        self.agent.chat_history.extend([
            f"User: {user_input}",
            f"Assistant: {output}"
        ])
        self.agent.logger.log_interaction(user_input, output)
