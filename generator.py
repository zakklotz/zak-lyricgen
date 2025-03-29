import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
import json
from transformers import PreTrainedTokenizer, PreTrainedModel
from .config import Config
from .logger import print_vram_usage
from typing import List

class ResponseGenerator:
    RECURSION_TEMPLATES = [
        "Extract key themes from: {content}",
        "Add emotional depth to: {content}",
        "Expand into poetic structure: {content}",
        "Refine for lyrical coherence: {content}"
    ]

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, config: Config):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = config.PADDING_SIDE
        self.hidden_state_index = config.LATENT_LAYER_INDEX
        self.use_cuda = torch.cuda.is_available()
        self.max_recursion_depth = config.MAX_RECURSION_DEPTH
        self.refinement_threshold = config.RECURSION_SIMILARITY_THRESHOLD

        print("[ResponseGenerator] Initialized with model:", config.MODEL_ID)

        # Add unloading capability
        self.model.recursive_unload = lambda: (
            self.model.to("cpu"), torch.cuda.empty_cache()
        )

    def generate(self, prompt: str, context: list = []) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
        print_vram_usage("Before generation")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                pad_token_id=self.tokenizer.pad_token_id,
                output_hidden_states=False
            )

        print_vram_usage("After generation")
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_latent_representation(self, prompt: str) -> dict:
        print_vram_usage("Before embedding extraction")
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.config.MAX_CONTEXT_LENGTH).to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states[self.hidden_state_index]
        input_ids = inputs["input_ids"]
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print_vram_usage("After embedding extraction")

        # Save to file
        os.makedirs("output_dumps", exist_ok=True)
        dump = {
            "prompt": prompt,
            "decoded_text": decoded_text,
            "input_ids": input_ids.tolist(),
            "hidden_state_shape": list(hidden_states.shape),
            "timestamp": datetime.now().isoformat()
        }
        path = os.path.join("output_dumps", f"latent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"[💾] Latent dumped to: {path}")

        return {
            "input_ids": input_ids,
            "hidden_state": hidden_states,
            "decoded_text": decoded_text
        }

    def recursive_generate(self, initial_prompt: str) -> str:
        previous_output = initial_prompt
        full_output = ""

        try:
            for step in range(self.max_recursion_depth):
                self.model.to("cuda")

                if step < len(self.RECURSION_TEMPLATES):
                    prompt = self.RECURSION_TEMPLATES[step].format(content=previous_output)
                else:
                    prompt = f"Refine the following:\n{previous_output}"

                new_output = self.generate(prompt, [])
                similarity = self._cosine_similarity(previous_output, new_output)

                print(f"[🌀] Recursion Step {step+1} | Cosine Similarity: {similarity:.4f}")

                latent = self.get_latent_representation(new_output)
                if similarity > self.refinement_threshold or self._is_quality_degrading(latent):
                    print("[⛔] Stopping refinement due to similarity/quality threshold.")
                    break

                full_output += f"\n[Recursion Layer {step+1}]\n{new_output}"
                previous_output = new_output
                self.model.recursive_unload()

        finally:
            self.model.to("cuda")

        return full_output.strip()

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_latent_representation(text1)["hidden_state"].mean(dim=1)
        emb2 = self.get_latent_representation(text2)["hidden_state"].mean(dim=1)
        return F.cosine_similarity(emb1, emb2, dim=1).item()

    def _is_quality_degrading(self, latent: dict) -> bool:
        hs = latent["hidden_state"]
        return (hs.std().item() < 0.1) or (hs.mean().abs().item() > 10)
    
    def embed_theme_block(self, text: str) -> list:
        latent = self.get_latent_representation(text)
        mean_vector = latent["hidden_state"].mean(dim=1).squeeze().tolist()
    
        os.makedirs("output_dumps", exist_ok=True)
        dump = {
            "theme_text": text,
            "embedding": mean_vector,
            "timestamp": datetime.now().isoformat()
        }
        path = os.path.join("output_dumps", f"theme_embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"[💾] Theme embedding saved to: {path}")
    
        return mean_vector

    def build_theme_context(self, theme_text: str) -> str:
        """
        Wrap raw theme text into a structured context block.
        You could later extract structured tags automatically.
        """
        # Simulated parsing – replace with real tag extraction later if desired
        theme_keywords = theme_text.lower()
        emotional_tags = []
        thematic_tags = []
        mood_tags = []

        # Basic keyword extraction
        if "longing" in theme_keywords: emotional_tags.append("Longing")
        if "heartbreak" in theme_keywords: emotional_tags.append("Heartbreak")
        if "nostalgic" in theme_keywords: emotional_tags.append("Nostalgia")
        if "melancholy" in theme_keywords: emotional_tags.append("Melancholy")
        if "futuristic" in theme_keywords: mood_tags.append("Futuristic")
        if "synth" in theme_keywords: mood_tags.append("Synth-heavy")
        if "time travel" in theme_keywords: thematic_tags.append("Time Travel")
        if "parallel" in theme_keywords: thematic_tags.append("Parallel Universes")
        if "cosmic" in theme_keywords: thematic_tags.append("Cosmic Wonder")

        # Format block
        return (
            f"# Theme Summary:\n"
            f"- Emotional Tags: {', '.join(emotional_tags) or 'N/A'}\n"
            f"- Thematic Tags: {', '.join(thematic_tags) or 'N/A'}\n"
            f"- Mood: {', '.join(mood_tags) or 'N/A'}\n\n"
            f"# Task:\n"
        )

    def save_lyrics_json(
        self,
        title: str,
        lyrics: str,
        theme_text: str,
        emotional_tags: List[str],
        thematic_tags: List[str],
        mood_tags: List[str],
        embedding: List[float],
        output_dir: str = "output_dumps"
    ) -> str:
        """
        Saves a lyrics generation result into a structured .json format for downstream use.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"symbolic_output_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
    
        output = {
            "title": title,
            "lyrics": lyrics,
            "theme_text": theme_text,
            "emotional_tags": emotional_tags,
            "thematic_tags": thematic_tags,
            "mood_tags": mood_tags,
            "embedding_vector": embedding,
            "timestamp": datetime.now().isoformat()
        }
    
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
    
        return filepath

    def generate_lyrics_from_themes(self, themes_text: str) -> str:
        print(f"[🎤] Generating lyrics from themes:\n{themes_text}\n")

        recursive_templates = [
            "Write a first draft of poetic song lyrics based on the theme summary and prior content:\n{content}",
            "Refine the lyrics for lyrical flow, coherence, and structure:\n{content}",
            "Add emotional depth and vivid imagery to the lyrics:\n{content}"
        ]

        previous_output = themes_text
        full_output = ""
        context_block = self.build_theme_context(themes_text)

        for step, template in enumerate(recursive_templates):
            self.model.to("cuda")
            prompt = context_block + template.format(content=previous_output)
            new_output = self.generate(prompt, [])
            similarity = self._cosine_similarity(previous_output, new_output)

            print(f"[🎵] Lyrics Refinement Step {step+1} | Cosine Similarity: {similarity:.4f}")
            latent = self.get_latent_representation(new_output)
            if similarity > self.refinement_threshold or self._is_quality_degrading(latent):
                print("[⛔] Stopping lyrics refinement.")
                break

            full_output += f"\n[Lyrics Layer {step+1}]\n{new_output}"
            previous_output = new_output
            self.model.recursive_unload()

        self.model.to("cuda")
        # 🔍 Re-extract the theme embedding vector (based on original theme text)
        theme_vector = self.get_latent_representation(themes_text)["hidden_state"].mean(dim=1).squeeze().tolist()
        
        # Rebuild tags using the same method as the context builder
        emotional_tags, thematic_tags, mood_tags = self.extract_tags_from_text(themes_text)
        
        # Optionally extract title from lyrics
        title = "Untitled"
        for line in full_output.splitlines():
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
                break
        
        # 💾 Save everything to JSON
        output_path = self.save_lyrics_json(
            title=title,
            lyrics=full_output.strip(),
            theme_text=themes_text,
            emotional_tags=emotional_tags,
            thematic_tags=thematic_tags,
            mood_tags=mood_tags,
            embedding=theme_vector
        )
        print(f"[💾] Symbolic output saved to: {output_path}")
        return full_output.strip()
    
    def extract_tags_from_text(self, theme_text: str):
        theme_keywords = theme_text.lower()
        emotional_tags = []
        thematic_tags = []
        mood_tags = []
    
        if "longing" in theme_keywords: emotional_tags.append("Longing")
        if "heartbreak" in theme_keywords: emotional_tags.append("Heartbreak")
        if "nostalgic" in theme_keywords: emotional_tags.append("Nostalgia")
        if "melancholy" in theme_keywords: emotional_tags.append("Melancholy")
        if "futuristic" in theme_keywords: mood_tags.append("Futuristic")
        if "synth" in theme_keywords: mood_tags.append("Synth-heavy")
        if "time travel" in theme_keywords: thematic_tags.append("Time Travel")
        if "parallel" in theme_keywords: thematic_tags.append("Parallel Universes")
        if "cosmic" in theme_keywords: thematic_tags.append("Cosmic Wonder")
    
        return emotional_tags, thematic_tags, mood_tags
    