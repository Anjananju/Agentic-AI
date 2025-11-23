# src/llm_provider.py

import os
import json
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextGenerationPipeline
)
import torch


class LLMProvider:
    """Base class for any LLM backend."""

    def __init__(self, model_name: Optional[str] = None, api_key_env: Optional[str] = None):
        self.model_name = model_name
        self.api_key = os.getenv(api_key_env) if api_key_env else None

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Override in subclasses."""
        raise NotImplementedError



class HuggingFaceProvider(LLMProvider):
    """
    HuggingFace Inference Provider.
    Works for Gemma, Llama, Phi-3, Mistral, etc.
    Uses local models from Transformers.
    """

    def __init__(self, model_name: str = "google/gemma-2b-it"):
        super().__init__(model_name=model_name)

        # Load tokenizer & model safely (GPU or CPU)
        try:
            print(f"[HF Provider] Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto"
            )

            self.pipeline: TextGenerationPipeline = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                pad_token_id=self.tokenizer.eos_token_id
            )

        except Exception as e:
            print(f"[HF Provider] Failed to load model. Using mock mode. Error: {e}")
            self.pipeline = None



    def _format_gemma_prompt(self, user_prompt: str) -> str:
        """
        Gemma uses a chat-style template.
        This dramatically improves performance and prevents prompt repetition.
        """
        return (
            f"<start_of_turn>user\n"
            f"{user_prompt}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )



    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Generate text using Gemma or fallback."""
        if not self.pipeline:
            return f"[MOCK OUTPUT] {prompt[:200]}..."

        # Apply Gemma chat template
        formatted_prompt = self._format_gemma_prompt(prompt)

        try:
            output = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id
            )

            text = output[0]["generated_text"]

            # Remove Gemma tags
            text = text.replace("<start_of_turn>model", "")
            text = text.replace("<end_of_turn>", "")
            text = text.replace("<start_of_turn>user", "")

            return text.strip()

        except Exception as e:
            return f"[HF ERROR] {e}\nPROMPT: {prompt[:200]}"



class HTTPProvider(LLMProvider):
    """For custom REST API LLMs."""

    def __init__(self, endpoint: str, api_key_env: Optional[str] = None):
        super().__init__(model_name=None, api_key_env=api_key_env)
        self.endpoint = endpoint

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        import requests

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
            return data.get("text") or json.dumps(data)

        except Exception as e:
            return f"[HTTP ERROR] {e}"
