# src/llm_provider.py
import os
import json
from typing import Optional

# Default provider interface
class LLMProvider:
    def __init__(self, model_name: Optional[str]=None, api_key_env: Optional[str]=None):
        self.model_name = model_name or os.getenv('HF_MODEL', 'mistral-small')
        self.api_key = os.getenv(api_key_env) if api_key_env else None

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        raise NotImplementedError

# HuggingFace provider (uses transformers pipeline if available)
class HuggingFaceProvider(LLMProvider):
    def __init__(self, model_name: str = None):
        super().__init__(model_name=model_name)
        # Lazy import to avoid heavy dependency if not used
        try:
            from transformers import pipeline
            # default small model name if none provided
            model = self.model_name or 'mistral-small'
            self.pipeline = pipeline('text-generation', model=model, device_map="auto")
        except Exception:
            self.pipeline = None

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        if self.pipeline:
            out = self.pipeline(prompt, max_length=max_tokens, do_sample=False)
            if isinstance(out, list) and out:
                return out[0].get('generated_text', '')
            return str(out)
        # Fallback mock response for offline/demo
        return f"[MOCK-HF] Generated text for prompt: {prompt[:200]}..."

# Simple HTTP JSON provider for custom endpoints
class HTTPProvider(LLMProvider):
    def __init__(self, endpoint: str, api_key_env: Optional[str]=None):
        super().__init__(model_name=None, api_key_env=api_key_env)
        self.endpoint = endpoint

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        import requests
        payload = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            # Expecting {'text': '...'} or similar
            return data.get('text') or json.dumps(data)
        except Exception as e:
            return f"[MOCK-HTTP-ERROR] {e}"
