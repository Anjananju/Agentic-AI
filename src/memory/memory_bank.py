# src/memory/memory_bank.py
import json
import os
from typing import Optional

class MemoryBank:
    def __init__(self, path: str = 'memory_bank.json'):
        self.path = path
        self._data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    def set_user_profile(self, user_id: str, profile: dict):
        self._data[user_id] = profile
        self._save()

    def get_user_profile(self, user_id: str) -> Optional[dict]:
        return self._data.get(user_id)
