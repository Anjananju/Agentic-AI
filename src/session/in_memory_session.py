# src/session/in_memory_session.py
import threading

class InMemorySessionService:
    def __init__(self):
        self._lock = threading.Lock()
        self._store = {}

    def create_session(self, session_id: str, data: dict):
        with self._lock:
            self._store[session_id] = data.copy()

    def update_session(self, session_id: str, data: dict):
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}
            self._store[session_id].update(data)

    def get_session(self, session_id: str):
        with self._lock:
            return self._store.get(session_id)

    def list_sessions(self):
        with self._lock:
            return list(self._store.keys())
