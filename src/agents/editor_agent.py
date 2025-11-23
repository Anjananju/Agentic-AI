# src/agents/editor_agent.py
class EditorAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, draft_text: str) -> str:
        prompt = (f"Edit the following blog section for clarity, grammar, and conciseness.\n\n{draft_text}\n\nProvide only the edited section.")
        return self.llm.complete(prompt, max_tokens=400)
