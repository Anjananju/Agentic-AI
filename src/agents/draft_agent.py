# src/agents/draft_agent.py
from typing import List

class DraftAgent:
    def __init__(self, llm):
        self.llm = llm

    def expand(self, heading: str, bullets: List[str], target_words: int = 300) -> str:
        prompt = (
            f"Write a {target_words}-word section with heading: '{heading}'. Use these bullet points: {bullets}. "
            "Write clear, human-friendly prose suitable for a blog post."
        )
        return self.llm.complete(prompt, max_tokens=target_words+200)
