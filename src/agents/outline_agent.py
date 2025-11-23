# src/agents/outline_agent.py
from typing import List

class OutlineAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, topic: str, audience: str, length: str = 'medium') -> List[dict]:
        prompt = (
            f"Create a blog post outline for the topic: '{topic}' targeting '{audience}'. "
            "Return 5 sections as JSON array where each section has 'heading' and 'bullets'."
        )
        resp = self.llm.complete(prompt, max_tokens=400)
        # Try to parse JSON in the response; fallback to naive split
        import json
        try:
            # If model gives raw JSON
            parsed = json.loads(resp)
            return parsed
        except Exception:
            # Heuristic: split by lines
            lines = [l.strip('- ') for l in resp.split('\n') if l.strip()]
            sections = []
            for i, line in enumerate(lines[:5]):
                sections.append({'heading': line[:120], 'bullets': ['Point 1', 'Point 2']})
            # If nothing usable, create generic sections
            if not sections:
                sections = [
                    {'heading': 'Introduction', 'bullets': ['What this article covers']},
                    {'heading': 'Background', 'bullets': ['Context and definitions']},
                    {'heading': 'Main Points', 'bullets': ['Key idea 1', 'Key idea 2']},
                    {'heading': 'Examples', 'bullets': ['Example 1', 'Example 2']},
                    {'heading': 'Conclusion', 'bullets': ['Summary and next steps']},
                ]
            return sections
