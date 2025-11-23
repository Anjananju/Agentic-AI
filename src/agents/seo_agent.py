# src/agents/seo_agent.py
class SEOAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, title: str, content: str) -> dict:
        prompt = f"Given this page title: {title} and content: {content[:400]}, generate: meta_title, meta_description (<=160 chars), keywords (comma separated). Return JSON."
        resp = self.llm.complete(prompt, max_tokens=200)
        import json
        try:
            return json.loads(resp)
        except Exception:
            return {
                'meta_title': title,
                'meta_description': (content[:150].replace('\n', ' ') + '...') if content else '',
                'keywords': ', '.join(title.lower().split()[:6])
            }
