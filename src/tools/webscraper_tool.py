# src/tools/webscraper_tool.py
import requests
from bs4 import BeautifulSoup

class WebScraperTool:
    def __init__(self, user_agent: str = 'agent/1.0'):
        self.headers = {'User-Agent': user_agent}

    def fetch_text(self, url: str, max_chars: int = 3000) -> str:
        try:
            r = requests.get(url, headers=self.headers, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            texts = soup.find_all('p')
            joined = ' '.join([t.get_text(strip=True) for t in texts])
            return joined[:max_chars]
        except Exception:
            return ''

    def summarize(self, url: str) -> str:
        text = self.fetch_text(url)
        if not text:
            return ''
        # Very simple heuristic summary: first 3 sentences
        sentences = text.split('.')
        return '.'.join(sentences[:3]).strip() + ('.' if len(sentences) >= 1 else '')
