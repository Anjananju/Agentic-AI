# src/agents/research_agent.py
from tools.webscraper_tool import WebScraperTool

class ResearchAgent:
    def __init__(self, scraper: WebScraperTool = None):
        self.scraper = scraper or WebScraperTool()

    def run(self, prompt: str, references: list = None) -> dict:
        """Returns a research summary and list of citations (simple)."""
        references = references or []
        summaries = []
        for url in references:
            s = self.scraper.summarize(url)
            if s:
                summaries.append({'url': url, 'summary': s})
        # If no references, return empty research (the DraftAgent will still work)
        return {'research_summary': '\n'.join([r['summary'] for r in summaries]), 'citations': summaries}
