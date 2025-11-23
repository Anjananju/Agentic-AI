# src/agents/supervisor_agent.py
import uuid
import json
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from observability.logger import get_logger
from session.in_memory_session import InMemorySessionService
from memory.memory_bank import MemoryBank

log = get_logger('supervisor')


class SupervisorAgent:
    def __init__(self, llm_provider, workers: int = 4):
        self.llm = llm_provider
        self.session_svc = InMemorySessionService()
        self.memory = MemoryBank()
        self.workers = workers
        self.executor = ThreadPoolExecutor(max_workers=workers)

        # Lazy import
        from agents.outline_agent import OutlineAgent
        from agents.draft_agent import DraftAgent
        from agents.editor_agent import EditorAgent
        from agents.research_agent import ResearchAgent
        from agents.seo_agent import SEOAgent

        self.research_agent = ResearchAgent()
        self.outline_agent = OutlineAgent(self.llm)
        self.draft_agent = DraftAgent(self.llm)
        self.editor_agent = EditorAgent(self.llm)
        self.seo_agent = SEOAgent(self.llm)

    # ----------------------------------------------------
    # JOB ENTRY
    # ----------------------------------------------------
    def start_job(self, topic: str, audience: str, references: list = None) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        self.session_svc.create_session(job_id, {'status': 'running', 'topic': topic})
        log.info(f"Starting job {job_id} for topic={topic}")

        # 1. Research
        research = self.research_agent.run(topic, references)
        self.session_svc.update_session(job_id, {'research': research})

        # 2. Outline
        outline = self.outline_agent.run(topic, audience)
        self.session_svc.update_session(job_id, {'outline': outline})

        # 3. Batch generate sections
        sections = self.generate_sections_batch(outline)

        # 4. Assemble
        full_content = "\n\n".join([
            f"## {sec['heading']}\n\n{sec['content']}"
            for sec in sections
        ])

        # 5. SEO optimization
        seo = self.seo_agent.run(topic, full_content)

        # Result
        result = {
            'job_id': job_id,
            'topic': topic,
            'outline': outline,
            'sections': sections,
            'content': full_content,
            'seo': seo,
            'research': research
        }

        self.session_svc.update_session(job_id, {'status': 'completed', 'result': result})
        log.info(f"Completed job {job_id}")
        return result

    # ----------------------------------------------------
    # FAST BATCH GENERATION
    # ----------------------------------------------------
    def generate_sections_batch(self, outline: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        prompt = (
            "You are an expert writer.\n"
            "Generate a JSON array of sections with:\n"
            "{ \"heading\": \"...\", \"content\": \"...\" }\n\n"
            f"Outline:\n{json.dumps(outline, indent=2)}\n\n"
            "Return ONLY valid JSON."
        )

        raw = self.llm.generate(prompt)

        # Parse
        try:
            sections = json.loads(raw)
            if not isinstance(sections, list):
                raise ValueError("Expected JSON list")
        except Exception:
            return self.generate_sections_fallback(outline)

        # EDIT IN ONE CALL
        edit_prompt = (
            "Improve clarity, grammar, structure.\n"
            "Return only JSON.\n\n"
            f"SECTIONS:\n{json.dumps(sections, indent=2)}"
        )

        edited_raw = self.llm.generate(edit_prompt)

        try:
            return json.loads(edited_raw)
        except Exception:
            return sections

    # ----------------------------------------------------
    # FALLBACK: PER-SECTION
    # ----------------------------------------------------
    def generate_sections_fallback(self, outline: List[Dict[str, Any]]):
        futures = []
        sections = []
        for s in outline:
            futures.append(self.executor.submit(
                self._draft_and_edit,
                s["heading"],
                s.get("bullets", [])
            ))

        for f in futures:
            sections.append(f.result())

        return sections

    def _draft_and_edit(self, heading, bullets):
        draft = self.draft_agent.expand(heading, bullets)
        edited = self.editor_agent.run(draft)
        return {'heading': heading, 'content': edited}

    # ----------------------------------------------------
    # PAUSE / RESUME
    # ----------------------------------------------------
    def pause_job(self, job_id: str):
        self.session_svc.update_session(job_id, {"status": "paused"})
        log.info(f"Paused job {job_id}")

    def resume_job(self, job_id: str) -> Dict[str, Any]:
        sess = self.session_svc.get_session(job_id)
        if not sess:
            raise KeyError("job not found")

        if sess.get("status") != "paused":
            return sess

        self.session_svc.update_session(job_id, {"status": "running"})
        log.info(f"Resumed job {job_id}")
        return sess
