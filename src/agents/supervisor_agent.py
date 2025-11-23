# src/agents/supervisor_agent.py
import uuid
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

        # lazy-import agents to avoid circular issues
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

    def start_job(self, topic: str, audience: str, references: list = None) -> dict:
        job_id = str(uuid.uuid4())
        self.session_svc.create_session(job_id, {'status': 'running', 'topic': topic})
        log.info(f"Starting job {job_id} for topic={topic}")

        # Step 1: research (may be empty if no refs)
        research = self.research_agent.run(topic, references)
        self.session_svc.update_session(job_id, {'research': research})

        # Step 2: outline
        outline = self.outline_agent.run(topic, audience)
        self.session_svc.update_session(job_id, {'outline': outline})

        # Step 3: draft sections in parallel
        futures = []
        sections = []
        for s in outline:
            heading = s.get('heading', 'Untitled')
            bullets = s.get('bullets', [])
            futures.append(self.executor.submit(self._draft_and_edit, heading, bullets))

        for f in futures:
            sections.append(f.result())

        # Step 4: assemble
        full_content = '\n\n'.join([f"## {sec['heading']}\n\n{sec['content']}" for sec in sections])

        # Step 5: SEO
        seo = self.seo_agent.run(topic, full_content)

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

    def _draft_and_edit(self, heading, bullets):
        # Draft
        draft = self.draft_agent.expand(heading, bullets)
        # Edit
        edited = self.editor_agent.run(draft)
        return {'heading': heading, 'content': edited}

    def pause_job(self, job_id: str):
        self.session_svc.update_session(job_id, {'status': 'paused'})
        log.info(f"Paused job {job_id}")

    def resume_job(self, job_id: str) -> dict:
        sess = self.session_svc.get_session(job_id)
        if not sess:
            raise KeyError('job not found')
        status = sess.get('status')
        if status != 'paused':
            return sess
        # naive resume: just mark running again (real resume should restore pipeline state)
        self.session_svc.update_session(job_id, {'status': 'running'})
        log.info(f"Resumed job {job_id}")
        return sess
