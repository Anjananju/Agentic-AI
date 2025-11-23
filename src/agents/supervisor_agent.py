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

        # lazy import agents
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

    # ================================
    # JOB ENTRYPOINT
    # ================================
    def start_job(self, topic: str, audience: str, references: list = None) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        self.session_svc.create_session(job_id, {'status': 'running', 'topic': topic})
        log.info(f"Starting job {job_id} for topic={topic}")

        # Step 1: Research
        research = self.research_agent.run(topic, references)
        self.session_svc.update_session(job_id, {'research': research})

        # Step 2: Outline
        outline = self.outline_agent.run(topic, audience)
        self.session_svc.update_session(job_id, {'outline': outline})

        # Step 3: Generate all sections (fast batch mode)
        sections = self.generate_sections_batch(outline)

        # Step 4: Assemble full content
        full_content = "\n\n".join([
            f"## {sec['heading']}\n\n{sec['content']}"
            for sec in sections
        ])

        # Step 5: SEO optimization
        seo = self.seo_agent.run(topic, full_content)

        # Final result
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

    # ================================
    # NEW: FAST BATCH SECTION GENERATION
    # ================================
    def generate_sections_batch(self, outline: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Single LLM call → generates all drafts.
        Single LLM call → edits all drafts.
        Much faster + avoids prompt leak.
        """
        prompt = (
            "You are an expert content generator.\n"
            "Generate a JSON array of blog sections.\n\n"
            "Each section must have this structure:\n"
            "{ \"heading\": \"...\", \"content\": \"...\" }\n\n"
            "Generate detailed sections based on this outline:\n"
            f"{json.dumps(outline, indent=2)}\n\n"
            "Return ONLY valid JSON. No commentary."
        )

        raw = self.llm.generate(prompt)

        try:
            sections = json.loads(raw)
            if isinstance(sections, list):
                pass
            else:
                raise ValueError("Not a list")
        except Exception:
            log.warning("Batch JSON failed. Falling back to per-section mode.")
            return self.generate_sections_fallback(outline)

        # EDIT ALL SECTIONS IN ONE CALL
        edit_prompt = (
            "You are an expert editor.\n"
            "Improve clarity, grammar, structure.\n"
            "Return ONLY JSON of this form:\n"
            "[ {\"heading\": \"...\", \"content\": \"...\"}, ... ]\n\n"
            "EDIT THESE SECTIONS:\n"
            f"{json.dumps(sections, indent=2)}\n"
        )

        edited_raw = self.llm.generate(edit_prompt)

        try:
            edited_sections = json.loads(edited_raw)
            return edited_sections
        except Exception:
            log.warning("Batch edit failed, returning unedited sections.")
            return sections

    # ================================
    # FALLBACK (OLD MODE)
    # ================================
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

    # OLD draft + edit
    def _draft_and_edit(self, heading, bullets):
        draft = self.draft_agent.expand(heading, bullets)
        edited = self.editor_agent.run(draft)
        return {'heading': heading, 'content': edited}

    # Pause & Resume
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
            log.info(f"Batch generation failed ({e}), falling back to per-section generation")
            sections = self._generate_sections_fallback(outline)

        # Step 4: Edit all sections in a single LLM call (preferred)
        try:
            edited_sections = self._edit_all_sections(sections)
            if not isinstance(edited_sections, list) or len(edited_sections) == 0:
                raise ValueError("Invalid edited sections")
        except Exception as e:
            log.info(f"Batch editing failed ({e}), falling back to per-section editing")
            edited_sections = self._edit_sections_fallback(sections)

        # Step 5: assemble
        full_content = '\n\n'.join([f"## {sec.get('heading','Untitled')}\n\n{sec.get('content','')}" for sec in edited_sections])

        # Step 6: SEO
        seo = self.seo_agent.run(topic, full_content)

        result = {
            'job_id': job_id,
            'topic': topic,
            'outline': outline,
            'sections': edited_sections,
            'content': full_content,
            'seo': seo,
            'research': research
        }
        self.session_svc.update_session(job_id, {'status': 'completed', 'result': result})
        log.info(f"Completed job {job_id}")
        return result

    # ---------------------------
    # Batch generation utilities
    # ---------------------------
    def _generate_all_sections_from_outline(self, outline: List[Dict[str, Any]], topic: str, audience: str) -> List[Dict[str, Any]]:
        """
        Ask the LLM to expand the entire outline into a JSON array of sections.
        The LLM is asked to return strict JSON like:
        [{"heading":"...","content":"..."}, ...]
        """
        # Prepare a compact outline JSON to include in the prompt
        safe_outline = [{'heading': s.get('heading', f'Section {i+1}'), 'bullets': s.get('bullets', [])} for i, s in enumerate(outline)]
        outline_json = json.dumps(safe_outline, ensure_ascii=False)

        draft_prompt = (
            f"Expand the following article outline into full sections for a blog post.\n\n"
            f"Topic: {topic}\nAudience: {audience}\n\n"
            f"Outline (JSON):\n{outline_json}\n\n"
            "For each entry in the outline, produce a section with keys 'heading' and 'content'. "
            "Write each section as roughly 150-250 words. Use the bullets to guide content. "
            "Return ONLY valid JSON: an array of objects like [{\"heading\":\"...\",\"content\":\"...\"}, ...]."
        )

        raw = self.llm.complete(draft_prompt, max_tokens=2000, temperature=0.2)

        sections = self._extract_json_array_from_text(raw)
        if sections is None:
            # Try a second attempt with a tighter instruction to ensure JSON only
            draft_prompt2 = draft_prompt + "\nIMPORTANT: Output must be valid JSON only — nothing else."
            raw2 = self.llm.complete(draft_prompt2, max_tokens=2000, temperature=0.1)
            sections = self._extract_json_array_from_text(raw2)

        if sections is None:
            raise RuntimeError("Could not parse JSON sections from model output")

        # Normalize sections to expected shape
        normalized = []
        for s in sections:
            heading = s.get('heading') or s.get('title') or 'Untitled'
            content = s.get('content') or s.get('body') or ''
            normalized.append({'heading': heading, 'content': content})
        return normalized

    def _edit_all_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Edit all sections in one LLM call. Input: JSON array of sections.
        Output: same JSON array but with edited 'content' fields.
        """
        sections_json = json.dumps(sections, ensure_ascii=False)
        edit_prompt = (
            "Edit the following JSON array of article sections for clarity, grammar, and conciseness.\n\n"
            "Input JSON:\n" + sections_json + "\n\n"
            "Return ONLY valid JSON with the same structure: [{\"heading\":\"...\",\"content\":\"...\"}, ...]."
        )

        raw = self.llm.complete(edit_prompt, max_tokens=2000, temperature=0.0)
        edited = self._extract_json_array_from_text(raw)
        if edited is None:
            # fallback: attempt one more time
            raw2 = self.llm.complete(edit_prompt + "\nIMPORTANT: return valid JSON only.", max_tokens=2000, temperature=0.0)
            edited = self._extract_json_array_from_text(raw2)

        if edited is None:
            raise RuntimeError("Could not parse edited sections JSON")

        normalized = []
        for s in edited:
            heading = s.get('heading') or s.get('title') or 'Untitled'
            content = s.get('content') or s.get('body') or ''
            normalized.append({'heading': heading, 'content': content})
        return normalized

    # ---------------------------
    # Fallback per-section methods
    # ---------------------------
    def _generate_sections_fallback(self, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback that uses the existing DraftAgent.expand per section (sequentially).
        """
        sections = []
        for s in outline:
            heading = s.get('heading', 'Untitled')
            bullets = s.get('bullets', [])
            draft = self.draft_agent.expand(heading, bullets)
            sections.append({'heading': heading, 'content': draft})
        return sections

    def _edit_sections_fallback(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback that uses EditorAgent.run per section (sequentially).
        """
        edited = []
        for s in sections:
            heading = s.get('heading', 'Untitled')
            content = s.get('content', '')
            out = self.editor_agent.run(content)
            edited.append({'heading': heading, 'content': out})
        return edited

    # ---------------------------
    # Utility: robust JSON extraction
    # ---------------------------
    def _extract_json_array_from_text(self, text: str):
        """
        Attempt to extract a JSON array from model output.
        Returns Python list if success, otherwise None.
        """
        if not text or not isinstance(text, str):
            return None

        text = text.strip()

        # Quick attempt: direct JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            # If top-level dict with 'sections' key
            if isinstance(parsed, dict) and 'sections' in parsed and isinstance(parsed['sections'], list):
                return parsed['sections']
        except Exception:
            pass

        # Try to locate first [...] JSON array in text
        array_match = re.search(r'(\[\\s*{.*}\\s*\])', text, flags=re.DOTALL)
        if array_match:
            candidate = array_match.group(1)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        # More lenient: find the first '[' and last ']' and try to parse
        try:
            first = text.index('[')
            last = text.rindex(']')
            candidate = text[first:last+1]
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # If still nothing, return None
        return None

    # ---------------------------
    # Pause / Resume
    # ---------------------------
    def pause_job(self, job_id: str):
        self.session_svc.update_session(job_id, {'status': 'paused'})
        log.info(f"Paused job {job_id}")

    def resume_job(self, job_id: str) -> Dict[str, Any]:
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
