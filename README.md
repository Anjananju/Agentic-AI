Automated Blog Writer — Multi-Agent LLM System
Kaggle AI Agents Capstone Project Submission
This project implements a multi-agent AI system that generates fully structured, edited, SEO-optimized blog articles from a topic prompt.
It showcases all required AI Agent Capstone features:
 Multi-agent orchestration
— Research Agent
— Outline Agent
— Draft Agent
— Editor Agent
— SEO Agent
— Supervisor Agent (Coordinator + Parallel Execution)
Tool usage
— Custom WebScraperTool for URL-based content extraction
— Optional custom HTTP LLM endpoint
— Built-in LLM provider abstraction (HuggingFace / Custom APIs)
Session & Memory
— InMemorySessionService
— Persistent MemoryBank (JSON-backed)
Observability
— Structured logging
— Execution trace by job_id
— Supports pause/resume

Project Overview
The Automated Blog Writer Agent takes:
A topic
A target audience
Optional reference URLs
Then creates a complete blog article through a multi-step LLM pipeline:
Agent Pipeline
ResearchAgent → Fetches & summarizes reference URLs
OutlineAgent → Builds a JSON-based article outline
DraftAgent → Expands all sections in parallel
EditorAgent → Cleans & rewrites each section
SEOAgent → Produces SEO metadata
SupervisorAgent → Runs, monitors, pauses, resumes the job

Architecture Diagram
User Input
   │
   ▼
SupervisorAgent
   │
   ├── ResearchAgent  →  WebScraperTool
   ├── OutlineAgent
   ├── DraftAgent  (Parallel threads)
   ├── EditorAgent
   └── SEOAgent
   │
   ▼
 Final Article (Markdown + SEO)


 Repository Structure
src/
│
├── llm_provider.py
├── demo_app.py
│
├── agents/
│   ├── supervisor_agent.py
│   ├── research_agent.py
│   ├── outline_agent.py
│   ├── draft_agent.py
│   ├── editor_agent.py
│   └── seo_agent.py
│
├── tools/
│   └── webscraper_tool.py
│
├── session/
│   └── in_memory_session.py
│
├── memory/
│   └── memory_bank.py
│
└── observability/
    └── logger.py

requirements.txt
README.md


🛠️ Installation
Local Setup
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
Run the demo UI:
streamlit run src/demo_app.py


🧩 Running the Agent
Here is a minimal example:
from llm_provider import HuggingFaceProvider
from agents.supervisor_agent import SupervisorAgent

llm = HuggingFaceProvider(model_name="gpt2")  # or your own HF model
agent = SupervisorAgent(llm)

result = agent.start_job("Future of AI Agents", "Developers")
print(result["content"])


🤖 HuggingFace / Custom LLM Support
You can use:
HuggingFace local models
HuggingFace Inference API
Custom HTTP LLM endpoint
Any other model by creating a new provider class
Example (HTTP endpoint):
from llm_provider import HTTPProvider
llm = HTTPProvider(endpoint="https://your-model-endpoint")


🧪 Parallelism
The system uses a ThreadPoolExecutor to draft sections simultaneously:
DraftAgent.expand(...)  ← executed in parallel workers
EditorAgent.run(...)
Adjust workers:
SupervisorAgent(llm, workers=4)

💾 Memory & Session
Short-term session:
InMemorySessionService stores job snapshots
Long-term memory:
MemoryBank stores user tone profile or reusable data
👀 Observability
Every step logs:
Agent name
Timestamp
Job ID
Status
Logs appear automatically in the console/Streamlit.


🧮 Pause/Resume
agent.pause_job(job_id)
agent.resume_job(job_id)
🧪 Requirements
streamlit
requests
beautifulsoup4
transformers
torch
