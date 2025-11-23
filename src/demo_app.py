# src/demo_app.py
import streamlit as st
import sys
from pathlib import Path

# Ensure src package imports work if running from repo root
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from llm_provider import HuggingFaceProvider, HTTPProvider
from agents.supervisor_agent import SupervisorAgent

st.set_page_config(page_title='Automated Blog Writer Agent', layout='wide')

st.title('Automated Blog Writer — Demo')

col1, col2 = st.columns([2,1])

with col1:
    topic = st.text_input('Topic', 'Benefits of Unit Testing')
    audience = st.text_input('Audience', 'Software engineers')
    refs = st.text_area('Reference URLs (one per line)', '')
    run = st.button('Generate Article')

with col2:
    st.markdown('**LLM Provider**')
    provider_type = st.selectbox('Provider', ['HuggingFace (default mock)','HTTP Endpoint (custom)'])
    if provider_type.startswith('HTTP'):
        endpoint = st.text_input('HTTP Endpoint', '')

if 'supervisor' not in st.session_state:
    # default to mock HF provider (uses small/gpt2-like default)
    hf = HuggingFaceProvider(model_name=os.getenv('HF_MODEL', 'gpt2'))
    st.session_state.supervisor = SupervisorAgent(hf)

if run:
    refs_list = [r.strip() for r in refs.split('\n') if r.strip()]
    with st.spinner('Running agents...'):
        res = st.session_state.supervisor.start_job(topic, audience, refs_list)
    st.success('Done')
    st.subheader('Generated Article')
    st.markdown(f"# {topic}\n\n{res['content']}")
    st.subheader('SEO')
    st.json(res.get('seo'))
    st.subheader('Outline')
    st.json(res.get('outline'))
