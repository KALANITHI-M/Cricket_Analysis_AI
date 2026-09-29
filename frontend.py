"""Streamlit frontend for the Cricket Match Analyst."""
import asyncio
import os
import sys

import streamlit as st
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from cricket_analyst import guardrails
from cricket_analyst.graph import build_graph


st.set_page_config(
    page_title="Cricket Match Analyst",
    page_icon="🏏",
    layout="centered",
)

load_dotenv()

SAMPLE_QUESTIONS = [
    "How does umpire's call work in DRS?",
    "Build me a fantasy XI for CSK vs MI at Chepauk within 100 credits",
    "Compare CSK and MI at Chepauk",
]


async def _create_graph():
    client = MultiServerMCPClient(
        {
            "cricket": {
                "command": sys.executable,
                "args": ["-m", "cricket_analyst.mcp_server"],
                "transport": "stdio",
            }
        }
    )
    return build_graph(await client.get_tools())


@st.cache_resource(show_spinner=False)
def get_graph():
    """Create the backend graph once per Streamlit process."""
    return asyncio.run(_create_graph())


def ask_question(graph, question):
    return asyncio.run(graph.ainvoke({"query": question}))["answer"]


def has_groq_key():
    key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(key) and key != "your_groq_api_key_here"


st.title("🏏 Cricket Match Analyst")
st.caption("Ask about cricket rules, match statistics, team news, or fantasy XIs.")

with st.sidebar:
    st.header("Configuration")
    if has_groq_key():
        st.success("Groq API key configured")
    else:
        st.warning("Groq API key missing")
        st.caption("Copy .env.example to .env and add your own key, then restart Streamlit.")
    st.header("Example questions")
    for sample in SAMPLE_QUESTIONS:
        if st.button(sample, use_container_width=True):
            st.session_state["question"] = sample

question = st.text_area(
    "Your question",
    key="question",
    height=120,
    placeholder="For example: Build me a fantasy XI for CSK vs MI at Chepauk within 100 credits",
)

if st.button("Ask analyst", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Enter a question before submitting.")
    elif guardrails.is_obviously_out_of_scope(question):
        st.info(guardrails.OUT_OF_SCOPE)
    elif not has_groq_key():
        st.error(
            "Add your Groq API key to a local .env file before asking a question. "
            "The key is required to route and answer requests."
        )
    else:
        with st.spinner("Processing your question..."):
            try:
                answer = ask_question(get_graph(), question.strip())
            except TypeError as exc:
                if "authentication" in str(exc).lower() or "api key" in str(exc).lower():
                    st.error(
                        "Groq authentication failed. Check GROQ_API_KEY in "
                        "your local .env file and restart Streamlit."
                    )
                else:
                    raise
            except Exception as exc:
                error_text = str(exc).lower()
                if "credit balance is too low" in error_text or "plans & billing" in error_text:
                    st.error(
                        "Your provider account has no available credits. Add credits "
                        "in the provider billing settings, "
                        "then restart Streamlit and try again."
                    )
                elif "model_not_found" in error_text or "does not exist" in error_text:
                    st.error(
                        "The configured Groq model is unavailable. Update CRICKET_MODEL "
                        "in .env to an active Groq model and restart Streamlit."
                    )
                else:
                    st.error(
                        "The analyst could not process this question. Check the "
                        "backend configuration and try again."
                    )
            else:
                st.subheader("Result")
                st.markdown(answer)
