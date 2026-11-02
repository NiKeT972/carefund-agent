"""LLM access. Returns None when no API key is configured so every node can fall back to rules."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load GROQ_API_KEY (and optional GROQ_MODEL) from <repo>/.env for local runs.
# override=False: a variable already set in the real environment (e.g. on Vercel) always wins.
# Vercel has no .env file, so this does nothing there.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# Groq retires models over time (llama-3.3-70b-versatile was shut down on 2026-08-16 and then
# returned 404 model_not_found). Default to Groq's recommended replacement; override with GROQ_MODEL
# (see https://console.groq.com/docs/models) without changing code.
DEFAULT_MODEL = "openai/gpt-oss-120b"

log = logging.getLogger("carefund")


def llm_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def get_llm(temperature: float = 0.2):
    if not llm_available():
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
        temperature=temperature,
        timeout=30,
        max_retries=1,
    )


def llm_failed(node: str, err: Exception) -> None:
    """Every LLM node falls back to rules on error. Log the reason so a silent fallback is visible when debugging."""
    log.warning("LLM call in %s failed, using rule-based fallback: %s: %s", node, type(err).__name__, err)


def model_name() -> str:
    return (os.getenv("GROQ_MODEL") or DEFAULT_MODEL) if llm_available() else "built-in rules"
