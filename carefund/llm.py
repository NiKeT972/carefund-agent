"""LLM access. Returns None when no API key is configured so every node can fall back to rules."""
import os


def llm_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def get_llm(temperature: float = 0.2):
    if not llm_available():
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        timeout=30,
        max_retries=1,
    )


def model_name() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") if llm_available() else "built-in rules"
