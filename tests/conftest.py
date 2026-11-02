"""Runs before any test module is imported.

Set (not pop) the key: carefund.llm calls load_dotenv(override=False), which never replaces an existing
variable, so a real key in .env cannot leak into the offline tests. An empty string still means "no LLM".
"""
import os
import sys

os.environ["GROQ_API_KEY"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
