# CareFund: notes for Claude Code

LangGraph agent for medical crowdfunding: extract case → check documents → fraud risk (rules + LLM) → human fraud review → story draft with a self-check loop → organiser approval → publish, plus a donor Q&A RAG chain. Portfolio project by Niket Korgaonkar; he must be able to explain every line in interviews, so keep code simple and explain changes.

Live: https://carefund-agent.vercel.app (Vercel **auto-deploys on push to `main`**; there is no vercel.json, Vercel detects the FastAPI `app` in `app.py`).

## Layout
- `app.py`: FastAPI. `POST /api/campaign/start`, `POST /api/campaign/resume` (`{state, paused_at, decision, note?, feedback?}`), `POST /api/donor/ask`, `GET /api/meta`, `GET /api/health`; `/` serves `public/index.html`.
- `carefund/graph.py`: nodes, routers, `build_graph()`, `start_campaign()`, `resume_campaign()`. The module docstring has the flow.
- `carefund/knowledge.py`: rule data: required documents per category, cost benchmarks, payment/pressure phrases, banned story phrases, platform policies.
- `carefund/donor.py`: donor Q&A: `BM25FactRetriever` → medical-advice guardrail (regex) → `PROMPT | llm`; with no LLM it returns the top retrieved fact (`mode: retrieval`).
- `carefund/llm.py`: `get_llm()` returns `ChatGroq` or `None` when `GROQ_API_KEY` is unset. Loads `.env` with `override=False`. `llm_failed()` logs why a node fell back.
- `carefund/samples.py`: `clean` (low risk → organiser approval) and `risky` (risk 100 → fraud review).
- `scripts/live_check.py`: drives both samples through every human path (organiser changes/approve; fraud approve/request_info/reject) plus donor Q&A against a running server, printing each node's output.

## Run
```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app:app --reload            # http://127.0.0.1:8000
.venv/bin/python scripts/live_check.py        # in a 2nd terminal, against the running server
```
Key: put `GROQ_API_KEY=...` in `.env` (gitignored; template in `.env.example`). Never commit it or ask for it in chat. With no key, everything runs on rules.

## Design rules / gotchas
- **Every LLM node has a rule-based fallback** (`extract_case`, `assess_risk`'s analyst concerns, `draft_story`, donor answer). Pattern: `if llm: try: ...; except Exception as e: llm_failed(node, e)` → rules path. Keep both paths working; trace entries carry `mode` (`llm`/`rules`/`human`) and the UI shows it.
- **Stateless human-in-the-loop:** `interrupt_before=["fraud_review", "organiser_approval"]`; each request builds a fresh graph with `MemorySaver`. `resume_campaign()` re-seeds with `update_state(cfg, state, as_node=RESUME_FROM[paused_at])` (`fraud_review` ← `assess_risk`, `organiser_approval` ← `check_story`). Don't rely on the checkpointer between requests (serverless).
- Fraud decisions: `approve` → `draft_story`, `reject` → `reject_case`, **anything else** → `request_documents`.
- Risk score = sum of flag weights (high 35, medium 15, low 5), capped at 100; ≥30 → fraud review. LLM concerns are added as `low` flags, so the LLM can nudge but never dominate the score.
- `check_story` loop: max 3 attempts per round; `organiser_approval` resets `story_attempts` when changes are requested. The goal amount check looks for the raw digits (e.g. `500000`) in the digit-stripped text, so `₹5,00,000` passes.
- `CaseFacts.category` is a `Literal`; if the model returns a value outside the list, validation fails and `extract_case` falls back to keyword rules.
- Retrieval-only donor mode returns the top BM25 fact even when it doesn't answer the question (no "I don't know" without the LLM).
- **Tests must never hit the network:** `tests/conftest.py` sets `GROQ_API_KEY=""` before any import (not `pop`), so `load_dotenv(override=False)` can't pull a real key from `.env`. LLM paths are tested with stub models in `tests/test_llm_path.py`.
- `uvicorn` and `pytest` live in `requirements-dev.txt`; `requirements.txt` is what Vercel installs.
- The cloud Claude Code container blocks `api.groq.com` by default; allow it in the environment's network settings to test the live LLM there.
