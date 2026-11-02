# CareFund: notes for Claude Code

LangGraph agent for medical crowdfunding: extract case → check documents → fraud risk (rules + LLM) → human fraud review → story draft with a self-check loop → organiser approval → publish, plus a donor Q&A RAG chain. Portfolio project by Niket Korgaonkar; he must be able to explain every line in interviews, so keep code simple and explain changes.

Live: https://carefund-agent.vercel.app (Vercel **auto-deploys on push to `main`**; there is no vercel.json, Vercel detects the FastAPI `app` in `app.py`).

## Layout
- `app.py`: FastAPI. `POST /api/campaign/start`, `POST /api/campaign/resume` (`{token, decision, note?, feedback?}`), `POST /api/donor/ask` (`{question, campaign_token}`), `POST /api/campaign/view`, `GET /c` (public page `public/campaign.html`), `GET /api/meta`, `GET /api/health`; `/` serves `public/index.html`.
- `carefund/graph.py`: nodes, routers, `build_graph()`, `start_campaign()`, `resume_campaign()`. The module docstring has the flow.
- `carefund/knowledge.py`: rule data: required documents per category, cost benchmarks, payment/pressure phrases, banned story phrases, platform policies.
- `carefund/donor.py`: donor Q&A: `BM25FactRetriever` → medical-advice guardrail (regex) → `PROMPT | llm`; with no LLM it answers from the best-covering fact or says it doesn't know (`mode: retrieval`).
- `carefund/llm.py`: `get_llm()` returns `ChatGroq` or `None` when `GROQ_API_KEY` is unset. Loads `.env` with `override=False`. `llm_failed()` logs why a node fell back.
- `carefund/samples.py`: `clean` (low risk → organiser approval) and `risky` (risk 100 → fraud review).
- `scripts/live_check.py`: drives both samples through every human path (organiser changes/approve; fraud approve/request_info/reject) plus donor Q&A against a running server, printing each node's output.

CI: `.github/workflows/tests.yml` runs `pytest -q` on push/PR (no keys needed).

## Run
```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app:app --reload            # http://127.0.0.1:8000
.venv/bin/python scripts/live_check.py        # in a 2nd terminal, against the running server
```
Key: put `GROQ_API_KEY=...` in `.env` (gitignored; template in `.env.example`). Never commit it or ask for it in chat. With no key, everything runs on rules.

## Design rules / gotchas
- **Model:** `DEFAULT_MODEL = "openai/gpt-oss-120b"` in `llm.py`, overridable with `GROQ_MODEL`. Groq retired `llama-3.3-70b-versatile` (2026-08-16); it returned 404 `model_not_found` and every node silently fell back to rules. If all nodes show `rules` with a key set, check the logs for `model_not_found` first.
- **Every LLM node has a rule-based fallback** (`extract_case`, `assess_risk`'s analyst concerns, `draft_story`, donor answer). Pattern: `if llm: try: ...; except Exception as e: llm_failed(node, e)` → rules path. Keep both paths working; trace entries carry `mode` (`llm`/`rules`/`human`) and the UI shows it.
- **Stateless human-in-the-loop:** `interrupt_before=["fraud_review", "organiser_approval"]`; each request builds a fresh graph with `MemorySaver`. `resume_campaign()` re-seeds with `update_state(cfg, state, as_node=RESUME_FROM[paused_at])` (`fraud_review` ← `assess_risk`, `organiser_approval` ← `check_story`). Don't rely on the checkpointer between requests (serverless).
- **Signed paused state:** start/resume return `token` (HMAC-SHA256 over zlib+base64 JSON of `{state, paused_at}`, `carefund/state_token.py`) when paused. `/api/campaign/resume` takes `{token, decision, note, feedback}`; `paused_at` comes from the token, never the client. Key from `STATE_SIGNING_KEY`; without it a public dev key is used (`/api/health` shows `state_signing: false`).
- **Published campaigns:** `_respond` adds `campaign_token = seal({campaign, status})` when status is `live`/`awaiting_documents`. It is the share link (`/c#<token>`, fragment so it stays out of logs) and the only source of donor Q&A facts. `_campaign_from()` rejects tokens without `campaign`; resume rejects tokens without `paused_at`, so the two token kinds can't be swapped.
- Decisions are validated per checkpoint in `app.py` (`DECISIONS`): fraud review `approve|request_info|reject`, organiser `approve|changes` (changes needs feedback). In the graph, fraud `approve` → `draft_story`, `reject` → `reject_case`, anything else → `request_documents`.
- Risk score = sum of flag weights (high 35, medium 15, low 5), capped at 100; ≥30 → fraud review. LLM concerns are added as `low` flags, so the LLM can nudge but never dominate the score.
- `check_story` loop: max 3 attempts per round; `organiser_approval` resets `story_attempts` when changes are requested. The goal amount check looks for the raw digits (e.g. `500000`) in the digit-stripped text, so `₹5,00,000` passes.
- `CaseFacts.category` is a `Literal`; if the model returns a value outside the list, validation fails and `extract_case` falls back to keyword rules.
- Retrieval-only donor mode answers with the retrieved fact that best `coverage()`s the question's non-generic words (with `_EXPAND` synonyms); below 0.6 it returns `NOT_FOUND`. `DONOR_CASES` in `tests/test_flows.py` is the regression table: add a row when changing `_EXPAND`/`_GENERIC`. The LLM path still gets the top-4 facts and decides itself.
- **Tests must never hit the network:** `tests/conftest.py` sets `GROQ_API_KEY=""` before any import (not `pop`), so `load_dotenv(override=False)` can't pull a real key from `.env`. LLM paths are tested with stub models in `tests/test_llm_path.py`.
- `uvicorn` and `pytest` live in `requirements-dev.txt`; `requirements.txt` is what Vercel installs.
- The cloud Claude Code container blocks `api.groq.com` by default; allow it in the environment's network settings to test the live LLM there.
