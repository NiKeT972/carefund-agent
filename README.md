# CareFund — AI campaign agent for medical crowdfunding

[![tests](https://github.com/NiKeT972/carefund-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/NiKeT972/carefund-agent/actions/workflows/tests.yml)

A **LangGraph** agent that takes a patient's case from an organiser, verifies the documents, scores fraud risk, sends risky cases to a human reviewer, writes the fundraiser story with a self-correcting quality loop, gets the organiser's sign-off, and publishes a campaign page with a **LangChain RAG** assistant that answers donor questions.

**Live demo:** https://carefund-agent.vercel.app

Built for the kind of workflow platforms like ImpactGuru, Milaap and Ketto run every day: every campaign must be verified before it goes live, stories must be honest, and donors ask the same questions again and again.

## What the agent does

```mermaid
graph TD
    START([start]) --> extract_case
    extract_case --> check_documents
    check_documents --> assess_risk
    assess_risk -. "risk low" .-> draft_story
    assess_risk -. "risk medium/high" .-> fraud_review
    fraud_review:::human -. approve .-> draft_story
    fraud_review -. "request info" .-> request_documents --> END([end])
    fraud_review -. reject .-> reject_case --> END
    draft_story --> check_story
    check_story -. "issues & attempts < 3" .-> draft_story
    check_story -. passed .-> organiser_approval
    organiser_approval:::human -. "changes requested" .-> draft_story
    organiser_approval -. approved .-> publish --> END
    classDef human fill:#efebff,stroke:#5b3fd1
```

| Node | Type | What it does |
|---|---|---|
| `extract_case` | LLM (structured output) | Pulls category, plain-language condition, urgency, family situation and factual claims out of the form and free-text story into a Pydantic `CaseFacts` model |
| `check_documents` | Rules | Works out the required documents for the case type (e.g. transplant → authorisation committee approval) and cross-checks the pasted hospital estimate: hospital name, patient name, amount within 15% |
| `assess_risk` | Rules + LLM | 12 fraud rules (cost far above benchmark, goal above funding gap, personal UPI requests, unrelated organiser without authorisation, estimate mismatches, pressure language…) plus an LLM analyst that adds evidence-based concerns. Produces a 0–100 risk score |
| `fraud_review` | **Human-in-the-loop** | Graph pauses (`interrupt_before`). Reviewer approves, requests documents or rejects |
| `draft_story` | LLM | Writes a warm, factual campaign (title, share summary, story) with strict rules: no cure guarantees, no pressure, funds go to the hospital, ₹ amounts in Indian format |
| `check_story` | Rules | Validates the draft (names patient + hospital, states the goal, banned phrases, length) and **loops back** to `draft_story` with the issues, max 3 attempts |
| `organiser_approval` | **Human-in-the-loop** | Organiser approves or asks for changes, which go back into the draft |
| `publish` | Rules | Builds the campaign page and the donor knowledge base; status is `live` or `awaiting_documents` |

**Donor Q&A** (`carefund/donor.py`) is a LangChain chain: a custom `BaseRetriever` (BM25 with query expansion, light enough for serverless) → guardrail (refuses medical predictions like "will she recover?") → `ChatPromptTemplate | ChatGroq` answering only from retrieved facts, or handing the question to the organiser.

## Engineering notes

- **Stateless human-in-the-loop on serverless.** Vercel functions don't keep memory between requests, so the paused graph state is returned to the client. On resume the API rebuilds the graph and calls `update_state(config, state, as_node=<node before the checkpoint>)`; LangGraph then continues from the interrupted node without re-triggering the interrupt.
- **Tamper-proof paused state.** If the server trusted the state the browser sends back, an organiser could lower `risk_level` and skip fraud review. The paused state (and the checkpoint it paused at) travels inside an HMAC-SHA256-signed token (`carefund/state_token.py`); the API resumes only from tokens it signed and checks the decision is valid for that checkpoint. Set `STATE_SIGNING_KEY` in production.
- **Honest donor answers without an LLM.** BM25 always ranks *some* fact first, so in retrieval-only mode an answer is given only if a fact covers most of the question's meaningful words (with synonyms); otherwise the question goes to the organiser. A table of real donor questions in the tests checks both cases.
- **Graceful degradation.** Every LLM node falls back to deterministic logic if no API key is set or the LLM call fails, so the app always works. The UI shows which nodes ran on the LLM, rules or a human.
- **Tests** cover every path through the graph (clean → publish, risky → reject / request info / approve, organiser revisions, guardrails) and the LLM paths with a stub model: `pytest -q`.

## Stack

LangGraph 1.x · LangChain Core · langchain-groq (Llama 3.3 70B) · Pydantic · rank-bm25 · FastAPI · vanilla JS UI · Vercel

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                 # app deps + pytest + uvicorn
cp .env.example .env                                # then put your GROQ_API_KEY in .env (optional; rules mode without it)
uvicorn app:app --reload                            # open http://localhost:8000
pytest -q
python scripts/live_check.py                        # with the server running: every review path + donor Q&A
```

The key is read from `.env` via python-dotenv. Real environment variables take precedence, which is how Vercel supplies it.

## API

| Method | Path | Body |
|---|---|---|
| POST | `/api/campaign/start` | `{form, documents[], estimate_text}` → `{state, paused_at, token?}` |
| POST | `/api/campaign/resume` | `{token, decision, note?, feedback?}`: `token` comes from the paused response; decisions: `approve`/`request_info`/`reject` at fraud review, `approve`/`changes` at organiser approval |
| POST | `/api/donor/ask` | `{question, knowledge_base[]}` |
| GET | `/api/meta` | samples, document labels, Mermaid graph |

The cost benchmarks are illustrative ranges for flagging outliers, not medical pricing. All people and documents in the samples are fictional.

---
Built by **Niket Korgaonkar** · [GitHub](https://github.com/NiKeT972)
