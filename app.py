"""CareFund API (FastAPI). Deployed on Vercel as a Python function; UI is served from public/."""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from carefund import donor, graph
from carefund.knowledge import DOCUMENT_LABELS
from carefund.llm import llm_available, model_name
from carefund.samples import SAMPLES

app = FastAPI(title="CareFund campaign agent")
ROOT = Path(__file__).parent


class StartReq(BaseModel):
    form: dict
    documents: list[str] = []
    estimate_text: str = ""


class ResumeReq(BaseModel):
    state: dict
    paused_at: str
    decision: str
    note: Optional[str] = ""
    feedback: Optional[str] = ""


class AskReq(BaseModel):
    question: str
    knowledge_base: list[str]


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm_available(), "model": model_name()}


@app.get("/api/meta")
def meta():
    return {"samples": SAMPLES, "documents": DOCUMENT_LABELS, "graph": graph.mermaid()}


@app.post("/api/campaign/start")
def start(req: StartReq):
    return graph.start_campaign(req.form, req.documents, req.estimate_text)


@app.post("/api/campaign/resume")
def resume(req: ResumeReq):
    try:
        return graph.resume_campaign(req.state, req.paused_at,
                                     {"decision": req.decision, "note": req.note, "feedback": req.feedback})
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/donor/ask")
def ask(req: AskReq):
    return donor.answer(req.question, req.knowledge_base)


@app.get("/")
def index():
    return FileResponse(ROOT / "public" / "index.html")
