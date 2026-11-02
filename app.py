"""CareFund API (FastAPI). Deployed on Vercel as a Python function; UI is served from public/."""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from carefund import donor, graph
from carefund.knowledge import DOCUMENT_LABELS
from carefund.llm import llm_available, model_name
from carefund.samples import SAMPLES
from carefund.state_token import InvalidToken, seal, signing_configured, unseal

app = FastAPI(title="CareFund campaign agent")
ROOT = Path(__file__).parent

# Which decisions each human checkpoint accepts. Anything else is a 422 instead of silently
# falling through to a default branch.
DECISIONS = {"fraud_review": {"approve", "request_info", "reject"}, "organiser_approval": {"approve", "changes"}}


class StartReq(BaseModel):
    form: dict
    documents: list[str] = Field(default_factory=list, max_length=20)
    estimate_text: str = Field("", max_length=5_000)

    @field_validator("form")
    @classmethod
    def form_not_huge(cls, v: dict) -> dict:
        # Size cap keeps a public demo from sending huge prompts to the LLM.
        if len(json.dumps(v, ensure_ascii=False)) > 12_000:
            raise ValueError("form is too large")
        return v


class ResumeReq(BaseModel):
    token: str                                   # signed paused state from the previous response
    decision: str
    note: str = Field("", max_length=1_000)      # fraud reviewer's note
    feedback: str = Field("", max_length=1_000)  # organiser's requested changes


class AskReq(BaseModel):
    question: str = Field(max_length=500)
    campaign_token: str          # signed published campaign; the facts come from it, not from the browser


class ViewReq(BaseModel):
    token: str


def _respond(result: dict) -> dict:
    """Add a signed token when the graph is paused; the browser sends only that token back to resume.
    The token also carries paused_at, so the client can't choose which checkpoint to resume from.
    A published campaign gets its own signed token: it's the shareable public page (/c#<token>) and the
    source of facts for donor Q&A, so neither can be edited by whoever holds the link."""
    st = result["state"]
    if result["paused_at"]:
        result["token"] = seal({"state": st, "paused_at": result["paused_at"]})
    elif st.get("campaign") and st.get("status") in ("live", "awaiting_documents"):
        result["campaign_token"] = seal({"campaign": st["campaign"], "status": st["status"]})
    return result


def _campaign_from(token: str) -> dict:
    try:
        payload = unseal(token)
    except InvalidToken:
        raise HTTPException(400, "This campaign link is invalid or was modified.")
    if "campaign" not in payload:            # e.g. a paused-run token pasted into a campaign link
        raise HTTPException(400, "This campaign link is invalid or was modified.")
    return payload


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm_available(), "model": model_name(), "state_signing": signing_configured()}


@app.get("/api/meta")
def meta():
    return {"samples": SAMPLES, "documents": DOCUMENT_LABELS, "graph": graph.mermaid()}


@app.post("/api/campaign/start")
def start(req: StartReq):
    return _respond(graph.start_campaign(req.form, req.documents, req.estimate_text))


@app.post("/api/campaign/resume")
def resume(req: ResumeReq):
    try:
        paused = unseal(req.token)
    except InvalidToken as e:
        raise HTTPException(400, str(e))
    if "paused_at" not in paused:            # e.g. a campaign token sent to resume
        raise HTTPException(400, "This paused run is no longer valid. Please run the agent again.")
    at = paused["paused_at"]
    if req.decision not in DECISIONS.get(at, set()):
        raise HTTPException(422, f"At {at}, decision must be one of {sorted(DECISIONS.get(at, []))}")
    if at == "organiser_approval" and req.decision == "changes" and not req.feedback.strip():
        raise HTTPException(422, "Say what to change in the story.")
    return _respond(graph.resume_campaign(paused["state"], at,
                                          {"decision": req.decision, "note": req.note, "feedback": req.feedback}))


@app.post("/api/donor/ask")
def ask(req: AskReq):
    return donor.answer(req.question, _campaign_from(req.campaign_token)["campaign"]["knowledge_base"])


@app.post("/api/campaign/view")
def view(req: ViewReq):
    """Public campaign page data. The signature proves CareFund's agent produced and verified this campaign."""
    p = _campaign_from(req.token)
    c = {k: v for k, v in p["campaign"].items() if k != "knowledge_base"}
    return {"campaign": c, "status": p["status"], "facts": len(p["campaign"]["knowledge_base"])}


@app.get("/c")
def campaign_page():
    """Shareable campaign page. The signed campaign travels in the URL fragment (/c#<token>), which browsers never
    send to the server in the request line, so it doesn't end up in access logs."""
    return FileResponse(ROOT / "public" / "campaign.html")


@app.get("/")
def index():
    return FileResponse(ROOT / "public" / "index.html")
