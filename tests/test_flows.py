import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fastapi.testclient import TestClient
from app import app
from carefund.samples import CLEAN, RISKY

c = TestClient(app)

def nodes(r): return [t["node"] for t in r["state"]["trace"]]

def test_clean_case_publishes():
    r = c.post("/api/campaign/start", json=CLEAN).json()
    assert r["state"]["risk_level"] == "low", r["state"]["flags"]
    assert r["paused_at"] == "organiser_approval"
    assert r["state"]["story_issues"] == []
    # organiser asks for a change -> redraft -> paused again
    r2 = c.post("/api/campaign/resume", json={"state": r["state"], "paused_at": r["paused_at"],
               "decision": "changes", "feedback": "mention dad is the donor"}).json()
    assert r2["paused_at"] == "organiser_approval"
    assert nodes(r2).count("draft_story") == 2
    r3 = c.post("/api/campaign/resume", json={"state": r2["state"], "paused_at": "organiser_approval", "decision": "approve"}).json()
    assert r3["paused_at"] is None and r3["state"]["status"] == "live"
    kb = r3["state"]["campaign"]["knowledge_base"]
    a = c.post("/api/donor/ask", json={"question": "Where does my money go?", "knowledge_base": kb}).json()
    assert "hospital" in a["answer"].lower(), a
    a2 = c.post("/api/donor/ask", json={"question": "Will she survive the surgery?", "knowledge_base": kb}).json()
    assert a2["mode"] == "guardrail"

def test_risky_case_flags_and_reject():
    r = c.post("/api/campaign/start", json=RISKY).json()
    s = r["state"]
    codes = {f["code"] for f in s["flags"]}
    assert s["risk_level"] == "high"
    assert {"cost_outlier", "goal_above_gap", "direct_payment", "no_estimate", "unrelated_organiser"} <= codes, codes
    assert r["paused_at"] == "fraud_review"
    rj = c.post("/api/campaign/resume", json={"state": s, "paused_at": "fraud_review", "decision": "reject", "note": "UPI ask"}).json()
    assert rj["state"]["status"] == "rejected"

def test_risky_request_info_and_approve():
    s = c.post("/api/campaign/start", json=RISKY).json()["state"]
    ri = c.post("/api/campaign/resume", json={"state": s, "paused_at": "fraud_review", "decision": "request_info"}).json()
    assert ri["state"]["status"] == "needs_info" and "Cost estimate" in ri["state"]["message"]
    ap = c.post("/api/campaign/resume", json={"state": s, "paused_at": "fraud_review", "decision": "approve"}).json()
    assert ap["paused_at"] == "organiser_approval"

def test_meta():
    m = c.get("/api/meta").json()
    assert "graph" in m and "fraud_review" in m["graph"]

def test_feedback_applied_without_llm_and_guardrail():
    r = c.post("/api/campaign/start", json=CLEAN).json()
    r2 = c.post("/api/campaign/resume", json={"state": r["state"], "paused_at": "organiser_approval",
               "decision": "changes", "feedback": "Mention that her husband is the kidney donor"}).json()
    assert "Her husband is the kidney donor." in r2["state"]["story"]["body"]
    assert r2["state"]["story_version"] == 2
    for q in ["Will she recover?", "what are her chances", "will the surgery work"]:
        assert c.post("/api/donor/ask", json={"question": q, "knowledge_base": ["x"]}).json()["mode"] == "guardrail"
