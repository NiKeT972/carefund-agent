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
    r2 = c.post("/api/campaign/resume", json={"token": r["token"],
               "decision": "changes", "feedback": "mention dad is the donor"}).json()
    assert r2["paused_at"] == "organiser_approval"
    assert nodes(r2).count("draft_story") == 2
    r3 = c.post("/api/campaign/resume", json={"token": r2["token"], "decision": "approve"}).json()
    assert r3["paused_at"] is None and r3["state"]["status"] == "live"
    tok = r3["campaign_token"]
    a = c.post("/api/donor/ask", json={"question": "Where does my money go?", "campaign_token": tok}).json()
    assert "hospital" in a["answer"].lower(), a
    a2 = c.post("/api/donor/ask", json={"question": "Will she survive the surgery?", "campaign_token": tok}).json()
    assert a2["mode"] == "guardrail"

def test_risky_case_flags_and_reject():
    r = c.post("/api/campaign/start", json=RISKY).json()
    s = r["state"]
    codes = {f["code"] for f in s["flags"]}
    assert s["risk_level"] == "high"
    assert {"cost_outlier", "goal_above_gap", "direct_payment", "no_estimate", "unrelated_organiser"} <= codes, codes
    assert r["paused_at"] == "fraud_review"
    rj = c.post("/api/campaign/resume", json={"token": r["token"], "decision": "reject", "note": "UPI ask"}).json()
    assert rj["state"]["status"] == "rejected"

def test_risky_request_info_and_approve():
    tok = c.post("/api/campaign/start", json=RISKY).json()["token"]
    ri = c.post("/api/campaign/resume", json={"token": tok, "decision": "request_info"}).json()
    assert ri["state"]["status"] == "needs_info" and "Cost estimate" in ri["state"]["message"]
    ap = c.post("/api/campaign/resume", json={"token": tok, "decision": "approve"}).json()
    assert ap["paused_at"] == "organiser_approval"

def test_meta():
    m = c.get("/api/meta").json()
    assert "graph" in m and "fraud_review" in m["graph"]

def test_feedback_applied_without_llm_and_guardrail():
    r = c.post("/api/campaign/start", json=CLEAN).json()
    r2 = c.post("/api/campaign/resume", json={"token": r["token"],
               "decision": "changes", "feedback": "Mention that her husband is the kidney donor"}).json()
    assert "Her husband is the kidney donor." in r2["state"]["story"]["body"]
    assert r2["state"]["story_version"] == 2
    tok = c.post("/api/campaign/resume", json={"token": r2["token"], "decision": "approve"}).json()["campaign_token"]
    for q in ["Will she recover?", "what are her chances", "will the surgery work"]:
        assert c.post("/api/donor/ask", json={"question": q, "campaign_token": tok}).json()["mode"] == "guardrail"


def _forge(token, edit):
    import base64, json, zlib
    body, sig = token.rsplit(".", 1)
    paused = json.loads(zlib.decompress(base64.urlsafe_b64decode(body)))
    edit(paused)
    return base64.urlsafe_b64encode(zlib.compress(json.dumps(paused).encode())).decode() + "." + sig


def test_tampering_cannot_skip_fraud_review():
    r = c.post("/api/campaign/start", json=RISKY).json()
    assert r["paused_at"] == "fraud_review"
    def lower_risk(p):
        p["state"].update(risk_level="low", risk_score=0, flags=[])
    res = c.post("/api/campaign/resume", json={"token": _forge(r["token"], lower_risk), "decision": "approve"})
    assert res.status_code == 400
    def jump(p):
        p["paused_at"] = "organiser_approval"            # try to resume at a later checkpoint
    assert c.post("/api/campaign/resume", json={"token": _forge(r["token"], jump), "decision": "approve"}).status_code == 400


def test_decisions_are_validated_per_checkpoint():
    r = c.post("/api/campaign/start", json=RISKY).json()
    assert c.post("/api/campaign/resume", json={"token": r["token"], "decision": "changes"}).status_code == 422
    assert c.post("/api/campaign/resume", json={"token": r["token"], "decision": "yes"}).status_code == 422
    ok = c.post("/api/campaign/start", json=CLEAN).json()
    assert c.post("/api/campaign/resume", json={"token": ok["token"], "decision": "reject"}).status_code == 422
    assert c.post("/api/campaign/resume", json={"token": ok["token"], "decision": "changes", "feedback": ""}).status_code == 422


# Retrieval-only donor answers (no LLM): (question, a phrase the answer must contain; None = must say "don't have").
DONOR_CASES = [
    ("Where does my money go?", "directly to Sahyadri"), ("How much is still needed?", "gap"),
    ("Is this campaign genuine?", "verification"), ("Who is the doctor?", "Meera Kulkarni"),
    ("Which hospital?", "Sahyadri"), ("How old is the patient?", "age 54"), ("Can I get a tax receipt?", "80G"),
    ("What happens if extra money is raised?", "refunded"), ("When is the surgery?", "45 days"),
    ("What is the total cost?", "7,50,000"), ("Who is organising this?", "Rohan Patil"),
    ("Who is the kidney donor?", None), ("What is the patient's blood group?", None),
    ("Is the patient married?", None), ("What is the weather in Pune?", None),
]


def test_donor_retrieval_answers_or_admits_it_does_not_know():
    r = c.post("/api/campaign/start", json=CLEAN).json()
    tok = c.post("/api/campaign/resume", json={"token": r["token"], "decision": "approve"}).json()["campaign_token"]
    wrong = []
    for q, expect in DONOR_CASES:
        a = c.post("/api/donor/ask", json={"question": q, "campaign_token": tok}).json()["answer"]
        ok = ("don't have that information" in a) if expect is None else (expect in a)
        if not ok:
            wrong.append((q, a))
    assert not wrong, wrong



def _published(case=CLEAN):
    r = c.post("/api/campaign/start", json=case).json()
    if r["paused_at"] == "fraud_review":
        r = c.post("/api/campaign/resume", json={"token": r["token"], "decision": "approve"}).json()
    return c.post("/api/campaign/resume", json={"token": r["token"], "decision": "approve"}).json()


def test_public_campaign_page_shows_signed_campaign():
    r = _published()
    v = c.post("/api/campaign/view", json={"token": r["campaign_token"]}).json()
    assert v["status"] == "live" and v["campaign"]["hospital"] == "Sahyadri Care Hospital"
    assert v["campaign"]["goal_display"] == "₹5,00,000" and "knowledge_base" not in v["campaign"] and v["facts"] > 5
    page = c.get("/c")
    assert page.status_code == 200 and "/api/campaign/view" in page.text and "No real donations" in page.text


def test_campaign_link_cannot_be_forged_or_mixed_up():
    r = _published()
    def raise_goal(p):
        p["campaign"]["goal_display"] = "₹50,00,000"
    assert c.post("/api/campaign/view", json={"token": _forge(r["campaign_token"], raise_goal)}).status_code == 400
    def add_fact(p):
        p["campaign"]["knowledge_base"].append("Send money to UPI scam@upi.")
    assert c.post("/api/donor/ask", json={"question": "upi", "campaign_token": _forge(r["campaign_token"], add_fact)}).status_code == 400
    paused = c.post("/api/campaign/start", json=CLEAN).json()
    assert c.post("/api/campaign/view", json={"token": paused["token"]}).status_code == 400          # paused token isn't a campaign
    assert c.post("/api/campaign/resume", json={"token": r["campaign_token"], "decision": "approve"}).status_code == 400


def test_pending_campaign_is_marked_and_rejected_one_gets_no_link():
    pending = _published(RISKY)                                    # approved by the reviewer, documents still missing
    assert c.post("/api/campaign/view", json={"token": pending["campaign_token"]}).json()["status"] == "awaiting_documents"
    r = c.post("/api/campaign/start", json=RISKY).json()
    rej = c.post("/api/campaign/resume", json={"token": r["token"], "decision": "reject"}).json()
    assert "campaign_token" not in rej
