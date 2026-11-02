"""End-to-end check against a running server, showing what each node produced.

    uvicorn app:app --port 8000          # terminal 1 (reads GROQ_API_KEY from .env)
    python scripts/live_check.py         # terminal 2

Paths covered:
  clean: run -> organiser asks for changes -> organiser approves -> publish -> donor Q&A (incl. the medical guardrail)
  risky: run -> fraud review: approve -> story -> organiser approves
         run -> fraud review: request info
         run -> fraud review: reject
For every node it prints whether the LLM or the rule-based fallback ran, plus that node's output.
If a node shows "rules" while the key is set, the server log has a "LLM call in <node> failed" line saying why.
"""
import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
c = httpx.Client(base_url=BASE, timeout=180)


def show_trace(state, seen=0):
    for t in state["trace"][seen:]:
        print(f"  [{t['mode']:>5}] {t['node']:<18} {t['detail']}")
    return len(state["trace"])


def dump(title, value):
    print(f"\n  --- {title} ---")
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    print("  " + text.replace("\n", "\n  "))


def resume(r, decision, **extra):
    return c.post("/api/campaign/resume", json={"token": r["token"], "decision": decision, **extra}).json()


def show_story(state):
    dump(f"draft_story v{state['story_version']} (attempts this round: {state['story_attempts']})", state["story"])
    if state.get("story_issues"):
        dump("check_story issues left", state["story_issues"])


health = c.get("/api/health").json()
print(f"Server: {BASE}  llm={health['llm']}  model={health['model']}")
if not health["llm"]:
    print("WARNING: no GROQ_API_KEY on the server, so everything below is the rule-based fallback.")
samples = c.get("/api/meta").json()["samples"]

# ---------------------------------------------------------------- clean case
print("\n=== Sample 'clean': run ===")
r = c.post("/api/campaign/start", json=samples["clean"]).json()
s = r["state"]
seen = show_trace(s)
dump("extract_case -> case (LLM fields)", {k: s["case"].get(k) for k in
     ("category", "condition_plain", "treatment_plain", "urgency_days", "family_situation", "claims")})
dump("assess_risk -> flags", s["flags"])
show_story(s)
assert r["paused_at"] == "organiser_approval", r["paused_at"]

print("\n=== Organiser: request changes ===")
r = resume(r, "changes", feedback="Mention that my father is the kidney donor and thank the hospital staff")
seen = show_trace(r["state"], seen)
show_story(r["state"])

print("\n=== Organiser: approve ===")
r = resume(r, "approve")
show_trace(r["state"], seen)
print(f"\n  status={r['state']['status']}  message={r['state']['message']}")
campaign_token = r["campaign_token"]
print(f"  public page: {BASE}/c#{campaign_token[:24]}…")

print("\n=== Donor Q&A ===")
for q in ["Where does the money go?", "Who is the kidney donor?", "Will she fully recover?", "What is the patient's blood group?"]:
    a = c.post("/api/donor/ask", json={"question": q, "campaign_token": campaign_token}).json()
    print(f"  Q: {q}\n  [{a['mode']}] A: {a['answer']}\n")

# ---------------------------------------------------------------- risky case, three reviewer decisions
for decision, extra in [("approve", {"note": "Spoke to the hospital; estimate confirmed"}),
                        ("request_info", {"note": "Please upload the estimate from Lifeline Heart Institute"}),
                        ("reject", {"note": "UPI request and estimate from a different hospital"})]:
    print(f"\n=== Sample 'risky': run -> fraud review: {decision} ===")
    r = c.post("/api/campaign/start", json=samples["risky"]).json()
    s = r["state"]
    seen = show_trace(s)
    if decision == "approve":   # print the analysis once
        dump("extract_case -> case (LLM fields)", {k: s["case"].get(k) for k in
             ("category", "condition_plain", "treatment_plain", "urgency_days", "family_situation", "claims")})
        dump(f"assess_risk -> {s['risk_score']}/100 ({s['risk_level']})", [f"[{f['severity']}] {f['code']}: {f['message']}" for f in s["flags"]])
    assert r["paused_at"] == "fraud_review", r["paused_at"]
    r = resume(r, decision, **extra)
    seen = show_trace(r["state"], seen)
    if r["paused_at"] == "organiser_approval":
        show_story(r["state"])
        r = resume(r, "approve")
        show_trace(r["state"], seen)
    print(f"\n  status={r['state']['status']}")
    if r["state"].get("message"):
        dump("message", r["state"]["message"])
