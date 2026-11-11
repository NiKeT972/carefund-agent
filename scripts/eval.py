"""Measure CareFund on the hand-labelled evaluation set.

    python scripts/eval.py            # uses the LLM if GROQ_API_KEY is set, else the built-in rules
    python scripts/eval.py --rules    # force rules mode

Writes evals/report_<mode>.json (shown in the UI's Evaluation panel and served by GET /api/eval).

evals/cases.jsonl        32 campaigns: 16 genuine, 16 that a fraud analyst would want to see. Each is labelled
                         by hand, before running the agent, with the category, whether it needs human fraud
                         review, which red flags must be raised and which must not.
evals/holdout.jsonl      10 more campaigns written AFTER the rules were fixed against cases.jsonl; the rules
                         were not tuned on them (one real bug they exposed was fixed and is recorded in the report). Reported separately: it is the honest estimate of how well the rules generalise.
evals/donor_cases.jsonl  20 donor questions on the published demo campaign: answerable (with a phrase the
                         answer must contain), not in the campaign facts, or a medical question to refuse.

What "task success" means here:
  genuine case  -> skips fraud review, the story passes every check, and after the organiser approves it is published
  flagged case  -> the run stops at fraud review AND every labelled red flag was raised
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name):
    return [json.loads(line) for line in (ROOT / "evals" / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def run_cases(graph, name: str) -> list[dict]:
    rows = []
    for c in load(name):
        t0 = time.perf_counter()
        out = graph.start_campaign(c["form"], c["documents"], c["estimate_text"])
        st = out["state"]
        published = None
        if out["paused_at"] == "organiser_approval":
            done = graph.resume_campaign(st, "organiser_approval", {"decision": "approve"})
            published = done["state"].get("status") in ("live", "awaiting_documents")
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"id": c["id"], "note": c["note"], "label": c["label"], "ms": ms,
                     "category": st["case"]["category"], "held": out["paused_at"] == "fraud_review",
                     "risk": st["risk_score"], "flags": sorted({f["code"] for f in st["flags"]}),
                     "story_ok": out["paused_at"] == "organiser_approval" and not st.get("story_issues"),
                     "story_attempts": st.get("story_attempts", 0), "published": published})
    return rows


def success(r) -> bool:
    if r["label"]["needs_review"]:
        return r["held"] and set(r["label"]["must_flag"]) <= set(r["flags"])
    return (not r["held"]) and r["story_ok"] and bool(r["published"])


def failed(rows) -> list[dict]:
    return [{"id": r["id"], "note": r["note"], "risk": r["risk"], "held": r["held"],
             "flags": r["flags"], "expected_flags": r["label"]["must_flag"]} for r in rows if not success(r)]


def run(force_rules: bool = False) -> dict:
    if force_rules:
        os.environ["GROQ_API_KEY"] = ""
    from carefund import donor, graph
    from carefund.llm import model_name
    from carefund.samples import CLEAN

    rows = run_cases(graph, "cases.jsonl")
    L = lambda r: r["label"]
    n = len(rows)
    flagged = [r for r in rows if L(r)["needs_review"]]
    genuine = [r for r in rows if not L(r)["needs_review"]]

    # Red flags: recall over (case, required flag) pairs; "wrong flags" = a flag the label says must NOT appear.
    need = [(r["id"], code) for r in rows for code in L(r)["must_flag"]]
    found = [(i, code) for i, code in need if code in next(r for r in rows if r["id"] == i)["flags"]]
    wrong = [(r["id"], code) for r in rows for code in L(r)["must_not_flag"] if code in r["flags"]]

    # Donor Q&A on the published demo campaign.
    demo = graph.start_campaign(CLEAN["form"], CLEAN["documents"], CLEAN["estimate_text"])
    kb = graph.resume_campaign(demo["state"], "organiser_approval", {"decision": "approve"})["state"]["campaign"]["knowledge_base"]
    drows = []
    for d in load("donor_cases.jsonl"):
        a = donor.answer(d["question"], kb)
        if a["mode"] == "guardrail":
            got = "guardrail"
        elif a["answer"] == donor.NOT_FOUND or "don't have that information" in a["answer"].lower():
            got = "not_found"
        else:
            got = "answer"
        ok = got == d["expect"] and (d["expect"] != "answer" or d["must_contain"].lower() in a["answer"].lower())
        drows.append({"id": d["id"], "question": d["question"], "expect": d["expect"], "got": got, "ok": ok,
                      "answer": a["answer"]})
    by = lambda kind: [r for r in drows if r["expect"] == kind]

    hold = run_cases(graph, "holdout.jsonl")
    h_flag = [r for r in hold if r["label"]["needs_review"]]
    report = {
        "mode": model_name(), "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "cases": n, "donor_questions": len(drows),
        "metrics": {
            "task_success": pct(sum(success(r) for r in rows), n),
            "category_accuracy": pct(sum(r["category"] == L(r)["category"] for r in rows), n),
            "fraud_review_recall": pct(sum(r["held"] for r in flagged), len(flagged)),
            "fraud_review_false_alarms": sum(r["held"] for r in genuine),
            "red_flag_recall": pct(len(found), len(need)),
            "wrong_flags": len(wrong),
            "stories_passing_check": pct(sum(r["story_ok"] for r in genuine if not r["held"]), sum(not r["held"] for r in genuine)),
            "avg_story_drafts": round(sum(r["story_attempts"] for r in genuine) / max(len(genuine), 1), 2),
            "donor_answer_accuracy": pct(sum(r["ok"] for r in by("answer")), len(by("answer"))),
            "donor_says_not_found_correctly": pct(sum(r["ok"] for r in by("not_found")), len(by("not_found"))),
            "medical_question_refusals": pct(sum(r["ok"] for r in by("guardrail")), len(by("guardrail"))),
            "avg_ms_per_case": round(sum(r["ms"] for r in rows) / n),
        },
        # Written after the rule fixes and never tuned on: the honest estimate of how the rules generalise.
        "holdout": {
            "cases": len(hold),
            # Recorded on its first run (11 Nov 2026), before fixing the bug it exposed: "upi" matched inside
            # "Jupiter Hospital" (h01). After that fix h01 is no longer unseen, so both numbers are reported.
            "first_run_task_success": 80.0,
            "task_success": pct(sum(success(r) for r in hold), len(hold)),
            "category_accuracy": pct(sum(r["category"] == r["label"]["category"] for r in hold), len(hold)),
            "fraud_review_recall": pct(sum(r["held"] for r in h_flag), len(h_flag)),
            "fraud_review_false_alarms": sum(r["held"] for r in hold if not r["label"]["needs_review"]),
            "failed_cases": failed(hold),
        },
        "category_confusions": dict(Counter(f"{L(r)['category']} → {r['category']}" for r in rows
                                            if r["category"] != L(r)["category"]).most_common()),
        "failed_cases": failed(rows),
        "missed_flags": [{"id": i, "flag": code} for i, code in need if (i, code) not in found],
        "wrong_flags": [{"id": i, "flag": code} for i, code in wrong],
        "donor_misses": [r for r in drows if not r["ok"]],
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", action="store_true", help="force rule-based mode")
    ap.add_argument("--out", help="report file name (default report_<mode>.json)")
    args = ap.parse_args()
    rep = run(args.rules)
    name = args.out or f"report_{'rules' if rep['mode'] == 'built-in rules' else 'llm'}.json"
    (ROOT / "evals" / name).write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"CareFund eval · {rep['mode']} · {rep['cases']} cases · {rep['donor_questions']} donor questions")
    for k, v in rep["metrics"].items():
        print(f"  {k:32} {v}")
    for f in rep["failed_cases"]:
        print(f"  FAIL {f['id']} ({f['note']}): risk {f['risk']}, held={f['held']}, flags={f['flags']}, expected {f['expected_flags']}")
    h = rep["holdout"]
    print(f"  held-out ({h['cases']} cases, never tuned on): task_success {h['task_success']}, category {h['category_accuracy']}, "
          f"review recall {h['fraud_review_recall']}, false alarms {h['fraud_review_false_alarms']}")
    for f in h["failed_cases"]:
        print(f"  HELD-OUT FAIL {f['id']} ({f['note']}): risk {f['risk']}, held={f['held']}, flags={f['flags']}, expected {f['expected_flags']}")
    for d in rep["donor_misses"]:
        print(f"  DONOR {d['id']} expected {d['expect']}, got {d['got']}: {d['question']} -> {d['answer'][:80]}")
    print(f"wrote evals/{name}")


if __name__ == "__main__":
    main()
