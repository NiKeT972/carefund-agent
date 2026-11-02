"""CareFund campaign agent, built as a LangGraph state machine.

Flow:
    extract_case -> check_documents -> assess_risk
        -> (risk medium/high) fraud_review  [human-in-the-loop]
              -> approve -> draft_story | request_info -> request_documents -> END | reject -> reject_case -> END
        -> (risk low) draft_story
    draft_story -> check_story -> (issues, attempts < 3) draft_story  [self-correction loop]
                               -> organiser_approval  [human-in-the-loop]
                                      -> changes requested -> draft_story | approved -> publish -> END

Human review is implemented with `interrupt_before`. The API is stateless (serverless), so the paused
state is returned to the client and re-seeded with `update_state(..., as_node=...)` on resume.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import knowledge as K
from .llm import get_llm, llm_failed


# --------------------------------------------------------------------------- state


class CampaignState(TypedDict, total=False):
    form: dict                # raw organiser input
    documents: list           # list of document type keys the organiser uploaded
    estimate_text: str        # pasted text of the hospital cost estimate (optional)
    case: dict                # structured facts extracted from the input
    missing_documents: list
    document_checks: list
    flags: list
    risk_score: int
    risk_level: str
    review: dict              # fraud team decision {decision, note}
    story: dict               # {title, summary, body}
    story_issues: list
    story_attempts: int
    story_version: int
    organiser_feedback: str
    organiser_decision: str
    status: str
    message: str
    campaign: dict
    trace: list


def _trace(state: CampaignState, node: str, detail: str, mode: str = "rules") -> list:
    return list(state.get("trace", [])) + [{"node": node, "detail": detail, "mode": mode}]


def _inr(n: float | int | None) -> str:
    if n is None:
        return "₹0"
    n = int(round(float(n)))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = head + "," + tail
    return ("-" if n < 0 else "") + "₹" + s


def _num(v: Any) -> float:
    try:
        return float(str(v).replace(",", "").replace("₹", "").strip() or 0)
    except ValueError:
        return 0.0


def _amounts_in_text(text: str) -> list[float]:
    """Find rupee amounts in free text, understanding 'lakh' / 'crore'."""
    out = []
    for m in re.finditer(r"(?:₹|rs\.?|inr)?\s*([\d][\d,]*(?:\.\d+)?)\s*(lakh|lakhs|lac|crore|cr|l)?\b", text.lower()):
        raw, unit = m.group(1), m.group(2)
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if unit in ("lakh", "lakhs", "lac", "l"):
            val *= 100_000
        elif unit in ("crore", "cr"):
            val *= 10_000_000
        if val >= 10_000:
            out.append(val)
    return out


# --------------------------------------------------------------------------- 1. extract


class CaseFacts(BaseModel):
    category: Literal["organ_transplant", "cancer", "cardiac", "neonatal", "accident_trauma", "neuro", "other"]
    condition_plain: str = Field(description="The medical condition explained in plain, non-technical words, max 15 words")
    treatment_plain: str = Field(description="The treatment needed in plain words, max 15 words")
    urgency_days: Optional[int] = Field(None, description="Days until treatment is needed, if stated")
    family_situation: str = Field(description="One sentence on the family's situation and why they need help, only from the input")
    claims: list[str] = Field(default_factory=list, description="Factual claims made in the description (amounts, dates, hospital, doctor)")


def _classify(text: str) -> str:
    t = text.lower()
    for cat, words in K.CATEGORY_KEYWORDS.items():
        if any(w in t for w in words):
            return cat
    return "other"


def extract_case(state: CampaignState) -> dict:
    f = state["form"]
    est, arranged, goal = _num(f.get("estimated_cost")), _num(f.get("amount_arranged")), _num(f.get("goal_amount"))
    base = {
        "patient_name": f.get("patient_name", "").strip(),
        "patient_age": _num(f.get("patient_age")),
        "city": f.get("city", "").strip(),
        "hospital": f.get("hospital_name", "").strip(),
        "doctor": f.get("doctor_name", "").strip(),
        "condition": f.get("condition", "").strip(),
        "treatment": f.get("treatment", "").strip(),
        "organiser": f.get("organiser_name", "").strip(),
        "relation": f.get("relation", "").strip().lower(),
        "estimated_cost": est,
        "amount_arranged": arranged,
        "funding_gap": max(est - arranged, 0),
        "goal_amount": goal,
    }
    text = " ".join([base["condition"], base["treatment"], f.get("description", "")])

    llm = get_llm(0)
    if llm is not None:
        try:
            facts: CaseFacts = llm.with_structured_output(CaseFacts).invoke(
                "Extract structured facts for a medical crowdfunding case review. Use only what is stated.\n\n"
                f"Form fields: {base}\n\nOrganiser's description:\n{f.get('description', '')}"
            )
            case = {**base, **facts.model_dump()}
            return {"case": case, "trace": _trace(state, "extract_case",
                    f"Classified as {K.CATEGORY_LABELS[case['category']]}; {len(case['claims'])} claims extracted", "llm")}
        except Exception as e:  # fall back to rules on any LLM error
            llm_failed("extract_case", e)
            err = f" (LLM error: {type(e).__name__})"
        else:
            err = ""
    else:
        err = ""

    case = {
        **base,
        "category": _classify(text),
        "condition_plain": base["condition"],
        "treatment_plain": base["treatment"],
        "urgency_days": None,
        "family_situation": "",
        "claims": [],
    }
    m = re.search(r"within (\d+) days", f.get("description", "").lower())
    if m:
        case["urgency_days"] = int(m.group(1))
    return {"case": case, "trace": _trace(state, "extract_case",
            f"Classified as {K.CATEGORY_LABELS[case['category']]} by keyword rules{err}")}


# --------------------------------------------------------------------------- 2. documents


def check_documents(state: CampaignState) -> dict:
    case = state["case"]
    have = set(state.get("documents", []))
    required = list(K.BASE_DOCUMENTS) + K.CATEGORY_DOCUMENTS.get(case["category"], [])
    if case["relation"] not in K.CLOSE_RELATIONS:
        required.append("authorisation_letter")
    missing = [d for d in required if d not in have]

    checks = []
    est_text = (state.get("estimate_text") or "").strip()
    if est_text:
        hosp = case["hospital"].lower()
        hosp_words = [w for w in re.findall(r"[a-z]+", hosp) if len(w) > 3 and w not in {"hospital", "hospitals", "clinic", "centre", "center"}]
        hosp_ok = bool(hosp) and (hosp in est_text.lower() or (hosp_words and all(w in est_text.lower() for w in hosp_words)))
        checks.append({"check": "Hospital on estimate matches form", "ok": hosp_ok,
                       "detail": "Found" if hosp_ok else f"'{case['hospital']}' not found in the estimate text"})
        name_ok = bool(case["patient_name"]) and case["patient_name"].split()[0].lower() in est_text.lower()
        checks.append({"check": "Patient name on estimate", "ok": name_ok,
                       "detail": "Found" if name_ok else "Patient name not found in the estimate text"})
        amounts = _amounts_in_text(est_text)
        if amounts and case["estimated_cost"]:
            closest = min(amounts, key=lambda a: abs(a - case["estimated_cost"]))
            diff = abs(closest - case["estimated_cost"]) / case["estimated_cost"]
            checks.append({"check": "Estimate amount matches form", "ok": diff <= 0.15,
                           "detail": f"Estimate says {_inr(closest)}, form says {_inr(case['estimated_cost'])}"})
        else:
            checks.append({"check": "Estimate amount matches form", "ok": False, "detail": "No amount found in the estimate text"})

    detail = f"{len(required) - len(missing)}/{len(required)} required documents present"
    if checks:
        detail += f"; {sum(c['ok'] for c in checks)}/{len(checks)} estimate cross-checks passed"
    return {"missing_documents": missing, "document_checks": checks,
            "trace": _trace(state, "check_documents", detail)}


# --------------------------------------------------------------------------- 3. risk


class LLMReview(BaseModel):
    concerns: list[str] = Field(default_factory=list, description="Specific inconsistencies or red flags, max 4. Empty if none.")


def assess_risk(state: CampaignState) -> dict:
    case, f = state["case"], state["form"]
    desc = (f.get("description") or "").lower()
    flags: list[dict] = []

    def flag(code, severity, message):
        flags.append({"code": code, "severity": severity, "message": message})

    lo, hi = K.COST_BENCHMARKS[case["category"]]
    if case["estimated_cost"] > hi * 1.25:
        flag("cost_outlier", "high", f"Estimated cost {_inr(case['estimated_cost'])} is far above the typical range for "
             f"{K.CATEGORY_LABELS[case['category']].lower()} ({_inr(lo)}–{_inr(hi)}).")
    if case["goal_amount"] > case["funding_gap"] * 1.10 and case["funding_gap"] > 0:
        flag("goal_above_gap", "medium", f"Goal {_inr(case['goal_amount'])} is more than the funding gap "
             f"{_inr(case['funding_gap'])} (estimate minus money already arranged).")
    if case["estimated_cost"] <= 0:
        flag("no_cost", "high", "No treatment cost estimate given.")
    if case["category"] == "neonatal" and case["patient_age"] > 1:
        flag("age_mismatch", "medium", f"Newborn/NICU case but patient age is {int(case['patient_age'])}.")
    if not (0 <= case["patient_age"] <= 110):
        flag("age_invalid", "medium", "Patient age looks invalid.")
    if "medical_estimate" in state.get("missing_documents", []):
        flag("no_estimate", "high", "Hospital cost estimate not uploaded, so the amount cannot be verified.")
    if "diagnosis_report" in state.get("missing_documents", []):
        flag("no_diagnosis", "medium", "Diagnosis report not uploaded.")
    if case["relation"] not in K.CLOSE_RELATIONS and "authorisation_letter" in state.get("missing_documents", []):
        flag("unrelated_organiser", "medium", f"Organiser is '{case['relation'] or 'not stated'}' to the patient and "
             "there is no authorisation letter from the family.")
    for c in state.get("document_checks", []):
        if not c["ok"]:
            flag("estimate_mismatch", "high" if "amount" in c["check"].lower() or "Hospital" in c["check"] else "medium",
                 f"{c['check']}: {c['detail']}.")
    if any(p in desc for p in K.PAYMENT_PHRASES):
        flag("direct_payment", "high", "Description asks donors to pay a personal account/UPI directly instead of through the platform.")
    if any(p in desc for p in K.PRESSURE_PHRASES):
        flag("pressure_language", "low", "Description uses high-pressure or exaggerated language.")
    stated = _amounts_in_text(desc)
    if stated and case["estimated_cost"]:
        big = max(stated)
        if big > case["estimated_cost"] * 1.3:
            flag("amount_inconsistent", "medium", f"Description mentions {_inr(big)}, higher than the form estimate "
                 f"{_inr(case['estimated_cost'])}.")

    mode = "rules"
    llm = get_llm(0)
    if llm is not None:
        try:
            r: LLMReview = llm.with_structured_output(LLMReview).invoke(
                "You are a fraud analyst at an Indian medical crowdfunding platform. List concrete inconsistencies or "
                "red flags in this case that are NOT already in the existing flags. Do not speculate without evidence.\n\n"
                f"Case: {case}\nDescription: {f.get('description', '')}\nEstimate text: {state.get('estimate_text', '')}\n"
                f"Existing flags: {[x['message'] for x in flags]}"
            )
            for c in r.concerns[:4]:
                flag("llm_review", "low", c)
            mode = "llm"
        except Exception as e:
            llm_failed("assess_risk", e)

    weights = {"high": 35, "medium": 15, "low": 5}
    score = min(100, sum(weights[x["severity"]] for x in flags))
    level = "high" if score >= 60 else "medium" if score >= 30 else "low"
    return {"flags": flags, "risk_score": score, "risk_level": level,
            "trace": _trace(state, "assess_risk", f"Risk {score}/100 ({level}); {len(flags)} flag(s)", mode)}


def route_after_risk(state: CampaignState) -> str:
    return "fraud_review" if state["risk_level"] in ("medium", "high") else "draft_story"


# --------------------------------------------------------------------------- 4. fraud review (HITL)


def fraud_review(state: CampaignState) -> dict:
    r = state.get("review") or {}
    return {"trace": _trace(state, "fraud_review", f"Reviewer decision: {r.get('decision', 'none')}"
                            + (f" — “{r['note']}”" if r.get("note") else ""), "human")}


def route_after_review(state: CampaignState) -> str:
    d = (state.get("review") or {}).get("decision")
    return {"approve": "draft_story", "reject": "reject_case"}.get(d, "request_documents")


def reject_case(state: CampaignState) -> dict:
    note = (state.get("review") or {}).get("note") or "The case did not pass verification."
    return {"status": "rejected", "message": f"Campaign rejected by the verification team. Reason: {note}",
            "trace": _trace(state, "reject_case", "Campaign closed")}


def request_documents(state: CampaignState) -> dict:
    case = state["case"]
    items = [K.DOCUMENT_LABELS[d] for d in state.get("missing_documents", [])]
    extra = (state.get("review") or {}).get("note")
    lines = [f"Hi {case['organiser'] or 'there'}, thanks for starting a campaign for {case['patient_name']}.",
             "Before we can publish it, our verification team needs:"]
    lines += [f"  • {i}" for i in items] or ["  • Clarification on the points below"]
    if extra:
        lines.append(f"Reviewer note: {extra}")
    lines.append("Reply with these and we'll review your campaign again within 24 hours.")
    return {"status": "needs_info", "message": "\n".join(lines),
            "trace": _trace(state, "request_documents", f"Asked organiser for {len(items)} item(s)")}


# --------------------------------------------------------------------------- 5. story


class Story(BaseModel):
    title: str = Field(description="Campaign title, max 12 words, names the patient and the treatment")
    summary: str = Field(description="One-sentence summary for share cards, max 30 words")
    body: str = Field(description="Campaign story, 180-320 words, first person from the organiser, plain paragraphs")


def _template_story(case: dict, feedback: str) -> dict:
    name = case["patient_name"] or "my family member"
    first = name.split()[0]
    age = f", {int(case['patient_age'])}," if case["patient_age"] else ""
    rel = case["relation"] if case["relation"] and case["relation"] != "self" else "family member"
    low = lambda x: (x[:1].lower() + x[1:]) if x and not x[:2].isupper() else x
    body = [
        f"My name is {case['organiser'] or 'the organiser'}, and I am raising funds for my {rel} {name}{age} from {case['city'] or 'India'}.",
        f"{first} has been diagnosed with {low(case['condition_plain'] or case['condition'])}. The doctors at {case['hospital']}"
        + (f", led by {case['doctor']}," if case["doctor"] else "")
        + f" have advised a {low(case['treatment_plain'] or case['treatment'])}"
        + (f", which needs to happen within {case['urgency_days']} days." if case.get("urgency_days") else "."),
        f"The hospital has estimated the total cost at {_inr(case['estimated_cost'])}. Our family has already arranged "
        f"{_inr(case['amount_arranged'])} through savings, insurance and help from relatives, but we still have a gap of "
        f"{_inr(case['funding_gap'])}.",
        f"We are asking for {_inr(case['goal_amount'])} to cover this gap. All funds raised go directly to {case['hospital']} "
        f"for {first}'s treatment, and we will share updates and hospital bills here as treatment progresses.",
        f"If you can contribute, thank you. If you can't, sharing this page with your friends and family helps just as much.",
    ]
    if case.get("family_situation"):
        body.insert(2, case["family_situation"])
    for req in [x.strip() for x in re.split(r"[;\n]", feedback) if x.strip().lower().startswith("organiser asked:")]:
        fact = re.sub(r"^organiser asked:\s*", "", req, flags=re.I)
        fact = re.sub(r"^(please\s+)?(mention|add|say|include|write|state)\s+(that\s+)?", "", fact, flags=re.I).strip()
        if fact:
            fact = fact[0].upper() + fact[1:]
            body.insert(2, fact if fact.endswith((".", "!", "?")) else fact + ".")
    return {
        "title": f"Help {name} get a {low(case['treatment']) or 'treatment'}"[:90],
        "summary": f"{name} needs {low(case['treatment']) or 'urgent treatment'} at {case['hospital']}. Help close a "
                   f"{_inr(case['funding_gap'])} gap.",
        "body": "\n\n".join(body),
    }


def draft_story(state: CampaignState) -> dict:
    case = state["case"]
    attempts = state.get("story_attempts", 0) + 1
    version = state.get("story_version", 0) + 1
    feedback = "; ".join(state.get("story_issues", []))
    if state.get("organiser_feedback"):
        feedback = (feedback + "; " if feedback else "") + "Organiser asked: " + state["organiser_feedback"]

    llm = get_llm(0.5)
    if llm is not None:
        try:
            s: Story = llm.with_structured_output(Story).invoke(
                "Write a medical crowdfunding campaign for an Indian platform. Rules: warm but factual; no exaggeration, "
                "no guarantees of cure, no 'last chance' pressure; mention the patient's name, the hospital, the total "
                "estimate, the money already arranged, the funding gap and the goal amount in ₹ with Indian digit "
                "grouping; say funds go directly to the hospital; never include bank or UPI details.\n\n"
                f"Case facts: {case}\n"
                + (f"Fix these problems from the previous draft: {feedback}\n" if feedback else "")
                + (f"Previous draft: {state['story']}\n" if state.get("story") and feedback else "")
            )
            return {"story": s.model_dump(), "story_attempts": attempts, "story_version": version, "organiser_feedback": "",
                    "trace": _trace(state, "draft_story", f"Version {version} written" + (" (revised)" if feedback else ""), "llm")}
        except Exception as e:
            llm_failed("draft_story", e)
    return {"story": _template_story(case, feedback), "story_attempts": attempts, "story_version": version,
            "organiser_feedback": "",
            "trace": _trace(state, "draft_story", f"Version {version} from template" + (" (revised)" if feedback else ""))}


def check_story(state: CampaignState) -> dict:
    case, story = state["case"], state["story"]
    text = (story["title"] + " " + story["body"]).lower()
    digits = re.sub(r"[^\d]", "", text)
    issues = []
    if case["patient_name"] and case["patient_name"].split()[0].lower() not in text:
        issues.append("Mention the patient's name")
    if case["hospital"] and case["hospital"].split()[0].lower() not in text:
        issues.append(f"Name the hospital ({case['hospital']})")
    if case["goal_amount"] and str(int(case["goal_amount"])) not in digits:
        issues.append(f"State the goal amount {_inr(case['goal_amount'])}")
    for b in K.STORY_BANNED:
        if b in text:
            issues.append(f"Remove the phrase '{b}'")
    words = len(story["body"].split())
    if words < 120:
        issues.append(f"Story is too short ({words} words); aim for 180–320")
    if words > 450:
        issues.append(f"Story is too long ({words} words); aim for 180–320")
    return {"story_issues": issues,
            "trace": _trace(state, "check_story", "Story passed all checks" if not issues else f"{len(issues)} issue(s): " + "; ".join(issues))}


def route_after_story_check(state: CampaignState) -> str:
    if state.get("story_issues") and state.get("story_attempts", 0) < 3:
        return "draft_story"
    return "organiser_approval"


# --------------------------------------------------------------------------- 6. organiser approval (HITL)


def organiser_approval(state: CampaignState) -> dict:
    d = state.get("organiser_decision", "")
    det = "Organiser approved the story" if d == "approve" else f"Organiser asked for changes: “{state.get('organiser_feedback', '')}”"
    out = {"trace": _trace(state, "organiser_approval", det, "human")}
    if d != "approve":
        out.update({"story_attempts": 0, "story_issues": []})
    return out


def route_after_organiser(state: CampaignState) -> str:
    return "publish" if state.get("organiser_decision") == "approve" else "draft_story"


# --------------------------------------------------------------------------- 7. publish


def publish(state: CampaignState) -> dict:
    case, story = state["case"], state["story"]
    missing = state.get("missing_documents", [])
    status = "live" if not missing else "awaiting_documents"
    kb = [
        f"Patient: {case['patient_name']}, age {int(case['patient_age'])}, from {case['city']}.",
        f"Condition: {case['condition']} ({case['condition_plain']}).",
        f"Treatment: {case['treatment']} at {case['hospital']}" + (f" under {case['doctor']}." if case["doctor"] else "."),
        f"Total hospital estimate is {_inr(case['estimated_cost'])}. The family has arranged {_inr(case['amount_arranged'])}. "
        f"Funding gap is {_inr(case['funding_gap'])}. Campaign goal is {_inr(case['goal_amount'])}.",
        f"The campaign is organised by {case['organiser']}; the patient is the organiser's {case['relation']}.",
        "Verified documents: " + ", ".join(K.DOCUMENT_LABELS[d] for d in state.get("documents", []) if d in K.DOCUMENT_LABELS) + ".",
    ]
    if missing:
        kb.append("Still pending verification: " + ", ".join(K.DOCUMENT_LABELS[d] for d in missing) + ".")
    if case.get("urgency_days"):
        kb.append(f"Treatment is needed within {case['urgency_days']} days.")
    kb += [p for p in re.split(r"(?<=[.!?])\s+", story["body"]) if len(p) > 30]
    kb += K.PLATFORM_POLICIES
    campaign = {**story, "goal": case["goal_amount"], "goal_display": _inr(case["goal_amount"]),
                "hospital": case["hospital"], "category": K.CATEGORY_LABELS[case["category"]],
                "risk_level": state["risk_level"], "knowledge_base": kb}
    msg = ("Campaign is live and ready to share." if status == "live"
           else "Campaign page is ready; it goes live once the pending documents are verified.")
    return {"status": status, "message": msg, "campaign": campaign,
            "trace": _trace(state, "publish", f"Status: {status}; donor knowledge base has {len(kb)} facts")}


# --------------------------------------------------------------------------- graph


HUMAN_NODES = ["fraud_review", "organiser_approval"]
# The node that runs just before each human checkpoint (used to re-seed state on resume).
RESUME_FROM = {"fraud_review": "assess_risk", "organiser_approval": "check_story"}


def build_graph():
    g = StateGraph(CampaignState)
    for name, fn in [("extract_case", extract_case), ("check_documents", check_documents), ("assess_risk", assess_risk),
                     ("fraud_review", fraud_review), ("reject_case", reject_case), ("request_documents", request_documents),
                     ("draft_story", draft_story), ("check_story", check_story),
                     ("organiser_approval", organiser_approval), ("publish", publish)]:
        g.add_node(name, fn)
    g.add_edge(START, "extract_case")
    g.add_edge("extract_case", "check_documents")
    g.add_edge("check_documents", "assess_risk")
    g.add_conditional_edges("assess_risk", route_after_risk, ["fraud_review", "draft_story"])
    g.add_conditional_edges("fraud_review", route_after_review, ["draft_story", "reject_case", "request_documents"])
    g.add_edge("reject_case", END)
    g.add_edge("request_documents", END)
    g.add_edge("draft_story", "check_story")
    g.add_conditional_edges("check_story", route_after_story_check, ["draft_story", "organiser_approval"])
    g.add_conditional_edges("organiser_approval", route_after_organiser, ["publish", "draft_story"])
    g.add_edge("publish", END)
    return g.compile(checkpointer=MemorySaver(), interrupt_before=HUMAN_NODES)


def _finish(graph, cfg) -> dict:
    snap = graph.get_state(cfg)
    values = dict(snap.values)
    paused = snap.next[0] if snap.next else None
    if paused == "fraud_review":
        values["status"] = "awaiting_fraud_review"
    elif paused == "organiser_approval":
        values["status"] = "awaiting_organiser_approval"
    return {"state": values, "paused_at": paused}


def start_campaign(form: dict, documents: list, estimate_text: str = "") -> dict:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "campaign"}}
    graph.invoke({"form": form, "documents": documents, "estimate_text": estimate_text, "trace": [],
                  "story_attempts": 0}, cfg)
    return _finish(graph, cfg)


def resume_campaign(state: dict, paused_at: str, decision: dict) -> dict:
    """Continue a paused run. `decision` is {decision, note} for fraud_review or {decision, feedback} for organiser_approval."""
    if paused_at not in RESUME_FROM:
        raise ValueError(f"Cannot resume at {paused_at!r}")
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "campaign"}}
    seed = dict(state)
    if paused_at == "fraud_review":
        seed["review"] = {"decision": decision.get("decision"), "note": decision.get("note", "")}
    else:
        seed["organiser_decision"] = decision.get("decision")
        seed["organiser_feedback"] = decision.get("feedback", "")
    graph.update_state(cfg, seed, as_node=RESUME_FROM[paused_at])
    graph.invoke(None, cfg)
    return _finish(graph, cfg)


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()
