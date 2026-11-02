"""Exercise the LLM code paths with a stub model (no network)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from langchain_core.messages import AIMessage
import carefund.graph as G, carefund.donor as D
from carefund.samples import CLEAN, RISKY


class Stub:
    def with_structured_output(self, schema):
        class R:
            def invoke(_, prompt):
                n = schema.__name__
                if n == "CaseFacts":
                    return schema(category="organ_transplant", condition_plain="kidneys have stopped working",
                                  treatment_plain="a new kidney from her husband", urgency_days=45,
                                  family_situation="Her husband is retired and the son is the only earner.",
                                  claims=["2.5 lakh arranged"])
                if n == "LLMReview":
                    return schema(concerns=["Estimate is from a different hospital than the form"])
                if n == "Story":
                    return schema(title="Help Sunita Patil get a kidney transplant", summary="s",
                                  body=("Sunita is being treated at Sahyadri Care Hospital. We need ₹5,00,000. " * 20))
                raise AssertionError(n)
        return R()

    def __call__(self, prompt_value):
        return AIMessage(content="Funds go directly to Sahyadri Care Hospital.")


def test_llm_paths(monkeypatch):
    monkeypatch.setattr(G, "get_llm", lambda *a, **k: Stub())
    monkeypatch.setattr(D, "get_llm", lambda *a, **k: Stub())
    r = G.start_campaign(CLEAN["form"], CLEAN["documents"], CLEAN["estimate_text"])
    modes = {t["node"]: t["mode"] for t in r["state"]["trace"]}
    assert modes["extract_case"] == "llm" and modes["draft_story"] == "llm"
    assert r["state"]["case"]["family_situation"]
    r = G.resume_campaign(r["state"], "organiser_approval", {"decision": "approve"})
    a = D.answer("where does money go", r["state"]["campaign"]["knowledge_base"])
    assert a["mode"] == "llm"
    rr = G.start_campaign(RISKY["form"], RISKY["documents"], RISKY["estimate_text"])
    assert any(f["code"] == "llm_review" for f in rr["state"]["flags"])


def test_llm_errors_fall_back(monkeypatch):
    class Broken:
        def with_structured_output(self, s): raise RuntimeError("down")
        def __call__(self, p): raise RuntimeError("down")
    monkeypatch.setattr(G, "get_llm", lambda *a, **k: Broken())
    r = G.start_campaign(CLEAN["form"], CLEAN["documents"], CLEAN["estimate_text"])
    assert r["paused_at"] == "organiser_approval"


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    from dotenv import load_dotenv
    import carefund.llm as L
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=from-dotenv-file\n")
    monkeypatch.setenv("GROQ_API_KEY", "")
    load_dotenv(env, override=False)            # same call llm.py makes at import
    assert os.environ["GROQ_API_KEY"] == ""     # the existing (empty) value wins
    assert not L.llm_available() and L.get_llm() is None


def test_llm_errors_are_logged_and_donor_falls_back(monkeypatch, caplog):
    class Broken:
        def with_structured_output(self, s):
            class R:
                def invoke(_, p): raise ValueError("tool_use_failed")   # Groq's error for malformed tool JSON
            return R()
        def __call__(self, p): raise RuntimeError("down")
    monkeypatch.setattr(G, "get_llm", lambda *a, **k: Broken())
    monkeypatch.setattr(D, "get_llm", lambda *a, **k: Broken())
    with caplog.at_level("WARNING", logger="carefund"):
        r = G.start_campaign(CLEAN["form"], CLEAN["documents"], CLEAN["estimate_text"])
        r = G.resume_campaign(r["state"], "organiser_approval", {"decision": "approve"})
        a = D.answer("where does the money go", r["state"]["campaign"]["knowledge_base"])
    assert all(t["mode"] != "llm" for t in r["state"]["trace"])
    assert a["mode"] == "retrieval" and a["answer"]
    for node in ("extract_case", "draft_story", "donor_answer"):
        assert f"{node} failed" in caplog.text
