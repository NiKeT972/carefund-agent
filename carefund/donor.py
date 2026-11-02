"""Donor Q&A: a small LangChain RAG chain over the published campaign's knowledge base.

retriever (BM25 over campaign facts + story + platform policy) -> guardrails -> grounded answer.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from .llm import get_llm, llm_failed

_STOP = set("a an the is are was were be to of in on for and or with what how who when where why does do did can i you it this that my your our their there".split())


# Small query-expansion map so everyday donor wording matches the campaign's wording.
_EXPAND = {
    "money": "funds raised paid", "donation": "funds raised", "donate": "funds raised", "go": "paid directly",
    "goes": "paid directly", "spent": "paid directly", "receipt": "80g tax receipt", "tax": "80g receipt deductible",
    "cost": "estimate total", "costs": "estimate total", "need": "gap goal", "needed": "gap goal", "still": "gap", "remaining": "gap", "left": "gap", "refund": "unused refunded",
    "real": "verified verification documents", "genuine": "verified verification documents", "fake": "verified report",
    "doctor": "dr under", "update": "updates", "updates": "updates", "who": "organised organiser",
    "when": "within days", "surgery": "treatment transplant operation", "operation": "treatment surgery",
    "old": "age", "extra": "unused refunded", "leftover": "unused refunded", "organising": "organised organiser",
    "organizer": "organiser organised", "hospital": "hospital",
}

# Words too generic to show that a fact answers the question (almost every fact is about the patient/campaign).
_GENERIC = {"patient", "campaign", "she", "he", "her", "him", "his", "they", "much", "many", "get", "will", "have", "has",
            "happen", "happens", "tell", "me", "about", "know", "please", "any", "fundraiser", "which", "whom"}

NOT_FOUND = "I don't have that information yet. I've passed your question to the organiser."


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 4 and w.endswith("s") else w


def _tok(t: str, expand: bool = False) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _STOP or (expand and w in _EXPAND)]
    if expand:
        words += [x for w in words for x in _EXPAND.get(w, "").split()]
    return [_stem(w) for w in words if w not in _STOP]


def coverage(question: str, fact: str) -> float:
    """Share of the question's meaningful words that the fact contains, directly or through a synonym.

    BM25 always ranks *something* first, even for "What is the weather in Pune?". Without an LLM to judge
    the retrieved facts, this score decides whether the top fact actually answers the question.
    """
    fact_words = set(_tok(fact))
    terms = [w for w in _tok(question) if w not in _GENERIC]
    if not terms:
        return 1.0
    def hit(w):
        return w in fact_words or any(_stem(x) in fact_words for x in _EXPAND.get(w, "").split())
    return sum(hit(w) for w in terms) / len(terms)


class BM25FactRetriever(BaseRetriever):
    """Keyword retriever over short fact strings. Light enough for serverless (no embedding model)."""

    facts: list[str]
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        corpus = [_tok(f) for f in self.facts]
        bm = BM25Okapi(corpus)
        scores = bm.get_scores(_tok(query, expand=True))
        ranked = sorted(range(len(self.facts)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [Document(page_content=self.facts[i], metadata={"score": float(scores[i])}) for i in ranked if scores[i] > 0]


MEDICAL_ADVICE = re.compile(r"\b(surviv\w*|recover\w*|prognosis|cure\w*|chances?|life expectancy|will (it|the surgery|"
                            r"treatment) work|success rate|which hospital is better|what medicine|side effects?|be ok(ay)?|get better)\b")

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer donor questions about one medical crowdfunding campaign. Use ONLY the context. "
     "If the context does not contain the answer, say you don't have that information and that the question has been "
     "passed to the organiser. Never give medical opinions or predictions. Keep answers to 1-3 sentences. "
     "Use ₹ amounts exactly as written in the context."),
    ("human", "Context:\n{context}\n\nDonor question: {question}"),
])


def answer(question: str, knowledge_base: list[str]) -> dict:
    q = question.strip()
    if not q:
        return {"answer": "Please type a question.", "sources": [], "mode": "rules"}
    if MEDICAL_ADVICE.search(q.lower()):
        return {"answer": "I can't give medical opinions or predict outcomes. The treating doctors are the right people "
                          "for that; the organiser will post treatment updates on this page.",
                "sources": [], "mode": "guardrail"}

    docs = BM25FactRetriever(facts=knowledge_base).invoke(q)
    if not docs:
        return {"answer": NOT_FOUND, "sources": [], "mode": "rules"}
    sources = [d.page_content for d in docs]

    llm = get_llm(0)
    if llm is not None:
        try:
            chain = PROMPT | llm
            out = chain.invoke({"context": "\n".join(f"- {s}" for s in sources), "question": q})
            return {"answer": out.content.strip(), "sources": sources, "mode": "llm"}
        except Exception as e:
            llm_failed("donor_answer", e)
    # No LLM: answer with the retrieved fact that best covers the question, or admit we don't know.
    best = max(sources, key=lambda f: coverage(q, f))      # max() keeps the first (best BM25 rank) on ties
    if coverage(q, best) < 0.6:
        return {"answer": NOT_FOUND, "sources": sources, "mode": "retrieval"}
    return {"answer": best, "sources": sources, "mode": "retrieval"}
