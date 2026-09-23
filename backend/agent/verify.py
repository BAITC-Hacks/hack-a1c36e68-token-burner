"""Judge: pure Python, no LLM. Every citation must point at a real clause and quote it verbatim.

A quote passes when, after normalisation (case, ё, punctuation, whitespace, PDF hyphenation):
  * rapidfuzz partial_ratio against the clause text >= settings.quote_min_score, and
  * at most MAX_FOREIGN_WORDS quote words are absent from the clause (partial_ratio alone
    lets a few substituted words through in long quotes).
A clause "contains" its sub-items (a quote from 5.3.3.б verifies against 5.3.3), and a sub-item
quote may start with its parent's lead-in («взаимодействуют ... в части: использования ...»).
A clause that is empty or a single token (e.g. «;») verifies only by exact equality.

On top of quotes, findings must have the right shape: a transfer/change cites both sides,
a duplication cites two different "after" clauses, and a "loss" must not be quoted verbatim
anywhere in the "after" set.
"""
import re

from rapidfuzz import fuzz

from backend.agent.schemas import AnalysisResult, Document, Evidence, Finding, Stats
from backend.config import settings

MAX_FOREIGN_WORDS = 2
MIN_QUOTE_WORDS = 2
SKIP_CLAUSES = ("preamble", "toc")
_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")
_NON_WORD = re.compile(r"[^\w]+")


def normalize(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1-\2", text.lower().replace("ё", "е"))
    return _NON_WORD.sub(" ", text).strip()


def quote_score(quote: str, text: str) -> tuple[float, int]:
    """(partial_ratio, number of quote words missing from text) on normalised strings."""
    q, t = normalize(quote), normalize(text)
    if not q or not t:
        return 0.0, len(q.split())
    words = set(t.split())
    return fuzz.partial_ratio(q, t), sum(w not in words for w in q.split())


def quote_matches(quote: str, text: str) -> bool:
    score, foreign = quote_score(quote, text)
    return score >= settings.quote_min_score and foreign <= MAX_FOREIGN_WORDS


def clause_texts(doc: Document, clause_id: str) -> list[str]:
    """Texts a quote for this clause may come from: itself, itself + sub-items, parent lead-in + itself."""
    clause = doc.clause(clause_id)
    if clause is None:
        return []
    texts = [clause.text, doc.text_with_children(clause_id)]
    parent = doc.clause(clause.parent_id or "")
    if parent:
        texts.append(parent.text + " " + clause.text)
    return texts


def check_evidence(e: Evidence, docs: dict[str, Document]) -> str | None:
    """None if the evidence is valid, otherwise a human-readable rejection reason (Russian)."""
    doc = docs.get(e.doc_id)
    if doc is None:
        return f"документ {e.doc_id} не загружен"
    clause = doc.clause(e.clause_id)
    if clause is None:
        return f"пункт {e.clause_id} не найден в документе «{doc.name}»"
    if e.quote.strip() and e.quote.strip() == clause.text.strip():
        return None
    if len(normalize(e.quote).split()) < MIN_QUOTE_WORDS:
        return f"цитата к п. {e.clause_id} слишком короткая"
    texts = clause_texts(doc, e.clause_id)
    if any(quote_matches(e.quote, t) for t in texts):
        return None
    score, foreign = quote_score(e.quote, texts[1])
    return (f"цитата не совпадает с текстом п. {e.clause_id} «{doc.name}»: сходство {score:.0f}% "
            f"(порог {settings.quote_min_score}%), слов не из пункта: {foreign}")


def locate_quote(doc: Document, quote: str, hint: str | None = None) -> str | None:
    """clause_id that contains the quote: the hinted clause first, then the best match in the document."""
    if hint and doc.clause(hint) and quote_matches(quote, doc.clause(hint).text):
        return hint
    best, best_score = None, 0.0
    for clause in doc.clauses:
        if clause.clause_id in SKIP_CLAUSES:
            continue
        score, foreign = quote_score(quote, clause.text)
        if score >= settings.quote_min_score and foreign <= MAX_FOREIGN_WORDS and score > best_score:
            best, best_score = clause.clause_id, score
    if best is None and hint and any(quote_matches(quote, t) for t in clause_texts(doc, hint)):
        return hint
    return best


def find_in_side(docs: dict[str, Document], side: str, quote: str) -> Evidence | None:
    """First clause of the given side that quotes `quote` verbatim (used to refute 'loss' findings)."""
    for doc in docs.values():
        if doc.side != side:
            continue
        clause_id = locate_quote(doc, quote)
        if clause_id:
            return Evidence(doc_id=doc.doc_id, clause_id=clause_id, quote=quote, page=doc.clause(clause_id).page)
    return None


def _shape_error(f: Finding, docs: dict[str, Document]) -> str | None:
    sides = [docs[e.doc_id].side for e in f.evidence]
    if f.type in ("transfer", "change") and not ("before" in sides and "after" in sides):
        return "перенос/изменение должны ссылаться на пункты обеих редакций"
    if f.type == "loss":
        if "before" not in sides:
            return "потеря должна ссылаться на пункт редакции «до»"
        for e in f.evidence:
            if docs[e.doc_id].side == "before" and len(normalize(e.quote).split()) >= 4:
                found = find_in_side(docs, "after", e.quote)
                if found:
                    return (f"цитата п. {e.clause_id} найдена в комплекте «после» (п. {found.clause_id} "
                            f"«{docs[found.doc_id].name}») — это не потеря")
    if f.type == "duplication":
        after = {(e.doc_id, e.clause_id) for e in f.evidence if docs[e.doc_id].side == "after"}
        if len(after) < 2:
            return "дублирование должно ссылаться минимум на два разных пункта редакции «после»"
    return None


def with_page(e: Evidence, docs: dict[str, Document]) -> Evidence:
    clause = docs[e.doc_id].clause(e.clause_id)
    return e.model_copy(update={"page": clause.page})


def verify_finding(f: Finding, docs: dict[str, Document]) -> Finding:
    if not f.evidence:
        return f.model_copy(update={"verified": False, "rejection_reason": "нет ни одной ссылки на пункт"})
    reasons = [r for r in (check_evidence(e, docs) for e in f.evidence) if r]
    if not reasons:
        shape = _shape_error(f, docs)
        reasons = [shape] if shape else []
    if reasons:
        return f.model_copy(update={"verified": False, "rejection_reason": "; ".join(reasons)})
    return f.model_copy(update={"verified": True, "rejection_reason": None,
                                "evidence": [with_page(e, docs) for e in f.evidence]})


def verify_result(result: AnalysisResult, docs: dict[str, Document]) -> AnalysisResult:
    """Verify all findings, recompute stats; rejected findings stay in the result."""
    findings = [verify_finding(f, docs) for f in result.findings]
    verified = sum(f.verified for f in findings)
    stats = result.stats.model_copy(update={"findings_total": len(findings), "verified": verified,
                                            "rejected": len(findings) - verified})
    return result.model_copy(update={"findings": findings, "stats": stats})
