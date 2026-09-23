"""Agent 1 "Extractor": one document -> units, roles, atomic functions (each with a verified source clause)."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from pydantic import BaseModel

from backend.agent.llm import LLMError, call_structured
from backend.agent.schemas import Document, Extraction, ExtractedFunction, TraceStep
from backend.agent.verify import locate_quote
from backend.config import ROOT, settings

PROMPT = (ROOT / "backend" / "prompts" / "extract.md").read_text(encoding="utf-8")
SKIP_CLAUSES = {"preamble", "toc"}


class DocExtraction(BaseModel):
    doc_id: str
    ok: bool
    error: str | None = None
    extraction: Extraction = Extraction(units=[], roles=[], functions=[])
    repaired: int = 0
    dropped: int = 0


def render_document(doc: Document) -> str:
    lines = [f"[{c.clause_id}] {c.text}" for c in doc.clauses if c.clause_id not in SKIP_CLAUSES]
    return f"Документ: «{doc.name}»\n<document>\n" + "\n".join(lines) + "\n</document>"


def clean(doc: Document, extraction: Extraction) -> tuple[Extraction, int, int]:
    """Point every function at the clause that really contains its quote; drop unsupported ones."""
    functions: list[ExtractedFunction] = []
    seen, repaired, dropped = set(), 0, 0
    for fn in extraction.functions:
        clause_id = locate_quote(doc, fn.quote, hint=fn.clause_id)
        if clause_id is None:
            dropped += 1
            continue
        if clause_id != fn.clause_id:
            repaired += 1
            fn = fn.model_copy(update={"clause_id": clause_id})
        key = (fn.owner.lower(), fn.action.lower(), fn.object.lower(), fn.clause_id)
        if key not in seen:
            seen.add(key)
            functions.append(fn)
    known = {c.clause_id for c in doc.clauses}
    units = [u.model_copy(update={"clause_ids": [c for c in u.clause_ids if c in known]}) for u in extraction.units]
    roles = [r.model_copy(update={"clause_ids": [c for c in r.clause_ids if c in known]}) for r in extraction.roles]
    return Extraction(units=units, roles=roles, functions=functions), repaired, dropped


def extract_document(doc: Document, client=None) -> tuple[DocExtraction, TraceStep]:
    started = datetime.now(timezone.utc)
    step = TraceStep(step=f"extract:{doc.doc_id}", started_at=started, model=settings.model_extract)
    if not doc.clauses:
        step.finished_at, step.notes = datetime.now(timezone.utc), "документ пуст, пропущен"
        return DocExtraction(doc_id=doc.doc_id, ok=False, error="документ не разобран"), step
    try:
        call = call_structured(model=settings.model_extract, reasoning=settings.reasoning_extract,
                               instructions=PROMPT, input=render_document(doc), schema=Extraction, client=client)
    except LLMError as exc:
        step.finished_at = datetime.now(timezone.utc)
        step.input_tokens, step.output_tokens = exc.input_tokens, exc.output_tokens
        step.notes = f"ошибка после повторной попытки: {exc}"
        return DocExtraction(doc_id=doc.doc_id, ok=False, error=str(exc)), step
    extraction, repaired, dropped = clean(doc, call.parsed)
    step.finished_at = datetime.now(timezone.utc)
    step.input_tokens, step.output_tokens = call.input_tokens, call.output_tokens
    step.notes = (f"{len(extraction.units)} подразделений, {len(extraction.roles)} ролей, "
                  f"{len(extraction.functions)} функций; ссылок исправлено {repaired}, отброшено {dropped}; "
                  f"попыток {call.attempts}, {call.seconds:.1f} с")
    return DocExtraction(doc_id=doc.doc_id, ok=True, extraction=extraction, repaired=repaired, dropped=dropped), step


def extract_all(docs: list[Document], client=None) -> tuple[dict[str, DocExtraction], list[TraceStep]]:
    """Extract all documents in parallel; results keep the input order."""
    with ThreadPoolExecutor(max_workers=max(1, len(docs))) as pool:
        results = list(pool.map(lambda d: extract_document(d, client), docs))
    return {r.doc_id: r for r, _ in results}, [step for _, step in results]


if __name__ == "__main__":
    # python -m backend.agent.extract before.pdf after.pdf -> data/jobs/extract_<doc_id>.json
    from backend.ingest import ingest_file

    paths = sys.argv[1:]
    docs = [ingest_file(p, side, doc_id=f"{side}-1") for p, side in zip(paths, ("before", "after"))]
    results, trace = extract_all(docs)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        r = results[doc.doc_id]
        (settings.jobs_dir / f"extract_{doc.doc_id}.json").write_text(
            r.model_dump_json(indent=2), encoding="utf-8")
    (settings.jobs_dir / "extract_trace.json").write_text(
        json.dumps([s.model_dump(mode="json") for s in trace], ensure_ascii=False, indent=2), encoding="utf-8")
    for step in trace:
        print(step.step, step.model, step.input_tokens, step.output_tokens, step.notes)
