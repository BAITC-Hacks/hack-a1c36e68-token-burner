"""Agent 1 "Extractor": one document -> units, roles, atomic functions (each with a verified source clause).

One call over the whole document extracts the structure (units, roles). Functions are extracted
by parallel calls over fragments of ~CHUNK_CLAUSES clauses: a single long call makes the model
sample functions instead of listing all of them. Fragments never split a second-level group
(5.5, 5.5.1, ..., 5.5.14), so the owner heading «Директор ДККМ:» stays with its items.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from pydantic import BaseModel

from backend.agent.llm import LLMCall, LLMError, call_structured
from backend.agent.schemas import Clause, Document, Extraction, ExtractedFunction, TraceStep
from backend.agent.verify import locate_quote
from backend.config import ROOT, settings

PROMPT = (ROOT / "backend" / "prompts" / "extract.md").read_text(encoding="utf-8")
SKIP_CLAUSES = {"preamble", "toc"}
CHUNK_CLAUSES = 40
MAX_PARALLEL_CALLS = 16

STRUCTURE_TASK = ("\n\n## Текущее задание\nЗаполни только `units` и `roles` по всему документу. "
                  "`functions` оставь пустым списком.")
FUNCTIONS_TASK = ("\n\n## Текущее задание\nНа вход подан фрагмент документа. Заполни только `functions`: "
                  "извлеки ВСЕ атомарные функции из КАЖДОГО пункта фрагмента, где есть исполнитель и действие, "
                  "не пропускай пункты и подпункты. `units` и `roles` оставь пустыми списками.")


class DocExtraction(BaseModel):
    doc_id: str
    ok: bool
    error: str | None = None
    extraction: Extraction = Extraction(units=[], roles=[], functions=[])
    repaired: int = 0
    dropped: int = 0


def _content(doc: Document) -> list[Clause]:
    return [c for c in doc.clauses if c.clause_id not in SKIP_CLAUSES]


def render(doc: Document, clauses: list[Clause], context: str = "") -> str:
    lines = [f"[{c.clause_id}] {c.text}" for c in clauses]
    head = f"Документ: «{doc.name}»\n" + (f"Раздел: {context}\n" if context else "")
    return head + "<document>\n" + "\n".join(lines) + "\n</document>"


def render_document(doc: Document) -> str:
    return render(doc, _content(doc))


def chunk_clauses(doc: Document, size: int = CHUNK_CLAUSES) -> list[tuple[str, list[Clause]]]:
    """(section heading, clauses) fragments; second-level groups like 5.5.* are never split."""
    groups: list[list[Clause]] = []
    key = None
    for c in _content(doc):
        k = ".".join(c.clause_id.split(".")[:2])
        if k != key:
            groups.append([])
            key = k
        groups[-1].append(c)
    headings = {c.clause_id: c.text for c in doc.clauses if "." not in c.clause_id}
    chunks: list[tuple[str, list[Clause]]] = []
    for group in groups:
        section = group[0].clause_id.split(".")[0]
        same_section = chunks and chunks[-1][1][0].clause_id.split(".")[0] == section
        if same_section and len(chunks[-1][1]) + len(group) <= size:
            chunks[-1][1].extend(group)
        else:
            chunks.append((headings.get(section, "")[:120], list(group)))
    return chunks


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


def _call(task: str, text: str, client) -> LLMCall | LLMError:
    try:
        return call_structured(model=settings.model_extract, reasoning=settings.reasoning_extract,
                               instructions=PROMPT + task, input=text, schema=Extraction, client=client)
    except LLMError as exc:
        return exc


def extract_document(doc: Document, client=None) -> tuple[DocExtraction, TraceStep]:
    started = datetime.now(timezone.utc)
    step = TraceStep(step=f"extract:{doc.doc_id}", started_at=started, model=settings.model_extract)
    if not doc.clauses:
        step.finished_at, step.notes = datetime.now(timezone.utc), "документ пуст, пропущен"
        return DocExtraction(doc_id=doc.doc_id, ok=False, error="документ не разобран"), step

    jobs = [(STRUCTURE_TASK, render_document(doc))]
    jobs += [(FUNCTIONS_TASK, render(doc, clauses, context)) for context, clauses in chunk_clauses(doc)]
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_CALLS, len(jobs))) as pool:
        calls = list(pool.map(lambda job: _call(*job, client), jobs))

    step.finished_at = datetime.now(timezone.utc)
    step.input_tokens = sum(c.input_tokens for c in calls)
    step.output_tokens = sum(c.output_tokens for c in calls)
    failed = [c for c in calls if isinstance(c, LLMError)]
    ok_calls = [c for c in calls if isinstance(c, LLMCall)]
    merged = Extraction(
        units=calls[0].parsed.units if isinstance(calls[0], LLMCall) else [],
        roles=calls[0].parsed.roles if isinstance(calls[0], LLMCall) else [],
        functions=[f for c in ok_calls[1 if isinstance(calls[0], LLMCall) else 0:] for f in c.parsed.functions],
    )
    extraction, repaired, dropped = clean(doc, merged)
    retries = sum(c.attempts - 1 for c in ok_calls)
    step.notes = (f"{len(extraction.units)} подразделений, {len(extraction.roles)} ролей, "
                  f"{len(extraction.functions)} функций; вызовов {len(calls)} (повторов {retries}, "
                  f"неудачных {len(failed)}); ссылок исправлено {repaired}, отброшено {dropped}")
    error = f"{len(failed)} из {len(calls)} вызовов не удались: {failed[0]}" if failed else None
    return DocExtraction(doc_id=doc.doc_id, ok=not failed, error=error, extraction=extraction,
                         repaired=repaired, dropped=dropped), step


def extract_all(docs: list[Document], client=None) -> tuple[dict[str, DocExtraction], list[TraceStep]]:
    """Extract all documents in parallel; trace keeps the input order."""
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
        (settings.jobs_dir / f"extract_{doc.doc_id}.json").write_text(
            results[doc.doc_id].model_dump_json(indent=2), encoding="utf-8")
    (settings.jobs_dir / "extract_trace.json").write_text(
        json.dumps([s.model_dump(mode="json") for s in trace], ensure_ascii=False, indent=2), encoding="utf-8")
    for step in trace:
        secs = (step.finished_at - step.started_at).total_seconds()
        print(step.step, step.model, f"in={step.input_tokens} out={step.output_tokens} {secs:.1f}s |", step.notes)
