"""ingest -> extract -> prematch -> match -> verify -> report, with progress events and trace."""
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from backend.agent import extract as extract_mod
from backend.agent.match import match
from backend.agent.prematch import notes as prematch_notes, prematch
from backend.agent.report import report
from backend.agent.schemas import AnalysisResult, Document, TraceStep
from backend.agent.verify import verify_result
from backend.config import settings
from backend.ingest import ingest_set

ProgressFn = Callable[[str, float, str], None]
INCOMPLETE_LOSS = ("Входные данные обработаны не полностью: отсутствие функции в комплекте «после» "
                   "не подтверждается")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _cache_path(doc: Document) -> Path:
    key = "\n".join([settings.model_extract, settings.reasoning_extract, extract_mod.PROMPT,
                     str(extract_mod.CHUNK_CLAUSES), extract_mod.render_document(doc)])
    return settings.jobs_dir / "cache" / f"extract_{hashlib.sha256(key.encode()).hexdigest()[:20]}.json"


def run_extract(docs: list[Document], client=None, use_cache: bool = True):
    results, trace, todo = {}, [], []
    for doc in docs:
        path = _cache_path(doc)
        if use_cache and path.exists():
            cached = extract_mod.DocExtraction.model_validate_json(path.read_text(encoding="utf-8"))
            results[doc.doc_id] = cached.model_copy(update={"doc_id": doc.doc_id})
            trace.append(TraceStep(step=f"extract:{doc.doc_id}", started_at=_now(), finished_at=_now(),
                                   model=settings.model_extract, input_tokens=0, output_tokens=0, cost_usd=0.0,
                                   notes=f"из кэша: {len(cached.extraction.functions)} функций"))
        else:
            todo.append(doc)
    if todo:
        fresh, steps = extract_mod.extract_all(todo, client=client)
        results.update(fresh)
        trace += steps
        for doc in todo:
            if fresh[doc.doc_id].ok:
                _cache_path(doc).parent.mkdir(parents=True, exist_ok=True)
                _cache_path(doc).write_text(fresh[doc.doc_id].model_dump_json(), encoding="utf-8")
    order = {d.doc_id: i for i, d in enumerate(docs)}
    trace.sort(key=lambda s: order.get(s.step.split(":", 1)[-1], 0))
    return results, trace


def run(before: list[str | Path], after: list[str | Path], progress: ProgressFn | None = None,
        client=None, use_cache: bool = True) -> AnalysisResult:
    say = progress or (lambda step, fraction, message: None)
    started = time.monotonic()
    trace: list[TraceStep] = []
    warnings: list[str] = []

    say("ingest", 0.02, "Чтение документов")
    t = _now()
    docs = ingest_set(before, "before") + ingest_set(after, "after")
    for d in docs:
        warnings += [f"«{d.name}»: {w}" for w in d.warnings]
    trace.append(TraceStep(step="ingest", started_at=t, finished_at=_now(),
                           notes=", ".join(f"{d.name}: {len(d.clauses)} пунктов" for d in docs)))
    result = AnalysisResult(documents=[d.info() for d in docs], trace=trace)
    if not any(d.side == "before" and d.clauses for d in docs) or not any(d.side == "after" and d.clauses for d in docs):
        result.analysis_complete = False
        result.warnings = warnings + ["Нужен хотя бы один читаемый документ в каждом комплекте"]
        return result

    say("extract", 0.08, f"Извлечение функций из {len(docs)} документов")
    extracted, steps = run_extract(docs, client=client, use_cache=use_cache)
    trace += steps
    warnings += [f"Извлечение «{d.name}»: {extracted[d.doc_id].error}" for d in docs if not extracted[d.doc_id].ok]

    say("prematch", 0.45, "Детерминированное сопоставление функций")
    t = _now()
    extractions = {k: v.extraction for k, v in extracted.items()}
    pm = prematch(docs, extractions)
    trace.append(TraceStep(step="prematch", started_at=t, finished_at=_now(), notes=prematch_notes(pm)))

    say("match", 0.5, f"Сопоставление остатка ({pm.residual_count} функций) моделью")
    matched, step = match(docs, extractions, pm, client=client)
    trace.append(step)
    warnings += matched.warnings

    say("verify", 0.88, "Проверка цитат")
    t = _now()
    complete = not warnings and matched.complete
    result = AnalysisResult(documents=[d.info() for d in docs], units=matched.units, functions=matched.functions,
                            findings=matched.findings, analysis_complete=complete, warnings=warnings, trace=trace)
    result = verify_result(result, {d.doc_id: d for d in docs})
    if not complete:  # never confirm losses on incomplete input
        result.findings = [f.model_copy(update={"verified": False, "rejection_reason": INCOMPLETE_LOSS})
                           if f.type == "loss" and f.verified else f for f in result.findings]
        verified = sum(f.verified for f in result.findings)
        result.stats.verified, result.stats.rejected = verified, len(result.findings) - verified
    trace.append(TraceStep(step="verify", started_at=t, finished_at=_now(),
                           notes=f"подтверждено {result.stats.verified}, отклонено {result.stats.rejected}"))

    say("report", 0.92, "Формирование заключения")
    result.conclusion_md, step = report(result, client=client)
    trace.append(step)

    costs = [s.cost_usd for s in trace if s.model]
    result.stats.cost_usd = None if any(c is None for c in costs) else round(sum(costs), 4)
    result.stats.duration_s = round(time.monotonic() - started, 1)
    result.trace = trace
    say("done", 1.0, "Готово")
    return result
