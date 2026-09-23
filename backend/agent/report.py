"""Agent 3 "Reporter": conclusion = template over verified findings + one LLM-written summary paragraph."""
import re
from datetime import datetime, timezone

from pydantic import BaseModel

from backend.agent.llm import LLMError, call_structured, cost_usd
from backend.agent.schemas import AnalysisResult, Evidence, Finding, TraceStep
from backend.config import ROOT, settings

PROMPT = (ROOT / "backend" / "prompts" / "report.md").read_text(encoding="utf-8")
SECTIONS = [
    ("structure", "Структура"),
    ("loss", 'Функции, не найденные в загруженном комплекте "после"'),
    ("change", "Изменённые и суженные функции"),
    ("transfer", "Перенесённые функции"),
    ("duplication", "Дублирование функций"),
    ("conflict", "Конфликт интересов — кандидаты на проверку сотрудником"),
    ("overlap", "Пересечение ответственности"),
]
REJECTED_TITLE = "Отклонено верификатором (в выводах не учитывается)"
FORBIDDEN = re.compile(r"утрач|уничтож|ликвидир", re.I)


class ReportSummary(BaseModel):
    summary: str


DISCLAIMER = "_Выводы носят рекомендательный характер и требуют проверки ответственным сотрудником._"
NOT_FOUND = 'не найдена в загруженном комплекте "после"'
LEVEL = {"high": "высокая", "medium": "средняя", "low": "низкая"}
COUNT_LABELS = {"structure": "структура", "loss": "не найдено в «после»", "change": "изменено",
                "transfer": "перенесено", "duplication": "дублирование", "conflict": "конфликт интересов",
                "overlap": "пересечение ответственности"}
MAX_CITES_PER_SIDE = 6


def _subject(summary: str) -> str:
    m = re.search(r"у «([^»]+)»", summary)
    return m.group(1) if m else "подразделения"


def _owners(summary: str) -> str:
    m = re.search(r"у: ([^.]+)\.", summary)
    names = [x.strip() for x in m.group(1).split(",")] if m else []
    return " и ".join(names) if names else "подразделениями"


def default_recommendation(f: Finding) -> str:
    """One line per finding when the model gave none: built from the finding type and the unit."""
    if f.type == "loss":
        return ("Закрепить функцию за подразделением или зафиксировать намеренное исключение "
                "в распорядительном документе.")
    if f.type == "duplication":
        return f"Разграничить зоны ответственности между {_owners(f.summary)} в положениях."
    if f.type == "conflict":
        return (f"Разделить выполнение и контроль одной деятельности у «{_subject(f.summary)}»: "
                f"передать контроль независимому подразделению или зафиксировать компенсирующие меры.")
    if f.type == "overlap":
        return f"Уточнить формулировки обязанностей «{_subject(f.summary)}», чтобы разграничить выполнение и контроль."
    if f.type == "transfer":
        return "Отразить перенос в положении принимающего подразделения и в должностных инструкциях."
    if f.type == "change":
        return "Подтвердить, что изменение объёма функции сделано намеренно, и отразить его в положениях."
    return "Отразить изменение структуры в положениях о подразделениях и штатном расписании."


def fill_recommendations(result: AnalysisResult) -> int:
    """Fill missing recommendations in place; model-written ones are kept. Returns how many were filled."""
    filled = 0
    for f in result.findings:
        if not (f.recommendation or "").strip():
            f.recommendation = default_recommendation(f)
            filled += 1
    return filled


def cite(e: Evidence) -> str:
    return f"п. {e.clause_id}" + (f" (с. {e.page})" if e.page else "")


def _citations(f: Finding, docs: dict[str, tuple[str, str]]) -> list[str]:
    """One line per side, grouped by document, so both editions are always visible."""
    lines = []
    for side, label in (("before", "До"), ("after", "После")):
        evidence = [e for e in f.evidence if docs.get(e.doc_id, ("", ""))[1] == side]
        if not evidence:
            continue
        shown, rest = evidence[:MAX_CITES_PER_SIDE], len(evidence) - MAX_CITES_PER_SIDE
        by_doc: dict[str, list[str]] = {}
        for e in shown:
            by_doc.setdefault(docs[e.doc_id][0], []).append(cite(e))
        text = "; ".join(f"«{name}» {', '.join(refs)}" for name, refs in by_doc.items())
        lines.append(f"  - {label}: {text}" + (f" и ещё {rest}" if rest > 0 else ""))
    return lines


def _label(f: Finding) -> str:
    if f.type == "conflict" and f.confidence:
        return f"уверенность {LEVEL[f.confidence]}"
    return f"важность {LEVEL[f.severity]}"


def _summary(f: Finding) -> str:
    if f.type == "loss" and NOT_FOUND not in f.summary:
        return f"{f.summary.rstrip('.')} — {NOT_FOUND}."
    return f.summary


def fallback_summary(result: AnalysisResult) -> str:
    ok = [f for f in result.findings if f.verified]
    text = (f"Сравнение дало {len(ok)} подтверждённых выводов; ниже они сгруппированы по типам, "
            f"у каждого указаны пункты обеих редакций.")
    if not result.analysis_complete:
        text = "Анализ неполный: часть входных данных не обработана, выводы о потерях предварительные. " + text
    return text


def render_conclusion(result: AnalysisResult, summary: str) -> str:
    docs = {d.doc_id: (d.name, d.side) for d in result.documents}
    before = ", ".join(f"«{d.name}»" for d in result.documents if d.side == "before")
    after = ", ".join(f"«{d.name}»" for d in result.documents if d.side == "after")
    ok = [f for f in result.findings if f.verified]
    counts = ", ".join(f"{COUNT_LABELS[t]} — {n}" for t, _ in SECTIONS if (n := sum(f.type == t for f in ok)))
    lines = ["# Заключение по сравнению организационных документов", "", DISCLAIMER, "",
             f"Комплект «до»: {before or '—'}. Комплект «после»: {after or '—'}.",
             f"Подтверждено верификатором {len(ok)} из {len(result.findings)} выводов"
             + (f": {counts}." if counts else "."), ""]
    if not result.analysis_complete:
        lines += ["> **Анализ неполный.** " + " ".join(result.warnings or ["Часть данных не обработана."]), ""]
    lines += [summary, ""]
    for ftype, title in SECTIONS:
        group = [f for f in ok if f.type == ftype]
        if not group:
            continue
        lines += [f"## {title}", ""]
        for f in group:
            lines.append(f"- **{f.id}** · {_label(f)}. {_summary(f)}")
            lines += _citations(f, docs)
            if f.recommendation:
                lines.append(f"  - Рекомендация: {f.recommendation}")
        lines.append("")
    rejected = [f for f in result.findings if not f.verified]
    if rejected:  # always the last block
        lines += [f"## {REJECTED_TITLE}", ""]
        lines += [f"- {f.id}: {f.summary} — причина: {f.rejection_reason}" for f in rejected]
    return "\n".join(lines).rstrip() + "\n"


def _summary_input(result: AnalysisResult) -> str:
    ok = [f for f in result.findings if f.verified]
    head = f"Анализ полный: {'да' if result.analysis_complete else 'нет'}. Подтверждённых выводов: {len(ok)}."
    body = "\n".join(f"- [{f.type}, {f.severity}] {f.summary[:300]}" for f in ok[:80])
    return head + "\n" + body


def report(result: AnalysisResult, client=None) -> tuple[str, TraceStep]:
    step = TraceStep(step="report", started_at=datetime.now(timezone.utc), model=settings.model_report)
    fill_recommendations(result)
    summary, note = fallback_summary(result), "вводный абзац по шаблону"
    try:
        call = call_structured(model=settings.model_report, reasoning="low", instructions=PROMPT,
                               input=_summary_input(result), schema=ReportSummary, client=client,
                               max_output_tokens=4000)
        step.input_tokens, step.output_tokens, step.cached_tokens = call.input_tokens, call.output_tokens, call.cached_tokens
        text = call.parsed.summary.strip()
        if text and not FORBIDDEN.search(text):
            summary, note = text, "вводный абзац от модели"
        else:
            note = "ответ модели отклонён (запрещённые формулировки), абзац по шаблону"
    except LLMError as exc:
        step.input_tokens, step.output_tokens = exc.input_tokens, exc.output_tokens
        note = f"модель недоступна ({exc}), абзац по шаблону"
    step.cost_usd = cost_usd(settings.model_report, step.input_tokens or 0, step.output_tokens or 0,
                             step.cached_tokens or 0)
    step.finished_at = datetime.now(timezone.utc)
    step.notes = note
    return render_conclusion(result, summary), step
