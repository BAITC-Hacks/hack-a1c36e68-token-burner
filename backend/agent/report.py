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
]
FORBIDDEN = re.compile(r"утрач|уничтож|ликвидир", re.I)
MAX_CITES = 4


class ReportSummary(BaseModel):
    summary: str


def cite(e: Evidence, names: dict[str, str]) -> str:
    page = f", с. {e.page}" if e.page else ""
    return f"«{names.get(e.doc_id, e.doc_id)}», п. {e.clause_id}{page}"


def fallback_summary(result: AnalysisResult) -> str:
    ok = [f for f in result.findings if f.verified]
    count = {t: sum(f.type == t for f in ok) for t, _ in SECTIONS}
    text = (f"Сравнение выявило {len(ok)} подтверждённых выводов: изменений структуры — {count['structure']}, "
            f'функций, не найденных в загруженном комплекте "после", — {count["loss"]}, изменённых функций — '
            f"{count['change']}, групп перенесённых функций — {count['transfer']}, случаев дублирования — "
            f"{count['duplication']}, кандидатов на проверку конфликта интересов — {count['conflict']}.")
    if not result.analysis_complete:
        text = "Анализ неполный: часть входных данных не обработана, выводы о потерях предварительные. " + text
    return text


def render_conclusion(result: AnalysisResult, summary: str) -> str:
    names = {d.doc_id: d.name for d in result.documents}
    before = ", ".join(f"«{d.name}»" for d in result.documents if d.side == "before")
    after = ", ".join(f"«{d.name}»" for d in result.documents if d.side == "after")
    lines = ["# Заключение по сравнению организационных документов", "",
             f"Комплект «до»: {before or '—'}. Комплект «после»: {after or '—'}.", ""]
    if not result.analysis_complete:
        lines += ["> **Анализ неполный.** " + " ".join(result.warnings or ["Часть данных не обработана."]), ""]
    lines += [summary, ""]
    ok = [f for f in result.findings if f.verified]
    for ftype, title in SECTIONS:
        group = [f for f in ok if f.type == ftype]
        if not group:
            continue
        lines += [f"## {title}", ""]
        for f in group:
            refs = "; ".join(cite(e, names) for e in f.evidence[:MAX_CITES])
            more = f" и ещё {len(f.evidence) - MAX_CITES}" if len(f.evidence) > MAX_CITES else ""
            lines.append(f"- **{f.id}** ({f.severity}). {f.summary} [{refs}{more}]")
            if f.recommendation:
                lines.append(f"  - Рекомендация: {f.recommendation}")
        lines.append("")
    rejected = [f for f in result.findings if not f.verified]
    if rejected:
        lines += ["## Отклонено верификатором (в выводы не включено)", ""]
        lines += [f"- {f.id}: {f.summary} — причина: {f.rejection_reason}" for f in rejected]
        lines.append("")
    lines.append(f"_Подтверждено верификатором: {len(ok)} из {len(result.findings)}._")
    return "\n".join(lines)


def _summary_input(result: AnalysisResult) -> str:
    ok = [f for f in result.findings if f.verified]
    head = f"Анализ полный: {'да' if result.analysis_complete else 'нет'}. Подтверждённых выводов: {len(ok)}."
    body = "\n".join(f"- [{f.type}, {f.severity}] {f.summary[:300]}" for f in ok[:80])
    return head + "\n" + body


def report(result: AnalysisResult, client=None) -> tuple[str, TraceStep]:
    step = TraceStep(step="report", started_at=datetime.now(timezone.utc), model=settings.model_report)
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
