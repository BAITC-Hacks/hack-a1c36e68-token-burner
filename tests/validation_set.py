"""Validation set from CLAUDE.md for the red8 -> red9 sample (test data only — the backend knows nothing of it).

python -m tests.validation_set data/demo_result.json   -> prints the found / missed / false table
"""
import json
import sys
from dataclasses import dataclass

from backend.agent.schemas import AnalysisResult, Finding

BEFORE, AFTER = "before", "after"


@dataclass
class Row:
    item: str
    expected: str
    verdict: str  # найдено | пропущено | ложное
    detail: str
    required: bool = True


def _side(result: AnalysisResult, doc_id: str) -> str:
    return next(d.side for d in result.documents if d.doc_id == doc_id)


def _cites(f: Finding, result: AnalysisResult, side: str, clause: str) -> bool:
    return any(_side(result, e.doc_id) == side and (e.clause_id == clause or e.clause_id.startswith(clause + "."))
               for e in f.evidence)


def _findings(result: AnalysisResult, ftype: str, side: str, clause: str) -> list[Finding]:
    return [f for f in result.findings if f.verified and f.type == ftype and _cites(f, result, side, clause)]


def _fn_status(result: AnalysisResult, clause: str) -> set[str]:
    return {f.status for f in result.functions
            if any(_side(result, e.doc_id) == BEFORE and (e.clause_id == clause or e.clause_id.startswith(clause + "."))
                   for e in f.evidence[:1])}


def _unit(result: AnalysisResult, needle: str):
    return next((u for u in result.units if needle.lower() in f"{u.name} {u.before_ref} {u.after_ref}".lower()), None)


def check(result: AnalysisResult) -> list[Row]:
    rows = []
    for abbr, clause in (("ДИТААД", "3.4.а"), ("ДОА", "3.4.б")):
        u = _unit(result, abbr)
        rows.append(Row(f"{abbr} создан (red9 {clause})", "created",
                        "найдено" if u and u.status == "created" else "пропущено", u.status if u else "нет в units"))
    for abbr in ("ДНМ", "ДККМ"):
        u = _unit(result, abbr)
        ok = u and u.status == "preserved"
        rows.append(Row(f"{abbr} сохранён", "preserved", "найдено" if ok else ("ложное" if u else "пропущено"),
                        u.status if u else "нет в units"))
    u = _unit(result, "Директор направления внутреннего аудита")
    verdict = "найдено" if u and u.status == "transformed" else ("ложное" if u and u.status == "eliminated" else "пропущено")
    rows.append(Row("Директор направления ВА → ДИТААД/ДОА (red8 3.5.а, 5.3)", "transformed", verdict,
                    f"{u.status} → {u.after_ref}" if u else "нет в units"))
    comp = [f for f in result.findings if f.verified and f.type == "structure" and "ДККМ" in f.summary
            and "аудит" in f.summary.lower() and ("исключ" in f.summary.lower() or "состав" in f.summary.lower())]
    rows.append(Row("ДККМ лишился исполнителей аудита (red8 3.8 → red9 3.9)", "structure",
                    "найдено" if comp else "пропущено", comp[0].id if comp else ""))

    for clause, what in (("5.6.2", "право формировать группы контроля качества"),
                         ("5.6.3", "предложения по объёму внешней оценки"),
                         ("5.7.2", "доведение результатов консультаций")):
        found = _findings(result, "loss", BEFORE, clause)
        rows.append(Row(f"red8 {clause} {what}", "loss", "найдено" if found else "пропущено",
                        ", ".join(f.id for f in found) or f"статус: {_fn_status(result, clause) or '—'}"))

    for clause, expected in (("5.5.10", "changed/transferred"), ("5.5.8", "changed/transferred"),
                             ("5.5.5", "changed"), ("5.4.4", "transferred → 5.3.3"), ("9.37", "transferred")):
        lost = _findings(result, "loss", BEFORE, clause)
        status = _fn_status(result, clause)
        moved = status & {"transferred", "changed"}
        verdict = "ложное" if lost else ("найдено" if moved else "пропущено")
        detail = f"в потерях: {', '.join(f.id for f in lost)}" if lost else f"статусы функций: {sorted(status) or '—'}"
        if clause == "5.4.4":
            target = [f for f in result.findings if f.verified and f.type in ("transfer", "change")
                      and _cites(f, result, BEFORE, clause) and _cites(f, result, AFTER, "5.3.3")]
            detail += f"; ссылка на red9 5.3.3: {'да' if target else 'нет'}"
        rows.append(Row(f"red8 {clause} не потеря", expected, verdict, detail))

    for clauses, what in ((("5.3.8", "5.4.5"), "анализ результатов непрерывного аудита"),
                          (("5.1.4", "5.3.7", "5.5.5"), "контроль устранения недостатков")):
        dups = [f for f in result.findings if f.verified and f.type == "duplication"]
        best = max(dups, key=lambda f: sum(_cites(f, result, AFTER, c) for c in clauses), default=None)
        hit = [c for c in clauses if best and _cites(best, result, AFTER, c)]
        verdict = "найдено" if len(hit) == len(clauses) else ("частично" if len(hit) >= 2 else "пропущено")
        rows.append(Row(f"дубль red9 {' / '.join(clauses)} ({what})", "duplication", verdict,
                        f"{best.id}: {', '.join(hit)}" if best and hit else ""))

    conf = [f for f in result.findings if f.verified and f.type == "conflict" and "ДККМ" in f.summary
            and any(_side(result, e.doc_id) == BEFORE for e in f.evidence)]
    resolved = [f for f in conf if "устранён" in f.summary]
    downgraded = [f for f in result.findings if f.verified and f.type == "overlap" and "ДККМ" in f.summary
                  and any(_side(result, e.doc_id) == BEFORE for e in f.evidence)]
    rows.append(Row("конфликт ДККМ в red8, устранён в red9", "conflict (resolved)",
                    "найдено" if resolved else ("ложное" if downgraded else ("частично" if conf else "пропущено")),
                    ", ".join(f"{f.id} {f.confidence}" for f in conf + downgraded)))
    dual = [f for f in result.findings if f.verified and _cites(f, result, BEFORE, "3.6")
            and f.type in ("conflict", "overlap")]
    conflict = [f for f in dual if f.type == "conflict"]
    rows.append(Row("red8 3.6 двойное подчинение", "conflict (не overlap)",
                    "найдено" if conflict else ("ложное" if dual else "пропущено"),
                    ", ".join(f"{f.id} {f.type}/{f.confidence}" for f in dual)))
    zdo = [f for f in result.findings if f.verified and _cites(f, result, AFTER, "4.4")]
    rows.append(Row("red9 4.4 раскрытие КИ Главного аудитора в ДЗО", "conflict/change",
                    "найдено" if zdo else "пропущено", ", ".join(f.id for f in zdo), required=False))
    defect = [f for f in result.findings if f.verified and _cites(f, result, BEFORE, "5.5.3") and "пуст" in f.summary]
    rows.append(Row("red8 5.5.3 пустой пункт", "structure (low)", "найдено" if defect else "пропущено",
                    ", ".join(f.id for f in defect), required=False))
    return rows


def extra_losses(result: AnalysisResult) -> list[Finding]:
    expected = ("5.6.2", "5.6.3", "5.7.2")
    return [f for f in result.findings if f.verified and f.type == "loss"
            and not any(_cites(f, result, BEFORE, c) for c in expected)]


def table(result: AnalysisResult) -> str:
    rows = check(result)
    lines = ["| пункт Validation set | ожидается | результат | детали |", "|---|---|---|---|"]
    lines += [f"| {r.item}{'' if r.required else ' (желательно)'} | {r.expected} | {r.verdict} | {r.detail} |"
              for r in rows]
    extra = extra_losses(result)
    if extra:
        lines.append("\nПотери вне Validation set (проверить вручную):")
        lines += [f"- {f.id}: {f.summary}" for f in extra]
    lines.append("\nКонфликты и пересечения:")
    lines += [f"- {f.id} {f.type}/{f.confidence}: {f.summary[:220]}" for f in result.findings
              if f.type in ("conflict", "overlap")]
    return "\n".join(lines)


if __name__ == "__main__":
    res = AnalysisResult.model_validate(json.load(open(sys.argv[1], encoding="utf-8")))
    print(table(res))
    print(f"\nstats: {res.stats.model_dump()}  complete={res.analysis_complete}")
