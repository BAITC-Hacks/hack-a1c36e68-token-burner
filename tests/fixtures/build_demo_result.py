"""Builds the hand-written mock data/demo_result.json for the frontend (red8 -> red9).

Quotes are checked against the parsed sample clauses, so the mock stays verifiable.
Run: python -m tests.fixtures.build_demo_result
Once the pipeline works, `python -m backend.cli ... -o data/demo_result.json` replaces this.
"""
import json
from datetime import datetime, timedelta, timezone

from rapidfuzz import fuzz

from backend.agent.schemas import (
    AnalysisResult, Evidence, Finding, Function, Stats, TraceStep, Unit,
)
from backend.config import ROOT, settings
from backend.ingest import ingest_file

BEFORE = ingest_file(ROOT / "data/samples/red8.pdf", "before", "before-1")
AFTER = ingest_file(ROOT / "data/samples/red9.pdf", "after", "after-1")
DOCS = {"before-1": BEFORE, "after-1": AFTER}
SHORT = {"before-1": "ред. 8", "after-1": "ред. 9"}


def ev(doc_id: str, clause_id: str, quote: str | None = None, *, must_match: bool = True) -> Evidence:
    clause = DOCS[doc_id].clause(clause_id)
    assert clause, f"{doc_id} {clause_id} missing"
    quote = quote or clause.text
    if must_match:
        assert quote in clause.text, f"{doc_id} {clause_id}: quote not in text"
    return Evidence(doc_id=doc_id, clause_id=clause_id, quote=quote, page=clause.page)


B = lambda cid, q=None: ev("before-1", cid, q)  # noqa: E731
A = lambda cid, q=None: ev("after-1", cid, q)  # noqa: E731

DITAAD_DOA = "ДИТААД и ДОА (Директоры департаментов и направлений)"

UNITS = [
    Unit(name="Департамент ИТ-аудита и анализа данных (ДИТААД)", status="created",
         after_ref="ДИТААД", evidence=[A("3.4.а")]),
    Unit(name="Департамент операционного аудита (ДОА)", status="created",
         after_ref="ДОА", evidence=[A("3.4.б")]),
    Unit(name="Департамент непрерывного мониторинга СВК (ДНМ)", status="preserved",
         before_ref="ДНМ", after_ref="ДНМ", evidence=[B("3.4.а"), A("3.4.в")]),
    Unit(name="Департамент контроля качества аудита и методологии (ДККМ)", status="preserved",
         before_ref="ДККМ", after_ref="ДККМ",
         comment="Подразделение сохранено, но из его состава исключены исполнители аудита "
                 "«Директор проектов ДККМ» и «Менеджер по аудиту».",
         evidence=[B("3.8.г", "Менеджер по аудиту."), A("3.9.б", "Руководитель направления.")]),
    Unit(name="Директор направления внутреннего аудита", status="transformed",
         before_ref="Директор направления внутреннего аудита", after_ref=DITAAD_DOA,
         comment="Должность упразднена, её функции распределены между руководителями ДИТААД и ДОА.",
         evidence=[B("3.5.а", "Директор направления внутреннего аудита."),
                   B("5.3", "Директор направления внутреннего аудита:"),
                   A("5.3", "Директоры департаментов и Директоры направлений ДИТААД и ДОА:")]),
    Unit(name="Главный аудитор", status="preserved", before_ref="Главный аудитор", after_ref="Главный аудитор",
         evidence=[B("1.4"), A("1.4")]),
]

FUNCTIONS = [
    Function(id="fn-1", action="формировать", object="группы контроля качества", area="контроль качества аудита",
             owner_before="ДККМ", status="lost",
             comment='Не найдена в загруженном комплекте "после".',
             evidence=[B("5.6.2", "формировать группы контроля качества с привлечением работников БВА")],
             recommendation="Закрепить право формирования групп контроля качества за Директором ДККМ в п. 5.6."),
    Function(id="fn-2", action="выносить предложения", object="объём и содержание внешней оценки БВА",
             area="внешняя оценка качества", owner_before="ДККМ", status="lost",
             comment='Не найдена в загруженном комплекте "после"; п. 11.5 сохраняет лишь право Главного аудитора корректировать объём.',
             evidence=[B("5.6.3", "выносить предложения по объему и содержанию внешней оценки БВА Главному аудитору"),
                       A("11.5", "Объем и содержание внешней оценки могут быть скорректированы по усмотрению Главного аудитора")],
             recommendation="Определить, кто готовит предложения по объёму внешней оценки качества."),
    Function(id="fn-3", action="доводить до сведения", object="результаты консультационных услуг",
             area="консультирование руководства", owner_before="ДНМ", status="lost",
             comment='Не найдена в загруженном комплекте "после".',
             evidence=[B("5.7.2", "доводить до сведения Руководителей Общества результаты по запросу оказания консультационных услуг")],
             recommendation="Вернуть в права Директора ДНМ или закрепить за другим подразделением."),
    Function(id="fn-4", action="взаимодействует", object="субъекты СВК (Карта гарантий)", area="координация с СВК",
             owner_before="ДНМ", owner_after=DITAAD_DOA, status="transferred",
             evidence=[B("5.4.4", "взаимодействует с субъектами СВК Общества в части:"),
                       A("5.3.3", "взаимодействуют с субъектами СВК Общества в части:")]),
    Function(id="fn-5", action="осуществлять контроль по поручению Главного аудитора", object="выполнение проверки",
             area="надзор за проверками", owner_before="Директор направления внутреннего аудита",
             owner_after="Директор операционного аудита", status="transferred",
             evidence=[B("9.37", "может поручить/делегировать соответствующие функции Директору направления внутреннего аудита"),
                       A("9.37", "может поручить соответствующие функции Директору операционного аудита")],
             recommendation="Уточнить наименование: в п. 3.5 ред. 9 должность называется «Директор ДОА»."),
    Function(id="fn-6", action="готовит предложения", object="план работ БВА", area="планирование",
             owner_before="ДККМ", owner_after=f"{DITAAD_DOA}; ДНМ", status="transferred",
             evidence=[B("5.5.10", "готовит предложения для включения в план работ БВА"),
                       A("5.3.3", "готовят предложения для включения в план работ БВА"),
                       A("5.4.2", "готовит предложения для включения в план работ БВА")]),
    Function(id="fn-7", action="выносит предложения", object="повышение профессионального уровня работников БВА",
             area="развитие персонала", owner_before="ДККМ", owner_after=f"{DITAAD_DOA}; ДНМ", status="transferred",
             evidence=[B("5.5.8", "выносит предложения по повышению профессионального уровня работников БВА"),
                       A("5.3.9", "выносят предложения по повышению профессионального уровня работников БВА")]),
    Function(id="fn-8", action="готовит отчёты", object="итоги выполнения плана работы БВА", area="отчётность",
             owner_before="ДККМ", owner_after="ДККМ", status="changed",
             comment="Ежеквартальная периодичность у ДККМ снята; ежеквартальные отчёты представляет Главный аудитор (п. 5.1.6).",
             evidence=[B("5.5.5", "на ежеквартальной основе и по итогам года"),
                       A("5.5.3", "готовит отчеты об итогах выполнения плана работы БВА в соответствии с требованиями настоящего Положения"),
                       A("5.1.6", "на ежеквартальной основе и по итогам года")]),
    Function(id="fn-9", action="анализирует", object="результаты непрерывного аудита", area="непрерывный аудит",
             owner_before="ДККМ", owner_after=f"{DITAAD_DOA}; ДНМ", status="duplicated",
             evidence=[B("5.5.4", "анализирует результаты непрерывного аудита"),
                       A("5.3.8", "анализируют результаты проверок БВА и непрерывного аудита"),
                       A("5.4.5", "анализирует результаты непрерывного аудита")],
             recommendation="Разграничить: ДНМ — анализ данных мониторинга, ДИТААД/ДОА — использование в проверках."),
    Function(id="fn-10", action="организует контроль", object="устранение недостатков и нарушений по итогам проверок",
             area="мониторинг корректирующих мер", owner_before="Главный аудитор",
             owner_after=f"Главный аудитор; {DITAAD_DOA}; ДККМ", status="duplicated",
             evidence=[A("5.1.4", "организует контроль устранения недостатков и нарушений"),
                       A("5.3.7", "организуют контроль устранения недостатков и нарушений"),
                       A("5.5.5", "организует контроль качества устранения недостатков и нарушений")],
             recommendation="Назначить одного владельца процесса контроля устранения недостатков."),
    Function(id="fn-11", action="проводит", object="аудиторские проверки", area="внутренний аудит",
             owner_before="ДККМ (Менеджер по аудиту, Директор проектов ДККМ)", owner_after=DITAAD_DOA,
             status="transferred", conflict_of_interest=True,
             comment="В ред. 8 ДККМ одновременно участвовал в проверках и контролировал их качество; "
                     "в ред. 9 исполнители аудита из ДККМ исключены.",
             evidence=[B("3.8.г", "Менеджер по аудиту."), B("5.5.2", "организует непрерывный мониторинг качества деятельности внутреннего аудита"),
                       A("3.9", "Директору ДККМ подчиняются работники ДККМ")]),
]

FINDINGS = [
    Finding(id="F1", type="structure", severity="medium", function_ids=["fn-5"],
            summary="Должность «Директор направления внутреннего аудита» упразднена; её функции переданы "
                    "Директорам департаментов и направлений ДИТААД и ДОА (преобразование, не ликвидация функций).",
            evidence=UNITS[4].evidence),
    Finding(id="F2", type="loss", severity="high", function_ids=["fn-1"],
            summary='Право формировать группы контроля качества (ДККМ) не найдено в загруженном комплекте "после".',
            evidence=FUNCTIONS[0].evidence, recommendation=FUNCTIONS[0].recommendation),
    Finding(id="F3", type="loss", severity="medium", function_ids=["fn-2"],
            summary='Право ДККМ выносить предложения по объёму и содержанию внешней оценки БВА не найдено в загруженном комплекте "после".',
            evidence=FUNCTIONS[1].evidence, recommendation=FUNCTIONS[1].recommendation),
    Finding(id="F4", type="loss", severity="medium", function_ids=["fn-3"],
            summary='Право ДНМ доводить результаты консультаций до Руководителей Общества не найдено в загруженном комплекте "после".',
            evidence=FUNCTIONS[2].evidence, recommendation=FUNCTIONS[2].recommendation),
    Finding(id="F5", type="transfer", severity="low", function_ids=["fn-4"],
            summary="Взаимодействие с субъектами СВК (Карта гарантий) перенесено от ДНМ к ДИТААД и ДОА.",
            evidence=FUNCTIONS[3].evidence),
    Finding(id="F6", type="transfer", severity="medium", function_ids=["fn-5"],
            summary="Делегирование контроля над выполнением проверки перенесено с «Директора направления внутреннего аудита» "
                    "на «Директора операционного аудита»; такая должность в разделе 3 ред. 9 названа иначе.",
            evidence=FUNCTIONS[4].evidence, recommendation=FUNCTIONS[4].recommendation),
    Finding(id="F7", type="duplication", severity="medium", function_ids=["fn-9"],
            summary="Анализ результатов непрерывного аудита закреплён одновременно за ДИТААД/ДОА и ДНМ.",
            evidence=FUNCTIONS[8].evidence[1:], recommendation=FUNCTIONS[8].recommendation),
    Finding(id="F8", type="duplication", severity="medium", function_ids=["fn-10"],
            summary="Контроль устранения недостатков закреплён за тремя субъектами: Главный аудитор, ДИТААД/ДОА, ДККМ.",
            evidence=FUNCTIONS[9].evidence, recommendation=FUNCTIONS[9].recommendation),
    Finding(id="F9", type="conflict", severity="low", function_ids=["fn-11"],
            summary="Кандидат на проверку сотрудником: в ред. 8 ДККМ и выполнял проверки (Менеджер по аудиту), и контролировал "
                    "их качество (надзор над выполнением проверок). В ред. 9 конфликт устранён реорганизацией.",
            evidence=[B("3.8.г", "Менеджер по аудиту."),
                      B("10.7.а", "осуществляют надзор над выполнением проверок и прочей деятельностью внутреннего аудита"),
                      A("3.9.а", "Директор проектов."), A("3.9.б", "Руководитель направления.")]),
]

# A finding the verifier must reject: the quote does not match clause 5.5.5 (and the function exists in 5.1.6).
BAD_QUOTE = "готовит ежемесячные отчеты о загрузке аудиторов и направляет их Президенту Общества"
bad = ev("before-1", "5.5.5", BAD_QUOTE, must_match=False)
score = fuzz.partial_ratio(BAD_QUOTE, BEFORE.clause("5.5.5").text)
assert score < settings.quote_min_score
REJECTED = Finding(
    id="F10", type="loss", severity="high", verified=False,
    summary="Ежемесячная отчётность ДККМ перед Президентом утрачена.",
    evidence=[bad],
    rejection_reason=f"Цитата не совпадает с текстом п. 5.5.5 ({SHORT['before-1']}): "
                     f"сходство {score:.0f}% < {settings.quote_min_score}%.",
)


def cite(e: Evidence) -> str:
    return f"{SHORT[e.doc_id]}, п. {e.clause_id}, с. {e.page}"


def conclusion(findings: list[Finding]) -> str:
    ok = [f for f in findings if f.verified]
    sections = [("structure", "Изменения структуры"), ("loss", 'Функции, не найденные в комплекте "после"'),
                ("transfer", "Перенесённые функции"), ("duplication", "Дублирование"),
                ("conflict", "Конфликт интересов (кандидаты на проверку сотрудником)")]
    lines = [
        "# Заключение по реорганизации БВА (ред. 8 → ред. 9)",
        "",
        "Реорганизация выделила два новых департамента (ДИТААД и ДОА) и упразднила должность Директора направления "
        "внутреннего аудита; основная часть его функций перенесена в новые департаменты. Три права ДККМ и ДНМ не найдены в "
        "новой редакции, два участка работы закреплены за несколькими подразделениями одновременно. Конфликт "
        "интересов ДККМ, существовавший в ред. 8, устранён.",
        "",
    ]
    for ftype, title in sections:
        group = [f for f in ok if f.type == ftype]
        if not group:
            continue
        lines += [f"## {title}", ""]
        for f in group:
            refs = "; ".join(cite(e) for e in f.evidence)
            lines.append(f"- **{f.id}** ({f.severity}). {f.summary} [{refs}]")
            if f.recommendation:
                lines.append(f"  - Рекомендация: {f.recommendation}")
        lines.append("")
    lines.append(f"_Подтверждено верификатором: {len(ok)} из {len(findings)}. "
                 "Отклонённые выводы в заключение не включены._")
    return "\n".join(lines)


def build() -> AnalysisResult:
    findings = [f.model_copy(update={"verified": True}) for f in FINDINGS] + [REJECTED]
    t0 = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
    steps = [("ingest", None, 2, None, None, "2 документа, 885 пунктов"),
             ("extract", settings.model_extract, 41, 38210, 9120, "параллельно по документам"),
             ("match", settings.model_match, 96, 61480, 7340, "сопоставление функций и структуры"),
             ("verify", None, 1, None, None, "9 подтверждено, 1 отклонён"),
             ("report", settings.model_report, 12, 5210, 610, "шаблон + вводный абзац")]
    trace, t = [], t0
    for step, model, secs, tin, tout, notes in steps:
        trace.append(TraceStep(step=step, started_at=t, finished_at=t + timedelta(seconds=secs), model=model,
                               input_tokens=tin, output_tokens=tout, notes=notes))
        t += timedelta(seconds=secs)
    for doc, name in ((BEFORE, "red8.pdf"), (AFTER, "red9.pdf")):
        doc.name = name
    return AnalysisResult(
        documents=[BEFORE.info(), AFTER.info()],
        units=UNITS,
        functions=FUNCTIONS,
        findings=findings,
        conclusion_md=conclusion(findings),
        stats=Stats(findings_total=len(findings), verified=sum(f.verified for f in findings),
                    rejected=sum(not f.verified for f in findings)),
        analysis_complete=True,
        trace=trace,
    )


if __name__ == "__main__":
    result = build()
    settings.demo_result_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{settings.demo_result_path}: {result.stats}")
