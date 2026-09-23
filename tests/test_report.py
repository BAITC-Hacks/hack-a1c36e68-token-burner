from backend.agent.report import fallback_summary, render_conclusion, report
from backend.agent.schemas import AnalysisResult, DocumentInfo, Evidence, Finding
from tests.conftest import FakeClient


def result():
    ev = Evidence(doc_id="before-1", clause_id="5.6.2", quote="формировать группы", page=10)
    return AnalysisResult(
        documents=[DocumentInfo(doc_id="before-1", name="red8.pdf", side="before", pages=25, clause_count=1),
                   DocumentInfo(doc_id="after-1", name="red9.pdf", side="after", pages=25, clause_count=1)],
        findings=[Finding(id="F1", type="loss", severity="high", summary="ДККМ: формировать группы — не найдена.",
                          evidence=[ev], verified=True, recommendation="Закрепить право."),
                  Finding(id="F2", type="duplication", severity="low", summary="ложный дубль", evidence=[ev],
                          verified=False, rejection_reason="цитата не совпадает")])


def test_template_uses_only_verified_findings_and_lists_rejected_separately():
    md = render_conclusion(result(), "Абзац.")
    body, rejected = md.split("## Отклонено верификатором (в выводах не учитывается)")
    lines = md.splitlines()
    assert lines[0].startswith("# ") and lines[2] == "_Выводы носят рекомендательный характер и требуют проверки " \
                                                      "ответственным сотрудником._"
    assert "**F1** · важность высокая" in body and "  - До: «red8.pdf» п. 5.6.2 (с. 10)" in body
    assert 'не найдена в загруженном комплекте "после"' in body and "Рекомендация: Закрепить право." in body
    assert "Подтверждено верификатором 1 из 2 выводов: не найдено в «после» — 1." in body
    assert "F2" not in body and "## Дублирование" not in body
    assert "F2: ложный дубль — причина: цитата не совпадает" in rejected
    assert "## " not in rejected  # the rejected block is the last one


def test_incomplete_analysis_is_stated():
    r = result()
    r.analysis_complete, r.warnings = False, ["страница 3 без текста"]
    assert "Анализ неполный" in render_conclusion(r, fallback_summary(r))


def test_forbidden_wording_from_model_falls_back_to_template():
    md, step = report(result(), client=FakeClient(summary="Функция утрачена навсегда."))
    assert "утрачена" not in md and "по шаблону" in step.notes
    md, step = report(result(), client=FakeClient(summary="Выявлены изменения структуры."))
    assert "Выявлены изменения структуры." in md and step.notes == "вводный абзац от модели"


def test_both_editions_are_cited_even_with_many_references():
    r = result()
    many = [Evidence(doc_id="before-1", clause_id=f"5.{i}", quote="q", page=1) for i in range(1, 10)]
    r.findings[0] = r.findings[0].model_copy(update={
        "type": "transfer", "evidence": many + [Evidence(doc_id="after-1", clause_id="5.3.3", quote="q", page=8)]})
    md = render_conclusion(r, "Абзац.")
    assert "  - До: «red8.pdf» п. 5.1 (с. 1)" in md and "и ещё 3" in md
    assert "  - После: «red9.pdf» п. 5.3.3 (с. 8)" in md


def test_missing_recommendations_are_filled_by_type_and_model_ones_kept():
    from backend.agent.report import fill_recommendations

    r = result()
    ev = r.findings[0].evidence
    r.findings = [
        Finding(id="F1", type="loss", severity="high", summary="ДККМ: «x» — не найдена.", evidence=ev, verified=True),
        Finding(id="F2", type="duplication", severity="low", summary="Дублирование «y» (z) у: ДИТААД/ДОА, ДНМ. Текст.",
                evidence=ev, verified=True),
        Finding(id="F3", type="conflict", severity="low", summary="Конфликт интересов в редакции «до» у «ДККМ» (уверенность: high).",
                evidence=ev, verified=True),
        Finding(id="F4", type="overlap", severity="low", summary="Пересечение ответственности в редакции «после» у «ДНМ»: …",
                evidence=ev, verified=True),
        Finding(id="F5", type="change", severity="low", summary="s", evidence=ev, verified=True, recommendation="Своя."),
    ]
    assert fill_recommendations(r) == 4
    rec = [f.recommendation for f in r.findings]
    assert rec[0].startswith("Закрепить функцию за подразделением или зафиксировать намеренное исключение")
    assert rec[1] == "Разграничить зоны ответственности между ДИТААД/ДОА и ДНМ в положениях."
    assert rec[2].startswith("Разделить выполнение и контроль одной деятельности у «ДККМ»")
    assert rec[3].startswith("Уточнить формулировки обязанностей «ДНМ»")
    assert rec[4] == "Своя."
    md, _ = report(r, client=FakeClient(summary="Итог."))
    assert md.count("Рекомендация:") == 5
