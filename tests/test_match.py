"""Pre-delivery tests 1–6: prematch -> match -> verify on synthetic documents with a fake LLM."""
from backend.agent.match import PROMPT, DupDecision, Verdict, match
from backend.agent.prematch import OwnerNormalizer, prematch
from backend.agent.schemas import AnalysisResult
from backend.agent.verify import verify_result
from tests.conftest import BASE, STRUCTURE, FakeClient, extraction_for, make_doc


def analyse(before_clauses, after_clauses, client_kwargs=None):
    before = make_doc("before-1", "before", before_clauses)
    after = make_doc("after-1", "after", after_clauses)
    docs = [before, after]
    extractions = {d.doc_id: extraction_for(d) for d in docs}
    pm = prematch(docs, extractions)
    client = FakeClient(docs, **(client_kwargs or {}))
    result, step = match(docs, extractions, pm, client=client)
    verified = verify_result(AnalysisResult(documents=[d.info() for d in docs], units=result.units,
                                            functions=result.functions, findings=result.findings),
                             {d.doc_id: d for d in docs})
    return pm, verified, client


def verdict(fid, status, after_ids=(), comment=""):
    return Verdict(before_id=fid, status=status, after_ids=list(after_ids), after_refs=[], search_note="искал везде",
                   comment=comment, severity="medium", recommendation=None)


def findings(result, ftype, verified=True):
    return [f for f in result.findings if f.type == ftype and f.verified == verified]


def test_owner_normalizer_folds_unit_names_in_any_case():
    norm = OwnerNormalizer([extraction_for(make_doc("b", "before", BASE))])
    assert norm.key("Отдел планирования") == "ОП"
    assert norm.key("Начальник отдела планирования (далее Начальник ОП)") == "ОП"
    assert norm.key("работники Отдела контроля") == "ОК"


def test_1_same_document_gives_no_changes():
    pm, result, client = analyse(BASE, BASE)
    assert pm.residual_count == 0 and len(pm.preserved) == len(pm.before)
    assert {u.status for u in result.units} == {"preserved"}
    assert not findings(result, "loss") and not findings(result, "duplication")
    assert all(f.status == "preserved" for f in result.functions)


def test_2_renumbered_clauses_give_no_findings():
    renumbered = STRUCTURE + [("3", "Функции")] + [(f"3.{i}", t) for i, (_, t) in enumerate(BASE[3:], 1)]
    pm, result, _ = analyse(BASE, renumbered)
    assert pm.residual_count == 0
    assert [f.type for f in result.findings] == []


def test_3_moved_function_is_transferred_not_lost():
    moved = [c if c[0] != "2.4" else ("2.4", "Отдел планирования ведет реестр выявленных нарушений.") for c in BASE]
    pm, _, _ = analyse(BASE, moved, {})
    assert len(pm.transfer_candidates) == 1 and not pm.unmatched
    cand = pm.transfer_candidates[0]
    pm, result, _ = analyse(BASE, moved, {"verdicts": lambda ids, text: [verdict(cand.before, "transferred", cand.after)]})
    fn = next(f for f in result.functions if f.id == cand.before)
    assert fn.status == "transferred" and fn.owner_before == "ОК" and fn.owner_after == "ОП"
    assert findings(result, "transfer") and not findings(result, "loss")


def test_3b_verifier_rejects_a_loss_that_exists_in_after_set():
    moved = [c if c[0] != "2.4" else ("2.4", "Отдел планирования ведет реестр выявленных нарушений.") for c in BASE]
    wrong = {"verdicts": lambda ids, text: [verdict(i, "lost") for i in ids]}
    _, result, _ = analyse(BASE, moved, wrong)
    rejected = findings(result, "loss", verified=False)
    assert rejected and "найдена в комплекте «после»" in rejected[0].rejection_reason


def test_4_deleted_clause_is_lost_with_evidence():
    deleted = [c for c in BASE if c[0] != "2.2"]
    pm, _, _ = analyse(BASE, deleted)
    assert len(pm.unmatched) == 1
    lost_id = pm.unmatched[0]
    _, result, _ = analyse(BASE, deleted, {"verdicts": lambda ids, text: [verdict(i, "lost") for i in ids]})
    loss = findings(result, "loss")
    assert len(loss) == 1 and loss[0].function_ids == [lost_id]
    assert [(e.doc_id, e.clause_id) for e in loss[0].evidence] == [("before-1", "2.2")]
    assert 'не найдена в загруженном комплекте "после"' in loss[0].summary
    assert next(f for f in result.functions if f.id == lost_id).status == "lost"


def confirm_all(text):
    import re
    first = re.search(r"## Кандидат в дубли 1\n(.+?)(?:\n## |\n## Что сделать)", text, re.S).group(1)
    ids = re.findall(r"^(A\d+) ", first, re.M)
    return [DupDecision(candidate=1, is_duplicate=True, after_ids=ids, action="вести", object="реестр нарушений",
                        area="управление", reason="общий объект", severity="medium", recommendation=None)]


def test_5_copied_clause_is_duplicated():
    copied = BASE + [("2.5", "Отдел планирования ведет реестр выявленных нарушений.")]
    pm, result, _ = analyse(BASE, copied, {"dup_answer": confirm_all})
    assert pm.duplicate_candidates
    dup = findings(result, "duplication")
    assert len(dup) == 1
    assert {e.clause_id for e in dup[0].evidence} == {"2.4", "2.5"}
    assert any(f.status == "duplicated" for f in result.functions)


def test_6_same_wording_different_area_is_not_duplication():
    areas = BASE + [("2.5", "Отдел контроля проводит проверки ИТ-систем блока."),
                    ("2.6", "Отдел планирования проводит проверки операционных процессов блока.")]
    reject = {"dup_answer": lambda text: [DupDecision(candidate=1, is_duplicate=False, after_ids=[], action="",
                                                      object="", area="", reason="разные области", severity="low",
                                                      recommendation=None)]}
    _, result, _ = analyse(areas, areas, reject)
    assert not findings(result, "duplication")
    assert "одинаковая формулировка в разных областях" in PROMPT


def test_prompt_has_the_three_false_positive_rules_and_definitions():
    for rule in ("Отсутствие пункта ≠ утрата функции", "Перенос ≠ утрата", "Перенумерация и редакционные правки ≠ изменение",
                 "Кандидат на проверку сотрудником", 'Не найдена в загруженном комплекте "после"'):
        assert rule in PROMPT


def test_missing_verdicts_get_one_follow_up_call():
    deleted = [c for c in BASE if c[0] not in ("2.2", "2.4")]
    seen = []

    def lazy(ids, text):  # answers only the first id each time
        seen.append(list(ids))
        return [verdict(ids[0], "lost")]

    _, result, client = analyse(BASE, deleted, {"verdicts": lazy})
    assert len(seen) == 2 and len(seen[1]) == 1
    assert all(f.status == "lost" for f in result.functions if f.id in {i for s in seen for i in s})


CONFLICT_DOC = STRUCTURE + [
    ("2", "Функции"),
    ("2.1", "Отдел контроля проводит проверки закупок."),                 # B1 ОК performs
    ("2.2", "Отдел контроля контролирует качество проверок закупок."),    # B2 ОК controls the same activity
    ("2.3", "Отдел планирования составляет график инвентаризации."),      # B3 ОП performs
    ("2.4", "Отдел планирования оценивает качество инвентаризации."),     # B4 ОП controls the same activity
]


def test_conflict_confidence_is_graded_by_checkable_evidence():
    from backend.agent.match import ConflictCheck, Ref, StructureOutput

    def check(subject, performs, controls, refs=(), staff=()):
        return ConflictCheck(side="before", subject=subject, controls="проверки", control_ids=list(controls),
                             performs_ids=list(performs), performer_roles=list(staff), shared_staff=[],
                             refs=list(refs), is_conflict=True, resolved_in_after=True,
                             comment="Кандидат на проверку сотрудником: совмещение.", severity="high")

    def grade(*checks):
        _, result, _ = analyse(CONFLICT_DOC, CONFLICT_DOC,
                               {"structure": StructureOutput(checks=list(checks), composition_changes=[])})
        return result, [f for f in result.findings if f.type in ("conflict", "overlap")]

    structure_ref = Ref(doc_id="before-1", clause_id="1.1", quote="Блок состоит из Отдела планирования (ОП)")
    result, found = grade(check("Отдел контроля", ["B1"], ["B2"], [structure_ref], ["Отдел планирования"]),
                          check("Отдел планирования", ["B3"], ["B4"]))
    assert {(f.type, f.confidence) for f in found} == {("conflict", "high"), ("conflict", "medium")}
    high = next(f for f in found if f.confidence == "high")
    assert high.verified and "устранён" in high.summary and high.severity == "low"
    assert {e.clause_id for e in high.evidence} == {"2.1", "2.2", "1.1"}
    assert any(f.conflict_of_interest for f in result.functions)

    # performs belongs to another unit -> only the control is the subject's: one condition -> overlap
    result, found = grade(check("Отдел контроля", ["B3"], ["B2"]))
    assert [(f.type, f.confidence, f.severity) for f in found] == [("overlap", "low", "low")]
    assert "Пересечение ответственности" in found[0].summary
    assert not any(f.conflict_of_interest for f in result.functions)

    # nothing owned by the subject, the subject named as its own staff -> dropped
    _, found = grade(check("Рабочая группа", ["B1"], ["B2"], [structure_ref], ["Рабочая группа"]))
    assert found == []

    # a subject outside the org structure is at most an overlap
    _, found = grade(check("Внешний эксперт", ["B1"], ["B2"]))
    assert found == []  # its functions are not its own either


def test_phrase_cuts_the_action_tail_repeated_in_the_object():
    from backend.agent.match import phrase

    assert phrase("контролировать сроки выполнения", "сроки выполнения графика по проверке") == \
        "контролировать сроки выполнения графика по проверке"
    assert phrase("организовать работу проектной команды", "работу проектной команды по проверке") == \
        "организовать работу проектной команды по проверке"
    assert phrase("готовить предложения", "предложения для включения в план работ") == \
        "готовить предложения для включения в план работ"
    assert phrase("запрашивать информацию", "информация у Руководителей Общества") == \
        "запрашивать информацию у Руководителей Общества"          # same word, another case
    assert phrase("анализировать", "результаты непрерывного аудита") == "анализировать результаты непрерывного аудита"
    assert phrase("вести", "реестр договоров; реестр нарушений") == "вести реестр договоров; реестр нарушений"
    assert phrase("проводить проверку", "проверку") == "проводить проверку"  # object fully repeated
    assert phrase("определять цели проверок", "цели проверок, критериев, порядка оценки") == \
        "определять цели проверок, критериев, порядка оценки"             # punctuation of the cut words is kept


def test_findings_use_the_deduplicated_phrase():
    from backend.agent.schemas import ExtractedFunction
    from tests.conftest import extraction_for

    before = make_doc("before-1", "before", BASE)
    after = make_doc("after-1", "after", [c for c in BASE if c[0] != "2.2"])
    docs = [before, after]
    extractions = {d.doc_id: extraction_for(d) for d in docs}
    fn = next(f for f in extractions["before-1"].functions if f.clause_id == "2.2")
    doubled = fn.model_copy(update={"action": "согласует бюджет", "object": "бюджет затрат блока с финансовой службой"})
    extractions["before-1"].functions = [doubled if f is fn else f for f in extractions["before-1"].functions]
    pm = prematch(docs, extractions)
    client = FakeClient(docs, verdicts=lambda ids, text: [verdict(i, "lost") for i in ids])
    result, _ = match(docs, extractions, pm, client=client)
    loss = next(f for f in result.findings if f.type == "loss")
    assert "«согласует бюджет затрат блока с финансовой службой»" in loss.summary
    assert "бюджет бюджет" not in loss.summary
