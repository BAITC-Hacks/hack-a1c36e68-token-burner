"""Pre-delivery test 7: the verifier (pure Python, no LLM)."""
import pytest

from backend.agent.schemas import AnalysisResult, Clause, Document, Evidence, Finding
from backend.agent.verify import check_evidence, locate_quote, verify_finding, verify_result

TEXT = ("готовит отчеты об итогах выполнения плана работы БВА на ежеквартальной основе и по итогам года "
        "в соответствии с требованиями настоящего Положения;")


@pytest.fixture
def docs():
    doc = Document(doc_id="before-1", name="red8.pdf", side="before", pages=1, clauses=[
        Clause(clause_id="5.5", text="Директор ДККМ:", page=9),
        Clause(clause_id="5.5.5", text=TEXT, page=9, parent_id="5.5"),
        Clause(clause_id="5.3.3", text="организует руководство проверками, в том числе:", page=8),
        Clause(clause_id="5.3.3.б", text="определение целей проверок, критериев, порядка оценки эффективности "
                                         "бизнес-процессов;", page=8, parent_id="5.3.3"),
        Clause(clause_id="3.10", text="Работники могут выполнять функции с использованием информационно- "
                                      "телекоммуникационных сетей общего пользования", page=6),
    ])
    return {doc.doc_id: doc}


def finding(clause_id, quote):
    return Finding(id="F1", type="loss", severity="high", summary="s",
                   evidence=[Evidence(doc_id="before-1", clause_id=clause_id, quote=quote)])


def test_quote_exact_is_accepted(docs):
    f = verify_finding(finding("5.5.5", "на ежеквартальной основе и по итогам года"), docs)
    assert f.verified and f.rejection_reason is None
    assert f.evidence[0].page == 9  # page is filled from the parsed clause


def test_fake_clause_id_is_rejected(docs):
    f = verify_finding(finding("5.5.15", "на ежеквартальной основе и по итогам года"), docs)
    assert not f.verified
    assert "5.5.15 не найден" in f.rejection_reason


def test_unknown_document_is_rejected(docs):
    e = Evidence(doc_id="after-7", clause_id="5.5.5", quote="на ежеквартальной основе")
    assert "не загружен" in check_evidence(e, docs)


def test_quote_mismatch_is_rejected(docs):
    # three words changed: ежеквартальной -> ежемесячной, года -> квартала, Положения -> Регламента
    quote = ("готовит отчеты об итогах выполнения плана работы БВА на ежемесячной основе и по итогам квартала "
             "в соответствии с требованиями настоящего Регламента")
    f = verify_finding(finding("5.5.5", quote), docs)
    assert not f.verified
    assert "не совпадает" in f.rejection_reason


def test_three_changed_words_in_short_quote_are_rejected(docs):
    f = verify_finding(finding("5.5.5", "готовит справки об итогах исполнения бюджета работы БВА"), docs)
    assert not f.verified


def test_punctuation_and_whitespace_changes_are_accepted(docs):
    quote = ("Готовит  отчёты об итогах выполнения плана работы БВА — на ежеквартальной основе, и по итогам года,\n"
             "в соответствии с требованиями настоящего положения.")
    assert verify_finding(finding("5.5.5", quote), docs).verified


def test_pdf_hyphenation_is_accepted(docs):
    quote = "с использованием информационно-телекоммуникационных сетей общего пользования"
    assert verify_finding(finding("3.10", quote), docs).verified


def test_quote_from_sub_item_verifies_against_parent(docs):
    assert verify_finding(finding("5.3.3", "определение целей проверок, критериев"), docs).verified


def test_too_short_quote_is_rejected(docs):
    assert not verify_finding(finding("5.5", "ДККМ"), docs).verified


def test_one_bad_evidence_rejects_the_finding(docs):
    f = finding("5.5.5", "на ежеквартальной основе и по итогам года")
    f.evidence.append(Evidence(doc_id="before-1", clause_id="9.99", quote="несуществующий пункт"))
    assert not verify_finding(f, docs).verified


def test_verify_result_keeps_rejected_and_counts(docs):
    result = AnalysisResult(findings=[finding("5.5.5", "по итогам года в соответствии с требованиями"),
                                      finding("1.1", "что-то чего нет")])
    out = verify_result(result, docs)
    assert len(out.findings) == 2
    assert (out.stats.findings_total, out.stats.verified, out.stats.rejected) == (2, 1, 1)


def test_locate_quote_repairs_clause_id(docs):
    doc = docs["before-1"]
    assert locate_quote(doc, "порядка оценки эффективности бизнес-процессов", hint="5.3.3") == "5.3.3.б"
    assert locate_quote(doc, "на ежеквартальной основе", hint="5.5.5") == "5.5.5"
    assert locate_quote(doc, "совершенно другой текст про закупки") is None


def two_side_docs():
    before = Document(doc_id="before-1", name="b.pdf", side="before", pages=1, clauses=[
        Clause(clause_id="5.5.3", text=";", page=9),
        Clause(clause_id="5.6.2", text="формировать группы контроля качества с привлечением работников БВА;", page=10)])
    after = Document(doc_id="after-1", name="a.pdf", side="after", pages=1, clauses=[
        Clause(clause_id="5.3.3", text="готовят предложения для включения в план работ БВА;", page=8),
        Clause(clause_id="5.4.2", text="готовит предложения для включения в план работ БВА;", page=9)])
    return {d.doc_id: d for d in (before, after)}


def ev(doc_id, clause_id, quote):
    return Evidence(doc_id=doc_id, clause_id=clause_id, quote=quote)


def test_empty_clause_verifies_only_by_exact_text():
    docs = two_side_docs()
    assert check_evidence(ev("before-1", "5.5.3", ";"), docs) is None
    assert check_evidence(ev("before-1", "5.5.3", ","), docs)


def test_transfer_needs_both_sides():
    docs = two_side_docs()
    one_side = Finding(id="F1", type="transfer", severity="low", summary="s",
                       evidence=[ev("before-1", "5.6.2", "формировать группы контроля качества")])
    assert "обеих редакций" in verify_finding(one_side, docs).rejection_reason


def test_duplication_needs_two_after_clauses():
    docs = two_side_docs()
    dup = Finding(id="F1", type="duplication", severity="low", summary="s",
                  evidence=[ev("after-1", "5.3.3", "готовят предложения для включения в план работ БВА")])
    assert not verify_finding(dup, docs).verified
    dup.evidence.append(ev("after-1", "5.4.2", "готовит предложения для включения в план работ БВА"))
    assert verify_finding(dup, docs).verified


def test_loss_quoted_in_after_set_is_rejected():
    docs = two_side_docs()
    docs["after-1"].clauses.append(Clause(clause_id="5.7.1", text="формировать группы контроля качества с привлечением "
                                                                    "работников БВА;", page=10))
    loss = Finding(id="F1", type="loss", severity="high", summary="s",
                   evidence=[ev("before-1", "5.6.2", "формировать группы контроля качества с привлечением работников БВА")])
    out = verify_finding(loss, docs)
    assert not out.verified and "5.7.1" in out.rejection_reason
