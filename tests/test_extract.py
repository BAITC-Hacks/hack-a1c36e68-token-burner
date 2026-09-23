"""Extractor with a fake OpenAI client: prompt input, retry, graceful failure, citation cleanup."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.agent.extract import extract_all, extract_document, render_document
from backend.agent.schemas import Clause, Document, Extraction, ExtractedFunction, ExtractedUnit


def make_doc(doc_id="before-1"):
    return Document(doc_id=doc_id, name="unit.docx", side="before", pages=None, clauses=[
        Clause(clause_id="preamble", text="УТВЕРЖДЕНО Советом директоров", page=None),
        Clause(clause_id="1", text="Структура", page=None),
        Clause(clause_id="1.1", text="Блок состоит из Отдела учета (ОУ).", page=None),
        Clause(clause_id="2.1", text="Начальник ОУ анализирует результаты проверок и готовит предложения в план работ;",
               page=None),
        Clause(clause_id="2.1.а", text="ведет реестр договоров;", page=None, parent_id="2.1"),
        Clause(clause_id="toc", text="1. СТРУКТУРА 1", page=None),
    ])


def fn(clause_id, quote, action="анализировать"):
    return ExtractedFunction(owner="Начальник ОУ", modality="duty", action=action, object="результаты",
                             area="учет", clause_id=clause_id, quote=quote)


def extraction(*functions):
    return Extraction(units=[ExtractedUnit(name="Отдел учета", short_name="ОУ", kind="division", parent=None,
                                           clause_ids=["1.1", "9.9"])],
                      roles=[], functions=list(functions))


class FakeClient:
    """Mimics client.responses.parse; each item in `script` is a parsed object or an exception."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []
        self.responses = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(output_parsed=item, status="completed", incomplete_details=None,
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def validation_error():
    try:
        Extraction.model_validate({"units": "oops"})
    except ValidationError as exc:
        return exc


def test_input_excludes_preamble_and_toc():
    text = render_document(make_doc())
    assert "[1.1] Блок состоит" in text and "[2.1.а] ведет реестр" in text
    assert "УТВЕРЖДЕНО" not in text and "[toc]" not in text


def test_success_logs_tokens_and_cleans_citations():
    client = FakeClient(extraction(
        fn("2.1", "анализирует результаты проверок"),                      # correct
        fn("2.1", "ведет реестр договоров", action="вести"),              # wrong clause -> repaired to 2.1.а
        fn("2.1", "утверждает бюджет департамента", action="утверждать"),  # nowhere in the doc -> dropped
        fn("2.1", "анализирует результаты проверок"),                      # duplicate -> removed
    ))
    result, step = extract_document(make_doc(), client=client)
    assert result.ok
    assert [(f.action, f.clause_id) for f in result.extraction.functions] == [("анализировать", "2.1"), ("вести", "2.1.а")]
    assert (result.repaired, result.dropped) == (1, 1)
    assert result.extraction.units[0].clause_ids == ["1.1"]
    assert (step.input_tokens, step.output_tokens) == (100, 20)
    assert step.finished_at and step.model and "попыток 1" in step.notes
    assert client.calls[0]["text_format"] is Extraction
    assert client.calls[0]["reasoning"]["effort"]


def test_one_retry_on_validation_error():
    client = FakeClient(validation_error(), extraction(fn("2.1", "анализирует результаты проверок")))
    result, step = extract_document(make_doc(), client=client)
    assert result.ok and len(client.calls) == 2
    assert "попыток 2" in step.notes


def test_second_failure_is_graceful():
    client = FakeClient(validation_error(), validation_error())
    result, step = extract_document(make_doc(), client=client)
    assert not result.ok and result.error
    assert result.extraction.functions == []
    assert "ошибка" in step.notes and len(client.calls) == 2


def test_extract_all_runs_every_document():
    client = FakeClient(*[extraction(fn("2.1", "анализирует результаты проверок")) for _ in range(2)])
    results, trace = extract_all([make_doc("before-1"), make_doc("after-1")], client=client)
    assert set(results) == {"before-1", "after-1"} and all(r.ok for r in results.values())
    assert [s.step for s in trace] == ["extract:before-1", "extract:after-1"]


@pytest.mark.integration
def test_extract_samples_live():
    from backend.agent.verify import check_evidence
    from backend.agent.schemas import Evidence
    from backend.config import ROOT
    from backend.ingest import ingest_file

    docs = [ingest_file(ROOT / "data/samples/red8.pdf", "before", "before-1"),
            ingest_file(ROOT / "data/samples/red9.pdf", "after", "after-1")]
    results, _ = extract_all(docs)
    by_id = {d.doc_id: d for d in docs}
    names = {k: " | ".join(f"{u.name} {u.short_name or ''}" for u in r.extraction.units) for k, r in results.items()}
    for abbr in ("ДИТААД", "ДОА", "ДНМ", "ДККМ"):
        assert abbr in names["after-1"]
    for abbr in ("ДНМ", "ДККМ"):
        assert abbr in names["before-1"]
    assert any("директор направления внутреннего аудита" in r.name.lower()
               for r in results["before-1"].extraction.roles)
    for doc_id, r in results.items():
        for f in r.extraction.functions:
            assert check_evidence(Evidence(doc_id=doc_id, clause_id=f.clause_id, quote=f.quote), by_id) is None
