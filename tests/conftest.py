"""Synthetic documents + a fake OpenAI client that answers every agent by its output schema."""
import re
import threading
from types import SimpleNamespace

import pytest

from backend.agent.match import DupOutput, StructureOutput, VerdictOutput
from backend.agent.schemas import Clause, Document, Extraction, ExtractedFunction, ExtractedUnit


def make_doc(doc_id: str, side: str, clauses: list[tuple[str, str]]) -> Document:
    return Document(doc_id=doc_id, name=f"{doc_id}.txt", side=side, pages=None,
                    clauses=[Clause(clause_id=c, text=t, page=None,
                                    parent_id=c.rsplit(".", 1)[0] if c.count(".") > 1 else None)
                             for c, t in clauses])


# Two departments; each clause «Отдел X <verb> ...» gives one function owned by that unit.
STRUCTURE = [
    ("1", "Структура"),
    ("1.1", "Блок состоит из Отдела планирования (ОП) и Отдела контроля (ОК)."),
]
BASE = STRUCTURE + [
    ("2", "Функции"),
    ("2.1", "Отдел планирования разрабатывает годовой план работ блока."),
    ("2.2", "Отдел планирования согласует бюджет затрат блока с финансовой службой."),
    ("2.3", "Отдел контроля проверяет исполнение плана работ блока."),
    ("2.4", "Отдел контроля ведет реестр выявленных нарушений."),
]
UNITS = [ExtractedUnit(name="Отдел планирования (ОП)", short_name="ОП", kind="division", parent=None, clause_ids=["1.1"]),
         ExtractedUnit(name="Отдел контроля (ОК)", short_name="ОК", kind="division", parent=None, clause_ids=["1.1"])]
_FN = re.compile(r"^(Отдел \w+)\s+(\w+)\s+(.+?)\.?$")


def extraction_for(doc: Document) -> Extraction:
    """What a perfect extractor returns for the synthetic documents."""
    functions = []
    for c in doc.clauses:
        m = _FN.match(c.text)
        if m:
            owner, verb, obj = m.groups()
            area = "ИТ-аудит" if "ИТ" in obj else ("операционный аудит" if "операцион" in obj else "управление")
            functions.append(ExtractedFunction(owner=owner, modality="duty", action=verb, object=obj, area=area,
                                               clause_id=c.clause_id, quote=f"{verb} {obj}"))
    return Extraction(units=UNITS, roles=[], functions=functions)


class FakeClient:
    """client.responses.parse replacement. `verdicts(ids)` decides statuses, `dup_answer(cands)` duplicates."""

    def __init__(self, docs=(), verdicts=None, dup_answer=None, structure=None, summary="Сводка."):
        self.docs = {d.doc_id: d for d in docs}
        self.verdicts = verdicts or (lambda ids, text: [])
        self.dup_answer = dup_answer or (lambda text: [])
        self.structure = structure or StructureOutput(checks=[], composition_changes=[])
        self.summary = summary
        self.calls = []
        self.lock = threading.Lock()
        self.responses = self

    def parse(self, **kw):
        with self.lock:
            self.calls.append(kw)
        schema, text = kw["text_format"], kw["input"]
        if schema is Extraction:
            doc = next(d for d in self.docs.values() if f"«{d.name}»" in text)
            parsed = extraction_for(doc)
            if "Заполни только `units`" in kw["instructions"]:
                parsed = Extraction(units=parsed.units, roles=[], functions=[])
            else:
                parsed = Extraction(units=[], roles=[], functions=[f for f in parsed.functions
                                                                   if f"[{f.clause_id}]" in text])
        elif schema is VerdictOutput:
            ids = re.search(r"вердикт ровно для \d+ функций: ([^.]+)\.", text).group(1).split(", ")
            parsed = VerdictOutput(verdicts=self.verdicts(ids, text))
        elif schema is DupOutput:
            parsed = DupOutput(decisions=self.dup_answer(text))
        elif schema is StructureOutput:
            parsed = self.structure
        else:  # report summary
            parsed = schema(summary=self.summary)
        return SimpleNamespace(output_parsed=parsed, status="completed", incomplete_details=None,
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                                     input_tokens_details=SimpleNamespace(cached_tokens=0)))


@pytest.fixture
def base_clauses():
    return list(BASE)
