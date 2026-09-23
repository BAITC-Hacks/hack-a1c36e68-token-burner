"""Deterministic stand-in for the OpenAI client (`client.responses.parse`) for pipeline tests.

It "reads" the synthetic fixtures in tests/fixtures/org, where every function clause is written as
`<АББР> <глагол> <объект>[ в области <область>].` and every unit as `<Название> (<АББР>).`
Answers are dispatched by the requested schema (`text_format`):
  * Extraction      -> units / functions parsed from the `[clause_id] text` lines of the input;
  * VerdictOutput   -> every "before" function of the current group: the candidate prematch offered
                       (transferred / changed) or `lost` when there is none;
  * DupOutput       -> a duplicate only when all candidates share the same area of responsibility;
  * StructureOutput -> no conflicts, no composition changes;
  * anything else   -> the schema filled with a fixed summary string (Reporter).
The test checks the plumbing around the model (prematch, assembly, verifier), not the model itself.
"""
import re
import threading
from types import SimpleNamespace

CLAUSE = re.compile(r"^\[([^\]]+)\] (.*)$")
UNIT = re.compile(r"^(?P<name>[^()]+?) \((?P<abbr>[А-ЯЁ]{2,})\)\.?$")
FUNCTION = re.compile(r"^(?P<owner>[А-ЯЁ]{2,}) (?P<quote>(?P<action>\S+) (?P<object>.+?))\.?$")
AREA_SPLIT = " в области "
DEFAULT_AREA = "внутренний контроль"
BEFORE_LINE = re.compile(r"^(B\d+) \[")
CANDIDATE_LINE = re.compile(r"^\s+кандидат (transferred|changed): (A\d+)")
DUP_HEADER = re.compile(r"^## Кандидат в дубли (\d+)")
DUP_LINE = re.compile(r"^(A\d+) \[.*?\| (?P<action>\S+) → (?P<object>.+?) \| область: (?P<area>.+?) \|")


def _clauses(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in map(CLAUSE.match, text.splitlines()) if m]


def extraction(instructions: str, text: str) -> dict:
    units, functions = [], []
    structure_task = "Заполни только `units`" in instructions
    for clause_id, body in _clauses(text):
        if structure_task:
            m = UNIT.match(body)
            if m:
                units.append({"name": m["name"], "short_name": m["abbr"], "kind": "division", "parent": None,
                              "clause_ids": [clause_id]})
            continue
        m = FUNCTION.match(body)
        if m:
            obj, _, area = m["object"].partition(AREA_SPLIT)
            functions.append({"owner": m["owner"], "modality": "duty", "action": m["action"], "object": obj,
                              "area": area or DEFAULT_AREA, "clause_id": clause_id, "quote": m["quote"]})
    return {"units": units, "roles": [], "functions": functions}


def _current_group(text: str) -> list[str]:
    start = text.rfind("# Текущая группа")
    return text[start:].splitlines() if start >= 0 else []


def verdicts(text: str) -> dict:
    out, current = [], None
    for line in _current_group(text):
        if m := BEFORE_LINE.match(line):
            current = {"before_id": m.group(1), "status": "lost", "after_ids": [], "after_refs": [],
                       "search_note": "", "comment": "", "severity": "medium", "recommendation": None}
            out.append(current)
        elif (m := CANDIDATE_LINE.match(line)) and current:
            current["status"] = m.group(1)
            current["after_ids"].append(m.group(2))
    return {"verdicts": out}


def duplicates(text: str) -> dict:
    clusters: dict[int, list[re.Match]] = {}
    number = None
    for line in _current_group(text):
        if m := DUP_HEADER.match(line):
            number = int(m.group(1))
            clusters[number] = []
        elif number is not None and (m := DUP_LINE.match(line)):
            clusters[number].append(m)
    decisions = []
    for n, rows in clusters.items():
        same_area = len({r["area"] for r in rows}) == 1
        decisions.append({"candidate": n, "is_duplicate": same_area, "after_ids": [r.group(1) for r in rows],
                          "action": rows[0]["action"], "object": rows[0]["object"], "area": rows[0]["area"],
                          "reason": "одна область ответственности" if same_area else "разные области",
                          "severity": "medium", "recommendation": None})
    return {"decisions": decisions}


class FakeLLM:
    def __init__(self):
        self.calls: list[str] = []  # schema names, in call order
        self.lock = threading.Lock()
        self.responses = self

    def parse(self, *, text_format, instructions, input, **_):
        name = text_format.__name__
        with self.lock:
            self.calls.append(name)
        if name == "Extraction":
            data = extraction(instructions, input)
        elif name == "VerdictOutput":
            data = verdicts(input)
        elif name == "DupOutput":
            data = duplicates(input)
        elif name == "StructureOutput":
            data = {"checks": [], "composition_changes": []}
        else:
            data = {field: "Сводка по результатам сравнения." for field in text_format.model_fields}
        return SimpleNamespace(
            output_parsed=text_format.model_validate(data), status="completed", incomplete_details=None,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  input_tokens_details=SimpleNamespace(cached_tokens=0)))
