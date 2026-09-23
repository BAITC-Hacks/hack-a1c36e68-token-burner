"""Pydantic models = the backend/frontend contract.

Run `python -m backend.agent.schemas` to regenerate contracts/result.schema.json.
"""
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Side = Literal["before", "after"]
UnitStatus = Literal["created", "preserved", "transformed", "eliminated"]
FunctionStatus = Literal["preserved", "transferred", "changed", "lost", "duplicated"]
FindingType = Literal["loss", "change", "transfer", "duplication", "conflict", "structure"]
Severity = Literal["high", "medium", "low"]


# --- parsed input (internal, but DocumentInfo is part of the result) ---


class Clause(BaseModel):
    clause_id: str = Field(description="'5.3.3', sub-item '5.3.3.б', section heading '10', text before the first clause 'preamble'")
    text: str = Field(description="Exact clause text without its number; the verifier matches quotes against it")
    page: int | None = Field(description="1-based page where the clause starts; null for DOCX")
    parent_id: str | None = None


class Document(BaseModel):
    doc_id: str
    name: str
    side: Side
    pages: int | None
    clauses: list[Clause]
    warnings: list[str] = []

    def clause(self, clause_id: str) -> Clause | None:
        return next((c for c in self.clauses if c.clause_id == clause_id), None)

    def text_with_children(self, clause_id: str) -> str:
        """Clause text followed by all nested clauses/sub-items, e.g. 5.3.3 + 5.3.3.а + 5.3.3.б."""
        prefix = clause_id + "."
        parts = [c.text for c in self.clauses if c.clause_id == clause_id or c.clause_id.startswith(prefix)]
        return " ".join(parts)

    def info(self) -> "DocumentInfo":
        return DocumentInfo(
            doc_id=self.doc_id,
            name=self.name,
            side=self.side,
            pages=self.pages,
            clause_count=len(self.clauses),
            parse_ok=not self.warnings,
            warnings=self.warnings,
        )


# --- Extractor output (internal, not part of the exported contract) ---
# No defaults here: OpenAI strict structured outputs require every field to be present.


class ExtractedUnit(BaseModel):
    name: str = Field(description="Полное наименование подразделения, как в документе")
    short_name: str | None = Field(description="Аббревиатура, если есть (ДНМ)")
    kind: Literal["block", "department", "division", "other"]
    parent: str | None = Field(description="Подразделение, в которое входит")
    clause_ids: list[str] = Field(description="Пункты, где подразделение определено или упомянуто в структуре")


class ExtractedRole(BaseModel):
    name: str = Field(description="Должность, как в документе")
    unit: str | None = Field(description="Подразделение, к которому относится должность")
    reports_to: list[str] = Field(description="Кому подчиняется (должности)")
    clause_ids: list[str]


class ExtractedFunction(BaseModel):
    owner: str = Field(description="Подразделение или должность — исполнитель, как в документе")
    modality: Literal["duty", "right", "responsibility"] = Field(description="обязанность/функция, право, ответственность")
    action: str = Field(description="Действие, глагол в начальной форме: 'анализировать'")
    object: str = Field(description="Объект действия: 'результаты непрерывного аудита'")
    area: str = Field(description="Область ответственности: 'непрерывный аудит', 'ИТ-аудит'")
    clause_id: str = Field(description="Идентификатор пункта-источника из входного списка, например 5.3.3.б")
    quote: str = Field(description="Дословный непрерывный фрагмент текста пункта, подтверждающий функцию")


class Extraction(BaseModel):
    units: list[ExtractedUnit]
    roles: list[ExtractedRole]
    functions: list[ExtractedFunction]


# --- result contract ---


class DocumentInfo(BaseModel):
    doc_id: str
    name: str
    side: Side
    pages: int | None
    clause_count: int
    parse_ok: bool = True
    warnings: list[str] = []


class Evidence(BaseModel):
    doc_id: str
    clause_id: str
    quote: str = Field(description="Verbatim fragment of the clause text")
    page: int | None = None


class Unit(BaseModel):
    name: str
    status: UnitStatus
    before_ref: str | None = Field(None, description="Name of the unit/position in the 'before' set")
    after_ref: str | None = Field(None, description="Name of the unit/position in the 'after' set")
    comment: str | None = None
    evidence: list[Evidence] = []


class Function(BaseModel):
    id: str
    action: str
    object: str
    area: str
    owner_before: str | None = None
    owner_after: str | None = None
    status: FunctionStatus
    conflict_of_interest: bool = False
    comment: str | None = None
    evidence: list[Evidence] = []
    recommendation: str | None = None


class Finding(BaseModel):
    id: str
    type: FindingType
    severity: Severity
    summary: str
    evidence: list[Evidence] = Field(min_length=1)
    function_ids: list[str] = []
    recommendation: str | None = None
    verified: bool = False
    rejection_reason: str | None = None


class Stats(BaseModel):
    findings_total: int = 0
    verified: int = 0
    rejected: int = 0
    duration_s: float | None = None
    cost_usd: float | None = Field(None, description="Sum of LLM costs; null if a model price is unknown")


class TraceStep(BaseModel):
    step: str
    started_at: datetime
    finished_at: datetime | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    notes: str = ""


class AnalysisResult(BaseModel):
    documents: list[DocumentInfo] = []
    units: list[Unit] = []
    functions: list[Function] = []
    findings: list[Finding] = []
    conclusion_md: str = ""
    stats: Stats = Stats()
    analysis_complete: bool = True
    warnings: list[str] = []
    trace: list[TraceStep] = []


def export_schema(path=None) -> dict:
    from backend.config import ROOT

    schema = AnalysisResult.model_json_schema()
    path = path or ROOT / "contracts" / "result.schema.json"
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return schema


if __name__ == "__main__":
    export_schema()
    print("contracts/result.schema.json written")
