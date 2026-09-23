"""Pre-delivery tests 1–6: the whole pipeline on small synthetic fixtures with a fake LLM.

Fixtures: tests/fixtures/org/*.txt (base.txt is the "before" set of most cases). The fake model
(tests/fixtures/fake_llm.py) only confirms what the deterministic layers propose, so these tests
pin down ingest -> prematch -> assembly -> verifier, not the quality of the prompts.
"""
from dataclasses import replace
from pathlib import Path

import pytest

from backend.config import settings
from tests.fixtures.fake_llm import FakeLLM

pipeline = pytest.importorskip("backend.agent.pipeline", reason="backend/agent/pipeline.py is not in this branch yet")

ORG = Path(__file__).parent / "fixtures" / "org"
CHANGE_TYPES = {"loss", "change", "transfer", "duplication"}


@pytest.fixture(autouse=True)
def tmp_jobs(tmp_path, monkeypatch):
    """Keep debug dumps (match_raw.json, extract cache) out of data/jobs."""
    import importlib

    local = replace(settings, jobs_dir=tmp_path / "jobs")
    for name in ("backend.agent.pipeline", "backend.agent.match", "backend.agent.extract"):
        try:
            monkeypatch.setattr(importlib.import_module(name), "settings", local, raising=False)
        except ImportError:
            pass


def run(before: str | Path, after: str | Path):
    fake = FakeLLM()
    before = before if isinstance(before, Path) else ORG / before
    after = after if isinstance(after, Path) else ORG / after
    result = pipeline.run([before], [after], client=fake, use_cache=False)
    return result, fake


def functions_by_clause(result, status: str) -> set[str]:
    return {e.clause_id for f in result.functions if f.status == status for e in f.evidence
            if e.doc_id.startswith("before")}


def verified(result, ftype: str):
    return [f for f in result.findings if f.type == ftype and f.verified]


def docx_copy(src: Path, dst: Path) -> Path:
    import docx

    document = docx.Document()
    for line in src.read_text(encoding="utf-8").splitlines():
        document.add_paragraph(line)
    document.save(dst)
    return dst


@pytest.mark.parametrize("fmt", ["txt", "docx"])
def test_1_same_document_no_changes(fmt, tmp_path):
    path = ORG / "base.txt" if fmt == "txt" else docx_copy(ORG / "base.txt", tmp_path / "base.docx")
    result, _ = run(path, path)
    assert result.analysis_complete, result.warnings
    assert result.units and all(u.status == "preserved" for u in result.units)
    assert {f.status for f in result.functions} == {"preserved"}
    assert not [f for f in result.findings if f.type in CHANGE_TYPES]


def test_2_renumbered_clauses_no_findings():
    result, _ = run("base.txt", "renumbered.txt")
    assert result.analysis_complete, result.warnings
    assert all(u.status == "preserved" for u in result.units)
    assert {f.status for f in result.functions} == {"preserved"}
    assert result.findings == []


def test_3_moved_function_is_transferred_not_lost():
    result, _ = run("base.txt", "transferred.txt")
    assert functions_by_clause(result, "transferred") == {"2.4"}
    moved = next(f for f in result.functions if f.status == "transferred")
    assert (moved.owner_before, moved.owner_after) == ("ОК", "ОУ")
    assert not [f for f in result.findings if f.type == "loss"]
    assert [f for f in verified(result, "transfer") if "2.4" in {e.clause_id for e in f.evidence}]


def test_4_deleted_clause_is_lost_with_evidence():
    result, _ = run("base.txt", "deleted.txt")
    assert functions_by_clause(result, "lost") == {"2.2"}
    [loss] = verified(result, "loss")
    assert [(e.doc_id, e.clause_id) for e in loss.evidence] == [("before-1", "2.2")]
    assert "ведет реестр договоров" in loss.evidence[0].quote
    assert "не найдена в загруженном комплекте" in loss.summary.lower()
    assert "утрач" not in result.conclusion_md.lower()


def test_5_copied_clause_is_duplicated():
    result, fake = run("base.txt", "duplicated.txt")
    assert "DupOutput" in fake.calls
    [dup] = verified(result, "duplication")
    assert {e.clause_id for e in dup.evidence} == {"2.3", "2.5"}
    assert all(e.doc_id == "after-1" for e in dup.evidence)
    assert any(f.status == "duplicated" for f in result.functions)
    assert not [f for f in result.findings if f.type == "loss"]


def test_6_same_wording_different_area_is_not_duplication():
    result, fake = run("areas.txt", "areas.txt")
    assert "DupOutput" in fake.calls  # the candidate was considered and rejected, not missed
    assert not [f for f in result.findings if f.type == "duplication"]
    assert not any(f.status == "duplicated" for f in result.functions)
