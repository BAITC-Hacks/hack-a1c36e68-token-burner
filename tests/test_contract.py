"""The mock and the exported JSON schema must stay in sync with backend/agent/schemas.py."""
import json
from pathlib import Path

import pytest
from backend.agent.schemas import AnalysisResult
from backend.agent.verify import check_evidence, verify_result
from backend.config import ROOT, settings
from backend.ingest import ingest_file


@pytest.fixture(scope="module")
def demo():
    return AnalysisResult.model_validate_json(settings.demo_result_path.read_text(encoding="utf-8"))


def test_schema_file_is_up_to_date():
    exported = json.loads((ROOT / "contracts" / "result.schema.json").read_text(encoding="utf-8"))
    assert exported == AnalysisResult.model_json_schema(), "run: python -m backend.agent.schemas"


def test_demo_stats_match_findings(demo):
    assert 8 <= len(demo.findings) <= 10
    assert demo.stats.findings_total == len(demo.findings)
    assert demo.stats.verified == sum(f.verified for f in demo.findings)
    assert demo.stats.rejected == 1
    rejected = next(f for f in demo.findings if not f.verified)
    assert rejected.rejection_reason
    assert f"**{rejected.id}**" not in demo.conclusion_md
    for f in demo.findings:
        if f.verified:
            assert f"**{f.id}**" in demo.conclusion_md


def test_demo_evidence_matches_sample_text(demo):
    docs = {d.doc_id: ingest_file(ROOT / "data" / "samples" / d.name, d.side, d.doc_id) for d in demo.documents}
    evidence = [(f.verified, e) for f in demo.findings for e in f.evidence]
    evidence += [(True, e) for x in demo.units + demo.functions for e in x.evidence]
    for verified, e in evidence:
        assert docs[e.doc_id].clause(e.clause_id).page == e.page
        assert (check_evidence(e, docs) is None) == verified, (e.clause_id, check_evidence(e, docs))
    rerun = verify_result(demo, docs)
    assert [f.verified for f in rerun.findings] == [f.verified for f in demo.findings]


def test_loss_wording(demo):
    text = demo.conclusion_md + " ".join(f.summary for f in demo.findings if f.verified)
    assert "утрачен" not in text.lower() and "уничтож" not in text.lower()
    assert 'не найден' in text
