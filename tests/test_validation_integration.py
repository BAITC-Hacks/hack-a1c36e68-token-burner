"""Opt-in: full pipeline with the real LLM on red8 -> red9, checked against the Validation set in CLAUDE.md.

pytest -m integration tests/test_validation_integration.py
The result is saved to data/jobs/integration_result.json (and its table printed with -s).
"""
import json

import pytest

from backend.agent.pipeline import run
from backend.config import ROOT, settings
from tests.validation_set import check, table


@pytest.mark.integration
def test_validation_set_on_samples():
    result = run([ROOT / "data/samples/red8.pdf"], [ROOT / "data/samples/red9.pdf"])
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    (settings.jobs_dir / "integration_result.json").write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + table(result))
    assert result.analysis_complete
    failed = [f"{r.item}: {r.verdict} ({r.detail})" for r in check(result) if r.required and r.verdict != "найдено"]
    assert not failed, "\n".join(failed)
