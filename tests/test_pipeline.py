"""End-to-end pipeline on synthetic TXT files with a fake LLM: progress, trace, stats, verification."""
import pytest

from backend.agent import pipeline
from backend.agent.match import Verdict
from backend.config import settings
from tests.conftest import BASE, FakeClient, make_doc


def write(path, clauses):
    path.write_text("\n".join(f"{c}. {t}" for c, t in clauses) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def tmp_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "jobs_dir", tmp_path / "jobs", raising=False)


def test_pipeline_end_to_end(tmp_path):
    before = write(tmp_path / "before.txt", BASE)
    after = write(tmp_path / "after.txt", [c for c in BASE if c[0] != "2.2"])
    docs = [make_doc("before-1", "before", BASE), make_doc("after-1", "after", BASE)]
    docs[0].name, docs[1].name = "before.txt", "after.txt"
    lost = lambda ids, text: [Verdict(before_id=i, status="lost", after_ids=[], after_refs=[], search_note="искал",
                                      comment="", severity="high", recommendation=None) for i in ids]
    client = FakeClient(docs, verdicts=lost)
    events = []
    result = pipeline.run([before], [after], progress=lambda s, f, m: events.append(s), client=client, use_cache=False)
    assert [e for e in events if e != "done"] == ["ingest", "extract", "prematch", "match", "verify", "report"]
    assert result.analysis_complete
    loss = [f for f in result.findings if f.type == "loss"]
    assert len(loss) == 1 and loss[0].verified and loss[0].evidence[0].clause_id == "2.2"
    assert "**" + loss[0].id + "**" in result.conclusion_md
    assert result.stats.findings_total == len(result.findings) and result.stats.rejected == 0
    steps = [s.step for s in result.trace]
    assert steps[:3] == ["ingest", "extract:before-1", "extract:after-1"] and "prematch" in steps and "match" in steps
    assert result.stats.duration_s is not None


def test_unreadable_file_fails_the_job_and_names_the_file(tmp_path):
    before = write(tmp_path / "before.txt", BASE)
    broken = tmp_path / "after.pdf"
    broken.write_bytes(b"not a pdf")
    with pytest.raises(pipeline.UnreadableInput, match="«after.pdf»"):
        pipeline.run([before], [broken], client=FakeClient(), use_cache=False)


def test_stats_count_corrected_and_dropped_citations(tmp_path, monkeypatch):
    from backend.agent import extract

    real_clean = extract.clean

    def noisy_clean(doc, extraction):
        fns = extraction.functions
        if fns:  # one citation points at the wrong clause, one quote is invented
            fns = [fns[0].model_copy(update={"clause_id": "1.1"})] + fns[1:]
            fns.append(fns[-1].model_copy(update={"quote": "утверждает бюджет департамента закупок"}))
        return real_clean(doc, extraction.model_copy(update={"functions": fns}))

    monkeypatch.setattr(extract, "clean", noisy_clean)
    before = write(tmp_path / "before.txt", BASE)
    after = write(tmp_path / "after.txt", BASE)
    docs = [make_doc("before-1", "before", BASE), make_doc("after-1", "after", BASE)]
    docs[0].name, docs[1].name = "before.txt", "after.txt"
    result = pipeline.run([before], [after], client=FakeClient(docs), use_cache=False)
    assert result.stats.citations_corrected >= 2 and result.stats.functions_dropped >= 2
