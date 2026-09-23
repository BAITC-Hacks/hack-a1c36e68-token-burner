"""API: jobs, progress, input errors, demo mode, clause lookup. The pipeline is mocked."""
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.agent.schemas import AnalysisResult
from backend.config import ROOT, settings

ORG = Path(__file__).parent / "fixtures" / "org"
SAMPLES = ROOT / "data" / "samples"
STEPS = ["ingest", "extract", "prematch", "match", "verify", "report"]


def fake_pipeline(before, after, progress):
    for i, step in enumerate(STEPS[1:], 1):
        progress(step, i / len(STEPS), f"шаг {step}")
    progress("done", 1.0, "Готово")
    return AnalysisResult(conclusion_md=f"до: {[Path(p).name for p in before]}, после: {[Path(p).name for p in after]}")


def failing_pipeline(before, after, progress):
    raise RuntimeError("модель недоступна")


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "settings", replace(settings, jobs_dir=tmp_path / "jobs", demo_mode=False))
    monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "DEMO_STEP_SECONDS", 0)
    monkeypatch.setattr(main, "JOBS", {})
    monkeypatch.setattr(main, "DOCS", {})
    monkeypatch.setattr(main, "CURRENT", {"docs": "demo"})
    with TestClient(main.app) as client:
        yield client


def upload(client, before=(), after=()):
    files = [("before[]", f) for f in before] + [("after[]", f) for f in after]
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 200
    return response.json()["job_id"]


def txt(name="base.txt"):
    return (name, (ORG / name).read_bytes(), "text/plain")


def wait(client, job_id, timeout=15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def test_analyze_runs_pipeline_with_progress_and_persists(api):
    job_id = upload(api, before=[txt()], after=[txt("deleted.txt"), txt("duplicated.txt")])
    job = wait(api, job_id)
    assert job["status"] == "done", job["error"]
    assert job["progress"] == 1.0 and job["step"] == "done"
    assert [s["step"] for s in job["trace"]] == STEPS
    assert all(s["finished_at"] for s in job["trace"])
    result = AnalysisResult.model_validate(job["result"])
    assert result.conclusion_md == "до: ['base.txt'], после: ['deleted.txt', 'duplicated.txt']"
    saved = json.loads((main.settings.jobs_dir / job_id / "job.json").read_text(encoding="utf-8"))
    assert saved["status"] == "done" and saved["result"] == job["result"]


@pytest.mark.parametrize("before, after, expected", [
    ([txt()], [], "Не загружен комплект «после»"),
    ([], [txt()], "Не загружен комплект «до»"),
    ([txt()], [("empty.pdf", b"", "application/pdf")], "«empty.pdf»: файл пустой"),
    ([txt()], [("old.doc", b"binary", "application/msword")], "«old.doc»: формат .doc не поддерживается"),
])
def test_bad_upload_is_a_job_error_not_500(api, before, after, expected):
    job = wait(api, upload(api, before, after))
    assert job["status"] == "error"
    assert expected in job["error"]


def test_broken_pdf_is_a_job_error(api):
    broken = ("broken.pdf", b"%PDF-1.4\n this is not really a pdf", "application/pdf")
    job = wait(api, upload(api, [txt()], [broken]))
    assert job["status"] == "error"
    assert "«broken.pdf»" in job["error"] and "повреждён" in job["error"]


def test_pipeline_exception_is_a_job_error(api, monkeypatch):
    monkeypatch.setattr(main, "run_pipeline", failing_pipeline)
    job = wait(api, upload(api, [txt()], [txt()]))
    assert job["status"] == "error"
    assert "модель недоступна" in job["error"]
    assert job["trace"][-1]["finished_at"]


def test_demo_mode_replays_steps_without_llm(api, monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, demo_mode=True))
    monkeypatch.setattr(main, "run_pipeline", failing_pipeline)
    files = [(p.name, p.read_bytes(), "application/pdf") for p in (SAMPLES / "red8.pdf", SAMPLES / "red9.pdf")]
    job = wait(api, upload(api, [files[0]], [files[1]]))
    assert job["status"] == "done", job["error"]
    assert job["demo"] is True
    assert [s["step"] for s in job["trace"]] == STEPS
    assert job["result"] == json.loads(settings.demo_result_path.read_text(encoding="utf-8"))


def test_clause_with_neighbors_for_finished_job(api):
    job_id = upload(api, [txt()], [txt("deleted.txt")])
    wait(api, job_id)  # polling a finished job makes its documents the current ones
    clause = api.get("/api/clause/before-1/2.2").json()
    assert clause["text"] == "ОУ ведет реестр договоров."
    assert clause["page"] is None and clause["doc_name"] == "base.txt"
    assert [n["clause_id"] for n in clause["neighbors"]] == ["2", "2.1", "2.3", "2.4"]
    assert api.get(f"/api/clause/after-1/2.3?job_id={job_id}").json()["text"].startswith("ОК контролирует")
    assert api.get("/api/clause/after-1/2.2").status_code == 404
    assert api.get("/api/clause/after-9/2.2").status_code == 404


def test_demo_result_and_its_clauses(api):
    demo = api.get("/api/demo")
    assert demo.status_code == 200
    evidence = next(e for f in demo.json()["findings"] for e in f["evidence"])
    clause = api.get(f"/api/clause/{evidence['doc_id']}/{evidence['clause_id']}").json()
    assert clause["page"] == evidence["page"]
    assert evidence["quote"][:30] in clause["text"]


def test_unknown_job_is_404(api):
    assert api.get("/api/jobs/nope").status_code == 404


def test_frontend_served_at_root(api):
    response = api.get("/")
    assert response.status_code == 200 and "<html" in response.text.lower()


def test_restart_marks_unfinished_jobs_as_error(api):
    job_dir = main.settings.jobs_dir / "stale1"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({"job_id": "stale1", "status": "running", "step": "match",
                                                  "progress": 0.5, "trace": []}), encoding="utf-8")
    main.load_jobs()
    job = api.get("/api/jobs/stale1").json()
    assert job["status"] == "error" and "перезапущен" in job["error"]
