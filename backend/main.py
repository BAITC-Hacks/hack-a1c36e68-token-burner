"""FastAPI app: upload a before/after set, run the pipeline in a background thread, poll the job.

POST /api/analyze                 multipart before[] / after[] -> {job_id}
GET  /api/jobs/{id}               -> {job_id, status, step, progress, message, trace, result?, error?}
GET  /api/clause/{doc_id}/{cid}   -> {doc_id, doc_name, clause_id, text, page, neighbors}
GET  /api/demo                    -> data/demo_result.json
GET  /                            -> frontend/ (static)

Jobs live in a dict and in data/jobs/<id>/job.json (uploads next to it). Input problems (no files on
a side, empty or unreadable file, unsupported format) end the job with status=error and a readable
message, never with HTTP 500. DEMO_MODE=1 replays the pipeline steps and returns the demo result.
"""
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.agent.schemas import AnalysisResult, Document
from backend.config import settings
from backend.ingest import ingest_file, ingest_set

log = logging.getLogger(__name__)
SIDES = ("before", "after")
SIDE_NAMES = {"before": "«до»", "after": "«после»"}
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}
DEMO_STEPS = [
    ("ingest", "Чтение документов"),
    ("extract", "Извлечение структуры и функций"),
    ("prematch", "Детерминированное сопоставление функций"),
    ("match", "Сопоставление редакций"),
    ("verify", "Проверка цитат"),
    ("report", "Формирование заключения"),
]
DEMO_STEP_SECONDS = 0.6
NEIGHBORS = 2
PUBLIC_FIELDS = ("job_id", "status", "step", "progress", "message", "demo", "created_at", "updated_at",
                 "trace", "error", "result")

JOBS: dict[str, dict] = {}
DOCS: dict[str, dict[str, Document]] = {}  # job_id (or "demo") -> doc_id -> parsed document
CURRENT = {"docs": "demo"}  # whose documents /api/clause resolves when no job_id is given
LOCK = threading.RLock()


class InputError(Exception):
    """Problem with the uploaded files; the message is shown to the user as is."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_pipeline(before: list[Path], after: list[Path], progress) -> AnalysisResult:
    """Public pipeline entry; imported lazily so the API starts even without the agent modules."""
    from backend.agent.pipeline import run

    return run(before, after, progress=progress)


# --- job storage ---


def _job_dir(job_id: str) -> Path:
    return settings.jobs_dir / job_id


def _save(job: dict) -> None:
    path = _job_dir(job["job_id"]) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _update(job_id: str, **fields) -> None:
    with LOCK:
        job = JOBS[job_id]
        job.update(fields, updated_at=_now())
        _save(job)


def _progress(job_id: str, step: str, fraction: float, message: str = "") -> None:
    """Progress event: the current step, overall fraction and a trace of step start/finish times."""
    with LOCK:
        job = JOBS[job_id]
        trace = job["trace"]
        now = _now()
        if step != job["step"]:
            if trace and trace[-1]["finished_at"] is None:
                trace[-1]["finished_at"] = now
            if step != "done":
                trace.append({"step": step, "started_at": now, "finished_at": None, "notes": message})
        _update(job_id, status="running", step=step, progress=round(max(job["progress"], fraction), 3),
                message=message)


def _finish(job_id: str, **fields) -> None:
    with LOCK:
        trace = JOBS[job_id]["trace"]
        if trace and trace[-1]["finished_at"] is None:
            trace[-1]["finished_at"] = _now()
        _update(job_id, **fields)


def load_jobs() -> None:
    """Restore jobs from data/jobs after a restart; unfinished ones cannot resume."""
    for path in sorted(settings.jobs_dir.glob("*/job.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if job.get("status") in ("queued", "running"):
            job.update(status="error", error="Сервер был перезапущен во время анализа. Запустите анализ заново.")
        JOBS[job["job_id"]] = job


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    load_jobs()
    yield


app = FastAPI(title="Token burner: сравнение оргдокументов", lifespan=lifespan)


# --- input checks ---


def _safe_name(name: str | None, index: int) -> str:
    name = Path((name or "").replace("\\", "/")).name.strip()
    name = re.sub(r"[\x00-\x1f]", "", name)
    return name if name and name not in (".", "..") else f"file{index}"


async def _store_uploads(request: Request, job_id: str) -> tuple[dict[str, list[Path]], list[str]]:
    """Save before[]/after[] parts (plain before/after also accepted); return paths and input problems."""
    form = await request.form()
    paths: dict[str, list[Path]] = {side: [] for side in SIDES}
    problems: list[str] = []
    for side in SIDES:
        uploads = [f for key in (f"{side}[]", side) for f in form.getlist(key) if hasattr(f, "filename")]
        for i, upload in enumerate(uploads, 1):
            name = _safe_name(upload.filename, i)
            data = await upload.read()
            suffix = Path(name).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                problems.append(f"«{name}»: формат {suffix or 'без расширения'} не поддерживается, "
                                "загрузите PDF или DOCX")
                continue
            if not data.strip():
                problems.append(f"«{name}»: файл пустой")
                continue
            path = _job_dir(job_id) / side / str(i) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            paths[side].append(path)
        if not uploads:
            problems.append(f"Не загружен комплект {SIDE_NAMES[side]}: нужен хотя бы один файл с каждой стороны")
    return paths, problems


def check_documents(paths: dict[str, list[Path]]) -> dict[str, Document]:
    """Parse every file the same way the pipeline does; unreadable ones raise InputError."""
    docs = ingest_set(paths["before"], "before") + ingest_set(paths["after"], "after")
    problems = []
    for doc in docs:
        if not doc.clauses:
            reason = "; ".join(doc.warnings) or "текст не извлечён"
            problems.append(f"«{doc.name}»: {reason}. Файл повреждён или это скан без текстового слоя "
                            "(OCR не поддерживается)")
    if problems:
        raise InputError("\n".join(problems))
    return {d.doc_id: d for d in docs}


# --- workers ---


def _work(job_id: str, paths: dict[str, list[Path]], demo: bool) -> None:
    try:
        _progress(job_id, "ingest", 0.02, "Чтение документов")
        docs = check_documents(paths)
        with LOCK:
            DOCS[job_id] = docs
        if demo:  # served as is, exactly like GET /api/demo
            _replay_demo(job_id)
            result = json.loads(settings.demo_result_path.read_text(encoding="utf-8"))
        else:
            result = run_pipeline(paths["before"], paths["after"],
                                  progress=lambda step, fraction, message="": _progress(job_id, step, fraction, message))
            result = result.model_dump(mode="json")
        _finish(job_id, status="done", step="done", progress=1.0, message="Готово", result=result)
    except InputError as exc:
        _finish(job_id, status="error", error=str(exc))
    except Exception as exc:  # the job reports the failure; the server keeps running
        log.exception("job %s failed", job_id)
        _finish(job_id, status="error", error=f"Ошибка анализа: {exc}")


def _replay_demo(job_id: str) -> None:
    for i, (step, message) in enumerate(DEMO_STEPS):
        _progress(job_id, step, i / len(DEMO_STEPS), f"{message} (демо-режим)")
        time.sleep(DEMO_STEP_SECONDS)


# --- API ---


@app.post("/api/analyze")
async def analyze(request: Request) -> dict:
    job_id = uuid.uuid4().hex[:12]
    demo = settings.demo_mode
    job = {"job_id": job_id, "status": "queued", "step": "", "progress": 0.0, "message": "", "demo": demo,
           "created_at": _now(), "updated_at": _now(), "trace": [], "error": None, "result": None}
    with LOCK:
        JOBS[job_id] = job
    try:
        paths, problems = await _store_uploads(request, job_id)
    except Exception as exc:  # malformed multipart body
        paths, problems = {side: [] for side in SIDES}, [f"Не удалось принять файлы: {exc}"]
    job["files"] = {side: [str(p) for p in paths[side]] for side in SIDES}
    if problems:
        _update(job_id, status="error", error="\n".join(problems))
        return {"job_id": job_id}
    _save(job)
    threading.Thread(target=_work, args=(job_id, paths, demo), daemon=True, name=f"job-{job_id}").start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, f"Задание {job_id} не найдено")
        if job["status"] == "done":
            CURRENT["docs"] = "demo" if job.get("demo") else job_id
        return {k: job.get(k) for k in PUBLIC_FIELDS}


@app.get("/api/demo")
def demo_result() -> FileResponse:
    if not settings.demo_result_path.exists():
        raise HTTPException(404, "Демо-результат не найден: data/demo_result.json")
    CURRENT["docs"] = "demo"
    return FileResponse(settings.demo_result_path, media_type="application/json")


def demo_documents() -> dict[str, Document]:
    """Documents referenced by the demo result, parsed from data/samples/<name>."""
    with LOCK:
        if "demo" not in DOCS:
            result = json.loads(settings.demo_result_path.read_text(encoding="utf-8"))
            docs = {}
            for info in result.get("documents", []):
                path = settings.data_dir / "samples" / info["name"]
                if path.exists():
                    docs[info["doc_id"]] = ingest_file(path, info["side"], info["doc_id"])
            DOCS["demo"] = docs
        return DOCS["demo"]


def job_documents(job_id: str) -> dict[str, Document]:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, f"Задание {job_id} не найдено")
        if job.get("demo"):
            return demo_documents()
        if job_id not in DOCS:  # e.g. after a restart: parse the stored uploads again
            files = job.get("files") or {}
            DOCS[job_id] = {d.doc_id: d for side in SIDES for d in ingest_set(files.get(side, []), side)}
        return DOCS[job_id]


@app.get("/api/clause/{doc_id}/{clause_id}")
def get_clause(doc_id: str, clause_id: str, job_id: str | None = None) -> dict:
    """Clause text and page plus neighbouring clauses; ?job_id= picks the job, else the last shown result."""
    source = job_id or CURRENT["docs"]
    docs = demo_documents() if source == "demo" else job_documents(source)
    doc = docs.get(doc_id)
    if doc is None:
        raise HTTPException(404, f"Документ {doc_id} не найден")
    clauses = [c for c in doc.clauses if c.clause_id != "toc"]
    index = next((i for i, c in enumerate(clauses) if c.clause_id == clause_id), None)
    if index is None:
        raise HTTPException(404, f"Пункт {clause_id} не найден в документе «{doc.name}»")
    clause = clauses[index]
    around = clauses[max(0, index - NEIGHBORS):index] + clauses[index + 1:index + 1 + NEIGHBORS]
    return {"doc_id": doc_id, "doc_name": doc.name, "clause_id": clause_id, "text": clause.text, "page": clause.page,
            "neighbors": [{"clause_id": c.clause_id, "text": c.text, "page": c.page} for c in around]}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "demo_mode": settings.demo_mode}


if settings.frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
