# AGENTS.md

Guidance for Codex working in this repository. You own the frontend; a separate Claude Code
session owns the backend. Read `CLAUDE.md` for the product, domain rules and the result
contract — do not duplicate it here.

## Ownership

- You edit only `frontend/` (and, when explicitly asked later, `README.md`, `Dockerfile`,
  `docker-compose.yml`).
- Never edit `backend/`, `tests/`, `contracts/`, `data/`. If the contract blocks you, say
  exactly what you need and stop.
- Source of truth: `contracts/result.schema.json` and the mock `data/demo_result.json`
  (served at `GET /api/demo`). Build against the mock first; the live API comes later.

## Frontend rules

- One file: `frontend/index.html`. Vue 3 from `frontend/vendor/vue.global.prod.js`
  (vendored, no CDN — the demo may run offline). No npm, no build step, no frameworks.
- Plain CSS inside the file. Clean, dense, "audit tool" look: neutral palette, one accent
  color, monospace for clause ids. Russian UI text.
- Must work at 1280px on a projector; test with `python -m http.server` from `frontend/`
  against a copy of the mock if the backend is down.

## API (from backend)

```
POST /api/analyze        multipart: before[] , after[]   -> {job_id}
GET  /api/jobs/{id}      -> {status: queued|running|done|error, step, progress, result?, error?}
GET  /api/clause/{doc_id}/{clause_id} -> {text, page, neighbors}
GET  /api/demo           -> AnalysisResult (mock / cached)
```
Poll `/api/jobs/{id}` every 1.5 s. `?demo=1` in the page URL loads `/api/demo` directly.

## Screens

1. Upload: two drop zones («До реорганизации», «После»), multiple files each, PDF/DOCX,
   button «Анализировать», link «Открыть демо-результат».
2. Progress: vertical list of pipeline steps from `trace`/`step` (Ingest → Extractor →
   Matcher → Verifier → Reporter) with elapsed time; the current step is highlighted.
3. Results, header with counters: «N выводов · V подтверждено · R отклонено верификатором»
   and `analysis_complete` warning if false. Tabs:
   - «Структура»: units grouped by status (created / preserved / transformed / eliminated),
     each with evidence chips.
   - «Функции»: table: function · было (owner_before) · стало (owner_after) · статус ·
     конфликт. Filters by status and unit. Each row expands into a lineage block:
     two columns «Ред. до» / «Ред. после», each with clause id + quote; missing side shows
     «не найдена в комплекте "после"».
   - «Заключение»: rendered `conclusion_md` (use a tiny inline markdown renderer or
     pre-rendered HTML from the backend if provided) + button «Скачать» (opens print view).
   - «Отклонено верификатором»: findings with `verified=false` and `rejection_reason`.
4. Evidence chip click: right-side drawer that calls `/api/clause/{doc_id}/{clause_id}` and
   shows the clause with the quoted fragment highlighted, plus neighbors for context.

## Working agreements

- Commit after each screen works against the mock.
- Keep it under ~800 lines. No animations beyond a spinner.
- Final message: what works, what is stubbed, screenshots path if you made any.

## Later task (only when asked, ~3:30): README + Docker

README structure: what it does (3 sentences) · screenshot · quick start (`docker compose up`
and bare `uvicorn`) · architecture (mermaid, from `CLAUDE.md`) · how the verifier works ·
demo scenario on `data/samples` · limitations · roadmap. Dockerfile: python:3.12-slim,
copy backend + frontend + data, `uvicorn backend.main:app --host 0.0.0.0 --port 8000`.