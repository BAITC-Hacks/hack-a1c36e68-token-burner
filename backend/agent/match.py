"""Matcher layer 2 (LLM) and assembly of units / functions / findings.

Input is only the residual of prematch: before functions without a pair (lost/changed candidates),
transfer and change candidates to confirm. Calls are grouped by owner (unit or role) and run in
parallel; every call also gets the whole "after" set (text + functions by owner) so the model can
search for moved/rephrased functions, duplicates and conflicts. The shared part goes first in the
input so identical prefixes can hit the prompt cache.

Everything the model returns is referenced by function ids or by (clause, verbatim quote);
findings are built here from those references and are verified afterwards by verify.py.
"""
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from backend.agent.llm import LLMCall, LLMError, call_structured, cost_usd
from backend.agent.prematch import FnRef, OwnerNormalizer, Prematch, same_owner
from backend.agent.schemas import (
    Document, Evidence, Extraction, Finding, Function, Severity, TraceStep, Unit,
)
from backend.agent.verify import check_evidence, normalize, with_page
from backend.config import ROOT, settings

PROMPT = (ROOT / "backend" / "prompts" / "match.md").read_text(encoding="utf-8")
MIN_GROUP_FUNCTIONS = 20  # owners with fewer functions go to the shared "прочие" group
MAX_VERDICTS_PER_CALL = 50
WARMUP_DELAY_S = 10  # the first call fills the prompt cache for the shared prefix before the others start
MAX_PARALLEL_CALLS = 8
OTHERS = "прочие исполнители"
STRUCTURE = "структура и конфликты"
NOT_FOUND = 'Не найдена в загруженном комплекте "после"'
CANDIDATE = "Кандидат на проверку сотрудником"
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


# --- LLM output (strict structured output: no defaults) ---


class Ref(BaseModel):
    doc_id: str
    clause_id: str
    quote: str


class Verdict(BaseModel):
    before_id: str
    status: Literal["preserved", "transferred", "changed", "lost"]
    after_ids: list[str]
    after_refs: list[Ref]
    search_note: str
    comment: str
    severity: Severity
    recommendation: str | None


class Duplicate(BaseModel):
    after_ids: list[str]
    action: str
    object: str
    area: str
    comment: str
    severity: Severity
    recommendation: str | None


class Conflict(BaseModel):
    side: Literal["before", "after"]
    subject: str
    performs_ids: list[str]
    controls_ids: list[str]
    performer_roles: list[str] = []
    shared_staff: list[str] = []
    refs: list[Ref]
    resolved_in_after: bool | None
    comment: str
    severity: Severity


class CompositionChange(BaseModel):
    unit: str
    removed_roles: list[str]
    added_roles: list[str]
    refs: list[Ref]
    comment: str


class VerdictOutput(BaseModel):
    verdicts: list[Verdict]


class DupDecision(BaseModel):
    candidate: int
    is_duplicate: bool
    after_ids: list[str]
    action: str
    object: str
    area: str
    reason: str
    severity: Severity
    recommendation: str | None


class DupOutput(BaseModel):
    decisions: list[DupDecision]


class ConflictCheck(BaseModel):
    side: Literal["before", "after"]
    subject: str
    controls: str  # what the subject controls / supervises
    control_ids: list[str]
    performs_ids: list[str]           # step 1: own functions performing the controlled activity
    performer_roles: list[str]        # step 2: staff positions performing it
    shared_staff: list[str]           # step 3: staff subordinated both to it and to the controlled manager
    refs: list[Ref]
    is_conflict: bool
    resolved_in_after: bool | None
    comment: str
    severity: Severity


class StructureOutput(BaseModel):
    checks: list[ConflictCheck]
    composition_changes: list[CompositionChange]


class Group(BaseModel):
    name: str
    keys: list[str]
    verdict_ids: list[str]
    full_tasks: bool  # conflicts + composition for this owner
    duplicates: list[list[str]] = []  # only in the dedicated duplicates call


class MatchResult(BaseModel):
    units: list[Unit]
    functions: list[Function]
    findings: list[Finding]
    warnings: list[str]
    complete: bool


# --- prompt input ---


def _cid(doc_id: str, clause_id: str) -> str:
    return f"[{doc_id}:{clause_id}]"


def _fn_line(f: FnRef) -> str:
    return f"{f.id} {_cid(f.doc_id, f.clause_id)} {f.action} → {f.object} | область: {f.area}"


def _structure_ids(doc: Document, ex: Extraction | None) -> set[str]:
    """Clauses about units and positions: those cited by the extracted structure, their parents and list items."""
    if not ex:
        return set()
    ids = {c for x in ex.units + ex.roles for c in x.clause_ids}
    ids |= {doc.clause(c).parent_id for c in list(ids) if doc.clause(c) and doc.clause(c).parent_id}
    ids |= {c.clause_id for c in doc.clauses if c.parent_id in ids}
    return ids


def _structure_clauses(docs: list[Document], extractions: dict[str, Extraction]) -> list[str]:
    lines = []
    for doc in docs:
        ex = extractions.get(doc.doc_id)
        if not ex:
            continue
        ids = _structure_ids(doc, ex)
        side = "до" if doc.side == "before" else "после"
        lines += [f"{_cid(doc.doc_id, c.clause_id)} ({side}) {c.text}" for c in doc.clauses if c.clause_id in ids]
    return lines


def render_shared(docs: list[Document], extractions: dict[str, Extraction], pm: Prematch) -> str:
    parts = ["# Комплект «после»: полный текст"]
    for doc in docs:
        if doc.side != "after":
            continue
        parts.append(f'<document id="{doc.doc_id}" name="{doc.name}">')
        parts += [f"[{c.clause_id}] {c.text}" for c in doc.clauses if c.clause_id not in ("preamble", "toc")]
        parts.append("</document>")
    parts.append("\n# Функции комплекта «после» по исполнителям")
    by_owner: dict[str, list[FnRef]] = defaultdict(list)
    for f in pm.after:
        by_owner[f.owner_key].append(f)
    for owner in sorted(by_owner, key=lambda k: -len(by_owner[k])):
        parts.append(f"## {owner}")
        parts += [_fn_line(f) for f in by_owner[owner]]
    parts.append("\n# Оргструктура обеих редакций (пункты о подразделениях и должностях)")
    parts += _structure_clauses(docs, extractions)
    return "\n".join(parts)


def render_group(group: Group, pm: Prematch, docs: dict[str, Document]) -> str:
    candidates = {p.before: ("transferred", p.after) for p in pm.transfer_candidates}
    candidates |= {p.before: ("changed", p.after) for p in pm.changed_candidates}
    title = group.name if group.name != OTHERS else f"{OTHERS}: " + ", ".join(group.keys)
    parts = [f"\n# Текущая группа: {title}", f"## Функции «до» для классификации ({len(group.verdict_ids)})"]
    for fid in group.verdict_ids:
        f = pm.fn(fid)
        clause = docs[f.doc_id].clause(f.clause_id)
        parts.append(f"{f.id} {_cid(f.doc_id, f.clause_id)} исполнитель: «{f.owner}» ({f.owner_key}) | "
                     f"{f.action} → {f.object} | область: {f.area} | цитата: «{f.quote}»")
        if clause and len(clause.text) > len(f.quote) + 20:
            parts.append(f"    текст пункта: «{clause.text[:500]}»")
        if fid in candidates:
            kind, after_ids = candidates[fid]
            for aid in after_ids:
                a = pm.fn(aid)
                a_clause = docs[a.doc_id].clause(a.clause_id)
                parts.append(f"    кандидат {kind}: {a.id} {_cid(a.doc_id, a.clause_id)} ({a.owner_key}) "
                             f"{a.action} → {a.object} | текст пункта «после»: «{a_clause.text[:500] if a_clause else a.quote}»")
    for n, cluster in enumerate(group.duplicates, 1):
        parts.append(f"## Кандидат в дубли {n}")
        for aid in cluster:
            a = pm.fn(aid)
            a_clause = docs[a.doc_id].clause(a.clause_id)
            parts.append(f"{a.id} {_cid(a.doc_id, a.clause_id)} исполнитель: {a.owner_key} | {a.action} → {a.object} | "
                         f"область: {a.area} | текст пункта: «{a_clause.text[:400] if a_clause else a.quote}»")
    if group.full_tasks:
        parts.append("## Функции комплекта «до» по исполнителям (для задач 3–4)")
        by_owner: dict[str, list[FnRef]] = defaultdict(list)
        for f in pm.before:
            by_owner[f.owner_key].append(f)
        for owner in sorted(by_owner, key=lambda k: -len(by_owner[k])):
            parts.append(f"### {owner}")
            parts += [_fn_line(f) for f in by_owner[owner]]
    parts.append("## Что сделать")
    if group.verdict_ids:
        parts.append(f"Только задача 1: вердикт ровно для {len(group.verdict_ids)} функций: "
                     + ", ".join(group.verdict_ids) + ". Ответ — `verdicts`.")
    elif group.duplicates:
        parts.append(f"Только задача 2: решение по каждому из {len(group.duplicates)} кандидатов в дубли "
                     "(номер кандидата в `candidate`). Ответ — `decisions`.")
    else:
        parts.append("Только задачи 3 и 4 для всех подразделений и должностей обеих редакций. "
                     "Ответ — `checks` (по каждому контролирующему субъекту в каждой редакции) и "
                     "`composition_changes`.")
    return "\n".join(parts)


def build_groups(pm: Prematch) -> list[Group]:
    residual = [p.before for p in pm.transfer_candidates + pm.changed_candidates] + pm.unmatched
    counts = Counter(f.owner_key for f in pm.before + pm.after)
    major = {k for k, n in counts.items() if n >= MIN_GROUP_FUNCTIONS}
    by_group: dict[str, list[str]] = defaultdict(list)
    for fid in residual:
        key = pm.fn(fid).owner_key
        by_group[key if key in major else OTHERS].append(fid)
    order = [k for k, _ in counts.most_common() if k in by_group] + ([OTHERS] if OTHERS in by_group else [])
    groups = []
    for name in order:
        ids = sorted(by_group.get(name, []), key=lambda i: int(i[1:]))
        keys = [name] if name != OTHERS else sorted({pm.fn(i).owner_key for i in ids})
        batches = [ids[i:i + MAX_VERDICTS_PER_CALL] for i in range(0, len(ids), MAX_VERDICTS_PER_CALL)]
        for n, batch in enumerate(batches):
            groups.append(Group(name=name, keys=keys, verdict_ids=batch, full_tasks=False))
    groups.append(Group(name=STRUCTURE, keys=[], verdict_ids=[], full_tasks=True))
    if pm.duplicate_candidates:
        groups.append(Group(name="дублирование", keys=[], verdict_ids=[], full_tasks=False,
                            duplicates=pm.duplicate_candidates))
    return groups


# --- calls ---


def _schema(group: Group) -> type[BaseModel]:
    if group.name == STRUCTURE:
        return StructureOutput
    return DupOutput if group.duplicates else VerdictOutput


def _call(text: str, schema: type[BaseModel], client) -> LLMCall | LLMError:
    try:
        return call_structured(model=settings.model_match, reasoning=settings.reasoning_match,
                               instructions=PROMPT, input=text, schema=schema, client=client)
    except LLMError as exc:
        return exc


def _run_group(group: Group, shared: str, pm: Prematch, docs: dict[str, Document], client):
    """One call per group, plus one follow-up call for verdicts the model skipped."""
    calls = [_call(shared + "\n" + render_group(group, pm, docs), _schema(group), client)]
    if isinstance(calls[0], LLMCall) and group.verdict_ids:
        got = {v.before_id for v in calls[0].parsed.verdicts}
        missing = [i for i in group.verdict_ids if i not in got]
        if missing:
            retry = group.model_copy(update={"verdict_ids": missing})
            calls.append(_call(shared + "\n" + render_group(retry, pm, docs), VerdictOutput, client))
    return group, calls


# --- assembly helpers ---


def _ev(f: FnRef) -> Evidence:
    return Evidence(doc_id=f.doc_id, clause_id=f.clause_id, quote=f.quote, page=f.page)


def _ev_ref(r: Ref) -> Evidence:
    return Evidence(doc_id=r.doc_id, clause_id=r.clause_id, quote=r.quote)


def _ev_clause(doc: Document, clause_id: str, words: int = 20) -> Evidence | None:
    clause = doc.clause(clause_id)
    if clause is None:
        return None
    return Evidence(doc_id=doc.doc_id, clause_id=clause_id, quote=" ".join(clause.text.split()[:words]),
                    page=clause.page)


def _dedupe(evidence: list[Evidence]) -> list[Evidence]:
    seen, out = set(), []
    for e in evidence:
        if (e.doc_id, e.clause_id) not in seen:
            seen.add((e.doc_id, e.clause_id))
            out.append(e)
    return out


def _max_severity(values) -> Severity:
    return max(values, key=lambda s: SEVERITY_RANK[s], default="low")


def _join(values) -> str:
    return "; ".join(dict.fromkeys(v for v in values if v))


def _valid_ids(ids: list[str], pm: Prematch, side: str) -> list[FnRef]:
    return [f for f in (pm.fn(i) for i in dict.fromkeys(ids)) if f is not None and f.side == side]


def _clean_evidence(evidence: list[Evidence], docs: dict[str, Document]) -> list[Evidence]:
    """Functions and units keep only citations that pass the verifier (findings keep all and get judged)."""
    return [with_page(e, docs) for e in _dedupe(evidence) if check_evidence(e, docs) is None]


def _append(findings: list[Finding], fields: dict) -> None:
    """A finding needs at least one citation; without any it is not reported at all."""
    if fields.get("evidence"):
        findings.append(Finding(**fields))


def empty_clause_findings(docs: list[Document]) -> list[Finding]:
    out = []
    for doc in docs:
        for c in doc.clauses:
            if c.clause_id not in ("preamble", "toc") and not normalize(c.text):
                out.append(Finding(id="", type="structure", severity="low",
                                   summary=f"Дефект документа «{doc.name}»: пункт {c.clause_id} пустой (текст «{c.text}»).",
                                   evidence=[Evidence(doc_id=doc.doc_id, clause_id=c.clause_id, quote=c.text,
                                                      page=c.page)],
                                   recommendation="Заполнить или удалить пустой пункт."))
    return out


# --- conflicts ---


class GradedConflict(BaseModel):
    confidence: Literal["high", "medium", "low"]
    finding: dict
    functions: list[FnRef]


CONTROL_STEMS = ("контр", "надзо", "оцен", "монит", "качес", "ревиз")
GENERIC_STEMS = {"обще", "общес", "работ", "деяте", "внутр", "проце", "функц", "вопро", "докум", "получ",
                 "рамка", "рамки", "целях", "части", "соотв", "работн"}


def _stem_set(text: str) -> set[str]:
    return {w[:5] for w in normalize(text).split() if len(w) >= 4} - GENERIC_STEMS


def _is_control(f: FnRef) -> bool:
    return any(stem in normalize(f"{f.action} {f.object}") for stem in CONTROL_STEMS)


GENERIC_SHARE = 0.10  # a word found in more than 10% of a side's functions says nothing about the activity


GENERIC_CLAUSE_SHARE = 0.05  # ... or in more than 5% of its clauses («осуществлять», «Положение»)
GENERIC_MIN_COUNT = 3  # small documents: a word seen once or twice is never "generic"


def _frequent_stems(fns: list[FnRef], docs: list[Document]) -> set[str]:
    counts = Counter(stem for f in fns for stem in _stem_set(f"{f.object} {f.area}"))
    generic = {stem for stem, n in counts.items() if n >= GENERIC_MIN_COUNT and n > GENERIC_SHARE * len(fns)}
    clauses = [c.text for d in docs for c in d.clauses]
    in_clauses = Counter(stem for text in clauses for stem in _stem_set(text))
    return generic | {stem for stem, n in in_clauses.items()
                      if n >= GENERIC_MIN_COUNT and n > GENERIC_CLAUSE_SHARE * len(clauses)}


def _same_activity(performs: list[FnRef], controls: list[FnRef], generic: set[str]) -> tuple[list[FnRef], list[FnRef]]:
    """Keep performs/controls pairs about the same activity: shared distinctive words in object or area."""
    pairs = [(p, c) for p in performs for c in controls
             if (_stem_set(f"{p.object} {p.area}") & _stem_set(f"{c.object} {c.area}")) - generic]
    return list({p.id: p for p, _ in pairs}.values()), list({c.id: c for _, c in pairs}.values())


def _examples(fns: list[FnRef], n: int = 2) -> str:
    return "; ".join(f"«{f.action} {f.object}» (п. {f.clause_id})" for f in fns[:n])


def grade_conflicts(conflicts: list[Conflict], docs: list[Document], extractions: dict[str, Extraction],
                    pm: Prematch) -> list[GradedConflict]:
    """Three checkable conditions, each backed by a citation:
    1. the subject itself performs the activity (a non-control function owned by the subject);
    2. the subject itself controls the same activity (a control/supervision/quality function owned by the
       subject that shares significant object/area words with a performed one);
    3. staff or subordination confirms it (a verified quote from an org-structure clause naming a position
       other than the subject).
    3 of 3 -> conflict/high, 2 -> conflict/medium, 1 -> overlap (low), 0 -> dropped.
    A subject that is not a unit or position of the org structure is at most an overlap.
    One result per (side, subject): the best-supported candidate wins."""
    by_id = {d.doc_id: d for d in docs}
    norm = OwnerNormalizer(list(extractions.values()))
    structure = {(d.doc_id, c) for d in docs for c in _structure_ids(d, extractions.get(d.doc_id))}
    in_structure = {side: {norm.key(x.short_name if hasattr(x, "short_name") and x.short_name else x.name)
                           for d in docs if d.side == side and extractions.get(d.doc_id)
                           for x in extractions[d.doc_id].units + extractions[d.doc_id].roles}
                    for side in ("before", "after")}
    generic = {side: _frequent_stems(pm.before if side == "before" else pm.after,
                                     [d for d in docs if d.side == side]) for side in ("before", "after")}
    best: dict[tuple[str, str], tuple[int, GradedConflict]] = {}
    for c in conflicts:
        subject = norm.key(c.subject)

        def own(ids: list[str]) -> list[FnRef]:
            fns = (pm.fn(i) for i in dict.fromkeys(ids))
            return [f for f in fns if f and f.side == c.side and same_owner(f.owner_key, subject)]

        controls = [f for f in own(c.controls_ids) if _is_control(f)]
        performs = [f for f in own(c.performs_ids) if not _is_control(f) and f.id not in {x.id for x in controls}]
        if performs and controls:
            performs, controls = _same_activity(performs, controls, generic[c.side])
        refs = [e for e in (_ev_ref(r) for r in c.refs)
                if (e.doc_id, e.clause_id) in structure and by_id.get(e.doc_id) and by_id[e.doc_id].side == c.side
                and check_evidence(e, by_id) is None]
        staff = [s for s in dict.fromkeys(c.performer_roles + c.shared_staff)
                 if s.strip() and not same_owner(norm.key(s), subject)]
        met = [bool(performs), bool(controls), bool(refs and staff)]
        score = sum(met)
        if not any(same_owner(subject, k) for k in in_structure[c.side]):
            score = min(score, 1)  # conflicts are assessed for units and positions of the org structure
        if score == 0:
            continue
        confidence = {3: "high", 2: "medium", 1: "low"}[score]
        where = "в редакции «до»" if c.side == "before" else "в редакции «после»"
        parts = []
        if performs:
            parts.append(f"выполняет: {_examples(performs)}")
        if controls:
            parts.append(f"контролирует: {_examples(controls)}")
        if refs and staff:
            parts.append(f"по оргструктуре: {', '.join(staff[:4])}")
        facts = "; ".join(parts)
        comment = c.comment.removeprefix(CANDIDATE).lstrip(" :.—")
        evidence = _dedupe([_ev(f) for f in performs[:2] + controls[:2]] + refs[:2])
        if confidence == "low":
            finding = dict(id="", type="overlap", severity="low", confidence="low", evidence=evidence,
                           summary=f"Пересечение ответственности {where} у «{subject}»: {facts}. {comment} "
                                   f"Для конфликта интересов подтверждено одно условие из трёх.".strip(),
                           function_ids=[f.id for f in performs + controls if f.side == "before"])
        else:
            resolved = c.side == "before" and c.resolved_in_after
            tail = " В редакции «после» конфликт устранён реорганизацией." if resolved else ""
            severity = "low" if resolved else (c.severity if confidence == "high" else
                                               min(c.severity, "medium", key=lambda s: SEVERITY_RANK[s]))
            finding = dict(id="", type="conflict", severity=severity, confidence=confidence, evidence=evidence,
                           summary=f"Конфликт интересов {where} у «{subject}» (уверенность: {confidence}). "
                                   f"{CANDIDATE}: {facts}. {comment}{tail}",
                           function_ids=[f.id for f in performs + controls if f.side == "before"])
        graded = GradedConflict(confidence=confidence, finding=finding, functions=performs + controls)
        key = (c.side, subject)
        if key not in best or score > best[key][0]:
            best[key] = (score, graded)
    return [g for _, g in best.values()]


def _flag_conflict(fns: list[FnRef], functions: dict[str, Function]) -> None:
    for f in fns:
        ids = [f.id] if f.side == "before" else [b for b, fn in functions.items()
                                                  if any(e.clause_id == f.clause_id and e.doc_id == f.doc_id
                                                         for e in fn.evidence)]
        for fid in ids:
            if fid in functions:
                functions[fid].conflict_of_interest = True


# --- units ---


def unit_diff(docs: list[Document], extractions: dict[str, Extraction], pm: Prematch,
              functions: dict[str, Function], compositions: list[CompositionChange]) -> tuple[list[Unit], list[Finding]]:
    by_id = {d.doc_id: d for d in docs}
    norm = OwnerNormalizer(list(extractions.values()))
    sides: dict[str, dict[str, tuple[str, list[str], str]]] = {"before": {}, "after": {}}  # key -> (name, clauses, doc)
    for doc in docs:
        ex = extractions.get(doc.doc_id)
        if not ex:
            continue
        for u in ex.units:
            key = norm.key(u.short_name or u.name)
            sides[doc.side].setdefault(key, (u.name, u.clause_ids, doc.doc_id))
    unit_keys = set(sides["before"]) | set(sides["after"])
    for doc in docs:  # standalone positions that own functions (not heads of a unit)
        ex = extractions.get(doc.doc_id)
        for r in (ex.roles if ex else []):
            key = norm.key(r.name)
            if "/" in key or key in unit_keys:
                continue
            owns = [f for f in (pm.before if doc.side == "before" else pm.after) if same_owner(f.owner_key, key)]
            if owns:
                sides[doc.side].setdefault(key, (r.name, r.clause_ids, doc.doc_id))

    def evidence(key: str, side: str) -> list[Evidence]:
        _, clause_ids, doc_id = sides[side][key]
        ev = [_ev_clause(by_id[doc_id], c) for c in clause_ids[:2]]
        return [e for e in ev if e]

    def present(key: str, side: str) -> bool:
        fns = pm.before if side == "before" else pm.after
        return any(same_owner(key, k) for k in sides[side]) or any(same_owner(key, f.owner_key) for f in fns)

    units, findings = [], []
    comp_by_unit = {norm.key(c.unit): c for c in compositions}
    for key in sorted(set(sides["before"]) | set(sides["after"])):
        in_before, in_after = present(key, "before"), present(key, "after")
        if in_before and in_after:
            if key not in sides["before"] or key not in unit_keys:
                continue  # listed once under its "before" spelling; unchanged positions are not listed
            after_key = next((k for k in sides["after"] if same_owner(k, key)), None)
            name = sides["after"][after_key][0] if after_key else sides["before"][key][0]
            comp = comp_by_unit.get(key)
            ev = evidence(key, "before") + (evidence(after_key, "after") if after_key else [])
            units.append(Unit(name=name, status="preserved", before_ref=key, after_ref=after_key or key,
                              comment=comp.comment if comp else None, evidence=_clean_evidence(ev, by_id)))
            if comp and (comp.removed_roles or comp.added_roles):
                parts = []
                if comp.removed_roles:
                    parts.append("исключены должности: " + ", ".join(comp.removed_roles))
                if comp.added_roles:
                    parts.append("добавлены: " + ", ".join(comp.added_roles))
                _append(findings, dict(id="", type="structure", severity="medium",
                                        summary=f"Изменён состав «{key}»: {'; '.join(parts)}. {comp.comment}".strip(),
                                        evidence=[_ev_ref(r) for r in comp.refs] or ev))
        elif key in sides["after"] and not in_before:
            name = sides["after"][key][0]
            ev = evidence(key, "after")
            units.append(Unit(name=name, status="created", after_ref=key, evidence=_clean_evidence(ev, by_id)))
            _append(findings, dict(id="", type="structure", severity="low",
                                    summary=f"Создано: «{name}».", evidence=ev))
        elif key in sides["before"] and not in_after:
            name = sides["before"][key][0]
            owned = [functions[f.id] for f in pm.before if same_owner(f.owner_key, key) and f.id in functions]
            moved = [f for f in owned if f.status != "lost" and f.owner_after]
            lost = [f for f in owned if f.status == "lost"]
            ev = evidence(key, "before")
            if moved and len(moved) >= len(lost):
                target = Counter(f.owner_after for f in moved).most_common(1)[0][0]
                after_ev = next((e for f in moved for e in f.evidence if by_id[e.doc_id].side == "after"), None)
                units.append(Unit(name=name, status="transformed", before_ref=key, after_ref=target,
                                  comment=f"В редакции «после» отсутствует; {len(moved)} из {len(owned)} функций "
                                          f"найдены у «{target}» и других исполнителей.",
                                  evidence=_clean_evidence(ev + ([after_ev] if after_ev else []), by_id)))
                _append(findings, dict(id="", type="structure", severity="medium",
                                        summary=f"«{name}» отсутствует в редакции «после», но это преобразование: "
                                                f"{len(moved)} из {len(owned)} функций перешли к «{target}» и др., "
                                                f"не найдено {len(lost)}.",
                                        evidence=ev + ([after_ev] if after_ev else []),
                                        function_ids=[f.id for f in owned]))
            else:
                units.append(Unit(name=name, status="eliminated", before_ref=key,
                                  comment=f"{NOT_FOUND}; функций без аналога: {len(lost)} из {len(owned)}.",
                                  evidence=_clean_evidence(ev, by_id)))
                _append(findings, dict(id="", type="structure", severity="high",
                                        summary=f"«{name}» и его функции не найдены в загруженном комплекте \"после\".",
                                        evidence=ev, function_ids=[f.id for f in owned]))
    return units, findings


# --- main ---


def _dump_raw(results) -> None:
    """Raw model answers per group, for debugging (data/jobs is not committed)."""
    raw = [{"group": g.name, "verdict_ids": g.verdict_ids,
            "calls": [c.parsed.model_dump() if isinstance(c, LLMCall) else str(c) for c in calls]}
           for g, calls in results]
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    (settings.jobs_dir / "match_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")


def _collect(results) -> tuple[dict[str, Verdict], list[Duplicate], list[Conflict], list[CompositionChange], list, list]:
    verdicts, dups, conflicts, comps, calls, failed = {}, [], [], [], [], []
    for group, group_calls in results:
        for c in group_calls:
            calls.append(c)
            if isinstance(c, LLMError):
                failed.append(f"{group.name}: {c}")
                continue
            out = c.parsed
            if isinstance(out, VerdictOutput):
                allowed = set(group.verdict_ids)
                verdicts.update({v.before_id: v for v in out.verdicts if v.before_id in allowed})
            elif isinstance(out, DupOutput):
                dups += [Duplicate(after_ids=d.after_ids, action=d.action, object=d.object, area=d.area,
                                   comment=d.reason, severity=d.severity, recommendation=d.recommendation)
                         for d in out.decisions if d.is_duplicate]
            elif isinstance(out, StructureOutput):
                comps += out.composition_changes
                for ch in out.checks:
                    if not ch.is_conflict:
                        continue
                    conflicts.append(Conflict(side=ch.side, subject=ch.subject, performs_ids=ch.performs_ids,
                                              controls_ids=ch.control_ids, performer_roles=ch.performer_roles,
                                              shared_staff=ch.shared_staff, refs=ch.refs,
                                              resolved_in_after=ch.resolved_in_after, comment=ch.comment,
                                              severity=ch.severity))
    return verdicts, dups, conflicts, comps, calls, failed


def match(docs: list[Document], extractions: dict[str, Extraction], pm: Prematch,
          client=None) -> tuple[MatchResult, TraceStep]:
    step = TraceStep(step="match", started_at=datetime.now(timezone.utc), model=settings.model_match)
    by_id = {d.doc_id: d for d in docs}
    groups = build_groups(pm)
    shared = render_shared(docs, extractions, pm)
    with ThreadPoolExecutor(max_workers=max(1, min(MAX_PARALLEL_CALLS, len(groups)))) as pool:
        first = pool.submit(_run_group, groups[0], shared, pm, by_id, client)
        if len(groups) > 1 and client is None:
            time.sleep(WARMUP_DELAY_S)
        rest = [pool.submit(_run_group, g, shared, pm, by_id, client) for g in groups[1:]]
        results = [first.result()] + [f.result() for f in rest]
    verdicts, dups, conflicts, comps, calls, failed = _collect(results)
    if client is None:  # real runs only; tests inject a fake client
        _dump_raw(results)

    # statuses: prematch (deterministic) + model verdicts
    for p in pm.preserved:
        verdicts[p.before] = Verdict(before_id=p.before, status="preserved", after_ids=p.after, after_refs=[],
                                     search_note="", comment="", severity="low", recommendation=None)
    functions: dict[str, Function] = {}
    unresolved = []
    for b in pm.before:
        v = verdicts.get(b.id)
        if v is None:
            unresolved.append(b.id)
            continue
        after = _valid_ids(v.after_ids, pm, "after")
        status = v.status
        if status in ("transferred", "changed", "preserved") and not after and not v.after_refs:
            status = "lost" if status != "preserved" else status
        comment = v.comment or None
        if status == "lost" and (not comment or not comment.startswith(NOT_FOUND)):
            comment = f"{NOT_FOUND}. {comment or ''}".strip()
        evidence = [_ev(b)] + [_ev(a) for a in after[:3]] + [_ev_ref(r) for r in v.after_refs[:2]]
        owner_after = _join(a.owner_key for a in after) or (b.owner_key if status == "preserved" else None)
        functions[b.id] = Function(id=b.id, action=b.action, object=b.object, area=b.area, owner_before=b.owner_key,
                                   owner_after=owner_after, status=status, comment=comment,
                                   evidence=_clean_evidence(evidence, by_id), recommendation=v.recommendation)
        verdicts[b.id] = v.model_copy(update={"status": status, "after_ids": [a.id for a in after], "comment": comment or ""})

    findings: list[Finding] = []
    # losses and changes: one finding per source clause
    for status, ftype in (("lost", "loss"), ("changed", "change")):
        by_clause: dict[tuple, list[FnRef]] = defaultdict(list)
        for b in pm.before:
            if b.id in functions and functions[b.id].status == status:
                by_clause[(b.doc_id, b.clause_id)].append(b)
        for (doc_id, clause_id), fns in by_clause.items():
            vs = [verdicts[b.id] for b in fns]
            what = "; ".join(f"«{b.action} {b.object}»" for b in fns)
            if status == "lost":
                summary = f"{fns[0].owner_key}: {what} — {NOT_FOUND[0].lower() + NOT_FOUND[1:]}."
                evidence = [_ev(b) for b in fns]
            else:
                summary = f"{fns[0].owner_key}: {what} — изменено. {_join(v.comment for v in vs)}"
                evidence = [_ev(b) for b in fns] + [_ev(pm.fn(a)) for v in vs for a in v.after_ids[:2]]
                evidence += [_ev_ref(r) for v in vs for r in v.after_refs[:1]]
            _append(findings, dict(id="", type=ftype, severity=_max_severity(v.severity for v in vs),
                                    summary=summary, evidence=_dedupe(evidence), function_ids=[b.id for b in fns],
                                    recommendation=next((v.recommendation for v in vs if v.recommendation), None)))
    # transfers: one finding per (from, to)
    moves: dict[tuple[str, str], list[FnRef]] = defaultdict(list)
    for b in pm.before:
        f = functions.get(b.id)
        if f and f.status == "transferred":
            moves[(b.owner_key, f.owner_after or "?")].append(b)
    for (src, dst), fns in moves.items():
        examples = "; ".join(f"«{b.action} {b.object}» (п. {b.clause_id})" for b in fns[:3])
        more = f" и ещё {len(fns) - 3}" if len(fns) > 3 else ""
        evidence = []
        for b in fns[:4]:
            v = verdicts[b.id]
            evidence.append(_ev(b))
            evidence += [_ev(pm.fn(a)) for a in v.after_ids[:1]] or [_ev_ref(r) for r in v.after_refs[:1]]
        if same_owner(src, dst):
            summary = (f"«{src}»: изменён исполнитель или адресат, которому передаётся функция: {examples}{more}. "
                       f"{_join(verdicts[b.id].comment for b in fns[:3])}")
        else:
            summary = f"Перенос {len(fns)} функц. от «{src}» к «{dst}»: {examples}{more}."
        _append(findings, dict(id="", type="transfer", severity="low" if len(fns) < 5 else "medium",
                                summary=summary,
                                evidence=_dedupe(evidence), function_ids=[b.id for b in fns]))
    # duplicates (deduplicated across groups)
    seen_dups, dup_functions = set(), []
    for d in dups:
        fns = _valid_ids(d.after_ids, pm, "after")
        owners = {f.owner_key for f in fns}
        key = frozenset((f.doc_id, f.clause_id) for f in fns)
        if len(fns) < 2 or len(owners) < 2 or key in seen_dups:
            continue
        seen_dups.add(key)
        fid = f"D{len(dup_functions) + 1}"
        evidence = [_ev(f) for f in fns]
        dup_functions.append(Function(id=fid, action=d.action, object=d.object, area=d.area,
                                      owner_after=_join(sorted(owners)), status="duplicated", comment=d.comment,
                                      evidence=_clean_evidence(evidence, by_id), recommendation=d.recommendation))
        _append(findings, dict(id="", type="duplication", severity=d.severity,
                                summary=f"Дублирование «{d.action} {d.object}» ({d.area}) у: {', '.join(sorted(owners))}. "
                                        f"{d.comment}".strip(),
                                evidence=_dedupe(evidence), function_ids=[fid],
                                recommendation=d.recommendation))
    # conflicts: the model proposes, the code grades the evidence
    for grade in grade_conflicts(conflicts, docs, extractions, pm):
        _append(findings, grade.finding)
        if grade.confidence in ("high", "medium"):
            _flag_conflict(grade.functions, functions)

    units, structure = unit_diff(docs, extractions, pm, functions, comps)
    findings = structure + empty_clause_findings(docs) + findings
    order = {"structure": 0, "loss": 1, "change": 2, "transfer": 3, "duplication": 4, "conflict": 5, "overlap": 6}
    findings.sort(key=lambda f: order[f.type])
    findings = [f.model_copy(update={"id": f"F{i}"}) for i, f in enumerate(findings, 1)]

    warnings = [f"Сопоставление: вызов не удался — {e}" for e in failed]
    if unresolved:
        warnings.append(f"Для {len(unresolved)} функций «до» статус не определён: {', '.join(unresolved[:20])}")
    ok_calls = [c for c in calls if isinstance(c, LLMCall)]
    step.finished_at = datetime.now(timezone.utc)
    step.input_tokens = sum(c.input_tokens for c in calls)
    step.output_tokens = sum(c.output_tokens for c in calls)
    step.cached_tokens = sum(c.cached_tokens for c in calls)
    step.cost_usd = cost_usd(settings.model_match, step.input_tokens, step.output_tokens, step.cached_tokens)
    status_counts = Counter(f.status for f in functions.values())
    step.notes = (f"групп {len(groups)}, вызовов {len(calls)} (повторов {sum(c.attempts - 1 for c in ok_calls)}, "
                  f"неудачных {len(failed)}); вердиктов модели {len(verdicts) - len(pm.preserved)}; "
                  f"статусы: {dict(status_counts)}; дублей {len(dup_functions)}, кандидатов в конфликты {len(conflicts)}; "
                  f"не определено {len(unresolved)}")
    result = MatchResult(units=units, functions=list(functions.values()) + dup_functions, findings=findings,
                         warnings=warnings, complete=not failed and not unresolved)
    return result, step
