"""Matcher layer 1, no LLM: canonical owners + deterministic before/after function matching.

Owners. Extraction returns owners as written: «Директор департамента непрерывного мониторинга
системы внутреннего контроля», «Директор ДНМ», «работники ДНМ». They are reduced to a key:
  1. «... (далее Директор ДККМ)» -> «Директор ДККМ»;
  2. full unit names from the structure (any case, compared by word stems) -> abbreviation;
  3. owner mentions unit abbreviations -> key is the unit(s): «ДНМ», «ДИТААД/ДОА»
     (the top-level block is dropped when a concrete unit is named);
  4. otherwise the longest known role the owner starts with: «Главный аудитор или уполномоченное
     им лицо» -> «Главный аудитор»;
  5. otherwise a short owner contained in a unit name («внутренний аудит» -> БВА), else the owner itself.

Functions. A before function matches an after function when the quotes agree
(rapidfuzz >= QUOTE_MATCH) and action+object agree (>= ACTION_OBJECT_MATCH). Matched with the same
owner and an unchanged clause text -> preserved (closed here). Same owner but the clause text
changed -> "changed" candidate. Another owner -> "transferred" candidate. No match -> residual for the LLM.
"""
import re
from collections import defaultdict

from pydantic import BaseModel
from rapidfuzz import fuzz

from backend.agent.schemas import Document, Extraction, Side
from backend.agent.verify import normalize

QUOTE_MATCH = 90
ACTION_OBJECT_MATCH = 70
CLAUSE_UNCHANGED = 90
MAX_CANDIDATES = 3
DUPLICATE_MATCH = 85      # action+object similarity for duplicate candidates in the "after" set
MAX_DUPLICATE_CLUSTERS = 40
_DALEE = re.compile(r"\(\s*далее\s*[-–—]?\s*([^()]+?)\s*\)", re.I)
_PARENS = re.compile(r"\([^()]*\)")


class FnRef(BaseModel):
    id: str  # "B12" / "A40": short ids for prompts and result
    doc_id: str
    side: Side
    owner: str
    owner_key: str
    modality: str
    action: str
    object: str
    area: str
    clause_id: str
    quote: str
    page: int | None = None


class Pair(BaseModel):
    before: str
    after: list[str]
    score: float


class Prematch(BaseModel):
    before: list[FnRef]
    after: list[FnRef]
    preserved: list[Pair]
    transfer_candidates: list[Pair]
    changed_candidates: list[Pair]
    unmatched: list[str]
    unit_keys: dict[str, str]  # abbreviation/unit key -> display name
    block_keys: list[str]
    duplicate_candidates: list[list[str]] = []  # clusters of "after" function ids

    def fn(self, fn_id: str) -> FnRef | None:
        return self._index().get(fn_id)

    def _index(self) -> dict[str, FnRef]:
        if not hasattr(self, "_idx"):
            object.__setattr__(self, "_idx", {f.id: f for f in self.before + self.after})
        return self._idx

    @property
    def residual_count(self) -> int:
        return len(self.unmatched) + len(self.transfer_candidates) + len(self.changed_candidates)


def _stems(text: str) -> list[str]:
    return [w[:5] for w in normalize(text).split() if len(w) >= 3]


def _find_seq(hay: list[str], needle: list[str]) -> int:
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return -1


class OwnerNormalizer:
    def __init__(self, extractions: list[Extraction]):
        self.units: list[tuple[list[str], str]] = []  # (stems of the full name, abbreviation)
        self.unit_names: dict[str, str] = {}
        self.blocks: set[str] = set()
        roles: set[str] = set()
        for ex in extractions:
            for u in ex.units:
                abbr = (u.short_name or "").strip() or _PARENS.sub("", u.name).strip()
                stems = _stems(_PARENS.sub("", u.name))
                if stems and (stems, abbr) not in self.units:
                    self.units.append((stems, abbr))
                self.unit_names.setdefault(abbr, _PARENS.sub("", u.name).strip())
                if u.kind == "block":
                    self.blocks.add(abbr)
            roles.update(r.name.strip() for r in ex.roles)
        self.units.sort(key=lambda x: -len(x[0]))
        self.abbrs = sorted({a for _, a in self.units}, key=len, reverse=True)
        # roles keyed by their canonical form (unit names already folded into abbreviations)
        self.roles = sorted({self._fold_units(r) for r in roles}, key=lambda r: -len(_stems(r)))

    def _fold_units(self, text: str) -> str:
        words = text.split()
        for stems, abbr in self.units:
            if len(stems) < 2:
                continue
            word_stems = [w[:5] for w in (normalize(x) for x in words)]
            # align stems over words that survive the length filter
            idx_map = [i for i, w in enumerate(word_stems) if len(normalize(words[i]).replace(" ", "")) >= 3]
            seq = [word_stems[i] for i in idx_map]
            pos = _find_seq(seq, stems)
            if pos >= 0:
                start, end = idx_map[pos], idx_map[pos + len(stems) - 1]
                words = words[:start] + [abbr] + words[end + 1:]
        return " ".join(words)

    def key(self, owner: str) -> str:
        text = owner.strip()
        m = _DALEE.search(text)
        if m:
            text = m.group(1)
        text = _PARENS.sub("", text).strip(" ,.;")
        text = self._fold_units(text)
        found = [a for a in self.abbrs if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text)]
        concrete = [a for a in found if a not in self.blocks]
        if concrete or found:
            return "/".join(sorted(concrete or found))
        stems = _stems(text)
        best = None
        for role in self.roles:
            rs = _stems(role)
            if rs and stems[:len(rs)] == rs and (best is None or len(rs) > len(_stems(best))):
                best = role
        if best:
            return best
        if 0 < len(stems) <= 3:
            for ustems, abbr in self.units:
                if _find_seq(ustems, stems) >= 0:
                    return abbr
        return text[:1].upper() + text[1:] if text else owner


def same_owner(a: str, b: str) -> bool:
    if a == b:
        return True
    sa, sb = _stems(a), _stems(b)
    return bool(sa) and sa == sb  # short abbreviations have no stems and compare only exactly


def build_refs(docs: list[Document], extractions: dict[str, Extraction], norm: OwnerNormalizer) -> list[FnRef]:
    refs, counters = [], {"before": 0, "after": 0}
    for doc in docs:
        ex = extractions.get(doc.doc_id)
        if ex is None:
            continue
        for f in ex.functions:
            counters[doc.side] += 1
            clause = doc.clause(f.clause_id)
            refs.append(FnRef(id=f"{doc.side[0].upper()}{counters[doc.side]}", doc_id=doc.doc_id, side=doc.side,
                              owner=f.owner, owner_key=norm.key(f.owner), modality=f.modality, action=f.action,
                              object=f.object, area=f.area, clause_id=f.clause_id, quote=f.quote,
                              page=clause.page if clause else None))
    return refs


def quote_similarity(a: str, b: str) -> float:
    qa, qb = normalize(a), normalize(b)
    if not qa or not qb:
        return 0.0
    score = fuzz.ratio(qa, qb)
    if min(len(qa), len(qb)) / max(len(qa), len(qb)) >= 0.5:
        score = max(score, fuzz.partial_ratio(qa, qb))
    return score


def prematch(docs: list[Document], extractions: dict[str, Extraction]) -> Prematch:
    norm = OwnerNormalizer(list(extractions.values()))
    refs = build_refs(docs, extractions, norm)
    before = [r for r in refs if r.side == "before"]
    after = [r for r in refs if r.side == "after"]
    by_doc = {d.doc_id: d for d in docs}

    def clause_text(r: FnRef) -> str:
        c = by_doc[r.doc_id].clause(r.clause_id)
        return normalize(c.text) if c else ""

    after_ao = [normalize(f"{a.action} {a.object}") for a in after]
    preserved, transfers, changed, unmatched = [], [], [], []
    for b in before:
        b_ao = normalize(f"{b.action} {b.object}")
        matches = []
        for a, a_ao in zip(after, after_ao):
            if fuzz.token_set_ratio(b_ao, a_ao) < ACTION_OBJECT_MATCH:
                continue
            q = quote_similarity(b.quote, a.quote)
            if q >= QUOTE_MATCH:
                matches.append((same_owner(a.owner_key, b.owner_key), q, a))
        if not matches:
            unmatched.append(b.id)
            continue
        matches.sort(key=lambda m: (m[0], m[1]), reverse=True)
        own = [m for m in matches if m[0]]
        if own:
            top = own[0]
            if fuzz.ratio(clause_text(b), clause_text(top[2])) >= CLAUSE_UNCHANGED:
                preserved.append(Pair(before=b.id, after=[top[2].id], score=top[1]))
            else:
                changed.append(Pair(before=b.id, after=[m[2].id for m in own[:MAX_CANDIDATES]], score=top[1]))
        else:
            transfers.append(Pair(before=b.id, after=[m[2].id for m in matches[:MAX_CANDIDATES]], score=matches[0][1]))
    return Prematch(before=before, after=after, preserved=preserved, transfer_candidates=transfers,
                    changed_candidates=changed, unmatched=unmatched, unit_keys=norm.unit_names,
                    block_keys=sorted(norm.blocks), duplicate_candidates=duplicate_candidates(after, norm.blocks))


def _owner_parts(key: str) -> set[str]:
    return set(key.split("/"))


def duplicate_candidates(after: list[FnRef], blocks: set[str]) -> list[list[str]]:
    """Clusters of "after" functions with the same action+object at different, non-overlapping owners.
    Candidates only: the model decides whether the area of responsibility is really the same."""
    fns = [a for a in after if not (_owner_parts(a.owner_key) & blocks)]
    texts = [normalize(f"{a.action} {a.object}") for a in fns]
    parent = list(range(len(fns)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    scored = []
    for i in range(len(fns)):
        for j in range(i + 1, len(fns)):
            a, b = fns[i], fns[j]
            if (a.doc_id, a.clause_id) == (b.doc_id, b.clause_id) or same_owner(a.owner_key, b.owner_key):
                continue
            if _owner_parts(a.owner_key) & _owner_parts(b.owner_key):
                continue
            score = fuzz.token_set_ratio(texts[i], texts[j])
            if score >= DUPLICATE_MATCH and fuzz.ratio(texts[i], texts[j]) >= 60:
                scored.append((score, i, j))
    for _, i, j in sorted(scored, reverse=True):
        parent[root(i)] = root(j)
    clusters: dict[int, list[str]] = defaultdict(list)
    for i in {x for _, a, b in scored for x in (a, b)}:
        clusters[root(i)].append(fns[i].id)
    ranked = sorted(clusters.values(), key=len, reverse=True)
    return [sorted(c, key=lambda x: int(x[1:])) for c in ranked if 2 <= len(c) <= 6][:MAX_DUPLICATE_CLUSTERS]


def owners_summary(refs: list[FnRef]) -> dict[str, list[FnRef]]:
    groups: dict[str, list[FnRef]] = defaultdict(list)
    for r in refs:
        groups[r.owner_key].append(r)
    return dict(groups)


def notes(pm: Prematch) -> str:
    return (f"функций «до» {len(pm.before)}, «после» {len(pm.after)}; закрыто детерминированно (preserved) "
            f"{len(pm.preserved)}; в модель: кандидатов transferred {len(pm.transfer_candidates)}, "
            f"changed {len(pm.changed_candidates)}, без пары {len(pm.unmatched)}; "
            f"кандидатов в дубли «после»: {len(pm.duplicate_candidates)}")
