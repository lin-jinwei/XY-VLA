from __future__ import annotations

import re
from typing import Any

_LAP_COUNT_TRAIL = r"(?!\s*(?:laps?|circles?|rounds?|turns?|revolutions?|loops?|times)\b)"

_BILLBOARD_ID_CMD_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"billboard_id\s*=\s*(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bbillboard\s+id\s+(\d{1,2})\b", re.IGNORECASE),
    re.compile(
        r"\bportal\s+billboard\s+(?:number|marker)\s+(\d{1,2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bportal\s+billboard\s+(\d{{1,2}}){_LAP_COUNT_TRAIL}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbillboard\s+(?:number|marker)\s+(\d{1,2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bbillboard\s+(\d{{1,2}}){_LAP_COUNT_TRAIL}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btop\s*label\s+(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"rect(?:angular)?\s*frame\s*(\d{1,2})", re.IGNORECASE),
    re.compile(r"frame\s*(\d{1,2})", re.IGNORECASE),
    re.compile(
        r"portal(?:\s*(?:billboard|frame))?\s*(?:number|no\.?)?\s*(\d{1,2})",
        re.IGNORECASE,
    ),
)

_ORBIT_COUNT_SPAN_RE = re.compile(
    r"(?:\b\d+\s*(?:laps?|circles?|rounds?|turns?|revolutions?|loops?)\b"
    r"|\b(?:once|twice|thrice)\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+times\b"
    r"|\b\d+\s*times\b)",
    re.IGNORECASE,
)

_CMD_THROUGH_VERB_CLAUSE = re.compile(
    r"\b(?:(?:pass|fly|navigate|go|move)\s+through|(?:navigate|go)\s+to|pass\s+over)\b",
    re.IGNORECASE,
)

_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:[;.]|\b(?:then|and\s+then|after\s+that|next)\b)\s*",
    re.IGNORECASE,
)


def split_mission_clauses(text: str) -> list[str]:

    ins = str(text or "").strip()
    if not ins:
        return []
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(ins) if c.strip()] or [ins]


def mission_clause_for_billboard(clauses: list[str], billboard_id: int) -> str | None:

    if not clauses:
        return None
    bid = int(billboard_id)
    hit = [i for i, c in enumerate(clauses) if bid in portal_billboard_ids_in_text(c)]
    if not hit:
        return None
    anchor = hit[0]
    start = anchor
    for j in range(anchor - 1, -1, -1):
        ids = portal_billboard_ids_in_text(clauses[j])
        if ids and any(int(x) != bid for x in ids):
            break
        start = j
    parts: list[str] = []
    for j in range(start, len(clauses)):
        ids = portal_billboard_ids_in_text(clauses[j])
        if j > anchor and ids and any(int(x) != bid for x in ids):
            break
        parts.append(clauses[j])
    if not parts:
        return None
    return "; ".join(parts) if len(parts) > 1 else parts[0]


def _orbit_count_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _ORBIT_COUNT_SPAN_RE.finditer(str(text or ""))]


def _digit_in_orbit_count_context(start: int, end: int, orbit_spans: list[tuple[int, int]]) -> bool:
    for span_start, span_end in orbit_spans:
        if start < span_end and end > span_start:
            return True
    return False


def portal_billboard_ids_in_text(text: str) -> list[int]:

    ins = str(text or "")
    if not ins.strip():
        return []
    orbit_spans = _orbit_count_spans(ins)
    hits: list[tuple[int, int]] = []
    seen_digit_spans: set[tuple[int, int]] = set()
    for rx in _BILLBOARD_ID_CMD_RES:
        for m in rx.finditer(ins):
            try:
                n = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if not (1 <= n <= 20):
                continue
            d_start, d_end = m.start(1), m.end(1)
            if d_start < 0 or _digit_in_orbit_count_context(d_start, d_end, orbit_spans):
                continue
            if (d_start, d_end) in seen_digit_spans:
                continue
            seen_digit_spans.add((d_start, d_end))
            hits.append((d_start, n))
    hits.sort(key=lambda item: item[0])
    return [n for _, n in hits]


def extract_ordered_traversal_billboard_ids(instruction: str) -> list[int]:

    if not instruction or not str(instruction).strip():
        return []
    ins = str(instruction).strip()
    ordered: list[int] = []

    clauses = split_mission_clauses(ins)
    if not clauses:
        clauses = [ins]

    for clause in clauses:
        if _CMD_THROUGH_VERB_CLAUSE.search(clause) is None:
            continue
        ordered.extend(portal_billboard_ids_in_text(clause))

    if not ordered:
        if _CMD_THROUGH_VERB_CLAUSE.search(ins):
            ordered = portal_billboard_ids_in_text(ins)

    out: list[int] = []
    for bid in ordered:
        if out and out[-1] == bid:
            continue
        out.append(bid)
    return out


def extract_ordered_mission_billboard_ids(instruction: str) -> list[int]:

    if not instruction or not str(instruction).strip():
        return []
    ins = str(instruction).strip()
    clauses = split_mission_clauses(ins)
    ordered: list[int] = []
    for clause in clauses:
        for bid in portal_billboard_ids_in_text(clause):
            if ordered and ordered[-1] == bid:
                continue
            ordered.append(bid)
    if not ordered:
        ordered = portal_billboard_ids_in_text(ins)
    out: list[int] = []
    for bid in ordered:
        if out and out[-1] == bid:
            continue
        out.append(bid)
    return out


def extract_instruction_portal_billboard_id_single(instruction: str) -> int | None:

    ids = extract_ordered_traversal_billboard_ids(instruction)
    if len(ids) == 1:
        return ids[0]
    if not ids:
        found = portal_billboard_ids_in_text(instruction)
        if not found:
            return None
        first = found[0]
        if any(x != first for x in found):
            return None
        return first
    return None


def find_rect_portal_by_billboard_id(placed_cubes: list[dict], billboard_id: int) -> dict | None:

    cand: list[dict] = []
    bid = int(billboard_id)
    for c in placed_cubes:
        if c.get("portal_label") != bid:
            continue
        sh = str(c.get("shape", "")).lower()
        if sh == "rect_frame" or "rect" in sh or sh in ("square_frame", "frame"):
            cand.append(c)
    if len(cand) == 1:
        return cand[0]
    return None


def leg_sub_instruction_for_billboard_id(billboard_id: int) -> str:
    return (
        f"Fly through only the rectangular portal marked billboard_id={int(billboard_id)}. "
        f"Pass through the opening center along the inward normal; do not graze frame sides."
    )
