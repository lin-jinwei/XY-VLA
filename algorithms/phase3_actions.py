from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .instruction_parse import (
    _BILLBOARD_ID_CMD_RES,
    extract_ordered_mission_billboard_ids,
    find_rect_portal_by_billboard_id,
    mission_clause_for_billboard,
    portal_billboard_ids_in_text,
    split_mission_clauses,
)
from .portal_geometry import (
    is_portal_object,
    opening_center_deviation_m,
    point_in_portal_opening,
    portal_bar_world_aabbs,
    portal_cuboid_half_extents_local,
    portal_frame_surface_point,
    portal_hover_world_position,
    portal_is_solid_cuboid,
    portal_local_coords,
    portal_opening_axes,
    portal_opening_center_world,
    portal_opening_half_diagonal,
    portal_pass_through_direction,
    portal_solid_pass_over_crossing,
    project_point_to_portal_axis,
    segment_capsule_collision_free,
    sphere_hits_inflated_aabb,
)

DEFAULT_COLLISION_PAD_M = 0.07


class BasicAction(str, Enum):
    FLY_BY = "fly_by"
    PASS_THROUGH = "pass_through"
    ORBIT = "orbit"
    COLLISION = "collision"
    HOVER = "hover"


_ACTION_PATTERNS: tuple[tuple[BasicAction, re.Pattern[str]], ...] = (
    (
        BasicAction.COLLISION,
        re.compile(
            r"\b(?:collid(?:e|ing|es)?|crash(?:es|ing)?|hit(?:s|ting)?|"
            r"ram(?:s|ming)?|strike(?:s|ing)?|impact(?:s|ing)?|smash(?:es|ing)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        BasicAction.HOVER,
        re.compile(
            r"\b(?:hover(?:ing|s)?|loiter(?:ing|s)?|hold\s+(?:position|still|above|over)|"
            r"stay\s+(?:above|over|still))\b",
            re.IGNORECASE,
        ),
    ),
    (
        BasicAction.ORBIT,
        re.compile(
            r"\b(?:orbit(?:ing|s)?|circle(?:s|ing)?|loop\s+around|"
            r"inspect(?:ing|s)?|observe(?:s|ing)?|recon(?:naiss(?:ance)?)?|"
            r"surveil(?:lance)?|scout(?:ing|s)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        BasicAction.PASS_THROUGH,
        re.compile(
            r"\b(?:(?:pass|fly|navigate|go|move|cross|enter)\s+through|"
            r"(?:enter|cross)\s+(?:the\s+)?(?:portal|opening|frame)|"
            r"(?:navigate|go)\s+to|pass\s+over)\b",
            re.IGNORECASE,
        ),
    ),
    (
        BasicAction.FLY_BY,
        re.compile(
            r"\b(?:fly\s+(?:by|past)|pass\s+(?:by|past)|skirt(?:s|ing)?|bypass(?:es|ing)?|"
            r"beside|nearby|graze(?:s|ing)?|without\s+entering)\b",
            re.IGNORECASE,
        ),
    ),
)

_ORBIT_LAPS_RE = re.compile(
    r"(\d+)\s*(?:laps?|circles?|rounds?|turns?|revolutions?|loops?)",
    re.IGNORECASE,
)

_ORBIT_LAPS_WORD_RES: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bonce\b|\bone\s+(?:full\s+)?circle\b|\bone\s+lap\b", re.IGNORECASE), 1),
    (re.compile(r"\btwice\b|\btwo\s+times\b|\bdouble\b|\b2\s*times\b", re.IGNORECASE), 2),
    (re.compile(r"\bthrice\b|\bthree\s+times\b|\btriple\b|\b3\s*times\b", re.IGNORECASE), 3),
    (re.compile(r"\bfour\s+times\b|\bquadruple\b|\b4\s*times\b", re.IGNORECASE), 4),
)

_ORBIT_CW_RE = re.compile(
    r"\b(?:clockwise|cw|right[-\s]?hand(?:ed)?)\b",
    re.IGNORECASE,
)
_ORBIT_CCW_RE = re.compile(
    r"\b(?:counter[-\s]?clockwise|anticlockwise|ccw|left[-\s]?hand(?:ed)?)\b",
    re.IGNORECASE,
)
_ORBIT_OPENING_PLANE_RE = re.compile(
    r"\b(?:around\s+the\s+opening|in\s+(?:the\s+)?(?:plane\s+of\s+)?(?:the\s+)?(?:portal|frame|opening)|"
    r"face\s+of\s+the\s+(?:portal|frame)|vertical\s+orbit|opening\s+plane)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OrbitParams:


    laps: int = 1
    turn_sign: int = 1
    horizontal_above: bool = True


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return (v / n).astype(np.float64)


def _aabb_lo_hi(aabb: tuple | list | None) -> tuple[np.ndarray, np.ndarray]:
    if aabb is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    lo = np.asarray(aabb[0], dtype=np.float64).reshape(3)
    hi = np.asarray(aabb[1], dtype=np.float64).reshape(3)
    return lo, hi


def _aabb_top_z(aabb: tuple | list | None, fallback: float) -> float:
    _, hi = _aabb_lo_hi(aabb)
    if float(np.linalg.norm(hi)) < 1e-9:
        return float(fallback)
    return float(hi[2])


def _drone_radius(drone_body_half: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(drone_body_half, dtype=np.float64).reshape(3)))


def _safe_clearance_xy(aabb: tuple | list | None, zone_radius: float, drone_r: float) -> float:
    lo, hi = _aabb_lo_hi(aabb)
    ext = hi[:2] - lo[:2]
    half_xy = float(max(ext[0], ext[1]) * 0.5) if float(np.linalg.norm(ext)) > 1e-9 else zone_radius
    return max(float(zone_radius), half_xy) + float(drone_r) + 0.04


def _parse_orbit_laps(clause: str, *, default: int = 1) -> int:
    return int(_parse_orbit_params(clause, default_laps=default).laps)


def _parse_orbit_params(clause: str, *, default_laps: int = 1) -> OrbitParams:

    text = str(clause or "")
    laps = int(default_laps)
    for m in _ORBIT_LAPS_RE.finditer(text):
        for g in m.groups():
            if g is not None:
                try:
                    laps = max(laps, max(1, int(g)))
                except ValueError:
                    continue
    for rx, n in _ORBIT_LAPS_WORD_RES:
        if rx.search(text):
            laps = max(laps, int(n))

    turn_sign = 1
    if _ORBIT_CCW_RE.search(text):
        turn_sign = 1
    elif _ORBIT_CW_RE.search(text):
        turn_sign = -1

    horizontal_above = _ORBIT_OPENING_PLANE_RE.search(text) is None
    return OrbitParams(laps=int(laps), turn_sign=int(turn_sign), horizontal_above=horizontal_above)


def _action_from_clause(clause: str) -> BasicAction | None:
    for action, rx in _ACTION_PATTERNS:
        if rx.search(clause):
            return action
    return None


def _zone_billboard_id(
    zone_name: str,
    zone_center: np.ndarray,
    placed_cubes: list[dict] | None,
) -> int | None:
    for rx in _BILLBOARD_ID_CMD_RES:
        m = rx.search(zone_name)
        if m:
            try:
                bid = int(m.group(1))
                if 1 <= bid <= 20:
                    return bid
            except (ValueError, IndexError):
                pass
    if not placed_cubes:
        return None
    cen = np.asarray(zone_center, dtype=np.float64).reshape(3)
    best: tuple[float, int] | None = None
    for c in placed_cubes:
        bid = c.get("portal_label")
        if bid is None:
            continue
        pos = np.asarray(c.get("pos", [0, 0, 0]), dtype=np.float64).reshape(3)
        d = float(np.linalg.norm(pos - cen))
        if best is None or d < best[0]:
            best = (d, int(bid))
    if best is not None and best[0] < 0.25:
        return best[1]
    return None


def _zone_in_mission_billboards(
    mission_cmd: str | None,
    *,
    zone_name: str,
    zone_center: np.ndarray,
    placed_cubes: list[dict] | None,
) -> bool:

    ins = str(mission_cmd or "").strip()
    if not ins:
        return True
    mission_bids = extract_ordered_mission_billboard_ids(ins)
    if not mission_bids:
        return True
    bid = _zone_billboard_id(zone_name, zone_center, placed_cubes)
    if bid is None:
        return False
    return int(bid) in {int(x) for x in mission_bids}


def resolve_mission_clause_for_zone(
    mission_cmd: str | None,
    *,
    zone_name: str,
    zone_center: np.ndarray,
    placed_cubes: list[dict] | None = None,
) -> str:

    ins = str(mission_cmd or "").strip()
    if not ins:
        return str(zone_name)
    clauses = split_mission_clauses(ins)
    bid = _zone_billboard_id(zone_name, zone_center, placed_cubes)
    mission_bids = extract_ordered_mission_billboard_ids(ins)
    mission_bid_set = {int(x) for x in mission_bids}

    if bid is not None:
        merged = mission_clause_for_billboard(clauses, int(bid))
        if merged:
            return merged
        if mission_bid_set and int(bid) not in mission_bid_set:
            return str(zone_name)

    unique: list[int] = []
    for b in mission_bids:
        if b not in unique:
            unique.append(b)
    if len(unique) == 1 and bid is not None and int(bid) == int(unique[0]):
        merged = mission_clause_for_billboard(clauses, unique[0])
        if merged:
            return merged

    name_l = zone_name.lower()
    skip_tokens = frozenset(("rect", "frame", "portal", "billboard", "number", "gate", "cube"))
    for i, c in enumerate(clauses):
        tokens = re.findall(r"[a-z]{3,}", c.lower())
        if not any(t in name_l for t in tokens if t not in skip_tokens):
            continue
        if placed_cubes:
            for portal in placed_cubes:
                col = str(portal.get("color_name", portal.get("color", ""))).lower()
                if col and col in name_l and portal.get("portal_label") is not None:
                    pb = int(portal["portal_label"])
                    if mission_bid_set and pb not in mission_bid_set:
                        continue
                    merged = mission_clause_for_billboard(clauses, pb)
                    if merged:
                        return merged
        if bid is not None and mission_bid_set and int(bid) not in mission_bid_set:
            continue
        parts = [clauses[i]]
        for j in range(i + 1, len(clauses)):
            ids = portal_billboard_ids_in_text(clauses[j])
            if ids:
                break
            parts.append(clauses[j])
        return "; ".join(parts) if len(parts) > 1 else parts[0]

    if len(clauses) == 1 and _zone_in_mission_billboards(
        ins, zone_name=zone_name, zone_center=zone_center, placed_cubes=placed_cubes
    ):
        return clauses[0]
    return str(zone_name)


def entry_key(name: str, center: np.ndarray | list) -> str:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    return f"{name}|{c[0]:.4f},{c[1]:.4f},{c[2]:.4f}"


def _parse_entry_key_center(key: str) -> np.ndarray | None:
    if "|" not in key:
        return None
    try:
        xyz = key.rsplit("|", 1)[1].split(",")
        return np.asarray([float(x) for x in xyz[:3]], dtype=np.float64)
    except (ValueError, IndexError):
        return None


def lookup_action_cache(
    action_cache: dict[str, tuple[BasicAction, int]] | None,
    *,
    zone_name: str,
    zone_center: np.ndarray,
    placed_cubes: list[dict] | None = None,
) -> tuple[BasicAction, int] | None:

    if not action_cache:
        return None
    cen = np.asarray(zone_center, dtype=np.float64).reshape(3)
    key = entry_key(str(zone_name), cen)
    if key in action_cache:
        act, laps = action_cache[key]
        return act, int(laps)

    bid = _zone_billboard_id(str(zone_name), cen, placed_cubes)
    best: tuple[float, tuple[BasicAction, int]] | None = None
    for cache_key, val in action_cache.items():
        cache_name = cache_key.split("|", 1)[0]
        cache_cen = _parse_entry_key_center(cache_key)
        cache_bid = _zone_billboard_id(cache_name, cache_cen if cache_cen is not None else cen, placed_cubes)
        if bid is not None and cache_bid is not None and int(cache_bid) == int(bid):
            dist = (
                float(np.linalg.norm(cache_cen - cen))
                if cache_cen is not None
                else 0.0
            )
            if best is None or dist < best[0]:
                best = (dist, val)
            continue
        if cache_cen is not None and float(np.linalg.norm(cache_cen - cen)) < 0.06:
            if str(cache_name).lower() == str(zone_name).lower():
                dist = float(np.linalg.norm(cache_cen - cen))
                if best is None or dist < best[0]:
                    best = (dist, val)
    if best is not None:
        act, laps = best[1]
        return act, int(laps)
    return None


def _action_laps_from_clause(clause: str) -> tuple[BasicAction, int] | None:

    act = _action_from_clause(clause)
    if act is None:
        return None
    laps = _parse_orbit_laps(clause, default=1) if act == BasicAction.ORBIT else 1
    return act, int(laps)


def basic_action_from_clause_keywords(clause: str) -> tuple[BasicAction, int] | None:

    return _action_laps_from_clause(clause)


def has_explicit_basic_action_verb(clause: str) -> bool:

    return _action_from_clause(clause) is not None


def basic_action_from_mission_zone(
    mission_cmd: str | None,
    *,
    zone_name: str,
    zone_center: np.ndarray,
    placed_cubes: list[dict] | None = None,
) -> tuple[BasicAction, int] | None:

    ins = str(mission_cmd or "").strip()
    if not ins:
        return None
    if not _zone_in_mission_billboards(
        ins,
        zone_name=str(zone_name),
        zone_center=zone_center,
        placed_cubes=placed_cubes,
    ):
        return None
    clause = resolve_mission_clause_for_zone(
        ins,
        zone_name=str(zone_name),
        zone_center=zone_center,
        placed_cubes=placed_cubes,
    )
    return basic_action_from_clause_keywords(clause)


def resolve_basic_action_for_zone(
    mission_cmd: str | None,
    *,
    zone_name: str,
    zone_center: np.ndarray,
    is_target: bool = False,
    placed_cubes: list[dict] | None = None,
    action_cache: dict[str, tuple[BasicAction, int]] | None = None,
) -> tuple[BasicAction, int]:

    ins = str(mission_cmd or "").strip()
    laps = 1
    if not ins:
        return (BasicAction.PASS_THROUGH if is_target else BasicAction.FLY_BY, laps)

    if not _zone_in_mission_billboards(
        ins,
        zone_name=str(zone_name),
        zone_center=zone_center,
        placed_cubes=placed_cubes,
    ):
        return BasicAction.FLY_BY, laps

    clause = resolve_mission_clause_for_zone(
        ins,
        zone_name=str(zone_name),
        zone_center=zone_center,
        placed_cubes=placed_cubes,
    )
    keyword = basic_action_from_clause_keywords(clause)
    if keyword is not None:
        return keyword

    cached = lookup_action_cache(
        action_cache,
        zone_name=str(zone_name),
        zone_center=zone_center,
        placed_cubes=placed_cubes,
    )
    if cached is not None:
        return cached

    if is_target or (
        clause
        and re.search(
            r"through|pass\s+over|(?:navigate|go)\s+to",
            clause,
            re.IGNORECASE,
        )
    ):
        return BasicAction.PASS_THROUGH, laps
    return BasicAction.FLY_BY, laps


def _find_trajectory_zone_span(
    trajectory: np.ndarray,
    zone_center: np.ndarray,
    zone_radius: float,
    drone_r: float,
    *,
    contact_idx: int,
) -> tuple[int, int]:

    pts = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0, 0
    contact_idx = int(max(0, min(contact_idx, pts.shape[0] - 1)))
    thresh = float(zone_radius) + float(drone_r) + 0.01

    def _inside(i: int) -> bool:
        return float(np.linalg.norm(pts[i] - zone_center)) <= thresh

    i0 = contact_idx
    while i0 > 0 and _inside(i0 - 1):
        i0 -= 1
    i1 = contact_idx
    while i1 < pts.shape[0] - 1 and _inside(i1 + 1):
        i1 += 1
    return i0, i1


def _append_unique(points: list[np.ndarray], p: np.ndarray, *, eps: float = 1e-4) -> None:
    q = np.asarray(p, dtype=np.float64).reshape(3)
    if points and float(np.linalg.norm(points[-1] - q)) < eps:
        return
    points.append(q.copy())


def _placed_object_aabb_lo_hi_world(c: dict) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
    bh = c.get("bounds_half")
    if bh is not None:
        h = np.asarray(bh, dtype=np.float64).reshape(3)
        return pos - h, pos + h
    half = float(c.get("half", 0.025))
    h = np.array([half, half, half], dtype=np.float64)
    return pos - h, pos + h


def _portal_opening_axes(portal: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return portal_opening_axes(portal)


def _same_portal(a: dict, b: dict) -> bool:
    la = a.get("portal_label")
    lb = b.get("portal_label")
    if la is not None and lb is not None:
        return int(la) == int(lb)
    pa = np.asarray(a.get("pos", [0, 0, 0]), dtype=np.float64).reshape(3)
    pb = np.asarray(b.get("pos", [0, 0, 0]), dtype=np.float64).reshape(3)
    return float(np.linalg.norm(pa - pb)) < 0.05


def _resolve_portal_for_zone(
    zone_name: str,
    zone_center: np.ndarray,
    placed_cubes: list[dict] | None,
) -> dict | None:
    if not placed_cubes:
        return None
    bid = _zone_billboard_id(zone_name, zone_center, placed_cubes)
    if bid is not None:
        portal = find_rect_portal_by_billboard_id(placed_cubes, bid)
        if portal is not None:
            return portal
    cen = np.asarray(zone_center, dtype=np.float64).reshape(3)
    best: tuple[float, dict] | None = None
    for c in placed_cubes:
        sh = str(c.get("shape", "")).lower()
        if sh not in ("rect_frame", "square_frame", "frame") and "rect" not in sh:
            continue
        pos = np.asarray(c.get("pos", [0, 0, 0]), dtype=np.float64).reshape(3)
        d = float(np.linalg.norm(pos - cen))
        if best is None or d < best[0]:
            best = (d, c)
    if best is not None and best[0] < 0.35:
        return best[1]
    return None


def _portal_pass_through_direction(portal: dict, entry: np.ndarray) -> np.ndarray:
    return portal_pass_through_direction(portal, entry)


def _portal_pass_offsets(portal: dict, drone_r: float) -> tuple[float, float]:
    if portal_is_solid_cuboid(portal):
        half_u, _, _ = portal_cuboid_half_extents_local(portal)
        offset = max(0.15, float(half_u) + float(drone_r) + 0.12)
        return offset, offset
    thickness = float(portal.get("thickness", 0.01))
    depth = float(portal.get("depth", thickness))
    frame_extent = max(thickness, depth * 0.5)
    offset = max(0.15, frame_extent + float(drone_r) + 0.08)
    return offset, offset


def _obstacle_aabbs_excluding_portal_bars(
    placed_cubes: list[dict] | None,
    portal: dict,
) -> list[tuple[np.ndarray, np.ndarray]]:

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for c in placed_cubes or []:
        if is_portal_object(c) and _same_portal(c, portal):
            continue
        if is_portal_object(c):
            out.extend(portal_bar_world_aabbs(c))
        else:
            lo, hi = _placed_object_aabb_lo_hi_world(c)
            out.append((lo.astype(np.float64), hi.astype(np.float64)))
    return out


def _pass_through_link_collision_free(
    p0: np.ndarray,
    p1: np.ndarray,
    portal: dict,
    placed_cubes: list[dict] | None,
    drone_r: float,
    pad: float,
) -> bool:

    a = np.asarray(p0, dtype=np.float64).reshape(3)
    b = np.asarray(p1, dtype=np.float64).reshape(3)
    mid = (a + b) * 0.5
    other = _obstacle_aabbs_excluding_portal_bars(placed_cubes, portal)
    if point_in_portal_opening(mid, portal, margin=0.05):
        if not segment_capsule_collision_free(a, b, other, drone_r, pad):
            return False
        bar_pad = max(0.012, float(pad) * 0.35)
        return segment_capsule_collision_free(
            a, b, portal_bar_world_aabbs(portal), drone_r, bar_pad
        )
    all_obs = _obstacle_aabbs_from_placed(placed_cubes, focus_portal=portal)
    return segment_capsule_collision_free(a, b, all_obs, drone_r, pad)


def _obstacle_aabbs_from_placed(
    placed_cubes: list[dict] | None,
    *,
    exclude_portal: dict | None = None,
    focus_portal: dict | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for c in placed_cubes or []:
        if focus_portal is not None and _same_portal(c, focus_portal):
            out.extend(portal_bar_world_aabbs(focus_portal))
            continue
        if exclude_portal is not None and _same_portal(c, exclude_portal):
            continue
        if is_portal_object(c):
            out.extend(portal_bar_world_aabbs(c))
        else:
            lo, hi = _placed_object_aabb_lo_hi_world(c)
            out.append((lo.astype(np.float64), hi.astype(np.float64)))
    return out


def _point_collision_free(
    p: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
    pad: float,
) -> bool:
    pt = np.asarray(p, dtype=np.float64).reshape(3)
    for lo, hi in aabbs:
        if sphere_hits_inflated_aabb(pt, lo, hi, radius, pad):
            return False
    return True


def _segment_collision_free(
    p0: np.ndarray,
    p1: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
    pad: float,
    *,
    samples: int = 12,
) -> bool:
    return segment_capsule_collision_free(p0, p1, aabbs, radius, pad)


def _min_z_clearance_for_xy_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    *,
    xy_pad: float,
) -> float:
    z_need = max(float(p0[2]), float(p1[2]))
    hit = False
    for lo, hi in aabbs:
        lo2 = lo.copy()
        hi2 = hi.copy()
        lo2[0] -= xy_pad
        lo2[1] -= xy_pad
        hi2[0] += xy_pad
        hi2[1] += xy_pad
        if _segment_intersects_rect_2d(
            float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]),
            float(lo2[0]), float(lo2[1]), float(hi2[0]), float(hi2[1]),
        ):
            z_need = max(z_need, float(hi[2]))
            hit = True
    return z_need if hit else max(float(p0[2]), float(p1[2]))


def _segments_intersect_proper_2d(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
    *, eps: float = 1e-9,
) -> bool:
    def orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        return (by - ay) * (cx - bx) - (bx - ax) * (cy - by)

    def on_seg(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
        return (
            min(ax, cx) - eps <= bx <= max(ax, cx) + eps
            and min(ay, cy) - eps <= by <= max(ay, cy) + eps
        )

    o1 = orient(x1, y1, x2, y2, x3, y3)
    o2 = orient(x1, y1, x2, y2, x4, y4)
    o3 = orient(x3, y3, x4, y4, x1, y1)
    o4 = orient(x3, y3, x4, y4, x2, y2)
    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True
    if abs(o1) <= eps and on_seg(x1, y1, x3, y3, x2, y2):
        return True
    if abs(o2) <= eps and on_seg(x1, y1, x4, y4, x2, y2):
        return True
    if abs(o3) <= eps and on_seg(x3, y3, x1, y1, x4, y4):
        return True
    if abs(o4) <= eps and on_seg(x3, y3, x2, y2, x4, y4):
        return True
    return False


def _segment_intersects_rect_2d(
    x0: float, y0: float, x1: float, y1: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
    *, eps: float = 1e-9,
) -> bool:
    a, b = min(rx0, rx1), max(rx0, rx1)
    c, d = min(ry0, ry1), max(ry0, ry1)

    def inside(x: float, y: float) -> bool:
        return a - eps <= x <= b + eps and c - eps <= y <= d + eps

    if inside(x0, y0) or inside(x1, y1):
        return True
    if abs(x0 - x1) < eps and abs(y0 - y1) < eps:
        return False
    edges = ((a, c, b, c), (b, c, b, d), (b, d, a, d), (a, d, a, c))
    for ex0, ey0, ex1, ey1 in edges:
        if _segments_intersect_proper_2d(x0, y0, x1, y1, ex0, ey0, ex1, ey1, eps=eps):
            return True
    return False


def _lift_point_collision_free(
    p: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
    pad: float,
    *,
    z_step: float = 0.025,
    max_lift_m: float = 2.5,
) -> np.ndarray:
    q = np.asarray(p, dtype=np.float64).reshape(3).copy()
    if _point_collision_free(q, aabbs, radius, pad):
        return q
    steps = max(1, int(max_lift_m / max(z_step, 1e-6)))
    for _ in range(steps):
        q[2] += z_step
        if _point_collision_free(q, aabbs, radius, pad):
            return q
    return q


def _enforce_collision_free_polyline(
    raw_pts: list[np.ndarray],
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
    *,
    pad: float = DEFAULT_COLLISION_PAD_M,
    z_step: float = 0.025,
    max_lift_m: float = 2.5,
) -> np.ndarray:

    if not raw_pts:
        return np.zeros((0, 3), dtype=np.float64)
    if not aabbs:
        return np.stack(raw_pts, axis=0)

    out: list[np.ndarray] = []
    clearance_z = radius + pad
    for raw in raw_pts:
        q = _lift_point_collision_free(
            raw, aabbs, radius, pad, z_step=z_step, max_lift_m=max_lift_m
        )
        if out and not _segment_collision_free(out[-1], q, aabbs, radius, pad):
            z_arc = _min_z_clearance_for_xy_segment(out[-1], q, aabbs, xy_pad=pad)
            z_arc = max(z_arc + clearance_z, float(out[-1][2]), float(q[2]))
            climb_from = out[-1].copy()
            climb_from[2] = z_arc
            climb_to = q.copy()
            climb_to[2] = z_arc
            _append_unique(out, climb_from)
            if not _segment_collision_free(out[-1], climb_to, aabbs, radius, pad):
                q[2] = max(float(q[2]), z_arc)
            else:
                _append_unique(out, climb_to)
                continue
        _append_unique(out, q)
    return np.stack(out, axis=0)


def _through_direction(
    entry: np.ndarray,
    center: np.ndarray,
    exit_hint: np.ndarray | None,
) -> np.ndarray:

    e = np.asarray(entry, dtype=np.float64).reshape(3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    if exit_hint is not None:
        x = np.asarray(exit_hint, dtype=np.float64).reshape(3)
        n = _normalize(x - e)
        if float(np.linalg.norm(n)) >= 1e-9:
            return n
    n = _normalize(c - e)
    if float(np.linalg.norm(n)) >= 1e-9:
        return n
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _build_pass_through_segment(
    entry: np.ndarray,
    center: np.ndarray,
    exit_hint: np.ndarray | None,
    *,
    aabb: tuple | list | None,
    zone_radius: float,
    drone_r: float,
    zone_name: str = "",
    placed_cubes: list[dict] | None = None,
    collision_pad: float = DEFAULT_COLLISION_PAD_M,
) -> np.ndarray:
    portal = _resolve_portal_for_zone(zone_name, center, placed_cubes)
    e = np.asarray(entry, dtype=np.float64).reshape(3)
    solid_over = portal is not None and portal_is_solid_cuboid(portal)

    if portal is not None:
        n = _portal_pass_through_direction(portal, e)
        approach_d, exit_d = _portal_pass_offsets(portal, drone_r)
        if solid_over:
            c = portal_solid_pass_over_crossing(
                portal, drone_r=drone_r, pad=collision_pad
            )
        else:
            c = portal_opening_center_world(portal)
    else:
        c = np.asarray(center, dtype=np.float64).reshape(3)
        n = _through_direction(entry, center, exit_hint)
        approach_d = max(0.15, float(zone_radius) * 0.55)
        exit_d = max(0.15, float(zone_radius) * 0.55)

    raw: list[np.ndarray] = []
    _append_unique(raw, e)
    approach = c - n * approach_d
    if solid_over:
        approach[2] = max(float(approach[2]), float(c[2]))
    _append_unique(raw, approach)
    _append_unique(raw, c.copy())
    exit_pt = c + n * exit_d
    if solid_over:
        exit_pt[2] = max(float(exit_pt[2]), float(c[2]))
    _append_unique(raw, exit_pt)
    if exit_hint is not None:
        xh = np.asarray(exit_hint, dtype=np.float64).reshape(3)
        if solid_over:
            xh = xh.copy()
            xh[2] = max(float(xh[2]), float(c[2]))
        _append_unique(raw, xh)

    obstacles = _obstacle_aabbs_from_placed(
        placed_cubes,
        focus_portal=portal,
    )
    seg = _enforce_collision_free_polyline(
        raw, obstacles, drone_r, pad=collision_pad
    )
    if solid_over:
        return _finalize_solid_pass_over_segment(
            seg,
            portal,
            e,
            exit_hint,
            obstacles,
            placed_cubes,
            drone_r,
            collision_pad,
        )
    if portal is not None:
        return _finalize_pass_through_segment(
            seg,
            portal,
            e,
            exit_hint,
            obstacles,
            placed_cubes,
            drone_r,
            collision_pad,
        )
    return seg


def _portal_local_coords(p: np.ndarray, portal: dict) -> tuple[float, float, float]:
    return portal_local_coords(p, portal)


def _point_in_portal_opening(p: np.ndarray, portal: dict, *, margin: float = 0.0) -> bool:
    return point_in_portal_opening(p, portal, margin=margin)


def _insert_mandatory_waypoint(
    seg: np.ndarray,
    waypoint: np.ndarray,
    *,
    dedupe_eps: float = 0.025,
) -> np.ndarray:
    pts = np.asarray(seg, dtype=np.float64).reshape(-1, 3)
    w = np.asarray(waypoint, dtype=np.float64).reshape(3)
    if pts.shape[0] == 0:
        return w.reshape(1, 3)
    dists = np.linalg.norm(pts - w.reshape(1, 3), axis=1)
    idx = int(np.argmin(dists))
    if float(dists[idx]) <= float(dedupe_eps):
        out = pts.copy()
        out[idx] = w
        return out
    insert_at = idx + 1 if idx + 1 <= pts.shape[0] else idx
    return np.insert(pts, insert_at, w.reshape(1, 3), axis=0)


def _snap_pass_through_polyline_to_axis(
    seg: np.ndarray,
    portal: dict,
    entry: np.ndarray,
    *,
    axis_snap_m: float = 0.12,
) -> np.ndarray:

    c = portal_opening_center_world(portal)
    n = _portal_pass_through_direction(portal, entry)
    out = np.asarray(seg, dtype=np.float64).reshape(-1, 3).copy()
    for i, p in enumerate(out):
        if float(np.linalg.norm(p - c)) <= axis_snap_m or _point_in_portal_opening(
            p, portal, margin=0.03
        ):
            out[i] = project_point_to_portal_axis(p, portal, entry)
    return out


def _finalize_solid_pass_over_segment(
    seg: np.ndarray,
    portal: dict,
    entry: np.ndarray,
    exit_hint: np.ndarray | None,
    obstacles: list[tuple[np.ndarray, np.ndarray]],
    placed_cubes: list[dict] | None,
    drone_r: float,
    collision_pad: float,
) -> np.ndarray:

    c_geo = portal_opening_center_world(portal)
    c_over = portal_solid_pass_over_crossing(
        portal, drone_r=drone_r, pad=collision_pad
    )
    min_z = float(c_over[2])
    n = _portal_pass_through_direction(portal, entry)
    approach_d, exit_d = _portal_pass_offsets(portal, drone_r)
    e = np.asarray(entry, dtype=np.float64).reshape(3)

    def _elevated_axis_point(base: np.ndarray) -> np.ndarray:
        p = project_point_to_portal_axis(base, portal, entry)
        p[2] = max(float(p[2]), min_z)
        return p

    raw: list[np.ndarray] = [e.copy()]
    _append_unique(raw, _elevated_axis_point(c_geo - n * approach_d))
    _append_unique(raw, c_over.copy())
    _append_unique(raw, _elevated_axis_point(c_geo + n * exit_d))
    if exit_hint is not None:
        xh = np.asarray(exit_hint, dtype=np.float64).reshape(3).copy()
        xh[2] = max(float(xh[2]), min_z)
        _append_unique(raw, xh)

    rebuilt = _enforce_collision_free_polyline(
        raw, obstacles, drone_r, pad=collision_pad
    )
    rebuilt = _insert_mandatory_waypoint(rebuilt, c_over)
    out = np.asarray(rebuilt, dtype=np.float64).reshape(-1, 3).copy()
    for i, p in enumerate(out):
        if float(np.linalg.norm(p[:2] - c_geo[:2])) <= 0.35:
            snapped = project_point_to_portal_axis(p, portal, entry)
            out[i, 0] = snapped[0]
            out[i, 1] = snapped[1]
            out[i, 2] = max(float(out[i, 2]), min_z)

    coll = 0
    for i in range(out.shape[0] - 1):
        if not segment_capsule_collision_free(
            out[i], out[i + 1], obstacles, drone_r, collision_pad
        ):
            coll += 1
    if coll > 0:
        print(
            f"[Phase3][pass_through] WARN: {coll} segment(s) still collide after "
            "solid over-top refine; keeping best-effort polyline."
        )
    return out if out.shape[0] >= 1 else seg


def _finalize_pass_through_segment(
    seg: np.ndarray,
    portal: dict,
    entry: np.ndarray,
    exit_hint: np.ndarray | None,
    obstacles: list[tuple[np.ndarray, np.ndarray]],
    placed_cubes: list[dict] | None,
    drone_r: float,
    collision_pad: float,
) -> np.ndarray:

    c = portal_opening_center_world(portal)
    n = _portal_pass_through_direction(portal, entry)
    approach_d, exit_d = _portal_pass_offsets(portal, drone_r)
    e = np.asarray(entry, dtype=np.float64).reshape(3)

    raw: list[np.ndarray] = [e.copy()]
    _append_unique(raw, project_point_to_portal_axis(c - n * approach_d, portal, entry))
    _append_unique(raw, c.copy())
    _append_unique(raw, project_point_to_portal_axis(c + n * exit_d, portal, entry))
    if exit_hint is not None:
        _append_unique(raw, np.asarray(exit_hint, dtype=np.float64).reshape(3))

    rebuilt = _enforce_collision_free_polyline(
        raw, obstacles, drone_r, pad=collision_pad
    )
    rebuilt = _insert_mandatory_waypoint(rebuilt, c)
    rebuilt = _snap_pass_through_polyline_to_axis(rebuilt, portal, entry)

    if not _point_in_portal_opening(c, portal, margin=0.01):
        print("[Phase3][pass_through] WARN: opening center not inside portal hole model.")
    dev = opening_center_deviation_m(rebuilt, portal)
    if dev > 0.04:
        print(
            f"[Phase3][pass_through] WARN: min deviation from opening center "
            f"{dev:.3f}m > 0.04m - pinning center."
        )
        rebuilt = _insert_mandatory_waypoint(rebuilt, c, dedupe_eps=0.05)

    coll = 0
    for i in range(rebuilt.shape[0] - 1):
        if not _pass_through_link_collision_free(
            rebuilt[i],
            rebuilt[i + 1],
            portal,
            placed_cubes,
            drone_r,
            collision_pad,
        ):
            coll += 1
    if coll > 0:
        print(
            f"[Phase3][pass_through] WARN: {coll} segment(s) still collide after refine; "
            "keeping best-effort polyline."
        )
    return rebuilt if rebuilt.shape[0] >= 1 else seg


def _finalize_fly_by_segment(
    seg: np.ndarray,
    portal: dict,
    lateral_dir: np.ndarray,
    clearance: float,
    center: np.ndarray,
    obstacles: list[tuple[np.ndarray, np.ndarray]],
    drone_r: float,
    collision_pad: float,
) -> np.ndarray:

    out_pts: list[np.ndarray] = []
    for p in np.asarray(seg, dtype=np.float64).reshape(-1, 3):
        q = _ensure_fly_by_outside_opening(
            p, portal, lateral_dir, clearance, center
        )
        _append_unique(out_pts, q)
    if not out_pts:
        return seg
    refined = _enforce_collision_free_polyline(
        out_pts, obstacles, drone_r, pad=collision_pad
    )
    for p in refined:
        if _point_in_portal_opening(p, portal, margin=0.015):
            print("[Phase3][fly_by] WARN: waypoint still inside opening after lateral push.")
            break
    return refined


def _portal_fly_by_lateral_dir(
    portal: dict,
    entry: np.ndarray,
    exit_hint: np.ndarray | None,
    center: np.ndarray,
) -> np.ndarray:

    c = np.asarray(center, dtype=np.float64).reshape(3)
    e = np.asarray(entry, dtype=np.float64).reshape(3)
    nx, ny, _ = _portal_opening_axes(portal)
    if exit_hint is not None:
        travel = _normalize(np.asarray(exit_hint, dtype=np.float64).reshape(3) - e)
    else:
        travel = _normalize(_portal_pass_through_direction(portal, e))

    if float(np.linalg.norm(travel)) < 1e-9:
        lat = ny
    elif abs(float(np.dot(travel, nx))) > 0.82:
        lat = ny
    else:
        tin = travel - nx * float(np.dot(travel, nx))
        if float(np.linalg.norm(tin)) < 0.25:
            lat = ny
        else:
            lat = _normalize(np.cross(nx, _normalize(tin)))
            if float(np.linalg.norm(lat)) < 1e-9:
                lat = ny

    side = -1.0 if float(np.dot(e - c, lat)) >= 0.0 else 1.0
    return _normalize(lat) * side


def _portal_fly_by_lateral_clearance(portal: dict, drone_r: float, pad: float) -> float:
    L = float(portal.get("side", 0.392))
    t = float(portal.get("thickness", 0.026))
    return 0.5 * L + t + float(drone_r) + float(pad) + 0.05


def _ensure_fly_by_outside_opening(
    p: np.ndarray,
    portal: dict,
    lateral_dir: np.ndarray,
    clearance: float,
    center: np.ndarray,
) -> np.ndarray:
    q = np.asarray(p, dtype=np.float64).reshape(3).copy()
    if not _point_in_portal_opening(q, portal, margin=0.02):
        return q
    c = np.asarray(center, dtype=np.float64).reshape(3)
    lat = _normalize(lateral_dir)
    q = c + lat * float(clearance)
    if _point_in_portal_opening(q, portal, margin=0.02):
        _, _, nz = _portal_opening_axes(portal)
        q = q + nz * float(clearance) * 0.55
    return q


def _build_fly_by_segment(
    entry: np.ndarray,
    center: np.ndarray,
    exit_hint: np.ndarray | None,
    *,
    aabb: tuple | list | None,
    zone_radius: float,
    drone_r: float,
    zone_name: str = "",
    placed_cubes: list[dict] | None = None,
    collision_pad: float = DEFAULT_COLLISION_PAD_M,
) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    e = np.asarray(entry, dtype=np.float64).reshape(3)
    x_hint = (
        np.asarray(exit_hint, dtype=np.float64).reshape(3)
        if exit_hint is not None
        else e + np.array([0.3, 0.0, 0.0], dtype=np.float64)
    )

    portal = _resolve_portal_for_zone(zone_name, center, placed_cubes)
    if portal is not None:
        nx_dir = _portal_pass_through_direction(portal, e)
        lat_dir = _portal_fly_by_lateral_dir(portal, e, exit_hint, c)
        clearance = _portal_fly_by_lateral_clearance(portal, drone_r, collision_pad)
        approach_d, exit_d = _portal_pass_offsets(portal, drone_r)
        pts: list[np.ndarray] = []
        _append_unique(pts, e.copy())
        for along, lat_scale in (
            (-approach_d, 0.35),
            (-approach_d * 0.55, 0.72),
            (0.0, 1.0),
            (exit_d * 0.55, 0.72),
            (exit_d, 0.35),
        ):
            p = c + nx_dir * float(along) + lat_dir * (clearance * float(lat_scale))
            p = _ensure_fly_by_outside_opening(p, portal, lat_dir, clearance, c)
            _append_unique(pts, p)
        if exit_hint is not None:
            _append_unique(pts, np.asarray(exit_hint, dtype=np.float64).reshape(3))
        obstacles = _obstacle_aabbs_from_placed(placed_cubes)
        seg = _enforce_collision_free_polyline(pts, obstacles, drone_r, pad=collision_pad)
        return _finalize_fly_by_segment(
            seg, portal, lat_dir, clearance, c, obstacles, drone_r, collision_pad
        )

    travel = _normalize(x_hint - e)
    if float(np.linalg.norm(travel)) < 1e-9:
        travel = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    perp = np.array([-travel[1], travel[0], 0.0], dtype=np.float64)
    perp = _normalize(perp)
    if float(np.linalg.norm(perp)) < 1e-9:
        perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    clearance = _safe_clearance_xy(aabb, zone_radius, drone_r)
    side = 1.0 if float(np.dot(e - c, perp)) <= 0.0 else -1.0
    lat = perp * side
    z = max(float(e[2]), _aabb_top_z(aabb, c[2]) + float(drone_r) + 0.03)
    span = max(
        float(np.linalg.norm(x_hint - e)),
        clearance * 1.6,
        float(zone_radius) * 1.2,
    )
    approach_d = max(0.15, span * 0.45)
    exit_d = approach_d

    pts: list[np.ndarray] = []
    _append_unique(pts, e.copy())
    for along, lat_scale in (
        (-approach_d, 0.35),
        (-approach_d * 0.55, 0.72),
        (0.0, 1.0),
        (exit_d * 0.55, 0.72),
        (exit_d, 0.35),
    ):
        p = c + travel * float(along) + lat * (clearance * float(lat_scale))
        p[2] = max(float(p[2]), z)
        _append_unique(pts, p)
    if exit_hint is not None:
        xh = np.asarray(exit_hint, dtype=np.float64).reshape(3).copy()
        xh[2] = max(float(xh[2]), z)
        _append_unique(pts, xh)
    obstacles = _obstacle_aabbs_from_placed(placed_cubes)
    return _enforce_collision_free_polyline(pts, obstacles, drone_r, pad=collision_pad)


def _build_orbit_segment(
    center: np.ndarray,
    *,
    aabb: tuple | list | None,
    zone_radius: float,
    drone_r: float,
    laps: int = 1,
    turn_sign: int = 1,
    horizontal_above: bool = True,
    n_points_per_lap: int = 36,
    placed_cubes: list[dict] | None = None,
    zone_name: str = "",
    collision_pad: float = DEFAULT_COLLISION_PAD_M,
) -> np.ndarray:

    c = np.asarray(center, dtype=np.float64).reshape(3)
    sign = -1.0 if int(turn_sign) < 0 else 1.0
    portal = _resolve_portal_for_zone(zone_name, center, placed_cubes)
    total = max(8, int(n_points_per_lap) * max(1, int(laps)))
    pts: list[np.ndarray] = []

    if portal is not None and not horizontal_above:
        nx, ny, nz = _portal_opening_axes(portal)
        orbit_r = portal_opening_half_diagonal(portal) + float(drone_r) + float(collision_pad) + 0.08
        for i in range(total + 1):
            ang = sign * 2.0 * math.pi * float(i) / float(max(1, n_points_per_lap))
            p = c + ny * (orbit_r * math.cos(ang)) + nz * (orbit_r * math.sin(ang))
            p = p + nx * (float(drone_r) + float(collision_pad) + 0.06)
            _append_unique(pts, p)
    elif portal is not None:
        hover = portal_hover_world_position(
            portal,
            zone_radius=float(zone_radius),
            drone_r=float(drone_r),
        )
        orbit_r = portal_opening_half_diagonal(portal) + float(drone_r) + float(collision_pad) + 0.08
        for i in range(total + 1):
            ang = sign * 2.0 * math.pi * float(i) / float(max(1, n_points_per_lap))
            p = np.array(
                [
                    float(hover[0]) + orbit_r * math.cos(ang),
                    float(hover[1]) + orbit_r * math.sin(ang),
                    float(hover[2]),
                ],
                dtype=np.float64,
            )
            _append_unique(pts, p)
    else:
        orbit_r = _safe_clearance_xy(aabb, zone_radius, drone_r) * 0.85
        z = _aabb_top_z(aabb, c[2]) + max(float(zone_radius), float(drone_r) * 2.0) + 0.03
        for i in range(total + 1):
            ang = sign * 2.0 * math.pi * float(i) / float(max(1, n_points_per_lap))
            p = np.array(
                [c[0] + orbit_r * math.cos(ang), c[1] + orbit_r * math.sin(ang), z],
                dtype=np.float64,
            )
            _append_unique(pts, p)
    obstacles = _obstacle_aabbs_from_placed(placed_cubes)
    return _enforce_collision_free_polyline(pts, obstacles, drone_r, pad=collision_pad)


def _collision_impact_point(
    *,
    entry: np.ndarray,
    center: np.ndarray,
    portal: dict | None,
    aabb: tuple | list | None,
    zone_radius: float,
) -> np.ndarray:

    e = np.asarray(entry, dtype=np.float64).reshape(3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    if portal is not None:
        opening = portal_opening_center_world(portal)
        approach = opening - e
        horiz = float(np.linalg.norm(approach[:2]))
        vert = abs(float(approach[2]))
        if vert > horiz * 0.85:
            return opening.copy()
        return portal_frame_surface_point(portal, e, drone_r=0.0, pad=0.0)
    n = _normalize(c - e)
    if float(np.linalg.norm(n)) < 1e-9:
        n = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    lo, hi = _aabb_lo_hi(aabb)
    half = np.maximum(np.abs(hi - c), np.abs(lo - c))
    extent = float(max(half[0], half[1], half[2], zone_radius * 0.35))
    reach = min(extent, float(np.linalg.norm(c - e)) * 0.9)
    return c - n * reach


def _build_direct_ram_polyline(
    entry: np.ndarray,
    impact: np.ndarray,
    *,
    step_m: float = 0.07,
    min_points: int = 3,
) -> np.ndarray:

    e = np.asarray(entry, dtype=np.float64).reshape(3)
    imp = np.asarray(impact, dtype=np.float64).reshape(3)
    dist = float(np.linalg.norm(imp - e))
    if dist < 1e-6:
        return np.stack([e, imp], axis=0)
    n_pts = max(int(min_points), int(math.ceil(dist / max(float(step_m), 1e-6))) + 1)
    pts: list[np.ndarray] = []
    for i in range(n_pts):
        t = float(i) / float(n_pts - 1) if n_pts > 1 else 1.0
        _append_unique(pts, e + t * (imp - e))
    if pts:
        pts[-1] = imp
    return np.stack(pts, axis=0)


def _build_collision_segment(
    entry: np.ndarray,
    center: np.ndarray,
    *,
    aabb: tuple | list | None,
    zone_radius: float,
    zone_name: str = "",
    placed_cubes: list[dict] | None = None,
    drone_r: float = 0.07,
    collision_pad: float = DEFAULT_COLLISION_PAD_M,
) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    e = np.asarray(entry, dtype=np.float64).reshape(3)
    portal = _resolve_portal_for_zone(zone_name, center, placed_cubes)
    impact = _collision_impact_point(
        entry=e,
        center=c,
        portal=portal,
        aabb=aabb,
        zone_radius=float(zone_radius),
    )
    return _build_direct_ram_polyline(
        e,
        impact,
        step_m=max(0.06, float(drone_r)),
        min_points=3,
    )


def _build_hover_segment(
    center: np.ndarray,
    *,
    aabb: tuple | list | None,
    zone_radius: float,
    drone_r: float,
    hold_points: int = 6,
    placed_cubes: list[dict] | None = None,
    zone_name: str = "",
    collision_pad: float = DEFAULT_COLLISION_PAD_M,
    workspace_z_cap_world: float | None = None,
) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    portal = _resolve_portal_for_zone(zone_name, center, placed_cubes)
    if portal is not None:
        hover = portal_hover_world_position(
            portal,
            zone_radius=float(zone_radius),
            drone_r=float(drone_r),
            workspace_z_cap_world=workspace_z_cap_world,
        )
    else:
        z = _aabb_top_z(aabb, c[2]) + 2.0 * float(zone_radius)
        z = max(z, c[2] + float(drone_r) + 0.05)
        if workspace_z_cap_world is not None:
            z = min(float(z), float(workspace_z_cap_world))
        hover = np.array([c[0], c[1], z], dtype=np.float64)
    pts: list[np.ndarray] = []
    for _ in range(max(2, int(hold_points))):
        _append_unique(pts, hover)
    obstacles = _obstacle_aabbs_from_placed(placed_cubes)
    return _enforce_collision_free_polyline(pts, obstacles, drone_r, pad=collision_pad)


def build_basic_action_segment(
    action: BasicAction,
    *,
    entry: np.ndarray,
    center: np.ndarray,
    exit_hint: np.ndarray | None,
    aabb: tuple | list | None,
    zone_radius: float,
    drone_r: float,
    orbit_laps: int = 1,
    orbit_clause: str | None = None,
    zone_name: str = "",
    placed_cubes: list[dict] | None = None,
    collision_pad: float = DEFAULT_COLLISION_PAD_M,
) -> np.ndarray:

    common = dict(
        aabb=aabb,
        zone_radius=zone_radius,
        drone_r=drone_r,
        placed_cubes=placed_cubes,
        collision_pad=collision_pad,
    )
    if action == BasicAction.PASS_THROUGH:
        return _build_pass_through_segment(
            entry,
            center,
            exit_hint,
            zone_name=zone_name,
            **common,
        )
    if action == BasicAction.FLY_BY:
        return _build_fly_by_segment(
            entry, center, exit_hint, zone_name=zone_name, **common
        )
    if action == BasicAction.ORBIT:
        if orbit_clause and str(orbit_clause).strip():
            op = _parse_orbit_params(orbit_clause, default_laps=int(orbit_laps))
        else:
            op = OrbitParams(laps=int(orbit_laps), turn_sign=1, horizontal_above=True)
        return _build_orbit_segment(
            center,
            laps=op.laps,
            turn_sign=op.turn_sign,
            horizontal_above=op.horizontal_above,
            zone_name=zone_name,
            **common,
        )
    if action == BasicAction.COLLISION:
        return _build_collision_segment(
            entry,
            center,
            aabb=aabb,
            zone_radius=zone_radius,
            zone_name=zone_name,
            placed_cubes=placed_cubes,
            drone_r=drone_r,
            collision_pad=collision_pad,
        )
    if action == BasicAction.HOVER:
        return _build_hover_segment(center, zone_name=zone_name, **common)
    return np.stack(
        [
            np.asarray(entry, dtype=np.float64).reshape(3),
            np.asarray(center, dtype=np.float64).reshape(3),
        ],
        axis=0,
    )


def _portal_feedback_zone_radius(portal: dict, *, fallback: float = 0.25) -> float:
    bh = portal.get("bounds_half")
    if bh is not None:
        h = np.asarray(bh, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(h)) * 0.55 + 0.08
    return float(fallback)


def compute_action_leg_goal_world(
    action: BasicAction,
    portal: dict,
    from_xyz: np.ndarray,
    placed_cubes: list[dict],
    *,
    exit_hint: np.ndarray | None = None,
    orbit_laps: int = 1,
    orbit_clause: str | None = None,
    drone_r: float = 0.07,
    collision_pad: float = DEFAULT_COLLISION_PAD_M,
) -> np.ndarray:

    c = np.asarray(portal["pos"], dtype=np.float64).reshape(3)
    e = np.asarray(from_xyz, dtype=np.float64).reshape(3)
    lo, hi = _placed_object_aabb_lo_hi_world(portal)
    zone_radius = _portal_feedback_zone_radius(portal)
    bid = portal.get("portal_label")
    zone_name = f"billboard_id={bid}" if bid is not None else "rect_frame portal"

    xh = (
        np.asarray(exit_hint, dtype=np.float64).reshape(3)
        if exit_hint is not None
        else c + _portal_pass_through_direction(portal, e) * 0.4
    )
    seg = build_basic_action_segment(
        action,
        entry=e,
        center=c,
        exit_hint=xh,
        aabb=(lo, hi),
        zone_radius=zone_radius,
        drone_r=float(drone_r),
        orbit_laps=int(orbit_laps),
        orbit_clause=orbit_clause,
        zone_name=zone_name,
        placed_cubes=placed_cubes,
        collision_pad=collision_pad,
    )
    if seg.shape[0] == 0:
        return c.copy()
    return np.asarray(seg[-1], dtype=np.float64).reshape(3)


def splice_trajectory_segment(
    trajectory: np.ndarray,
    i0: int,
    i1: int,
    replacement: np.ndarray,
    *,
    dedupe_eps: float = 1e-4,
) -> np.ndarray:

    pts = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    rep = np.asarray(replacement, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return rep.copy()
    i0 = int(max(0, min(i0, pts.shape[0] - 1)))
    i1 = int(max(i0, min(i1, pts.shape[0] - 1)))
    prefix = pts[:i0]
    suffix = pts[i1 + 1 :]
    chunks: list[np.ndarray] = []
    if prefix.shape[0]:
        chunks.append(prefix)
    if rep.shape[0]:
        if chunks and float(np.linalg.norm(chunks[-1][-1] - rep[0])) < dedupe_eps:
            chunks.append(rep[1:])
        else:
            chunks.append(rep)
    if suffix.shape[0]:
        if chunks and float(np.linalg.norm(chunks[-1][-1] - suffix[0])) < dedupe_eps:
            chunks.append(suffix[1:])
        else:
            chunks.append(suffix)
    if not chunks:
        return pts.copy()
    return np.concatenate(chunks, axis=0)


def refine_trajectory_at_zone_contact(
    trajectory: np.ndarray,
    *,
    zone_name: str,
    zone_center: np.ndarray,
    zone_radius: float,
    is_target: bool,
    aabb: tuple | list | None,
    contact_pos: np.ndarray,
    contact_traj_idx: int,
    mission_cmd: str | None,
    placed_cubes: list[dict] | None,
    drone_body_half: tuple[float, float, float],
    action_cache: dict[str, tuple[BasicAction, int]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:

    drone_r = _drone_radius(drone_body_half)
    action, orbit_laps = resolve_basic_action_for_zone(
        mission_cmd,
        zone_name=zone_name,
        zone_center=zone_center,
        is_target=is_target,
        placed_cubes=placed_cubes,
        action_cache=action_cache,
    )
    i0, i1 = _find_trajectory_zone_span(
        trajectory,
        zone_center,
        zone_radius,
        drone_r,
        contact_idx=contact_traj_idx,
    )
    entry = np.asarray(trajectory[i0], dtype=np.float64).reshape(3)
    exit_hint = (
        np.asarray(trajectory[i1 + 1], dtype=np.float64).reshape(3)
        if i1 + 1 < np.asarray(trajectory).shape[0]
        else None
    )
    orbit_clause = resolve_mission_clause_for_zone(
        mission_cmd,
        zone_name=str(zone_name),
        zone_center=zone_center,
        placed_cubes=placed_cubes,
    )
    segment = build_basic_action_segment(
        action,
        entry=entry,
        center=zone_center,
        exit_hint=exit_hint,
        aabb=aabb,
        zone_radius=float(zone_radius),
        drone_r=drone_r,
        orbit_laps=orbit_laps,
        orbit_clause=orbit_clause,
        zone_name=zone_name,
        placed_cubes=placed_cubes,
    )
    refined = splice_trajectory_segment(trajectory, i0, i1, segment)
    meta = {
        "zone_name": zone_name,
        "action": action.value,
        "orbit_laps": int(orbit_laps),
        "span_i0": int(i0),
        "span_i1": int(i1),
        "segment_points": int(segment.shape[0]),
        "refined_points": int(refined.shape[0]),
        "segment_xyz": segment.copy(),
        "coarse_span_xyz": np.asarray(trajectory[i0 : i1 + 1], dtype=np.float64).copy(),
    }
    return refined, meta


def nearest_trajectory_index(trajectory: np.ndarray, pos: np.ndarray) -> int:
    pts = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0
    d = np.linalg.norm(pts - np.asarray(pos, dtype=np.float64).reshape(1, 3), axis=1)
    return int(np.argmin(d))


def validate_clause_action_consistency(clause: str, action: BasicAction) -> bool:

    parsed = _action_from_clause(clause)
    if parsed is None:
        return True
    return parsed == action


def _placed_cubes_excluding_centers(
    placed_cubes: list[dict] | None,
    exclude_object_centers: list[Any] | None,
    *,
    tol_xy_m: float = 0.3,
) -> list[dict]:

    if not placed_cubes:
        return []
    if not exclude_object_centers:
        return list(placed_cubes)
    centers = [np.asarray(c, dtype=np.float64).reshape(3)[:2] for c in exclude_object_centers]
    kept: list[dict] = []
    for c in placed_cubes:
        pos = np.asarray(c.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)[:2]
        if any(float(np.linalg.norm(pos - ctr)) <= float(tol_xy_m) for ctr in centers):
            continue
        kept.append(c)
    return kept


def _count_collision_segments(
    pts: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    *,
    drone_r: float,
    collision_pad: float,
) -> int:
    coll = 0
    for i in range(pts.shape[0] - 1):
        if not segment_capsule_collision_free(
            pts[i], pts[i + 1], aabbs, float(drone_r), float(collision_pad)
        ):
            coll += 1
    return coll


def compute_path_quality_metrics(
    trajectory: np.ndarray,
    placed_cubes: list[dict] | None,
    *,
    drone_r: float,
    collision_pad: float = DEFAULT_COLLISION_PAD_M,
    exclude_object_centers: list[Any] | None = None,
) -> dict[str, Any]:

    pts = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        out0: dict[str, Any] = {
            "waypoint_count": int(pts.shape[0]),
            "collision_segment_count": 0,
            "max_turn_deg": 0.0,
            "path_length_m": 0.0,
        }
        if exclude_object_centers is not None:
            out0["collision_segment_count_avoidable"] = 0
        return out0
    aabbs = _obstacle_aabbs_from_placed(placed_cubes)
    coll = _count_collision_segments(pts, aabbs, drone_r=drone_r, collision_pad=collision_pad)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    path_len = float(np.sum(seg_len))
    max_turn = 0.0
    for i in range(1, pts.shape[0] - 1):
        v0 = pts[i] - pts[i - 1]
        v1 = pts[i + 1] - pts[i]
        n0 = float(np.linalg.norm(v0))
        n1 = float(np.linalg.norm(v1))
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        c = float(np.clip(np.dot(v0 / n0, v1 / n1), -1.0, 1.0))
        max_turn = max(max_turn, math.degrees(math.acos(c)))
    out: dict[str, Any] = {
        "waypoint_count": int(pts.shape[0]),
        "collision_segment_count": int(coll),
        "max_turn_deg": float(max_turn),
        "path_length_m": float(path_len),
        "collision_pad_m": float(collision_pad),
        "drone_radius_m": float(drone_r),
    }
    if exclude_object_centers is not None:
        avoid_cubes = _placed_cubes_excluding_centers(placed_cubes, exclude_object_centers)
        avoid_aabbs = _obstacle_aabbs_from_placed(avoid_cubes)
        out["collision_segment_count_avoidable"] = int(
            _count_collision_segments(pts, avoid_aabbs, drone_r=drone_r, collision_pad=collision_pad)
        )
    return out
