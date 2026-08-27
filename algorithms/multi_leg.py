from __future__ import annotations

import datetime
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .instruction_parse import (
    extract_ordered_mission_billboard_ids,
    find_rect_portal_by_billboard_id,
    leg_sub_instruction_for_billboard_id,
    mission_clause_for_billboard,
    split_mission_clauses,
)
from .phase1 import run_single_leg_phase1_and_phase2
from .phase3 import Phase3FeedbackZones
from .phase3_actions import (
    BasicAction,
    _find_trajectory_zone_span,
    _placed_object_aabb_lo_hi_world,
    _portal_feedback_zone_radius,
    _zone_billboard_id,
    build_basic_action_segment,
    compute_action_leg_goal_world,
    nearest_trajectory_index,
    splice_trajectory_segment,
)
from .phase_recording import (
    PHASE2_DIR,
    STEREO45_PNG,
    TOPDOWN_PNG,
    resolve_experiment_base_folder,
    resolve_phase_folder,
)

@dataclass
class TraversalLegSpec:
    leg_index: int
    billboard_id: int
    portal: dict
    sub_instruction: str
    goal_xyz_world: np.ndarray
    start_xyz_world: np.ndarray

@dataclass
class MissionActionLegSpec:


    leg_index: int
    billboard_id: int
    portal: dict
    clause: str
    sub_instruction: str
    action: BasicAction
    orbit_laps: int
    goal_xyz_world: np.ndarray
    start_xyz_world: np.ndarray

def _ensure_multi_action_portal_feedback_zones(
    p: Any,
    registry: Phase3FeedbackZones,
    leg_specs: list[MissionActionLegSpec],
) -> None:

    from .phase3 import (
        PHASE3_RGBA_IDLE,
        Phase3FeedbackZone,
        apply_feedback_sphere_visual,
        create_feedback_sphere_visual_shape,
    )
    from .phase3_actions import entry_key

    existing = registry.zone_keys()
    for spec in leg_specs:
        portal = spec.portal
        lo, hi = _portal_aabb_for_spec(portal)
        center = (lo + hi) * 0.5
        col = str(portal.get("color_name", portal.get("color", "?")))
        sh = str(portal.get("shape", "rect_frame")).lower()
        kind = "gate" if sh in ("rect_frame", "frame", "portal") else sh
        name = f"{col} {kind}"
        key = entry_key(name, center)
        if key in existing:
            continue
        existing.add(key)
        radius = float(_portal_feedback_zone_radius(portal))
        vis = create_feedback_sphere_visual_shape(p, radius, PHASE3_RGBA_IDLE)
        body_uid = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=center.tolist(),
        )
        zone = Phase3FeedbackZone(
            name=name,
            center=center,
            radius=radius,
            is_target=True,
            body_uid=int(body_uid),
            entered=False,
            aabb=(lo, hi),
        )
        apply_feedback_sphere_visual(p, zone)
        registry.zones.append(zone)
        print(
            f"[Multi-action] Pre-created feedback sphere: {name} "
            f"(billboard_id={spec.billboard_id}, action={spec.action.value})"
        )

def _find_feedback_zone_for_portal(
    registry: Phase3FeedbackZones,
    portal: dict,
) -> Any | None:
    bid = portal.get("portal_label")
    c = np.asarray(portal["pos"], dtype=np.float64).reshape(3)
    if bid is not None:
        for zone in registry.zones:
            zbid = _zone_billboard_id(zone.name, zone.center, None)
            if zbid is not None and int(zbid) == int(bid):
                return zone
    best: tuple[float, Any] | None = None
    for zone in registry.zones:
        d = float(np.linalg.norm(np.asarray(zone.center, dtype=np.float64).reshape(3) - c))
        if best is None or d < best[0]:
            best = (d, zone)
    if best is not None and best[0] < 0.35:
        return best[1]
    return None

def _portal_aabb_for_spec(portal: dict) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _placed_object_aabb_lo_hi_world(portal)
    return lo, hi

def _geometry_fallback_leg_trajectory(
    spec: MissionActionLegSpec,
    placed_cubes: list[dict],
    *,
    drone_r: float,
) -> np.ndarray:

    from .portal_geometry import portal_opening_center_world

    start = np.asarray(spec.start_xyz_world, dtype=np.float64).reshape(3)
    goal = np.asarray(spec.goal_xyz_world, dtype=np.float64).reshape(3)
    portal_c = portal_opening_center_world(spec.portal)
    lo, hi = _portal_aabb_for_spec(spec.portal)
    zone_radius = float(_portal_feedback_zone_radius(spec.portal))
    zone_name = (
        f"billboard_id={spec.billboard_id} "
        f"{spec.portal.get('color_name', spec.portal.get('color', '?'))} gate"
    )
    next_spec_goal = goal
    segment = build_basic_action_segment(
        spec.action,
        entry=start,
        center=portal_c,
        exit_hint=next_spec_goal,
        aabb=(lo, hi),
        zone_radius=zone_radius,
        drone_r=float(drone_r),
        orbit_laps=int(spec.orbit_laps),
        orbit_clause=spec.clause,
        zone_name=zone_name,
        placed_cubes=placed_cubes,
    )
    if segment.shape[0] >= 2:
        return anchor_leg_trajectory_endpoints(segment, start, goal).astype(np.float32)
    return anchor_leg_trajectory_endpoints(
        np.stack([start, goal], axis=0), start, goal
    ).astype(np.float32)

def _refine_leg_trajectory_for_action(
    p: Any,
    traj: np.ndarray,
    spec: MissionActionLegSpec,
    registry: Phase3FeedbackZones,
    *,
    mission_cmd: str | None,
    placed_cubes: list[dict],
    action_cache: dict[str, tuple[BasicAction, int]] | None,
) -> tuple[np.ndarray, dict[str, Any] | None]:

    pts = np.asarray(traj, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 1:
        return traj, None

    portal_c = np.asarray(spec.portal["pos"], dtype=np.float64).reshape(3)
    zone = _find_feedback_zone_for_portal(registry, spec.portal)
    zone_center = portal_c if zone is None else np.asarray(zone.center, dtype=np.float64).reshape(3)
    lo, hi = _portal_aabb_for_spec(spec.portal)
    zone_radius = float(
        zone.radius if zone is not None else _portal_feedback_zone_radius(spec.portal)
    )
    aabb = zone.aabb if zone is not None and zone.aabb is not None else (lo, hi)
    zone_name = (
        zone.name
        if zone is not None
        else f"billboard_id={spec.billboard_id} {spec.portal.get('color', '?')} gate"
    )

    contact_idx = nearest_trajectory_index(pts, zone_center)
    contact_pos = pts[contact_idx]

    if zone is not None and not zone.entered:
        registry.apply_feedback_color(
            p,
            zone,
            contact_pos=contact_pos,
            mission_cmd=spec.sub_instruction or mission_cmd,
            placed_cubes=placed_cubes,
        )
    elif zone is None:
        print(
            f"[Multi-action] Leg {spec.leg_index} billboard_id={spec.billboard_id}: "
            "feedback sphere not created; still refining trajectory for assigned action."
        )

    drone_r = float(np.linalg.norm(np.asarray(registry.drone_body_half, dtype=np.float64).reshape(3)))
    i0, i1 = _find_trajectory_zone_span(
        pts,
        zone_center,
        zone_radius,
        drone_r,
        contact_idx=int(contact_idx),
    )
    entry = pts[i0]
    exit_hint = pts[i1 + 1] if i1 + 1 < pts.shape[0] else None
    segment = build_basic_action_segment(
        spec.action,
        entry=entry,
        center=zone_center,
        exit_hint=exit_hint,
        aabb=aabb,
        zone_radius=zone_radius,
        drone_r=drone_r,
        orbit_laps=int(spec.orbit_laps),
        orbit_clause=spec.clause,
        zone_name=zone_name,
        placed_cubes=placed_cubes,
    )
    refined = splice_trajectory_segment(pts, i0, i1, segment)
    meta = {
        "zone_name": zone_name,
        "action": spec.action.value,
        "orbit_laps": int(spec.orbit_laps),
        "span_i0": int(i0),
        "span_i1": int(i1),
        "segment_points": int(segment.shape[0]),
        "refined_points": int(refined.shape[0]),
        "leg_index": spec.leg_index,
        "billboard_id": spec.billboard_id,
        "segment_xyz": segment.copy(),
        "coarse_span_xyz": pts[i0 : i1 + 1].copy(),
    }
    print(
        f"[Multi-action][refine] leg {spec.leg_index} billboard_id={spec.billboard_id} "
        f"action={spec.action.value}, waypoints {pts.shape[0]} → {refined.shape[0]}"
    )
    return refined, meta

def parse_mission_action_legs(
    mission_cmd: str,
    placed_cubes: list[dict],
    start_xyz_world: np.ndarray,
    *,
    action_cache: dict[str, tuple[BasicAction, int]] | None = None,
    drone_r: float = 0.07,
) -> list[MissionActionLegSpec]:

    ins = str(mission_cmd or "").strip()
    if not ins:
        return []

    from .phase3_actions import resolve_basic_action_for_zone, validate_clause_action_consistency

    clauses = split_mission_clauses(ins)
    bids = extract_ordered_mission_billboard_ids(ins)
    raw: list[tuple[int, dict, str, BasicAction, int]] = []
    for bid in bids:
        merged = mission_clause_for_billboard(clauses, int(bid))
        if not merged:
            print(
                f"[Multi-action] WARN: billboard_id={bid} has no matching clause fragment; skipping."
            )
            continue
        portal = find_rect_portal_by_billboard_id(placed_cubes, int(bid))
        if portal is None:
            print(f"[Multi-action] WARN: billboard_id={bid} not found; skipping clause.")
            continue
        lo, hi = _portal_aabb_for_spec(portal)
        c = (lo + hi) * 0.5
        col = str(portal.get("color_name", portal.get("color", "?")))
        sh = str(portal.get("shape", "rect_frame")).lower()
        kind = "gate" if sh in ("rect_frame", "frame", "portal") else sh
        name = f"{col} {kind}"
        action, laps = resolve_basic_action_for_zone(
            ins,
            zone_name=name,
            zone_center=c,
            is_target=True,
            placed_cubes=placed_cubes,
            action_cache=action_cache,
        )
        if not validate_clause_action_consistency(merged, action):
            print(
                f"[Multi-action] WARN: clause action mismatch billboard_id={bid} "
                f"clause={merged!r} assigned={action.value}"
            )
        raw.append((int(bid), portal, merged, action, int(laps)))

    if not raw:
        return []

    legs: list[MissionActionLegSpec] = []
    cur_start = np.asarray(start_xyz_world, dtype=np.float64).reshape(3)
    for i, (bid, portal, clause, action, laps) in enumerate(raw):
        cur_c = np.asarray(portal["pos"], dtype=np.float64).reshape(3)
        if i + 1 < len(raw):
            next_c = np.asarray(raw[i + 1][1]["pos"], dtype=np.float64).reshape(3)
            vec = next_c - cur_c
            n = float(np.linalg.norm(vec))
            exit_hint = (
                cur_c + vec * (0.45 / n)
                if n > 1e-9
                else cur_c + np.array([0.45, 0.0, 0.0], dtype=np.float64)
            )
        else:
            exit_hint = None
        goal = compute_action_leg_goal_world(
            action,
            portal,
            cur_start,
            placed_cubes,
            exit_hint=exit_hint,
            orbit_laps=laps,
            orbit_clause=clause,
            drone_r=drone_r,
        )
        legs.append(
            MissionActionLegSpec(
                leg_index=len(legs),
                billboard_id=bid,
                portal=portal,
                clause=clause,
                sub_instruction=clause,
                action=action,
                orbit_laps=laps,
                goal_xyz_world=goal,
                start_xyz_world=cur_start.copy(),
            )
        )
        cur_start = goal.copy()
    return legs

DEFAULT_LEG_BRIDGE_MAX_STEP_M = 0.04

def polyline_yaw_rotations(trajectory: np.ndarray) -> list[np.ndarray]:

    pts = np.asarray(trajectory, dtype=np.float32).reshape(-1, 3)
    eye = np.eye(3, dtype=np.float32)
    if pts.shape[0] == 0:
        return [eye.copy()]
    out: list[np.ndarray] = []
    for i in range(pts.shape[0]):
        if i + 1 < pts.shape[0]:
            d = pts[i + 1] - pts[i]
        elif i > 0:
            d = pts[i] - pts[i - 1]
        else:
            out.append(eye.copy())
            continue
        dn = float(np.linalg.norm(d))
        if dn < 1e-9:
            out.append(eye.copy())
            continue
        yaw = math.atan2(float(d[1]), float(d[0]))
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        out.append(R)
    return out

def _exit_hint_for_portals(portal: dict, next_portal: dict | None) -> np.ndarray | None:
    if next_portal is None:
        return None
    cur_c = np.asarray(portal["pos"], dtype=np.float64).reshape(3)
    next_c = np.asarray(next_portal["pos"], dtype=np.float64).reshape(3)
    vec = next_c - cur_c
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return cur_c + np.array([0.45, 0.0, 0.0], dtype=np.float64)
    return cur_c + vec * (0.45 / n)

def _refresh_leg_spec_endpoints(
    spec: MissionActionLegSpec,
    *,
    start_xyz: np.ndarray,
    next_spec: MissionActionLegSpec | None,
    placed_cubes: list[dict],
    drone_r: float,
) -> None:

    from .phase3_actions import compute_action_leg_goal_world

    spec.start_xyz_world = np.asarray(start_xyz, dtype=np.float64).reshape(3).copy()
    next_portal = next_spec.portal if next_spec is not None else None
    exit_hint = _exit_hint_for_portals(spec.portal, next_portal)
    spec.goal_xyz_world = compute_action_leg_goal_world(
        spec.action,
        spec.portal,
        spec.start_xyz_world,
        placed_cubes,
        exit_hint=exit_hint,
        orbit_laps=int(spec.orbit_laps),
        orbit_clause=spec.clause,
        drone_r=float(drone_r),
    )

def _endpoint_tangent(pts: np.ndarray, *, at_start: bool) -> np.ndarray:

    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 2:
        return np.zeros(3, dtype=np.float64)
    if at_start:
        v = p[1] - p[0]
    else:
        v = p[-1] - p[-2]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return (v / n).astype(np.float64)

def _hermite_bridge_3d(
    p0: np.ndarray,
    t0: np.ndarray,
    p1: np.ndarray,
    t1: np.ndarray,
    max_step_m: float,
    *,
    include_first: bool = False,
    include_last: bool = True,
    tangent_scale: float = 0.35,
) -> np.ndarray:

    a = np.asarray(p0, dtype=np.float64).reshape(3)
    b = np.asarray(p1, dtype=np.float64).reshape(3)
    m0 = np.asarray(t0, dtype=np.float64).reshape(3) * float(tangent_scale)
    m1 = np.asarray(t1, dtype=np.float64).reshape(3) * float(tangent_scale)
    chord = float(np.linalg.norm(b - a))
    if chord < 1e-9:
        if include_first or include_last:
            return a.reshape(1, 3).astype(np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    step = max(1e-6, float(max_step_m))
    n_seg = max(3, int(math.ceil(chord / step)))
    ts = np.linspace(0.0, 1.0, n_seg + 1, dtype=np.float64)
    if not include_first:
        ts = ts[1:]
    if not include_last:
        ts = ts[:-1]
    if ts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    out: list[np.ndarray] = []
    for u in ts:
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        pt = h00 * a + h10 * m0 + h01 * b + h11 * m1
        out.append(pt.astype(np.float64))
    return np.stack(out, axis=0).astype(np.float32)

def _linear_bridge_3d(
    p0: np.ndarray,
    p1: np.ndarray,
    max_step_m: float,
    *,
    include_first: bool = False,
    include_last: bool = True,
) -> np.ndarray:

    a = np.asarray(p0, dtype=np.float64).reshape(3)
    b = np.asarray(p1, dtype=np.float64).reshape(3)
    dist = float(np.linalg.norm(b - a))
    if dist < 1e-9:
        if include_first or include_last:
            return a.reshape(1, 3).astype(np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    step = max(1e-6, float(max_step_m))
    n_seg = max(1, int(math.ceil(dist / step)))
    ts = np.linspace(0.0, 1.0, n_seg + 1, dtype=np.float64)
    if not include_first:
        ts = ts[1:]
    if not include_last:
        ts = ts[:-1]
    if ts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    bridge = np.stack([(1.0 - t) * a + t * b for t in ts], axis=0)
    return bridge.astype(np.float32)

def _merge_trajectory_chunks(
    chunks: list[np.ndarray],
    *,
    dedupe_eps: float = 1e-3,
) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.asarray(chunks[0], dtype=np.float32).reshape(-1, 3)
    for chunk in chunks[1:]:
        c = np.asarray(chunk, dtype=np.float32).reshape(-1, 3)
        if c.shape[0] == 0:
            continue
        if out.shape[0] and float(np.linalg.norm(out[-1] - c[0])) < dedupe_eps:
            out = np.concatenate([out, c[1:]], axis=0) if c.shape[0] > 1 else out
        else:
            out = np.concatenate([out, c], axis=0)
    return out

def densify_trajectory_polyline(
    trajectory: np.ndarray,
    *,
    max_step_m: float = DEFAULT_LEG_BRIDGE_MAX_STEP_M,
    dedupe_eps: float = 1e-3,
) -> np.ndarray:

    pts = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pts.astype(np.float32)

    chunks: list[np.ndarray] = [pts[0:1].astype(np.float32)]
    for i in range(1, pts.shape[0]):
        a = pts[i - 1]
        b = pts[i]
        dist = float(np.linalg.norm(b - a))
        if dist <= float(max_step_m) + 1e-9:
            chunks.append(b.reshape(1, 3).astype(np.float32))
        else:
            bridge = _linear_bridge_3d(
                a,
                b,
                max_step_m,
                include_first=False,
                include_last=True,
            )
            if bridge.shape[0]:
                chunks.append(bridge)
            else:
                chunks.append(b.reshape(1, 3).astype(np.float32))
    return _merge_trajectory_chunks(chunks, dedupe_eps=dedupe_eps)

def smooth_stitch_trajectory_legs(
    legs: list[np.ndarray],
    *,
    dedupe_eps: float = 1e-3,
    bridge_max_step_m: float = DEFAULT_LEG_BRIDGE_MAX_STEP_M,
) -> np.ndarray:

    stitched = stitch_trajectory_legs(
        legs,
        dedupe_eps=dedupe_eps,
        bridge_max_step_m=bridge_max_step_m,
    )
    return densify_trajectory_polyline(
        stitched,
        max_step_m=bridge_max_step_m,
        dedupe_eps=dedupe_eps,
    )

def anchor_leg_trajectory_endpoints(
    trajectory: np.ndarray,
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
    *,
    bridge_max_step_m: float = DEFAULT_LEG_BRIDGE_MAX_STEP_M,
    dedupe_eps: float = 1e-3,
) -> np.ndarray:

    pts = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    start = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    goal = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    if pts.shape[0] == 0:
        return _linear_bridge_3d(
            start,
            goal,
            bridge_max_step_m,
            include_first=True,
            include_last=True,
        )

    chunks: list[np.ndarray] = []
    if float(np.linalg.norm(pts[0] - start)) > dedupe_eps:
        chunks.append(
            _linear_bridge_3d(
                start,
                pts[0],
                bridge_max_step_m,
                include_first=True,
                include_last=False,
            )
        )
    else:
        pts = pts.copy()
        pts[0] = start

    chunks.append(pts.astype(np.float32))

    tail = np.asarray(chunks[-1], dtype=np.float64).reshape(-1, 3)
    if float(np.linalg.norm(tail[-1] - goal)) > dedupe_eps:
        chunks.append(
            _linear_bridge_3d(
                tail[-1],
                goal,
                bridge_max_step_m,
                include_first=False,
                include_last=True,
            )
        )
    else:
        tail = tail.copy()
        tail[-1] = goal
        chunks[-1] = tail.astype(np.float32)

    merged = _merge_trajectory_chunks(chunks, dedupe_eps=dedupe_eps)
    return densify_trajectory_polyline(
        merged,
        max_step_m=bridge_max_step_m,
        dedupe_eps=dedupe_eps,
    )

def stitch_trajectory_legs(
    legs: list[np.ndarray],
    *,
    dedupe_eps: float = 1e-3,
    bridge_max_step_m: float = DEFAULT_LEG_BRIDGE_MAX_STEP_M,
    z_bridge_max_step: float | None = None,
    smooth_leg_junctions: bool = True,
) -> np.ndarray:

    if z_bridge_max_step is not None and bridge_max_step_m == DEFAULT_LEG_BRIDGE_MAX_STEP_M:
        bridge_max_step_m = float(z_bridge_max_step)
    if not legs:
        return np.zeros((0, 3), dtype=np.float32)
    chunks: list[np.ndarray] = [np.asarray(legs[0], dtype=np.float32).reshape(-1, 3)]
    for leg in legs[1:]:
        p = np.asarray(leg, dtype=np.float32).reshape(-1, 3)
        if p.shape[0] == 0:
            continue
        prev = chunks[-1]
        if not prev.shape[0]:
            chunks.append(p)
            continue
        gap = float(np.linalg.norm(prev[-1] - p[0]))
        if gap < dedupe_eps:
            p_use = p[1:] if p.shape[0] > 1 else np.zeros((0, 3), dtype=np.float32)
        else:
            if bool(smooth_leg_junctions) and gap > dedupe_eps * 4.0:
                t0 = _endpoint_tangent(prev, at_start=False)
                t1 = _endpoint_tangent(p, at_start=True)
                if float(np.linalg.norm(t0)) > 1e-6 and float(np.linalg.norm(t1)) > 1e-6:
                    bridge = _hermite_bridge_3d(
                        prev[-1],
                        t0,
                        p[0],
                        t1,
                        bridge_max_step_m,
                        include_first=False,
                        include_last=False,
                    )
                else:
                    bridge = _linear_bridge_3d(
                        prev[-1],
                        p[0],
                        bridge_max_step_m,
                        include_first=False,
                        include_last=False,
                    )
            else:
                bridge = _linear_bridge_3d(
                    prev[-1],
                    p[0],
                    bridge_max_step_m,
                    include_first=False,
                    include_last=False,
                )
            if bridge.shape[0]:
                chunks.append(bridge)
            p_use = p
        if p_use.shape[0]:
            chunks.append(p_use)
    return _merge_trajectory_chunks(chunks, dedupe_eps=dedupe_eps)

def parse_traversal_legs(
    billboard_ids: list[int],
    placed_cubes: list[dict],
    start_xyz_world: np.ndarray,
    *,
    portal_leg_goal_fn: Callable[[dict, np.ndarray], np.ndarray],
) -> list[TraversalLegSpec]:

    legs: list[TraversalLegSpec] = []
    cur_start = np.asarray(start_xyz_world, dtype=np.float64).reshape(3)
    for i, bid in enumerate(billboard_ids):
        portal = find_rect_portal_by_billboard_id(placed_cubes, bid)
        if portal is None:
            print(f"[Multi-leg] WARN: billboard_id={bid} not found — skipping leg {i}")
            continue
        goal = np.asarray(portal_leg_goal_fn(portal, cur_start), dtype=np.float64).reshape(3)
        legs.append(
            TraversalLegSpec(
                leg_index=len(legs),
                billboard_id=int(bid),
                portal=portal,
                sub_instruction=leg_sub_instruction_for_billboard_id(bid),
                goal_xyz_world=goal,
                start_xyz_world=cur_start.copy(),
            )
        )
        cur_start = goal.copy()
    return legs

def _draw_blue_path_on_bgr(
    img_bgr: np.ndarray,
    pts_world: np.ndarray,
    *,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
) -> None:
    import cv2

    pix_ln: list[tuple[int, int]] = []
    for row in np.asarray(pts_world, dtype=np.float64).reshape(-1, 3):
        pix = wproj(float(row[0]), float(row[1]), float(row[2]))
        if pix is not None:
            pix_ln.append((int(pix[0]), int(pix[1])))
    blue_bgr = (255, 0, 0)
    if len(pix_ln) >= 2:
        arr_pix = np.array(pix_ln, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            img_bgr,
            [arr_pix],
            isClosed=False,
            color=blue_bgr,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    for xy in pix_ln:
        cv2.circle(img_bgr, xy, 4, blue_bgr, -1, lineType=cv2.LINE_AA)

def run_multi_leg_navigation_phase12(
    *,
    billboard_ids: list[int],
    portal_leg_goal_fn: Callable[[dict, np.ndarray], np.ndarray],
    phase12_kwargs: dict[str, Any],
    initial_drone_pos: np.ndarray,
    create_feedback_spheres: bool = True,
    feedback_zones_registry: Any | None = None,
) -> dict[str, Any] | None:

    placed_cubes: list[dict] = phase12_kwargs["placed_cubes"]
    leg_specs = parse_traversal_legs(
        billboard_ids,
        placed_cubes,
        initial_drone_pos,
        portal_leg_goal_fn=portal_leg_goal_fn,
    )
    if len(leg_specs) < 2:
        print("[Multi-leg] Fewer than 2 valid legs after parsing — falling back to single-leg.")
        return None

    print("\n" + "=" * 60)
    print(
        f"[Multi-leg] {len(leg_specs)} traversal legs (billboard_id order): "
        f"{[s.billboard_id for s in leg_specs]}"
    )

    rec_folder = phase12_kwargs.get("rec_folder")
    recording_folder = phase12_kwargs.get("recording_folder")
    root_dir = phase12_kwargs.get("root_dir")
    experiment_base = resolve_experiment_base_folder(
        rec_folder=rec_folder,
        recording_folder=recording_folder,
        root_dir=root_dir,
    )
    phase1_base = resolve_phase_folder(experiment_base, "phase1", mkdir=True)
    phase2_base = resolve_phase_folder(experiment_base, PHASE2_DIR, mkdir=True)
    phase3_folder = resolve_phase_folder(experiment_base, "phase3", mkdir=False)

    leg_trajs: list[np.ndarray] = []
    leg_snapshots: list[dict[str, Any]] = []
    feedback_registry = (
        feedback_zones_registry
        if feedback_zones_registry is not None
        else Phase3FeedbackZones()
    )
    last_global_img: np.ndarray | None = None
    last_wproj: Callable[[float, float, float], tuple[int, int] | None] | None = None
    last_stereo_img: np.ndarray | None = None
    last_wproj_stereo: Callable[[float, float, float], tuple[int, int] | None] | None = None

    n_legs = len(leg_specs)
    for idx, spec in enumerate(leg_specs):
        if idx > 0 and leg_trajs:
            actual_start = np.asarray(leg_trajs[-1][-1], dtype=np.float64)
            spec.start_xyz_world = actual_start.copy()
            spec.goal_xyz_world = np.asarray(
                portal_leg_goal_fn(spec.portal, actual_start), dtype=np.float64
            ).reshape(3)
            print(
                f"[Multi-leg] Leg {spec.leg_index} start synced from leg {idx - 1} end: "
                f"{spec.start_xyz_world.round(3).tolist()}"
            )
        leg_subfolder = f"leg{spec.leg_index}_billboard_{spec.billboard_id}"
        is_last = spec.leg_index == n_legs - 1
        print(
            f"\n[Multi-leg] Leg {spec.leg_index + 1}/{n_legs}: "
            f"billboard_id={spec.billboard_id} "
            f"start={spec.start_xyz_world.round(3).tolist()} "
            f"goal={spec.goal_xyz_world.round(3).tolist()}"
        )
        kw = dict(phase12_kwargs)
        kw["drone_pos"] = spec.start_xyz_world
        kw["g_world"] = spec.goal_xyz_world
        kw["mission_cmd"] = spec.sub_instruction
        kw["cur_instruction"] = spec.sub_instruction
        kw["leg_subfolder"] = leg_subfolder
        kw["phase2_sidecar_name"] = f"navigation_phase2_xy_leg{spec.leg_index}.json"
        kw["phase2_png_filename"] = TOPDOWN_PNG
        kw["phase2_stereo_png_filename"] = STEREO45_PNG
        kw["sync_root_config"] = bool(phase12_kwargs.get("navigation_phase2_sync_root_config")) and is_last
        kw["sync_qs"] = bool(phase12_kwargs.get("navigation_phase2_sync_qs")) and is_last
        kw["leg_index"] = spec.leg_index
        kw["billboard_id"] = spec.billboard_id
        kw["target_portal_ref"] = spec.portal
        kw["create_feedback_spheres"] = bool(create_feedback_spheres)
        kw["feedback_zones_registry"] = feedback_registry

        result = run_single_leg_phase1_and_phase2(**kw)
        if result is None or result.get("trajectory") is None:
            print(f"[Multi-leg] Leg {spec.leg_index} produced no trajectory.")
            continue
        traj = anchor_leg_trajectory_endpoints(
            np.asarray(result["trajectory"], dtype=np.float32),
            spec.start_xyz_world,
            spec.goal_xyz_world,
        ).astype(np.float32)
        leg_trajs.append(traj)
        if result.get("phase2_snapshot"):
            snap = dict(result["phase2_snapshot"])
            snap["leg_index"] = spec.leg_index
            snap["billboard_id"] = spec.billboard_id
            leg_snapshots.append(snap)
        if result.get("img_marked_bgr") is not None and result.get("wproj") is not None:
            last_global_img = result["img_marked_bgr"]
            last_wproj = result["wproj"]
        if result.get("img_stereo_bgr") is not None and result.get("wproj_stereo") is not None:
            last_stereo_img = result["img_stereo_bgr"]
            last_wproj_stereo = result["wproj_stereo"]

    if not leg_trajs:
        print("[Multi-leg] No leg trajectories — abort.")
        return None

    global_traj = smooth_stitch_trajectory_legs(leg_trajs)
    print(
        f"\n[Multi-leg] Stitched global path: {global_traj.shape[0]} waypoints "
        f"from {len(leg_trajs)} legs"
    )
    for i, row in enumerate(global_traj):
        print(
            f"  g{i}: XY=[{float(row[0]):.6f}, {float(row[1]):.6f}]  Z={float(row[2]):.6f}"
        )

    try:
        import cv2

        p_client = phase12_kwargs.get("p")
        world_recording_view_proj_fn = phase12_kwargs.get("world_recording_view_proj_fn")
        render_world_recording_rgb_fn = phase12_kwargs.get("render_world_recording_rgb_fn")
        world_xyz_to_recording_image_pixel_fn = phase12_kwargs.get(
            "world_xyz_to_recording_image_pixel_fn"
        )
        workspace_lo = np.asarray(phase12_kwargs.get("workspace_lo"), dtype=np.float64)
        workspace_hi = np.asarray(phase12_kwargs.get("workspace_hi"), dtype=np.float64)
        virtual_base_world = np.asarray(
            phase12_kwargs.get("virtual_base_world"), dtype=np.float64
        ).reshape(3)
        tw, th = 800, 800
        if (
            p_client is not None
            and world_recording_view_proj_fn is not None
            and render_world_recording_rgb_fn is not None
            and world_xyz_to_recording_image_pixel_fn is not None
            and global_traj.shape[0] >= 1
        ):
            route_pts = np.vstack(
                [
                    np.asarray(initial_drone_pos, dtype=np.float64).reshape(1, 3),
                    np.asarray(global_traj, dtype=np.float64).reshape(-1, 3),
                ]
            )
            route_lo = route_pts.min(axis=0)
            route_hi = route_pts.max(axis=0)
            ws_lo_w = workspace_lo + virtual_base_world
            ws_hi_w = workspace_hi + virtual_base_world
            scene_lo = np.minimum(ws_lo_w, route_lo)
            scene_hi = np.maximum(ws_hi_w, route_hi)
            pad = np.array([0.12, 0.12, 0.05], dtype=np.float64)
            view_lo = scene_lo - pad
            view_hi = scene_hi + pad
            view_m, proj_m = world_recording_view_proj_fn(
                p_client,
                view_lo.astype(np.float32),
                view_hi.astype(np.float32),
                view_kind="top",
                width=tw,
                height=th,
                fov=float(phase12_kwargs.get("recording_scene_fov", 48.0)),
                margin=float(phase12_kwargs.get("recording_scene_margin", 1.75)),
                distance_scale=float(phase12_kwargs.get("recording_camera_distance_scale", 0.4)),
                top_view_distance_scale=float(
                    phase12_kwargs.get("recording_topview_distance_scale", 1.22)
                ),
            )
            top_rgb = render_world_recording_rgb_fn(
                p_client, view_m, proj_m, width=tw, height=th
            )

            def wproj_global(wx: float, wy: float, wz: float) -> tuple[int, int] | None:
                return world_xyz_to_recording_image_pixel_fn(
                    np.array([wx, wy, wz], dtype=np.float64),
                    view_m,
                    proj_m,
                    width=tw,
                    height=th,
                )

            g_copy = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
            _draw_blue_path_on_bgr(g_copy, global_traj, wproj=wproj_global)
            g_path = phase2_base / "global_topdown.png"
            cv2.imwrite(str(g_path), g_copy)
            print(f"[Multi-leg] Saved global top-down PNG (scene-wide): {g_path}")

            view_st, proj_st = world_recording_view_proj_fn(
                p_client,
                view_lo.astype(np.float32),
                view_hi.astype(np.float32),
                view_kind="45deg",
                width=tw,
                height=th,
                fov=float(phase12_kwargs.get("recording_scene_fov", 48.0)),
                margin=float(phase12_kwargs.get("recording_scene_margin", 1.75)),
                distance_scale=float(phase12_kwargs.get("recording_camera_distance_scale", 0.4)),
                top_view_distance_scale=float(
                    phase12_kwargs.get("recording_topview_distance_scale", 1.22)
                ),
                stereo45_view_distance_scale=float(
                    phase12_kwargs.get("recording_stereo45_distance_scale", 1.5)
                ),
            )
            stereo_rgb = render_world_recording_rgb_fn(
                p_client, view_st, proj_st, width=tw, height=th
            )

            def wproj_st_global(wx: float, wy: float, wz: float) -> tuple[int, int] | None:
                return world_xyz_to_recording_image_pixel_fn(
                    np.array([wx, wy, wz], dtype=np.float64),
                    view_st,
                    proj_st,
                    width=tw,
                    height=th,
                )

            st_copy = cv2.cvtColor(stereo_rgb, cv2.COLOR_RGB2BGR)
            _draw_blue_path_on_bgr(st_copy, global_traj, wproj=wproj_st_global)
            st_path = phase2_base / "global_stereo45deg.png"
            cv2.imwrite(str(st_path), st_copy)
            print(f"[Multi-leg] Saved global stereo PNG (scene-wide): {st_path}")
        elif last_global_img is not None and last_wproj is not None:
            g_copy = np.asarray(last_global_img, dtype=np.uint8).copy()
            _draw_blue_path_on_bgr(g_copy, global_traj, wproj=last_wproj)
            g_path = phase2_base / "global_topdown.png"
            cv2.imwrite(str(g_path), g_copy)
            print(f"[Multi-leg] Saved global top-down PNG: {g_path}")

            if last_stereo_img is not None and last_wproj_stereo is not None:
                st_copy = np.asarray(last_stereo_img, dtype=np.uint8).copy()
                _draw_blue_path_on_bgr(st_copy, global_traj, wproj=last_wproj_stereo)
                st_path = phase2_base / "global_stereo45deg.png"
                cv2.imwrite(str(st_path), st_copy)
                print(f"[Multi-leg] Saved global stereo PNG: {st_path}")
    except Exception as exc:
        print(f"[Multi-leg] WARNING: could not save global stitched PNG — {exc}")

    multi_snapshot = {
        "phase": "navigation_phase_multi_xy",
        "saved_at": datetime.datetime.now().isoformat(),
        "billboard_ids_ordered": [s.billboard_id for s in leg_specs],
        "num_legs": len(leg_specs),
        "legs": leg_snapshots,
        "global_waypoints": global_traj.astype(float).tolist(),
        "global_xy_keypoints": [[float(r[0]), float(r[1])] for r in global_traj],
        "mission_cmd_excerpt": str(phase12_kwargs.get("mission_cmd") or "")[:800],
    }
    multi_path = phase2_base / "navigation_phase_multi_xy.json"
    with open(multi_path, "w", encoding="utf-8") as f:
        json.dump(multi_snapshot, f, indent=2, ensure_ascii=False)
    print(f"[Multi-leg] Sidecar JSON: {multi_path}")

    if bool(phase12_kwargs.get("navigation_phase2_sync_root_config")) and phase12_kwargs.get(
        "config_json_path"
    ):
        try:
            read_config_fn = phase12_kwargs["read_config_fn"]
            cfg = read_config_fn(phase12_kwargs["config_json_path"])
            cfg["_navigation_phase_multi_snapshot"] = multi_snapshot
            with open(phase12_kwargs["config_json_path"], "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print(f"[Multi-leg] Wrote `_navigation_phase_multi_snapshot` into {phase12_kwargs['config_json_path']}")
        except Exception as exc:
            print(f"[Multi-leg] WARNING: config merge failed — {exc}")

    phase3_zones = feedback_registry if feedback_registry.num_zones > 0 else None

    print("[Multi-leg] Completed multi-leg Phase1+2 + global stitch.")
    print("=" * 60 + "\n")
    return {
        "trajectory": global_traj,
        "phase3_zones": phase3_zones,
        "multi_snapshot": multi_snapshot,
        "experiment_base": experiment_base,
        "phase1_folder": phase1_base,
        "phase2_folder": phase2_base,
        "phase3_folder": phase3_folder,
        "img_marked_bgr": last_global_img,
    }

def _leg_refinement_json_safe(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    for key in ("segment_xyz", "coarse_span_xyz"):
        val = out.get(key)
        if val is not None:
            out[key] = np.asarray(val, dtype=np.float64).reshape(-1, 3).tolist()
    return out

def run_multi_action_navigation_phase12(
    *,
    mission_cmd: str,
    phase12_kwargs: dict[str, Any],
    initial_drone_pos: np.ndarray,
    create_feedback_spheres: bool = True,
    feedback_zones_registry: Any | None = None,
    action_cache: dict[str, tuple[BasicAction, int]] | None = None,
) -> dict[str, Any] | None:

    placed_cubes: list[dict] = phase12_kwargs["placed_cubes"]
    p_client = phase12_kwargs.get("p")
    leg_specs = parse_mission_action_legs(
        mission_cmd,
        placed_cubes,
        initial_drone_pos,
        action_cache=action_cache,
    )
    if len(leg_specs) < 2:
        print("[Multi-action] Fewer than 2 valid action legs — falling back to single-leg.")
        return None

    print("\n" + "=" * 60)
    print(
        f"[Multi-action] {len(leg_specs)} basic-action legs: "
        + ", ".join(
            f"id={s.billboard_id}:{s.action.value}" for s in leg_specs
        )
    )

    rec_folder = phase12_kwargs.get("rec_folder")
    recording_folder = phase12_kwargs.get("recording_folder")
    root_dir = phase12_kwargs.get("root_dir")
    experiment_base = resolve_experiment_base_folder(
        rec_folder=rec_folder,
        recording_folder=recording_folder,
        root_dir=root_dir,
    )
    phase1_base = resolve_phase_folder(experiment_base, "phase1", mkdir=True)
    phase2_base = resolve_phase_folder(experiment_base, PHASE2_DIR, mkdir=True)
    phase3_folder = resolve_phase_folder(experiment_base, "phase3", mkdir=False)

    leg_trajs: list[np.ndarray] = []
    leg_trajs_coarse: list[np.ndarray] = []
    leg_snapshots: list[dict[str, Any]] = []
    leg_refinements: list[dict[str, Any]] = []
    feedback_registry = (
        feedback_zones_registry
        if feedback_zones_registry is not None
        else Phase3FeedbackZones()
    )
    last_global_img: np.ndarray | None = None
    last_wproj: Callable[[float, float, float], tuple[int, int] | None] | None = None
    last_stereo_img: np.ndarray | None = None
    last_wproj_stereo: Callable[[float, float, float], tuple[int, int] | None] | None = None

    n_legs = len(leg_specs)
    drone_r = float(phase12_kwargs.get("navigation_phase2_astar_obstacle_pad_m", 0.078)) * 0.9
    if drone_r <= 0.0:
        drone_r = 0.07
    if p_client is not None and create_feedback_spheres:
        _ensure_multi_action_portal_feedback_zones(p_client, feedback_registry, leg_specs)

    for idx, spec in enumerate(leg_specs):
        if idx > 0 and leg_trajs:
            actual_start = np.asarray(leg_trajs[-1][-1], dtype=np.float64)
            _refresh_leg_spec_endpoints(
                spec,
                start_xyz=actual_start,
                next_spec=leg_specs[idx + 1] if idx + 1 < n_legs else None,
                placed_cubes=placed_cubes,
                drone_r=drone_r,
            )
            print(
                f"[Multi-action] Leg {spec.leg_index} start synced from leg {idx - 1} end: "
                f"{spec.start_xyz_world.round(3).tolist()}"
            )
        leg_subfolder = f"leg{spec.leg_index}_billboard_{spec.billboard_id}_{spec.action.value}"
        is_last = spec.leg_index == n_legs - 1
        lap_txt = f" {spec.orbit_laps} laps" if spec.action == BasicAction.ORBIT else ""
        print(
            f"\n[Multi-action] Leg {spec.leg_index + 1}/{n_legs}: "
            f"billboard_id={spec.billboard_id} action={spec.action.value}{lap_txt}"
        )
        print(
            f"  start={spec.start_xyz_world.round(3).tolist()} "
            f"goal={spec.goal_xyz_world.round(3).tolist()}"
        )
        kw = dict(phase12_kwargs)
        kw["drone_pos"] = spec.start_xyz_world
        phase2_goal = np.asarray(spec.goal_xyz_world, dtype=np.float64).reshape(3)
        if spec.action == BasicAction.PASS_THROUGH:
            from .portal_geometry import portal_opening_center_world

            phase2_goal = portal_opening_center_world(spec.portal)
            print(
                f"[Multi-action] pass_through leg {spec.leg_index}: "
                f"Phase2 goal → opening center {phase2_goal.round(3).tolist()}"
            )
        kw["g_world"] = phase2_goal
        kw["mission_cmd"] = spec.sub_instruction
        kw["cur_instruction"] = spec.sub_instruction
        kw["leg_subfolder"] = leg_subfolder
        kw["phase2_sidecar_name"] = f"navigation_phase2_xy_leg{spec.leg_index}.json"
        kw["phase2_png_filename"] = TOPDOWN_PNG
        kw["phase2_stereo_png_filename"] = STEREO45_PNG
        kw["sync_root_config"] = bool(phase12_kwargs.get("navigation_phase2_sync_root_config")) and is_last
        kw["sync_qs"] = bool(phase12_kwargs.get("navigation_phase2_sync_qs")) and is_last
        kw["leg_index"] = spec.leg_index
        kw["billboard_id"] = spec.billboard_id
        kw["target_portal_ref"] = spec.portal
        kw["create_feedback_spheres"] = bool(create_feedback_spheres)
        kw["feedback_zones_registry"] = feedback_registry

        result = run_single_leg_phase1_and_phase2(**kw)
        traj: np.ndarray | None = None
        if result is not None and result.get("trajectory") is not None:
            cand = np.asarray(result["trajectory"], dtype=np.float32).reshape(-1, 3)
            if cand.shape[0] >= 1:
                traj = cand
        if traj is None:
            print(
                f"[Multi-action] Leg {spec.leg_index} Phase2 unsolved; "
                f"using basic-action geometric fallback trajectory (billboard_id={spec.billboard_id} "
                f"action={spec.action.value})。"
            )
            traj = _geometry_fallback_leg_trajectory(
                spec, placed_cubes, drone_r=drone_r
            )
            leg_trajs_coarse.append(
                anchor_leg_trajectory_endpoints(
                    traj,
                    spec.start_xyz_world,
                    spec.goal_xyz_world,
                ).astype(np.float32)
            )
        else:
            leg_trajs_coarse.append(
                anchor_leg_trajectory_endpoints(
                    traj,
                    spec.start_xyz_world,
                    spec.goal_xyz_world,
                ).astype(np.float32)
            )
            if result is not None and result.get("phase2_snapshot"):
                snap = dict(result["phase2_snapshot"])
                snap["leg_index"] = spec.leg_index
                snap["billboard_id"] = spec.billboard_id
                snap["basic_action"] = spec.action.value
                leg_snapshots.append(snap)
            if result is not None and result.get("img_marked_bgr") is not None and result.get("wproj") is not None:
                last_global_img = result["img_marked_bgr"]
                last_wproj = result["wproj"]
            if (
                result is not None
                and result.get("img_stereo_bgr") is not None
                and result.get("wproj_stereo") is not None
            ):
                last_stereo_img = result["img_stereo_bgr"]
                last_wproj_stereo = result["wproj_stereo"]
        if p_client is not None:
            traj_f, meta = _refine_leg_trajectory_for_action(
                p_client,
                traj,
                spec,
                feedback_registry,
                mission_cmd=mission_cmd,
                placed_cubes=placed_cubes,
                action_cache=action_cache,
            )
            traj = np.asarray(traj_f, dtype=np.float32)
            if meta is not None:
                meta["leg_index"] = spec.leg_index
                meta["billboard_id"] = spec.billboard_id
                leg_refinements.append(meta)
        traj = anchor_leg_trajectory_endpoints(
            traj,
            spec.start_xyz_world,
            spec.goal_xyz_world,
        ).astype(np.float32)
        leg_trajs.append(traj)

    if not leg_trajs:
        print("[Multi-action] No leg trajectories — abort.")
        return None

    global_traj = smooth_stitch_trajectory_legs(leg_trajs)
    global_traj_coarse = (
        smooth_stitch_trajectory_legs(leg_trajs_coarse)
        if leg_trajs_coarse
        else global_traj.copy()
    )
    print(
        f"\n[Multi-action] Stitched global path: {global_traj.shape[0]} waypoints "
        f"from {len(leg_trajs)} legs ({len(leg_refinements)} refined in-leg)"
    )
    if len(leg_refinements) < n_legs:
        missing = n_legs - len(leg_refinements)
        print(
            f"[Multi-action] WARN: only {len(leg_refinements)}/{n_legs} legs action-refined; "
            f"missing {missing} (check Phase2 fallback and p_client)."
        )
    for i, row in enumerate(global_traj):
        print(
            f"  g{i}: XY=[{float(row[0]):.6f}, {float(row[1]):.6f}]  Z={float(row[2]):.6f}"
        )

    try:
        import cv2

        world_recording_view_proj_fn = phase12_kwargs.get("world_recording_view_proj_fn")
        render_world_recording_rgb_fn = phase12_kwargs.get("render_world_recording_rgb_fn")
        world_xyz_to_recording_image_pixel_fn = phase12_kwargs.get(
            "world_xyz_to_recording_image_pixel_fn"
        )
        workspace_lo = np.asarray(phase12_kwargs.get("workspace_lo"), dtype=np.float64)
        workspace_hi = np.asarray(phase12_kwargs.get("workspace_hi"), dtype=np.float64)
        virtual_base_world = np.asarray(
            phase12_kwargs.get("virtual_base_world"), dtype=np.float64
        ).reshape(3)
        tw, th = 800, 800
        if (
            p_client is not None
            and world_recording_view_proj_fn is not None
            and render_world_recording_rgb_fn is not None
            and world_xyz_to_recording_image_pixel_fn is not None
            and global_traj.shape[0] >= 1
        ):
            route_pts = np.vstack(
                [
                    np.asarray(initial_drone_pos, dtype=np.float64).reshape(1, 3),
                    np.asarray(global_traj, dtype=np.float64).reshape(-1, 3),
                ]
            )
            route_lo = route_pts.min(axis=0)
            route_hi = route_pts.max(axis=0)
            ws_lo_w = workspace_lo + virtual_base_world
            ws_hi_w = workspace_hi + virtual_base_world
            scene_lo = np.minimum(ws_lo_w, route_lo)
            scene_hi = np.maximum(ws_hi_w, route_hi)
            pad = np.array([0.12, 0.12, 0.05], dtype=np.float64)
            view_lo = scene_lo - pad
            view_hi = scene_hi + pad
            view_m, proj_m = world_recording_view_proj_fn(
                p_client,
                view_lo.astype(np.float32),
                view_hi.astype(np.float32),
                view_kind="top",
                width=tw,
                height=th,
                fov=float(phase12_kwargs.get("recording_scene_fov", 48.0)),
                margin=float(phase12_kwargs.get("recording_scene_margin", 1.75)),
                distance_scale=float(phase12_kwargs.get("recording_camera_distance_scale", 0.4)),
                top_view_distance_scale=float(
                    phase12_kwargs.get("recording_topview_distance_scale", 1.22)
                ),
            )
            top_rgb = render_world_recording_rgb_fn(
                p_client, view_m, proj_m, width=tw, height=th
            )

            def wproj_global(wx: float, wy: float, wz: float) -> tuple[int, int] | None:
                return world_xyz_to_recording_image_pixel_fn(
                    np.array([wx, wy, wz], dtype=np.float64),
                    view_m,
                    proj_m,
                    width=tw,
                    height=th,
                )

            g_copy = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
            _draw_blue_path_on_bgr(g_copy, global_traj, wproj=wproj_global)
            g_path = phase2_base / "global_topdown.png"
            cv2.imwrite(str(g_path), g_copy)
            print(f"[Multi-action] Saved global top-down PNG: {g_path}")
        elif last_global_img is not None and last_wproj is not None:
            g_copy = np.asarray(last_global_img, dtype=np.uint8).copy()
            _draw_blue_path_on_bgr(g_copy, global_traj, wproj=last_wproj)
            g_path = phase2_base / "global_topdown.png"
            cv2.imwrite(str(g_path), g_copy)
            print(f"[Multi-action] Saved global top-down PNG: {g_path}")
    except Exception as exc:
        print(f"[Multi-action] WARNING: could not save global stitched PNG — {exc}")

    multi_snapshot = {
        "phase": "navigation_phase_multi_action_xy",
        "saved_at": datetime.datetime.now().isoformat(),
        "billboard_ids_ordered": [s.billboard_id for s in leg_specs],
        "actions_ordered": [s.action.value for s in leg_specs],
        "num_legs": len(leg_specs),
        "legs": leg_snapshots,
        "leg_refinements": [_leg_refinement_json_safe(m) for m in leg_refinements],
        "global_waypoints": global_traj.astype(float).tolist(),
        "global_xy_keypoints": [[float(r[0]), float(r[1])] for r in global_traj],
        "mission_cmd_excerpt": str(mission_cmd or "")[:800],
    }
    multi_path = phase2_base / "navigation_phase_multi_action_xy.json"
    with open(multi_path, "w", encoding="utf-8") as f:
        json.dump(multi_snapshot, f, indent=2, ensure_ascii=False)
    print(f"[Multi-action] Sidecar JSON: {multi_path}")

    phase3_zones = feedback_registry if feedback_registry.num_zones > 0 else None
    print("[Multi-action] Completed per-action Phase1+2+refinement + global stitch.")
    print("=" * 60 + "\n")
    return {
        "trajectory": global_traj,
        "trajectory_coarse": global_traj_coarse,
        "phase3_zones": phase3_zones,
        "multi_snapshot": multi_snapshot,
        "experiment_base": experiment_base,
        "phase1_folder": phase1_base,
        "phase2_folder": phase2_base,
        "phase3_folder": phase3_folder,
        "img_marked_bgr": last_global_img,
        "leg_refinements": leg_refinements,
    }
