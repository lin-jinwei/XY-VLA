from __future__ import annotations

import math
import re
from typing import Any, Callable

import numpy as np

from .instruction_parse import (
    extract_ordered_mission_billboard_ids,
    extract_ordered_traversal_billboard_ids,
    find_rect_portal_by_billboard_id,
    leg_sub_instruction_for_billboard_id,
)
from .phase3_actions import (
    BasicAction,
    _action_from_clause,
    _parse_orbit_laps,
    basic_action_from_clause_keywords,
    basic_action_from_mission_zone,
    _zone_billboard_id,
    entry_key,
    resolve_basic_action_for_zone,
    resolve_mission_clause_for_zone,
)

def _placed_obj_nav_kind(c: dict) -> str:
    sh = str(c.get("shape", "cube")).lower()
    if sh in ("rect_frame", "square_frame", "frame"):
        return "gate"
    if sh == "sphere":
        return "sphere"
    if sh == "ramp":
        return "ramp"
    return "cube"

def _object_aabb_lo_hi(c: dict) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
    bh = c.get("bounds_half")
    if bh is not None:
        h = np.asarray(bh, dtype=np.float64).reshape(3)
        return pos - h, pos + h
    half = float(c.get("half", 0.025))
    h = np.array([half, half, half], dtype=np.float64)
    return pos - h, pos + h

def _object_display_name(c: dict) -> str:
    col = str(c.get("color_name", c.get("color", "?")))
    return f"{col} {_placed_obj_nav_kind(c)}"

def _object_spec_from_portal(portal: dict, *, is_target: bool = True) -> dict[str, Any]:
    lo, hi = _object_aabb_lo_hi(portal)
    center = (lo + hi) * 0.5
    return {
        "name": _object_display_name(portal),
        "center": center,
        "aabb": (lo.tolist(), hi.tolist()),
        "is_target": bool(is_target),
        "billboard_id": portal.get("portal_label"),
        "ref": portal,
    }

def collect_specified_objects_from_mission(
    mission_cmd: str | None,
    placed_cubes: list[dict],
    *,
    first_rect_portal_fn: Callable[..., dict | None] | None = None,
    prefer_near_xyz: np.ndarray | None = None,
) -> list[dict[str, Any]]:

    ins = str(mission_cmd or "").strip()
    if not ins or not placed_cubes:
        return []

    specs: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    bids = extract_ordered_mission_billboard_ids(ins)
    if bids:
        for bid in bids:
            portal = find_rect_portal_by_billboard_id(placed_cubes, int(bid))
            if portal is None:
                print(f"[InstrAnalysis] WARN: billboard_id={bid} not found; skipping action classification.")
                continue
            spec = _object_spec_from_portal(portal, is_target=True)
            key = entry_key(str(spec["name"]), spec["center"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            specs.append(spec)
        return specs

    portal = None
    if first_rect_portal_fn is not None:
        try:
            portal = first_rect_portal_fn(
                ins,
                placed_cubes,
                prefer_near_xyz=prefer_near_xyz,
            )
        except TypeError:
            portal = first_rect_portal_fn(ins, placed_cubes)
    if portal is not None:
        specs.append(_object_spec_from_portal(portal, is_target=True))
    return specs

def capture_workspace_topdown_rgb(
    p: Any,
    *,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    virtual_base_world: np.ndarray,
    world_recording_view_proj_fn: Callable[..., tuple[list[float], list[float]]],
    render_world_recording_rgb_fn: Callable[..., np.ndarray],
    recording_scene_fov: float = 48.0,
    recording_scene_margin: float = 1.75,
    recording_camera_distance_scale: float = 0.4,
    recording_topview_distance_scale: float = 1.22,
    width: int = 800,
    height: int = 800,
) -> np.ndarray:

    ws_lo = np.asarray(workspace_lo, dtype=np.float64) + np.asarray(
        virtual_base_world, dtype=np.float64
    ).reshape(3)
    ws_hi = np.asarray(workspace_hi, dtype=np.float64) + np.asarray(
        virtual_base_world, dtype=np.float64
    ).reshape(3)
    view_m, proj_m = world_recording_view_proj_fn(
        p,
        ws_lo.astype(np.float32),
        ws_hi.astype(np.float32),
        view_kind="top",
        width=int(width),
        height=int(height),
        fov=float(recording_scene_fov),
        margin=float(recording_scene_margin),
        distance_scale=float(recording_camera_distance_scale),
        top_view_distance_scale=float(recording_topview_distance_scale),
    )
    return np.asarray(
        render_world_recording_rgb_fn(p, view_m, proj_m, width=int(width), height=int(height)),
        dtype=np.uint8,
    )

def _user_clause_for_object(
    mission_cmd: str | None,
    *,
    zone_name: str,
    zone_center: np.ndarray,
    placed_cubes: list[dict] | None,
) -> str:
    ins = str(mission_cmd or "").strip()
    if not ins:
        return str(zone_name)
    clause = resolve_mission_clause_for_zone(
        ins,
        zone_name=str(zone_name),
        zone_center=zone_center,
        placed_cubes=placed_cubes,
    )
    bid = _zone_billboard_id(zone_name, zone_center, placed_cubes)
    if bid is not None and clause == ins and _action_from_clause(clause) is None:
        return leg_sub_instruction_for_billboard_id(int(bid))
    return clause

def _compose_xvla_task(
    task: str,
    *,
    compose_instruction_fn: Callable[..., str] | None,
    scene_catalog: str | None,
    xvla_scene_semantic_context: bool,
    xvla_path_planning_instruction_suffix: str,
) -> str:
    text = str(task).strip()
    if compose_instruction_fn is None:
        return text
    return compose_instruction_fn(
        text,
        scene_catalog=str(scene_catalog or ""),
        enabled=bool(xvla_scene_semantic_context and str(scene_catalog or "").strip()),
        planning_suffix=str(xvla_path_planning_instruction_suffix or ""),
    )

def _query_xvla_actions(
    *,
    query_xvla_fn: Callable[..., Any],
    server_url: str,
    image_rgb: np.ndarray,
    proprio: np.ndarray,
    instruction: str,
    xvla_steps: int,
    timeout: float,
) -> np.ndarray | None:
    try:
        actions = query_xvla_fn(
            server_url,
            np.asarray(image_rgb, dtype=np.uint8),
            np.asarray(proprio, dtype=np.float32).reshape(-1),
            str(instruction),
            steps=max(1, int(xvla_steps)),
            timeout=float(timeout),
        )
        return np.asarray(actions, dtype=np.float64)
    except Exception as exc:
        print(f"[InstrAnalysis][X-VLA] /act failed ({instruction[:72]!r}…): {exc}")
        return None

def _normalize_xy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(2, dtype=np.float64)
    return v / n

def _infer_basic_action_from_xvla_actions(
    actions: np.ndarray,
    *,
    drone_pos: np.ndarray,
    object_center: np.ndarray,
    user_clause: str,
    is_target: bool,
) -> BasicAction:

    explicit = _action_from_clause(user_clause)
    if explicit is not None:
        return explicit

    pos = np.asarray(drone_pos, dtype=np.float64).reshape(3)
    cen = np.asarray(object_center, dtype=np.float64).reshape(3)
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    step0 = arr[0, 0:3] if arr.shape[1] >= 3 else np.zeros(3, dtype=np.float64)
    move = step0 - pos
    to_obj = cen - pos
    move_xy = move[:2]
    to_obj_xy = to_obj[:2]
    move_z = float(move[2])

    if move_z > 0.035 and float(np.linalg.norm(move_xy)) < 0.025:
        return BasicAction.HOVER
    if float(np.linalg.norm(move)) < 1e-5:
        return BasicAction.HOVER

    move_n = _normalize_xy(move_xy)
    to_n = _normalize_xy(to_obj_xy)
    align = float(np.dot(move_n, to_n)) if float(np.linalg.norm(move_n)) > 1e-9 else 0.0

    if arr.shape[0] >= 4:
        angs: list[float] = []
        for row in arr[:, 0:2]:
            v = row - cen[:2]
            if float(np.linalg.norm(v)) > 1e-4:
                angs.append(float(math.atan2(v[1], v[0])))
        if len(angs) >= 3:
            angs.sort()
            spread = float(angs[-1] - angs[0])
            if spread > 0.75 * math.pi:
                return BasicAction.ORBIT

    if align > 0.72:
        if re.search(
            r"collid|crash|hit|ram|strike|impact",
            user_clause,
            re.IGNORECASE,
        ):
            return BasicAction.COLLISION
        return BasicAction.PASS_THROUGH
    if align < 0.25 and float(np.linalg.norm(move_xy)) > 0.015:
        return BasicAction.FLY_BY
    if re.search(r"orbit|circle|inspect", user_clause, re.IGNORECASE):
        return BasicAction.ORBIT
    if re.search(r"hover|loiter", user_clause, re.IGNORECASE):
        return BasicAction.HOVER

    return BasicAction.PASS_THROUGH if is_target else BasicAction.FLY_BY

def classify_basic_action_via_xvla(
    *,
    zone_name: str,
    zone_center: np.ndarray,
    is_target: bool,
    placed_cubes: list[dict] | None,
    mission_cmd: str | None,
    query_xvla_fn: Callable[..., Any],
    server_url: str,
    topdown_rgb: np.ndarray,
    proprio: np.ndarray,
    drone_pos: np.ndarray | None = None,
    compose_instruction_fn: Callable[..., str] | None = None,
    scene_catalog: str | None = None,
    xvla_scene_semantic_context: bool = True,
    xvla_path_planning_instruction_suffix: str = "",
    xvla_steps: int = 1,
    xvla_act_request_timeout_s: float = 300.0,
) -> tuple[BasicAction, int, str]:

    user_clause = _user_clause_for_object(
        mission_cmd,
        zone_name=zone_name,
        zone_center=zone_center,
        placed_cubes=placed_cubes,
    )
    keyword = basic_action_from_mission_zone(
        mission_cmd,
        zone_name=zone_name,
        zone_center=zone_center,
        placed_cubes=placed_cubes,
    )
    if keyword is None:
        keyword = basic_action_from_clause_keywords(user_clause)
    if keyword is not None:
        act, laps = keyword
        return act, int(laps), "keyword"

    user_instr = _compose_xvla_task(
        user_clause,
        compose_instruction_fn=compose_instruction_fn,
        scene_catalog=scene_catalog,
        xvla_scene_semantic_context=xvla_scene_semantic_context,
        xvla_path_planning_instruction_suffix=xvla_path_planning_instruction_suffix,
    )
    actions = _query_xvla_actions(
        query_xvla_fn=query_xvla_fn,
        server_url=server_url,
        image_rgb=topdown_rgb,
        proprio=proprio,
        instruction=user_instr,
        xvla_steps=max(1, int(xvla_steps)),
        timeout=xvla_act_request_timeout_s,
    )
    if actions is None:
        act, laps = resolve_basic_action_for_zone(
            mission_cmd,
            zone_name=zone_name,
            zone_center=zone_center,
            is_target=is_target,
            placed_cubes=placed_cubes,
        )
        return act, laps, "heuristic_fallback"

    dpos = np.asarray(proprio, dtype=np.float64).reshape(-1)[:3] if drone_pos is None else np.asarray(
        drone_pos, dtype=np.float64
    ).reshape(3)
    act = _infer_basic_action_from_xvla_actions(
        actions,
        drone_pos=dpos,
        object_center=zone_center,
        user_clause=user_clause,
        is_target=is_target,
    )
    laps = _parse_orbit_laps(user_clause, default=1) if act == BasicAction.ORBIT else 1
    return act, int(laps), "xvla"

def classify_mission_basic_actions_early(
    object_specs: list[dict[str, Any]],
    *,
    mission_cmd: str | None,
    placed_cubes: list[dict] | None,
    query_xvla_fn: Callable[..., Any],
    server_url: str,
    topdown_rgb: np.ndarray,
    proprio: np.ndarray,
    drone_pos: np.ndarray,
    compose_instruction_fn: Callable[..., str] | None = None,
    scene_catalog: str | None = None,
    xvla_scene_semantic_context: bool = True,
    xvla_path_planning_instruction_suffix: str = "",
    xvla_steps: int = 1,
    xvla_act_request_timeout_s: float = 300.0,
) -> dict[str, tuple[BasicAction, int, str]]:

    out: dict[str, tuple[BasicAction, int, str]] = {}
    img = np.asarray(topdown_rgb, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        print("[InstrAnalysis] Skipped: top-down RGB invalid.")
        return out
    if not object_specs:
        print("[InstrAnalysis] Skipped: no specified objects parsed from instruction.")
        return out

    xvla_specs: list[dict[str, Any]] = []
    for spec in object_specs:
        name = str(spec["name"])
        center = np.asarray(spec["center"], dtype=np.float64).reshape(3)
        keyword = basic_action_from_mission_zone(
            mission_cmd,
            zone_name=name,
            zone_center=center,
            placed_cubes=placed_cubes,
        )
        key = entry_key(name, center)
        if keyword is not None:
            act, laps = keyword
            out[key] = (act, int(laps), "keyword")
        else:
            xvla_specs.append(spec)

    n_kw = len(out)
    n_xvla = len(xvla_specs)
    if n_kw and n_xvla:
        print(
            f"\n[InstrAnalysis] First --cmd parse: {n_kw} object(s) matched by keywords, "
            f"{n_xvla} object(s) will call X-VLA /act (steps={max(1, int(xvla_steps))})…"
        )
    elif n_kw:
        print(f"\n[InstrAnalysis] First --cmd parse: all {n_kw} object(s) matched keywords; skipping X-VLA /act.")
    else:
        print(
            f"\n[InstrAnalysis][X-VLA] First --cmd parse: classify basic actions for {n_xvla} object(s) "
            f"(1 /act per object, steps={max(1, int(xvla_steps))})…"
        )

    for spec in object_specs:
        name = str(spec["name"])
        center = np.asarray(spec["center"], dtype=np.float64).reshape(3)
        key = entry_key(name, center)
        if key in out:
            act, laps, src = out[key]
        else:
            act, laps, src = classify_basic_action_via_xvla(
                zone_name=name,
                zone_center=center,
                is_target=bool(spec.get("is_target", True)),
                placed_cubes=placed_cubes,
                mission_cmd=mission_cmd,
                query_xvla_fn=query_xvla_fn,
                server_url=server_url,
                topdown_rgb=img,
                proprio=proprio,
                drone_pos=np.asarray(drone_pos, dtype=np.float64).reshape(3),
                compose_instruction_fn=compose_instruction_fn,
                scene_catalog=scene_catalog,
                xvla_scene_semantic_context=xvla_scene_semantic_context,
                xvla_path_planning_instruction_suffix=xvla_path_planning_instruction_suffix,
                xvla_steps=xvla_steps,
                xvla_act_request_timeout_s=xvla_act_request_timeout_s,
            )
            out[key] = (act, laps, src)

        act, laps, src = out[key]
        lap_txt = f", {laps} lap(s)" if act == BasicAction.ORBIT else ""
        src_label = {"keyword": "keyword", "xvla": "X-VLA", "heuristic_fallback": "keyword fallback"}.get(
            src, src
        )
        print(
            f"  - {name!r} → {act.value}{lap_txt} "
            f"({src_label})"
        )
    return out

def classify_all_zone_actions_via_xvla(
    zones: list[Any],
    *,
    mission_cmd: str | None,
    placed_cubes: list[dict] | None,
    query_xvla_fn: Callable[..., Any],
    server_url: str,
    topdown_rgb: np.ndarray,
    proprio: np.ndarray,
    drone_pos: np.ndarray | None = None,
    compose_instruction_fn: Callable[..., str] | None = None,
    scene_catalog: str | None = None,
    xvla_scene_semantic_context: bool = True,
    xvla_path_planning_instruction_suffix: str = "",
    xvla_steps: int = 1,
    xvla_act_request_timeout_s: float = 300.0,
) -> dict[str, tuple[BasicAction, int, str]]:

    specs = [
        {
            "name": str(z.name),
            "center": np.asarray(z.center, dtype=np.float64).reshape(3),
            "is_target": bool(z.is_target),
        }
        for z in zones
    ]
    dpos = (
        np.asarray(proprio, dtype=np.float64).reshape(-1)[:3]
        if drone_pos is None
        else np.asarray(drone_pos, dtype=np.float64).reshape(3)
    )
    return classify_mission_basic_actions_early(
        specs,
        mission_cmd=mission_cmd,
        placed_cubes=placed_cubes,
        query_xvla_fn=query_xvla_fn,
        server_url=server_url,
        topdown_rgb=topdown_rgb,
        proprio=proprio,
        drone_pos=dpos,
        compose_instruction_fn=compose_instruction_fn,
        scene_catalog=scene_catalog,
        xvla_scene_semantic_context=xvla_scene_semantic_context,
        xvla_path_planning_instruction_suffix=xvla_path_planning_instruction_suffix,
        xvla_steps=xvla_steps,
        xvla_act_request_timeout_s=xvla_act_request_timeout_s,
    )
