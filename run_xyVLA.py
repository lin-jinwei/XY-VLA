from __future__ import annotations

import argparse
import datetime
import heapq
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import requests
import json_numpy

from algorithms.phase1 import run_navigation_phase1_and_phase2_topdown
from algorithms.phase3 import Phase3FeedbackZones, hide_feedback_spheres_for_recording, run_phase3_apply_feedback_colors, set_phase3_feedback_alpha
from algorithms.phase_recording import (
    configure_pybullet_gui_opengl_transparency,
    recording_experiment_folder_name,
    render_camera_rgb,
    set_recording_prefer_opengl,
    sharpen_topdown_portal_labels_rgb,
)
from algorithms.phase3_xvla_actions import (
    capture_workspace_topdown_rgb,
    classify_mission_basic_actions_early,
    collect_specified_objects_from_mission,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "xVLAModel"
CONFIG_DEFAULT_PATH = ROOT / "config.json"

_CONFIG_STRUCT_KEYS = frozenset(
    {
        "schemes",
        "scheme",
        "presets",
        "preset",
        "profiles",
        "profile",
        "common",
        "comment",
        "description",
        "_comment",
        "readme_scheme",
        "_readme_scheme_switch",
        "widowx_scheme",
    }
)


def read_config_json(path: Path | None = None) -> dict[str, Any]:

    p = path or CONFIG_DEFAULT_PATH
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"WARNING: Could not load config {p}: {exc}", file=sys.stderr)
        return {}


def load_qs_policies(path: Path) -> list[dict[str, str]]:

    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[vvla] could not read {path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, str]] = []
    for item in data:
        if isinstance(item, dict) and "Q" in item and "S" in item:
            out.append({"Q": str(item["Q"]).strip(), "S": str(item["S"]).strip()})
    return out


def _sanitize_policy_indices(raw: list[Any], n_policies: int) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for x in raw:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_policies and i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


_QS_OFFLINE_STOP: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "at", "for", "into",
    "with", "without", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those", "each", "any", "not", "no", "only", "just", "then",
    "when", "what", "which", "who", "how", "can", "could", "should", "would", "may", "might",
    "must", "will", "shall", "use", "using", "have", "has", "had", "does", "did", "doing", "done",
    "such", "than", "about", "also", "there", "here", "where", "why", "both", "once", "after",
    "before", "during", "given", "give", "gives", "but", "asks", "asks", "ask", "need", "needs",
    "instruction", "instructions", "drone", "scene", "object", "objects", "specific", "concrete",
    "provide", "provides", "information", "task", "whether", "while", "them", "they", "their",
    "one", "two", "first", "every", "all", "some", "many", "most", "more", "less", "very",
    "other", "another", "yet", "still", "even", "away", "back", "long", "longer", "high",
    "higher", "low", "same", "own", "its", "does", "say", "says", "say", "whether",
})
_QS_OFFLINE_SHORT_OK: frozenset[str] = frozenset({
    "map", "all", "patrol", "fire", "path", "gas", "gps", "lap", "laps",
    "wait", "home", "rise", "slot", "gate", "rth", "uav",
})

_QS_OFFLINE_PHRASES: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("navigate", "go to", "goto", "waypoint", "destination", "navigation")),
    (1, ("pass through", "fly through", " go through", "through the", "through a")),
    (2, ("trajectory", "flight path", "path in", "altitude", "height ")),
    (3, ("no location", "no target", "unspecified", "explore", "global map", "where to")),
    (4, ("speed", "velocity", "faster", "slower")),
    (5, ("every object", "all objects", "inspect all", "visit every", "traverse")),
    (6, ("patrol", "patrolling", "surveil", "surveillance")),
    (7, ("search for", "locate", "find the", "find a", "first marker", "find one")),
    (8, ("find all", "every marker", "all markers", "enumerate")),
    (9, ("fire", "firefighting", "extinguish", "hose", "blaze")),
    (10, ("obstacle avoidance", "avoid obstacles", "no obstacle requirement", "collision")),
    (11, ("deliver", "delivery", "courier", "pickup", "pick up", "drop off", "dropoff", "parcel")),
    (12, ("rectangular", "yellow frame", "rectangular frame", "portal", "slot", "gate", "the yellow", "the red", "the blue", "through the")),
    (13, ("safely", "carefully", "smoothly", "slowly", "harsh")),
    (14, ("drone", "uav", "quadcopter", "flying robot", "autonomous flight")),
    (15, ("hover", "hold position", "pause", "loiter")),
    (16, ("approach the", "land near", "beside", "dock with")),
    (17, ("return to", "go home", "launch", "initial pose")),
    (18, ("orbit", "circle", "loop around")),
    (19, ("ascend", "climb", "rise", "gain altitude")),
    (20, ("descend", "lower altitude", "go down")),
    (21, ("inspect", "follow", "track")),
    (22, ("only the", "the yellow", "the red", "the green", "the orange")),
    (23, ("verify", "centered", "in camera")),
    (
        24,
        (
            "figure eight",
            "figure-eight",
            "pattern flight",
            "smooth circuit",
            "racetrack",
            "oval",
            "ellipse",
            "diamond",
            "in the air",
            "in the sky",
            "above the blocks",
            "over the workspace",
            "demonstration",
        ),
    ),
)


def _qs_policy_is_nonportal_scenic_airshow(p: dict[str, str]) -> bool:

    return "non-portal scenic" in str(p.get("S", "")).lower()


def _offline_user_cmd_requires_portal_flight(cmd: str) -> bool:

    u = (cmd or "").strip().lower()
    if not u:
        return False
    if any(h in u for h in ("fly through", "pass through", "go through", "move through")):
        return True
    if ("through" in u or "traverse" in u) and any(
        k in u for k in ("frame", "portal", "gate", "slot")
    ):
        return True
    if "opening" in u and ("frame" in u or "portal" in u or "gate" in u or "slot" in u):
        return True
    if "opening" in u and "rectangular" in u:
        return True
    return False


def _strip_conflicting_nonportal_scenic_policies(
    user_cmd: str,
    policies: list[dict[str, str]],
    indices: list[int],
) -> list[int]:

    if not indices:
        return indices
    if not _offline_user_cmd_requires_portal_flight(user_cmd):
        return indices
    return [i for i in indices if not _qs_policy_is_nonportal_scenic_airshow(policies[i])]


def _qs_offline_sig_tokens(text: str) -> set[str]:
    toks = set(re.findall(r"[a-z][a-z0-9]{2,}", text.lower()))
    out: set[str] = set()
    for t in toks:
        if t in _QS_OFFLINE_STOP:
            continue
        if len(t) >= 4 or t in _QS_OFFLINE_SHORT_OK:
            out.add(t)
    return out


def offline_match_qs_policy_indices(user_cmd: str, policies: list[dict[str, str]]) -> list[int]:

    u0 = user_cmd.strip()
    if not u0:
        return []
    u = u0.lower()
    u_tok = _qs_offline_sig_tokens(u0)
    hits: list[int] = []
    for i, p in enumerate(policies):
        q_raw = p.get("Q", "")
        q_tok = _qs_offline_sig_tokens(q_raw)
        if u_tok & q_tok:
            hits.append(i)
            continue
        hit = False
        for t in q_tok:
            if len(t) >= 4 and t in u:
                hit = True
                break
        if hit:
            hits.append(i)
            continue
    extra: list[int] = []
    for idx, phrases in _QS_OFFLINE_PHRASES:
        if 0 <= idx < len(policies) and any(ph in u for ph in phrases):
            extra.append(idx)
    seen: set[int] = set()
    out: list[int] = []
    for i in hits + extra:
        if 0 <= i < len(policies) and i not in seen:
            seen.add(i)
            out.append(i)
    return _strip_conflicting_nonportal_scenic_policies(u0, policies, out)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        text = "\n".join(lines).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON object in model response")
    return json.loads(m.group(0))


def _vvla_http_match(
    url: str,
    user_cmd: str,
    policies: list[dict[str, str]],
    *,
    timeout: float = 120.0,
) -> list[int]:

    payload = {"user_command": user_cmd, "policies": policies}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    idx = data.get("indices")
    if idx is None:
        idx = data.get("match_indices")
    if idx is None:
        idx = data.get("matches")
    if idx is None:
        return []
    if not isinstance(idx, list):
        return []
    return [int(x) for x in idx]


def _vvla_chat_match(
    user_cmd: str,
    policies: list[dict[str, str]],
    *,
    api_url: str,
    model: str,
    api_key: str,
    timeout: float = 120.0,
) -> list[int]:

    lines = [f"{i}: {p['Q']}" for i, p in enumerate(policies)]
    user_block = (
        "User drone / robot command (any language):\n"
        f"{user_cmd}\n\n"
        "Indexed situation list (when Q applies to the command, include that index):\n"
        + "\n".join(lines)
        + '\n\nReply with ONLY JSON: {"indices":[...]} — use [] if none apply.'
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You reply with only a JSON object, no markdown."},
            {"role": "user", "content": user_block},
        ],
        "temperature": 0,
    }
    r = requests.post(api_url, json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    resp = r.json()
    choices = resp.get("choices")
    if not choices:
        raise RuntimeError(f"chat completions: no choices in {str(resp)[:400]}")
    content = choices[0].get("message", {}).get("content", "")
    obj = _extract_json_object(str(content))
    idx = obj.get("indices")
    if idx is None:
        return []
    if not isinstance(idx, list):
        return []
    return [int(x) for x in idx]


def vvla_select_policy_indices(
    user_cmd: str,
    policies: list[dict[str, str]],
    *,
    vvla_http_url: str | None = None,
    chat_url: str | None = None,
    chat_model: str = "gpt-4o-mini",
    chat_api_key: str | None = None,
    timeout_http: float = 120.0,
    timeout_chat: float = 120.0,
) -> list[int]:

    n = len(policies)
    if n == 0:
        return []
    if vvla_http_url:
        raw = _vvla_http_match(
            vvla_http_url, user_cmd, policies, timeout=timeout_http
        )
        return _strip_conflicting_nonportal_scenic_policies(
            user_cmd, policies, _sanitize_policy_indices(raw, n)
        )
    if chat_url and chat_api_key:
        raw = _vvla_chat_match(
            user_cmd,
            policies,
            api_url=chat_url,
            model=chat_model,
            api_key=chat_api_key,
            timeout=timeout_chat,
        )
        return _strip_conflicting_nonportal_scenic_policies(
            user_cmd, policies, _sanitize_policy_indices(raw, n)
        )
    off = offline_match_qs_policy_indices(user_cmd, policies)
    if off:
        print(
            f"[vvla] offline keyword/phrase QS match (no HTTP/chat endpoint): indices {off}"
        )
    else:
        print(
            "[vvla] no HTTP/chat v-vla and offline QS match found no row; "
            "using raw --cmd. Optional: --vvla-url or chat API keys.",
            file=sys.stderr,
        )
    return off


def enrich_cmd_with_qs_policies(
    user_cmd: str,
    policies: list[dict[str, str]],
    indices: list[int],
) -> str:

    if not indices:
        return user_cmd
    parts_lines = [user_cmd.strip(), "", "Additional flight-policy constraints:", ""]
    for i in indices:
        if 0 <= i < len(policies):
            parts_lines.append(f"- {policies[i]['S']}")
    return "\n".join(parts_lines)


@dataclass
class InferenceRoutePlan:


    mode: str = "blend"
    w_xvla: float = 0.5
    w_local_kb: float = 0.5
    qs_attachment_frac: float = 1.0
    prefer_scenic_geometry: bool = True
    xvla_trajectory_once: bool | None = None
    infer_every_factor: float = 1.0
    source: str = "heuristic"


def _clamp_unit_interval(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _trim_qs_indices(idxs: list[int], frac: float) -> list[int]:

    if not idxs:
        return []
    if frac >= 1.0 - 1e-9:
        return list(idxs)
    if frac <= 0.0:
        return []
    k = max(0, int(math.ceil(len(idxs) * float(frac))))
    return idxs[:k]


def _normalize_route_dict(obj: dict[str, Any]) -> InferenceRoutePlan:
    mode = str(obj.get("mode", "blend")).strip().lower()
    if mode not in ("xvla", "local", "blend"):
        mode = "blend"
    wx = _clamp_unit_interval(float(obj.get("w_xvla", 0.5)))
    wl = _clamp_unit_interval(float(obj.get("w_local_kb", 0.5)))
    s = wx + wl
    if s > 1e-6:
        wx, wl = wx / s, wl / s
    else:
        wx, wl = 0.5, 0.5
    qs_f = _clamp_unit_interval(float(obj.get("qs_attachment_frac", 1.0)))
    raw_ps = obj.get("prefer_scenic_geometry")
    if raw_ps is None:
        prefer = wl >= 0.45
    else:
        prefer = bool(raw_ps)
    xonce_raw = obj.get("xvla_trajectory_once")
    xv_once: bool | None
    if xonce_raw is None or (
        isinstance(xonce_raw, str) and str(xonce_raw).strip().lower() in ("null", "none", "")
    ):
        xv_once = None
    else:
        xv_once = bool(xonce_raw)
    inf_f = float(obj.get("infer_every_factor", 1.0))
    inf_f = float(min(3.0, max(0.25, inf_f)))
    return InferenceRoutePlan(
        mode=mode,
        w_xvla=wx,
        w_local_kb=wl,
        qs_attachment_frac=qs_f,
        prefer_scenic_geometry=prefer,
        xvla_trajectory_once=xv_once,
        infer_every_factor=inf_f,
        source=str(obj.get("_source", "chat")),
    )


def xvla_inference_router_chat(
    user_cmd: str,
    *,
    api_url: str,
    model: str,
    api_key: str,
    timeout: float = 60.0,
) -> InferenceRoutePlan:

    user_block = (
        "You decide how to run a drone demo that blends:\n"
        "(A) Neural X-VLA policy via /act (vision + language, repeated every infer_every steps).\n"
        "(B) Local QS.json expert constraints plus geometric scenic planners "
        "(lawnmower/raster, figure-eight, orbit, spiral, zigzag, etc.).\n\n"
        "User command:\n"
        f"{user_cmd.strip()}\n\n"
        "Reply with ONLY a JSON object (no markdown, no prose):\n"
        "{\n"
        '  "mode": "xvla" | "local" | "blend",\n'
        '  "w_xvla": float in [0,1],\n'
        '  "w_local_kb": float in [0,1],\n'
        '  "qs_attachment_frac": float in [0,1],\n'
        '  "prefer_scenic_geometry": boolean,\n'
        '  "xvla_trajectory_once": boolean | null,\n'
        '  "infer_every_factor": float in [0.25, 3.0]\n'
        "}\n\n"
        "Guidelines:\n"
        '- Use mode \"local\" when the command is mainly a named aerial pattern / coverage path '
        "where templates excel (raster scan, lawnmower, figure-eight, orbit, spiral, zigzag).\n"
        '- Use mode \"xvla\" when the command needs visual grounding or dexterity: pick/place, '
        'portals / pass-through slots, obstacles, vague goals.\n'
        '- Use mode \"blend\" when both matter; keep w_xvla + w_local_kb summing to ~1.\n'
        "- qs_attachment_frac: fraction of matched QS constraint rows to attach (lower if policies "
        "might contradict the user).\n"
        "- prefer_scenic_geometry: true when scenic XY planning should run for pattern verbs.\n"
        '- xvla_trajectory_once: null to defer to runtime defaults; false for reactive /act; '
        "true for one-shot scenic polyline playback when applicable.\n"
        "- infer_every_factor: multiply CLI --infer-every; values <1 call /act more often; >1 less.\n"
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You reply with only a JSON object."},
            {"role": "user", "content": user_block},
        ],
        "temperature": 0,
    }
    r = requests.post(api_url, json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    resp = r.json()
    choices = resp.get("choices")
    if not choices:
        raise RuntimeError(f"router chat: no choices ({str(resp)[:400]})")
    content = choices[0].get("message", {}).get("content", "")
    obj = _extract_json_object(str(content))
    obj["_source"] = "chat"
    return _normalize_route_dict(obj)


def heuristic_inference_route(user_cmd: str) -> InferenceRoutePlan:

    uc = user_cmd.strip()
    if not uc:
        return InferenceRoutePlan(source="heuristic")
    s = uc.lower()
    scenic = (
        _language_cmd_requests_serpentine_scan(uc)
        or _language_cmd_requests_figure8(uc)
        or _language_cmd_requests_circle_orbit_uav(uc)
        or _language_cmd_requests_oval_racetrack(uc)
        or _language_cmd_requests_spiral(uc)
        or _language_cmd_requests_cross_axis_shuttle(uc)
        or _language_cmd_requests_shuttle_line_xy(uc)
    )
    portal = bool(re.search(r"\b(pass|fly|go)\s+through\b", s))
    manip = any(
        k in s
        for k in (
            "pick up",
            "pickup",
            "grasp",
            "grab",
            "place the",
            "stack",
            "push the",
            "pull the",
        )
    )
    if scenic and not portal:
        return InferenceRoutePlan(
            mode="local",
            w_xvla=0.35,
            w_local_kb=0.65,
            qs_attachment_frac=0.75,
            prefer_scenic_geometry=True,
            xvla_trajectory_once=True,
            infer_every_factor=1.15,
            source="heuristic",
        )
    if portal or manip:
        return InferenceRoutePlan(
            mode="xvla",
            w_xvla=0.72,
            w_local_kb=0.28,
            qs_attachment_frac=1.0,
            prefer_scenic_geometry=False,
            xvla_trajectory_once=False,
            infer_every_factor=0.88,
            source="heuristic",
        )
    return InferenceRoutePlan(
        mode="blend",
        w_xvla=0.5,
        w_local_kb=0.5,
        qs_attachment_frac=0.9,
        prefer_scenic_geometry=True,
        xvla_trajectory_once=None,
        infer_every_factor=1.0,
        source="heuristic",
    )


def resolve_inference_route(
    user_cmd: str,
    *,
    enabled: bool,
    heuristic_only: bool,
    chat_url: str | None,
    chat_model: str,
    chat_key: str | None,
    router_model: str | None,
    timeout: float,
) -> InferenceRoutePlan | None:
    if not enabled or not str(user_cmd).strip():
        return None
    model_eff = (router_model or "").strip() or chat_model
    if (
        (not heuristic_only)
        and chat_url
        and chat_key
        and str(chat_url).strip()
    ):
        try:
            return xvla_inference_router_chat(
                user_cmd,
                api_url=str(chat_url).strip(),
                model=model_eff,
                api_key=str(chat_key),
                timeout=float(timeout),
            )
        except Exception as exc:
            print(
                f"[xvla-router] chat routing failed ({exc}); using heuristic fallback",
                file=sys.stderr,
            )
    return heuristic_inference_route(user_cmd)


def _normalize_scheme_dict_key(sel: Any) -> str:

    if isinstance(sel, str) and sel.strip().isdigit():
        return sel.strip()
    if isinstance(sel, (int, float)) and float(sel).is_integer():
        return str(int(sel))
    return str(sel).strip()


def merge_widowx_scheme_config(raw: dict[str, Any]) -> dict[str, Any]:

    if not raw:
        return {}

    scheme_map = raw.get("schemes") or raw.get("presets") or raw.get("profiles")
    if not isinstance(scheme_map, dict) or not scheme_map:
        return {k: v for k, v in raw.items() if k not in _CONFIG_STRUCT_KEYS}

    overlay = {k: v for k, v in raw.items() if k not in _CONFIG_STRUCT_KEYS}
    common = raw.get("common")
    if not isinstance(common, dict):
        common = {}

    sel = raw.get("scheme")
    if sel is None:
        sel = raw.get("preset", raw.get("profile"))
    if sel is None:
        sel = "widowx_ee6d"

    user_key = _normalize_scheme_dict_key(sel)
    chosen_user = scheme_map.get(user_key)

    if isinstance(chosen_user, dict) and "workspace_lo" in chosen_user:
        base_key = user_key
        perf_key = None
    else:
        perf_key = user_key if user_key in scheme_map else None
        legacy = raw.get("widowx_scheme", "widowx_ee6d")
        lk = _normalize_scheme_dict_key(legacy)
        base_try = scheme_map.get(lk)
        if isinstance(base_try, dict) and "workspace_lo" in base_try:
            base_key = lk
        else:
            base_key = "widowx_ee6d"

    base = scheme_map.get(base_key)
    if base is None:
        valid = ", ".join(sorted(scheme_map.keys()))
        raise ValueError(
            f'config.json: widowx workspace scheme {base_key!r} not found under "schemes". Valid keys: {valid}'
        )
    if not isinstance(base, dict):
        raise ValueError(f'config.json: schemes[{base_key!r}] must be a JSON object')

    merged = {**common, **base}

    if perf_key is not None and perf_key != base_key:
        perf = scheme_map.get(perf_key)
        if isinstance(perf, dict):
            _geo = frozenset({
                "workspace_lo",
                "workspace_hi",
                "camera_eye",
                "camera_look_at",
                "cubes",
                "task_sequence",
                "record_every",
                "sim_steps",
                "label",
                "_label",
            })
            perf_clean = {k: v for k, v in perf.items() if k not in _geo}
            merged = {**merged, **perf_clean}

    merged = {**merged, **overlay}
    merged["_resolved_scheme"] = str(base_key) if not perf_key else f"{base_key}|runtime={perf_key}"
    base_lbl = base.get("label", base.get("_label", ""))
    if perf_key and perf_key != base_key:
        perf_d = scheme_map.get(perf_key, {})
        perf_lbl = perf_d.get("label", "") if isinstance(perf_d, dict) else ""
        merged["_resolved_scheme_label"] = f"{base_lbl} · runtime {perf_key}: {perf_lbl}".strip(" ·:")
    else:
        merged["_resolved_scheme_label"] = str(base_lbl)
    return merged


def load_widowx_demo_config(
    path: Path | None = None, scheme_override: int | str | None = None
) -> dict[str, Any]:
    raw = read_config_json(path)
    if scheme_override is not None:
        raw = {**raw, "scheme": scheme_override}
    return merge_widowx_scheme_config(raw)


def config_infer_every(merged: dict[str, Any], fallback: int = 4) -> int:

    if merged.get("infer_every") is not None:
        return max(1, int(merged["infer_every"]))
    if merged.get("xvla_infer_gap") is not None:
        return max(1, int(merged["xvla_infer_gap"]))
    return max(1, int(fallback))


def apply_auto_motion_scales(
    *,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    infer_every: int,
    language_only: bool,
    pos_lerp_alpha: float,
    delta_pos_scale: float,
    ref_diag_m: float = 0.55,
) -> tuple[float, float]:

    extent = (np.asarray(workspace_hi, dtype=np.float64) - np.asarray(workspace_lo, dtype=np.float64)).reshape(
        3,
    )
    diag = float(np.linalg.norm(extent))
    k_ws = float(np.clip(diag / max(float(ref_diag_m), 1e-6), 0.62, 1.9))
    ie = max(1, int(infer_every))
    k_infer = float(np.clip(0.40 + 0.10 * float(ie - 1), 0.42, 2.05))
    k_task = 1.38 if language_only else 1.0

    pl0 = float(pos_lerp_alpha)
    pl = pl0 * k_ws * k_infer * k_task
    pl = float(np.clip(pl, 0.06, 0.88))

    ds0 = float(delta_pos_scale)
    ds = ds0 * k_ws * (1.12 if language_only else 1.0)
    ds = float(np.clip(ds, 0.016, 0.16))
    return pl, ds


def require_pybullet():
    try:
        import pybullet as p
        import pybullet_data
    except Exception as exc:
        raise RuntimeError(
            "pybullet is not installed. Run: conda run -n xy-vla pip install pybullet"
        ) from exc
    return p, pybullet_data


def ensure_xvla_server(server_url: str, timeout: float = 3.0) -> None:
    try:
        r = requests.get(server_url.replace("/act", "/docs"), timeout=timeout)
        if r.status_code < 400:
            return
    except Exception:
        pass
    raise RuntimeError(
        "X-VLA server is not reachable.\n"
        "If this URL should be local, set auto_start_xvla_server to true in config.json (default) "
        "and use 127.0.0.1; or start the service manually:\n"
        "python -c \"from transformers import AutoModel, AutoProcessor; "
        f"m=AutoModel.from_pretrained(r'{MODEL_DIR}', trust_remote_code=True, local_files_only=True); "
        f"p=AutoProcessor.from_pretrained(r'{MODEL_DIR}', trust_remote_code=True, local_files_only=True); "
        "m.run(p, host='127.0.0.1', port=8000)\"\n"
        "To skip auto-start and fail fast, use --no-auto-server or set auto_start_xvla_server to false."
    )


def xvla_server_reachable(server_url: str, timeout: float = 3.0) -> bool:
    try:
        r = requests.get(server_url.replace("/act", "/docs"), timeout=timeout)
        return r.status_code < 400
    except Exception:
        return False


def parse_act_url(server_url: str) -> tuple[str, int]:
    u = urlparse(server_url)
    host = u.hostname or "127.0.0.1"
    if u.port is not None:
        return host, int(u.port)
    if (u.scheme or "").lower() == "https":
        return host, 443
    return host, 8000


def is_loopback_act_host(host: str) -> bool:

    h = (host or "").strip().lower()
    if h in ("localhost", "::1", "0.0.0.0"):
        return True
    if h == "127.0.0.1":
        return True
    return h.startswith("127.")


def wait_port_open(host: str, port: int, timeout_s: float = 180.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def start_local_xvla_server(
    host: str,
    port: int,
    *,
    access_log: bool = False,
    log_level: str = "warning",
) -> subprocess.Popen:
    allowed = frozenset({"critical", "error", "warning", "info", "debug", "trace"})
    lv = str(log_level).strip().lower()
    if lv not in allowed:
        lv = "warning"
    acc_lit = "True" if access_log else "False"
    _suppress_hf_noise = (
        "import os,warnings; "
        "os.environ.setdefault('TRANSFORMERS_VERBOSITY','error'); "
        "import transformers as _tr; "
        "_tr.logging.set_verbosity_error(); "
        "warnings.filterwarnings('ignore',message='.*GenerationMixin.*'); "
        "warnings.filterwarnings('ignore',message='.*slow image processor.*'); "
        "warnings.filterwarnings('ignore',message='.*use_fast.*'); "
    )
    code = _suppress_hf_noise + (
        "from transformers import AutoModel, AutoProcessor; "
        f"mpath=r'{MODEL_DIR}'; "
        "m=AutoModel.from_pretrained(mpath, trust_remote_code=True, local_files_only=True); "
        "p=AutoProcessor.from_pretrained(mpath, trust_remote_code=True, local_files_only=True); "
        f"m.run(p, host='{host}', port={port}, access_log={acc_lit}, log_level='{lv}')"
    )
    proc = subprocess.Popen([sys.executable, "-u", "-c", code])
    if not wait_port_open(host, port, timeout_s=180.0):
        proc.terminate()
        raise RuntimeError("X-VLA server failed to start in time (model load / bind port).")
    return proc


def format_xyz_for_overlay(vec: Any, *, decimals: int = 6) -> str:

    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = max(0, int(decimals))
    fmt = f"{{:.{n}f}}"
    xs = [fmt.format(float(v[i])) if i < v.size else fmt.format(0.0) for i in range(3)]
    return "[" + ",".join(xs) + "]"


def frame_with_overlay(rgb: np.ndarray, line1: str, line2: str = "") -> np.ndarray:
    try:
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr,
            line1,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if line2:
            cv2.putText(
                bgr,
                line2,
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 200, 100),
                2,
                cv2.LINE_AA,
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return rgb


def save_navigation_video(frames: list[np.ndarray], out_path: Path, fps: float) -> None:
    if not frames:
        print("[record] No frames captured, skip save.")
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    mp4_errors: list[str] = []

    def try_imageio_mp4(path: Path) -> bool:
        try:
            import imageio.v2 as imageio

            imageio.mimsave(str(path), frames, fps=fps, codec="libx264", quality=8)
            return True
        except Exception as exc:
            mp4_errors.append(f"imageio: {exc}")
            return False

    def try_opencv_mp4(path: Path) -> bool:
        try:
            import cv2

            h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(path), fourcc, float(max(fps, 1e-3)), (w, h))
            if not vw.isOpened():
                mp4_errors.append("OpenCV VideoWriter: writer not opened (missing codec?)")
                return False
            for fr in frames:
                bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                vw.write(bgr)
            vw.release()
            return True
        except Exception as exc:
            mp4_errors.append(f"OpenCV VideoWriter: {exc}")
            return False

    def try_imageio_gif(path: Path) -> bool:
        try:
            import imageio.v2 as imageio

            imageio.mimsave(str(path), frames, fps=min(fps, 50))
            return True
        except Exception:
            return False

    def try_pillow_gif(path: Path) -> bool:
        try:
            from PIL import Image

            duration_ms = int(max(1000 / max(fps, 1e-3), 20))
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(
                str(path),
                save_all=True,
                append_images=imgs[1:],
                duration=duration_ms,
                loop=0,
            )
            return True
        except Exception:
            return False

    ok = False
    if suffix == ".mp4":
        ok = try_opencv_mp4(out_path)
        if not ok:
            ok = try_imageio_mp4(out_path)
        if not ok:
            for msg in mp4_errors:
                print(f"[record] MP4 failed: {msg}", file=sys.stderr)
            print(
                "[record] MP4 export failed (often missing ffmpeg for imageio, or codec issue). "
                "Saving GIF instead.",
                file=sys.stderr,
            )
            out_path = out_path.with_suffix(".gif")

    if out_path.suffix.lower() == ".gif":
        ok = try_imageio_gif(out_path) or try_pillow_gif(out_path)

    out_abs = out_path.resolve()
    if ok:
        print(f"[record] saved {len(frames)} frames -> {out_abs}")
    else:
        raise RuntimeError(
            "Cannot save recording. For GIF: imageio or Pillow. "
            "For MP4: OpenCV (mp4v) or imageio with ffmpeg on PATH."
        )


def resolve_recording_folder(
    base_dir: str,
    *,
    prefix: str = "",
    append_timestamp: bool = True,
) -> Path:

    p = Path(base_dir)
    if not p.is_absolute():
        p = ROOT / p
    ts = recording_experiment_folder_name()
    prefix = str(prefix).strip()
    if append_timestamp:
        folder_name = f"{prefix}_{ts}" if prefix else ts
    else:
        folder_name = prefix if prefix else ts
    return p / folder_name


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:

    R = np.asarray(R, dtype=np.float32)
    c1 = R[:, 0]
    c2 = R[:, 1]
    v6 = np.empty(6, dtype=np.float32)
    v6[0::2] = c1
    v6[1::2] = c2
    return v6


def rot6d_to_matrix(v6: np.ndarray) -> np.ndarray:

    v6 = np.asarray(v6, dtype=np.float32)
    a1 = v6[0:5:2]
    a2 = v6[1:6:2]
    n1 = np.linalg.norm(a1) + 1e-8
    b1 = a1 / n1
    proj = float(np.dot(b1, a2)) * b1
    a2p = a2 - proj
    n2 = np.linalg.norm(a2p) + 1e-8
    b2 = a2p / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def euler_xyz_from_matrix(R: np.ndarray) -> tuple[float, float, float]:

    R = np.asarray(R, dtype=np.float64)
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = float(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return roll, pitch, yaw


def matrix_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = float(np.cos(roll)), float(np.sin(roll))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


DRONE_BODY_HALF = 0.04
DRONE_ROTOR_RADIUS = 0.018
DRONE_ARM_LENGTH = 0.07
DRONE_MAX_DIAGONAL = float(
    np.linalg.norm(
        [
            2.0 * (DRONE_ARM_LENGTH + DRONE_ROTOR_RADIUS),
            2.0 * (DRONE_ARM_LENGTH + DRONE_ROTOR_RADIUS),
            2.0 * DRONE_BODY_HALF,
        ]
    )
)
DEFAULT_DRONE_START_OFFSET_TOPVIEW_UP_M = 2.0
DEFAULT_FRAME_THICKNESS = DRONE_MAX_DIAGONAL / 10.0
DEFAULT_FRAME_SIDE = DRONE_MAX_DIAGONAL * 15.0 / 10.0
RECT_FRAME_SLOT_WIDTH_SCALE = 0.9
RECT_FRAME_MAX_TILT_DEG = 30.0
RECT_FRAME_MAX_TILT_RAD = float(np.deg2rad(RECT_FRAME_MAX_TILT_DEG))


def _portal_frame_quaternion_bullet(p, *, tilt_rad: float, yaw_rad: float) -> list[float]:

    q_rx = p.getQuaternionFromEuler([float(tilt_rad), 0.0, 0.0])
    q_rz = p.getQuaternionFromEuler([0.0, 0.0, float(yaw_rad)])
    _pos_unused, orn = p.multiplyTransforms([0.0, 0.0, 0.0], q_rz, [0.0, 0.0, 0.0], q_rx)
    return list(orn)


def _world_corners_rotated_box(
    p,
    frame_center: np.ndarray,
    orn: list[float],
    offset_local: np.ndarray,
    half_extents: tuple[float, float, float],
) -> np.ndarray:
    hx, hy, hz = (float(half_extents[0]), float(half_extents[1]), float(half_extents[2]))
    base = np.asarray(offset_local, dtype=np.float64).reshape(3)
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                lc = base + np.array([sx * hx, sy * hy, sz * hz], dtype=np.float64)
                rotated = np.asarray(p.rotateVector(orn, lc.tolist()), dtype=np.float64)
                pts.append(frame_center.reshape(3) + rotated)
    return np.vstack(pts)


def create_tilted_rectangular_portal_frame(
    p,
    pos_xyz,
    *,
    long_side: float,
    thickness: float,
    depth: float | None = None,
    rgba: list[float] | None = None,
    short_side: float | None = None,
    tilt_rad: float | None = None,
    yaw_rad: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, float, np.ndarray]:

    pos = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
    color = list(rgba if rgba is not None else [0.8, 0.8, 0.8, 1.0])
    L = float(long_side)
    t = float(thickness)
    d = float(depth if depth is not None else thickness)
    W = float(short_side if short_side is not None else RECT_FRAME_SLOT_WIDTH_SCALE * DRONE_MAX_DIAGONAL)
    rng = rng if rng is not None else np.random.default_rng()
    if tilt_rad is None:
        tilt_rad = float(rng.uniform(0.0, RECT_FRAME_MAX_TILT_RAD))
    if yaw_rad is None:
        yaw_rad = float(rng.uniform(0.0, 2.0 * np.pi))
    tilt_rad = float(np.clip(tilt_rad, 0.0, RECT_FRAME_MAX_TILT_RAD))

    W_eff = float(min(W, max(L - 2.0 * t, t * 0.25)))

    Ly = 0.5 * L
    Wz = 0.5 * W_eff
    weld = float(max(t * 0.05, 1e-9))
    tw = float(t + weld)
    hz_vert = float(Wz + weld)
    hy_horiz = float(Ly + weld)

    bars_local: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
        ((d * 0.5, hy_horiz, tw * 0.5), (0.0, 0.0, Wz - tw * 0.5)),
        ((d * 0.5, hy_horiz, tw * 0.5), (0.0, 0.0, -Wz + tw * 0.5)),
        ((d * 0.5, tw * 0.5, hz_vert), (0.0, Ly - tw * 0.5, 0.0)),
        ((d * 0.5, tw * 0.5, hz_vert), (0.0, -Ly + tw * 0.5, 0.0)),
    ]

    quat = _portal_frame_quaternion_bullet(p, tilt_rad=tilt_rad, yaw_rad=yaw_rad)

    all_corners: list[np.ndarray] = []
    for half_extents, offset_local in bars_local:
        hc = _world_corners_rotated_box(
            p, pos, quat, np.asarray(offset_local, dtype=np.float64), half_extents
        )
        all_corners.append(hc)
    corners = np.vstack(all_corners)
    bounds_half = np.max(np.abs(corners - pos.reshape(1, 3)), axis=0)

    for half_extents, offset_local in bars_local:
        off = np.asarray(offset_local, dtype=np.float64).reshape(3)
        center_world = pos + np.asarray(p.rotateVector(quat, off.tolist()), dtype=np.float64)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents))
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half_extents), rgbaColor=color)
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=center_world.tolist(),
            baseOrientation=quat,
        )

    return tilt_rad, yaw_rad, W_eff, L, bounds_half


_PORTAL_LABEL_TEX_VER = "v8"


def _portal_label_png_path(digit: int) -> Path:
    d = ROOT / ".cache" / "portal_num_labels"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"n{int(digit):02d}_{_PORTAL_LABEL_TEX_VER}.png"


def _ensure_portal_label_png(digit: int) -> Path:

    n = int(np.clip(digit, 1, 20))
    path = _portal_label_png_path(n)
    if path.is_file():
        return path
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV (cv2) required to generate portal label textures") from exc
    px = 2048
    bg_bgr = (250, 236, 220)
    img = np.full((px, px, 3), bg_bgr, dtype=np.uint8)
    text = str(n)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if n < 10:
        scale = 34.0
    else:
        scale = 25.0
    thickness = max(8, int(round(scale * 0.24)))
    (tw, th), _bl = cv2.getTextSize(text, font, scale, thickness)
    org = (px // 2 - tw // 2, px // 2 + th // 2)
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness, cv2.LINE_8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[gray < 200] = (0, 0, 0)
    img[gray >= 200] = bg_bgr
    img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.flip(img, 1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _ensure_portal_label_obj(digit: int) -> Path:

    n = int(np.clip(digit, 1, 20))
    _ensure_portal_label_png(n)
    d = _portal_label_png_path(n).parent
    obj_p = d / f"quad_{n:02d}.obj"
    mtl_p = d / f"quad_{n:02d}.mtl"
    png_name = f"n{n:02d}_{_PORTAL_LABEL_TEX_VER}.png"
    mtl_p.write_text(
        "newmtl Mat\n"
        "Ka 1 1 1\n"
        "Kd 1 1 1\n"
        f"map_Kd {png_name}\n",
        encoding="utf-8",
    )
    obj_p.write_text(
        f"mtllib {mtl_p.name}\n"
        "o quad\n"
        "usemtl Mat\n"
        "v -0.5 -0.5 0.0\n"
        "v 0.5 -0.5 0.0\n"
        "v 0.5 0.5 0.0\n"
        "v -0.5 0.5 0.0\n"
        "vt 0.0 1.0\n"
        "vt 1.0 1.0\n"
        "vt 1.0 0.0\n"
        "vt 0.0 0.0\n"
        "f 1/1 2/2 3/3\n"
        "f 1/1 3/3 4/4\n",
        encoding="utf-8",
    )
    return obj_p


def _add_rect_portal_number_labels(p, placed: list[dict]) -> None:

    portals = [it for it in placed if it.get("shape") == "rect_frame"]
    if not portals:
        return
    orn = [0.0, 0.0, 0.0, 1.0]
    z_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for idx, it in enumerate(portals[:20]):
        num = idx + 1
        pos = np.asarray(it["pos"], dtype=np.float64).reshape(3)
        bh = np.asarray(it["bounds_half"], dtype=np.float64).reshape(3)
        L = float(it["side"])
        W = float(it["short_side"])
        side_m = 2.0 * float(min(L, W)) / 3.0
        side_m = float(np.clip(side_m, 0.024, 2.5))
        margin = max(0.008, 0.06 * side_m)
        center = pos + z_up * (float(bh[2]) + 0.5 * side_m + margin + 0.002)
        obj_path = _ensure_portal_label_obj(num)
        try:
            vis = p.createVisualShape(
                p.GEOM_MESH,
                fileName=str(obj_path.resolve()),
                meshScale=[float(side_m), float(side_m), 1.0],
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
            )
        except Exception:
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[side_m * 0.5, side_m * 0.5, 0.002],
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
            )
        uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=center.tolist(),
            baseOrientation=orn,
        )
        it["portal_label"] = int(num)
        it["portal_label_uid"] = int(uid)
        it["portal_label_center"] = center.tolist()
        it["portal_label_side_m"] = float(side_m)


DEFAULT_CUBES = [
    {"color": "red",    "pos": [0.18,  0.22, 0.06], "rgba": [0.90, 0.20, 0.20, 1.0], "half": 0.025},
    {"color": "green",  "pos": [0.52, -0.22, 0.34], "rgba": [0.20, 0.85, 0.30, 1.0], "half": 0.025},
    {"color": "blue",   "pos": [0.52,  0.22, 0.10], "rgba": [0.20, 0.45, 0.95, 1.0], "half": 0.025},
    {"color": "yellow", "pos": [0.18, -0.22, 0.30], "rgba": [0.95, 0.85, 0.20, 1.0], "half": 0.025},
]


def build_scene(p, with_objects: bool, cubes: list[dict] | None = None) -> list[dict]:

    p.loadURDF("plane.urdf")
    if not with_objects:
        return []
    cubes = cubes if cubes is not None else DEFAULT_CUBES
    scene_ori_seed: int | None = None
    for c in cubes:
        if c.get("frame_orientation_seed") is not None:
            scene_ori_seed = int(c["frame_orientation_seed"])
            break
    portal_rng = np.random.default_rng(scene_ori_seed)

    placed: list[dict] = []
    for c in cubes:
        pos = list(c["pos"])
        rgba = list(c.get("rgba", [0.8, 0.8, 0.8, 1.0]))
        shape = str(c.get("shape", "cube")).strip().lower()
        if shape in ("square_frame", "frame", "rect_frame"):
            side = float(c.get("side", DEFAULT_FRAME_SIDE))
            thickness = float(c.get("thickness", DEFAULT_FRAME_THICKNESS))
            depth = float(c.get("depth", thickness))
            short_side_val = c.get("short_side")
            short_side_f = float(short_side_val) if short_side_val is not None else None
            slot_scale = c.get("slot_width_scale")
            if slot_scale is not None and short_side_f is None:
                short_side_f = float(slot_scale) * DRONE_MAX_DIAGONAL
            tilt_rad = np.deg2rad(float(c["tilt_deg"])) if c.get("tilt_deg") is not None else None
            yaw_rad = np.deg2rad(float(c["yaw_deg"])) if c.get("yaw_deg") is not None else None
            t_r, y_r, W_eff, L, bounds_half = create_tilted_rectangular_portal_frame(
                p,
                pos,
                long_side=side,
                thickness=thickness,
                depth=depth,
                rgba=rgba,
                short_side=short_side_f,
                tilt_rad=tilt_rad,
                yaw_rad=yaw_rad,
                rng=portal_rng,
            )
            placed.append(
                {
                    "color": c.get("color", "?"),
                    "pos": pos,
                    "rgba": rgba,
                    "shape": "rect_frame",
                    "side": float(L),
                    "short_side": float(W_eff),
                    "tilt_deg": float(np.rad2deg(t_r)),
                    "yaw_deg": float(np.rad2deg(y_r)),
                    "thickness": thickness,
                    "bounds_half": [float(bounds_half[0]), float(bounds_half[1]), float(bounds_half[2])],
                }
            )
            continue
        half = float(c.get("half", 0.025))
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=rgba)
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        placed.append({"color": c.get("color", "?"), "pos": pos, "rgba": rgba, "half": half})
    _add_rect_portal_number_labels(p, placed)
    return placed


def create_floating_drone(p, pos_xyz, half=0.04, rotor_r=0.018, arm=0.07):

    orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
    body_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[half, half, half * 0.5], rgbaColor=[0.15, 0.45, 0.92, 1.0]
    )
    body_uid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=body_vis, basePosition=list(pos_xyz))

    rotor_uids = []
    offsets = [(arm, arm), (arm, -arm), (-arm, arm), (-arm, -arm)]
    for dx, dy in offsets:
        r_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=rotor_r, rgbaColor=[0.95, 0.25, 0.15, 1.0]
        )
        uid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=r_vis)
        p.resetBasePositionAndOrientation(
            uid, [pos_xyz[0] + dx, pos_xyz[1] + dy, pos_xyz[2]], orn
        )
        rotor_uids.append(uid)
    return body_uid, rotor_uids, offsets


def update_floating_drone(p, body_uid, rotor_uids, offsets, pos_xyz, R_world):

    quat = p.getQuaternionFromEuler(list(euler_xyz_from_matrix(R_world)))
    p.resetBasePositionAndOrientation(body_uid, list(pos_xyz), quat)
    for uid, (dx, dy) in zip(rotor_uids, offsets):
        local = np.array([dx, dy, 0.0], dtype=np.float32)
        world = pos_xyz + R_world @ local
        p.resetBasePositionAndOrientation(uid, world.tolist(), quat)


def _quat_from_z_axis_to_vector(v: np.ndarray) -> list[float]:

    direction = np.asarray(v, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 0.999999:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -0.999999:
        return [1.0, 0.0, 0.0, 0.0]
    axis = np.cross(z_axis, direction)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    half_angle = 0.5 * float(np.arccos(dot))
    s = float(np.sin(half_angle))
    return [float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), float(np.cos(half_angle))]


def create_trail_segment(
    p,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    radius: float = 0.006,
    rgba: list[float] | None = None,
    min_length: float = 0.01,
) -> bool:

    start = np.asarray(start_xyz, dtype=np.float64)
    end = np.asarray(end_xyz, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < float(min_length):
        return False
    color = rgba if rgba is not None else [1.0, 0.68, 0.32, 0.75]
    mid = (start + end) * 0.5
    vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(radius),
        length=length,
        rgbaColor=list(color),
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vis,
        basePosition=mid.tolist(),
        baseOrientation=_quat_from_z_axis_to_vector(delta),
    )
    return True


def get_workspace_rgb(p, eye, look_at, width=256, height=256, fov=55.0):
    view = p.computeViewMatrix(cameraEyePosition=list(eye), cameraTargetPosition=list(look_at), cameraUpVector=[0, 0, 1])
    aspect = width / max(height, 1)
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=0.02, farVal=5.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width=width, height=height, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER,
    )
    return np.reshape(rgba, (height, width, 4))[..., :3].astype(np.uint8)


def get_drone_fpv_rgb(
    p,
    drone_pos: np.ndarray,
    R_body_world: np.ndarray,
    *,
    cam_offset_body: np.ndarray,
    cam_look_body: np.ndarray,
    width: int = 256,
    height: int = 256,
    fov: float = 72.0,
) -> np.ndarray:

    R = np.asarray(R_body_world, dtype=np.float64).reshape(3, 3)
    pos = np.asarray(drone_pos, dtype=np.float64).reshape(3)
    off = np.asarray(cam_offset_body, dtype=np.float64).reshape(3)
    look = np.asarray(cam_look_body, dtype=np.float64).reshape(3)
    eye_w = pos + R @ off
    target_w = pos + R @ (off + look)
    up_w = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    un = float(np.linalg.norm(up_w))
    if un < 1e-12:
        up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up_w /= un
    view = p.computeViewMatrix(
        cameraEyePosition=eye_w.tolist(),
        cameraTargetPosition=target_w.tolist(),
        cameraUpVector=up_w.tolist(),
    )
    aspect = width / max(height, 1)
    proj = p.computeProjectionMatrixFOV(fov=float(fov), aspect=aspect, nearVal=0.015, farVal=8.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    return np.reshape(rgba, (height, width, 4))[..., :3].astype(np.uint8)


def _match_rect_portal_placed(
    placed_cubes: list[dict], goal_xyz: np.ndarray, *, tol: float
) -> dict | None:
    g = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    best: dict | None = None
    best_d = 1e18
    for o in placed_cubes:
        if o.get("shape") != "rect_frame":
            continue
        c = np.asarray(o["pos"], dtype=np.float64).reshape(3)
        d = float(np.linalg.norm(c - g))
        if d < best_d:
            best_d = d
            best = o
    if best is None or best_d > float(tol):
        return None
    return best


def _portal_pass_spec_for_task(
    p,
    placed_cubes: list[dict],
    task_target_xyz: np.ndarray,
    drone_pos: np.ndarray,
    *,
    match_tol: float,
    approach_offset: float,
    exit_offset: float,
) -> dict[str, np.ndarray] | None:

    portal = _match_rect_portal_placed(placed_cubes, task_target_xyz, tol=match_tol)
    if portal is None:
        return None
    c = np.asarray(portal["pos"], dtype=np.float64).reshape(3)
    tilt_r = np.deg2rad(float(portal["tilt_deg"]))
    yaw_r = np.deg2rad(float(portal["yaw_deg"]))
    quat = _portal_frame_quaternion_bullet(p, tilt_rad=tilt_r, yaw_rad=yaw_r)
    nx = np.asarray(p.rotateVector(quat, [1.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    un = float(np.linalg.norm(nx))
    if un < 1e-12:
        return None
    nx /= un
    dp = np.asarray(drone_pos, dtype=np.float64).reshape(3) - c
    s = float(np.dot(nx, dp))
    if abs(s) < 1e-8:
        n_through = nx.copy()
    else:
        n_through = (-nx if s > 0.0 else nx)
    n_through = n_through / max(float(np.linalg.norm(n_through)), 1e-12)

    da = max(float(approach_offset), 0.15)
    de = max(float(exit_offset), 0.15)

    return {
        "center": c.astype(np.float32),
        "n_through": n_through.astype(np.float32),
        "approach": (c - n_through * da).astype(np.float32),
        "exit": (c + n_through * de).astype(np.float32),
    }


def build_drone_R_aligned_to_rect_portal(
    p,
    gate_quat_xyzw: list[float],
    drone_pos: np.ndarray,
    gate_center: np.ndarray,
) -> np.ndarray:

    n0 = np.asarray(p.rotateVector(gate_quat_xyzw, [1.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    n0 /= max(float(np.linalg.norm(n0)), 1e-12)
    toward = np.asarray(gate_center, dtype=np.float64).reshape(3) - np.asarray(drone_pos, dtype=np.float64).reshape(
        3
    )
    if float(np.linalg.norm(toward)) >= 1e-6 and float(np.dot(n0, toward)) < 0.0:
        n0 = -n0
    long_w = np.asarray(p.rotateVector(gate_quat_xyzw, [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
    long_w = long_w - float(np.dot(long_w, n0)) * n0
    ln = float(np.linalg.norm(long_w))
    if ln < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        long_w = up - float(np.dot(up, n0)) * n0
        ln = float(np.linalg.norm(long_w))
    long_w /= max(ln, 1e-12)
    col2 = np.cross(n0, long_w)
    col2 /= max(float(np.linalg.norm(col2)), 1e-12)
    R = np.column_stack([n0, long_w, col2]).astype(np.float64)
    if float(np.linalg.det(R)) < 0.0:
        col2 = -col2
        R = np.column_stack([n0, long_w, col2])
    return R.astype(np.float32)


def build_drone_R_align_approach_and_inplane_long_twist(
    drone_pos: np.ndarray,
    gate_center: np.ndarray,
    *,
    twist_rad: float,
    world_up: np.ndarray | None = None,
) -> np.ndarray:

    gc = np.asarray(gate_center, dtype=np.float64).reshape(3)
    dp = np.asarray(drone_pos, dtype=np.float64).reshape(3)
    n0 = gc - dp
    un = float(np.linalg.norm(n0))
    if un < 1e-9:
        return np.eye(3, dtype=np.float32)
    n0 = n0 / un
    up = np.asarray(world_up if world_up is not None else [0.0, 0.0, 1.0], dtype=np.float64).reshape(3)
    long_w = up - float(np.dot(up, n0)) * n0
    ln = float(np.linalg.norm(long_w))
    if ln < 1e-8:
        orth = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        long_w = orth - float(np.dot(orth, n0)) * n0
        ln = float(np.linalg.norm(long_w))
    long_w /= max(ln, 1e-12)
    w = float(np.sin(twist_rad))
    c = float(np.cos(twist_rad))
    long_tw = long_w * c + np.cross(n0, long_w) * w
    long_tw /= max(float(np.linalg.norm(long_tw)), 1e-12)
    col2 = np.cross(n0, long_tw)
    col2 /= max(float(np.linalg.norm(col2)), 1e-12)
    R = np.column_stack([n0, long_tw, col2]).astype(np.float64)
    if float(np.linalg.det(R)) < 0.0:
        col2 = -col2
        R = np.column_stack([n0, long_tw, col2])
    return R.astype(np.float32)


def cv_estimate_colored_rect_long_edge_skew_rad(
    rgb_u8: np.ndarray,
    rgba01: list[float] | tuple[float, ...],
    *,
    rgb_pad: int = 45,
    min_area_ratio: float = 0.015,
    morph_px: int = 3,
) -> tuple[float | None, float]:

    try:
        import cv2
    except Exception:
        return None, 0.0
    if rgb_u8 is None or rgb_u8.size == 0:
        return None, 0.0
    h, w = rgb_u8.shape[:2]
    if h < 8 or w < 8:
        return None, 0.0
    mask = _portal_color_segment_mask_rgb(
        rgb_u8, rgba01, rgb_pad=int(rgb_pad), morph_px=int(morph_px)
    )
    contours, _h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    frac = area / float(max(h * w, 1))
    if frac < float(min_area_ratio):
        return None, frac
    rect = cv2.minAreaRect(cnt)
    (rw, rh), ang_deg = rect[1], float(rect[2])
    if rw < rh:
        rw, rh = rh, rw
        ang_deg = ang_deg - 90.0
    skew_rad = float(np.deg2rad(ang_deg))
    return skew_rad, frac


def estimate_rect_portal_pose_fpv(
    p,
    placed_cubes: list[dict],
    *,
    goal_world: np.ndarray,
    drone_pos: np.ndarray,
    drone_R: np.ndarray,
    match_tol: float,
    use_sim_truth: bool,
    cam_offset_body: np.ndarray,
    cam_look_body: np.ndarray,
    fpv_width: int,
    fpv_height: int,
    fpv_fov: float,
    gate_pose_use_dedicated_camera: bool = False,
    gate_pose_cam_offset_body: np.ndarray | None = None,
    gate_pose_cam_look_body: np.ndarray | None = None,
    gate_pose_cam_width: int | None = None,
    gate_pose_cam_height: int | None = None,
    gate_pose_cam_fov: float | None = None,
    gate_pose_cv_rgb_pad: int = 45,
    gate_pose_cv_min_area_ratio: float = 0.015,
    gate_pose_cv_fallback_map_pose: bool = True,
    gate_pose_estimator: str = "opencv",
    xvla_server_url: str | None = None,
    xvla_gate_instruction_template: str = (
        "Orient end-effector with long opening of rectangular portal billboard_id={billboard_id} ({color}); pass through."
    ),
    xvla_gate_steps: int = 4,
    xvla_gate_infer_width: int = 256,
    xvla_gate_infer_height: int = 256,
    proprio_20d: np.ndarray | None = None,
    decode_workspace_lo: np.ndarray | None = None,
    decode_workspace_hi: np.ndarray | None = None,
    decode_treat_pos_as: str = "absolute",
    decode_delta_pos_scale: float = 0.04,
    gate_pose_xvla_fallback_opencv: bool = True,
    xvla_act_request_timeout_s: float = 300.0,
) -> dict | None:
    portal = _match_rect_portal_placed(placed_cubes, goal_world, tol=match_tol)
    if portal is None:
        return None

    if gate_pose_use_dedicated_camera:
        off_b = np.asarray(gate_pose_cam_offset_body, dtype=np.float64).reshape(3)
        look_b = np.asarray(gate_pose_cam_look_body, dtype=np.float64).reshape(3)
        gw = int(gate_pose_cam_width if gate_pose_cam_width is not None else fpv_width)
        gh = int(gate_pose_cam_height if gate_pose_cam_height is not None else fpv_height)
        gfov = float(gate_pose_cam_fov if gate_pose_cam_fov is not None else fpv_fov)
    else:
        off_b = np.asarray(cam_offset_body, dtype=np.float64).reshape(3)
        look_b = np.asarray(cam_look_body, dtype=np.float64).reshape(3)
        gw, gh, gfov = int(fpv_width), int(fpv_height), float(fpv_fov)

    rgb = get_drone_fpv_rgb(
        p,
        drone_pos,
        drone_R,
        cam_offset_body=off_b,
        cam_look_body=look_b,
        width=gw,
        height=gh,
        fov=gfov,
    )
    gc = np.asarray(portal["pos"], dtype=np.float32)
    tilt_r = float(np.deg2rad(portal["tilt_deg"]))
    yaw_r = float(np.deg2rad(portal["yaw_deg"]))
    gq = _portal_frame_quaternion_bullet(p, tilt_rad=tilt_r, yaw_rad=yaw_r)

    base: dict[str, Any] = {
        "fpv_rgb": rgb,
        "portal": portal,
        "gate_pose_used_dedicated_cam": bool(gate_pose_use_dedicated_camera),
        "gate_pose_resolution": (gw, gh),
        "center_world": gc,
        "tilt_rad": tilt_r,
        "yaw_rad": yaw_r,
    }

    if use_sim_truth:
        Rslot = build_drone_R_aligned_to_rect_portal(p, gq, drone_pos, gc)
        return {
            **base,
            "R_body_world": Rslot,
            "gate_quat": gq,
            "pose_source": "sim_truth",
        }

    mode = str(gate_pose_estimator).strip().lower()
    if mode not in ("opencv", "xvla"):
        mode = "opencv"

    rgba = list(portal.get("rgba", [0.8, 0.8, 0.8, 1.0]))
    skew_rad: float | None = None
    area_frac = 0.0
    Rslot: np.ndarray | None = None
    pose_source: str | None = None
    gate_q_out: list[float] | None = None

    if mode == "xvla" and xvla_server_url and proprio_20d is not None and decode_workspace_lo is not None and decode_workspace_hi is not None:
        color_key = str(portal.get("color", "target"))
        lid0 = portal.get("portal_label")
        bill_txt = str(int(lid0)) if lid0 is not None else ""
        inst_text = _instruction_phrase_from_defaults(
            str(xvla_gate_instruction_template),
            color=color_key,
            billboard_id=bill_txt,
        )
        Rslot = gate_rotation_matrix_from_xvla(
            server_url=xvla_server_url,
            image_rgb=rgb,
            proprio_20d=np.asarray(proprio_20d, dtype=np.float32),
            language_instruction=inst_text,
            steps=int(xvla_gate_steps),
            infer_width=int(xvla_gate_infer_width),
            infer_height=int(xvla_gate_infer_height),
            workspace_lo=np.asarray(decode_workspace_lo, dtype=np.float32),
            workspace_hi=np.asarray(decode_workspace_hi, dtype=np.float32),
            treat_pos_as=str(decode_treat_pos_as),
            delta_pos_scale=float(decode_delta_pos_scale),
            act_request_timeout_s=float(xvla_act_request_timeout_s),
        )
        if Rslot is not None:
            pose_source = "xvla_action_rotation"

    if Rslot is None and mode == "xvla" and gate_pose_xvla_fallback_opencv:
        mode_eff = "opencv"
    else:
        mode_eff = mode

    if Rslot is None and mode_eff == "opencv":
        skew_rad, area_frac = cv_estimate_colored_rect_long_edge_skew_rad(
            rgb,
            rgba,
            rgb_pad=int(gate_pose_cv_rgb_pad),
            min_area_ratio=float(gate_pose_cv_min_area_ratio),
        )
        base["cv_long_edge_skew_rad"] = skew_rad
        base["cv_mask_area_frac"] = area_frac
        if skew_rad is not None:
            twist = -float(skew_rad)
            Rslot = build_drone_R_align_approach_and_inplane_long_twist(drone_pos, gc, twist_rad=twist)
            pose_source = "opencv_long_axis"
    else:
        base["cv_long_edge_skew_rad"] = skew_rad
        base["cv_mask_area_frac"] = area_frac

    if Rslot is None and gate_pose_cv_fallback_map_pose:
        Rslot = build_drone_R_aligned_to_rect_portal(p, gq, drone_pos, gc)
        if pose_source is None:
            pose_source = "map_pose_fallback"
        gate_q_out = gq

    return {
        **base,
        "R_body_world": Rslot,
        "gate_quat": gate_q_out,
        "pose_source": pose_source,
    }


def _world_recording_view_proj(
    p,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    *,
    view_kind: str,
    width: int,
    height: int,
    fov: float,
    margin: float,
    distance_scale: float,
    top_view_distance_scale: float,
    stereo45_view_distance_scale: float = 1.0,
) -> tuple[list[float], list[float]]:

    lo = np.asarray(workspace_lo, dtype=np.float64)
    hi = np.asarray(workspace_hi, dtype=np.float64)
    scene_pad = np.array([0.12, 0.12, 0.10], dtype=np.float64)
    lo = lo - scene_pad
    hi = hi + scene_pad
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm((hi - lo) * 0.5))
    fov_rad = np.deg2rad(max(float(fov), 1.0))
    ds = float(np.clip(distance_scale, 0.05, 100.0))
    dist = max(
        radius / max(np.sin(fov_rad * 0.5), 1e-6) * float(margin) * ds,
        0.8,
    )
    if view_kind == "top":
        dist *= float(np.clip(top_view_distance_scale, 0.05, 100.0))
    elif view_kind == "45deg":
        dist *= float(np.clip(stereo45_view_distance_scale, 0.05, 100.0))

    if view_kind == "front":
        direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        up = [0, 0, 1]
    elif view_kind == "right":
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = [0, 0, 1]
    elif view_kind == "top":
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = [0, 1, 0]
    elif view_kind == "45deg":
        direction = np.array([1.0, -1.0, np.sqrt(2.0)], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        up = [0, 0, 1]
    else:
        raise ValueError(f"Unknown recording view_kind: {view_kind!r}")

    eye = center + direction * dist
    near_val = max(0.01, dist - radius * 2.5)
    far_val = dist + radius * 3.5
    view = p.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=center.tolist(),
        cameraUpVector=up,
    )
    aspect = width / max(height, 1)
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near_val, farVal=far_val)
    return view, proj


def _draw_phase1_detection_overlay_bgr(
    img_bgr: np.ndarray,
    *,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    z_ground: float,
    start_xy: np.ndarray,
    start_wz: float,
    target_xyz: np.ndarray,
    obstacle_centers: list[tuple[float, float, float]],
) -> None:

    import cv2

    rect_w = [
        wproj(x_min, y_min, z_ground),
        wproj(x_max, y_min, z_ground),
        wproj(x_max, y_max, z_ground),
        wproj(x_min, y_max, z_ground),
    ]
    if all(pt is not None for pt in rect_w):
        poly = np.array([[int(p[0]), int(p[1])] for p in rect_w], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            img_bgr,
            [poly],
            isClosed=True,
            color=(144, 238, 144),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    sxy = np.asarray(start_xy, dtype=np.float64).reshape(-1)
    txyz = np.asarray(target_xyz, dtype=np.float64).reshape(-1)
    sp = wproj(float(sxy[0]), float(sxy[1]), float(start_wz))
    tp = wproj(float(txyz[0]), float(txyz[1]), float(txyz[2]))
    if sp is not None:
        cv2.circle(img_bgr, sp, 8, (0, 255, 0), -1)
        cv2.putText(
            img_bgr,
            "START",
            (sp[0] - 30, sp[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if tp is not None:
        cv2.circle(img_bgr, tp, 8, (255, 0, 255), -1)
        cv2.putText(
            img_bgr,
            "TARGET",
            (tp[0] - 30, tp[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    for i, (px, py, pz) in enumerate(obstacle_centers):
        opc = wproj(float(px), float(py), float(pz))
        if opc is None:
            continue
        cv2.circle(img_bgr, opc, 6, (0, 165, 255), -1)
        cv2.putText(
            img_bgr,
            f"OBJ{i+1}",
            (opc[0] - 20, opc[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )


def world_xyz_to_recording_image_pixel(
    world_xyz: np.ndarray,
    view: list[float],
    proj: list[float],
    *,
    width: int,
    height: int,
) -> tuple[int, int] | None:

    V = np.asarray(view, dtype=np.float64).reshape((4, 4), order="F")
    Pm = np.asarray(proj, dtype=np.float64).reshape((4, 4), order="F")
    pw = np.concatenate([np.asarray(world_xyz, dtype=np.float64).reshape(3), np.ones(1, dtype=np.float64)])
    cam = V @ pw
    clip = Pm @ cam
    wc = float(clip[3])
    if abs(wc) < 1e-10:
        return None
    ndc_x = float(clip[0] / wc)
    ndc_y = float(clip[1] / wc)
    u = (ndc_x * 0.5 + 0.5) * float(width) - 0.5
    v = (-ndc_y * 0.5 + 0.5) * float(height) - 0.5
    ui = int(np.clip(np.round(u), 0, max(width - 1, 0)))
    vi = int(np.clip(np.round(v), 0, max(height - 1, 0)))
    return ui, vi


def _render_world_recording_rgb(
    p,
    view: list[float],
    proj: list[float],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    return render_camera_rgb(p, view, proj, width=int(width), height=int(height))


def get_world_recording_rgb(
    p,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    *,
    view_kind: str,
    width: int = 640,
    height: int = 480,
    fov: float = 42.0,
    margin: float = 1.55,
    distance_scale: float = 1.0,
    top_view_distance_scale: float = 1.0,
    stereo45_view_distance_scale: float = 1.0,
) -> np.ndarray:

    view, proj = _world_recording_view_proj(
        p,
        workspace_lo,
        workspace_hi,
        view_kind=view_kind,
        width=int(width),
        height=int(height),
        fov=float(fov),
        margin=float(margin),
        distance_scale=float(distance_scale),
        top_view_distance_scale=float(top_view_distance_scale),
        stereo45_view_distance_scale=float(stereo45_view_distance_scale),
    )
    return _render_world_recording_rgb(p, view, proj, width=int(width), height=int(height))


def _world_recording_camera_pose(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    *,
    view_kind: str,
    fov: float,
    margin: float,
    distance_scale: float,
    top_view_distance_scale: float = 1.0,
    stereo45_view_distance_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lo = np.asarray(workspace_lo, dtype=np.float64)
    hi = np.asarray(workspace_hi, dtype=np.float64)
    scene_pad = np.array([0.12, 0.12, 0.10], dtype=np.float64)
    lo = lo - scene_pad
    hi = hi + scene_pad
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm((hi - lo) * 0.5))
    fov_rad = np.deg2rad(max(float(fov), 1.0))
    ds = float(np.clip(distance_scale, 0.05, 100.0))
    dist = max(
        radius / max(np.sin(fov_rad * 0.5), 1e-6) * float(margin) * ds,
        0.8,
    )
    if view_kind == "top":
        dist *= float(np.clip(top_view_distance_scale, 0.05, 100.0))
    elif view_kind == "45deg":
        dist *= float(np.clip(stereo45_view_distance_scale, 0.05, 100.0))

    if view_kind == "front":
        direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif view_kind == "right":
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif view_kind == "top":
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif view_kind == "45deg":
        direction = np.array([1.0, -1.0, np.sqrt(2.0)], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown recording view_kind: {view_kind!r}")

    eye = center + direction * dist
    return eye.astype(np.float64), center.astype(np.float64), up.astype(np.float64), float(fov)


def estimate_topdown_objects_xy_aabb_from_rgb(
    rgb: np.ndarray,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    *,
    width: int,
    height: int,
    fov: float,
    margin: float,
    distance_scale: float,
    virtual_base_world: np.ndarray,
    top_view_distance_scale: float = 1.0,
    min_area_ratio: float = 0.003,
    diff_thresh: float = 18.0,
) -> tuple[np.ndarray, np.ndarray, float] | None:

    h = int(height)
    w = int(width)
    if rgb is None or rgb.size == 0 or h <= 0 or w <= 0:
        return None
    img = np.asarray(rgb, dtype=np.uint8)
    if img.shape[0] != h or img.shape[1] != w:
        return None
    border = np.concatenate(
        [img[0, :, :], img[-1, :, :], img[:, 0, :], img[:, -1, :]],
        axis=0,
    )
    bg = np.median(border, axis=0).astype(np.float32)
    diff = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=2)
    mask = diff > float(diff_thresh)
    frac = float(np.mean(mask))
    if frac < float(min_area_ratio):
        return None
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    u0, u1 = int(xs.min()), int(xs.max())
    v0, v1 = int(ys.min()), int(ys.max())
    eye, center, up, fov_out = _world_recording_camera_pose(
        np.asarray(workspace_lo, dtype=np.float64) + np.asarray(virtual_base_world, dtype=np.float64),
        np.asarray(workspace_hi, dtype=np.float64) + np.asarray(virtual_base_world, dtype=np.float64),
        view_kind="top",
        fov=float(fov),
        margin=float(margin),
        distance_scale=float(distance_scale),
        top_view_distance_scale=float(top_view_distance_scale),
    )
    plane_z = float(np.asarray(workspace_lo, dtype=np.float64)[2] + float(virtual_base_world[2]))
    plane_point = np.array([0.0, 0.0, plane_z], dtype=np.float64)
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    hits: list[np.ndarray] = []
    for u, v in corners:
        o, d = global_workspace_cam_unit_ray(
            eye,
            center,
            width=w,
            height=h,
            fov_deg=fov_out,
            u=float(u),
            v=float(v),
            up_world=up,
        )
        hit = _ray_plane_intersect_3d(o, d, plane_point, plane_normal)
        if hit is None:
            return None
        hits.append(hit)
    pts = np.stack(hits, axis=0)
    lo = pts[:, 0:2].min(axis=0)
    hi = pts[:, 0:2].max(axis=0)
    w_lo = np.asarray(workspace_lo, dtype=np.float64) + np.asarray(virtual_base_world, dtype=np.float64)
    w_hi = np.asarray(workspace_hi, dtype=np.float64) + np.asarray(virtual_base_world, dtype=np.float64)
    lo = np.maximum(lo, w_lo[0:2])
    hi = np.minimum(hi, w_hi[0:2])
    if float(np.min(hi - lo)) <= 1e-5:
        return None
    return lo.astype(np.float64), hi.astype(np.float64), float(frac)


def build_proprio_widowx_ee6d(pos_arm_base: np.ndarray, R_world: np.ndarray, gripper: float = 0.0) -> np.ndarray:
    proprio = np.zeros(20, dtype=np.float32)
    proprio[0:3] = pos_arm_base.astype(np.float32)
    proprio[3:9] = rotmat_to_6d(R_world)
    proprio[9] = float(gripper)
    return proprio


_PORTAL_COLORS_CMD_HINT: tuple[str, ...] = (
    "light-purple",
    "light-green",
    "light-red",
    "yellow",
    "orange",
    "purple",
    "green",
    "cyan",
    "blue",
    "red",
)

def _portal_billboard_ids_in_instruction_text(text: str) -> list[int]:

    from algorithms.instruction_parse import portal_billboard_ids_in_text

    return portal_billboard_ids_in_text(text)


def extract_instruction_portal_billboard_id(instruction: str) -> int | None:

    from algorithms.instruction_parse import extract_ordered_traversal_billboard_ids

    ordered = extract_ordered_traversal_billboard_ids(instruction)
    if len(ordered) == 1:
        return int(ordered[0])
    if len(ordered) > 1:
        return None

    if not instruction or not str(instruction).strip():
        return None
    ins = str(instruction).strip()
    found = _portal_billboard_ids_in_instruction_text(ins)
    if not found:
        return None
    first = found[0]
    if any(x != first for x in found):
        return None
    return int(first)


def _instruction_phrase_from_defaults(template: str, *, color: str, billboard_id: str) -> str:

    t = str(template)
    try:
        return t.format(color=color, billboard_id=billboard_id)
    except Exception:
        pass
    try:
        return t.format(color=color)
    except Exception:
        pass
    return t.replace("{color}", color).replace("{billboard_id}", billboard_id)


def _instruction_mentions_color(ins_l: str, col: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(col)}(?![a-z0-9])", ins_l) is not None


def augment_instruction_color_disambiguation(instruction: str, placed_cubes: list[dict]) -> str:

    if not instruction or not placed_cubes:
        return instruction
    ins_l = instruction.lower()
    placed_colors = {str(c.get("color", "")).lower() for c in placed_cubes}
    for col in _PORTAL_COLORS_CMD_HINT:
        if col not in placed_colors:
            continue
        if not _instruction_mentions_color(ins_l, col):
            continue
        base = instruction.strip()
        return (
            f"{base} "
            f"(Important: pass only through the {col}-colored rectangular portal; "
            f"avoid all other colored frames.)"
        )
    return instruction


def augment_instruction_billboard_disambiguation(instruction: str, placed_cubes: list[dict]) -> str:

    if not instruction or not placed_cubes:
        return instruction
    bid = extract_instruction_portal_billboard_id(instruction)
    if bid is None:
        return instruction
    have = False
    for c in placed_cubes:
        if c.get("portal_label") != bid:
            continue
        sh = str(c.get("shape", "")).lower()
        if sh == "rect_frame" or "rect" in sh or sh in ("square_frame", "frame"):
            have = True
            break
    if not have:
        return instruction
    base = instruction.strip()
    return (
        f"{base} "
        f"(Important: fly only through numbered portal billboard_id={bid}; skip all other markers.)"
    )


def global_workspace_cam_unit_ray(
    eye: np.ndarray,
    look_at: np.ndarray,
    *,
    width: int,
    height: int,
    fov_deg: float,
    u: float,
    v: float,
    up_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    look = np.asarray(look_at, dtype=np.float64).reshape(3)
    upv = np.asarray(up_world if up_world is not None else [0.0, 0.0, 1.0], dtype=np.float64).reshape(3)
    forward = look - eye
    fn = float(np.linalg.norm(forward))
    if fn < 1e-12:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fn = 1.0
    forward /= fn
    right = np.cross(forward, upv)
    rn = float(np.linalg.norm(right))
    if rn < 1e-12:
        right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        rn = 1.0
    right /= rn
    up_cam = np.cross(right, forward)
    up_cam /= max(float(np.linalg.norm(up_cam)), 1e-12)
    aspect = float(width) / max(float(height), 1.0)
    tan_half = float(np.tan(np.deg2rad(float(fov_deg)) * 0.5))
    x_ndc = (float(u) + 0.5) / float(max(width, 1)) * 2.0 - 1.0
    y_ndc = -((float(v) + 0.5) / float(max(height, 1)) * 2.0 - 1.0)
    direc = forward + right * x_ndc * tan_half * aspect + up_cam * y_ndc * tan_half
    direc /= max(float(np.linalg.norm(direc)), 1e-12)
    return eye.copy(), direc


def _ray_plane_intersect_3d(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray | None:
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    p0 = np.asarray(plane_point, dtype=np.float64).reshape(3)
    n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    denom = float(np.dot(d, n))
    if abs(denom) < 1e-10:
        return None
    t = float(np.dot(p0 - o, n)) / denom
    if t < 0.0:
        return None
    return (o + t * d).astype(np.float64)


def portal_opening_plane_normal_world(
    p,
    portal: dict,
    *,
    toward: np.ndarray | None = None,
) -> np.ndarray:

    c = np.asarray(portal["pos"], dtype=np.float64).reshape(3)
    tilt_r = np.deg2rad(float(portal["tilt_deg"]))
    yaw_r = np.deg2rad(float(portal["yaw_deg"]))
    quat = _portal_frame_quaternion_bullet(p, tilt_rad=tilt_r, yaw_rad=yaw_r)
    nx = np.asarray(p.rotateVector(quat, [1.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    nx /= max(float(np.linalg.norm(nx)), 1e-12)
    if toward is not None:
        t = np.asarray(toward, dtype=np.float64).reshape(3) - c
        if float(np.linalg.norm(t)) > 1e-9 and float(np.dot(nx, t)) < 0.0:
            nx = -nx
    return nx.astype(np.float64)


def _rgba01_to_bgr_inrange_bounds(
    rgba01: list[float] | tuple[float, ...],
    rgb_pad: float,
) -> tuple[np.ndarray, np.ndarray]:

    rgba = np.asarray(rgba01, dtype=np.float64).reshape(-1)
    targ_rgb = np.clip(rgba[:3] * 255.0, 0.0, 255.0).astype(np.float64)
    targ_bgr = np.array([targ_rgb[2], targ_rgb[1], targ_rgb[0]], dtype=np.float64)
    pd = float(np.clip(rgb_pad, 0, 255))
    lo = np.clip(targ_bgr - pd, 0.0, 255.0).astype(np.uint8)
    hi = np.clip(targ_bgr + pd, 0.0, 255.0).astype(np.uint8)
    return lo, hi


def _portal_color_segment_mask_rgb(
    rgb_u8: np.ndarray,
    rgba01: list[float] | tuple[float, ...],
    *,
    rgb_pad: int,
    morph_px: int = 3,
) -> np.ndarray:

    import cv2

    bgr = cv2.cvtColor(rgb_u8.astype(np.uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ref = np.asarray(rgba01, dtype=np.float64).reshape(-1)
    rc = int(np.clip(ref[0] * 255.0, 0.0, 255.0))
    gc = int(np.clip(ref[1] * 255.0, 0.0, 255.0))
    bc = int(np.clip(ref[2] * 255.0, 0.0, 255.0))
    ref_bgr = np.uint8([[[bc, gc, rc]]])
    ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
    h0 = int(ref_hsv[0, 0, 0])
    s0 = int(ref_hsv[0, 0, 1])
    v0 = int(ref_hsv[0, 0, 2])
    pd = float(np.clip(rgb_pad, 0, 255))
    dh = int(np.clip(7 + pd / 5.0, 8, 26))
    s_floor = int(np.clip(s0 * 0.42 - pd * 0.08, 28.0, 165.0))
    v_floor = int(np.clip(v0 * 0.38 - pd * 0.08, 32.0, 200.0))

    if h0 <= max(12, dh) or h0 >= 180 - max(12, dh):
        w = min(22, 12 + dh // 2)
        lo1 = np.array([0, s_floor, v_floor], dtype=np.uint8)
        hi1 = np.array([w, 255, 255], dtype=np.uint8)
        lo2 = np.array([180 - w, s_floor, v_floor], dtype=np.uint8)
        hi2 = np.array([179, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
    else:
        lo_h = int(np.clip(h0 - dh, 0, 179))
        hi_h = int(np.clip(h0 + dh, 0, 179))
        hsv_mask = cv2.inRange(
            hsv, np.array([lo_h, s_floor, v_floor], dtype=np.uint8), np.array([hi_h, 255, 255], dtype=np.uint8)
        )

    pd_b = float(pd)
    lo_b, hi_b = _rgba01_to_bgr_inrange_bounds(rgba01, pd_b)
    bgr_mask = cv2.inRange(bgr, lo_b, hi_b)
    ksz = max(1, int(morph_px)) | 1
    kernel = np.ones((ksz, ksz), dtype=np.uint8)

    def _minus_near_white(m: np.ndarray) -> np.ndarray:

        wlo = np.array([0, 0, 215], dtype=np.uint8)
        whi = np.array([179, 55, 255], dtype=np.uint8)
        wm = cv2.inRange(hsv, wlo, whi)
        return cv2.bitwise_and(m, cv2.bitwise_not(wm))

    hsv_m = cv2.morphologyEx(_minus_near_white(hsv_mask), cv2.MORPH_CLOSE, kernel)
    bgr_m = cv2.morphologyEx(_minus_near_white(bgr_mask), cv2.MORPH_CLOSE, kernel)
    n = float(hsv_m.size)
    f_h = float(np.count_nonzero(hsv_m)) / max(n, 1.0)
    f_b = float(np.count_nonzero(bgr_m)) / max(n, 1.0)

    if 0.0012 <= f_h <= 0.48:
        return hsv_m
    if 0.0012 <= f_b <= 0.48:
        return bgr_m
    if f_h > 0.0 and (f_b < 0.0012 or f_h <= f_b):
        return hsv_m
    if f_b > 0.0:
        return bgr_m
    return hsv_m


def _first_rect_portal_for_instruction_color(
    instruction: str,
    placed_cubes: list[dict],
    *,
    prefer_near_xyz: np.ndarray | None = None,
) -> dict | None:

    if not instruction or not placed_cubes:
        return None
    from algorithms.instruction_parse import (
        extract_ordered_traversal_billboard_ids,
        find_rect_portal_by_billboard_id,
    )

    ordered_bb = extract_ordered_traversal_billboard_ids(instruction)
    if ordered_bb:
        portal0 = find_rect_portal_by_billboard_id(placed_cubes, int(ordered_bb[0]))
        if portal0 is not None:
            return portal0
    bb = extract_instruction_portal_billboard_id(instruction)
    if bb is not None:
        cand_bb: list[dict] = []
        for c in placed_cubes:
            if c.get("portal_label") != int(bb):
                continue
            sh = str(c.get("shape", "")).lower()
            if sh == "rect_frame" or "rect" in sh or sh in ("square_frame", "frame"):
                cand_bb.append(c)
        if len(cand_bb) == 1:
            return cand_bb[0]
    ins_l = instruction.lower()
    placed_colors = {str(c.get("color", "")).lower() for c in placed_cubes}
    p0 = (
        np.asarray(prefer_near_xyz, dtype=np.float64).reshape(3)
        if prefer_near_xyz is not None
        else None
    )
    for col in _PORTAL_COLORS_CMD_HINT:
        if col not in placed_colors or not _instruction_mentions_color(ins_l, col):
            continue
        cand: list[dict] = []
        for c in placed_cubes:
            if str(c.get("color", "")).lower() != col:
                continue
            sh = str(c.get("shape", "")).lower()
            if sh == "rect_frame" or "rect" in sh or sh in ("square_frame", "frame"):
                cand.append(c)
        if not cand:
            continue
        if p0 is not None and len(cand) > 1:
            def _dist_key(c: dict) -> float:
                pc = np.asarray(c["pos"], dtype=np.float64).reshape(3)
                return float(np.linalg.norm(pc - p0))

            cand.sort(key=_dist_key)
        return cand[0]
    return None


def _masked_color_centroid_frac(
    rgb_u8: np.ndarray,
    rgba01: list[float] | tuple[float, ...],
    *,
    rgb_pad: int,
    morph_px: int = 3,
) -> tuple[float | None, float | None, float]:

    try:
        import cv2
    except Exception:
        return None, None, 0.0
    if rgb_u8 is None or rgb_u8.size == 0:
        return None, None, 0.0
    h, w = rgb_u8.shape[:2]
    if h < 8 or w < 8:
        return None, None, 0.0
    mask = _portal_color_segment_mask_rgb(
        rgb_u8, rgba01, rgb_pad=int(rgb_pad), morph_px=int(morph_px)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    frac = area / float(max(h * w, 1))
    m = cv2.moments(cnt)
    if m["m00"] < 1e-6:
        return None, None, frac
    cx = float(m["m10"] / m["m00"])
    cy = float(m["m01"] / m["m00"])
    return cx, cy, frac


def resolve_cmd_target_from_global_workspace_camera(
    p,
    instruction: str,
    placed_cubes: list[dict],
    *,
    cam_eye_world: np.ndarray,
    cam_look_world: np.ndarray,
    cam_width: int,
    cam_height: int,
    cam_fov_deg: float,
    rgb_pad: int,
    min_area_ratio: float,
    prefer_near_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float, str]:

    portal = _first_rect_portal_for_instruction_color(
        instruction, placed_cubes, prefer_near_xyz=prefer_near_xyz
    )
    if portal is None:
        return None, 0.0, "no_portal_color_match"
    center_cat = np.asarray(portal["pos"], dtype=np.float32).reshape(3)
    rgb_img = get_workspace_rgb(
        p,
        np.asarray(cam_eye_world, dtype=np.float64),
        np.asarray(cam_look_world, dtype=np.float64),
        width=int(cam_width),
        height=int(cam_height),
        fov=float(cam_fov_deg),
    )
    rgba = portal.get("rgba", [0.8, 0.8, 0.8, 1.0])
    cx, cy, frac = _masked_color_centroid_frac(rgb_img, rgba, rgb_pad=int(rgb_pad))
    if cx is None or frac < float(min_area_ratio):
        return center_cat.copy(), float(frac), "weak_mask_use_catalog_center"

    eye = np.asarray(cam_eye_world, dtype=np.float64).reshape(3)
    _, direc = global_workspace_cam_unit_ray(
        eye,
        np.asarray(cam_look_world, dtype=np.float64).reshape(3),
        width=int(cam_width),
        height=int(cam_height),
        fov_deg=float(cam_fov_deg),
        u=cx,
        v=cy,
    )
    c = np.asarray(portal["pos"], dtype=np.float64).reshape(3)
    n = portal_opening_plane_normal_world(p, portal, toward=eye)
    hit = _ray_plane_intersect_3d(eye, direc, c, n)
    if hit is None:
        return center_cat.copy(), float(frac), "ray_parallel_use_catalog_center"
    return hit.astype(np.float32), float(frac), "ray_plane_opening"


def query_xvla(
    server_url: str,
    image_rgb: np.ndarray,
    proprio_20d: np.ndarray,
    instruction: str,
    steps: int,
    *,
    timeout: float = 300.0,
):
    payload = {
        "proprio": json_numpy.dumps(proprio_20d),
        "language_instruction": instruction,
        "image0": json_numpy.dumps(image_rgb),
        "domain_id": 0,
        "steps": steps,
    }
    r = requests.post(server_url, json=payload, timeout=timeout)
    r.raise_for_status()
    out = r.json()
    if "action" not in out:
        raise RuntimeError(f"Unexpected /act response: {str(out)[:300]}")
    return np.asarray(out["action"], dtype=np.float32)


def wait_for_xvla_act_inference(
    server_url: str,
    *,
    probe_request_timeout_s: float = 300.0,
    max_wait_s: float = 900.0,
    retry_interval_s: float = 3.0,
    image_height: int = 256,
    image_width: int = 256,
) -> None:

    h = max(8, int(image_height))
    w = max(8, int(image_width))
    img = np.full((h, w, 3), 127, dtype=np.uint8)
    proprio = np.zeros(20, dtype=np.float32)
    instr = "ready_probe"
    t0 = time.monotonic()
    t_req = float(max(30.0, probe_request_timeout_s))
    t_max = float(max(t_req, max_wait_s))
    t_sleep = float(max(0.5, retry_interval_s))
    attempt = 0
    last_err: str = ""
    print(
        f"[xvla] waiting for first /act inference "
        f"(request_timeout={t_req:.0f}s, max_wait={t_max:.0f}s) …"
    )
    while True:
        attempt += 1
        try:
            _ = query_xvla(
                server_url,
                img,
                proprio,
                instr,
                steps=1,
                timeout=t_req,
            )
            dt = time.monotonic() - t0
            print(f"[xvla] /act inference ready (attempt {attempt}, {dt:.1f}s)")
            return
        except Exception as exc:
            last_err = str(exc)
            elapsed = time.monotonic() - t0
            if elapsed >= t_max:
                raise RuntimeError(
                    f"X-VLA /act did not become ready within {t_max:.0f}s "
                    f"(last error: {last_err})"
                ) from exc
            if attempt == 1 or attempt % 5 == 0:
                print(f"[xvla] /act probe retry ({elapsed:.0f}s / {t_max:.0f}s): {last_err}", file=sys.stderr)
            time.sleep(t_sleep)


def decode_action_widowx_ee6d(
    action_chunk: np.ndarray,
    *,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    treat_pos_as: str = "absolute",
    current_pos: np.ndarray | None = None,
    current_R: np.ndarray | None = None,
    delta_pos_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:

    a = action_chunk[0] if action_chunk.ndim == 2 else action_chunk
    pos = a[0:3].astype(np.float32)

    if treat_pos_as == "delta":
        if current_pos is None:
            raise ValueError("delta mode needs current_pos")
        target_pos = current_pos + pos * float(delta_pos_scale)
    else:
        target_pos = pos
    target_pos = np.minimum(np.maximum(target_pos, workspace_lo), workspace_hi).astype(np.float32)

    target_R = rot6d_to_matrix(a[3:9])

    g_logit = float(a[9])
    gripper = 1.0 / (1.0 + np.exp(-g_logit))
    return target_pos, target_R, gripper


def decode_action_chunk_to_local_waypoints(
    actions: np.ndarray,
    *,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    treat_pos_as: str,
    start_pos_local: np.ndarray,
    delta_pos_scale: float,
) -> tuple[np.ndarray, list[np.ndarray]]:

    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n, d = a.shape
    if d < 10:
        raise ValueError(f"action rows need ≥10 dims (got {d})")
    lo = np.asarray(workspace_lo, dtype=np.float32).reshape(3)
    hi = np.asarray(workspace_hi, dtype=np.float32).reshape(3)
    cur = np.asarray(start_pos_local, dtype=np.float32).reshape(3).copy()
    pts = np.zeros((n, 3), dtype=np.float32)
    Rs: list[np.ndarray] = []
    scl = float(delta_pos_scale)
    for i in range(n):
        row = a[i]
        pos = row[0:3].astype(np.float32)
        if treat_pos_as == "delta":
            cur = cur + pos * scl
        else:
            cur = pos.copy()
        cur = np.minimum(np.maximum(cur, lo), hi).astype(np.float32)
        pts[i] = cur
        Rs.append(rot6d_to_matrix(row[3:9]))
    return pts, Rs


def _clip_world_pos_to_workspace(
    p_world: np.ndarray,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    virtual_base_world: np.ndarray,
) -> np.ndarray:
    lo = workspace_lo.astype(np.float64) + virtual_base_world.astype(np.float64)
    hi = workspace_hi.astype(np.float64) + virtual_base_world.astype(np.float64)
    q = np.asarray(p_world, dtype=np.float64).reshape(3)
    return np.minimum(np.maximum(q, lo), hi).astype(np.float32)


def _expand_workspace_hi_local_for_world_z(
    workspace_hi: np.ndarray,
    *,
    z_target_world: float,
    virtual_base_z: float,
    slack_local: float = 0.05,
) -> np.ndarray:

    out = np.asarray(workspace_hi, dtype=np.float32).copy()
    z_hi_need_local = float(z_target_world - float(virtual_base_z) + float(slack_local))
    if z_hi_need_local > float(out[2]):
        old = float(out[2])
        out[2] = float(z_hi_need_local)
        print(
            "[demo] scenic Z: raised workspace_hi[2] (local) "
            f"{old:.3f}m → {out[2]:.3f}m so target cruise world Z≈{z_target_world:.3f}m is inside hull"
        )
    return out


def _scene_objects_xy_aabb_world(
    placed_cubes: list[dict],
) -> tuple[np.ndarray, np.ndarray] | None:

    if not placed_cubes:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for c in placed_cubes:
        pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
        bh = c.get("bounds_half")
        if bh is not None:
            h = np.asarray(bh, dtype=np.float64).reshape(3)
            xs.extend([float(pos[0] - h[0]), float(pos[0] + h[0])])
            ys.extend([float(pos[1] - h[1]), float(pos[1] + h[1])])
        else:
            half = float(c.get("half", 0.025))
            xs.extend([float(pos[0] - half), float(pos[0] + half)])
            ys.extend([float(pos[1] - half), float(pos[1] + half)])
    lo = np.array([min(xs), min(ys)], dtype=np.float64)
    hi = np.array([max(xs), max(ys)], dtype=np.float64)
    return lo, hi


def _scene_objects_aabb_world(
    placed_cubes: list[dict],
) -> tuple[np.ndarray, np.ndarray] | None:

    if not placed_cubes:
        return None
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for c in placed_cubes:
        pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
        bh = c.get("bounds_half")
        if bh is not None:
            h = np.asarray(bh, dtype=np.float64).reshape(3)
            xs.extend([float(pos[0] - h[0]), float(pos[0] + h[0])])
            ys.extend([float(pos[1] - h[1]), float(pos[1] + h[1])])
            zs.extend([float(pos[2] - h[2]), float(pos[2] + h[2])])
        else:
            half = float(c.get("half", 0.025))
            xs.extend([float(pos[0] - half), float(pos[0] + half)])
            ys.extend([float(pos[1] - half), float(pos[1] + half)])
            zs.extend([float(pos[2] - half), float(pos[2] + half)])
    lo = np.array([min(xs), min(ys), min(zs)], dtype=np.float64)
    hi = np.array([max(xs), max(ys), max(zs)], dtype=np.float64)
    return lo, hi


def _placed_object_aabb_lo_hi_world(c: dict) -> tuple[np.ndarray, np.ndarray]:

    pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
    bh = c.get("bounds_half")
    if bh is not None:
        h = np.asarray(bh, dtype=np.float64).reshape(3)
        return pos - h, pos + h
    half = float(c.get("half", 0.025))
    h = np.array([half, half, half], dtype=np.float64)
    return pos - h, pos + h


def _placed_obj_nav_kind(c: dict) -> str:

    sh = str(c.get("shape", "cube")).lower()
    if sh in ("rect_frame", "square_frame", "frame"):
        return "gate"
    if sh == "sphere":
        return "sphere"
    if sh == "ramp":
        return "ramp"
    return "cube"


def _phase1_include_placed_obj(c: dict) -> bool:
    return _placed_obj_nav_kind(c) in ("gate", "cube", "sphere", "ramp")


def _placed_obj_display_name(c: dict) -> str:
    col = str(c.get("color_name", c.get("color", "?")))
    return f"{col} {_placed_obj_nav_kind(c)}"


def _placed_object_pyb_aabb_pair(c: dict) -> tuple[list[float], list[float]]:
    lo, hi = _placed_object_aabb_lo_hi_world(c)
    return lo.tolist(), hi.tolist()


def _object_sphere_radius(
    c: dict,
    *,
    scale: float = 1.0,
    inflate: float = 0.0,
    min_r: float = 0.05,
) -> float:

    bh = c.get("bounds_half")
    if bh is not None:
        h = np.asarray(bh, dtype=np.float64).reshape(3)
        r = float(np.linalg.norm(h))
    else:
        half = float(c.get("half", 0.025))
        r = float(np.sqrt(3.0) * max(half, 0.0))
    r = max(float(r), float(min_r))
    r = float(r) * float(scale) + float(inflate)
    return max(float(r), 1e-6)


def _point_to_aabb_vector_outward(
    p: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[float, np.ndarray]:

    p = np.asarray(p, dtype=np.float64).reshape(3)
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    q = np.clip(p, lo, hi)
    delta = p - q
    dist_out = float(np.linalg.norm(delta))
    if dist_out > 1e-9:
        return dist_out, (delta / dist_out).astype(np.float64)
    d_lo = p - lo
    d_hi = hi - p
    pen = np.minimum(d_lo, d_hi)
    j = int(np.argmin(pen))
    n = np.zeros(3, dtype=np.float64)
    if float(d_lo[j]) < float(d_hi[j]):
        n[j] = -1.0
        return -float(d_lo[j]), n
    n[j] = 1.0
    return -float(d_hi[j]), n


def format_scene_semantic_catalog_for_xvla(
    placed_cubes: list[dict],
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
) -> str:

    if not placed_cubes:
        return ""
    lines: list[str] = []
    wlo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    whi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    lines.append(
        f"Workspace axis-aligned bounds (meters, world frame): lo={wlo.round(4).tolist()} hi={whi.round(4).tolist()}."
    )
    lines.append(
        f"Objects ({len(placed_cubes)}); each line: id, semantic color, shape (rect portals include billboard_id), center xyz, AABB half-extents:"
    )
    for i, c in enumerate(placed_cubes):
        pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
        color = str(c.get("color", "?"))
        shape = str(c.get("shape", "cube"))
        lo, hi = _placed_object_aabb_lo_hi_world(c)
        half = 0.5 * (hi - lo)
        extra = ""
        if shape == "rect_frame":
            lid = c.get("portal_label")
            lab = f" billboard_id={int(lid)}" if lid is not None else ""
            extra = (
                f" portal_long_m={float(c.get('side', 0.0)):.4f} portal_short_m={float(c.get('short_side', 0.0)):.4f}"
                f" tilt_deg={float(c.get('tilt_deg', 0.0)):.2f} yaw_deg={float(c.get('yaw_deg', 0.0)):.2f}{lab}"
            )
        lines.append(
            f"  - id={i} color={color} shape={shape} center={pos.round(4).tolist()}"
            f" aabb_half={half.round(4).tolist()}{extra}"
        )
    return "\n".join(lines)


def compose_xvla_navigation_instruction(
    user_task_instruction: str,
    *,
    scene_catalog: str,
    enabled: bool,
    planning_suffix: str,
) -> str:

    base = str(user_task_instruction).strip()
    if not enabled or not scene_catalog.strip():
        return base
    sfx = str(planning_suffix).strip()
    if sfx:
        return (
            "[Scene semantic catalog — world-frame meters; use with the workspace camera image.]\n"
            f"{scene_catalog.strip()}\n\n"
            f"User task:\n{base}\n\n{sfx}"
        ).strip()
    return (
        "[Scene semantic catalog — world-frame meters; use with the workspace camera image.]\n"
        f"{scene_catalog.strip()}\n\n"
        f"User task:\n{base}"
    ).strip()


DEFAULT_XVLA_PATH_PLANNING_INSTRUCTION_SUFFIX = (
    "Plan a smooth, continuous 3D path in the workspace that fulfills the user task. "
    "Maintain generous clearance (at least 0.1m) from all obstacles, blocks, and rectangular frames "
    "to ensure collision-free flight. If the task requires passing through a specific rectangular portal, "
    "first align the drone position and attitude with the portal center and the long opening's "
    "orientation, then fly through carefully. "
    "Use both the workspace camera image and the provided numeric semantic AABB catalog "
    "for precise spatial reasoning and obstacle avoidance."
)


def run_navigation_phase2_topdown_xvla_xy_plan(
    *,
    server_url: str,
    topdown_rgb: np.ndarray,
    img_marked_bgr: np.ndarray,
    view_m: list[float],
    proj_m: list[float],
    render_width: int,
    render_height: int,
    phase1_folder: Path,
    png_filename: str,
    stereo_img_marked_bgr: np.ndarray | None,
    stereo_view_m: list[float] | None,
    stereo_proj_m: list[float] | None,
    stereo_png_filename: str,
    drone_pos_world: np.ndarray,
    drone_pos_local: np.ndarray,
    drone_R: np.ndarray,
    virtual_base_world: np.ndarray,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    treat_pos_as: str,
    delta_pos_scale: float,
    gripper_state: float,
    mission_cmd: str | None,
    language_instruction: str,
    obs_lines_text: str,
    corridor_xy: tuple[float, float, float, float],
    goal_xyz_world: np.ndarray,
    scene_catalog_str: str,
    xvla_scene_semantic_context: bool,
    xvla_path_planning_instruction_suffix: str,
    workspace_camera_width: int,
    workspace_camera_height: int,
    phase2_steps: int,
    xvla_act_request_timeout_s: float,
    sync_root_config: bool,
    sync_qs: bool,
    qs_policy_path: Path | None,
    config_json_path: Path,
    phase2_extra_instruction: str,
    phase1_obstacle_entries: list[Any],
    navigation_phase2_geom_astar: bool = True,
    navigation_phase2_astar_cell_m: float = 0.04,
    navigation_phase2_astar_obstacle_pad_m: float = 0.07,
    navigation_phase2_optional_topdown_xvla: bool = False,
    navigation_phase2_z_clearance_enabled: bool = True,
    navigation_phase2_z_clearance_margin_m: float = 0.08,
    navigation_phase2_z_workspace_margin_m: float = 0.02,
) -> None:

    import cv2

    xmin, ymin, xmax, ymax = corridor_xy
    pw = np.asarray(drone_pos_world, dtype=np.float64).reshape(-1)
    sx, sy, sz = float(pw[0]), float(pw[1]), float(pw[2])
    gv = np.asarray(goal_xyz_world, dtype=np.float64).reshape(-1)
    gx, gy, gz = float(gv[0]), float(gv[1]), float(gv[2])

    mission_txt = str(mission_cmd).strip() if mission_cmd else "(no --cmd)"

    obstacle_boxes_xy = obstacle_xy_boxes_phase1_entries(
        phase1_obstacle_entries,
        inflate_m=float(navigation_phase2_astar_obstacle_pad_m),
        exclude_goal_fn=None,
    )

    vbz = float(np.asarray(virtual_base_world, dtype=np.float64).reshape(-1)[2])
    z_cap_world = float(workspace_hi[2]) + vbz - float(navigation_phase2_z_workspace_margin_m)

    pts_arr: np.ndarray | None = None
    infer_ms: float | None = None
    planner_primary = "none"
    z_clearance_meta: dict[str, Any] | None = None
    steps = max(4, int(phase2_steps))

    def _try_z_clearance_plan(reason: str) -> bool:
        nonlocal pts_arr, planner_primary, z_clearance_meta
        if not bool(navigation_phase2_z_clearance_enabled):
            return False
        got = phase2_trajectory_xyz_z_clearance_overfly(
            pw,
            gv,
            phase1_obstacle_entries,
            inflate_xy_m=float(navigation_phase2_astar_obstacle_pad_m),
            z_clearance_margin_m=float(navigation_phase2_z_clearance_margin_m),
            z_workspace_cap_world=float(z_cap_world),
        )
        if got is None:
            return False
        traj, meta = got
        pts_arr = traj
        planner_primary = "z_clearance_overfly_xy"
        z_clearance_meta = meta
        print(
            f"\n[Phase 2] Z-axis clearance ({reason}): straight segment over "
            f"{meta.get('obstacles_along_xy_segment', 0)} obstacle tops; "
            f"Z_cruise≈{meta.get('z_cruise_world', 0):.4f} m (workspace top≈{z_cap_world:.4f} m)"
        )
        return True

    def _xy_segment_underclear_at_cruise() -> bool:
        aabbs = phase1_entries_world_aabbs(phase1_obstacle_entries)
        z_top_along, n_hit = min_z_clearance_for_world_xy_segment(
            (sx, sy), (gx, gy), aabbs, inflate_xy_m=float(navigation_phase2_astar_obstacle_pad_m)
        )
        if n_hit <= 0:
            return False
        need = float(z_top_along) + float(max(0.0, navigation_phase2_z_clearance_margin_m))
        return max(sz, gz) < need - 1e-6

    if navigation_phase2_geom_astar:
        pts_geom = phase2_trajectory_xyz_from_astar(
            corridor_xy,
            pw,
            gv,
            obstacle_boxes_xy,
            cell_m=float(navigation_phase2_astar_cell_m),
        )
        if pts_geom is not None and pts_geom.shape[0] >= 1:
            pts_arr = pts_geom
            planner_primary = "astar_corridor_xy_inflate"
            print(
                f"\n[Phase 2] Corridor A* XY path: cells≈{navigation_phase2_astar_cell_m:.3f}m "
                f"inflate_pad={navigation_phase2_astar_obstacle_pad_m:.3f}m waypoints={pts_arr.shape[0]}"
            )
        else:
            if not _try_z_clearance_plan("top-down XY corridor A* unsolved"):
                pts_arr = np.stack(
                    [
                        np.array([sx, sy, sz], dtype=np.float32),
                        np.array([gx, gy, gz], dtype=np.float32),
                    ],
                    axis=0,
                )
                planner_primary = "los_fallback_start_goal"
                print(
                    "[Phase 2] WARN: A* failed inside corridor — using straight start→goal fallback "
                    "(may cut inflated obstacles / or workspace cap blocks overflight)."
                )
    else:
        pts_arr = np.stack(
            [
                np.array([sx, sy, sz], dtype=np.float32),
                np.array([gx, gy, gz], dtype=np.float32),
            ],
            axis=0,
        )
        planner_primary = "los_geom_astar_disabled_config"
        print("[Phase 2] navigation_phase2_geom_astar=false — using LOS start→goal only.")
        if _xy_segment_underclear_at_cruise() and _try_z_clearance_plan("A* disabled and XY segment below obstacle tops"):
            pass

    if navigation_phase2_optional_topdown_xvla:
        phase2_task = (
            "[Navigation Phase 2 OPTIONAL — TOP-DOWN RGB supplement]\n"
            "Birds-eye image; green corridor between START and TARGET.\n\n"
            f"Original mission (--cmd):\n{mission_txt}\n\n"
            f"Effective flight instruction:\n{language_instruction.strip()}\n\n"
            "World-frame anchors (meters):\n"
            f"- start XY: ({sx:.4f}, {sy:.4f}); drone Z≈{sz:.4f}\n"
            f"- goal XYZ: ({gx:.4f}, {gy:.4f}, {gz:.4f})\n"
            f"- corridor XY bounds: x∈[{xmin:.4f},{xmax:.4f}], y∈[{ymin:.4f},{ymax:.4f}]\n\n"
            "Obstacle XY footprints:\n"
            f"{obs_lines_text}\n\n"
            "Provide auxiliary ABSOLUTE EE targets consistent with geometric corridor planning.\n"
        )
        extra = str(phase2_extra_instruction).strip()
        if extra:
            phase2_task += "\nAdditional operator hint:\n" + extra + "\n"

        composed = compose_xvla_navigation_instruction(
            phase2_task.strip(),
            scene_catalog=scene_catalog_str,
            enabled=bool(xvla_scene_semantic_context and scene_catalog_str.strip()),
            planning_suffix=xvla_path_planning_instruction_suffix,
        )

        td = cv2.resize(
            np.asarray(topdown_rgb, dtype=np.uint8),
            (max(8, int(workspace_camera_width)), max(8, int(workspace_camera_height))),
            interpolation=cv2.INTER_AREA,
        )
        proprio = build_proprio_widowx_ee6d(
            np.asarray(drone_pos_local, dtype=np.float32),
            np.asarray(drone_R, dtype=np.float32),
            gripper=float(gripper_state),
        )
        print("\n[Phase 2] Optional top-down X-VLA /act (does not replace A* polyline unless integrated downstream) …")
        print(f"[Phase 2] xvla_steps={steps} infer_timeout_s={xvla_act_request_timeout_s:.0f}")
        try:
            t_req = time.perf_counter()
            actions = query_xvla(
                server_url,
                td,
                proprio,
                composed,
                steps=steps,
                timeout=float(xvla_act_request_timeout_s),
            )
            infer_ms = float((time.perf_counter() - t_req) * 1000.0)
            print(f"[Phase 2] optional top-down infer_ms={infer_ms:.1f}")
        except Exception as exc:
            print(f"[Phase 2] optional top-down X-VLA failed: {exc}")

    if pts_arr is not None and pts_arr.shape[0] >= 1:
        print("[Phase 2] XY keypoints — world frame (meters), primary planner:")
        print(f"  planner={planner_primary}")
        for i, row in enumerate(pts_arr):
            print(
                f"  k{i}: XY=[{float(row[0]):.6f}, {float(row[1]):.6f}]  Z={float(row[2]):.6f}"
            )

        pix_ln: list[tuple[int, int]] = []
        for row in pts_arr:
            wx, wy, wz = float(row[0]), float(row[1]), float(row[2])
            pix = world_xyz_to_recording_image_pixel(
                np.array([wx, wy, wz], dtype=np.float64),
                view_m,
                proj_m,
                width=int(render_width),
                height=int(render_height),
            )
            if pix is not None:
                pix_ln.append((int(pix[0]), int(pix[1])))
        blue_bgr = (255, 0, 0)
        if len(pix_ln) >= 2:
            arr_pix = np.array(pix_ln, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                img_marked_bgr,
                [arr_pix],
                isClosed=False,
                color=blue_bgr,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
        for xy in pix_ln:
            cv2.circle(img_marked_bgr, xy, 4, blue_bgr, -1, lineType=cv2.LINE_AA)

        out_png = phase1_folder / png_filename
        cv2.imwrite(str(out_png), img_marked_bgr)
        print(f"[Phase 2] Saved top-down PNG with blue (BGR {blue_bgr}) path: {out_png}")

        stereo_saved: str | None = None
        if (
            stereo_img_marked_bgr is not None
            and stereo_view_m is not None
            and stereo_proj_m is not None
        ):
            pix_st: list[tuple[int, int]] = []
            for row in pts_arr:
                wx, wy, wz = float(row[0]), float(row[1]), float(row[2])
                pix = world_xyz_to_recording_image_pixel(
                    np.array([wx, wy, wz], dtype=np.float64),
                    stereo_view_m,
                    stereo_proj_m,
                    width=int(render_width),
                    height=int(render_height),
                )
                if pix is not None:
                    pix_st.append((int(pix[0]), int(pix[1])))
            st_bgr = np.asarray(stereo_img_marked_bgr, dtype=np.uint8).copy()
            if len(pix_st) >= 2:
                arr_s = np.array(pix_st, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    st_bgr,
                    [arr_s],
                    isClosed=False,
                    color=blue_bgr,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
            for xy in pix_st:
                cv2.circle(st_bgr, xy, 4, blue_bgr, -1, lineType=cv2.LINE_AA)
            st_name = str(stereo_png_filename).strip() or "global_stereo45deg_detection_rectangle.png"
            st_path = phase1_folder / st_name
            cv2.imwrite(str(st_path), st_bgr)
            stereo_saved = str(st_path)
            print(f"[Phase 2] Saved stereo (45°) PNG with blue (BGR {blue_bgr}) path: {st_path}")

        snapshot = {
            "phase": "navigation_phase2_xy",
            "primary_planner": planner_primary,
            "geom_astar_enabled": bool(navigation_phase2_geom_astar),
            "optional_topdown_xvla": bool(navigation_phase2_optional_topdown_xvla),
            "astar_cell_m": float(navigation_phase2_astar_cell_m),
            "astar_obstacle_pad_m": float(navigation_phase2_astar_obstacle_pad_m),
            "z_clearance_enabled": bool(navigation_phase2_z_clearance_enabled),
            "z_clearance_margin_m": float(navigation_phase2_z_clearance_margin_m),
            "z_workspace_cap_world": float(z_cap_world),
            "z_clearance_plan_meta": z_clearance_meta,
            "saved_at": datetime.datetime.now().isoformat(),
            "corridor_xy": {"x_min": xmin, "y_min": ymin, "x_max": xmax, "y_max": ymax},
            "goal_xyz_world": [gx, gy, gz],
            "start_xy_world": [sx, sy],
            "infer_ms_optional_xvla": infer_ms,
            "xvla_steps_requested": steps,
            "xy_keypoints_world": [[float(r[0]), float(r[1])] for r in pts_arr],
            "xyz_keypoints_world": pts_arr.astype(float).tolist(),
            "mission_cmd": mission_txt,
            "language_instruction_excerpt": language_instruction.strip()[:800],
            "saved_topdown_png": str(out_png),
            "saved_stereo_png": stereo_saved,
            "stereo_png_filename": str(stereo_png_filename) if stereo_saved else None,
        }
        sidecar = phase1_folder / "navigation_phase2_xy.json"
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        print(f"[Phase 2] Sidecar JSON: {sidecar}")

        if sync_root_config:
            try:
                cfg = read_config_json(config_json_path)
                cfg["_navigation_phase2_snapshot"] = snapshot
                with open(config_json_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                print(f"[Phase 2] Wrote `_navigation_phase2_snapshot` into {config_json_path}")
            except Exception as exc:
                print(f"[Phase 2] WARNING: could not merge config.json — {exc}")

        if sync_qs and qs_policy_path is not None:
            try:
                policies = load_qs_policies(qs_policy_path)
                summary_s = (
                    f"[auto Phase2 {snapshot['saved_at']}] planner={planner_primary} "
                    f"waypoints={pts_arr.shape[0]} corridor [{xmin:.3f},{xmax:.3f}]×[{ymin:.3f},{ymax:.3f}]; "
                    "geometry-first corridor A* with inflated Phase1 footprints; optional top-down X-VLA auxiliary."
                )
                policies.append(
                    {
                        "Q": "Automatic Phase-2 XY corridor planning (runtime snapshot)",
                        "S": summary_s,
                    }
                )
                with open(qs_policy_path, "w", encoding="utf-8") as f:
                    json.dump(policies, f, indent=4, ensure_ascii=False)
                print(f"[Phase 2] Appended Phase-2 policy entry to {qs_policy_path}")
            except Exception as exc:
                print(f"[Phase 2] WARNING: could not append QS.json — {exc}")
    else:
        print("[Phase 2] No drawable trajectory; screenshot not updated for polyline.")

    print("[Phase 2] Completed. Stopping for debugging.\n")


def local_obstacle_repulsion_world(
    drone_pos: np.ndarray,
    placed_cubes: list[dict],
    *,
    robot_radius: float,
    influence_m: float,
    gain: float,
    max_delta_m: float,
    exclude_near_points: list[np.ndarray] | None = None,
    exclude_tol_m: float = 0.10,
) -> np.ndarray:

    if not placed_cubes or influence_m <= 1e-9 or gain <= 0.0:
        return np.zeros(3, dtype=np.float32)
    p = np.asarray(drone_pos, dtype=np.float64).reshape(3)
    r = max(0.0, float(robot_radius))
    inf = float(influence_m)
    g = float(gain)
    cap = float(max(0.0, max_delta_m))
    excl = exclude_near_points or []
    acc = np.zeros(3, dtype=np.float64)
    for c in placed_cubes:
        cen = np.asarray(c["pos"], dtype=np.float64).reshape(3)
        skip = False
        for ep in excl:
            ee = np.asarray(ep, dtype=np.float64).reshape(3)
            if float(np.linalg.norm(cen - ee)) <= float(exclude_tol_m):
                skip = True
                break
        if skip:
            continue
        lo0, hi0 = _placed_object_aabb_lo_hi_world(c)
        lo = lo0 - r
        hi = hi0 + r

        dist_s, direc = _point_to_aabb_vector_outward(p, lo, hi)

        if dist_s < 0.0:
            acc += direc * (g * 15.0)
            continue

        w = 1.0 - (float(dist_s) / inf)
        if w <= 0.0:
            continue

        w_scaled = w * w
        if dist_s < inf * 0.3:
            w_scaled = w * w * w * 2.0

        acc += direc * (g * w_scaled)
    n = float(np.linalg.norm(acc))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float32)
    if cap > 1e-9 and n > cap:
        acc = acc * (cap / n)
    return acc.astype(np.float32)


def _remap_polyline_xy_fill_rect(
    pts_world: np.ndarray,
    lo_tgt_xy: np.ndarray,
    hi_tgt_xy: np.ndarray,
) -> np.ndarray:

    out = np.asarray(pts_world, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] < 2:
        return np.asarray(pts_world, dtype=np.float32)
    xy = out[:, 0:2]
    lo_s = xy.min(axis=0)
    hi_s = xy.max(axis=0)
    span = np.maximum(hi_s - lo_s, 1e-6)
    t_lo = np.asarray(lo_tgt_xy, dtype=np.float64).reshape(2)
    t_hi = np.asarray(hi_tgt_xy, dtype=np.float64).reshape(2)
    t_sp = np.maximum(t_hi - t_lo, 1e-9)
    out[:, 0] = t_lo[0] + (out[:, 0] - lo_s[0]) / span[0] * t_sp[0]
    out[:, 1] = t_lo[1] + (out[:, 1] - lo_s[1]) / span[1] * t_sp[1]
    return out.astype(np.float32)


def _interpolate_pose_along_polyline(
    pts_world: np.ndarray,
    Rs: list[np.ndarray],
    u: float,
) -> tuple[np.ndarray, np.ndarray]:

    p = np.asarray(pts_world, dtype=np.float64)
    if p.size == 0:
        raise ValueError("empty polyline")
    n = p.shape[0]
    u = float(np.clip(u, 0.0, 1.0))
    if n == 1:
        return p[0].astype(np.float32), np.asarray(Rs[0], dtype=np.float32)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        return p[0].astype(np.float32), np.asarray(Rs[0], dtype=np.float32)
    d = u * total
    j = int(np.searchsorted(cum, d, side="right") - 1)
    j = int(np.clip(j, 0, n - 2))
    sl = float(seg[j])
    t = 0.0 if sl < 1e-9 else float(np.clip((d - cum[j]) / sl, 0.0, 1.0))
    pw = ((1.0 - t) * p[j] + t * p[j + 1]).astype(np.float32)


    e0 = np.array(euler_xyz_from_matrix(Rs[j]), dtype=np.float64)
    e1 = np.array(euler_xyz_from_matrix(Rs[j + 1]), dtype=np.float64)
    ew = (1.0 - t) * e0 + t * e1
    Rw = matrix_from_euler_xyz(float(ew[0]), float(ew[1]), float(ew[2]))
    return pw, Rw.astype(np.float32)


def boost_language_only_local_target(
    drone_local: np.ndarray,
    target_local: np.ndarray,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    *,
    amplify: float,
    min_step_m: float,
) -> np.ndarray:

    dl = np.asarray(drone_local, dtype=np.float64).reshape(3)
    tl = np.asarray(target_local, dtype=np.float64).reshape(3)
    dvec = tl - dl
    dist = float(np.linalg.norm(dvec))
    amp = float(np.clip(amplify, 0.0, 50.0))
    if amp != 1.0 and dist > 1e-9:
        tl = dl + dvec * amp
    ws_lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    ws_hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    tl = np.minimum(np.maximum(tl, ws_lo), ws_hi)
    d2 = tl - dl
    dist2 = float(np.linalg.norm(d2))
    mns = float(max(0.0, min_step_m))
    if mns > 0.0 and dist2 > 1e-9 and dist2 < mns:
        tl = dl + (d2 / dist2) * mns
        tl = np.minimum(np.maximum(tl, ws_lo), ws_hi)
    return tl.astype(np.float32)


def scale_infer_local_displacement(
    drone_local: np.ndarray,
    target_local: np.ndarray,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:

    s = float(np.clip(scale, 0.0, 50.0))
    if abs(s - 1.0) < 1e-9:
        return np.asarray(target_local, dtype=np.float32)
    dl = np.asarray(drone_local, dtype=np.float64).reshape(3)
    tl = np.asarray(target_local, dtype=np.float64).reshape(3)
    out = dl + (tl - dl) * s
    ws_lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    ws_hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    out = np.minimum(np.maximum(out, ws_lo), ws_hi)
    return out.astype(np.float32)


def gate_rotation_matrix_from_xvla(
    *,
    server_url: str,
    image_rgb: np.ndarray,
    proprio_20d: np.ndarray,
    language_instruction: str,
    steps: int,
    infer_width: int,
    infer_height: int,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    treat_pos_as: str,
    delta_pos_scale: float,
    act_request_timeout_s: float = 300.0,
) -> np.ndarray | None:

    try:
        im = image_rgb
        ih, iw = int(im.shape[0]), int(im.shape[1])
        if ih != int(infer_height) or iw != int(infer_width):
            try:
                import cv2

                im = cv2.resize(
                    im,
                    (int(infer_width), int(infer_height)),
                    interpolation=cv2.INTER_AREA,
                )
            except Exception:
                return None
        actions = query_xvla(
            server_url,
            im,
            proprio_20d,
            language_instruction,
            steps=int(steps),
            timeout=float(act_request_timeout_s),
        )
        curp = np.asarray(proprio_20d, dtype=np.float32).reshape(-1)
        if curp.size < 3:
            return None
        _tp, Rm, _g = decode_action_widowx_ee6d(
            actions,
            workspace_lo=workspace_lo,
            workspace_hi=workspace_hi,
            treat_pos_as=treat_pos_as,
            current_pos=curp[0:3],
            delta_pos_scale=delta_pos_scale,
        )
        return np.asarray(Rm, dtype=np.float32)
    except Exception:
        return None


def choose_virtual_base_world(
    drone_world: np.ndarray,
    goal_world: np.ndarray | None,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    prev_base_world: np.ndarray,
    *,
    margin: float,
) -> np.ndarray:

    lo = np.asarray(workspace_lo, dtype=np.float64)
    hi = np.asarray(workspace_hi, dtype=np.float64)
    center = (lo + hi) * 0.5
    eff_lo = lo + float(margin)
    eff_hi = hi - float(margin)
    if np.any(eff_hi <= eff_lo):
        eff_lo, eff_hi = lo, hi

    drone = np.asarray(drone_world, dtype=np.float64)
    prev = np.asarray(prev_base_world, dtype=np.float64)
    desired = drone - center
    if goal_world is None:
        return desired.astype(np.float32)

    goal = np.asarray(goal_world, dtype=np.float64)
    lower = np.maximum(drone - eff_hi, goal - eff_hi)
    upper = np.minimum(drone - eff_lo, goal - eff_lo)

    out = desired.copy()
    fit = lower <= upper
    out[fit] = np.minimum(np.maximum(prev[fit], lower[fit]), upper[fit])
    return out.astype(np.float32)


def cmd_goal_world_estimate_for_virtual_base(
    mission_cmd: str | None,
    placed_cubes: list[dict],
    *,
    prefer_near_xyz: np.ndarray,
) -> np.ndarray | None:

    if not mission_cmd or not str(mission_cmd).strip() or not placed_cubes:
        return None
    sp = _first_rect_portal_for_instruction_color(
        str(mission_cmd).strip(),
        placed_cubes,
        prefer_near_xyz=np.asarray(prefer_near_xyz, dtype=np.float64),
    )
    if sp is None:
        return None
    return np.asarray(sp["pos"], dtype=np.float64).reshape(3)


def drone_goal_both_inside_workspace_local(
    drone_world: np.ndarray,
    goal_world: np.ndarray,
    vb: np.ndarray,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    *,
    margin: float = 0.0,
) -> bool:
    lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3) + float(margin)
    hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3) - float(margin)
    if np.any(hi <= lo):
        lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
        hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    vbv = np.asarray(vb, dtype=np.float64).reshape(3)
    dl = np.asarray(drone_world, dtype=np.float64).reshape(3) - vbv
    gl = np.asarray(goal_world, dtype=np.float64).reshape(3) - vbv

    def _in(x: np.ndarray) -> bool:
        return bool(np.all(x >= lo) and np.all(x <= hi))

    return _in(dl) and _in(gl)


def smooth_and_cap_virtual_base_step(
    prev_vb: np.ndarray,
    raw_vb: np.ndarray,
    dt: float,
    *,
    alpha: float,
    max_speed_m_s: float,
) -> np.ndarray:

    prev = np.asarray(prev_vb, dtype=np.float64).reshape(3)
    raw = np.asarray(raw_vb, dtype=np.float64).reshape(3)
    a = float(np.clip(alpha, 0.0, 1.0))
    tgt = prev + a * (raw - prev)
    max_d = float(max(0.0, max_speed_m_s)) * float(max(dt, 1e-9))
    delta = tgt - prev
    norm = float(np.linalg.norm(delta))
    if norm > max_d > 0.0:
        tgt = prev + delta * (max_d / norm)
    return tgt.astype(np.float32)


def obstacle_xy_boxes_phase1_entries(
    entries: list[Any],
    *,
    inflate_m: float,
    exclude_goal_fn: Callable[[Any], bool] | None = None,
) -> list[tuple[float, float, float, float]]:

    boxes: list[tuple[float, float, float, float]] = []
    pad = float(max(0.0, inflate_m))
    for obj in entries:
        if exclude_goal_fn is not None:
            try:
                if exclude_goal_fn(obj):
                    continue
            except Exception:
                pass
        cdict: dict | Any = obj
        if isinstance(obj, tuple) and len(obj) >= 2:
            cdict = obj[-1]
        if isinstance(cdict, dict) and "aabb" in cdict:
            lo = np.asarray(cdict["aabb"][0], dtype=np.float64)
            hi = np.asarray(cdict["aabb"][1], dtype=np.float64)
        elif isinstance(cdict, dict):
            lo, hi = _placed_object_aabb_lo_hi_world(cdict)
        else:
            continue
        boxes.append(
            (
                float(lo[0]) - pad,
                float(lo[1]) - pad,
                float(hi[0]) + pad,
                float(hi[1]) + pad,
            )
        )
    return boxes


def plan_xy_astar_corridor_world(
    corridor_xy: tuple[float, float, float, float],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    obstacle_boxes_xy: list[tuple[float, float, float, float]],
    *,
    cell_m: float,
    max_grid_dim: int = 384,
) -> list[tuple[float, float]] | None:

    xmin, ymin, xmax, ymax = corridor_xy
    xmin, ymin = float(min(xmin, xmax)), float(min(ymin, ymax))
    xmax, ymax = float(max(xmin, xmax)), float(max(ymin, ymax))
    cell = float(max(1e-4, cell_m))
    span_x = xmax - xmin
    span_y = ymax - ymin
    if span_x < 1e-9 or span_y < 1e-9:
        return None
    nx = int(min(max_grid_dim, max(4, math.ceil(span_x / cell))))
    ny = int(min(max_grid_dim, max(4, math.ceil(span_y / cell))))
    dx = span_x / float(nx)
    dy = span_y / float(ny)

    def cell_blocked(ix: int, iy: int) -> bool:
        cx = xmin + (ix + 0.5) * dx
        cy = ymin + (iy + 0.5) * dy
        for ox0, oy0, ox1, oy1 in obstacle_boxes_xy:
            if cx >= ox0 and cx <= ox1 and cy >= oy0 and cy <= oy1:
                return True
        return False

    def to_ij(x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip(round((float(x) - xmin) / dx - 0.5), 0, nx - 1))
        iy = int(np.clip(round((float(y) - ymin) / dy - 0.5), 0, ny - 1))
        return ix, iy

    def nearest_free(ix: int, iy: int, wx: float, wy: float) -> tuple[int, int]:
        if not cell_blocked(ix, iy):
            return ix, iy
        best = (ix, iy)
        bd = float("inf")
        for ii in range(nx):
            for jj in range(ny):
                if cell_blocked(ii, jj):
                    continue
                cx = xmin + (ii + 0.5) * dx
                cy = ymin + (jj + 0.5) * dy
                d = (cx - wx) ** 2 + (cy - wy) ** 2
                if d < bd:
                    bd = d
                    best = (ii, jj)
        return best

    sx, sy = float(start_xy[0]), float(start_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    i0, j0 = to_ij(sx, sy)
    i1, j1 = to_ij(gx, gy)
    i0, j0 = nearest_free(i0, j0, sx, sy)
    i1, j1 = nearest_free(i1, j1, gx, gy)
    if cell_blocked(i0, j0) or cell_blocked(i1, j1):
        return None

    start_h = math.hypot(i1 - i0, j1 - j0)
    heap: list[tuple[float, float, int, int]] = [(start_h, 0.0, i0, j0)]
    came: dict[tuple[int, int], tuple[int, int] | None] = {(i0, j0): None}
    gscore: dict[tuple[int, int], float] = {(i0, j0): 0.0}

    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    while heap:
        _f, gc, i, j = heapq.heappop(heap)
        if (i, j) == (i1, j1):
            path_ij: list[tuple[int, int]] = []
            cur: tuple[int, int] | None = (i, j)
            while cur is not None:
                path_ij.append(cur)
                cur = came[cur]
            path_ij.reverse()
            way_xy: list[tuple[float, float]] = []
            for ix, iy in path_ij:
                cx = xmin + (ix + 0.5) * dx
                cy = ymin + (iy + 0.5) * dy
                way_xy.append((cx, cy))
            if len(way_xy) <= 2:
                return way_xy
            slim = [way_xy[0]]
            for k in range(1, len(way_xy) - 1):
                x0, y0 = slim[-1]
                x1, y1 = way_xy[k]
                x2, y2 = way_xy[k + 1]
                cross = (y1 - y0) * (x2 - x1) - (x1 - x0) * (y2 - y1)
                if abs(cross) > 1e-9 * max(
                    1.0,
                    math.hypot(x1 - x0, y1 - y0),
                    math.hypot(x2 - x1, y2 - y1),
                ):
                    slim.append((x1, y1))
            slim.append(way_xy[-1])
            return slim

        if gc > gscore.get((i, j), float("inf")) + 1e-9:
            continue

        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                continue
            if cell_blocked(ni, nj):
                continue
            step = math.hypot(di * dx, dj * dy)
            tentative = gc + step
            key = (ni, nj)
            if tentative < gscore.get(key, float("inf")):
                came[key] = (i, j)
                gscore[key] = tentative
                h = math.hypot(i1 - ni, j1 - nj) * min(dx, dy)
                heapq.heappush(heap, (tentative + h, tentative, ni, nj))

    return None


def phase2_trajectory_xyz_from_astar(
    corridor_xy: tuple[float, float, float, float],
    start_xyz_world: np.ndarray,
    goal_xyz_world: np.ndarray,
    obstacle_boxes_xy: list[tuple[float, float, float, float]],
    *,
    cell_m: float,
) -> np.ndarray | None:

    sw = np.asarray(start_xyz_world, dtype=np.float64).reshape(-1)
    gw = np.asarray(goal_xyz_world, dtype=np.float64).reshape(-1)
    xy_path = plan_xy_astar_corridor_world(
        corridor_xy,
        (float(sw[0]), float(sw[1])),
        (float(gw[0]), float(gw[1])),
        obstacle_boxes_xy,
        cell_m=float(cell_m),
    )
    if xy_path is None:
        return None
    n = len(xy_path)
    sz = float(sw[2])
    gz = float(gw[2])
    out = np.zeros((n, 3), dtype=np.float32)
    for i, (x, y) in enumerate(xy_path):
        u = float(i) / float(max(1, n - 1))
        out[i] = np.array([x, y, (1.0 - u) * sz + u * gz], dtype=np.float32)
    return out


def phase1_entries_world_aabbs(entries: list[Any]) -> list[tuple[np.ndarray, np.ndarray]]:

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for obj in entries:
        cdict: dict | Any = obj
        if isinstance(obj, tuple) and len(obj) >= 2:
            cdict = obj[-1]
        if isinstance(cdict, dict) and "aabb" in cdict:
            lo = np.asarray(cdict["aabb"][0], dtype=np.float64)
            hi = np.asarray(cdict["aabb"][1], dtype=np.float64)
        elif isinstance(cdict, dict):
            lo, hi = _placed_object_aabb_lo_hi_world(cdict)
        else:
            continue
        out.append((lo, hi))
    return out


def _segments_intersect_proper_2d(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
    *,
    eps: float = 1e-9,
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
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rx0: float,
    ry0: float,
    rx1: float,
    ry1: float,
    *,
    eps: float = 1e-9,
) -> bool:

    a, b = min(rx0, rx1), max(rx0, rx1)
    c, d = min(ry0, ry1), max(ry0, ry1)

    def inside(x: float, y: float) -> bool:
        return a - eps <= x <= b + eps and c - eps <= y <= d + eps

    if inside(x0, y0) or inside(x1, y1):
        return True
    if abs(x0 - x1) < eps and abs(y0 - y1) < eps:
        return False
    edges = (
        (a, c, b, c),
        (b, c, b, d),
        (b, d, a, d),
        (a, d, a, c),
    )
    for ex0, ey0, ex1, ey1 in edges:
        if _segments_intersect_proper_2d(x0, y0, x1, y1, ex0, ey0, ex1, ey1, eps=eps):
            return True
    return False


def min_z_clearance_for_world_xy_segment(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    world_aabbs: list[tuple[np.ndarray, np.ndarray]],
    *,
    inflate_xy_m: float,
) -> tuple[float, int]:

    pad = float(max(0.0, inflate_xy_m))
    sx, sy = float(start_xy[0]), float(start_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    z_top_max = -float("inf")
    hit = 0
    for lo, hi in world_aabbs:
        lo = np.asarray(lo, dtype=np.float64).reshape(3)
        hi = np.asarray(hi, dtype=np.float64).reshape(3)
        bx0, by0 = float(lo[0]) - pad, float(lo[1]) - pad
        bx1, by1 = float(hi[0]) + pad, float(hi[1]) + pad
        if _segment_intersects_rect_2d(sx, sy, gx, gy, bx0, by0, bx1, by1):
            z_top_max = max(z_top_max, float(hi[2]))
            hit += 1
    if hit == 0:
        return float("-inf"), 0
    return float(z_top_max), int(hit)


def phase2_trajectory_xyz_z_clearance_overfly(
    start_xyz_world: np.ndarray,
    goal_xyz_world: np.ndarray,
    phase1_obstacle_entries: list[Any],
    *,
    inflate_xy_m: float,
    z_clearance_margin_m: float,
    z_workspace_cap_world: float | None,
) -> tuple[np.ndarray, dict[str, Any]] | None:

    sw = np.asarray(start_xyz_world, dtype=np.float64).reshape(-1)
    gw = np.asarray(goal_xyz_world, dtype=np.float64).reshape(-1)
    if sw.shape[0] < 3 or gw.shape[0] < 3:
        return None
    sx, sy, sz = float(sw[0]), float(sw[1]), float(sw[2])
    gx, gy, gz = float(gw[0]), float(gw[1]), float(gw[2])
    aabbs = phase1_entries_world_aabbs(phase1_obstacle_entries)
    z_top_along, n_hit = min_z_clearance_for_world_xy_segment(
        (sx, sy), (gx, gy), aabbs, inflate_xy_m=float(inflate_xy_m)
    )
    zm = float(max(0.0, z_clearance_margin_m))
    z_need_top = float(z_top_along) + zm if n_hit > 0 else max(sz, gz)
    z_cruise = max(sz, gz, z_need_top)
    meta = {
        "obstacles_along_xy_segment": int(n_hit),
        "z_obstacle_top_world": float(z_top_along) if n_hit > 0 else None,
        "z_cruise_world": float(z_cruise),
        "z_clearance_margin_m": float(zm),
    }
    if z_workspace_cap_world is not None and z_cruise > float(z_workspace_cap_world) + 1e-6:
        return None
    z_tol = 1e-3
    xy_move = (gx - sx) ** 2 + (gy - sy) ** 2 > 1e-8
    raw: list[np.ndarray] = []

    def add_pt(x: float, y: float, z: float) -> None:
        p = np.array([x, y, z], dtype=np.float64)
        if not raw:
            raw.append(p)
            return
        if float(np.linalg.norm(p - raw[-1])) < 1e-4:
            return
        raw.append(p)

    add_pt(sx, sy, sz)
    if abs(z_cruise - sz) > z_tol:
        add_pt(sx, sy, z_cruise)
    if xy_move:
        add_pt(gx, gy, z_cruise)
    if abs(z_cruise - gz) > z_tol:
        add_pt(gx, gy, gz)
    elif not xy_move and abs(z_cruise - sz) <= z_tol:
        add_pt(gx, gy, gz)
    if len(raw) == 1:
        add_pt(gx, gy, gz)
    out = np.stack([r.astype(np.float32) for r in raw], axis=0)
    return out, meta


def _language_cmd_suggests_air_path_or_cruise(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if "figure" in s and ("eight" in s or "8" in s):
        return True
    needles = (
        "in the air",
        "in the sky",
        "above the blocks",
        "above all blocks",
        "airspace",
        "above the workspace",
        "above the table",
        "over the workspace",
        "over the table",
        "patrol",
        "orbit",
        "circuit",
        "racetrack",
        "oval",
        "ellipse",
        "diamond",
        "rhombus",
        "smooth path",
        "smooth circuit",
        "serpentine",
        "back and forth",
        " sweep",
        "sweep ",
        "cruise altitude",
        "fly a path",
        "hold altitude",
        "loop around",
        "spiral",
        "spiralling",
        "zigzag",
        "zig zag",
        "boustrophedon",
        "clover",
        "trefoil",
        "pentagram",
        "lissajous",
        "bow tie",
        "bowtie",
        "heart shaped",
        "heart-shaped",
    )
    if any(n in s for n in needles):
        return True
    if " circle" in s or s.startswith("circle ") or " circle " in s:
        return True
    return False


def _apply_language_only_cruise_z_to_local_target(
    target_pos_local: np.ndarray,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    *,
    margin_lo_frac: float,
    margin_hi_frac: float,
) -> np.ndarray:

    out = np.asarray(target_pos_local, dtype=np.float32).reshape(3).copy()
    z0 = float(workspace_lo[2])
    z1 = float(workspace_hi[2])
    span = max(z1 - z0, 1e-6)
    mlo = float(np.clip(margin_lo_frac, 0.0, 0.45))
    mhi = float(np.clip(margin_hi_frac, 0.0, 0.45))
    z_lo = z0 + mlo * span
    z_hi = z1 - mhi * span
    if z_hi <= z_lo + 1e-4:
        return out
    out[2] = float(np.clip(float(out[2]), z_lo, z_hi))
    return out


def _scene_object_max_height_above_floor(
    placed_cubes: list[dict],
    z_floor_world: float,
) -> tuple[float, float]:

    z_floor = float(z_floor_world)
    z_top_max = z_floor
    for c in placed_cubes:
        pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
        bh = c.get("bounds_half")
        if bh is not None:
            h = np.asarray(bh, dtype=np.float64).reshape(3)
            z_top_max = max(z_top_max, float(pos[2] + h[2]))
        else:
            half = float(c.get("half", 0.025))
            z_top_max = max(z_top_max, float(pos[2] + half))
    H_max = max(float(z_top_max - z_floor), 0.02)
    return float(z_top_max), H_max


def _language_cmd_demo_trajectory_no_portal(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return False
    if re.search(r"\bthrough\s+(the\s+)?(portal|frame|opening|slot|gate)\b", s):
        return False
    if re.search(r"\balign\s+(with|to)\s+the\b", s):
        return False
    if re.search(r"\bportal\s+crossing\b", s):
        return False
    if re.search(r"\bbillboard(?:\s*_?\s*id)?\s*[=:#]?\s*\d", s):
        return False
    if re.search(r"\bportal\s+(?:number|billboard(?:\s+number)?)\s*\d", s):
        return False
    if re.search(r"\b(orbit|circle|hover|inspect|recon|surveillance|scout)\b", s) and re.search(
        r"\b(portal|billboard|gate|frame|opening)\b", s
    ):
        return False
    if _language_cmd_requests_serpentine_scan(text):
        return True
    if "figure" in s and ("eight" in s or "8" in s):
        return True
    demo_hints = (
        "demonstrat",
        "pattern",
        "smooth path",
        "smooth circuit",
        "racetrack",
        "oval",
        "ellipse",
        "elliptic",
        "diamond",
        "rhombus",
        "lozenge",
        "trace a",
        "trace an",
        "in the air",
        "in the sky",
        "above the blocks",
        "above all blocks",
        "airspace",
        "above the workspace",
        "above the table",
        "over the workspace",
        "over the table",
        "orbit",
        "loop around",
        "fly a path",
        "figure eight",
        "serpentine",
        "back and forth",
        "sweep",
        "spiral",
        "spiralling",
        "zigzag",
        "zig zag",
        "boustrophedon",
        "clover",
        "trefoil",
        "pentagram",
        "lissajous",
        "bow tie",
        "heart shaped",
        "heart-shaped",
        "sine wave",
        "cosine wave",
        "cos wave",
        "beat wave",
        "damped sine",
        "triangle wave",
        "triangular wave",
        "tri wave",
        "tanh",
        "hyperbolic tangent",
        "squircle",
        "cycloid",
        "cardioid",
        "deltoid",
        "astroid",
        "lemniscate",
        "hypocycloid",
        "hypotrochoid",
        "epicycloid",
        "trochoid",
        "log spiral",
        "hyperbolic spiral",
        "involute",
        "folium",
        "tractrix",
        "clothoid",
        "cassini oval",
        "agnesi",
        "cornu spiral",
        "archimedean spiral",
        "parabolic arc",
        "euler spiral",
        "witch of agnesi",
        "parametric curve",
        "decorative orbit",
        "squircle orbit",
        "corkscrew",
        "helical",
        "helix",
        "3d spiral",
        "3d sine",
        "3d cosine",
        "oscillating altitude",
        "altitude cosine",
        "vertical cosine",
        "vertical sine",
        "holding pattern",
        "out and back",
        "crisscross",
        "cross pattern",
        "teardrop",
        "tear drop",
        "phyllotaxis",
        "sunflower pattern",
        "golden angle spiral",
        "golden angle pattern",
        "polygon",
        "triangle circuit",
        "hexagonal path",
        "rose curve",
        "rhodonea",
        "butterfly curve",
        "Gaussian",
        "catenary curve",
        "scallop orbit",
        "rounded rectangle perimeter",
        "pill shaped circuit",
        "epitrochoid",
        "nephroid",
        "limaçon",
        "lima con",
        "limacon",
        "cissoid",
        "diocles curve",
        "kidney-shaped",
        "strophoid",
        "kampyle",
        "eudoxus",
        "conchoid",
        "nicomedes",
    )
    return any(h in s for h in demo_hints)


def _language_cmd_requests_figure8(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    return "figure" in s and ("eight" in s or "8" in s)


def _language_cmd_requests_oval_racetrack(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if _language_cmd_requests_figure8(text):
        return False
    if "racetrack" in s:
        return True
    if re.search(r"\boval\b", s) or "oval-shaped" in s:
        return True
    if "ellipse" in s or "elliptic" in s:
        return True
    if "stadium" in s and ("track" in s or "loop" in s or "circuit" in s):
        return True
    return False


def _language_cmd_requests_circle_orbit_uav(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if _language_cmd_requests_figure8(text):
        return False
    if _language_cmd_requests_oval_racetrack(text):
        return False
    if not (re.search(r"\bcircle\b", s) or re.search(r"\bcircular\b", s)):
        return False
    return bool(
        ("orbit" in s)
        or ("loop" in s and "around" in s)
        or re.search(r"\bfly\b", s)
        or ("path" in s)
        or ("circuit" in s)
        or ("patrol" in s)
        or ("around" in s)
        or re.search(r"\bhover\b", s)
        or ("holding pattern" in s)
    )


def _language_cmd_requests_helical_vertical_profile(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    hits = (
        "corkscrew",
        "helix",
        "helical",
        "3d spiral",
        "3-d spiral",
        "vertical spiral",
        "climbing spiral",
        "descending spiral",
        "altitude weave",
        "vertical sine",
        "vertical cosine",
        "altitude sine",
        "altitude cosine",
        "oscillating altitude",
        "altitude oscillation",
        "bobbing altitude",
        "vertical bob",
        "z sine",
        "z cosine",
        "3d sine",
        "3d cosine",
        "three dimensional sine",
        "three dimensional cosine",
    )
    return any(h in s for h in hits)


def _infer_helical_turns_across_path(text: str) -> float:

    sk = str(text).lower()
    mk = re.search(r"(\d+(?:\.\d+)?)\s*(?:turn|turns|rotations?|cycles?)\b", sk)
    if mk:
        return float(np.clip(float(mk.group(1)), 0.25, 32.0))
    return 3.0


def _infer_scenic_z_trig_from_cmd(text: str) -> str:

    if _decorative_cmd_blocked(text):
        return "sin"
    s = str(text).lower().replace("-", " ")
    cos_hints = (
        "cosine altitude",
        "vertical cosine",
        "altitude cosine",
        "cosine bob",
        "cosine oscillation",
        "z cosine",
        "3d cosine",
        "three dimensional cosine",
        "helix cosine",
        "corkscrew cosine",
    )
    sin_hints = (
        "sine altitude",
        "vertical sine",
        "altitude sine",
        "sine bob",
        "sine oscillation",
        "z sine",
        "3d sine",
        "three dimensional sine",
    )
    has_cos = any(h in s for h in cos_hints)
    has_sin = any(h in s for h in sin_hints)
    if has_cos and not has_sin:
        return "cos"
    return "sin"


def _language_cmd_requests_diamond(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if _language_cmd_requests_figure8(text) or _language_cmd_requests_oval_racetrack(text):
        return False
    if re.search(r"\bdiamond\b", s) or "diamond shaped" in s:
        return True
    if "rhombus" in s or re.search(r"\blozenge\b", s):
        return True
    return False


def _language_cmd_requests_square_circuit(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if (
        _language_cmd_requests_figure8(text)
        or _language_cmd_requests_oval_racetrack(text)
        or _language_cmd_requests_diamond(text)
    ):
        return False
    sq = (
        re.search(r"\bsquare\b", s) is not None
        or "rectangular" in s
        or re.search(r"\brectangle\b", s) is not None
    )
    if not sq:
        return False
    circuitish = (
        "circuit" in s
        or "loop" in s
        or re.search(r"\bpath\b", s) is not None
        or "perimeter" in s
        or "track" in s
        or re.search(r"\bfour\s+corners\b", s) is not None
    )
    return circuitish


def _language_cmd_requests_spiral(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return False
    if (
        "logarithmic spiral" in s
        or "log spiral" in s
        or "equiangular spiral" in s
        or "hyperbolic spiral" in s
        or "archimedean spiral" in s
        or "arithmetic spiral" in s
        or "constant pitch spiral" in s
        or "clothoid" in s
        or "euler spiral" in s
        or ("cornu spiral" in s)
        or ("fresnel spiral" in s)
    ):
        return False
    if "spiral" in s or "spiralling" in s or "sprial" in s:
        return True
    return (
        "corkscrew" in s
        or "cork screw" in s
        or re.search(r"\bhelix\b", s) is not None
        or "helical" in s
        or "3d spiral" in s
        or "3-d spiral" in s
        or "vertical spiral" in s
        or "climbing spiral" in s
        or "descending spiral" in s
    )


def _language_cmd_requests_serpentine_scan(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return False
    if _language_cmd_requests_spiral(text):
        return False
    if "serpentine" in s or "zigzag" in s or "zig zag" in s:
        return True
    if "boustrophedon" in s:
        return True
    if "lawn" in s and "mower" in s:
        return True
    if "raster" in s:
        return True
    if "grid" in s and ("scan" in s or "coverage" in s or ("pattern" in s and "grid" in s)):
        return True
    return False


def _language_cmd_requests_shuttle_line_xy(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return False
    if _language_cmd_requests_serpentine_scan(text):
        return False
    if "back and forth" in s or "back-and-forth" in s:
        return True
    return any(
        h in s
        for h in (
            "reciprocating path",
            "reciprocating flight",
            "shuttle run",
            "shuttle path",
            "pendulum path",
            "linear traverse",
            "straight line shuttle",
            "straight shuttle",
            "out and back",
        )
    )


def _language_cmd_requests_cross_axis_shuttle(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return False
    if _language_cmd_requests_serpentine_scan(text):
        return False
    crosses = (
        "crisscross",
        "criss-cross",
        "cross pattern",
        "cross-pattern",
        "cross shape",
        "cross-shape",
        "cross shaped",
        "plus pattern",
        "plus-pattern",
        "plus sign pattern",
        "plus-sign pattern",
        "orthogonal cross",
        "cross traverse",
        "shape of a plus",
        "plus-shaped",
    )
    return any(h in s for h in crosses)


def _language_cmd_requests_teardrop_loop_xy(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    return "teardrop" in s or "tear drop" in s or "tear-drop" in s


def _language_cmd_requests_phyllotaxis_disk(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    if "logarithmic spiral" in s or "log spiral" in s:
        return False
    return (
        "phyllotaxis" in s
        or "phyllotactic" in s
        or "sunflower seed" in s
        or "sunflower disk" in s
        or "sunflower pattern" in s
        or "golden angle spiral" in s
        or "golden angle pattern" in s
        or "vogel spiral" in s
        or "fermat spiral" in s
        or "fermat's spiral" in s
    )


def _language_cmd_requests_clover_trefoil(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return False
    if "clover" in s:
        return True
    if "trefoil" in s:
        return True
    if "three loops" in s or "three-loops" in s.replace(" ", "-"):
        return True
    if re.search(r"\bthree\s+loop\b", s):
        return True
    return False


def _language_cmd_requests_pentagram_star(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return False
    if "pentagram" in s:
        return True
    if "five-point" in s or "five point" in s:
        return True
    if "five corners star" in s:
        return True
    return False


def _language_cmd_requests_lissajous(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return False
    return "lissajous" in s or "bowtie" in s or "bow tie" in s


def _language_cmd_requests_heart_loop(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return False
    if "heart-shaped" in s.replace(" ", "-").replace("--", "-") or "heart shaped" in s:
        return True
    if "heart" in s and (
        "shape" in s or "pattern" in s or "path" in s or "trace" in s or "fly" in s or "circuit" in s
    ):
        return True
    return False


def _decorative_cmd_blocked(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(pass|fly|go)\s+through\b", s):
        return True
    if re.search(r"\bthrough\s+(the\s+)?(portal|frame|opening|slot|gate)\b", s):
        return True
    if re.search(r"\balign\s+(with|to)\s+the\b", s):
        return True
    return False


def _infer_rose_petal_k(text: str) -> int | None:

    if _decorative_cmd_blocked(text):
        return None
    s = str(text).lower().replace("-", " ")
    mk = re.search(r"\b(\d{1,2})\s*-?\s*pet(?:al|alled|als)?\b", s)
    if mk is not None:
        k = int(mk.group(1))
        if k >= 2:
            return k
    if "rose" in s or "rhodonea" in s:
        return 7
    return None


def _infer_regular_polygon_outline_sides(text: str) -> int | None:

    if _decorative_cmd_blocked(text):
        return None
    s = str(text).lower().replace("-", " ")
    if re.search(r"\btriangle\b", s) or "triangular" in s or "trigon" in s:
        return 3
    if re.search(r"\bpentagon\b", s):
        return 5
    if re.search(r"\bhexagon\b", s) or "hexagonal" in s:
        return 6
    if "heptagon" in s or "septagon" in s:
        return 7
    if "octagon" in s or "octagonal" in s:
        return 8
    mq = re.search(r"\b([3-9]|[12]\d)\s*-?\s*gon\b|\b(\d+)\s*-?\s*sided\b", s)
    if mq:
        gs = mq.group(1) or mq.group(2)
        n = int(gs)
        if 3 <= n <= 24:
            return n
    return None


def _language_cmd_requests_damped_sine_wave_path(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    return (
        "damped sine" in s
        or "decaying sine" in s
        or "underdamped" in s
        or "decaying oscillation" in s
        or ("exponential decay" in s and "sine" in s)
    )


def _language_cmd_requests_trig_beat_wave_path(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    if _language_cmd_requests_damped_sine_wave_path(text):
        return False
    s = str(text).lower().replace("-", " ")
    return (
        "beat wave" in s
        or "beats pattern" in s
        or "wave beats" in s
        or "sin beat" in s
        or "cos beat" in s
    )


def _language_cmd_requests_triangle_wave_path(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    if _language_cmd_requests_damped_sine_wave_path(text):
        return False
    if _language_cmd_requests_trig_beat_wave_path(text):
        return False
    s = str(text).lower().replace("-", " ")
    return "triangle wave" in s or "triangular wave" in s or "tri wave" in s


def _language_cmd_requests_tanh_ribbon_path(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    if _language_cmd_requests_damped_sine_wave_path(text):
        return False
    if _language_cmd_requests_trig_beat_wave_path(text):
        return False
    if _language_cmd_requests_triangle_wave_path(text):
        return False
    s = str(text).lower().replace("-", " ")
    return (
        "tanh" in s
        or "hyperbolic tangent" in s
        or "hyperbolic tan" in s
        or "saturation s-curve" in s
        or "tanh ribbon" in s
    )


def _language_cmd_requests_cosine_wave_path(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    if _language_cmd_requests_damped_sine_wave_path(text):
        return False
    if _language_cmd_requests_trig_beat_wave_path(text):
        return False
    if _language_cmd_requests_triangle_wave_path(text):
        return False
    if _language_cmd_requests_tanh_ribbon_path(text):
        return False
    s = str(text).lower().replace("-", " ")
    hits = (
        "cosine wave",
        "cos wave",
        "cosinusoid",
        "cosinusoidal trace",
        "cosine ribbon",
        "fly a cosine",
        "trace a cosine",
    )
    return any(h in s for h in hits)


def _language_cmd_requests_sine_wave_path(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if _language_cmd_requests_damped_sine_wave_path(text):
        return False
    if _language_cmd_requests_cosine_wave_path(text):
        return False
    if _language_cmd_requests_trig_beat_wave_path(text):
        return False
    if _language_cmd_requests_triangle_wave_path(text):
        return False
    if _language_cmd_requests_tanh_ribbon_path(text):
        return False
    hits = (
        "sine wave",
        "sin wave",
        "sinusoid",
        "harmonic wave",
        "sinusoidal trace",
        "s curve",
    )
    return any(h in s for h in hits)


def _language_cmd_requests_stadium_capsule(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if (
        "stadium" in s
        and (
            "capsule" in s
            or "pill" in s
            or ("shape" in s and ("path" in s or "orbit" in s or "circuit" in s))
        )
    ):
        return True
    if "capsule" in s and (
        "path" in s or "orbit" in s or "track" in s or "circuit" in s or "fly" in s
    ):
        return True
    if "pill" in s and ("shape" in s or "circuit" in s or "orbit" in s):
        return True
    if ("rounded rectangle" in s or "rounded rectangular" in s) and (
        "perimeter" in s or "circuit" in s or "orbit" in s or "outline" in s
    ):
        return True
    return False


def _language_cmd_requests_single_row_cycloid(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if re.search(r"\bcycloids?\b", s):
        return True
    if "trochoid" in s and "hypo" not in s and "epi" not in s:
        return True
    return False


def _language_cmd_requests_cardioid_curve(text: str) -> bool:
    return not _decorative_cmd_blocked(text) and "cardioid" in str(text).lower()


def _language_cmd_requests_deltoid_hypocycloid(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if "deltoid" in s:
        return True
    if "hypocycloid" in s and ("three" in s or "tri" in s or "tri-cusp" in s.replace(" ", "-")):
        return True
    return False


def _language_cmd_requests_hypocycloid_astroid_xy(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if "astronomy" in s or "asteroid belt" in s:
        return False
    if re.search(r"\bastroid\b", s):
        return True
    if "hypocycloid" in s and ("four" in s or "quad" in s):
        return True
    return False


def _language_cmd_requests_regular_polygon_outline(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    if _language_cmd_requests_pentagram_star(text):
        return False
    return _infer_regular_polygon_outline_sides(text) is not None


def _language_cmd_requests_rose_rhodonea(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    pk = _infer_rose_petal_k(text)
    if pk is None:
        return False
    return (
        "rhodonea" in s
        or "rose curve" in s
        or "rose orbit" in s
        or "rose path" in s
        or re.search(r"\b\d+\s*-?\s*pet(?:al|alled|als)\b", s) is not None
    )


def _language_cmd_requests_epitrochoid_outer(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if "hypocycloid" in s or "hypotrochoid" in s:
        return False
    return "epitrochoid" in s


def _language_cmd_requests_epicycloid_outer(text: str) -> bool:

    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if "hypocycloid" in s or "hypotrochoid" in s:
        return False
    if "epitrochoid" in s:
        return False
    return "epicycloid" in s or "epicycle" in s or "spirograph" in s


def _language_cmd_requests_log_spiral_arc(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if "hyperbolic spiral" in s:
        return False
    return (
        "logarithmic spiral" in s
        or "log spiral" in s
        or "equiangular spiral" in s
    )


def _language_cmd_requests_hyperbolic_spiral_curve(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    return (
        not _decorative_cmd_blocked(text)
        and "hyperbolic spiral" in s
    )


def _language_cmd_requests_involute_approx(text: str) -> bool:
    return not _decorative_cmd_blocked(text) and (
        "involute" in str(text).lower()
        or ("gear tooth" in str(text).lower())
    )


def _language_cmd_requests_superellipse_sqircle(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    return "squircle" in s or "superellipse" in s


def _language_cmd_requests_butterfly_rice(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    return ("butterfly curve" in s) or ("butterfly flight" in s) or ("butterfly orbit" in s)


def _language_cmd_requests_gaussian_bump_path(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    return ("bell curve" in s or "bell-shaped" in s.replace(" ", "-")) and (
        "path" in s or "trail" in s or "orbit" in s or "pattern" in s or "circuit" in s
    )


def _language_cmd_requests_arc_chain_wave(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if "catenary" in s or "suspension bridge" in s:
        return True
    if "scallop" in s and ("path" in s or "trail" in s or "orbit" in s):
        return True
    return False


def _language_cmd_requests_lemniscate_extended(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    if _decorative_cmd_blocked(text):
        return False
    if "cassini" in s:
        return False
    return "huygens" in s or "lemniscate" in s


def _language_cmd_requests_cochleoid_approx(text: str) -> bool:
    return not _decorative_cmd_blocked(text) and "snail shell" in str(text).lower()


def _language_cmd_requests_steiner_folium_xy(text: str) -> bool:
    s = str(text).lower().replace("-", " ")
    return not _decorative_cmd_blocked(text) and ("folium" in s or "steiner curve" in s)


def _language_cmd_requests_archimedean_spiral(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    return (
        "archimedean spiral" in s
        or "arithmetic spiral" in s
        or "constant pitch spiral" in s
        or "constant spacing spiral" in s
    )


def _language_cmd_requests_clothoid_like(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    return (
        "clothoid" in s
        or "euler spiral" in s
        or "cornu spiral" in s
        or "fresnel spiral" in s
    )


def _language_cmd_requests_tractrix_curve(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    return "tractrix" in s or ("tow rope" in s and "curve" in s)


def _language_cmd_requests_witch_of_agnesi_xy(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    return (
        "witch of agnesi" in s
        or "agnesi curve" in s
        or "agnesi witch" in s
    )


def _language_cmd_requests_cassini_oval_xy(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    return (
        "cassini oval" in s
        or "cassinian oval" in s
        or "cassini curve" in s
        or "cassini lemniscate" in s
    )


def _language_cmd_requests_hypotrochoid_general(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    return "hypotrochoid" in str(text).lower()


def _language_cmd_requests_parabolic_arc_xy(text: str) -> bool:
    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    if "hyperbolic paraboloid" in s:
        return False
    has_par = ("parabolic" in s or re.search(r"\bparabola\b", s) is not None)
    if not has_par:
        return False
    return (
        "arc" in s
        or "path" in s
        or "orbit" in s
        or "curve" in s
        or "circuit" in s
        or "trace" in s
    )


def _language_cmd_requests_nephroid_xy(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    s_raw = str(text).lower().replace("-", " ")
    if "epicycloid" in s_raw or "epicycle" in s_raw or "epitrochoid" in s_raw:
        return False
    s_plain = s_raw.replace("-", " ")
    return (
        "nephroid" in s_raw
        or (
            "kidney" in s_raw
            and ("shaped path" in s_plain or "shaped orbit" in s_plain or "shaped circuit" in s_plain)
        )
    )


def _language_cmd_requests_limacon_pascal_xy(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    if "cardioid" in s:
        return False
    if "limaçon" in str(text).lower():
        return True
    s_ascii = str(text).lower().replace("ç", "c").replace("-", " ")
    return "limacon" in s_ascii


def _language_cmd_requests_cissoid_xy(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower()
    return "cissoid" in s


def _language_cmd_requests_strophoid_xy(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    return "strophoid" in str(text).lower()


def _language_cmd_requests_kampyle_eudoxus_xy(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ")
    return "kampyle" in s or (
        "eudoxus" in s
        and any(h in s for h in ("curve", "path", "orbit", "circuit"))
    )


def _language_cmd_requests_conchoid_nicomedes_xy(text: str) -> bool:

    if _decorative_cmd_blocked(text):
        return False
    s = str(text).lower().replace("-", " ").replace("ç", "c")
    if "conchoid" in s:
        return True
    if "nicomedes" not in s:
        return False
    return any(h in s for h in ("path", "orbit", "circuit", "curve", "trace", "flight", "fly"))


def _resample_xy_polyline_uniform(path_xy: np.ndarray, n_out: int) -> np.ndarray:

    p = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    nn = max(int(n_out), 2)
    if p.shape[0] < 2:
        cx = float(p[0, 0]) if p.size else 0.0
        cy = float(p[0, 1]) if p.size else 0.0
        return np.tile(np.array([[cx, cy]], dtype=np.float32), (nn, 1))
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cum[-1])
    if total < 1e-9:
        return np.tile(p[0:1].astype(np.float32), (nn, 1))
    tgt = np.linspace(0.0, total, nn, dtype=np.float64)
    out = np.zeros((nn, 2), dtype=np.float64)
    for i, ti in enumerate(tgt):
        j = int(np.searchsorted(cum, ti, side="right") - 1)
        j = int(np.clip(j, 0, p.shape[0] - 2))
        sl = float(cum[j + 1] - cum[j])
        if sl < 1e-12:
            out[i] = p[j]
        else:
            u = float(np.clip((ti - cum[j]) / sl, 0.0, 1.0))
            out[i] = (1.0 - u) * p[j] + u * p[j + 1]
    return out.astype(np.float32)


def _expand_rotation_keyframes_to_n(rlist: list[np.ndarray], n: int) -> list[np.ndarray]:

    n = int(max(n, 1))
    if len(rlist) == 0:
        raise ValueError("empty rlist")
    if n == 1:
        return [np.asarray(rlist[0], dtype=np.float32)]
    if n == len(rlist):
        return [np.asarray(R, dtype=np.float32) for R in rlist]
    out: list[np.ndarray] = []
    for i in range(n):
        u = float(i) / float(max(n - 1, 1))
        j = u * float(len(rlist) - 1)
        j0 = int(np.clip(int(np.floor(j)), 0, len(rlist) - 1))
        j1 = int(np.clip(int(np.ceil(j)), 0, len(rlist) - 1))
        t = float(np.clip(j - float(j0), 0.0, 1.0))
        e0 = np.asarray(euler_xyz_from_matrix(rlist[j0]), dtype=np.float64)
        e1 = np.asarray(euler_xyz_from_matrix(rlist[j1]), dtype=np.float64)
        ew = (1.0 - t) * e0 + t * e1
        out.append(
            matrix_from_euler_xyz(float(ew[0]), float(ew[1]), float(ew[2])).astype(np.float32)
        )
    return out


def _fit_scenic_xy_polyline_rows(path_xy: np.ndarray, n_rows: int) -> np.ndarray:

    n = max(int(n_rows), 2)
    p = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] == n:
        return p.astype(np.float32)
    if p.shape[0] < 2:
        c = p[0].astype(np.float64) if p.size else np.zeros(2, dtype=np.float64)
        return np.tile(c.reshape(1, 2), (n, 1)).astype(np.float32)
    return _resample_xy_polyline_uniform(p, n)


def _apply_scenic_helical_z_world(
    Wxyz: np.ndarray,
    *,
    z_lo_world: float,
    z_hi_world: float,
    n_turns: float,
    z_trig: str = "sin",
) -> np.ndarray:

    p = np.asarray(Wxyz, dtype=np.float64).reshape(-1, 3).copy()
    n = int(p.shape[0])
    lo = float(min(z_lo_world, z_hi_world))
    hi = float(max(z_lo_world, z_hi_world))
    zmode = str(z_trig).strip().lower()
    trig_fn = np.cos if zmode == "cos" else np.sin
    if n < 1:
        return p.astype(np.float32)
    if n < 2 or hi <= lo + 1e-9:
        p[:, 2] = np.clip(p[:, 2], lo, hi)
        return p.astype(np.float32)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cum[-1])
    if total < 1e-9:
        p[:, 2] = np.clip(p[:, 2], lo, hi)
        return p.astype(np.float32)
    turns = float(np.clip(float(n_turns), 0.25, 32.0))
    mid = 0.5 * (lo + hi)
    amp = 0.5 * (hi - lo)
    for i in range(n):
        ph = (2.0 * np.pi * turns) * (float(cum[i]) / total)
        p[i, 2] = mid + amp * float(trig_fn(ph))
    return p.astype(np.float32)


def _plan_expand_contract_spiral_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    r = 0.5 * ff * float(min(usable_x, usable_y))
    a = r
    b = r
    nn = max(int(n), 4)
    n_exp = max(nn // 2, 2)
    n_con = nn - n_exp
    rho_min = 0.05
    n_turn_half = float(max(5.0, min(18.0, nn / 3.8)))
    th_scale = n_turn_half * (2.0 * np.pi)
    xe: list[float] = []
    ye: list[float] = []
    for k in range(n_exp):
        u = float(k) / float(max(n_exp - 1, 1))
        rho = rho_min + (1.0 - rho_min) * u
        th = u * th_scale
        xe.append(cx + rho * a * np.cos(th))
        ye.append(cy + rho * b * np.sin(th))
    for k in range(n_con):
        u = float(k) / float(max(n_con - 1, 1))
        rho = 1.0 - (1.0 - rho_min) * u
        th = th_scale + u * th_scale
        xe.append(cx + rho * a * np.cos(th))
        ye.append(cy + rho * b * np.sin(th))
    if len(xe) < 2:
        return np.tile(np.array([[cx, cy]], dtype=np.float32), (max(nn, 2), 1))
    out = np.stack([np.asarray(xe, dtype=np.float64), np.asarray(ye, dtype=np.float64)], axis=1).astype(
        np.float32
    )
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _demo_plan_figure8_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = float(np.clip(fill_frac, 0.18, 0.50))
    ax = 0.5 * usable_x * ff
    ay = 0.5 * usable_y * ff
    t = float(phase)
    x_loc = cx + ax * float(np.sin(t))
    y_loc = cy + ay * float(np.sin(t) * np.cos(t))
    out = np.array([x_loc, y_loc], dtype=np.float32)
    out[0] = float(np.clip(out[0], lo[0], hi[0]))
    out[1] = float(np.clip(out[1], lo[1], hi[1]))
    return out


def _demo_plan_ellipse_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = float(np.clip(fill_frac, 0.18, 0.50))
    a = 0.5 * usable_x * ff
    b = 0.5 * usable_y * ff
    t = float(phase)
    x_loc = cx + a * float(np.cos(t))
    y_loc = cy + b * float(np.sin(t))
    out = np.array([x_loc, y_loc], dtype=np.float32)
    out[0] = float(np.clip(out[0], lo[0], hi[0]))
    out[1] = float(np.clip(out[1], lo[1], hi[1]))
    return out


def _demo_plan_circle_orbit_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = float(np.clip(fill_frac, 0.18, 0.50))
    radius = float(0.5 * min(usable_x, usable_y) * ff)
    t = float(phase)
    x_loc = cx + radius * float(np.cos(t))
    y_loc = cy + radius * float(np.sin(t))
    out = np.array([x_loc, y_loc], dtype=np.float32)
    out[0] = float(np.clip(out[0], lo[0], hi[0]))
    out[1] = float(np.clip(out[1], lo[1], hi[1]))
    return out


def _demo_plan_diamond_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = float(np.clip(fill_frac, 0.18, 0.50))
    rx = 0.5 * usable_x * ff
    ry = 0.5 * usable_y * ff
    verts = np.array(
        [
            [cx, cy + ry],
            [cx + rx, cy],
            [cx, cy - ry],
            [cx - rx, cy],
        ],
        dtype=np.float64,
    )
    edges = [float(np.linalg.norm(verts[(i + 1) % 4] - verts[i])) for i in range(4)]
    total = float(sum(edges))
    if total <= 1e-9:
        return np.array([cx, cy], dtype=np.float32)
    ang = float(phase) % (2.0 * np.pi)
    s = (ang / (2.0 * np.pi)) * total
    acc = 0.0
    for i in range(4):
        L = edges[i]
        if acc + L > s + 1e-12 or i == 3:
            u = float(np.clip((s - acc) / max(L, 1e-9), 0.0, 1.0))
            p0 = verts[i]
            p1 = verts[(i + 1) % 4]
            out = (p0 + u * (p1 - p0)).astype(np.float32)
            out[0] = float(np.clip(out[0], lo[0], hi[0]))
            out[1] = float(np.clip(out[1], lo[1], hi[1]))
            return out
        acc += L
    return np.array([cx, cy], dtype=np.float32)


def _demo_plan_square_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = float(np.clip(fill_frac, 0.18, 0.50))
    half = 0.5 * min(usable_x, usable_y) * ff
    hx = hy = half
    verts = np.array(
        [
            [cx + hx, cy + hy],
            [cx - hx, cy + hy],
            [cx - hx, cy - hy],
            [cx + hx, cy - hy],
        ],
        dtype=np.float64,
    )
    edges = [float(np.linalg.norm(verts[(i + 1) % 4] - verts[i])) for i in range(4)]
    total = float(sum(edges))
    if total <= 1e-9:
        return np.array([cx, cy], dtype=np.float32)
    ang = float(phase) % (2.0 * np.pi)
    s_arc = (ang / (2.0 * np.pi)) * total
    acc = 0.0
    for i in range(4):
        L = edges[i]
        if acc + L > s_arc + 1e-12 or i == 3:
            u = float(np.clip((s_arc - acc) / max(L, 1e-9), 0.0, 1.0))
            p0 = verts[i]
            p1 = verts[(i + 1) % 4]
            out = (p0 + u * (p1 - p0)).astype(np.float32)
            out[0] = float(np.clip(out[0], lo[0], hi[0]))
            out[1] = float(np.clip(out[1], lo[1], hi[1]))
            return out
        acc += L
    return np.array([cx, cy], dtype=np.float32)


def _demo_plan_spiral_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    hi = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = float(np.clip(fill_frac, 0.18, 0.50))
    r = 0.5 * ff * float(min(usable_x, usable_y))
    a = r
    b = r
    rho_min = 0.05
    turns = float(11.0)
    span = float(28.0 * np.pi)
    u = float((float(phase) % span) / span)
    th_scale = turns * (2.0 * np.pi)
    if u < 0.5:
        v = u * 2.0
        rho = rho_min + (1.0 - rho_min) * v
        th = v * th_scale
    else:
        v = (u - 0.5) * 2.0
        rho = 1.0 - (1.0 - rho_min) * v
        th = th_scale + v * th_scale
    x_loc = cx + rho * a * np.cos(th)
    y_loc = cy + rho * b * np.sin(th)
    out = np.array([x_loc, y_loc], dtype=np.float32)
    out[0] = float(np.clip(out[0], lo[0], hi[0]))
    out[1] = float(np.clip(out[1], lo[1], hi[1]))
    return out


def _demo_xy_sample_polyline_phase(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
    planner: Callable[..., np.ndarray],
    span: float,
    dense_n: int = 384,
) -> np.ndarray:

    lo3 = np.asarray(workspace_lo, dtype=np.float64).reshape(3)
    hi3 = np.asarray(workspace_hi, dtype=np.float64).reshape(3)
    lo2 = lo3[0:2]
    hi2 = hi3[0:2]
    path = planner(
        lo2,
        hi2,
        int(max(dense_n, 16)),
        fill_frac=float(fill_frac),
        edge_margin_frac=float(edge_margin_frac),
    )
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if path.shape[0] < 2:
        c = np.array([0.5 * (lo2[0] + hi2[0]), 0.5 * (lo2[1] + hi2[1])], dtype=np.float32)
        c[0] = float(np.clip(c[0], lo3[0], hi3[0]))
        c[1] = float(np.clip(c[1], lo3[1], hi3[1]))
        return c
    sp = float(span) if span > 1e-6 else float(32.0 * np.pi)
    u = float((float(phase) % sp) / sp)
    i = int(np.clip(round(u * float(path.shape[0] - 1)), 0, path.shape[0] - 1))
    out = path[i].astype(np.float32)
    out[0] = float(np.clip(out[0], lo3[0], hi3[0]))
    out[1] = float(np.clip(out[1], lo3[1], hi3[1]))
    return out


def _demo_plan_serpentine_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_serpentine_xy_polyline,
        span=float(112.0 * np.pi),
    )


def _demo_plan_clover_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_clover_xy_polyline,
        span=float(24.0 * np.pi),
    )


def _demo_plan_star_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_pentagram_xy_polyline,
        span=float(40.0 * np.pi),
    )


def _demo_plan_lissajous_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_lissajous_xy_polyline,
        span=float(28.0 * np.pi),
    )


def _demo_plan_heart_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_heart_xy_polyline,
        span=float(36.0 * np.pi),
    )


def _demo_plan_regular_polygon_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    sides: int,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    sd = max(3, min(int(sides), 24))

    def _pl(
        lo2: np.ndarray, hi2: np.ndarray, nv: int, *, fill_frac: float, edge_margin_frac: float
    ) -> np.ndarray:
        return _plan_regular_n_gon_xy_polyline(
            lo2,
            hi2,
            nv,
            sides=sd,
            fill_frac=float(fill_frac),
            edge_margin_frac=float(edge_margin_frac),
        )

    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_pl,
        span=float(sd * (32.0 * np.pi)),
    )


def _demo_plan_rose_petal_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    petals_k: int,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    kk = max(2, min(int(petals_k), 24))

    def _pl(
        lo2: np.ndarray, hi2: np.ndarray, nv: int, *, fill_frac: float, edge_margin_frac: float
    ) -> np.ndarray:
        return _plan_rose_petal_xy_polyline(
            lo2,
            hi2,
            nv,
            petals_k=kk,
            fill_frac=float(fill_frac),
            edge_margin_frac=float(edge_margin_frac),
        )

    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_pl,
        span=float(80.0 * np.pi),
    )


def _demo_plan_stadium_capsule_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_stadium_xy_polyline,
        span=float(112.0 * np.pi),
    )


def _demo_plan_sinewave_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_sinewave_xy_polyline,
        span=float(120.0 * np.pi),
    )


def _demo_plan_cosine_wave_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_cosine_wave_xy_polyline,
        span=float(120.0 * np.pi),
    )


def _demo_plan_damped_sine_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_damped_sine_xy_polyline,
        span=float(112.0 * np.pi),
    )


def _demo_plan_trig_beat_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_trig_beat_wave_xy_polyline,
        span=float(128.0 * np.pi),
    )


def _demo_plan_triangle_wave_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_triangle_wave_xy_polyline,
        span=float(128.0 * np.pi),
    )


def _demo_plan_tanh_ribbon_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_tanh_ribbon_xy_polyline,
        span=float(72.0 * np.pi),
    )


def _demo_plan_cycloid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_cycloid_one_arch_xy_polyline,
        span=float(48.0 * np.pi),
    )


def _demo_plan_cardioid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_cardioid_xy_polyline,
        span=float(64.0 * np.pi),
    )


def _demo_plan_deltoid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_deltoid_xy_polyline,
        span=float(96.0 * np.pi),
    )


def _demo_plan_astroid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_astroid_xy_polyline,
        span=float(72.0 * np.pi),
    )


def _demo_plan_epicycloid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_epicycloid_default_xy_polyline,
        span=float(512.0 * np.pi),
    )


def _demo_plan_epitrochoid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_epitrochoid_default_xy_polyline,
        span=float(640.0 * np.pi),
    )


def _demo_plan_shuttle_line_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_shuttle_line_xy_polyline,
        span=float(48.0 * np.pi),
    )


def _demo_plan_cross_axis_shuttle_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_cross_axis_shuttle_xy_polyline,
        span=float(96.0 * np.pi),
    )


def _demo_plan_teardrop_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_teardrop_xy_polyline,
        span=float(88.0 * np.pi),
    )


def _demo_plan_phyllotaxis_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_phyllotaxis_disk_xy_polyline,
        span=float(1400.0 * np.pi),
    )


def _demo_plan_logarithmic_spiral_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_logarithmic_spiral_xy_polyline,
        span=float(512.0 * np.pi),
    )


def _demo_plan_hyperbolic_spiral_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_hyperbolic_spiral_xy_polyline,
        span=float(1024.0 * np.pi),
    )


def _demo_plan_involute_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_involute_circle_xy_polyline,
        span=float(1400.0 * np.pi),
    )


def _demo_plan_superellipse_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_superellipse_xy_polyline,
        span=float(80.0 * np.pi),
    )


def _demo_plan_butterfly_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_butterfly_rice_xy_polyline,
        span=float(3600.0 * np.pi),
    )


def _demo_plan_gaussian_track_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_gaussian_track_xy_polyline,
        span=float(280.0 * np.pi),
    )


def _demo_plan_arc_chain_wave_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_arc_chain_wave_xy_polyline,
        span=float(160.0 * np.pi),
    )


def _demo_plan_bernoulli_lemniscate_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_bernoulli_lemniscate_xy_polyline,
        span=float(192.0 * np.pi),
    )


def _demo_plan_cochleoid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_cochleoid_xy_polyline,
        span=float(820.0 * np.pi),
    )


def _demo_plan_steiner_folium_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_steiner_folium_xy_polyline,
        span=float(1200.0 * np.pi),
    )


def _demo_plan_archimedean_spiral_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_archimedean_spiral_xy_polyline,
        span=float(640.0 * np.pi),
    )


def _demo_plan_clothoid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_clothoid_segment_xy_polyline,
        span=float(420.0 * np.pi),
    )


def _demo_plan_tractrix_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_tractrix_xy_polyline,
        span=float(180.0 * np.pi),
    )


def _demo_plan_witch_of_agnesi_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_witch_of_agnesi_xy_polyline,
        span=float(240.0 * np.pi),
    )


def _demo_plan_cassini_oval_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_cassini_oval_xy_polyline,
        span=float(220.0 * np.pi),
    )


def _demo_plan_hypotrochoid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_hypotrochoid_default_xy_polyline,
        span=float(900.0 * np.pi),
    )


def _demo_plan_parabolic_arc_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_parabolic_arc_xy_polyline,
        span=float(160.0 * np.pi),
    )


def _demo_plan_nephroid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_nephroid_xy_polyline,
        span=float(64.0 * np.pi),
    )


def _demo_plan_limacon_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_limacon_pascal_xy_polyline,
        span=float(72.0 * np.pi),
    )


def _demo_plan_cissoid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_cissoid_diocles_xy_polyline,
        span=float(320.0 * np.pi),
    )


def _demo_plan_strophoid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_strophoid_right_xy_polyline,
        span=float(340.0 * np.pi),
    )


def _demo_plan_kampyle_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_kampyle_eudoxus_xy_polyline,
        span=float(400.0 * np.pi),
    )


def _demo_plan_conchoid_xy_in_workspace(
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    phase: float,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    return _demo_xy_sample_polyline_phase(
        workspace_lo,
        workspace_hi,
        phase,
        fill_frac=fill_frac,
        edge_margin_frac=edge_margin_frac,
        planner=_plan_conchoid_nicomedes_xy_polyline,
        span=float(380.0 * np.pi),
    )


def _plan_figure8_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    ax = 0.5 * usable_x * ff
    ay = 0.5 * usable_y * ff
    nn = max(int(n), 2)
    t = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    x = cx + ax * np.sin(t)
    y = cy + ay * np.sin(2 * t) * 0.5
    out = np.stack([x, y], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_ellipse_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    a = 0.5 * usable_x * ff
    b = 0.5 * usable_y * ff
    nn = max(int(n), 3)
    t = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    x = cx + a * np.cos(t)
    y = cy + b * np.sin(t)
    out = np.stack([x, y], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_circle_orbit_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    radius = float(0.5 * min(usable_x, usable_y) * ff)
    nn = max(int(n), 3)
    t = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    x = cx + radius * np.cos(t)
    y = cy + radius * np.sin(t)
    out = np.stack([x, y], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_diamond_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    rx = 0.5 * usable_x * ff
    ry = 0.5 * usable_y * ff
    verts = np.array(
        [
            [cx, cy + ry],
            [cx + rx, cy],
            [cx, cy - ry],
            [cx - rx, cy],
        ],
        dtype=np.float64,
    )
    edges = [float(np.linalg.norm(verts[(i + 1) % 4] - verts[i])) for i in range(4)]
    total = float(sum(edges))
    nn = max(int(n), 4)
    if total <= 1e-9:
        return np.tile(np.array([[cx, cy]], dtype=np.float32), (nn, 1))
    out = np.zeros((nn, 2), dtype=np.float64)
    for k in range(nn):
        s = (float(k) / float(nn)) * total
        acc = 0.0
        placed = False
        for i in range(4):
            L = edges[i]
            if acc + L > s + 1e-12 or i == 3:
                u = float(np.clip((s - acc) / max(L, 1e-9), 0.0, 1.0))
                p0 = verts[i]
                p1 = verts[(i + 1) % 4]
                out[k] = p0 + u * (p1 - p0)
                placed = True
                break
            acc += L
        if not placed:
            out[k] = verts[0]
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out.astype(np.float32)


def _plan_square_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    half = 0.5 * min(usable_x, usable_y) * ff
    hx = hy = half
    verts = np.array(
        [
            [cx + hx, cy + hy],
            [cx - hx, cy + hy],
            [cx - hx, cy - hy],
            [cx + hx, cy - hy],
        ],
        dtype=np.float64,
    )
    edges = [float(np.linalg.norm(verts[(i + 1) % 4] - verts[i])) for i in range(4)]
    total = float(sum(edges))
    nn = max(int(n), 4)
    if total <= 1e-9:
        return np.tile(np.array([[cx, cy]], dtype=np.float32), (nn, 1))
    out = np.zeros((nn, 2), dtype=np.float64)
    for k in range(nn):
        s = (float(k) / float(nn)) * total
        acc = 0.0
        placed = False
        for i in range(4):
            L = edges[i]
            if acc + L > s + 1e-12 or i == 3:
                u = float(np.clip((s - acc) / max(L, 1e-9), 0.0, 1.0))
                p0 = verts[i]
                p1 = verts[(i + 1) % 4]
                out[k] = p0 + u * (p1 - p0)
                placed = True
                break
            acc += L
        if not placed:
            out[k] = verts[0]
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out.astype(np.float32)


def _plan_serpentine_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    lx, ly = float(lo[0]), float(lo[1])
    hx, hy = float(hi[0]), float(hi[1])
    nn = max(int(n), 8)
    ny = int(np.clip(max(nn // 8, 4), 4, 56))
    verts: list[tuple[float, float]] = [(lx, ly)]
    for j in range(ny):
        y = ly + (hy - ly) * (float(j) / float(max(ny - 1, 1)))
        if j % 2 == 0:
            verts.append((lx, y))
            verts.append((hx, y))
        else:
            verts.append((hx, y))
            verts.append((lx, y))
    path = np.asarray(verts, dtype=np.float64)
    out = _resample_xy_polyline_uniform(path, nn)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_clover_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    a = 0.5 * usable_x * ff
    b = 0.5 * usable_y * ff
    nn = max(int(n), 24)
    t = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    bump = 0.38 * np.cos(3.0 * t)
    rad = (1.0 + bump) / 1.38
    x = cx + a * np.cos(t) * rad
    y = cy + b * np.sin(t) * rad
    out = np.stack([x, y], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_pentagram_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    R = 0.5 * min(usable_x, usable_y) * ff
    verts = []
    for k in range(5):
        ang = np.pi / 2.0 + float(k) * (2.0 * np.pi / 5.0)
        verts.append([cx + R * np.cos(ang), cy + R * np.sin(ang)])
    order = [0, 2, 4, 1, 3, 0]
    segments: list[np.ndarray] = []
    for i in range(len(order) - 1):
        p0 = np.asarray(verts[order[i]], dtype=np.float64)
        p1 = np.asarray(verts[order[i + 1]], dtype=np.float64)
        segments.append(np.stack([p0, p1]))
    path = np.concatenate(segments, axis=0)
    nn = max(int(n), 10)
    out = _resample_xy_polyline_uniform(path, nn)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_lissajous_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    a = 0.5 * usable_x * ff
    b = 0.5 * usable_y * ff
    nn = max(int(n), 24)
    ta = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    x = cx + a * np.sin(3.0 * ta + 0.15)
    y = cy + b * np.sin(2.0 * ta)
    out = np.stack([x, y], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_heart_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    span_x = max(hi[0] - lo[0], 1e-6)
    span_y = max(hi[1] - lo[1], 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    ff = 0.95
    nn = max(int(n), 32)
    t = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    xh = 16.0 * np.sin(t) ** 3
    yh = 13.0 * np.cos(t) - 5.0 * np.cos(2.0 * t) - 2.0 * np.cos(3.0 * t) - np.cos(4.0 * t)
    xh_norm = xh / 16.0
    yh_norm = yh / 17.5
    x = cx + 0.5 * usable_x * ff * xh_norm
    y = cy + 0.5 * usable_y * ff * yh_norm
    out = np.stack([x, y], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _scenic_xy_usable_box(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    *,
    edge_margin_frac: float,
) -> tuple[float, float, float, float, float, float, float, float]:

    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    lx, ly = float(lo[0]), float(lo[1])
    hx, hy = float(hi[0]), float(hi[1])
    span_x = max(hx - lx, 1e-6)
    span_y = max(hy - ly, 1e-6)
    mf = float(np.clip(edge_margin_frac, 0.02, 0.30))
    usable_x = max(span_x * (1.0 - 2.0 * mf), 1e-6)
    usable_y = max(span_y * (1.0 - 2.0 * mf), 1e-6)
    return cx, cy, usable_x, usable_y, lx, ly, hx, hy


def _scenic_normalize_and_place_xy(xh: np.ndarray, yh: np.ndarray, *, cx: float, cy: float, fx: float, fy: float) -> np.ndarray:

    xh = np.asarray(xh, dtype=np.float64).reshape(-1)
    yh = np.asarray(yh, dtype=np.float64).reshape(-1)
    n = max(int(min(xh.size, yh.size)), 1)
    xh = xh[:n]
    yh = yh[:n]
    xm = float(0.5 * (xh.min() + xh.max()))
    ym = float(0.5 * (yh.min() + yh.max()))
    xr = max(0.5 * float(np.ptp(xh)), 1e-9)
    yr = max(0.5 * float(np.ptp(yh)), 1e-9)
    ux = ((xh - xm) / xr) * fx
    uy = ((yh - ym) / yr) * fy
    out = np.stack([cx + ux, cy + uy], axis=1).astype(np.float32)
    return out


def _plan_regular_n_gon_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    sides: int,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, _, _, _, _ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), max(int(sides) * 8, 8))
    m = max(3, min(int(sides), 24))
    ff = 0.95
    R = float(0.5 * min(ux, uy) * ff)
    verts = []
    for j in range(m):
        ang = np.pi / 2.0 + float(j) * (2.0 * np.pi / float(m))
        verts.append([cx + R * np.cos(ang), cy + R * np.sin(ang)])
    segments: list[np.ndarray] = []
    for i in range(m):
        p0 = np.asarray(verts[i], dtype=np.float64)
        p1 = np.asarray(verts[(i + 1) % m], dtype=np.float64)
        segments.append(np.stack([p0, p1]))
    path = np.concatenate(segments, axis=0)
    out = _resample_xy_polyline_uniform(path, nn)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_rose_petal_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    petals_k: int,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, _, _, _, _ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    k = max(2, min(int(petals_k), 24))
    nn = max(int(n), max(160, int(k * 24)))
    ff = 0.95
    ta = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    rad = np.cos(float(k) * ta)
    eff = rad * (rad >= 0.0)
    rad_n = rad / float(max(np.abs(rad).max(), 1e-9))
    ee = float(max(float(np.abs(eff).max()), 1e-9))
    eff_n = eff / ee
    use_eff = np.count_nonzero(eff > 1e-6) >= max(24, nn // (k + 6))
    r_use = np.where(use_eff & (rad > -1e-6), eff_n, rad_n)
    xh = np.cos(ta) * r_use * (0.5 * ux * ff)
    yh = np.sin(ta) * r_use * (0.5 * uy * ff)
    out = np.stack([xh + cx, yh + cy], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_stadium_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    lx, ly, hx, hy = float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])
    cx, cy, usable_x, usable_y, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    ff = 0.95
    ux = usable_x * ff * 0.5
    uy = usable_y * ff * 0.5

    verts: list[tuple[float, float]] = []

    def cap_pts_horizontal(w: float, h: float, n_arc: int) -> None:
        if w <= 1e-9 or h <= 1e-9:
            verts.append((cx, cy))
            return
        r = float(0.5 * min(w, h))
        straight = float(max(w - 2.0 * r, 0.0))
        xl_c = cx - 0.5 * straight
        xr_c = cx + 0.5 * straight
        for th in np.linspace(0.5 * np.pi, 1.5 * np.pi, max(n_arc, 8), endpoint=False):
            verts.append((xl_c + r * np.cos(th), cy + r * np.sin(th)))
        for xv in np.linspace(xl_c, xr_c, max(n_arc // 2, 2), endpoint=True):
            verts.append((xv, cy - r))
        for th in np.linspace(-0.5 * np.pi, 0.5 * np.pi, max(n_arc, 8), endpoint=False):
            verts.append((xr_c + r * np.cos(th), cy + r * np.sin(th)))
        for xv in np.linspace(xr_c, xl_c, max(n_arc // 2, 2), endpoint=True):
            verts.append((xv, cy + r))

    def rotate_pts(pairs: list[tuple[float, float]]) -> np.ndarray:
        arr = np.asarray(pairs, dtype=np.float64)
        xe = cy + (arr[:, 1] - cy)
        ye = cx + (arr[:, 0] - cx)
        return np.stack([xe, ye], axis=1)

    n_arc = max(int(n) // 6, 10)
    if usable_x >= usable_y:
        cap_pts_horizontal(2.0 * ux, 2.0 * uy, n_arc)
        path_xy = np.asarray(verts, dtype=np.float64)
    else:
        cap_pts_horizontal(2.0 * uy, 2.0 * ux, n_arc)
        path_xy = rotate_pts(verts)

    nn = max(int(n), 64)
    out = _resample_xy_polyline_uniform(path_xy, nn).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_sinewave_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 2)
    ff = 0.95
    if ux >= uy:
        xs = np.linspace(lx, hx, nn, dtype=np.float64)
        wav = np.sin(np.linspace(0.0, 4.0 * np.pi, nn, endpoint=True))
        ys = cy + wav * (0.5 * uy * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        ys = np.linspace(ly, hy, nn, dtype=np.float64)
        wav = np.sin(np.linspace(0.0, 4.0 * np.pi, nn, endpoint=True))
        xs = cx + wav * (0.5 * ux * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_cosine_wave_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 2)
    ff = 0.95
    ang = np.linspace(0.0, 4.0 * np.pi, nn, endpoint=True)
    if ux >= uy:
        xs = np.linspace(lx, hx, nn, dtype=np.float64)
        wav = np.cos(ang)
        ys = cy + wav * (0.5 * uy * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        ys = np.linspace(ly, hy, nn, dtype=np.float64)
        wav = np.cos(ang)
        xs = cx + wav * (0.5 * ux * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_damped_sine_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 2)
    ff = 0.95
    ang = np.linspace(0.0, 3.5 * np.pi, nn, endpoint=True)
    envelope = np.exp(-0.45 * (ang / np.pi))
    if ux >= uy:
        xs = np.linspace(lx, hx, nn, dtype=np.float64)
        wav = envelope * np.sin(ang)
        ys = cy + wav * (0.5 * uy * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        ys = np.linspace(ly, hy, nn, dtype=np.float64)
        wav = envelope * np.sin(ang)
        xs = cx + wav * (0.5 * ux * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_trig_beat_wave_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 2)
    ff = 0.95
    ang = np.linspace(0.0, 5.5 * np.pi, nn, endpoint=True)
    carrier = np.sin(ang)
    slow_mod = 0.78 + 0.22 * np.cos(ang * (13.0 / 37.0))
    wav = carrier * slow_mod
    if ux >= uy:
        xs = np.linspace(lx, hx, nn, dtype=np.float64)
        ys = cy + wav * (0.5 * uy * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        ys = np.linspace(ly, hy, nn, dtype=np.float64)
        xs = cx + wav * (0.5 * ux * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_triangle_wave_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 2)
    ff = 0.95
    ang = np.linspace(0.0, 4.0 * np.pi, nn, endpoint=True)
    wav = (2.0 / np.pi) * np.arcsin(np.sin(ang))
    if ux >= uy:
        xs = np.linspace(lx, hx, nn, dtype=np.float64)
        ys = cy + wav * (0.5 * uy * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        ys = np.linspace(ly, hy, nn, dtype=np.float64)
        xs = cx + wav * (0.5 * ux * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_tanh_ribbon_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 2)
    ff = 0.95
    u = np.linspace(-3.25, 3.25, nn, dtype=np.float64)
    wav = np.tanh(u)
    if ux >= uy:
        xs = np.linspace(lx, hx, nn, dtype=np.float64)
        ys = cy + wav * (0.5 * uy * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        ys = np.linspace(ly, hy, nn, dtype=np.float64)
        xs = cx + wav * (0.5 * ux * ff * 0.85)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_cycloid_one_arch_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    ff = 0.95
    tt = np.linspace(0.0, 2.0 * np.pi, max(int(n), 64), endpoint=False)
    xh = tt - np.sin(tt)
    yh = 1.0 - np.cos(tt)
    fx = float(0.5 * ux * ff)
    fy = float(0.5 * uy * ff)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_cardioid_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 64)
    tt = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    rad = 1.0 + np.cos(tt)
    xh = rad * np.cos(tt)
    yh = rad * np.sin(tt)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_deltoid_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    tt = np.linspace(0.0, 2.0 * np.pi, max(int(n), 96), endpoint=False)
    xh = 2.0 * np.cos(tt) + np.cos(2.0 * tt)
    yh = 2.0 * np.sin(tt) - np.sin(2.0 * tt)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_astroid_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    tt = np.linspace(0.0, 2.0 * np.pi, max(int(n), 96), endpoint=False)
    xh = np.cos(tt) ** 3
    yh = np.sin(tt) ** 3
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_epicycloid_default_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    RR, rr = 7.0, 2.0
    ratio = float((RR + rr) / max(rr, 1e-9))
    tn = max(int(round(720.0 * ratio)), max(int(n) * int(ratio)))
    tn = max(tn, 512)
    denom = float(max(math.gcd(int(RR), int(rr)), 1))
    tt = np.linspace(0.0, 2.0 * np.pi * rr / denom, tn, endpoint=False)
    xh = (RR + rr) * np.cos(tt) - rr * np.cos(ratio * tt)
    yh = (RR + rr) * np.sin(tt) - rr * np.sin(ratio * tt)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out0 = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out = _resample_xy_polyline_uniform(np.asarray(out0, dtype=np.float64), max(int(n), 256)).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_epitrochoid_default_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    RR, rr, dd = 5.0, 2.0, 3.25
    rri = int(max(1, round(rr)))
    RRi = int(max(1, round(RR)))
    ratio = float((RR + rr) / max(rr, 1e-9))
    denom = float(max(math.gcd(RRi, rri), 1))
    tn = max(int(round(880.0 * ratio)), max(int(n) * 10, 400))
    tt = np.linspace(0.0, 2.0 * np.pi * rr / denom, tn, endpoint=False)
    xh = (RR + rr) * np.cos(tt) - dd * np.cos(ratio * tt)
    yh = (RR + rr) * np.sin(tt) - dd * np.sin(ratio * tt)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out0 = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out = _resample_xy_polyline_uniform(np.asarray(out0, dtype=np.float64), max(int(n), 256)).astype(
        np.float32
    )
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_shuttle_line_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    ff = float(0.95)
    nn = max(int(n), 8)
    n_out = max(nn // 2, 2)
    n_ret = nn - n_out + 1
    if ux >= uy:
        rx = float(0.5 * ux * ff)
        x_fwd = np.linspace(cx - rx, cx + rx, n_out, dtype=np.float64)
        x_ret = np.linspace(cx + rx, cx - rx, n_ret, dtype=np.float64)[1:]
        x_all = np.concatenate([x_fwd, x_ret])
        y_all = np.full_like(x_all, cy)
    else:
        ry = float(0.5 * uy * ff)
        y_fwd = np.linspace(cy - ry, cy + ry, n_out, dtype=np.float64)
        y_ret = np.linspace(cy + ry, cy - ry, n_ret, dtype=np.float64)[1:]
        y_all = np.concatenate([y_fwd, y_ret])
        x_all = np.full_like(y_all, cx)
    out = np.stack([x_all, y_all], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_cross_axis_shuttle_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    ff = float(0.95)
    rx = float(0.5 * ux * ff)
    ry = float(0.5 * uy * ff)
    nn = max(int(n), 16)
    q = max(nn // 4, 2)
    n_h_out = q
    n_h_ret = q
    n_v_out = q
    n_v_ret = max(nn - n_h_out - n_h_ret - n_v_out, 2)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    x1 = np.linspace(cx - rx, cx + rx, n_h_out, dtype=np.float64)
    y1 = np.full(n_h_out, cy, dtype=np.float64)
    xs.append(x1)
    ys.append(y1)
    x2 = np.linspace(cx + rx, cx - rx, n_h_ret, dtype=np.float64)[1:]
    y2 = np.full_like(x2, cy)
    if x2.shape[0] > 0:
        xs.append(x2)
        ys.append(y2)
    y3 = np.linspace(cy - ry, cy + ry, n_v_out, dtype=np.float64)
    x3 = np.full_like(y3, cx)
    xs.append(x3)
    ys.append(y3)
    y4 = np.linspace(cy + ry, cy - ry, n_v_ret, dtype=np.float64)[1:]
    x4 = np.full_like(y4, cx)
    if y4.shape[0] > 0:
        xs.append(x4)
        ys.append(y4)
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    if x_all.shape[0] < nn:
        rep = nn - int(x_all.shape[0])
        x_all = np.concatenate([x_all, np.repeat(x_all[-1:], rep)])
        y_all = np.concatenate([y_all, np.repeat(y_all[-1:], rep)])
    elif x_all.shape[0] > nn:
        x_all = x_all[:nn]
        y_all = y_all[:nn]
    out = np.stack([x_all, y_all], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_teardrop_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 64)
    tt = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    xh = np.cos(tt)
    st = np.sin(0.5 * tt)
    yh = np.sin(tt) * (st * st) * 2.35
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out.astype(np.float32)


def _plan_phyllotaxis_disk_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    ga = np.pi * (3.0 - np.sqrt(5.0))
    nn = max(int(n), 80)
    k = np.arange(1, nn + 1, dtype=np.float64)
    rho = np.sqrt(k / float(max(nn, 1)))
    th = ga * k
    xh = rho * np.cos(th)
    yh = rho * np.sin(th)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out.astype(np.float32)


def _plan_logarithmic_spiral_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    ff = 0.95
    tn = np.linspace(0.0, 10.5 * np.pi, max(int(n), 384), endpoint=True)
    b = float(0.14)
    r = np.exp(b * tn)
    xh = np.cos(tn) * r
    yh = np.sin(tn) * r
    fx = float(0.5 * ux * ff * 0.55)
    fy = float(0.5 * uy * ff * 0.55)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_hyperbolic_spiral_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 256)
    tn = np.linspace(0.32, 22.5 * np.pi, nn)
    xh = np.cos(tn) / tn * 18.0
    yh = np.sin(tn) / tn * 18.0
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_involute_circle_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    tn = np.linspace(0.0, 15.7, max(int(n), 384), endpoint=False)
    xh = np.cos(tn) + tn * np.sin(tn)
    yh = np.sin(tn) - tn * np.cos(tn)
    fx = float(0.5 * ux * 0.95 * 0.22)
    fy = float(0.5 * uy * 0.95 * 0.22)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_superellipse_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
    power_n: float = 4.0,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nexp = float(max(2.1, float(power_n)))
    nn = max(int(n), 64)
    tn = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    c = np.cos(tn)
    s = np.sin(tn)
    xh = np.sign(c) * (np.abs(c) ** (2.0 / nexp))
    yh = np.sign(s) * (np.abs(s) ** (2.0 / nexp))
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_butterfly_rice_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 512)
    tn = np.linspace(0.0, 48.84, nn, endpoint=False)
    xh = np.sin(tn) * (
        np.exp(np.cos(tn)) - 2.0 * np.cos(4.0 * tn) + np.sin(tn / 12.0) ** 5
    )
    yh = np.cos(tn) * (
        np.exp(np.cos(tn)) - 2.0 * np.cos(4.0 * tn) + np.sin(tn / 12.0) ** 5
    )
    fx = float(0.45 * ux * 0.95)
    fy = float(0.45 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_gaussian_track_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:
    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 64)
    ff = 0.95
    if ux >= uy:
        xs = np.linspace(lx, hx, nn, dtype=np.float64)
        z = (xs - cx) / max(float(0.42 * ux * ff), 1e-6)
        wav = np.exp(-(z ** 2) * 2.4)
        ys = cy + (wav - 0.06) * (0.5 * uy * ff * 1.08)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        ys = np.linspace(ly, hy, nn, dtype=np.float64)
        z = (ys - cy) / max(float(0.42 * uy * ff), 1e-6)
        wav = np.exp(-(z ** 2) * 2.4)
        xs = cx + (wav - 0.06) * (0.5 * ux * ff * 1.08)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_arc_chain_wave_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    ff = 0.95
    xs = np.linspace(lx + 0.04 * ux, hx - 0.04 * ux, max(int(n), 96), dtype=np.float64)
    zs_u = np.clip((xs - cx) / max(0.5 * ux * ff, 1e-6), -2.9, 2.9)
    zs = np.cosh(zs_u)
    zs = zs - float(zs[len(zs) // 2])
    ys = cy + zs * (0.5 * uy * ff * 0.58)
    out = np.stack([xs.astype(np.float64), ys], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_bernoulli_lemniscate_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 256)
    th = np.linspace(-np.pi * 0.25 + 0.02, np.pi * 0.75 - 0.02, nn)
    c2 = np.cos(2.0 * th)
    r = np.sqrt(np.maximum(c2, 0.0))
    xh = r * np.cos(th)
    yh = r * np.sin(th)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_cochleoid_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 256)
    tn = np.linspace(-4.95 * np.pi, 5.95 * np.pi, nn)
    denom = tn + np.where(tn >= 0, 1e-6, -1e-6)
    rad = np.sin(tn) / denom
    xh = rad * np.cos(tn)
    yh = rad * np.sin(tn)
    fx = float(0.52 * ux * 0.95)
    fy = float(0.52 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_steiner_folium_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    uu = np.linspace(-4.95, -1.01, max(int(n) // 2, 96))
    uu = np.concatenate([uu, np.linspace(-0.99, 28.5, max(int(n), 192))])
    den = (1.0 + uu**3).astype(np.float64)
    den = np.where(np.abs(den) < 0.035, np.nan, den)
    xh = 3.0 * uu / den
    yh = (3.0 * uu**2) / den
    m = np.isfinite(xh) & np.isfinite(yh) & np.abs(yh) < 9e12
    xh = xh[m]
    yh = yh[m]
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out = _resample_xy_polyline_uniform(np.asarray(out, dtype=np.float64), max(int(n), 384)).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_archimedean_spiral_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 160)
    th_max = float(17.5 * np.pi)
    th = np.linspace(0.0, th_max, nn, dtype=np.float64)
    rho = (th / max(th_max, 1e-9)).astype(np.float64)
    ff = float(0.95)
    r_max = float(0.5 * ff * min(ux, uy))
    x = cx + r_max * rho * np.cos(th)
    y = cy + r_max * rho * np.sin(th)
    out = np.stack([x, y], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_clothoid_segment_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 224)
    smax = float(7.25)
    s = np.linspace(-smax, smax, nn, dtype=np.float64)
    kappa_scale = float(0.52)
    psi = kappa_scale * (s ** 2)
    vx = np.cos(psi)
    vy = np.sin(psi)
    xh = np.concatenate([[0.0], np.cumsum(0.5 * (vx[:-1] + vx[1:]) * np.diff(s))])
    yh = np.concatenate([[0.0], np.cumsum(0.5 * (vy[:-1] + vy[1:]) * np.diff(s))])
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_tractrix_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 96)
    eps = float(0.09)
    t = np.linspace(eps, np.pi - eps, nn)
    xh = np.log(np.tan(0.5 * t)) + np.cos(t)
    yh = np.sin(t)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_witch_of_agnesi_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 96)
    th = np.linspace(-1.48, 1.48, nn)
    xh = 2.0 * np.tan(th)
    yh = 2.0 * (np.cos(th) ** 2)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_cassini_oval_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 320)
    th = np.linspace(0.0, 2.0 * np.pi, nn, endpoint=False)
    a = float(1.0)
    k_shape = float(1.28)
    rhs = float(k_shape**2)
    cos_t = np.cos(th)
    cos2 = cos_t**2
    c1_coeff = 2.0 * a * a * (1.0 - 2.0 * cos2)
    c0_coeff = a**4 - rhs
    disc = c1_coeff * c1_coeff - 4.0 * c0_coeff
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    u1 = (-c1_coeff + sqrt_disc) * 0.5
    u2 = (-c1_coeff - sqrt_disc) * 0.5
    u_eff = np.maximum(np.maximum(u1, u2), 0.0)
    r = np.sqrt(np.maximum(u_eff, 0.0))
    xh = r * cos_t
    yh = r * np.sin(th)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_hypotrochoid_default_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    R, r, d = 8.0, 5.0, 4.55
    rr = float(max(r, 1e-9))
    ratio = float((R - r) / rr)
    gn = max(math.gcd(int(round(R)), int(round(r))), 1)
    tn = max(int(n) * 28, 768)
    tt = np.linspace(0.0, 2.0 * np.pi * rr / float(gn), tn, endpoint=False)
    xh = (R - r) * np.cos(tt) + d * np.cos(ratio * tt)
    yh = (R - r) * np.sin(tt) - d * np.sin(ratio * tt)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_parabolic_arc_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, lx, ly, hx, hy = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    nn = max(int(n), 48)
    ff = float(0.95)
    if ux >= uy:
        uu = np.linspace(-1.0, 1.0, nn, dtype=np.float64)
        xs = cx + uu * (0.5 * ux * ff)
        ys = cy + (uu**2) * (0.5 * uy * ff * 0.92)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        uu = np.linspace(-1.0, 1.0, nn, dtype=np.float64)
        ys = cy + uu * (0.5 * uy * ff)
        xs = cx + (uu**2) * (0.5 * ux * ff * 0.92)
        out = np.stack([xs, ys], axis=1).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], lx, hx)
    out[:, 1] = np.clip(out[:, 1], ly, hy)
    return out


def _plan_nephroid_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    tn = max(int(n), 224)
    tt = np.linspace(0.0, 2.0 * np.pi, tn, endpoint=False)
    xh = 3.0 * np.cos(tt) - np.cos(3.0 * tt)
    yh = 3.0 * np.sin(tt) - np.sin(3.0 * tt)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_limacon_pascal_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    tt = np.linspace(0.0, 2.0 * np.pi, max(int(n), 128), endpoint=False)
    ec = 1.0
    ecc = 0.74
    rho = ec + ecc * np.cos(tt)
    rho = np.maximum(rho, 0.02)
    xh = rho * np.cos(tt)
    yh = rho * np.sin(tt)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_cissoid_diocles_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    half = max(int(n) // 2, 96)
    tn = np.linspace(-6.25, 6.25, half, dtype=np.float64)
    den = 1.0 + tn**2
    xf = (tn**2) / den
    yf = (tn**3) / den
    xh = np.concatenate([xf, xf[::-1]], axis=0)
    yh = np.concatenate([yf, yf[::-1]], axis=0)
    fx = float(0.5 * ux * 0.96)
    fy = float(0.5 * uy * 0.96)
    out0 = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out = _resample_xy_polyline_uniform(np.asarray(out0, dtype=np.float64), max(int(n), half * 2)).astype(
        np.float32
    )
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_strophoid_right_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    half = max(int(n) // 2, 112)
    uu = np.linspace(-5.2, 5.2, half, dtype=np.float64)
    den = 1.0 + uu**2
    xf = (1.0 - uu**2) / den
    yf = uu * (1.0 - uu**2) / den
    xh = np.concatenate([xf, xf[::-1]], axis=0)
    yh = np.concatenate([yf, yf[::-1]], axis=0)
    fx = float(0.5 * ux * 0.96)
    fy = float(0.5 * uy * 0.96)
    out0 = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out = _resample_xy_polyline_uniform(np.asarray(out0, dtype=np.float64), max(int(n), half * 2)).astype(
        np.float32
    )
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_kampyle_eudoxus_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    half = max(int(n) // 2, 128)
    a = float(1.0)
    th = np.linspace(-1.32, 1.32, half, dtype=np.float64)
    c_raw = np.cos(th)
    s = np.sin(th)
    c_safe = np.sign(c_raw) * np.maximum(np.abs(c_raw), 8.0e-3)
    sec = a / c_safe
    xf = sec
    yf = sec * (s / c_safe)
    xh = np.concatenate([xf, xf[::-1]], axis=0)
    yh = np.concatenate([yf, yf[::-1]], axis=0)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out0 = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out = _resample_xy_polyline_uniform(np.asarray(out0, dtype=np.float64), max(int(n), half * 2)).astype(
        np.float32
    )
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _plan_conchoid_nicomedes_xy_polyline(
    rect_lo_xy: np.ndarray,
    rect_hi_xy: np.ndarray,
    n: int,
    *,
    fill_frac: float,
    edge_margin_frac: float,
) -> np.ndarray:

    _ = fill_frac
    lo = np.asarray(rect_lo_xy, dtype=np.float64).reshape(2)
    hi = np.asarray(rect_hi_xy, dtype=np.float64).reshape(2)
    cx, cy, ux, uy, *_ = _scenic_xy_usable_box(lo, hi, edge_margin_frac=edge_margin_frac)
    half = max(int(n) // 2, 128)
    th = np.linspace(-1.32, 1.32, half, dtype=np.float64)
    st = np.sin(th)
    c_raw = np.cos(th)
    c_safe = np.sign(c_raw) * np.maximum(np.abs(c_raw), 8.0e-3)
    a_lin = float(0.56)
    b_off = float(0.40)
    tan_t = st / c_safe
    xf = a_lin + b_off * np.cos(th)
    yf = a_lin * tan_t + b_off * st
    xh = np.concatenate([xf, xf[::-1]], axis=0)
    yh = np.concatenate([yf, yf[::-1]], axis=0)
    fx = float(0.5 * ux * 0.95)
    fy = float(0.5 * uy * 0.95)
    out0 = _scenic_normalize_and_place_xy(xh, yh, cx=cx, cy=cy, fx=fx, fy=fy)
    out = _resample_xy_polyline_uniform(np.asarray(out0, dtype=np.float64), max(int(n), half * 2)).astype(
        np.float32
    )
    out[:, 0] = np.clip(out[:, 0], lo[0], hi[0])
    out[:, 1] = np.clip(out[:, 1], lo[1], hi[1])
    return out


def _scenic_force_square_xy_in_workspace_footprint(
    w_lo_w: np.ndarray,
    w_hi_w: np.ndarray,
    margin_xy: float,
    *,
    prior_fill_tag: str,
    long_side_scale: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, str]:

    ws_span = w_hi_w[0:2] - w_lo_w[0:2]
    ws_center_xy = (w_lo_w[0:2] + w_hi_w[0:2]) * 0.5
    ws_long_side = float(np.max(ws_span))
    ls = float(np.clip(long_side_scale, 0.1, 0.99))
    target_scale = ws_long_side * ls
    tl = ws_center_xy - 0.5 * target_scale
    th = ws_center_xy + 0.5 * target_scale
    tl = np.maximum(tl, w_lo_w[0:2] + margin_xy)
    th = np.minimum(th, w_hi_w[0:2] - margin_xy)
    tag = f"scenic_force_square_{target_scale:.2f}m@{prior_fill_tag}"
    print(
        "[demo] scenic trajectory: unified XY force-square "
        f"(side≈{target_scale:.3f}m = {ls:.0%}×workspace_long_side {ws_long_side:.3f}m; "
        f"prior_rect_tag={prior_fill_tag!r})"
    )
    return tl, th, tag


def _resolve_task_sequence(
    raw_seq: list[dict] | None,
    placed_cubes: list[dict],
    default_arrive_radius: float,
    default_dwell_steps: int,
    default_phrase: str,
) -> list[dict]:

    if raw_seq is not None:
        if len(raw_seq) == 0:
            return []
    else:
        return [
            {
                "instruction": _instruction_phrase_from_defaults(
                    default_phrase,
                    color=str(c["color"]),
                    billboard_id=(
                        str(int(c["portal_label"]))
                        if c.get("portal_label") is not None
                        else ""
                    ),
                ),
                "target_xyz": list(c["pos"]),
                "arrive_radius": float(default_arrive_radius),
                "dwell_steps": int(default_dwell_steps),
                "label": (
                    f"through-billboard-{int(c['portal_label']):02d}"
                    if c.get("portal_label") is not None
                    else f"go-to-{c['color']}"
                ),
            }
            for c in placed_cubes
        ]
    out: list[dict] = []
    for i, t in enumerate(raw_seq):
        if "target_xyz" not in t:
            raise ValueError(f"task_sequence[{i}] missing 'target_xyz'")
        out.append(
            {
                "instruction": str(
                    t.get(
                        "instruction",
                        _instruction_phrase_from_defaults(default_phrase, color="block", billboard_id=""),
                    )
                ),
                "target_xyz": list(t["target_xyz"]),
                "arrive_radius": float(t.get("arrive_radius", default_arrive_radius)),
                "dwell_steps": int(t.get("dwell_steps", default_dwell_steps)),
                "label": str(t.get("label", f"task_{i}")),
            }
        )
    return out


def run_widowx_drone_demo(
    *,
    server_url: str,
    instruction: str,
    mission_cmd: str | None = None,
    sim_steps: int,
    xvla_steps: int,
    gui: bool,
    speed_scale: float,
    infer_every: int,
    dt: float,
    realtime_sleep: bool,
    log_every: int,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    treat_pos_as: str,
    delta_pos_scale: float,
    pos_lerp_alpha: float,
    rot_lerp_alpha: float,
    with_objects: bool,
    cam_eye: np.ndarray,
    cam_look: np.ndarray,
    record_visualization: bool,
    recording_folder: Path | None,
    recording_format: str,
    recording_fps: float,
    record_every: int,
    recording_width: int,
    recording_height: int,
    task_sequence_raw: list[dict] | None = None,
    task_default_phrase: str = (
        "Fly through rectangular portal billboard_id={billboard_id} ({color} visual); align with the long narrow opening."
    ),
    task_default_arrive_radius: float = 0.05,
    task_default_dwell_steps: int = 3,
    stop_when_all_visited: bool = True,
    cubes_override: list[dict] | None = None,
    target_pull_alpha: float = 0.0,
    target_snap_radius: float = 0.0,
    movement_deadband: float = 0.0,
    phase1_stall_window: int = 20,
    phase1_stall_threshold: float = 0.005,
    phase1_max_steps: int = 45,
    phase2_p_gain: float = 4.0,
    phase2_max_speed: float = 0.08,
    recording_scene_fov: float = 42.0,
    recording_scene_margin: float = 1.55,
    recording_camera_distance_scale: float = 1.0,
    recording_topview_distance_scale: float = 1.0,
    recording_stereo45_distance_scale: float = 1.0,
    recording_final_hold_frames: int = 18,
    trail_enabled: bool = True,
    trail_rgba: list[float] | None = None,
    trail_radius: float = 0.006,
    trail_min_distance: float = 0.01,
    virtual_workspace_enabled: bool = True,
    virtual_workspace_margin: float = 0.02,
    fpv_slot_align: bool = False,
    fpv_slot_align_dist: float = 0.35,
    fpv_use_sim_truth_pose: bool = True,
    portal_match_tol: float = 0.12,
    portal_pass_through_enabled: bool = True,
    portal_pass_approach_offset: float = 0.14,
    portal_pass_exit_offset: float = 0.14,
    portal_pass_stage_switch_radius: float | None = None,
    fpv_cam_offset_body: tuple[float, float, float] = (0.06, 0.0, 0.02),
    fpv_cam_look_body: tuple[float, float, float] = (0.85, 0.0, 0.0),
    fpv_cam_width: int = 256,
    fpv_cam_height: int = 256,
    fpv_cam_fov: float = 72.0,
    gate_pose_use_dedicated_camera: bool = False,
    gate_pose_cam_offset_body: tuple[float, float, float] = (0.06, 0.0, 0.032),
    gate_pose_cam_look_body: tuple[float, float, float] = (0.95, 0.0, -0.08),
    gate_pose_cam_width: int = 320,
    gate_pose_cam_height: int = 320,
    gate_pose_cam_fov: float = 82.0,
    gate_pose_cv_rgb_pad: int = 45,
    gate_pose_cv_min_area_ratio: float = 0.015,
    gate_pose_cv_fallback_map_pose: bool = True,
    gate_pose_estimator: str = "opencv",
    xvla_gate_instruction_template: str = (
        "Orient end-effector with long opening of rectangular portal billboard_id={billboard_id} ({color}); pass through."
    ),
    xvla_gate_steps: int = 4,
    xvla_gate_infer_width: int = 256,
    xvla_gate_infer_height: int = 256,
    gate_pose_xvla_fallback_opencv: bool = True,
    language_only_motion_amplify: float = 1.0,
    language_only_min_step_local_m: float = 0.0,
    cmd_color_disambiguation: bool = True,
    infer_displacement_scale: float = 1.0,
    cmd_use_global_camera_target: bool = False,
    cmd_global_cam_pull_alpha: float = 0.0,
    cmd_global_cam_weak_mask_scale: float = 0.35,
    cmd_global_cam_weak_mask_min_pull: float = 0.12,
    cmd_global_cam_portal_min_pull: float = 0.18,
    workspace_camera_width: int = 256,
    workspace_camera_height: int = 256,
    workspace_camera_fov: float = 55.0,
    cmd_global_cam_min_area_ratio: float = 0.015,
    cmd_coarse_plan_once: bool = False,
    cmd_coarse_plan_steps: int = 48,
    cmd_coarse_plan_smooth_window: int = 5,
    cmd_precision_zone_enable: bool = True,
    cmd_precision_zone_scale: float = 1.2,
    cmd_precision_zone_inflate_m: float = 0.08,
    cmd_precision_zone_min_r: float = 0.10,
    cmd_precision_infer_every: int = 1,
    language_only_cruise_z_clamp: bool = True,
    language_only_cruise_z_margin_lo_frac: float = 0.10,
    language_only_cruise_z_margin_hi_frac: float = 0.08,
    language_only_air_path_max_cam_pull: float | None = 0.22,
    language_only_demo_trajectory_fill: bool = True,
    language_only_demo_height_factor: float = 1.44,
    language_only_demo_plan_blend_beta: float = 0.38,
    language_only_demo_plan_fill_frac: float = 0.40,
    language_only_demo_plan_edge_margin_frac: float = 0.08,
    language_only_demo_traj_period_infers: int = 32,
    language_only_demo_xvla_trajectory_once: bool = True,
    language_only_demo_traj_plan_steps: int = 64,
    language_only_demo_plan_use_topdown_camera: bool = True,
    language_only_demo_fill_xy_margin_m: float = 0.025,
    language_only_demo_scenic_xy_force_long_scale: float = 0.85,
    language_only_demo_clearance_frac_above_tallest: float = 0.50,
    language_only_demo_clearance_above_scene_z_m: float = 0.0,
    language_only_demo_scenic_formula_min_waypoints: int = 512,
    xvla_act_request_timeout_s: float = 300.0,
    xvla_scene_semantic_context: bool = True,
    xvla_path_planning_instruction_suffix: str = DEFAULT_XVLA_PATH_PLANNING_INSTRUCTION_SUFFIX,
    local_avoidance_enabled: bool = True,
    local_avoidance_robot_radius: float = 0.048,
    local_avoidance_influence_m: float = 0.16,
    local_avoidance_gain: float = 0.065,
    local_avoidance_target_max_shift_m: float = 0.09,
    local_avoidance_phase2_gain: float = 0.85,
    local_avoidance_exclude_goal_tol_m: float = 0.11,
    navigation_phase2_xvla_steps: int = 48,
    navigation_phase2_sync_root_config: bool = False,
    navigation_phase2_sync_qs: bool = False,
    navigation_phase2_extra_instruction: str = "",
    qs_policy_path_for_sync: Path | None = None,
    navigation_phase2_geom_astar: bool = True,
    navigation_phase2_astar_cell_m: float = 0.04,
    navigation_phase1_corridor_margin_m: float = 0.18,
    navigation_phase1_corridor_bandwidth_m: float = 0.12,
    navigation_collision_pad_m: float | None = None,
    navigation_use_phase3_refined_path_in_sim: bool = False,
    navigation_sim_recording_right_distance_scale: float = 1.3,
    navigation_sim_recording_top_distance_scale: float = 1.1,
    navigation_sim_recording_stereo45_distance_scale: float = 0.7,
    navigation_phase2_astar_obstacle_pad_m: float = 0.07,
    navigation_phase2_optional_topdown_xvla: bool = False,
    navigation_phase2_z_clearance_enabled: bool = True,
    navigation_phase2_z_clearance_margin_m: float = 0.08,
    navigation_phase2_z_workspace_margin_m: float = 0.02,
    navigation_phase3_xvla_action_classify: bool = True,
    navigation_phase3_xvla_steps: int = 1,
    navigation_phase3_feedback_alpha: float = 0.10,
    navigation_recording_use_opengl: bool = True,
    cmd_goal_coupled_virtual_base: bool = True,
    vb_smooth_alpha: float = 0.42,
    vb_max_speed_m_s: float = 4.0,
    vb_jump_warn_m: float = 0.35,
):

    p, pybullet_data = require_pybullet()
    mode = p.GUI if gui else p.DIRECT
    cid = p.connect(mode)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet.")

    _use_opengl_recording = bool(gui) and bool(navigation_recording_use_opengl)
    set_recording_prefer_opengl(_use_opengl_recording)
    if gui:
        configure_pybullet_gui_opengl_transparency(p)
        if _use_opengl_recording:
            print("[render] GUI OpenGL enabled: feedback spheres use rgba alpha blending.")

    world_front_frames: list[np.ndarray] = []
    world_right_frames: list[np.ndarray] = []
    world_top_frames: list[np.ndarray] = []
    world_45deg_frames: list[np.ndarray] = []
    rec_folder = recording_folder if record_visualization else None
    steps_executed = 0
    infer_calls = 0

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        placed_cubes = build_scene(p, with_objects=with_objects, cubes=cubes_override)

        workspace_lo = np.asarray(workspace_lo, dtype=np.float32)
        workspace_hi = np.asarray(workspace_hi, dtype=np.float32)
        if with_objects and placed_cubes:
            obj_aabb = _scene_objects_aabb_world(placed_cubes)
            if obj_aabb is not None:
                o_lo, o_hi = obj_aabb
                pad = np.array([0.05, 0.05, 0.05], dtype=np.float64)
                new_lo = np.minimum(workspace_lo.astype(np.float64), o_lo - pad)
                new_hi = np.maximum(workspace_hi.astype(np.float64), o_hi + pad)

                workspace_lo = new_lo.astype(np.float32)
                workspace_hi = new_hi.astype(np.float32)
                print(
                    "[scene] expand workspace to include all objects: "
                    f"lo={workspace_lo.round(3).tolist()} hi={workspace_hi.round(3).tolist()}"
                )

        scene_catalog_str = ""
        if bool(xvla_scene_semantic_context) and bool(with_objects) and placed_cubes:
            scene_catalog_str = format_scene_semantic_catalog_for_xvla(
                placed_cubes, workspace_lo, workspace_hi
            )
            print(
                "[nav] X-VLA scene semantic catalog enabled: "
                f"{len(placed_cubes)} objects listed in language_instruction for /act."
            )

        def _instruction_for_xvla_call(lang_line: str) -> str:
            return compose_xvla_navigation_instruction(
                str(lang_line),
                scene_catalog=scene_catalog_str,
                enabled=bool(xvla_scene_semantic_context and scene_catalog_str),
                planning_suffix=xvla_path_planning_instruction_suffix,
            )

        task_sequence = _resolve_task_sequence(
            task_sequence_raw,
            placed_cubes if with_objects else [],
            task_default_arrive_radius,
            task_default_dwell_steps,
            task_default_phrase,
        )
        if mission_cmd:
            mc = str(mission_cmd).strip()
            if mc:
                for tt in task_sequence:
                    tt["instruction"] = f"Mission: {mc}. Subtask: {tt['instruction']}"

        eff_instruction = instruction
        if (
            mission_cmd
            and str(mission_cmd).strip()
            and with_objects
            and placed_cubes
        ):
            aug_b = augment_instruction_billboard_disambiguation(eff_instruction, placed_cubes)
            if aug_b != eff_instruction:
                print(
                    "[cmd] language_instruction augmented for billboard_id grounding — "
                    f"was: {eff_instruction!r}\n"
                    f"     now: {aug_b!r}"
                )
                eff_instruction = aug_b
        if (
            mission_cmd
            and str(mission_cmd).strip()
            and cmd_color_disambiguation
            and with_objects
            and placed_cubes
        ):
            aug = augment_instruction_color_disambiguation(eff_instruction, placed_cubes)
            if aug != eff_instruction:
                print(
                    "[cmd] language_instruction augmented for color grounding — "
                    f"was: {eff_instruction!r}\n"
                    f"     now: {aug!r}"
                )
                eff_instruction = aug

        cur_task_idx = 0
        dwell_count = 0
        visited_log: list[dict] = []

        nav_phase = 1
        task_step_count = 0
        dist_history: list[float] = []

        portal_pass_cached_idx = -1
        portal_pass_stage = 0
        portal_pass_ctx: dict[str, np.ndarray] | None = None

        ws_center = (workspace_lo + workspace_hi) * 0.5
        virtual_base_world = np.zeros(3, dtype=np.float32)
        drone_pos = ws_center.astype(np.float32).copy()
        drone_pos[1] += float(DEFAULT_DRONE_START_OFFSET_TOPVIEW_UP_M)
        drone_pos = np.minimum(np.maximum(drone_pos, workspace_lo), workspace_hi).astype(np.float32)
        drone_R = np.eye(3, dtype=np.float32)
        target_pos = drone_pos.copy()
        target_R = drone_R.copy()
        gripper_state = 0.0

        body_uid, rotor_uids, offsets = create_floating_drone(p, drone_pos.tolist())
        update_floating_drone(p, body_uid, rotor_uids, offsets, drone_pos, drone_R)
        trail_last_pos = drone_pos.copy()
        trail_color = trail_rgba if trail_rgba is not None else [1.0, 0.68, 0.32, 0.75]
        fpv_off_b = np.asarray(fpv_cam_offset_body, dtype=np.float64)
        fpv_look_b = np.asarray(fpv_cam_look_body, dtype=np.float64)
        gate_pose_off_b = np.asarray(gate_pose_cam_offset_body, dtype=np.float64)
        gate_pose_look_b = np.asarray(gate_pose_cam_look_body, dtype=np.float64)
        scene_points = [workspace_lo.astype(np.float32), workspace_hi.astype(np.float32)]
        for task in task_sequence:
            scene_points.append(np.asarray(task["target_xyz"], dtype=np.float32))
        for cube in placed_cubes:
            cube_pos = np.asarray(cube["pos"], dtype=np.float32)
            scene_points.append(cube_pos)
            bounds_half = cube.get("bounds_half")
            if bounds_half is not None:
                bh = np.asarray(bounds_half, dtype=np.float32)
                scene_points.append(cube_pos - bh)
                scene_points.append(cube_pos + bh)
        scene_arr = np.vstack(scene_points)
        recording_scene_lo = scene_arr.min(axis=0)
        recording_scene_hi = scene_arr.max(axis=0)

        gcam_target_log_once = False
        coarse_xvla_world_pts: np.ndarray | None = None
        coarse_xvla_world_Rs: list[np.ndarray] | None = None
        coarse_xvla_plan_step: int | None = None
        coarse_frozen_vb: np.ndarray | None = None
        coarse_plan_log_once = False
        phase3_sim_path_follow = False
        phase3_feedback: Phase3FeedbackZones | None = None
        _phase3_folder = None
        _run_phase123 = (
            (bool(cmd_coarse_plan_once) or bool(navigation_use_phase3_refined_path_in_sim))
            and with_objects
            and placed_cubes
        )
        if _run_phase123:

            def _portal_leg_exit_goal_coarse(portal_cube: dict, from_xyz: np.ndarray) -> np.ndarray:
                spec = _portal_pass_spec_for_task(
                    p,
                    placed_cubes,
                    np.asarray(portal_cube["pos"], dtype=np.float64),
                    np.asarray(from_xyz, dtype=np.float64),
                    match_tol=float(portal_match_tol),
                    approach_offset=float(portal_pass_approach_offset),
                    exit_offset=float(portal_pass_exit_offset),
                )
                if spec is not None:
                    return np.asarray(spec["exit"], dtype=np.float64)
                return np.asarray(portal_cube["pos"], dtype=np.float64).reshape(3)

            coarse_goal_xyz = None
            if mission_cmd and str(mission_cmd).strip():
                _sp_goal = _first_rect_portal_for_instruction_color(
                    str(mission_cmd), placed_cubes, prefer_near_xyz=np.asarray(drone_pos, dtype=np.float64)
                )
                if _sp_goal is not None:
                    coarse_goal_xyz = np.asarray(_sp_goal["pos"], dtype=np.float64).copy()
            if coarse_goal_xyz is None:
                if task_sequence:
                    coarse_goal_xyz = np.asarray(task_sequence[-1]["target_xyz"], dtype=np.float64).copy()
                else:
                    coarse_goal_xyz = np.asarray(placed_cubes[-1]["pos"], dtype=np.float64).copy()

            _drone_local_coarse = np.minimum(
                np.maximum(
                    drone_pos.astype(np.float32) - virtual_base_world.astype(np.float32),
                    workspace_lo,
                ),
                workspace_hi,
            ).astype(np.float32)

            phase3_feedback = Phase3FeedbackZones()
            set_phase3_feedback_alpha(float(navigation_phase3_feedback_alpha))
            if (
                bool(navigation_phase3_xvla_action_classify)
                and mission_cmd
                and str(mission_cmd).strip()
            ):
                _cmd_object_specs = collect_specified_objects_from_mission(
                    str(mission_cmd),
                    placed_cubes,
                    first_rect_portal_fn=_first_rect_portal_for_instruction_color,
                    prefer_near_xyz=np.asarray(drone_pos, dtype=np.float64),
                )
                if _cmd_object_specs:
                    import cv2

                    _cmd_top_rgb = capture_workspace_topdown_rgb(
                        p,
                        workspace_lo=workspace_lo,
                        workspace_hi=workspace_hi,
                        virtual_base_world=virtual_base_world,
                        world_recording_view_proj_fn=_world_recording_view_proj,
                        render_world_recording_rgb_fn=_render_world_recording_rgb,
                        recording_scene_fov=float(recording_scene_fov),
                        recording_scene_margin=float(recording_scene_margin),
                        recording_camera_distance_scale=float(recording_camera_distance_scale),
                        recording_topview_distance_scale=float(recording_topview_distance_scale),
                    )
                    _cmd_top_rgb = cv2.resize(
                        np.asarray(_cmd_top_rgb, dtype=np.uint8),
                        (
                            max(8, int(workspace_camera_width)),
                            max(8, int(workspace_camera_height)),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                    _cmd_proprio = build_proprio_widowx_ee6d(
                        _drone_local_coarse,
                        np.asarray(drone_R, dtype=np.float32),
                        float(gripper_state),
                    )
                    _cmd_action_raw = classify_mission_basic_actions_early(
                        _cmd_object_specs,
                        mission_cmd=str(mission_cmd),
                        placed_cubes=placed_cubes,
                        query_xvla_fn=query_xvla,
                        server_url=server_url,
                        topdown_rgb=_cmd_top_rgb,
                        proprio=_cmd_proprio,
                        drone_pos=np.asarray(drone_pos, dtype=np.float64).reshape(3),
                        compose_instruction_fn=compose_xvla_navigation_instruction,
                        scene_catalog=scene_catalog_str,
                        xvla_scene_semantic_context=bool(xvla_scene_semantic_context),
                        xvla_path_planning_instruction_suffix=str(
                            xvla_path_planning_instruction_suffix or ""
                        ),
                        xvla_steps=int(navigation_phase3_xvla_steps),
                        xvla_act_request_timeout_s=float(xvla_act_request_timeout_s),
                    )
                    phase3_feedback.action_cache = {
                        k: (v[0], int(v[1])) for k, v in _cmd_action_raw.items()
                    }
                    phase3_feedback.action_classify_source = {
                        k: str(v[2]) for k, v in _cmd_action_raw.items()
                    }

            _plan_result = run_navigation_phase1_and_phase2_topdown(
                p=p,
                placed_cubes=placed_cubes,
                mission_cmd=mission_cmd,
                cur_instruction=eff_instruction,
                g_world=coarse_goal_xyz,
                drone_pos=drone_pos,
                drone_pos_local=_drone_local_coarse,
                drone_R=drone_R,
                virtual_base_world=virtual_base_world,
                workspace_lo=workspace_lo,
                workspace_hi=workspace_hi,
                treat_pos_as=treat_pos_as,
                delta_pos_scale=delta_pos_scale,
                gripper_state=gripper_state,
                server_url=server_url,
                scene_catalog_str=scene_catalog_str,
                xvla_scene_semantic_context=bool(xvla_scene_semantic_context),
                xvla_path_planning_instruction_suffix=xvla_path_planning_instruction_suffix,
                workspace_camera_width=int(workspace_camera_width),
                workspace_camera_height=int(workspace_camera_height),
                navigation_phase2_xvla_steps=int(navigation_phase2_xvla_steps),
                xvla_act_request_timeout_s=float(xvla_act_request_timeout_s),
                navigation_phase2_sync_root_config=bool(navigation_phase2_sync_root_config),
                navigation_phase2_sync_qs=bool(navigation_phase2_sync_qs),
                qs_policy_path_for_sync=qs_policy_path_for_sync,
                config_json_path=CONFIG_DEFAULT_PATH,
                navigation_phase2_extra_instruction=str(navigation_phase2_extra_instruction),
                navigation_phase2_geom_astar=bool(navigation_phase2_geom_astar),
                navigation_phase2_astar_cell_m=float(navigation_phase2_astar_cell_m),
                navigation_phase1_corridor_margin_m=float(navigation_phase1_corridor_margin_m),
                navigation_phase1_corridor_bandwidth_m=float(navigation_phase1_corridor_bandwidth_m),
                navigation_collision_pad_m=navigation_collision_pad_m,
                navigation_phase2_astar_obstacle_pad_m=float(navigation_phase2_astar_obstacle_pad_m),
                navigation_phase2_optional_topdown_xvla=bool(navigation_phase2_optional_topdown_xvla),
                navigation_phase2_z_clearance_enabled=bool(navigation_phase2_z_clearance_enabled),
                navigation_phase2_z_clearance_margin_m=float(navigation_phase2_z_clearance_margin_m),
                navigation_phase2_z_workspace_margin_m=float(navigation_phase2_z_workspace_margin_m),
                recording_scene_fov=float(recording_scene_fov),
                recording_scene_margin=float(recording_scene_margin),
                recording_camera_distance_scale=float(recording_camera_distance_scale),
                recording_topview_distance_scale=float(recording_topview_distance_scale),
                recording_stereo45_distance_scale=float(recording_stereo45_distance_scale),
                rec_folder=rec_folder,
                recording_folder=recording_folder,
                root_dir=ROOT,
                world_recording_view_proj_fn=_world_recording_view_proj,
                render_world_recording_rgb_fn=_render_world_recording_rgb,
                world_xyz_to_recording_image_pixel_fn=world_xyz_to_recording_image_pixel,
                first_rect_portal_for_instruction_color_fn=_first_rect_portal_for_instruction_color,
                build_proprio_fn=build_proprio_widowx_ee6d,
                query_xvla_fn=query_xvla,
                compose_instruction_fn=compose_xvla_navigation_instruction,
                read_config_fn=read_config_json,
                load_qs_policies_fn=load_qs_policies,
                portal_leg_goal_fn=_portal_leg_exit_goal_coarse,
                create_feedback_spheres=True,
                feedback_zones_registry=phase3_feedback,
            )
            _stitched_arr: np.ndarray | None = None
            _phase3_trajectory: np.ndarray | None = None
            _phase3_trajectory_coarse: np.ndarray | None = None
            _leg_refinements: list[dict[str, Any]] = []
            if _plan_result is not None:
                phase3_feedback = _plan_result.get("phase3_zones") or phase3_feedback
                _phase3_folder = _plan_result.get("phase3_folder")
                _stitched = _plan_result.get("trajectory")
                _phase3_trajectory_coarse = _plan_result.get("trajectory_coarse")
                _leg_refinements = list(_plan_result.get("leg_refinements") or [])
                if _stitched is not None:
                    _stitched_arr = np.asarray(_stitched, dtype=np.float32).reshape(-1, 3)
                    if _stitched_arr.shape[0] >= 1:
                        print(
                            f"[Multi-leg] Stitched Phase1+2 geom trajectory (planning only): "
                            f"waypoints={_stitched_arr.shape[0]}"
                        )
                        for i, row in enumerate(_stitched_arr):
                            print(
                                f"  g{i}: XY=[{float(row[0]):.4f}, {float(row[1]):.4f}] "
                                f"Z={float(row[2]):.4f}"
                            )

            _phase3_trajectory = _stitched_arr
            if _stitched_arr is not None and _stitched_arr.shape[0] >= 1:
                _start = np.asarray(drone_pos, dtype=np.float64).reshape(1, 3)
                _phase3_trajectory = np.vstack([_start, _stitched_arr.astype(np.float64)])
                if _phase3_trajectory_coarse is not None:
                    _coarse = np.asarray(_phase3_trajectory_coarse, dtype=np.float64).reshape(-1, 3)
                    if _coarse.shape[0] >= 1:
                        _phase3_trajectory_coarse = np.vstack([_start, _coarse])
                    else:
                        _phase3_trajectory_coarse = None
                else:
                    _phase3_trajectory_coarse = None

            _phase3_recording_kw = {
                "virtual_base_world": virtual_base_world,
                "workspace_lo": workspace_lo,
                "workspace_hi": workspace_hi,
                "placed_cubes": placed_cubes,
                "world_recording_view_proj_fn": _world_recording_view_proj,
                "render_world_recording_rgb_fn": _render_world_recording_rgb,
                "world_xyz_to_recording_image_pixel_fn": world_xyz_to_recording_image_pixel,
                "recording_scene_fov": float(recording_scene_fov),
                "recording_scene_margin": float(recording_scene_margin),
                "recording_camera_distance_scale": float(recording_camera_distance_scale),
                "recording_topview_distance_scale": float(recording_topview_distance_scale),
                "recording_stereo45_distance_scale": float(recording_stereo45_distance_scale),
                "use_opengl_sphere_transparency": bool(_use_opengl_recording),
                "navigation_collision_pad_m": float(
                    navigation_collision_pad_m
                    if navigation_collision_pad_m is not None
                    else navigation_phase2_astar_obstacle_pad_m
                ),
            }
            _p3_out = run_phase3_apply_feedback_colors(
                p,
                phase3_feedback,
                trajectory=_phase3_trajectory,
                mission_cmd=mission_cmd,
                placed_cubes=placed_cubes,
                phase3_folder=_phase3_folder,
                recording_kwargs=_phase3_recording_kw if _phase3_folder is not None else None,
                prior_refinements=_leg_refinements if _leg_refinements else None,
                trajectory_coarse=_phase3_trajectory_coarse,
            )
            _refined_traj = _p3_out.get("trajectory")
            if _refined_traj is not None:
                _phase3_trajectory = np.asarray(_refined_traj, dtype=np.float64).reshape(-1, 3)
            if bool(navigation_use_phase3_refined_path_in_sim) and _phase3_trajectory is not None:
                from algorithms.multi_leg import polyline_yaw_rotations

                _follow_pts = np.asarray(_phase3_trajectory, dtype=np.float32).reshape(-1, 3)
                if _follow_pts.shape[0] >= 1:
                    coarse_xvla_world_pts = _follow_pts
                    coarse_xvla_world_Rs = polyline_yaw_rotations(_follow_pts)
                    coarse_xvla_plan_step = 0
                    coarse_frozen_vb = np.asarray(virtual_base_world, dtype=np.float32).copy()
                    phase3_sim_path_follow = True
                    print(
                        f"[sim] Phase3 refined trajectory loaded for waypoint following "
                        f"({coarse_xvla_world_pts.shape[0]} waypoints)"
                    )
            if bool(cmd_coarse_plan_once) and not bool(navigation_use_phase3_refined_path_in_sim):
                print(">> Phase1+2+3 complete; paused after Phase3 color feedback; not entering main simulation.")
                sys.exit(0)
            elif bool(navigation_use_phase3_refined_path_in_sim) and phase3_sim_path_follow:
                print(">> Phase1+2+3 complete; continuing to main simulation (Phase3 path replay + recording, no X-VLA inference).")
                if rec_folder is not None:
                    rec_folder = Path(rec_folder) / "sim"
                    rec_folder.mkdir(parents=True, exist_ok=True)
                    print(f"[record] Global simulation recording folder: {rec_folder.resolve()}")
                    print(
                        "[record] sim view distance scale (front unchanged): "
                        f"right×{float(navigation_sim_recording_right_distance_scale):.2f} "
                        f"top×{float(navigation_sim_recording_top_distance_scale):.2f} "
                        f"45deg×{float(navigation_sim_recording_stereo45_distance_scale):.2f}"
                    )
                elif record_visualization:
                    print("[record] WARN: record_visualization enabled but rec_folder is empty; simulation will not record.")
            elif bool(navigation_use_phase3_refined_path_in_sim) and not phase3_sim_path_follow:
                print(
                    "[sim] WARN: navigation_use_phase3_refined_path_in_sim=true but no Phase3 trajectory; "
                    "running normal --cmd main simulation."
                )

        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9, cameraYaw=55, cameraPitch=-25,
                cameraTargetPosition=ws_center.tolist(),
            )

        infer_every = max(1, int(infer_every))
        rec_every = max(1, int(record_every))
        precision_infer_every = max(1, int(cmd_precision_infer_every))

        seq_msg = (
            f"task_sequence={[t['label'] for t in task_sequence]}"
            if task_sequence else f"single_instruction={eff_instruction!r}"
        )
        print(
            "[sim] widowx-ee6d scene start | "
            f"workspace=({workspace_lo.tolist()} -> {workspace_hi.tolist()}) "
            f"dt={dt} infer_every={infer_every} treat_pos_as={treat_pos_as} "
            f"objects={with_objects} | "
            f"virtual_ws={virtual_workspace_enabled} "
            f"pull={target_pull_alpha} snap_r={target_snap_radius} deadband={movement_deadband} | "
            f"{seq_msg}"
        )
        if not task_sequence and (
            float(language_only_motion_amplify) != 1.0
            or float(language_only_min_step_local_m) > 0.0
        ):
            print(
                f"[motion] language-only target boost: amplify={language_only_motion_amplify}, "
                f"min_step_local_m={language_only_min_step_local_m}"
            )
        if float(infer_displacement_scale) != 1.0:
            print(f"[motion] infer_displacement_scale={infer_displacement_scale} (each /act local target stretch)")
        if (
            cmd_use_global_camera_target
            and float(cmd_global_cam_pull_alpha) > 0.0
            and mission_cmd
            and str(mission_cmd).strip()
        ):
            print(
                "[cmd] global workspace camera → color mask + ray/plane target | "
                f"pull={cmd_global_cam_pull_alpha} "
                f"cam={workspace_camera_width}x{workspace_camera_height} fov={workspace_camera_fov}"
            )
        if phase3_sim_path_follow:
            print(
                "[sim] Phase3 trajectory replay: no X-VLA /act during simulation; "
                "flying by interpolating Phase3 global refined polyline only"
            )
        elif cmd_coarse_plan_once and mission_cmd and str(mission_cmd).strip():
            print(
                "[cmd] coarse plan once: X-VLA one-shot path + smooth follow; "
                f"plan_steps={cmd_coarse_plan_steps} smooth_w={cmd_coarse_plan_smooth_window}"
            )
        if cmd_precision_zone_enable and not phase3_sim_path_follow and mission_cmd and str(mission_cmd).strip():
            print(
                "[precision] zone enabled: "
                f"scale={cmd_precision_zone_scale} inflate={cmd_precision_zone_inflate_m}m "
                f"min_r={cmd_precision_zone_min_r}m infer_every={precision_infer_every}"
            )
        if rec_folder is not None:
            r_scene = float(np.linalg.norm((recording_scene_hi - recording_scene_lo) * 0.5))
            fov_r = float(np.deg2rad(max(recording_scene_fov, 1.0)))
            est_d = max(
                r_scene / max(np.sin(fov_r * 0.5), 1e-6)
                * float(recording_scene_margin)
                * float(np.clip(recording_camera_distance_scale, 0.05, 100.0)),
                0.8,
            )
            est_d_top = float(est_d) * float(np.clip(recording_topview_distance_scale, 0.05, 100.0))
            est_d_stereo45 = float(est_d) * float(np.clip(recording_stereo45_distance_scale, 0.05, 100.0))
            print(
                f"[record] folder: {rec_folder.resolve()}  (~{recording_fps} fps, every {rec_every} step)\n"
                f"[record] world_cam: fov={recording_scene_fov} margin={recording_scene_margin} "
                f"distance_scale={recording_camera_distance_scale} "
                f"topview_dist_scale={recording_topview_distance_scale} "
                f"stereo45_dist_scale={recording_stereo45_distance_scale} "
                f"(~eye distance {est_d:.2f}m non-top / {est_d_top:.2f}m top / "
                f"{est_d_stereo45:.2f}m 45deg; "
                f"scene bbox radius {r_scene:.2f}m; "
                "smaller fov pushes camera farther — use margin / distance_scale to zoom in)"
            )

        vb_prev_smooth = np.zeros(3, dtype=np.float32)
        demo_traj_log_once = False
        demo_traj_phase = 0.0
        demo_xvla_world_pts: np.ndarray | None = None
        demo_xvla_world_Rs: list[np.ndarray] | None = None
        demo_xvla_plan_step: int | None = None
        demo_scenic_frozen_vb: np.ndarray | None = None
        demo_scenic_z_cap_warned: bool = False
        demo_trail_min_len: float = float(trail_min_distance)
        demo_topdown_bbox: tuple[np.ndarray, np.ndarray] | None = None
        demo_topdown_bbox_frac: float | None = None
        last_infer_ms: float | None = None
        precision_zone_last = False
        if phase3_feedback is not None and phase3_feedback.zones:
            hide_feedback_spheres_for_recording(p, phase3_feedback)
            print(f"[sim] Hiding {len(phase3_feedback.zones)} feedback sphere(s) (simulation phase)")
        for t in range(sim_steps):
            u_traj = 0.0
            if not p.getConnectionInfo().get("isConnected", True):
                print("PyBullet disconnected (window closed?). Stopping cleanly.")
                break

            cur_task = task_sequence[cur_task_idx] if task_sequence else None
            cur_instruction = cur_task["instruction"] if cur_task else eff_instruction

            portal_pass_spec: dict[str, np.ndarray] | None = None
            if cur_task is not None and portal_pass_through_enabled and placed_cubes:
                if portal_pass_cached_idx != cur_task_idx:
                    portal_pass_cached_idx = cur_task_idx
                    portal_pass_stage = 0
                    portal_pass_ctx = _portal_pass_spec_for_task(
                        p,
                        placed_cubes,
                        np.asarray(cur_task["target_xyz"], dtype=np.float32),
                        drone_pos,
                        match_tol=float(portal_match_tol),
                        approach_offset=float(portal_pass_approach_offset),
                        exit_offset=float(portal_pass_exit_offset),
                    )
                portal_pass_spec = portal_pass_ctx

            goal_center_xyz: np.ndarray | None = None
            nav_goal_xyz: np.ndarray | None = None
            if cur_task is not None:
                goal_center_xyz = np.asarray(cur_task["target_xyz"], dtype=np.float32)
                if portal_pass_spec is not None:
                    nav_goal_xyz = (
                        portal_pass_spec["approach"]
                        if portal_pass_stage == 0
                        else portal_pass_spec["exit"]
                    )
                else:
                    nav_goal_xyz = goal_center_xyz
                cur_goal_world = nav_goal_xyz.copy()
            else:
                cur_goal_world = None

            vb_goal_catalog = cmd_goal_world_estimate_for_virtual_base(
                mission_cmd,
                placed_cubes if with_objects else [],
                prefer_near_xyz=np.asarray(drone_pos, dtype=np.float64),
            )
            vb_goal_for_coupling: np.ndarray | None
            if cur_goal_world is not None:
                vb_goal_for_coupling = np.asarray(cur_goal_world, dtype=np.float64).reshape(3)
            else:
                vb_goal_for_coupling = vb_goal_catalog

            ref_demo = (
                str(mission_cmd).strip()
                if mission_cmd and str(mission_cmd).strip()
                else str(eff_instruction).strip()
            )
            demo_scenic = (
                cur_task is None
                and bool(language_only_demo_trajectory_fill)
                and bool(ref_demo)
                and _language_cmd_demo_trajectory_no_portal(ref_demo)
            )
            if phase3_sim_path_follow:
                demo_scenic = False
            _scenic_shape_taken = [False]

            def _scenic_take(shape_requested: bool) -> bool:
                if not demo_scenic or _scenic_shape_taken[0]:
                    return False
                if not shape_requested:
                    return False
                _scenic_shape_taken[0] = True
                return True

            demo_fig8 = _scenic_take(_language_cmd_requests_figure8(ref_demo))
            demo_oval = _scenic_take(_language_cmd_requests_oval_racetrack(ref_demo))
            demo_circle_orbit = _scenic_take(_language_cmd_requests_circle_orbit_uav(ref_demo))
            demo_diamond = _scenic_take(_language_cmd_requests_diamond(ref_demo))
            demo_square = _scenic_take(_language_cmd_requests_square_circuit(ref_demo))
            demo_phyllotaxis = _scenic_take(_language_cmd_requests_phyllotaxis_disk(ref_demo))
            demo_arch_spiral = _scenic_take(_language_cmd_requests_archimedean_spiral(ref_demo))
            demo_spiral = _scenic_take(_language_cmd_requests_spiral(ref_demo))
            demo_serpentine = _scenic_take(_language_cmd_requests_serpentine_scan(ref_demo))
            demo_cross_shuttle = _scenic_take(_language_cmd_requests_cross_axis_shuttle(ref_demo))
            demo_shuttle = _scenic_take(_language_cmd_requests_shuttle_line_xy(ref_demo))
            demo_clover = _scenic_take(_language_cmd_requests_clover_trefoil(ref_demo))
            demo_star = _scenic_take(_language_cmd_requests_pentagram_star(ref_demo))
            demo_lissajous = _scenic_take(_language_cmd_requests_lissajous(ref_demo))
            demo_heart = _scenic_take(_language_cmd_requests_heart_loop(ref_demo))
            demo_teardrop = _scenic_take(_language_cmd_requests_teardrop_loop_xy(ref_demo))
            demo_polygon = _scenic_take(_language_cmd_requests_regular_polygon_outline(ref_demo))
            demo_rose = _scenic_take(_language_cmd_requests_rose_rhodonea(ref_demo))
            demo_damped_sinewave = _scenic_take(_language_cmd_requests_damped_sine_wave_path(ref_demo))
            demo_trig_beat_wave = _scenic_take(_language_cmd_requests_trig_beat_wave_path(ref_demo))
            demo_triangle_wave = _scenic_take(_language_cmd_requests_triangle_wave_path(ref_demo))
            demo_tanh_ribbon = _scenic_take(_language_cmd_requests_tanh_ribbon_path(ref_demo))
            demo_cosine_wave = _scenic_take(_language_cmd_requests_cosine_wave_path(ref_demo))
            demo_sinewave = _scenic_take(_language_cmd_requests_sine_wave_path(ref_demo))
            demo_stadium_capsule = _scenic_take(_language_cmd_requests_stadium_capsule(ref_demo))
            demo_cycloid_arc = _scenic_take(_language_cmd_requests_single_row_cycloid(ref_demo))
            demo_cardioid = _scenic_take(_language_cmd_requests_cardioid_curve(ref_demo))
            demo_limacon = _scenic_take(_language_cmd_requests_limacon_pascal_xy(ref_demo))
            demo_deltoid = _scenic_take(_language_cmd_requests_deltoid_hypocycloid(ref_demo))
            demo_astroid = _scenic_take(_language_cmd_requests_hypocycloid_astroid_xy(ref_demo))
            demo_epitrochoid = _scenic_take(_language_cmd_requests_epitrochoid_outer(ref_demo))
            demo_nephroid = _scenic_take(_language_cmd_requests_nephroid_xy(ref_demo))
            demo_epicycloid = _scenic_take(_language_cmd_requests_epicycloid_outer(ref_demo))
            demo_logspiral = _scenic_take(_language_cmd_requests_log_spiral_arc(ref_demo))
            demo_hyp_spiral = _scenic_take(_language_cmd_requests_hyperbolic_spiral_curve(ref_demo))
            demo_involute = _scenic_take(_language_cmd_requests_involute_approx(ref_demo))
            demo_squircle = _scenic_take(_language_cmd_requests_superellipse_sqircle(ref_demo))
            demo_butterfly = _scenic_take(_language_cmd_requests_butterfly_rice(ref_demo))
            demo_gaussian = _scenic_take(_language_cmd_requests_gaussian_bump_path(ref_demo))
            demo_arc_chain = _scenic_take(_language_cmd_requests_arc_chain_wave(ref_demo))
            demo_bern_lemn = _scenic_take(_language_cmd_requests_lemniscate_extended(ref_demo))
            demo_cochleoid = _scenic_take(_language_cmd_requests_cochleoid_approx(ref_demo))
            demo_folium = _scenic_take(_language_cmd_requests_steiner_folium_xy(ref_demo))
            demo_clothoid = _scenic_take(_language_cmd_requests_clothoid_like(ref_demo))
            demo_tractrix = _scenic_take(_language_cmd_requests_tractrix_curve(ref_demo))
            demo_witch = _scenic_take(_language_cmd_requests_witch_of_agnesi_xy(ref_demo))
            demo_cassini = _scenic_take(_language_cmd_requests_cassini_oval_xy(ref_demo))
            demo_hypotroch = _scenic_take(_language_cmd_requests_hypotrochoid_general(ref_demo))
            demo_strophoid = _scenic_take(_language_cmd_requests_strophoid_xy(ref_demo))
            demo_kampyle = _scenic_take(_language_cmd_requests_kampyle_eudoxus_xy(ref_demo))
            demo_conchoid_nic = _scenic_take(_language_cmd_requests_conchoid_nicomedes_xy(ref_demo))
            demo_cissoid = _scenic_take(_language_cmd_requests_cissoid_xy(ref_demo))
            demo_parabola = _scenic_take(_language_cmd_requests_parabolic_arc_xy(ref_demo))
            scenic_polygon_sides = _infer_regular_polygon_outline_sides(ref_demo) if demo_polygon else None
            scenic_rose_petals = _infer_rose_petal_k(ref_demo) if demo_rose else None
            demo_helical_z = bool(demo_scenic) and bool(
                _language_cmd_requests_helical_vertical_profile(ref_demo)
            )
            scenic_helical_turns = float(_infer_helical_turns_across_path(ref_demo))
            scenic_z_trig = (
                _infer_scenic_z_trig_from_cmd(ref_demo) if demo_helical_z else "sin"
            )
            demo_xvla_once = demo_scenic and bool(language_only_demo_xvla_trajectory_once)

            demo_scenic_analytic_shape = (
                demo_fig8
                or demo_oval
                or demo_circle_orbit
                or demo_diamond
                or demo_square
                or demo_phyllotaxis
                or demo_arch_spiral
                or demo_spiral
                or demo_serpentine
                or demo_cross_shuttle
                or demo_shuttle
                or demo_clover
                or demo_star
                or demo_lissajous
                or demo_heart
                or demo_teardrop
                or demo_polygon
                or demo_rose
                or demo_damped_sinewave
                or demo_trig_beat_wave
                or demo_triangle_wave
                or demo_tanh_ribbon
                or demo_cosine_wave
                or demo_sinewave
                or demo_stadium_capsule
                or demo_cycloid_arc
                or demo_cardioid
                or demo_limacon
                or demo_deltoid
                or demo_astroid
                or demo_epitrochoid
                or demo_nephroid
                or demo_epicycloid
                or demo_logspiral
                or demo_hyp_spiral
                or demo_involute
                or demo_squircle
                or demo_butterfly
                or demo_gaussian
                or demo_arc_chain
                or demo_bern_lemn
                or demo_cochleoid
                or demo_folium
                or demo_clothoid
                or demo_tractrix
                or demo_witch
                or demo_cassini
                or demo_hypotroch
                or demo_strophoid
                or demo_kampyle
                or demo_conchoid_nic
                or demo_cissoid
                or demo_parabola
            )
            coarse_plan_enabled = bool(cmd_coarse_plan_once) and cur_task is None and not demo_scenic
            follow_world_path = (
                coarse_xvla_world_pts is not None
                and cur_task is None
                and not demo_scenic
                and (
                    coarse_plan_enabled
                    or bool(phase3_sim_path_follow)
                    or bool(navigation_use_phase3_refined_path_in_sim)
                )
            )
            if (
                phase3_sim_path_follow
                and coarse_xvla_world_pts is not None
                and cur_task is None
            ):
                follow_world_path = True
            vb_freeze = (
                demo_scenic_analytic_shape
                or (
                    virtual_workspace_enabled
                    and demo_xvla_once
                    and demo_xvla_world_pts is not None
                    and demo_scenic_frozen_vb is not None
                )
                or (
                    virtual_workspace_enabled
                    and follow_world_path
                    and coarse_frozen_vb is not None
                )
            )
            if vb_freeze:
                if demo_scenic_analytic_shape:
                    vb_candidate = np.zeros(3, dtype=np.float32)
                elif (
                    virtual_workspace_enabled
                    and demo_xvla_once
                    and demo_xvla_world_pts is not None
                    and demo_scenic_frozen_vb is not None
                ):
                    vb_candidate = demo_scenic_frozen_vb.astype(np.float32).copy()
                else:
                    vb_candidate = coarse_frozen_vb.astype(np.float32).copy()
                virtual_base_world = vb_candidate
                vb_prev_smooth = np.asarray(virtual_base_world, dtype=np.float32).copy()
            elif (
                virtual_workspace_enabled
                and cur_task is None
                and mission_cmd
                and str(mission_cmd).strip()
            ):
                if (
                    cmd_goal_coupled_virtual_base
                    and vb_goal_for_coupling is not None
                    and bool(with_objects)
                    and bool(placed_cubes)
                ):
                    vb_candidate = choose_virtual_base_world(
                        drone_pos,
                        vb_goal_for_coupling,
                        workspace_lo,
                        workspace_hi,
                        virtual_base_world,
                        margin=virtual_workspace_margin,
                    )
                    if not drone_goal_both_inside_workspace_local(
                        drone_pos,
                        vb_goal_for_coupling,
                        vb_candidate,
                        workspace_lo,
                        workspace_hi,
                        margin=1e-5,
                    ):
                        vb_candidate = choose_virtual_base_world(
                            drone_pos,
                            vb_goal_for_coupling,
                            workspace_lo,
                            workspace_hi,
                            np.zeros(3, dtype=np.float32),
                            margin=virtual_workspace_margin,
                        )
                    virtual_base_world = vb_candidate
                else:
                    virtual_base_world = np.zeros(3, dtype=np.float32)
                vb_smooth_applies = (
                    float(vb_smooth_alpha) > 0.0
                    and float(vb_max_speed_m_s) > 0.0
                )
                if vb_smooth_applies:
                    vb_smoothed = smooth_and_cap_virtual_base_step(
                        vb_prev_smooth,
                        virtual_base_world,
                        dt,
                        alpha=float(vb_smooth_alpha),
                        max_speed_m_s=float(vb_max_speed_m_s),
                    )
                    jmp = float(
                        np.linalg.norm(
                            np.asarray(vb_smoothed, dtype=np.float64)
                            - np.asarray(vb_prev_smooth, dtype=np.float64)
                        )
                    )
                    if vb_jump_warn_m > 0.0 and jmp > float(vb_jump_warn_m):
                        print(
                            f"[vb] smoothed virtual-base step {jmp:.3f}m "
                            f"(warn threshold {float(vb_jump_warn_m):.3f}m)"
                        )
                    virtual_base_world = vb_smoothed
                vb_prev_smooth = np.asarray(virtual_base_world, dtype=np.float32).copy()
            elif virtual_workspace_enabled:
                vb_candidate = choose_virtual_base_world(
                    drone_pos,
                    cur_goal_world,
                    workspace_lo,
                    workspace_hi,
                    virtual_base_world,
                    margin=virtual_workspace_margin,
                )
                virtual_base_world = vb_candidate
                vb_smooth_applies = float(vb_smooth_alpha) > 0.0 and float(vb_max_speed_m_s) > 0.0
                if vb_smooth_applies:
                    vb_smoothed = smooth_and_cap_virtual_base_step(
                        vb_prev_smooth,
                        virtual_base_world,
                        dt,
                        alpha=float(vb_smooth_alpha),
                        max_speed_m_s=float(vb_max_speed_m_s),
                    )
                    jmp = float(
                        np.linalg.norm(
                            np.asarray(vb_smoothed, dtype=np.float64)
                            - np.asarray(vb_prev_smooth, dtype=np.float64)
                        )
                    )
                    if vb_jump_warn_m > 0.0 and jmp > float(vb_jump_warn_m):
                        print(
                            f"[vb] smoothed virtual-base step {jmp:.3f}m "
                            f"(warn threshold {float(vb_jump_warn_m):.3f}m)"
                        )
                    virtual_base_world = vb_smoothed
                vb_prev_smooth = np.asarray(virtual_base_world, dtype=np.float32).copy()
            else:
                virtual_base_world = np.zeros(3, dtype=np.float32)
                vb_prev_smooth = np.asarray(virtual_base_world, dtype=np.float32).copy()
            drone_pos_local = np.minimum(
                np.maximum(drone_pos - virtual_base_world, workspace_lo),
                workspace_hi,
            ).astype(np.float32)

            precision_zone = False
            if (
                not phase3_sim_path_follow
                and cmd_precision_zone_enable
                and cur_task is None
                and with_objects
                and placed_cubes
            ):
                for c in placed_cubes:
                    cen = np.asarray(c["pos"], dtype=np.float64).reshape(3)
                    r = _object_sphere_radius(
                        c,
                        scale=float(cmd_precision_zone_scale),
                        inflate=float(cmd_precision_zone_inflate_m),
                        min_r=float(cmd_precision_zone_min_r),
                    )
                    if float(np.linalg.norm(drone_pos.astype(np.float64) - cen)) <= r:
                        precision_zone = True
                        break
            if precision_zone != precision_zone_last:
                print(
                    f"[precision] zone={'enter' if precision_zone else 'exit'} "
                    f"step={t} infer_every={precision_infer_every}"
                )
                precision_zone_last = precision_zone
            if cur_task is not None:
                task_step_count += 1

            inferred = (t % infer_every == 0) or (
                precision_zone and (t % precision_infer_every == 0)
            )
            if phase3_sim_path_follow:
                inferred = False
            if demo_xvla_once and demo_xvla_world_pts is not None:
                u_traj = float(t) / float(max(1, int(sim_steps) - 1))
                target_pos, target_R = _interpolate_pose_along_polyline(
                    demo_xvla_world_pts,
                    demo_xvla_world_Rs or [],
                    u_traj,
                )
                target_pos = _clip_world_pos_to_workspace(
                    target_pos,
                    workspace_lo,
                    workspace_hi,
                    virtual_base_world,
                )
            elif coarse_xvla_world_pts is not None and (
                phase3_sim_path_follow or (follow_world_path and not precision_zone)
            ):
                base_step = int(coarse_xvla_plan_step or 0)
                denom = max(1, int(sim_steps) - base_step - 1)
                u_traj = float(np.clip((t - base_step) / float(denom), 0.0, 1.0))
                target_pos, target_R = _interpolate_pose_along_polyline(
                    coarse_xvla_world_pts,
                    coarse_xvla_world_Rs or [],
                    u_traj,
                )
                target_pos = _clip_world_pos_to_workspace(
                    target_pos,
                    workspace_lo,
                    workspace_hi,
                    virtual_base_world,
                )
                target_pos_local = np.minimum(
                    np.maximum(target_pos - virtual_base_world, workspace_lo),
                    workspace_hi,
                ).astype(np.float32)
            elif inferred and not phase3_sim_path_follow:
                if coarse_plan_enabled and coarse_xvla_world_pts is None and not precision_zone:
                    plan_steps = max(int(xvla_steps), int(cmd_coarse_plan_steps))
                    img = get_workspace_rgb(
                        p,
                        cam_eye + virtual_base_world,
                        cam_look + virtual_base_world,
                        width=int(workspace_camera_width),
                        height=int(workspace_camera_height),
                        fov=float(workspace_camera_fov),
                    )
                    proprio = build_proprio_widowx_ee6d(
                        drone_pos_local, drone_R, gripper=gripper_state
                    )
                    t_req = time.perf_counter()
                    actions = query_xvla(
                        server_url,
                        img,
                        proprio,
                        _instruction_for_xvla_call(cur_instruction),
                        steps=int(plan_steps),
                        timeout=float(xvla_act_request_timeout_s),
                    )
                    last_infer_ms = (time.perf_counter() - t_req) * 1000.0
                    infer_calls += 1
                    loc_pts, loc_Rs = decode_action_chunk_to_local_waypoints(
                        actions,
                        workspace_lo=workspace_lo,
                        workspace_hi=workspace_hi,
                        treat_pos_as=treat_pos_as,
                        start_pos_local=drone_pos_local.copy(),
                        delta_pos_scale=float(delta_pos_scale),
                    )
                    a0 = np.asarray(actions, dtype=np.float32)
                    if a0.ndim == 1:
                        a0 = a0.reshape(1, -1)
                    g_logit = float(a0[-1, 9])
                    gripper_state = float(1.0 / (1.0 + np.exp(-g_logit)))
                    W = (loc_pts + virtual_base_world.astype(np.float32).reshape(1, 3)).astype(np.float32)
                    wlist = [W[i] for i in range(W.shape[0])]
                    if wlist:
                        d0 = float(
                            np.linalg.norm(
                                np.asarray(wlist[0], dtype=np.float64)
                                - drone_pos.astype(np.float64)
                            )
                        )
                        if d0 > 0.02:
                            wlist.insert(0, drone_pos.copy().astype(np.float32))
                            loc_Rs.insert(0, drone_R.copy().astype(np.float32))
                    coarse_xvla_world_pts = np.stack(wlist, axis=0) if wlist else None
                    coarse_xvla_world_Rs = loc_Rs
                    coarse_xvla_plan_step = int(t)
                    coarse_frozen_vb = virtual_base_world.astype(np.float32).copy()
                    if coarse_xvla_world_pts is not None:
                        k = max(1, int(cmd_coarse_plan_smooth_window))
                        if k >= 3 and coarse_xvla_world_pts.shape[0] > k:
                            pad = k // 2
                            P = np.pad(coarse_xvla_world_pts, ((pad, pad), (0, 0)), mode="edge")
                            out = []
                            for i in range(coarse_xvla_world_pts.shape[0]):
                                out.append(np.mean(P[i : i + k], axis=0))
                            coarse_xvla_world_pts = np.asarray(out, dtype=np.float32)
                    if not coarse_plan_log_once and coarse_xvla_world_pts is not None:
                        print(
                            "[cmd] coarse plan: one-shot X-VLA trajectory planned "
                            f"steps={plan_steps} waypoints={coarse_xvla_world_pts.shape[0]} "
                            f"smooth_w={int(cmd_coarse_plan_smooth_window)} infer_ms={last_infer_ms:.1f}"
                        )
                        coarse_plan_log_once = True
                    if coarse_xvla_world_pts is not None:
                        u_traj = float(t) / float(max(1, int(sim_steps) - 1))
                        target_pos, target_R = _interpolate_pose_along_polyline(
                            coarse_xvla_world_pts,
                            coarse_xvla_world_Rs or [],
                            u_traj,
                        )
                        target_pos = _clip_world_pos_to_workspace(
                            target_pos,
                            workspace_lo,
                            workspace_hi,
                            virtual_base_world,
                        )
                        target_pos_local = np.minimum(
                            np.maximum(target_pos - virtual_base_world, workspace_lo),
                            workspace_hi,
                        ).astype(np.float32)
                elif demo_xvla_once and demo_xvla_world_pts is None:
                    plan_steps = max(int(xvla_steps), int(language_only_demo_traj_plan_steps))
                    vb_plan = virtual_base_world.astype(np.float32).copy()
                    if bool(language_only_demo_plan_use_topdown_camera):
                        img = get_world_recording_rgb(
                            p,
                            workspace_lo + vb_plan,
                            workspace_hi + vb_plan,
                            view_kind="top",
                            width=int(workspace_camera_width),
                            height=int(workspace_camera_height),
                            fov=float(recording_scene_fov),
                            margin=float(recording_scene_margin),
                            distance_scale=float(recording_camera_distance_scale),
                            top_view_distance_scale=float(recording_topview_distance_scale),
                        )
                        topdown_bbox = estimate_topdown_objects_xy_aabb_from_rgb(
                            img,
                            workspace_lo,
                            workspace_hi,
                            width=int(workspace_camera_width),
                            height=int(workspace_camera_height),
                            fov=float(recording_scene_fov),
                            margin=float(recording_scene_margin),
                            distance_scale=float(recording_camera_distance_scale),
                            virtual_base_world=vb_plan,
                            top_view_distance_scale=float(recording_topview_distance_scale),
                            min_area_ratio=float(cmd_global_cam_min_area_ratio),
                        )
                        if topdown_bbox is not None:
                            demo_topdown_bbox, demo_topdown_bbox_frac = topdown_bbox[0:2], topdown_bbox[2]
                            print(
                                "[demo] topdown RGB bbox: "
                                f"mask_frac={demo_topdown_bbox_frac:.4f} "
                                f"lo={demo_topdown_bbox[0].round(3).tolist()} "
                                f"hi={demo_topdown_bbox[1].round(3).tolist()}"
                            )
                    else:
                        img = get_workspace_rgb(
                            p,
                            cam_eye + virtual_base_world,
                            cam_look + virtual_base_world,
                            width=int(workspace_camera_width),
                            height=int(workspace_camera_height),
                            fov=float(workspace_camera_fov),
                        )
                    proprio = build_proprio_widowx_ee6d(
                        drone_pos_local, drone_R, gripper=gripper_state
                    )
                    t_req = time.perf_counter()
                    actions = query_xvla(
                        server_url,
                        img,
                        proprio,
                        _instruction_for_xvla_call(cur_instruction),
                        steps=int(plan_steps),
                        timeout=float(xvla_act_request_timeout_s),
                    )
                    a_np = np.asarray(actions, dtype=np.float32)
                    n_act_rows = int(a_np.shape[0]) if a_np.ndim >= 1 else 1
                    last_infer_ms = (time.perf_counter() - t_req) * 1000.0
                    infer_calls += 1
                    loc_pts, loc_Rs = decode_action_chunk_to_local_waypoints(
                        actions,
                        workspace_lo=workspace_lo,
                        workspace_hi=workspace_hi,
                        treat_pos_as=treat_pos_as,
                        start_pos_local=drone_pos_local.copy(),
                        delta_pos_scale=float(delta_pos_scale),
                    )
                    z_floor_w = float(workspace_lo[2] + vb_plan[2])
                    if with_objects and placed_cubes:
                        z_top_max_world, Hmax = _scene_object_max_height_above_floor(
                            placed_cubes,
                            z_floor_w,
                        )
                    else:
                        z_top_max_world = float(z_floor_w)
                        Hmax = max(
                            float(workspace_hi[2] - workspace_lo[2]) * 0.35,
                            0.05,
                        )
                    z_floor_safe = float(z_floor_w + 0.01)
                    frac_h = float(
                        np.clip(
                            float(language_only_demo_clearance_frac_above_tallest), 0.0, 3.0
                        )
                    )
                    clear_z = float(
                        np.clip(float(language_only_demo_clearance_above_scene_z_m), 0.0, 2.0)
                    )
                    margin_z = float(frac_h * float(Hmax) + clear_z)
                    z_target_clear = float(z_top_max_world + margin_z)
                    workspace_hi = _expand_workspace_hi_local_for_world_z(
                        workspace_hi,
                        z_target_world=float(z_target_clear),
                        virtual_base_z=float(vb_plan[2]),
                    )
                    z_ceil_world = float(workspace_hi[2]) + float(vb_plan[2]) - 0.005
                    z_min_world = float(
                        np.clip(z_target_clear, z_floor_safe, z_ceil_world - 0.01)
                    )
                    if z_target_clear > z_ceil_world - 0.01 + 1e-6:
                        if not demo_scenic_z_cap_warned:
                            print(
                                "[demo] scenic Z: cruise world Z still above ceiling "
                                f"after hull expand (z_target≈{z_target_clear:.3f}m cap≈{z_ceil_world:.3f}m); "
                                "clamping to ceiling."
                            )
                            demo_scenic_z_cap_warned = True
                        z_min_world = float(
                            np.clip(z_min_world, z_floor_safe, z_ceil_world - 0.01)
                        )

                    wlist: list[np.ndarray] = []
                    rlist: list[np.ndarray] = []
                    for k in range(int(loc_pts.shape[0])):
                        w = vb_plan.astype(np.float64) + loc_pts[k].astype(np.float64)
                        if demo_helical_z:
                            w[2] = float(
                                np.clip(float(w[2]), z_floor_safe, float(z_ceil_world))
                            )
                        else:
                            w[2] = float(
                                np.clip(
                                    float(w[2]),
                                    float(z_min_world),
                                    float(z_ceil_world),
                                )
                            )
                        wf = _clip_world_pos_to_workspace(
                            w.astype(np.float32),
                            workspace_lo,
                            workspace_hi,
                            vb_plan,
                        )
                        wlist.append(wf)
                        rlist.append(loc_Rs[k])
                    W = np.stack(wlist, axis=0)
                    n_xy = int(W.shape[0])
                    margin_xy = float(np.clip(language_only_demo_fill_xy_margin_m, 0.0, 0.25))
                    w_lo_w = workspace_lo.astype(np.float64) + vb_plan.astype(np.float64)
                    w_hi_w = workspace_hi.astype(np.float64) + vb_plan.astype(np.float64)
                    t_lo2 = w_lo_w[0:2] + margin_xy
                    t_hi2 = w_hi_w[0:2] - margin_xy
                    fill_tag = "workspace_xy"
                    obj_bbox: tuple[np.ndarray, np.ndarray] | None = None
                    if with_objects and placed_cubes:
                        oxy = _scene_objects_xy_aabb_world(placed_cubes)
                        if oxy is not None:
                            lo_o, hi_o = oxy
                            lo_o = lo_o + margin_xy
                            hi_o = hi_o - margin_xy
                            obj_bbox = (lo_o, hi_o)
                    if demo_topdown_bbox is not None and obj_bbox is not None:
                        lo_a, hi_a = demo_topdown_bbox
                        lo_b, hi_b = obj_bbox
                        lo_u = np.minimum(lo_a, lo_b)
                        hi_u = np.maximum(hi_a, hi_b)
                        t_lo2 = np.maximum(lo_u + margin_xy, w_lo_w[0:2] + margin_xy)
                        t_hi2 = np.minimum(hi_u - margin_xy, w_hi_w[0:2] - margin_xy)
                        fill_tag = "topdown_rgb∪objects"
                    elif demo_topdown_bbox is not None:
                        lo_o, hi_o = demo_topdown_bbox
                        t_lo2 = np.maximum(lo_o + margin_xy, w_lo_w[0:2] + margin_xy)
                        t_hi2 = np.minimum(hi_o - margin_xy, w_hi_w[0:2] - margin_xy)
                        fill_tag = "topdown_rgb"
                    elif obj_bbox is not None:
                        lo_o, hi_o = obj_bbox
                        t_lo2 = np.maximum(lo_o, w_lo_w[0:2] + margin_xy)
                        t_hi2 = np.minimum(hi_o, w_hi_w[0:2] - margin_xy)
                        fill_tag = "object_xy∩workspace"
                    if demo_scenic and not demo_strophoid:
                        t_lo2, t_hi2, fill_tag = _scenic_force_square_xy_in_workspace_footprint(
                            w_lo_w,
                            w_hi_w,
                            margin_xy,
                            prior_fill_tag=fill_tag,
                            long_side_scale=float(
                                language_only_demo_scenic_xy_force_long_scale
                            ),
                        )

                    if float(np.min(t_hi2 - t_lo2)) > 1e-4:
                        diag_before = float(
                            np.linalg.norm(np.ptp(W[:, 0:2], axis=0))
                        )
                        scenic_n = max(
                            int(W.shape[0]),
                            int(language_only_demo_traj_plan_steps),
                            int(language_only_demo_scenic_formula_min_waypoints),
                        )
                        n_xy = int(scenic_n if demo_scenic_analytic_shape else W.shape[0])
                        if demo_scenic_analytic_shape:
                            rlist = _expand_rotation_keyframes_to_n(rlist, n_xy)
                            W = np.zeros((n_xy, 3), dtype=np.float32)
                            W[:, 2] = float(z_min_world)
                        if demo_fig8:
                            fig8_xy = _plan_figure8_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(fig8_xy, int(n_xy))
                            fill_tag = f"figure8@{fill_tag}"
                        elif demo_oval:
                            ell_xy = _plan_ellipse_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ell_xy, int(n_xy))
                            fill_tag = f"ellipse@{fill_tag}"
                        elif demo_circle_orbit:
                            circ_xy = _plan_circle_orbit_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(circ_xy, int(n_xy))
                            fill_tag = f"circle_orbit@{fill_tag}"
                        elif demo_diamond:
                            d_xy = _plan_diamond_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(d_xy, int(n_xy))
                            fill_tag = f"diamond@{fill_tag}"
                        elif demo_square:
                            sq_xy = _plan_square_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(sq_xy, int(n_xy))
                            fill_tag = f"square@{fill_tag}"
                        elif demo_phyllotaxis:
                            ph_xy = _plan_phyllotaxis_disk_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ph_xy, int(n_xy))
                            fill_tag = f"phyllotaxis@{fill_tag}"
                        elif demo_arch_spiral:
                            arch_xy = _plan_archimedean_spiral_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(arch_xy, int(n_xy))
                            fill_tag = f"archimedean_spiral@{fill_tag}"
                        elif demo_spiral:
                            sp_xy = _plan_expand_contract_spiral_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(sp_xy, int(n_xy))
                            fill_tag = f"spiral@{fill_tag}"
                        elif demo_serpentine:
                            serp_xy = _plan_serpentine_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(serp_xy, int(n_xy))
                            fill_tag = f"serpentine@{fill_tag}"
                        elif demo_cross_shuttle:
                            crs_xy = _plan_cross_axis_shuttle_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(crs_xy, int(n_xy))
                            fill_tag = f"cross_shuttle@{fill_tag}"
                        elif demo_shuttle:
                            sh_xy = _plan_shuttle_line_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(sh_xy, int(n_xy))
                            fill_tag = f"shuttle_line@{fill_tag}"
                        elif demo_clover:
                            cl_xy = _plan_clover_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(cl_xy, int(n_xy))
                            fill_tag = f"clover@{fill_tag}"
                        elif demo_star:
                            st_xy = _plan_pentagram_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(st_xy, int(n_xy))
                            fill_tag = f"star@{fill_tag}"
                        elif demo_lissajous:
                            lj_xy = _plan_lissajous_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(lj_xy, int(n_xy))
                            fill_tag = f"lissajous@{fill_tag}"
                        elif demo_heart:
                            ht_xy = _plan_heart_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ht_xy, int(n_xy))
                            fill_tag = f"heart@{fill_tag}"
                        elif demo_teardrop:
                            td_xy = _plan_teardrop_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(td_xy, int(n_xy))
                            fill_tag = f"teardrop@{fill_tag}"
                        elif demo_polygon:
                            _nsides = scenic_polygon_sides if scenic_polygon_sides is not None else 6
                            pg_xy = _plan_regular_n_gon_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                sides=int(_nsides),
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(pg_xy, int(n_xy))
                            fill_tag = f"polygon{_nsides}@{fill_tag}"
                        elif demo_rose:
                            _rk = scenic_rose_petals if scenic_rose_petals is not None else 5
                            rs_xy = _plan_rose_petal_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                petals_k=int(_rk),
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(rs_xy, int(n_xy))
                            fill_tag = f"rose{_rk}@{fill_tag}"
                        elif demo_damped_sinewave:
                            ds_xy = _plan_damped_sine_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ds_xy, int(n_xy))
                            fill_tag = f"damped_sine@{fill_tag}"
                        elif demo_trig_beat_wave:
                            bt_xy = _plan_trig_beat_wave_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(bt_xy, int(n_xy))
                            fill_tag = f"trig_beat@{fill_tag}"
                        elif demo_triangle_wave:
                            tw_xy = _plan_triangle_wave_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(tw_xy, int(n_xy))
                            fill_tag = f"triangle_wave@{fill_tag}"
                        elif demo_tanh_ribbon:
                            th_xy = _plan_tanh_ribbon_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(th_xy, int(n_xy))
                            fill_tag = f"tanh_ribbon@{fill_tag}"
                        elif demo_cosine_wave:
                            cos_xy = _plan_cosine_wave_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(cos_xy, int(n_xy))
                            fill_tag = f"cosine_wave@{fill_tag}"
                        elif demo_sinewave:
                            sin_xy = _plan_sinewave_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(sin_xy, int(n_xy))
                            fill_tag = f"sinewave@{fill_tag}"
                        elif demo_stadium_capsule:
                            st_xy = _plan_stadium_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(st_xy, int(n_xy))
                            fill_tag = f"stadium@{fill_tag}"
                        elif demo_cycloid_arc:
                            cy_xy = _plan_cycloid_one_arch_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(cy_xy, int(n_xy))
                            fill_tag = f"cycloid@{fill_tag}"
                        elif demo_cardioid:
                            cd_xy = _plan_cardioid_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(cd_xy, int(n_xy))
                            fill_tag = f"cardioid@{fill_tag}"
                        elif demo_limacon:
                            lm_xy = _plan_limacon_pascal_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(lm_xy, int(n_xy))
                            fill_tag = f"limacon@{fill_tag}"
                        elif demo_deltoid:
                            dl_xy = _plan_deltoid_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(dl_xy, int(n_xy))
                            fill_tag = f"deltoid@{fill_tag}"
                        elif demo_astroid:
                            as_xy = _plan_astroid_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(as_xy, int(n_xy))
                            fill_tag = f"astroid@{fill_tag}"
                        elif demo_epitrochoid:
                            ep2_xy = _plan_epitrochoid_default_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ep2_xy, int(n_xy))
                            fill_tag = f"epitrochoid@{fill_tag}"
                        elif demo_nephroid:
                            neph_xy = _plan_nephroid_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(neph_xy, int(n_xy))
                            fill_tag = f"nephroid@{fill_tag}"
                        elif demo_epicycloid:
                            ep_xy = _plan_epicycloid_default_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ep_xy, int(n_xy))
                            fill_tag = f"epicycloid@{fill_tag}"
                        elif demo_logspiral:
                            ls_xy = _plan_logarithmic_spiral_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ls_xy, int(n_xy))
                            fill_tag = f"log_spiral@{fill_tag}"
                        elif demo_hyp_spiral:
                            hs_xy = _plan_hyperbolic_spiral_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(hs_xy, int(n_xy))
                            fill_tag = f"hyp_spiral@{fill_tag}"
                        elif demo_involute:
                            iv_xy = _plan_involute_circle_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(iv_xy, int(n_xy))
                            fill_tag = f"involute@{fill_tag}"
                        elif demo_squircle:
                            sq_xy2 = _plan_superellipse_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(sq_xy2, int(n_xy))
                            fill_tag = f"squircle@{fill_tag}"
                        elif demo_butterfly:
                            bf_xy = _plan_butterfly_rice_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(bf_xy, int(n_xy))
                            fill_tag = f"butterfly@{fill_tag}"
                        elif demo_gaussian:
                            gz_xy = _plan_gaussian_track_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(gz_xy, int(n_xy))
                            fill_tag = f"gaussian_bump@{fill_tag}"
                        elif demo_arc_chain:
                            ac_xy = _plan_arc_chain_wave_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ac_xy, int(n_xy))
                            fill_tag = f"catenary_scallop@{fill_tag}"
                        elif demo_bern_lemn:
                            bl_xy = _plan_bernoulli_lemniscate_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(bl_xy, int(n_xy))
                            fill_tag = f"lemniscate_bern@{fill_tag}"
                        elif demo_cochleoid:
                            ck_xy = _plan_cochleoid_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ck_xy, int(n_xy))
                            fill_tag = f"cochleoid@{fill_tag}"
                        elif demo_folium:
                            fm_xy = _plan_steiner_folium_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(fm_xy, int(n_xy))
                            fill_tag = f"folium@{fill_tag}"
                        elif demo_clothoid:
                            cl_xy = _plan_clothoid_segment_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(cl_xy, int(n_xy))
                            fill_tag = f"clothoid@{fill_tag}"
                        elif demo_tractrix:
                            tx_xy = _plan_tractrix_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(tx_xy, int(n_xy))
                            fill_tag = f"tractrix@{fill_tag}"
                        elif demo_witch:
                            wc_xy = _plan_witch_of_agnesi_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(wc_xy, int(n_xy))
                            fill_tag = f"witch_agnesi@{fill_tag}"
                        elif demo_cassini:
                            cs_xy = _plan_cassini_oval_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(cs_xy, int(n_xy))
                            fill_tag = f"cassini@{fill_tag}"
                        elif demo_hypotroch:
                            hy_xy = _plan_hypotrochoid_default_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(hy_xy, int(n_xy))
                            fill_tag = f"hypotrochoid@{fill_tag}"
                        elif demo_strophoid:
                            st_xy = _plan_strophoid_right_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(st_xy, int(n_xy))
                            fill_tag = f"strophoid@{fill_tag}"
                        elif demo_kampyle:
                            ky_xy = _plan_kampyle_eudoxus_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ky_xy, int(n_xy))
                            fill_tag = f"kampyle@{fill_tag}"
                        elif demo_conchoid_nic:
                            cn_xy = _plan_conchoid_nicomedes_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(cn_xy, int(n_xy))
                            fill_tag = f"conchoid@{fill_tag}"
                        elif demo_cissoid:
                            ci_xy = _plan_cissoid_diocles_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(ci_xy, int(n_xy))
                            fill_tag = f"cissoid@{fill_tag}"
                        elif demo_parabola:
                            pb_xy = _plan_parabolic_arc_xy_polyline(
                                t_lo2,
                                t_hi2,
                                n_xy,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                            W[:, 0:2] = _fit_scenic_xy_polyline_rows(pb_xy, int(n_xy))
                            fill_tag = f"parabolic_arc@{fill_tag}"
                        else:
                            W = _remap_polyline_xy_fill_rect(W, t_lo2, t_hi2)
                        if demo_helical_z:
                            z_span_w = max(float(z_ceil_world - z_min_world), 0.0)
                            z_hi_wave = float(z_ceil_world - max(0.005, 0.02 * min(z_span_w, 1.0)))
                            if z_hi_wave <= z_min_world + 1e-4:
                                z_hi_wave = float(z_ceil_world - 0.005)
                            W = _apply_scenic_helical_z_world(
                                W,
                                z_lo_world=float(z_min_world),
                                z_hi_world=float(z_hi_wave),
                                n_turns=float(scenic_helical_turns),
                                z_trig=str(scenic_z_trig),
                            )
                            fill_tag = (
                                f"helical_z:{float(scenic_helical_turns):.2f}t:"
                                f"{str(scenic_z_trig)}@{fill_tag}"
                            )
                        diag_tgt = float(np.linalg.norm(t_hi2 - t_lo2))
                        for ii in range(W.shape[0]):
                            W[ii] = _clip_world_pos_to_workspace(
                                W[ii],
                                workspace_lo,
                                workspace_hi,
                                vb_plan,
                            )
                        wlist = [W[i] for i in range(W.shape[0])]
                        print(
                            "[demo] scenic XY fill: "
                            f"{fill_tag} rect_diag≈{diag_tgt:.3f}m "
                            f"poly_xy_ptp_before={diag_before:.3f}m"
                        )
                    else:
                        wlist = [W[i] for i in range(W.shape[0])]
                        print(
                            "[demo] scenic XY fill: skip (degenerate target rect)",
                            file=sys.stderr,
                        )
                    d0 = float(
                        np.linalg.norm(
                            np.asarray(wlist[0], dtype=np.float64)
                            - drone_pos.astype(np.float64)
                        )
                    )
                    if d0 > 0.02:
                        wlist.insert(0, drone_pos.copy().astype(np.float32))
                        rlist.insert(0, drone_R.copy().astype(np.float32))
                    demo_xvla_world_pts = np.stack(wlist, axis=0)
                    demo_xvla_world_Rs = rlist
                    demo_xvla_plan_step = int(t)
                    demo_scenic_frozen_vb = vb_plan.astype(np.float32).copy()
                    if demo_xvla_world_pts.shape[0] >= 2:
                        dif = np.diff(demo_xvla_world_pts, axis=0)
                        total_len = float(np.sum(np.linalg.norm(dif, axis=1)))
                        per_step = total_len / float(max(1, int(sim_steps) - 1))
                        demo_trail_min_len = min(
                            float(trail_min_distance),
                            max(0.002, per_step * 0.8),
                        )
                    a0 = np.asarray(actions, dtype=np.float32)
                    if a0.ndim == 1:
                        a0 = a0.reshape(1, -1)
                    g_logit = float(a0[-1, 9])
                    gripper_state = float(1.0 / (1.0 + np.exp(-g_logit)))
                    print(
                        "[demo] X-VLA one-shot trajectory planned: "
                        f"requested_steps={plan_steps} action_rows={n_act_rows} "
                        f"waypoints={demo_xvla_world_pts.shape[0]} "
                        f"z_scene_top={z_top_max_world:.3f} z_cruise={z_min_world:.3f} "
                        f"formula_pts={int(n_xy)} infer_ms={last_infer_ms:.1f} "
                        f"plan_cam={'topdown' if language_only_demo_plan_use_topdown_camera else 'workspace'} "
                        f"frozen_vb={demo_scenic_frozen_vb.round(3).tolist()}"
                    )
                    u_traj = float(t) / float(max(1, int(sim_steps) - 1))
                    target_pos, target_R = _interpolate_pose_along_polyline(
                        demo_xvla_world_pts,
                        demo_xvla_world_Rs,
                        u_traj,
                    )
                    target_pos = _clip_world_pos_to_workspace(
                        target_pos,
                        workspace_lo,
                        workspace_hi,
                        virtual_base_world,
                    )
                else:
                    img = get_workspace_rgb(
                        p,
                        cam_eye + virtual_base_world,
                        cam_look + virtual_base_world,
                        width=int(workspace_camera_width),
                        height=int(workspace_camera_height),
                        fov=float(workspace_camera_fov),
                    )
                    proprio = build_proprio_widowx_ee6d(
                        drone_pos_local, drone_R, gripper=gripper_state
                    )
                    t_req = time.perf_counter()
                    actions = query_xvla(
                        server_url,
                        img,
                        proprio,
                        _instruction_for_xvla_call(cur_instruction),
                        steps=xvla_steps,
                        timeout=float(xvla_act_request_timeout_s),
                    )
                    last_infer_ms = (time.perf_counter() - t_req) * 1000.0
                    infer_calls += 1
                    target_pos_local, target_R, g = decode_action_widowx_ee6d(
                        actions,
                        workspace_lo=workspace_lo,
                        workspace_hi=workspace_hi,
                        treat_pos_as=treat_pos_as,
                        current_pos=drone_pos_local.copy(),
                        current_R=drone_R.copy(),
                        delta_pos_scale=delta_pos_scale,
                    )
                    gripper_state = float(g)

                    if (
                        cur_task is None
                        and mission_cmd
                        and str(mission_cmd).strip()
                        and cmd_use_global_camera_target
                        and with_objects
                        and placed_cubes
                        and float(cmd_global_cam_pull_alpha) > 0.0
                    ):
                        g_world, frac, how = resolve_cmd_target_from_global_workspace_camera(
                            p,
                            cur_instruction,
                            placed_cubes,
                            cam_eye_world=np.asarray(
                                cam_eye + virtual_base_world, dtype=np.float64
                            ),
                            cam_look_world=np.asarray(
                                cam_look + virtual_base_world, dtype=np.float64
                            ),
                            cam_width=int(workspace_camera_width),
                            cam_height=int(workspace_camera_height),
                            cam_fov_deg=float(workspace_camera_fov),
                            rgb_pad=int(gate_pose_cv_rgb_pad),
                            min_area_ratio=float(cmd_global_cam_min_area_ratio),
                            prefer_near_xyz=np.asarray(drone_pos, dtype=np.float64),
                        )
                        if g_world is not None:
                            pull_g = float(np.clip(cmd_global_cam_pull_alpha, 0.0, 1.0))
                            weak_mask = float(frac) < float(cmd_global_cam_min_area_ratio)
                            if weak_mask:
                                pull_g = float(pull_g) * float(cmd_global_cam_weak_mask_scale)

                                _cmd_ref = str(mission_cmd or cur_instruction or "").lower()
                                _portal_cmd = bool(
                                    re.search(r"\b(pass|fly|go)\s+through\b", _cmd_ref)
                                    or re.search(r"\b(rectangular|portal|frame|opening|slot|gate)\b", _cmd_ref)
                                )
                                if _portal_cmd and cmd_global_cam_pull_alpha > 0.0:
                                    pull_g = max(float(pull_g), float(cmd_global_cam_portal_min_pull))

                            if pull_g > 0.0 and weak_mask:
                                pull_g = max(float(pull_g), float(cmd_global_cam_weak_mask_min_pull))
                            if (
                                language_only_air_path_max_cam_pull is not None
                                and mission_cmd
                                and str(mission_cmd).strip()
                                and _language_cmd_suggests_air_path_or_cruise(str(mission_cmd))
                            ):
                                pull_g = min(
                                    pull_g,
                                    max(0.0, float(language_only_air_path_max_cam_pull)),
                                )
                            goal_loc = np.minimum(
                                np.maximum(
                                    g_world.astype(np.float64)
                                    - virtual_base_world.astype(np.float64),
                                    workspace_lo.astype(np.float64),
                                ),
                                workspace_hi.astype(np.float64),
                            ).astype(np.float32)
                            target_pos_local = (
                                (1.0 - pull_g) * target_pos_local.astype(np.float64)
                                + pull_g * goal_loc.astype(np.float64)
                            ).astype(np.float32)
                            target_pos_local = np.minimum(
                                np.maximum(target_pos_local, workspace_lo), workspace_hi
                            ).astype(np.float32)
                            if not gcam_target_log_once:
                                print(
                                    f"[cmd] global-camera fuse: {how} mask_frac={frac:.4f} pull={pull_g:.3f} "
                                    f"goal_world={np.asarray(g_world).round(3).tolist()}"
                                )

                                def _portal_leg_exit_goal(portal_cube: dict, from_xyz: np.ndarray) -> np.ndarray:
                                    spec = _portal_pass_spec_for_task(
                                        p,
                                        placed_cubes,
                                        np.asarray(portal_cube["pos"], dtype=np.float64),
                                        np.asarray(from_xyz, dtype=np.float64),
                                        match_tol=float(portal_match_tol),
                                        approach_offset=float(portal_pass_approach_offset),
                                        exit_offset=float(portal_pass_exit_offset),
                                    )
                                    if spec is not None:
                                        return np.asarray(spec["exit"], dtype=np.float64)
                                    return np.asarray(portal_cube["pos"], dtype=np.float64).reshape(3)

                                _plan_gcam = run_navigation_phase1_and_phase2_topdown(
                                    p=p,
                                    placed_cubes=placed_cubes,
                                    mission_cmd=mission_cmd,
                                    cur_instruction=cur_instruction,
                                    g_world=g_world,
                                    drone_pos=drone_pos,
                                    drone_pos_local=drone_pos_local,
                                    drone_R=drone_R,
                                    virtual_base_world=virtual_base_world,
                                    workspace_lo=workspace_lo,
                                    workspace_hi=workspace_hi,
                                    treat_pos_as=treat_pos_as,
                                    delta_pos_scale=delta_pos_scale,
                                    gripper_state=gripper_state,
                                    server_url=server_url,
                                    scene_catalog_str=scene_catalog_str,
                                    xvla_scene_semantic_context=bool(xvla_scene_semantic_context),
                                    xvla_path_planning_instruction_suffix=xvla_path_planning_instruction_suffix,
                                    workspace_camera_width=int(workspace_camera_width),
                                    workspace_camera_height=int(workspace_camera_height),
                                    navigation_phase2_xvla_steps=int(navigation_phase2_xvla_steps),
                                    xvla_act_request_timeout_s=float(xvla_act_request_timeout_s),
                                    navigation_phase2_sync_root_config=bool(navigation_phase2_sync_root_config),
                                    navigation_phase2_sync_qs=bool(navigation_phase2_sync_qs),
                                    qs_policy_path_for_sync=qs_policy_path_for_sync,
                                    config_json_path=CONFIG_DEFAULT_PATH,
                                    navigation_phase2_extra_instruction=str(navigation_phase2_extra_instruction),
                                    navigation_phase2_geom_astar=bool(navigation_phase2_geom_astar),
                                    navigation_phase2_astar_cell_m=float(navigation_phase2_astar_cell_m),
                                    navigation_phase1_corridor_margin_m=float(navigation_phase1_corridor_margin_m),
                                    navigation_phase1_corridor_bandwidth_m=float(navigation_phase1_corridor_bandwidth_m),
                                    navigation_collision_pad_m=navigation_collision_pad_m,
                                    navigation_phase2_astar_obstacle_pad_m=float(navigation_phase2_astar_obstacle_pad_m),
                                    navigation_phase2_optional_topdown_xvla=bool(navigation_phase2_optional_topdown_xvla),
                                    navigation_phase2_z_clearance_enabled=bool(navigation_phase2_z_clearance_enabled),
                                    navigation_phase2_z_clearance_margin_m=float(navigation_phase2_z_clearance_margin_m),
                                    navigation_phase2_z_workspace_margin_m=float(navigation_phase2_z_workspace_margin_m),
                                    recording_scene_fov=float(recording_scene_fov),
                                    recording_scene_margin=float(recording_scene_margin),
                                    recording_camera_distance_scale=float(recording_camera_distance_scale),
                                    recording_topview_distance_scale=float(recording_topview_distance_scale),
                                    recording_stereo45_distance_scale=float(recording_stereo45_distance_scale),
                                    rec_folder=rec_folder,
                                    recording_folder=recording_folder,
                                    root_dir=ROOT,
                                    world_recording_view_proj_fn=_world_recording_view_proj,
                                    render_world_recording_rgb_fn=_render_world_recording_rgb,
                                    world_xyz_to_recording_image_pixel_fn=world_xyz_to_recording_image_pixel,
                                    first_rect_portal_for_instruction_color_fn=_first_rect_portal_for_instruction_color,
                                    build_proprio_fn=build_proprio_widowx_ee6d,
                                    query_xvla_fn=query_xvla,
                                    compose_instruction_fn=compose_xvla_navigation_instruction,
                                    read_config_fn=read_config_json,
                                    load_qs_policies_fn=load_qs_policies,
                                    portal_leg_goal_fn=_portal_leg_exit_goal,
                                    create_feedback_spheres=True,
                                )
                                if _plan_gcam is not None:
                                    phase3_feedback = _plan_gcam.get("phase3_zones")
                                gcam_target_log_once = True

                    if cur_task is not None and (
                        target_pull_alpha > 0.0 or target_snap_radius > 0.0
                    ):
                        assert nav_goal_xyz is not None
                        goal_xyz = nav_goal_xyz
                        goal_local = np.minimum(
                            np.maximum(goal_xyz - virtual_base_world, workspace_lo),
                            workspace_hi,
                        ).astype(np.float32)
                        pull = float(np.clip(target_pull_alpha, 0.0, 1.0))
                        if pull > 0.0:
                            target_pos_local = (
                                (1.0 - pull) * target_pos_local + pull * goal_local
                            ).astype(np.float32)
                        if target_snap_radius > 0.0:
                            if float(np.linalg.norm(drone_pos - goal_xyz)) <= float(
                                target_snap_radius
                            ):
                                target_pos_local = goal_local.astype(np.float32)
                        target_pos_local = np.minimum(
                            np.maximum(target_pos_local, workspace_lo), workspace_hi
                        ).astype(np.float32)
                    if cur_task is None and (
                        float(language_only_motion_amplify) != 1.0
                        or float(language_only_min_step_local_m) > 0.0
                    ):
                        target_pos_local = boost_language_only_local_target(
                            drone_pos_local,
                            target_pos_local,
                            workspace_lo,
                            workspace_hi,
                            amplify=float(language_only_motion_amplify),
                            min_step_m=float(language_only_min_step_local_m),
                        )
                    if float(infer_displacement_scale) != 1.0:
                        target_pos_local = scale_infer_local_displacement(
                            drone_pos_local,
                            target_pos_local,
                            workspace_lo,
                            workspace_hi,
                            scale=float(infer_displacement_scale),
                        )
                    if (
                        cur_task is None
                        and bool(language_only_cruise_z_clamp)
                        and mission_cmd
                        and str(mission_cmd).strip()
                        and _language_cmd_suggests_air_path_or_cruise(str(mission_cmd))
                    ):
                        target_pos_local = _apply_language_only_cruise_z_to_local_target(
                            target_pos_local,
                            workspace_lo,
                            workspace_hi,
                            margin_lo_frac=float(language_only_cruise_z_margin_lo_frac),
                            margin_hi_frac=float(language_only_cruise_z_margin_hi_frac),
                        )
                    if demo_scenic and (not bool(language_only_demo_xvla_trajectory_once)):
                        z_floor_w = float(workspace_lo[2] + virtual_base_world[2])
                        if with_objects and placed_cubes:
                            z_top_max_world, Hmax = _scene_object_max_height_above_floor(
                                placed_cubes,
                                z_floor_w,
                            )
                        else:
                            z_top_max_world = float(z_floor_w)
                            Hmax = max(
                                float(workspace_hi[2] - workspace_lo[2]) * 0.35,
                                0.05,
                            )
                        frac_h = float(
                            np.clip(
                                float(language_only_demo_clearance_frac_above_tallest), 0.0, 3.0
                            )
                        )
                        clear_z = float(
                            np.clip(float(language_only_demo_clearance_above_scene_z_m), 0.0, 2.0)
                        )
                        z_tgt_world = float(
                            z_top_max_world + float(frac_h * float(Hmax) + clear_z)
                        )
                        workspace_hi = _expand_workspace_hi_local_for_world_z(
                            workspace_hi,
                            z_target_world=float(z_tgt_world),
                            virtual_base_z=float(virtual_base_world[2]),
                        )
                        z_min_world = float(z_tgt_world)
                        z_ceil_w = float(workspace_hi[2] + virtual_base_world[2]) - 0.005
                        z_min_world = float(
                            np.clip(z_min_world, z_floor_w + 0.01, z_ceil_w - 0.01)
                        )
                        z_min_local = z_min_world - float(virtual_base_world[2])
                        target_pos_local = np.asarray(target_pos_local, dtype=np.float32).copy()
                        target_pos_local[2] = max(float(target_pos_local[2]), float(z_min_local))
                        target_pos_local[2] = float(
                            np.clip(
                                target_pos_local[2],
                                float(workspace_lo[2]),
                                float(workspace_hi[2]),
                            )
                        )
                        per = max(4, int(language_only_demo_traj_period_infers))
                        demo_traj_phase += (2.0 * float(np.pi)) / float(per)
                        if demo_fig8:
                            xy_pat = _demo_plan_figure8_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_oval:
                            xy_pat = _demo_plan_ellipse_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_circle_orbit:
                            xy_pat = _demo_plan_circle_orbit_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_diamond:
                            xy_pat = _demo_plan_diamond_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_square:
                            xy_pat = _demo_plan_square_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_phyllotaxis:
                            xy_pat = _demo_plan_phyllotaxis_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_arch_spiral:
                            xy_pat = _demo_plan_archimedean_spiral_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_spiral:
                            xy_pat = _demo_plan_spiral_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_serpentine:
                            xy_pat = _demo_plan_serpentine_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_cross_shuttle:
                            xy_pat = _demo_plan_cross_axis_shuttle_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_shuttle:
                            xy_pat = _demo_plan_shuttle_line_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_clover:
                            xy_pat = _demo_plan_clover_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_star:
                            xy_pat = _demo_plan_star_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_lissajous:
                            xy_pat = _demo_plan_lissajous_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_heart:
                            xy_pat = _demo_plan_heart_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_teardrop:
                            xy_pat = _demo_plan_teardrop_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_polygon:
                            _ns_blend = scenic_polygon_sides if scenic_polygon_sides is not None else 6
                            xy_pat = _demo_plan_regular_polygon_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                sides=int(_ns_blend),
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_rose:
                            _rk_blend = scenic_rose_petals if scenic_rose_petals is not None else 5
                            xy_pat = _demo_plan_rose_petal_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                petals_k=int(_rk_blend),
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        elif demo_damped_sinewave:
                            xy_pat = _demo_plan_damped_sine_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_trig_beat_wave:
                            xy_pat = _demo_plan_trig_beat_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_triangle_wave:
                            xy_pat = _demo_plan_triangle_wave_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_tanh_ribbon:
                            xy_pat = _demo_plan_tanh_ribbon_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_cosine_wave:
                            xy_pat = _demo_plan_cosine_wave_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_sinewave:
                            xy_pat = _demo_plan_sinewave_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_stadium_capsule:
                            xy_pat = _demo_plan_stadium_capsule_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_cycloid_arc:
                            xy_pat = _demo_plan_cycloid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_cardioid:
                            xy_pat = _demo_plan_cardioid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_limacon:
                            xy_pat = _demo_plan_limacon_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_deltoid:
                            xy_pat = _demo_plan_deltoid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_astroid:
                            xy_pat = _demo_plan_astroid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_epitrochoid:
                            xy_pat = _demo_plan_epitrochoid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_nephroid:
                            xy_pat = _demo_plan_nephroid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_epicycloid:
                            xy_pat = _demo_plan_epicycloid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_logspiral:
                            xy_pat = _demo_plan_logarithmic_spiral_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_hyp_spiral:
                            xy_pat = _demo_plan_hyperbolic_spiral_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_involute:
                            xy_pat = _demo_plan_involute_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_squircle:
                            xy_pat = _demo_plan_superellipse_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_butterfly:
                            xy_pat = _demo_plan_butterfly_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_gaussian:
                            xy_pat = _demo_plan_gaussian_track_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_arc_chain:
                            xy_pat = _demo_plan_arc_chain_wave_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_bern_lemn:
                            xy_pat = _demo_plan_bernoulli_lemniscate_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_cochleoid:
                            xy_pat = _demo_plan_cochleoid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_folium:
                            xy_pat = _demo_plan_steiner_folium_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_clothoid:
                            xy_pat = _demo_plan_clothoid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_tractrix:
                            xy_pat = _demo_plan_tractrix_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_witch:
                            xy_pat = _demo_plan_witch_of_agnesi_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_cassini:
                            xy_pat = _demo_plan_cassini_oval_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_hypotroch:
                            xy_pat = _demo_plan_hypotrochoid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_strophoid:
                            xy_pat = _demo_plan_strophoid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_kampyle:
                            xy_pat = _demo_plan_kampyle_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_conchoid_nic:
                            xy_pat = _demo_plan_conchoid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_cissoid:
                            xy_pat = _demo_plan_cissoid_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        elif demo_parabola:
                            xy_pat = _demo_plan_parabolic_arc_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(language_only_demo_plan_edge_margin_frac),
                            )
                        else:
                            xy_pat = _demo_plan_figure8_xy_in_workspace(
                                workspace_lo,
                                workspace_hi,
                                demo_traj_phase,
                                fill_frac=float(language_only_demo_plan_fill_frac),
                                edge_margin_frac=float(
                                    language_only_demo_plan_edge_margin_frac
                                ),
                            )
                        b = float(np.clip(language_only_demo_plan_blend_beta, 0.0, 1.0))
                        target_pos_local[0] = b * float(target_pos_local[0]) + (
                            1.0 - b
                        ) * float(xy_pat[0])
                        target_pos_local[1] = b * float(target_pos_local[1]) + (
                            1.0 - b
                        ) * float(xy_pat[1])
                        target_pos_local[0:2] = np.minimum(
                            np.maximum(target_pos_local[0:2], workspace_lo[0:2]),
                            workspace_hi[0:2],
                        ).astype(np.float32)
                        if demo_helical_z:
                            z_span_l = max(float(workspace_hi[2] - z_min_local), 0.0)
                            z_hi_l = float(z_min_local + 0.52 * z_span_l)
                            mid = 0.5 * (float(z_min_local) + z_hi_l)
                            amp = 0.5 * (z_hi_l - float(z_min_local))
                            if amp > 1e-6:
                                ph = float(scenic_helical_turns) * float(demo_traj_phase)
                                _zmode = str(scenic_z_trig).strip().lower()
                                z_osc = np.cos(ph) if _zmode == "cos" else np.sin(ph)
                                target_pos_local[2] = float(mid + amp * float(z_osc))
                            target_pos_local[2] = float(
                                np.clip(
                                    target_pos_local[2],
                                    float(workspace_lo[2]),
                                    float(workspace_hi[2]),
                                )
                            )
                        if not demo_traj_log_once:
                            print(
                                "[demo] scenic planar path + clearance: "
                                f"z_floor_world={z_floor_w:.3f} H_max={Hmax:.3f} "
                                f"z_min_world={z_min_world:.3f} blend_beta={b:.2f} period_infers={per}"
                            )
                            demo_traj_log_once = True
                    target_pos = (virtual_base_world + target_pos_local).astype(np.float32)

            excl_near: list[np.ndarray] = []
            if cur_task is not None:
                if goal_center_xyz is not None:
                    excl_near.append(np.asarray(goal_center_xyz, dtype=np.float64).reshape(3))
                if nav_goal_xyz is not None:
                    ng = np.asarray(nav_goal_xyz, dtype=np.float64).reshape(3)
                    if not excl_near:
                        excl_near.append(ng)
                    elif float(np.linalg.norm(ng - excl_near[0])) > 1e-3:
                        excl_near.append(ng)
            if (
                local_avoidance_enabled
                and not phase3_sim_path_follow
                and with_objects
                and placed_cubes
            ):
                rep_t = local_obstacle_repulsion_world(
                    drone_pos,
                    placed_cubes,
                    robot_radius=float(local_avoidance_robot_radius),
                    influence_m=float(local_avoidance_influence_m),
                    gain=float(local_avoidance_gain),
                    max_delta_m=float(local_avoidance_target_max_shift_m),
                    exclude_near_points=excl_near or None,
                    exclude_tol_m=float(local_avoidance_exclude_goal_tol_m),
                )
                if float(np.linalg.norm(rep_t)) > 1e-9:
                    target_pos = (
                        np.asarray(target_pos, dtype=np.float64) + rep_t.astype(np.float64)
                    ).astype(np.float32)
                    target_pos = _clip_world_pos_to_workspace(
                        target_pos,
                        workspace_lo,
                        workspace_hi,
                        virtual_base_world,
                    )

            alpha_r = float(np.clip(rot_lerp_alpha * speed_scale, 0.0, 1.0))

            if cur_task is not None:
                assert goal_center_xyz is not None and nav_goal_xyz is not None
                goal_center = goal_center_xyz
                dist_portal = float(np.linalg.norm(drone_pos - goal_center))
                dist_now = float(np.linalg.norm(drone_pos - nav_goal_xyz))

                if (
                    fpv_slot_align
                    and placed_cubes
                    and dist_portal <= float(fpv_slot_align_dist)
                ):
                    proprio_gate = build_proprio_widowx_ee6d(drone_pos_local, drone_R, gripper=gripper_state)
                    pinfo = estimate_rect_portal_pose_fpv(
                        p,
                        placed_cubes,
                        goal_world=goal_center,
                        drone_pos=drone_pos,
                        drone_R=drone_R,
                        match_tol=float(portal_match_tol),
                        use_sim_truth=bool(fpv_use_sim_truth_pose),
                        cam_offset_body=fpv_off_b,
                        cam_look_body=fpv_look_b,
                        fpv_width=int(fpv_cam_width),
                        fpv_height=int(fpv_cam_height),
                        fpv_fov=float(fpv_cam_fov),
                        gate_pose_use_dedicated_camera=bool(gate_pose_use_dedicated_camera),
                        gate_pose_cam_offset_body=gate_pose_off_b,
                        gate_pose_cam_look_body=gate_pose_look_b,
                        gate_pose_cam_width=int(gate_pose_cam_width),
                        gate_pose_cam_height=int(gate_pose_cam_height),
                        gate_pose_cam_fov=float(gate_pose_cam_fov),
                        gate_pose_cv_rgb_pad=int(gate_pose_cv_rgb_pad),
                        gate_pose_cv_min_area_ratio=float(gate_pose_cv_min_area_ratio),
                        gate_pose_cv_fallback_map_pose=bool(gate_pose_cv_fallback_map_pose),
                        gate_pose_estimator=str(gate_pose_estimator),
                        xvla_server_url=str(server_url),
                        xvla_gate_instruction_template=str(xvla_gate_instruction_template),
                        xvla_gate_steps=int(xvla_gate_steps),
                        xvla_gate_infer_width=int(xvla_gate_infer_width),
                        xvla_gate_infer_height=int(xvla_gate_infer_height),
                        proprio_20d=proprio_gate,
                        decode_workspace_lo=workspace_lo,
                        decode_workspace_hi=workspace_hi,
                        decode_treat_pos_as=str(treat_pos_as),
                        decode_delta_pos_scale=float(delta_pos_scale),
                        gate_pose_xvla_fallback_opencv=bool(gate_pose_xvla_fallback_opencv),
                        xvla_act_request_timeout_s=float(xvla_act_request_timeout_s),
                    )
                    if pinfo is not None and pinfo.get("R_body_world") is not None:
                        target_R = pinfo["R_body_world"]
                        alpha_r = 1.0

                dist_history.append(dist_now)
                if len(dist_history) > int(phase1_stall_window):
                    dist_history.pop(0)
                if nav_phase == 1 and len(dist_history) >= int(phase1_stall_window):
                    improvement = dist_history[0] - dist_history[-1]
                    if improvement < float(phase1_stall_threshold):
                        nav_phase = 2
                        print(
                            f"[phase] task={cur_task['label']} step={t} "
                            f"dist={dist_now:.3f} → Phase2 (P-controller): "
                            f"improvement={improvement:.4f} < {phase1_stall_threshold}"
                        )
                if nav_phase == 1 and task_step_count >= int(phase1_max_steps):
                    nav_phase = 2
                    print(
                        f"[phase] task={cur_task['label']} step={t} "
                        f"dist={dist_now:.3f} → Phase2 (P-controller): "
                        f"phase1_max_steps={phase1_max_steps}"
                    )

                if nav_phase == 1:
                    alpha_p = float(np.clip(pos_lerp_alpha * speed_scale, 0.0, 1.0))
                    disp = float(np.linalg.norm(target_pos - drone_pos))
                    if disp > float(movement_deadband):
                        drone_pos = ((1 - alpha_p) * drone_pos + alpha_p * target_pos).astype(np.float32)

                else:
                    err = nav_goal_xyz - drone_pos
                    if (
                        local_avoidance_enabled
                        and with_objects
                        and placed_cubes
                    ):
                        rep_p2 = local_obstacle_repulsion_world(
                            drone_pos,
                            placed_cubes,
                            robot_radius=float(local_avoidance_robot_radius),
                            influence_m=float(local_avoidance_influence_m),
                            gain=float(local_avoidance_gain),
                            max_delta_m=float(local_avoidance_target_max_shift_m),
                            exclude_near_points=excl_near or None,
                            exclude_tol_m=float(local_avoidance_exclude_goal_tol_m),
                        )
                        err = err + float(local_avoidance_phase2_gain) * rep_p2.astype(np.float64)
                    err_norm = float(np.linalg.norm(err))
                    snap_r = max(float(target_snap_radius), float(cur_task["arrive_radius"]) * 0.5)
                    if err_norm <= snap_r:
                        drone_pos = nav_goal_xyz.astype(np.float32)
                    elif err_norm > 1e-6:
                        step_size = float(phase2_p_gain) * float(dt) * err_norm
                        step_size = float(np.clip(step_size, 0.0, float(phase2_max_speed) * float(dt)))
                        drone_pos = (drone_pos + (err / err_norm) * step_size).astype(np.float32)
                    if not virtual_workspace_enabled:
                        drone_pos = np.minimum(np.maximum(drone_pos, workspace_lo), workspace_hi).astype(np.float32)

                if portal_pass_spec is not None and portal_pass_stage == 0:
                    sw_r = (
                        float(portal_pass_stage_switch_radius)
                        if portal_pass_stage_switch_radius is not None
                        else max(0.08, float(cur_task["arrive_radius"]) * 1.75)
                    )
                    if float(np.linalg.norm(drone_pos - portal_pass_spec["approach"])) <= sw_r:
                        portal_pass_stage = 1
                        nav_phase = 1
                        task_step_count = 0
                        dist_history.clear()
                        dwell_count = 0
                        print(
                            f"[portal] {cur_task['label']}: reached pre-opening hold → crossing to far side"
                        )

            else:
                alpha_p = float(np.clip(pos_lerp_alpha * speed_scale, 0.0, 1.0))
                disp = float(np.linalg.norm(target_pos - drone_pos))
                if disp > float(movement_deadband):
                    drone_pos = ((1 - alpha_p) * drone_pos + alpha_p * target_pos).astype(np.float32)
                drone_pos = _clip_world_pos_to_workspace(
                    drone_pos,
                    workspace_lo,
                    workspace_hi,
                    virtual_base_world,
                )

            drone_R = ((1 - alpha_r) * drone_R + alpha_r * target_R).astype(np.float32)
            try:
                u, _, vt = np.linalg.svd(drone_R)
                drone_R = (u @ vt).astype(np.float32)
                if np.linalg.det(drone_R) < 0:
                    u[:, -1] *= -1
                    drone_R = (u @ vt).astype(np.float32)
            except Exception:
                drone_R = np.eye(3, dtype=np.float32)

            update_floating_drone(p, body_uid, rotor_uids, offsets, drone_pos, drone_R)
            if trail_enabled:
                create_trail_segment(
                    p,
                    trail_last_pos,
                    drone_pos,
                    radius=trail_radius,
                    rgba=trail_color,
                    min_length=demo_trail_min_len,
                )
                trail_last_pos = drone_pos.copy()

            try:
                p.stepSimulation()
            except Exception as exc:
                if "physics server" in str(exc).lower() or "not connected" in str(exc).lower():
                    print(f"Simulation stopped: {exc}")
                    break
                raise

            steps_executed = t + 1

            cur_dist = float("nan")
            advanced = False
            stop_after_record = False
            if cur_task is not None:
                assert goal_center_xyz is not None and nav_goal_xyz is not None
                nav_goal = nav_goal_xyz
                cur_dist = float(np.linalg.norm(drone_pos - nav_goal))
                if portal_pass_spec is None or portal_pass_stage == 1:
                    if cur_dist < float(cur_task["arrive_radius"]):
                        dwell_count += 1
                    else:
                        dwell_count = 0
                else:
                    dwell_count = 0
                if dwell_count >= int(cur_task["dwell_steps"]):
                    drone_pos = nav_goal.astype(np.float32)
                    target_pos = drone_pos.copy()
                    cur_dist = 0.0
                    if trail_enabled:
                        create_trail_segment(
                            p,
                            trail_last_pos,
                            drone_pos,
                            radius=trail_radius,
                            rgba=trail_color,
                            min_length=1e-6,
                        )
                        trail_last_pos = drone_pos.copy()
                    visited_log.append(
                        {
                            "label": cur_task["label"],
                            "step": int(t),
                            "dist": cur_dist,
                            "pos": drone_pos.round(4).tolist(),
                        }
                    )
                    print(
                        f"[task] arrived: {cur_task['label']} "
                        f"step={t} dist={cur_dist:.3f} pos={drone_pos.round(3).tolist()}"
                    )
                    cur_task_idx += 1
                    dwell_count = 0
                    advanced = True
                    nav_phase = 1
                    task_step_count = 0
                    dist_history.clear()
                    if cur_task_idx >= len(task_sequence):
                        if stop_when_all_visited:
                            print(f"[task] all {len(task_sequence)} targets visited; stopping.")
                            stop_after_record = True
                        else:
                            cur_task_idx = 0

            if advanced:
                update_floating_drone(p, body_uid, rotor_uids, offsets, drone_pos, drone_R)

            if log_every <= 1 or (t % log_every == 0) or advanced:
                e = euler_xyz_from_matrix(drone_R)
                traj_play = (
                    cur_task is None
                    and demo_xvla_once
                    and demo_xvla_world_pts is not None
                )
                if nav_phase == 1:
                    if phase3_sim_path_follow:
                        ctrl_tag = f" Ph1(Phase3-replay) u={u_traj:.3f}"
                    elif traj_play:
                        if (
                            demo_xvla_plan_step is not None
                            and int(t) == int(demo_xvla_plan_step)
                            and inferred
                            and last_infer_ms is not None
                        ):
                            ctrl_tag = f" Ph1(traj) xvla_plan={last_infer_ms:.1f}ms"
                        else:
                            ctrl_tag = " Ph1(traj) interp"
                    else:
                        ctrl_tag = (
                            f" Ph1(VLA) xvla_infer={last_infer_ms:.1f}ms"
                            if (inferred and last_infer_ms is not None)
                            else " Ph1(VLA) xvla=reuse"
                        )
                else:
                    ctrl_tag = " Ph2(P-ctrl)"
                if cur_task is not None:
                    task_tag = (
                        f" task[{cur_task_idx}/{len(task_sequence)}]={cur_task['label']!r} "
                        f"dist={cur_dist:.3f} dwell={dwell_count}/{cur_task['dwell_steps']}"
                    )
                else:
                    task_tag = ""
                print(
                    f"step={t:04d} pos={drone_pos.round(3).tolist()} "
                    f"rpy=[{e[0]:+.3f},{e[1]:+.3f},{e[2]:+.3f}] gripper={gripper_state:.3f}"
                    f"{ctrl_tag}{task_tag}"
                )

            if rec_folder is not None and ((t % rec_every == 0) or stop_after_record):
                line1 = f"step={t}  pos={format_xyz_for_overlay(drone_pos, decimals=6)}"
                if cur_task is not None:
                    ph_str = "Ph2-Pctrl" if nav_phase == 2 else "Ph1-VLA"
                    task_display_idx = min(cur_task_idx + 1, len(task_sequence))
                    line2 = (
                        f"task {task_display_idx}/{len(task_sequence)} "
                        f"{cur_task['label']}  d={cur_dist:.2f}m  [{ph_str}]"
                    )
                else:
                    line2 = (
                        f"Phase3-follow  u={u_traj:.3f}"
                        if phase3_sim_path_follow
                        else f"rpy={[round(x,2) for x in euler_xyz_from_matrix(drone_R)]}"
                    )
                for view_kind, dst in (
                    ("front", world_front_frames),
                    ("right", world_right_frames),
                    ("top", world_top_frames),
                    ("45deg", world_45deg_frames),
                ):
                    cam_ds = float(recording_camera_distance_scale)
                    top_ds = float(recording_topview_distance_scale)
                    stereo_ds = float(recording_stereo45_distance_scale)
                    if phase3_sim_path_follow:
                        if view_kind == "right":
                            cam_ds *= float(
                                np.clip(navigation_sim_recording_right_distance_scale, 0.05, 3.0)
                            )
                        elif view_kind == "top":
                            top_ds *= float(
                                np.clip(navigation_sim_recording_top_distance_scale, 0.05, 3.0)
                            )
                        elif view_kind == "45deg":
                            stereo_ds *= float(
                                np.clip(navigation_sim_recording_stereo45_distance_scale, 0.05, 3.0)
                            )
                    snap = get_world_recording_rgb(
                        p,
                        recording_scene_lo,
                        recording_scene_hi,
                        view_kind=view_kind,
                        width=int(recording_width),
                        height=int(recording_height),
                        fov=recording_scene_fov,
                        margin=recording_scene_margin,
                        distance_scale=cam_ds,
                        top_view_distance_scale=top_ds,
                        stereo45_view_distance_scale=stereo_ds,
                    )
                    if view_kind == "top" and placed_cubes:
                        view_m, proj_m = _world_recording_view_proj(
                            p,
                            recording_scene_lo,
                            recording_scene_hi,
                            view_kind="top",
                            width=int(recording_width),
                            height=int(recording_height),
                            fov=float(recording_scene_fov),
                            margin=float(recording_scene_margin),
                            distance_scale=float(cam_ds),
                            top_view_distance_scale=float(top_ds),
                        )

                        def _wproj_top_lbl(wx: float, wy: float, wz: float) -> tuple[int, int] | None:
                            return world_xyz_to_recording_image_pixel(
                                np.array([wx, wy, wz], dtype=np.float64),
                                view_m,
                                proj_m,
                                width=int(recording_width),
                                height=int(recording_height),
                            )

                        snap = sharpen_topdown_portal_labels_rgb(
                            snap, placed_cubes, _wproj_top_lbl
                        )
                    dst.append(frame_with_overlay(snap, line1, line2))

            if stop_after_record:
                hold_frames = max(0, int(recording_final_hold_frames))
                if hold_frames > 0 and world_front_frames:
                    for _ in range(hold_frames):
                        world_front_frames.append(world_front_frames[-1].copy())
                        world_right_frames.append(world_right_frames[-1].copy())
                        world_top_frames.append(world_top_frames[-1].copy())
                        world_45deg_frames.append(world_45deg_frames[-1].copy())
                break

            if gui and realtime_sleep:
                time.sleep(min(dt, 1.0 / 30.0))
    finally:
        if steps_executed > 0:
            visited_tags = [v["label"] for v in visited_log]
            print(
                f"[sim] summary steps={steps_executed} xvla_infers={infer_calls} "
                f"visited={visited_tags} final_pos={np.asarray(drone_pos).round(3).tolist()}"
            )
        try:
            p.disconnect()
        except Exception:
            pass
        if rec_folder is not None:
            fmt = (recording_format or "mp4").lstrip(".").lower()
            if fmt not in ("gif", "mp4"):
                fmt = "mp4"
            rec_folder.mkdir(parents=True, exist_ok=True)
            nfr = len(world_front_frames)
            print(f"[record] saving {nfr} frames per view -> {rec_folder.resolve()}/", flush=True)
            saves: list[tuple[str, list[np.ndarray]]] = [
                ("world_front", world_front_frames),
                ("world_right", world_right_frames),
                ("world_top", world_top_frames),
                ("world_45deg", world_45deg_frames),
            ]
            for stem, frames in saves:
                out_path = rec_folder / f"{stem}.{fmt}"
                if not frames:
                    print(f"[record] skipped {stem}: no frames captured", file=sys.stderr)
                    continue
                frames_to_save = frames if len(frames) > 1 else [frames[0], frames[0]]
                try:
                    save_navigation_video(frames_to_save, out_path, recording_fps)
                    print(f"[record] saved {stem}: {out_path.name}")
                except Exception as exc_save:
                    print(f"[record] ERROR saving {stem}: {exc_save}", file=sys.stderr)
                    traceback.print_exc()


def _vec3(name: str, cfg: dict[str, Any], default: list[float]) -> np.ndarray:
    v = cfg.get(name, default)
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must be a length-3 list, got {v!r}")
    return arr


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=CONFIG_DEFAULT_PATH,
                     help=f"Path to JSON config (default: {CONFIG_DEFAULT_PATH.name}).")
    pre.add_argument("--scheme", type=str, default=None, metavar="KEY",
                     help='Override config "scheme" / preset key (e.g. widowx_ee6d, 1).')
    pre_args, remaining = pre.parse_known_args()
    help_only = "--help" in remaining or "-h" in remaining

    cli = pre_args.scheme.strip() if pre_args.scheme else None
    scheme_override: int | str | None = int(cli) if (cli is not None and cli.isdigit()) else cli

    try:
        cfg = load_widowx_demo_config(pre_args.config, scheme_override=scheme_override)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    scheme_info = cfg.pop("_resolved_scheme", None)
    scheme_label = cfg.pop("_resolved_scheme_label", None)
    for _meta in ("label", "_label"):
        cfg.pop(_meta, None)
    if scheme_info is not None and not help_only:
        lbl = f" ({scheme_label})" if scheme_label else ""
        print(f"[config] scheme={scheme_info}{lbl}")

    def ckey(name: str, fallback: Any) -> Any:
        return cfg.get(name, fallback)

    parser = argparse.ArgumentParser(
        description="X-VLA WidowX-EE6D drone demo (drone center == virtual gripper tip).",
        parents=[pre],
    )
    parser.add_argument("--server-url", default=ckey("server_url", "http://127.0.0.1:8000/act"))
    parser.add_argument("--no-auto-server", action="store_true")
    parser.add_argument(
        "--xvla-act-timeout",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "HTTP read timeout (seconds) for each POST /act inference during the sim "
            "(default: config xvla_act_request_timeout_s, else xvla_act_probe_request_timeout_s, "
            "else 300). Must be >= 30."
        ),
    )
    parser.add_argument(
        "--cmd",
        default=None,
        metavar="TEXT",
        help=(
            "Sole natural-language task for the drone; sent to X-VLA as language_instruction on "
            "each /act call. When set, overrides --instruction and config 'instruction', and "
            "disables task_sequence (auto cube waypoints or config JSON): no code-enforced visit "
            "order. Prefer multiple focused runs for long missions."
        ),
    )
    parser.add_argument("--instruction", default=ckey("instruction", "pick up the red block"))
    parser.add_argument("--sim-steps", type=int, default=int(ckey("sim_steps", 200)))
    parser.add_argument("--xvla-steps", type=int, default=int(ckey("xvla_steps", 6)))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--speed", type=float, default=float(ckey("speed", 1.0)))
    parser.add_argument("--infer-every", type=int, default=config_infer_every(cfg, 4))
    parser.add_argument("--dt", type=float, default=float(ckey("dt", 0.06)))
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--log-every", type=int, default=int(ckey("log_every", 5)))
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--recording-path", default=None)
    parser.add_argument("--recording-fps", type=float, default=float(ckey("recording_fps", 18.0)))
    parser.add_argument("--recording-format", choices=["gif", "mp4"], default=None)
    parser.add_argument("--no-objects", action="store_true",
                        help="Empty workspace (no Bridge-style objects).")
    parser.add_argument("--treat-pos-as", choices=["absolute", "delta"], default=None)
    parser.add_argument(
        "--auto-motion",
        action="store_true",
        help=(
            "Scale pos_lerp_alpha and delta_pos_scale from workspace size, infer-every gap, "
            "and task mode (pure --cmd / empty task_sequence vs scripted waypoints)."
        ),
    )
    parser.add_argument(
        "--no-auto-motion",
        action="store_true",
        help="Turn off auto motion scaling even if config sets auto_motion_scales.",
    )
    parser.add_argument(
        "--cmd-motion-amplify",
        type=float,
        default=None,
        metavar="K",
        help=(
            "Pure --cmd / empty task_sequence: scale local displacement from drone to decoded "
            "target (default: config language_only_motion_amplify, 1=unchanged)."
        ),
    )
    parser.add_argument(
        "--cmd-min-step",
        type=float,
        default=None,
        metavar="M",
        help=(
            "Pure language mode: after workspace clip, enforce at least M meters from drone to "
            "target in local wrist frame (default: language_only_min_step_local_m)."
        ),
    )
    parser.add_argument(
        "--infer-displacement-scale",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Every /act call: multiply wrist-local vector from current pose to decoded target by S "
            "after pull & language-only boost; 1=no change. Config: infer_displacement_scale."
        ),
    )
    parser.add_argument(
        "--cmd-global-cam-pull",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Pure --cmd: blend decoded wrist target toward global workspace RGB color mask centroid "
            "(ray-plane on matched portal opening). 0 disables blend. Config: cmd_global_cam_pull_alpha."
        ),
    )
    parser.add_argument(
        "--cmd-global-cam",
        action="store_true",
        help="Enable global-workspace-camera --cmd target fusion (config cmd_global_cam_pull_alpha).",
    )
    parser.add_argument(
        "--no-cmd-global-cam",
        action="store_true",
        help="Disable global-workspace-camera --cmd fusion even if config enables it.",
    )
    parser.add_argument(
        "--qs-path",
        default=None,
        metavar="PATH",
        help=(
            "QS.json expert policy path (default: config qs_policy_path, "
            "or <project>/QS.json)."
        ),
    )
    parser.add_argument(
        "--vvla-url",
        default=None,
        metavar="URL",
        help=(
            "v-vla HTTP endpoint: POST JSON {\"user_command\":str,\"policies\":[{\"Q\",\"S\"},...]} "
            "→ JSON {\"indices\":[int,...]} (or match_indices / matches)."
        ),
    )
    parser.add_argument(
        "--no-vvla",
        action="store_true",
        help="Disable QS.json + v-vla enrichment for --cmd (use raw command for X-VLA).",
    )
    parser.add_argument(
        "--no-inference-router",
        action="store_true",
        help=(
            "Disable automatic routing between X-VLA /act (neural) vs local QS + scenic geometry "
            "for --cmd."
        ),
    )
    parser.add_argument(
        "--inference-router-heuristic-only",
        action="store_true",
        help="Use keyword heuristic only for inference routing (no chat JSON router API call).",
    )
    parser.add_argument(
        "--no-xvla-scene-catalog",
        action="store_true",
        help="Disable prepending the full sim object catalog + path-planning suffix to language_instruction (/act).",
    )
    parser.add_argument(
        "--no-local-avoidance",
        action="store_true",
        help="Disable artificial-potential obstacle repulsion (near-field classical avoidance).",
    )
    args = parser.parse_args(remaining)

    user_cmd = (args.cmd or "").strip()
    effective_instruction = user_cmd if user_cmd else args.instruction
    if user_cmd:
        print(f"[cmd] user --cmd (raw): {user_cmd!r}")

    if args.gui and args.no_gui:
        parser.error("Use only one of --gui and --no-gui")
    use_gui = True if args.gui else (False if args.no_gui else bool(ckey("gui", True)))

    if args.realtime and args.no_realtime:
        parser.error("Use only one of --realtime and --no-realtime")
    realtime = True if args.realtime else (False if args.no_realtime else bool(ckey("realtime", False)))

    record_visualization = bool(ckey("record_visualization", True)) and not args.no_record
    _path_cfg = (
        args.recording_path if args.recording_path is not None
        else str(ckey("recording_path", str(ROOT / "recordings")))
    )
    _p = Path(_path_cfg)
    base_rec_dir = str(_p.parent if _p.suffix in (".gif", ".mp4") else _p)
    recording_fmt = args.recording_format if args.recording_format is not None \
        else str(ckey("recording_format", "mp4")).lstrip(".").lower()
    if recording_fmt not in ("gif", "mp4"):
        recording_fmt = "mp4"
    recording_folder = resolve_recording_folder(
        base_rec_dir,
        prefix=str(ckey("recording_folder_prefix", "")),
        append_timestamp=bool(ckey("recording_append_timestamp", True)),
    )
    if record_visualization and not help_only:
        print(f"[config] recording_format={recording_fmt!r} -> folder: {recording_folder}")

    workspace_lo = _vec3("workspace_lo", cfg, [0.15, -0.25, 0.02])
    workspace_hi = _vec3("workspace_hi", cfg, [0.55, 0.25, 0.40])
    cam_eye = _vec3("camera_eye", cfg, [-0.55, 0.05, 0.40])
    cam_look = _vec3("camera_look_at", cfg, [0.35, 0.00, 0.18])

    treat_pos_as = (
        args.treat_pos_as if args.treat_pos_as is not None
        else str(ckey("treat_pos_as", "absolute")).strip().lower()
    )
    if treat_pos_as not in ("absolute", "delta"):
        parser.error("treat_pos_as must be 'absolute' or 'delta'")

    delta_pos_scale = float(ckey("delta_pos_scale", 0.04))
    pos_lerp_alpha = float(ckey("pos_lerp_alpha", 0.25))
    rot_lerp_alpha = float(ckey("rot_lerp_alpha", 0.20))
    with_objects = (not args.no_objects) and bool(ckey("with_objects", True))

    raw_seq = ckey("task_sequence", None)
    if raw_seq is not None and not isinstance(raw_seq, list):
        parser.error("task_sequence must be a JSON list of {instruction,target_xyz,arrive_radius,dwell_steps,label}")
    if user_cmd:
        if raw_seq is not None and len(raw_seq) > 0:
            print(
                "[cmd] ignoring config task_sequence (--cmd mode: no scripted waypoint sequence)."
            )
        raw_seq = []
    language_only = bool(user_cmd) or (raw_seq is not None and len(raw_seq) == 0)

    if args.auto_motion and args.no_auto_motion:
        parser.error("Use only one of --auto-motion and --no-auto-motion")
    auto_motion = bool(ckey("auto_motion_scales", False))
    if args.auto_motion:
        auto_motion = True
    if args.no_auto_motion:
        auto_motion = False

    if auto_motion:
        _ref_d = float(ckey("auto_motion_ref_diag_m", 0.55))
        pl0, ds0 = pos_lerp_alpha, delta_pos_scale
        pos_lerp_alpha, delta_pos_scale = apply_auto_motion_scales(
            workspace_lo=workspace_lo,
            workspace_hi=workspace_hi,
            infer_every=int(args.infer_every),
            language_only=language_only,
            pos_lerp_alpha=pos_lerp_alpha,
            delta_pos_scale=delta_pos_scale,
            ref_diag_m=_ref_d,
        )
        if not help_only and (
            abs(pl0 - pos_lerp_alpha) > 1e-6 or abs(ds0 - delta_pos_scale) > 1e-6
        ):
            print(
                f"[motion] auto_motion_scales: pos_lerp {pl0:.4f}→{pos_lerp_alpha:.4f}, "
                f"delta_pos_scale {ds0:.4f}→{delta_pos_scale:.4f} "
                f"(language_only={language_only}, infer_every={int(args.infer_every)}, "
                f"ref_diag_m={_ref_d:.3f})"
            )

    cubes_override = ckey("cubes", None)
    if cubes_override is not None and not isinstance(cubes_override, list):
        parser.error("cubes must be a JSON list of {color,pos,rgba,half}")
    task_default_phrase = str(
        ckey(
            "task_default_phrase",
            (
                "Fly through rectangular portal billboard_id={billboard_id} ({color} visual); "
                "align with the long narrow opening."
            ),
        )
    )
    task_default_arrive_radius = float(ckey("task_default_arrive_radius", 0.05))
    task_default_dwell_steps = int(ckey("task_default_dwell_steps", 3))
    stop_when_all_visited = bool(ckey("stop_when_all_visited", True))

    target_pull_alpha = float(ckey("target_pull_alpha", 0.0))
    target_snap_radius = float(ckey("target_snap_radius", 0.0))
    movement_deadband = float(ckey("movement_deadband", 0.0))

    language_only_motion_amplify = float(ckey("language_only_motion_amplify", 1.0))
    language_only_min_step_local_m = float(ckey("language_only_min_step_local_m", 0.0))
    if args.cmd_motion_amplify is not None:
        language_only_motion_amplify = float(args.cmd_motion_amplify)
    if args.cmd_min_step is not None:
        language_only_min_step_local_m = float(args.cmd_min_step)

    cmd_color_disambiguation = bool(ckey("cmd_color_disambiguation", True))

    infer_displacement_scale = float(ckey("infer_displacement_scale", 1.0))
    if args.infer_displacement_scale is not None:
        infer_displacement_scale = float(args.infer_displacement_scale)

    fpv_slot_align = bool(ckey("fpv_slot_align", False))
    fpv_slot_align_dist = float(ckey("fpv_slot_align_dist", 0.35))
    fpv_use_sim_truth_pose = bool(ckey("fpv_use_sim_truth_pose", True))
    portal_match_tol = float(ckey("portal_match_tol", 0.12))
    portal_pass_through_enabled = bool(ckey("portal_pass_through_enabled", True))
    portal_pass_approach_offset = float(ckey("portal_pass_approach_offset", 0.14))
    portal_pass_exit_offset = float(ckey("portal_pass_exit_offset", 0.14))
    _psw = ckey("portal_pass_stage_switch_radius", None)
    portal_pass_stage_switch_radius = float(_psw) if _psw is not None else None
    fpv_cam_offset_body = _vec3("fpv_cam_offset_body", cfg, [0.06, 0.0, 0.02])
    fpv_cam_look_body = _vec3("fpv_cam_look_body", cfg, [0.85, 0.0, 0.0])
    fpv_cam_width = int(ckey("fpv_cam_width", 256))
    fpv_cam_height = int(ckey("fpv_cam_height", 256))
    fpv_cam_fov = float(ckey("fpv_cam_fov", 72.0))

    gate_pose_use_dedicated_camera = bool(ckey("gate_pose_use_dedicated_camera", False))
    gate_pose_cam_offset_body = _vec3("gate_pose_cam_offset_body", cfg, [0.06, 0.0, 0.032])
    gate_pose_cam_look_body = _vec3("gate_pose_cam_look_body", cfg, [0.95, 0.0, -0.08])
    gate_pose_cam_width = int(ckey("gate_pose_cam_width", 320))
    gate_pose_cam_height = int(ckey("gate_pose_cam_height", 320))
    gate_pose_cam_fov = float(ckey("gate_pose_cam_fov", 82.0))
    gate_pose_cv_rgb_pad = int(ckey("gate_pose_cv_rgb_pad", 45))
    gate_pose_cv_min_area_ratio = float(ckey("gate_pose_cv_min_area_ratio", 0.015))
    gate_pose_cv_fallback_map_pose = bool(ckey("gate_pose_cv_fallback_map_pose", True))

    cmd_use_global_camera_target = bool(ckey("cmd_use_global_camera_target", False))
    cmd_global_cam_pull_alpha = float(ckey("cmd_global_cam_pull_alpha", 0.0))
    cmd_global_cam_weak_mask_scale = float(ckey("cmd_global_cam_weak_mask_scale", 0.35))
    cmd_global_cam_weak_mask_min_pull = float(ckey("cmd_global_cam_weak_mask_min_pull", 0.12))
    cmd_global_cam_portal_min_pull = float(ckey("cmd_global_cam_portal_min_pull", 0.18))
    workspace_camera_width = int(ckey("workspace_camera_width", 256))
    workspace_camera_height = int(ckey("workspace_camera_height", 256))
    workspace_camera_fov = float(ckey("workspace_camera_fov", 55.0))
    cmd_global_cam_min_area_ratio = float(
        ckey("cmd_global_cam_min_area_ratio", gate_pose_cv_min_area_ratio)
    )
    cmd_coarse_plan_once = bool(ckey("cmd_coarse_plan_once", False))
    cmd_coarse_plan_steps = int(ckey("cmd_coarse_plan_steps", 48))
    cmd_coarse_plan_smooth_window = int(ckey("cmd_coarse_plan_smooth_window", 5))
    cmd_precision_zone_enable = bool(ckey("cmd_precision_zone_enable", True))
    cmd_precision_zone_scale = float(ckey("cmd_precision_zone_scale", 1.2))
    cmd_precision_zone_inflate_m = float(ckey("cmd_precision_zone_inflate_m", 0.08))
    cmd_precision_zone_min_r = float(ckey("cmd_precision_zone_min_r", 0.10))
    cmd_precision_infer_every = int(ckey("cmd_precision_infer_every", 1))

    _n2raw = ckey("navigation_phase2_xvla_steps", None)
    if _n2raw is None or (isinstance(_n2raw, str) and not str(_n2raw).strip()):
        navigation_phase2_xvla_steps = max(int(args.xvla_steps), int(cmd_coarse_plan_steps))
    else:
        navigation_phase2_xvla_steps = max(4, int(_n2raw))
    navigation_phase2_sync_root_config = bool(ckey("navigation_phase2_sync_root_config", False))
    navigation_phase2_sync_qs = bool(ckey("navigation_phase2_sync_qs", False))
    navigation_phase2_extra_instruction = str(ckey("navigation_phase2_extra_instruction", "") or "")
    navigation_phase2_geom_astar = bool(ckey("navigation_phase2_geom_astar", True))
    navigation_phase2_astar_cell_m = float(ckey("navigation_phase2_astar_cell_m", 0.04))
    navigation_phase1_corridor_margin_m = float(ckey("navigation_phase1_corridor_margin_m", 0.18))
    navigation_phase1_corridor_bandwidth_m = float(ckey("navigation_phase1_corridor_bandwidth_m", 0.12))
    _ncol_raw = ckey("navigation_collision_pad_m", None)
    navigation_collision_pad_m = (
        float(_ncol_raw) if _ncol_raw is not None else None
    )
    navigation_use_phase3_refined_path_in_sim = bool(
        ckey("navigation_use_phase3_refined_path_in_sim", False)
    )
    navigation_sim_recording_right_distance_scale = float(
        ckey("navigation_sim_recording_right_distance_scale", 1.3)
    )
    navigation_sim_recording_top_distance_scale = float(
        ckey("navigation_sim_recording_top_distance_scale", 1.1)
    )
    navigation_sim_recording_stereo45_distance_scale = float(
        ckey("navigation_sim_recording_stereo45_distance_scale", 0.7)
    )
    _astar_pad_default = float(ckey("local_avoidance_robot_radius", 0.048)) + 0.03
    _pad_raw = ckey("navigation_phase2_astar_obstacle_pad_m", _astar_pad_default)
    if _pad_raw is None:
        navigation_phase2_astar_obstacle_pad_m = float(_astar_pad_default)
    else:
        navigation_phase2_astar_obstacle_pad_m = float(_pad_raw)
    navigation_phase2_optional_topdown_xvla = bool(
        ckey("navigation_phase2_optional_topdown_xvla", False)
    )
    navigation_phase2_z_clearance_enabled = bool(ckey("navigation_phase2_z_clearance_enabled", True))
    navigation_phase2_z_clearance_margin_m = float(ckey("navigation_phase2_z_clearance_margin_m", 0.08))
    navigation_phase2_z_workspace_margin_m = float(ckey("navigation_phase2_z_workspace_margin_m", 0.02))
    navigation_phase3_xvla_action_classify = bool(ckey("navigation_phase3_xvla_action_classify", True))
    _p3steps_raw = ckey("navigation_phase3_xvla_steps", 1)
    navigation_phase3_xvla_steps = max(1, int(_p3steps_raw if _p3steps_raw is not None else 1))
    navigation_phase3_feedback_alpha = float(ckey("navigation_phase3_feedback_alpha", 0.10))
    navigation_recording_use_opengl = bool(ckey("navigation_recording_use_opengl", True))
    cmd_goal_coupled_virtual_base = bool(ckey("cmd_goal_coupled_virtual_base", True))
    vb_smooth_alpha = float(ckey("vb_smooth_alpha", 0.42))
    vb_max_speed_m_s = float(ckey("vb_max_speed_m_s", 4.0))
    vb_jump_warn_m = float(ckey("vb_jump_warn_m", 0.35))
    _qs_sync_cfg = str(ckey("qs_policy_path", "QS.json")).strip() or "QS.json"
    qs_policy_path_for_sync = Path(_qs_sync_cfg)
    if not qs_policy_path_for_sync.is_absolute():
        qs_policy_path_for_sync = ROOT / qs_policy_path_for_sync

    if args.cmd_global_cam and args.no_cmd_global_cam:
        parser.error("Use only one of --cmd-global-cam and --no-cmd-global-cam")
    if args.cmd_global_cam:
        cmd_use_global_camera_target = True
    if args.no_cmd_global_cam:
        cmd_use_global_camera_target = False
    if args.cmd_global_cam_pull is not None:
        cmd_global_cam_pull_alpha = float(args.cmd_global_cam_pull)
        if cmd_global_cam_pull_alpha > 0.0:
            cmd_use_global_camera_target = True

    gate_pose_estimator = str(ckey("gate_pose_estimator", "opencv")).strip().lower()
    if gate_pose_estimator not in ("opencv", "xvla"):
        gate_pose_estimator = "opencv"
    xvla_gate_instruction_template = str(
        ckey(
            "xvla_gate_instruction_template",
            (
                "Orient end-effector with long opening of rectangular portal billboard_id={billboard_id} "
                "({color}); pass through."
            ),
        )
    )
    xvla_gate_steps = int(ckey("xvla_gate_steps", 4))
    xvla_gate_infer_width = int(ckey("xvla_gate_infer_width", 256))
    xvla_gate_infer_height = int(ckey("xvla_gate_infer_height", 256))
    gate_pose_xvla_fallback_opencv = bool(ckey("gate_pose_xvla_fallback_opencv", True))

    infer_every_effective = int(args.infer_every)
    demo_traj_fill_eff = bool(ckey("language_only_demo_trajectory_fill", True))
    demo_xvla_once_eff = bool(ckey("language_only_demo_xvla_trajectory_once", True))
    route_plan: InferenceRoutePlan | None = None

    auto_start_server = bool(ckey("auto_start_xvla_server", True)) and not args.no_auto_server
    server_proc = None
    try:
        if not xvla_server_reachable(args.server_url):
            if auto_start_server:
                h, pnum = parse_act_url(args.server_url)
                if not is_loopback_act_host(h):
                    raise RuntimeError(
                        f"X-VLA not reachable at {args.server_url}. "
                        "Auto-start only applies to localhost; start the service on that host "
                        "or pass --no-auto-server."
                    )
                print(f"[xvla] no server at {args.server_url}; starting local model at http://{h}:{pnum} …")
                server_proc = start_local_xvla_server(
                    h, pnum,
                    access_log=bool(ckey("xvla_server_access_log", False)),
                    log_level=str(ckey("xvla_server_log_level", "warning")),
                )
            else:
                ensure_xvla_server(args.server_url)

        ensure_xvla_server(args.server_url)
        wait_for_xvla_act_inference(
            args.server_url,
            probe_request_timeout_s=float(ckey("xvla_act_probe_request_timeout_s", 300.0)),
            max_wait_s=float(ckey("xvla_act_probe_max_wait_s", 900.0)),
            retry_interval_s=float(ckey("xvla_act_probe_retry_s", 3.0)),
            image_height=int(ckey("xvla_act_probe_image_height", 256)),
            image_width=int(ckey("xvla_act_probe_image_width", 256)),
        )

        router_chat_model = str(ckey("vvla_chat_model", "gpt-4o-mini")).strip() or "gpt-4o-mini"
        router_model_override = str(ckey("xvla_inference_router_chat_model", "")).strip() or None
        router_chat_url = str(ckey("vvla_chat_completions_url", "")).strip() or None
        ck_router = ckey("vvla_chat_api_key", "")
        router_chat_key = (
            (str(ck_router).strip() if ck_router is not None and str(ck_router).strip() else "")
            or os.environ.get("VVLA_CHAT_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        ) or None
        if not router_chat_url and router_chat_key:
            router_chat_url = os.environ.get(
                "VVLA_CHAT_COMPLETIONS_URL", ""
            ).strip() or "https://api.openai.com/v1/chat/completions"
        router_timeout_s = float(ckey("xvla_inference_router_timeout_s", 55.0))
        router_enabled = (
            bool(user_cmd)
            and bool(ckey("xvla_inference_router_enabled", True))
            and (not args.no_inference_router)
            and (not help_only)
        )
        route_plan = resolve_inference_route(
            user_cmd,
            enabled=router_enabled,
            heuristic_only=bool(args.inference_router_heuristic_only),
            chat_url=router_chat_url,
            chat_model=router_chat_model,
            chat_key=router_chat_key,
            router_model=router_model_override,
            timeout=max(10.0, router_timeout_s),
        )
        if route_plan is not None:
            print(
                "[xvla-router] "
                f"source={route_plan.source} mode={route_plan.mode} "
                f"w_xvla={route_plan.w_xvla:.2f} w_local_kb={route_plan.w_local_kb:.2f} "
                f"qs_attach={route_plan.qs_attachment_frac:.2f} "
                f"scenic_geom={route_plan.prefer_scenic_geometry} "
                f"xvla_traj_once={route_plan.xvla_trajectory_once} "
                f"infer_every×={route_plan.infer_every_factor:.3f}"
            )
            infer_every_effective = max(
                1,
                min(
                    512,
                    int(
                        round(
                            float(args.infer_every) * float(route_plan.infer_every_factor)
                        )
                    ),
                ),
            )
            if infer_every_effective != int(args.infer_every):
                print(
                    f"[xvla-router] infer-every {int(args.infer_every)} "
                    f"→ {infer_every_effective}"
                )
            demo_traj_fill_eff = (
                demo_traj_fill_eff and route_plan.prefer_scenic_geometry
            )
            if route_plan.xvla_trajectory_once is not None:
                demo_xvla_once_eff = bool(route_plan.xvla_trajectory_once)

        if user_cmd and (not args.no_vvla) and (not help_only):
            print("[xvla] analyzing TransCMD (QS + v-vla) …")
            qs_cfg = str(ckey("qs_policy_path", "QS.json")).strip() or "QS.json"
            if args.qs_path:
                qs_p = Path(args.qs_path)
            else:
                qs_p = Path(qs_cfg) if Path(qs_cfg).is_absolute() else (ROOT / qs_cfg)
            policies = load_qs_policies(qs_p)
            if policies:
                if args.vvla_url is not None:
                    vvla_http = str(args.vvla_url).strip() or None
                else:
                    vv0 = ckey("vvla_url", "")
                    vvla_http = str(vv0).strip() if vv0 else None
                chat_url = str(ckey("vvla_chat_completions_url", "")).strip() or None
                chat_model = str(ckey("vvla_chat_model", "gpt-4o-mini")).strip() or "gpt-4o-mini"
                ck_cfg = ckey("vvla_chat_api_key", "")
                chat_key = (
                    (str(ck_cfg).strip() if ck_cfg is not None and str(ck_cfg).strip() else "")
                    or os.environ.get("VVLA_CHAT_API_KEY", "").strip()
                    or os.environ.get("OPENAI_API_KEY", "").strip()
                ) or None
                if not chat_url and chat_key:
                    chat_url = os.environ.get(
                        "VVLA_CHAT_COMPLETIONS_URL", ""
                    ).strip() or "https://api.openai.com/v1/chat/completions"
                try:
                    idxs = vvla_select_policy_indices(
                        user_cmd,
                        policies,
                        vvla_http_url=vvla_http,
                        chat_url=chat_url,
                        chat_model=chat_model,
                        chat_api_key=chat_key,
                    )
                except Exception as exc:
                    print(
                        f"[vvla] request failed ({exc}); using raw --cmd for X-VLA",
                        file=sys.stderr,
                    )
                    idxs = []
                if idxs:
                    idxs_use = (
                        _trim_qs_indices(idxs, route_plan.qs_attachment_frac)
                        if route_plan is not None
                        else idxs
                    )
                    if idxs_use != idxs:
                        print(
                            f"[xvla-router] QS indices trimmed {idxs} → {idxs_use} "
                            f"(qs_attachment_frac={route_plan.qs_attachment_frac:.2f})"
                        )
                    if idxs_use:
                        enriched = enrich_cmd_with_qs_policies(
                            user_cmd, policies, idxs_use
                        )
                    else:
                        enriched = user_cmd
                    if enriched.strip() != user_cmd.strip():
                        trans_cmd = enriched.strip()
                        effective_instruction = enriched
                        print(f"[vvla] matched QS indices (v-vla): {idxs_use}")
                        print(f"[===> TransCMD]: {trans_cmd}")
            elif not qs_p.exists():
                print(f"[vvla] policy file missing {qs_p}; skip QS enrichment", file=sys.stderr)
            elif not policies:
                print(f"[vvla] no valid Q/S entries in {qs_p}; skip QS enrichment", file=sys.stderr)

        if user_cmd and effective_instruction == user_cmd:
            print(f"[cmd] X-VLA language_instruction (user --cmd): {effective_instruction!r}")

        _air_pull_raw = ckey("language_only_air_path_max_cam_pull", 0.22)
        if _air_pull_raw is None or (isinstance(_air_pull_raw, str) and not str(_air_pull_raw).strip()):
            language_only_air_path_max_cam_pull = None
        else:
            language_only_air_path_max_cam_pull = float(_air_pull_raw)

        _xvla_act_to = float(
            ckey(
                "xvla_act_request_timeout_s",
                ckey("xvla_act_probe_request_timeout_s", 300.0),
            )
        )
        if args.xvla_act_timeout is not None:
            _xvla_act_to = float(args.xvla_act_timeout)
        xvla_act_request_timeout_s = float(max(30.0, _xvla_act_to))

        xvla_scene_semantic_context_eff = bool(
            ckey("xvla_scene_semantic_context", True)
        ) and not bool(args.no_xvla_scene_catalog)
        local_avoidance_enabled_eff = bool(ckey("local_avoidance_enabled", True)) and not bool(
            args.no_local_avoidance
        )
        _path_suffix_cfg_raw = ckey("xvla_path_planning_instruction_suffix", "")
        _pse = (
            str(_path_suffix_cfg_raw).strip()
            if _path_suffix_cfg_raw is not None and str(_path_suffix_cfg_raw).strip()
            else ""
        )
        xvla_plan_suffix_eff = (
            _pse if _pse != "" else DEFAULT_XVLA_PATH_PLANNING_INSTRUCTION_SUFFIX
        )

        run_widowx_drone_demo(
            server_url=args.server_url,
            instruction=effective_instruction,
            mission_cmd=user_cmd if user_cmd else None,
            sim_steps=args.sim_steps,
            xvla_steps=args.xvla_steps,
            gui=use_gui,
            speed_scale=args.speed,
            infer_every=infer_every_effective,
            dt=args.dt,
            realtime_sleep=realtime,
            log_every=args.log_every,
            workspace_lo=workspace_lo,
            workspace_hi=workspace_hi,
            treat_pos_as=treat_pos_as,
            delta_pos_scale=delta_pos_scale,
            pos_lerp_alpha=pos_lerp_alpha,
            rot_lerp_alpha=rot_lerp_alpha,
            with_objects=with_objects,
            cam_eye=cam_eye,
            cam_look=cam_look,
            record_visualization=record_visualization,
            recording_folder=recording_folder,
            recording_format=recording_fmt,
            recording_fps=max(args.recording_fps, 0.5),
            record_every=int(ckey("record_every", 1)),
            recording_width=int(ckey("recording_width", 640)),
            recording_height=int(ckey("recording_height", 480)),
            task_sequence_raw=raw_seq,
            task_default_phrase=task_default_phrase,
            task_default_arrive_radius=task_default_arrive_radius,
            task_default_dwell_steps=task_default_dwell_steps,
            stop_when_all_visited=stop_when_all_visited,
            cubes_override=cubes_override,
            target_pull_alpha=target_pull_alpha,
            target_snap_radius=target_snap_radius,
            movement_deadband=movement_deadband,
            phase1_stall_window=int(ckey("phase1_stall_window", 20)),
            phase1_stall_threshold=float(ckey("phase1_stall_threshold", 0.005)),
            phase1_max_steps=int(ckey("phase1_max_steps", 45)),
            phase2_p_gain=float(ckey("phase2_p_gain", 4.0)),
            phase2_max_speed=float(ckey("phase2_max_speed", 0.08)),
            recording_scene_fov=float(ckey("recording_scene_fov", 42.0)),
            recording_scene_margin=float(ckey("recording_scene_margin", 1.55)),
            recording_camera_distance_scale=float(ckey("recording_camera_distance_scale", 1.0)),
            recording_topview_distance_scale=float(ckey("recording_topview_distance_scale", 1.22)),
            recording_stereo45_distance_scale=float(ckey("recording_stereo45_distance_scale", 1.5)),
            recording_final_hold_frames=int(ckey("recording_final_hold_frames", 18)),
            trail_enabled=bool(ckey("trail_enabled", True)),
            trail_rgba=list(ckey("trail_rgba", [1.0, 0.68, 0.32, 0.75])),
            trail_radius=float(ckey("trail_radius", 0.006)),
            trail_min_distance=float(ckey("trail_min_distance", 0.01)),
            virtual_workspace_enabled=bool(ckey("virtual_workspace_enabled", True)),
            virtual_workspace_margin=float(ckey("virtual_workspace_margin", 0.02)),
            fpv_slot_align=fpv_slot_align,
            fpv_slot_align_dist=fpv_slot_align_dist,
            fpv_use_sim_truth_pose=fpv_use_sim_truth_pose,
            portal_match_tol=portal_match_tol,
            portal_pass_through_enabled=portal_pass_through_enabled,
            portal_pass_approach_offset=portal_pass_approach_offset,
            portal_pass_exit_offset=portal_pass_exit_offset,
            portal_pass_stage_switch_radius=portal_pass_stage_switch_radius,
            fpv_cam_offset_body=tuple(float(x) for x in fpv_cam_offset_body.tolist()),
            fpv_cam_look_body=tuple(float(x) for x in fpv_cam_look_body.tolist()),
            fpv_cam_width=fpv_cam_width,
            fpv_cam_height=fpv_cam_height,
            fpv_cam_fov=fpv_cam_fov,
            gate_pose_use_dedicated_camera=gate_pose_use_dedicated_camera,
            gate_pose_cam_offset_body=tuple(float(x) for x in gate_pose_cam_offset_body.tolist()),
            gate_pose_cam_look_body=tuple(float(x) for x in gate_pose_cam_look_body.tolist()),
            gate_pose_cam_width=gate_pose_cam_width,
            gate_pose_cam_height=gate_pose_cam_height,
            gate_pose_cam_fov=gate_pose_cam_fov,
            gate_pose_cv_rgb_pad=gate_pose_cv_rgb_pad,
            gate_pose_cv_min_area_ratio=gate_pose_cv_min_area_ratio,
            gate_pose_cv_fallback_map_pose=gate_pose_cv_fallback_map_pose,
            gate_pose_estimator=gate_pose_estimator,
            xvla_gate_instruction_template=xvla_gate_instruction_template,
            xvla_gate_steps=xvla_gate_steps,
            xvla_gate_infer_width=xvla_gate_infer_width,
            xvla_gate_infer_height=xvla_gate_infer_height,
            gate_pose_xvla_fallback_opencv=gate_pose_xvla_fallback_opencv,
            language_only_motion_amplify=language_only_motion_amplify,
            language_only_min_step_local_m=language_only_min_step_local_m,
            cmd_color_disambiguation=cmd_color_disambiguation,
            infer_displacement_scale=infer_displacement_scale,
            cmd_use_global_camera_target=cmd_use_global_camera_target,
            cmd_global_cam_pull_alpha=cmd_global_cam_pull_alpha,
            cmd_global_cam_weak_mask_scale=cmd_global_cam_weak_mask_scale,
            cmd_global_cam_weak_mask_min_pull=cmd_global_cam_weak_mask_min_pull,
            cmd_global_cam_portal_min_pull=cmd_global_cam_portal_min_pull,
            workspace_camera_width=workspace_camera_width,
            workspace_camera_height=workspace_camera_height,
            workspace_camera_fov=workspace_camera_fov,
            cmd_global_cam_min_area_ratio=cmd_global_cam_min_area_ratio,
            cmd_coarse_plan_once=cmd_coarse_plan_once,
            cmd_coarse_plan_steps=cmd_coarse_plan_steps,
            cmd_coarse_plan_smooth_window=cmd_coarse_plan_smooth_window,
            cmd_precision_zone_enable=cmd_precision_zone_enable,
            cmd_precision_zone_scale=cmd_precision_zone_scale,
            cmd_precision_zone_inflate_m=cmd_precision_zone_inflate_m,
            cmd_precision_zone_min_r=cmd_precision_zone_min_r,
            cmd_precision_infer_every=cmd_precision_infer_every,
            language_only_cruise_z_clamp=bool(ckey("language_only_cruise_z_clamp", True)),
            language_only_cruise_z_margin_lo_frac=float(
                ckey("language_only_cruise_z_margin_lo_frac", 0.10)
            ),
            language_only_cruise_z_margin_hi_frac=float(
                ckey("language_only_cruise_z_margin_hi_frac", 0.08)
            ),
            language_only_air_path_max_cam_pull=language_only_air_path_max_cam_pull,
            language_only_demo_trajectory_fill=demo_traj_fill_eff,
            language_only_demo_height_factor=float(ckey("language_only_demo_height_factor", 1.6)),
            language_only_demo_plan_blend_beta=float(
                ckey("language_only_demo_plan_blend_beta", 0.38)
            ),
            language_only_demo_plan_fill_frac=float(
                ckey("language_only_demo_plan_fill_frac", 0.40)
            ),
            language_only_demo_plan_edge_margin_frac=float(
                ckey("language_only_demo_plan_edge_margin_frac", 0.08)
            ),
            language_only_demo_traj_period_infers=int(
                ckey("language_only_demo_traj_period_infers", 32)
            ),
            language_only_demo_xvla_trajectory_once=demo_xvla_once_eff,
            language_only_demo_traj_plan_steps=int(
                ckey("language_only_demo_traj_plan_steps", 64)
            ),
            language_only_demo_plan_use_topdown_camera=bool(
                ckey("language_only_demo_plan_use_topdown_camera", True)
            ),
            language_only_demo_fill_xy_margin_m=float(
                ckey("language_only_demo_fill_xy_margin_m", 0.025)
            ),
            language_only_demo_scenic_xy_force_long_scale=float(
                ckey("language_only_demo_scenic_xy_force_long_scale", 0.85)
            ),
            language_only_demo_clearance_frac_above_tallest=float(
                ckey("language_only_demo_clearance_frac_above_tallest", 0.50)
            ),
            language_only_demo_clearance_above_scene_z_m=float(
                ckey("language_only_demo_clearance_above_scene_z_m", 0.0)
            ),
            language_only_demo_scenic_formula_min_waypoints=int(
                ckey("language_only_demo_scenic_formula_min_waypoints", 512)
            ),
            xvla_scene_semantic_context=xvla_scene_semantic_context_eff,
            xvla_path_planning_instruction_suffix=xvla_plan_suffix_eff,
            local_avoidance_enabled=local_avoidance_enabled_eff,
            local_avoidance_robot_radius=float(ckey("local_avoidance_robot_radius", 0.048)),
            local_avoidance_influence_m=float(ckey("local_avoidance_influence_m", 0.16)),
            local_avoidance_gain=float(ckey("local_avoidance_gain", 0.065)),
            local_avoidance_target_max_shift_m=float(
                ckey("local_avoidance_target_max_shift_m", 0.09)
            ),
            local_avoidance_phase2_gain=float(ckey("local_avoidance_phase2_gain", 0.85)),
            local_avoidance_exclude_goal_tol_m=float(
                ckey("local_avoidance_exclude_goal_tol_m", 0.11)
            ),
            xvla_act_request_timeout_s=xvla_act_request_timeout_s,
            navigation_phase2_xvla_steps=int(navigation_phase2_xvla_steps),
            navigation_phase2_sync_root_config=bool(navigation_phase2_sync_root_config),
            navigation_phase2_sync_qs=bool(navigation_phase2_sync_qs),
            navigation_phase2_extra_instruction=str(navigation_phase2_extra_instruction),
            qs_policy_path_for_sync=qs_policy_path_for_sync,
            navigation_phase2_geom_astar=bool(navigation_phase2_geom_astar),
            navigation_phase2_astar_cell_m=float(navigation_phase2_astar_cell_m),
            navigation_phase1_corridor_margin_m=float(navigation_phase1_corridor_margin_m),
            navigation_phase1_corridor_bandwidth_m=float(navigation_phase1_corridor_bandwidth_m),
            navigation_collision_pad_m=navigation_collision_pad_m,
            navigation_phase2_astar_obstacle_pad_m=float(navigation_phase2_astar_obstacle_pad_m),
            navigation_phase2_optional_topdown_xvla=bool(navigation_phase2_optional_topdown_xvla),
            navigation_phase2_z_clearance_enabled=bool(navigation_phase2_z_clearance_enabled),
            navigation_phase2_z_clearance_margin_m=float(navigation_phase2_z_clearance_margin_m),
            navigation_phase2_z_workspace_margin_m=float(navigation_phase2_z_workspace_margin_m),
            navigation_phase3_xvla_action_classify=bool(navigation_phase3_xvla_action_classify),
            navigation_phase3_xvla_steps=int(navigation_phase3_xvla_steps),
            navigation_phase3_feedback_alpha=float(navigation_phase3_feedback_alpha),
            navigation_use_phase3_refined_path_in_sim=bool(
                navigation_use_phase3_refined_path_in_sim
            ),
            navigation_sim_recording_right_distance_scale=float(
                navigation_sim_recording_right_distance_scale
            ),
            navigation_sim_recording_top_distance_scale=float(
                navigation_sim_recording_top_distance_scale
            ),
            navigation_sim_recording_stereo45_distance_scale=float(
                navigation_sim_recording_stereo45_distance_scale
            ),
            navigation_recording_use_opengl=bool(navigation_recording_use_opengl),
            cmd_goal_coupled_virtual_base=bool(cmd_goal_coupled_virtual_base),
            vb_smooth_alpha=float(vb_smooth_alpha),
            vb_max_speed_m_s=float(vb_max_speed_m_s),
            vb_jump_warn_m=float(vb_jump_warn_m),
        )
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=20)
            except Exception:
                server_proc.kill()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
