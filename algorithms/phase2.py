from __future__ import annotations

import datetime
import heapq
import json
import math
from collections import deque
from typing import Any, Callable

import numpy as np

from .portal_geometry import (
    min_z_clearance_for_xy_segment_aabbs,
    obstacle_xy_boxes_from_entries,
    unified_navigation_collision_pad_m,
    world_aabbs_from_entries,
)


def _placed_object_aabb_lo_hi_world(c: dict) -> tuple[np.ndarray, np.ndarray]:

    pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
    bh = c.get("bounds_half")
    if bh is not None:
        h = np.asarray(bh, dtype=np.float64).reshape(3)
        return pos - h, pos + h
    half = float(c.get("half", 0.025))
    h = np.array([half, half, half], dtype=np.float64)
    return pos - h, pos + h


def obstacle_xy_boxes_phase1_entries(
    entries: list[Any],
    *,
    inflate_m: float,
    exclude_goal_fn: Callable[[Any], bool] | None = None,
    decompose_portals: bool = True,
) -> list[tuple[float, float, float, float]]:

    return obstacle_xy_boxes_from_entries(
        entries,
        inflate_m=float(inflate_m),
        decompose_portals=bool(decompose_portals),
        exclude_goal_fn=exclude_goal_fn,
    )


_ASTAR_NEIGHBORS_8: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _nearest_free_grid_cell(
    ix: int,
    iy: int,
    wx: float,
    wy: float,
    *,
    nx: int,
    ny: int,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    cell_blocked: Callable[[int, int], bool],
    max_bfs_visit: int = 8192,
) -> tuple[int, int]:

    if not cell_blocked(ix, iy):
        return ix, iy
    best: tuple[int, int] | None = None
    best_d2 = float("inf")
    q: deque[tuple[int, int]] = deque([(ix, iy)])
    seen = {(ix, iy)}
    visits = 0
    while q and visits < max_bfs_visit:
        ci, cj = q.popleft()
        visits += 1
        if not cell_blocked(ci, cj):
            cx = xmin + (ci + 0.5) * dx
            cy = ymin + (cj + 0.5) * dy
            d2 = (cx - wx) ** 2 + (cy - wy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (ci, cj)
            continue
        for di, dj in _ASTAR_NEIGHBORS_8:
            ni, nj = ci + di, cj + dj
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                continue
            if (ni, nj) in seen:
                continue
            seen.add((ni, nj))
            q.append((ni, nj))
    return best if best is not None else (ix, iy)


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
        return _nearest_free_grid_cell(
            ix,
            iy,
            wx,
            wy,
            nx=nx,
            ny=ny,
            xmin=xmin,
            ymin=ymin,
            dx=dx,
            dy=dy,
            cell_blocked=cell_blocked,
        )

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

    neighbors = _ASTAR_NEIGHBORS_8

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


def phase2_apply_z_profile_to_xy_path(
    xy_path: list[tuple[float, float]],
    start_xyz_world: np.ndarray,
    goal_xyz_world: np.ndarray,
    world_aabbs: list[tuple[np.ndarray, np.ndarray]],
    *,
    inflate_xy_m: float,
    z_clearance_margin_m: float,
    z_workspace_cap_world: float | None = None,
) -> np.ndarray | None:

    if not xy_path:
        return None
    sw = np.asarray(start_xyz_world, dtype=np.float64).reshape(3)
    gw = np.asarray(goal_xyz_world, dtype=np.float64).reshape(3)
    pad = float(max(0.0, inflate_xy_m))
    zm = float(max(0.0, z_clearance_margin_m))
    raw: list[np.ndarray] = [sw.copy()]

    def _cap_z(z: float) -> float:
        zf = float(z)
        if z_workspace_cap_world is not None:
            zf = min(zf, float(z_workspace_cap_world))
        return zf

    def add_pt(x: float, y: float, z: float) -> None:
        p = np.array([x, y, _cap_z(z)], dtype=np.float64)
        if float(np.linalg.norm(p - raw[-1])) < 1e-4:
            return
        raw.append(p)

    def cruise_z_for_leg(p0: np.ndarray, x: float, y: float, *, end_z_hint: float) -> float:
        seg_end = np.array([x, y, float(end_z_hint)], dtype=np.float64)
        z_seg = min_z_clearance_for_xy_segment_aabbs(p0, seg_end, world_aabbs, xy_pad=pad)
        return _cap_z(max(float(p0[2]), z_seg + zm))

    for i, (x, y) in enumerate(xy_path):
        end_z = float(gw[2]) if i == len(xy_path) - 1 else float(raw[-1][2])
        z_cruise = cruise_z_for_leg(raw[-1], x, y, end_z_hint=end_z)
        prev = raw[-1]
        if abs(z_cruise - float(prev[2])) > 1e-3:
            add_pt(float(prev[0]), float(prev[1]), z_cruise)
        add_pt(x, y, z_cruise)

    if float(np.linalg.norm(raw[-1][:2] - gw[:2])) > 1e-4 or abs(raw[-1][2] - gw[2]) > 1e-3:
        z_out = cruise_z_for_leg(raw[-1], float(gw[0]), float(gw[1]), end_z_hint=float(gw[2]))
        z_out = _cap_z(max(float(gw[2]), z_out))
        if abs(z_out - float(raw[-1][2])) > 1e-3:
            add_pt(float(raw[-1][0]), float(raw[-1][1]), z_out)
        add_pt(float(gw[0]), float(gw[1]), float(gw[2]))
    else:
        raw[-1] = np.array([float(gw[0]), float(gw[1]), _cap_z(float(gw[2]))], dtype=np.float64)

    return np.stack(raw, axis=0).astype(np.float32)


def phase2_trajectory_xyz_from_astar(
    corridor_xy: tuple[float, float, float, float],
    start_xyz_world: np.ndarray,
    goal_xyz_world: np.ndarray,
    obstacle_boxes_xy: list[tuple[float, float, float, float]],
    *,
    cell_m: float,
    world_aabbs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    inflate_xy_m: float = 0.07,
    z_clearance_margin_m: float = 0.08,
    z_workspace_cap_world: float | None = None,
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
    if world_aabbs:
        prof = phase2_apply_z_profile_to_xy_path(
            xy_path,
            sw,
            gw,
            world_aabbs,
            inflate_xy_m=float(inflate_xy_m),
            z_clearance_margin_m=float(z_clearance_margin_m),
            z_workspace_cap_world=z_workspace_cap_world,
        )
        if prof is not None and prof.shape[0] >= 1:
            return prof
    n = len(xy_path)
    sz = float(sw[2])
    gz = float(gw[2])
    out = np.zeros((n, 3), dtype=np.float32)
    for i, (x, y) in enumerate(xy_path):
        u = float(i) / float(max(1, n - 1))
        out[i] = np.array([x, y, (1.0 - u) * sz + u * gz], dtype=np.float32)
    return out


def _xy_segment_hits_obstacle_boxes(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    obstacle_boxes_xy: list[tuple[float, float, float, float]],
) -> bool:
    sx, sy = float(start_xy[0]), float(start_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    for ox0, oy0, ox1, oy1 in obstacle_boxes_xy:
        if _segment_intersects_rect_2d(sx, sy, gx, gy, ox0, oy0, ox1, oy1):
            return True
    return False


def _corridor_clip_xy(
    x: float,
    y: float,
    corridor_xy: tuple[float, float, float, float],
) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = corridor_xy
    lo_x, hi_x = float(min(xmin, xmax)), float(max(xmin, xmax))
    lo_y, hi_y = float(min(ymin, ymax)), float(max(ymin, ymax))
    return (
        float(np.clip(x, lo_x, hi_x)),
        float(np.clip(y, lo_y, hi_y)),
    )


def _corridor_subgoal_xy_chain(
    corridor_xy: tuple[float, float, float, float],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    n_interior: int,
) -> list[tuple[float, float]]:

    sx, sy = float(start_xy[0]), float(start_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    out: list[tuple[float, float]] = [(sx, sy)]
    n = max(1, int(n_interior))
    for k in range(1, n + 1):
        t = float(k) / float(n + 1)
        x, y = _corridor_clip_xy(sx + t * (gx - sx), sy + t * (gy - sy), corridor_xy)
        if math.hypot(x - out[-1][0], y - out[-1][1]) > 1e-4:
            out.append((x, y))
    gx_c, gy_c = _corridor_clip_xy(gx, gy, corridor_xy)
    if math.hypot(gx_c - out[-1][0], gy_c - out[-1][1]) > 1e-4:
        out.append((gx_c, gy_c))
    return out


def _stitch_xy_paths(segments: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for seg in segments:
        if not seg:
            continue
        if not merged:
            merged.extend(seg)
            continue
        for pt in seg:
            if math.hypot(pt[0] - merged[-1][0], pt[1] - merged[-1][1]) > 1e-4:
                merged.append(pt)
    return merged


def phase2_trajectory_xyz_subgoal_astar(
    corridor_xy: tuple[float, float, float, float],
    start_xyz_world: np.ndarray,
    goal_xyz_world: np.ndarray,
    obstacle_boxes_xy: list[tuple[float, float, float, float]],
    *,
    cell_m: float,
    world_aabbs: list[tuple[np.ndarray, np.ndarray]] | None,
    inflate_xy_m: float,
    z_clearance_margin_m: float,
    z_workspace_cap_world: float | None,
    n_interior: int = 3,
) -> np.ndarray | None:

    sw = np.asarray(start_xyz_world, dtype=np.float64).reshape(-1)
    gw = np.asarray(goal_xyz_world, dtype=np.float64).reshape(-1)
    chain = _corridor_subgoal_xy_chain(
        corridor_xy,
        (float(sw[0]), float(sw[1])),
        (float(gw[0]), float(gw[1])),
        n_interior=int(n_interior),
    )
    if len(chain) < 2:
        return None
    xy_legs: list[list[tuple[float, float]]] = []
    for i in range(len(chain) - 1):
        leg_xy = plan_xy_astar_corridor_world(
            corridor_xy,
            chain[i],
            chain[i + 1],
            obstacle_boxes_xy,
            cell_m=float(cell_m),
        )
        if leg_xy is None or len(leg_xy) < 1:
            return None
        xy_legs.append(leg_xy)
    xy_path = _stitch_xy_paths(xy_legs)
    if len(xy_path) < 2:
        return None
    if world_aabbs:
        prof = phase2_apply_z_profile_to_xy_path(
            xy_path,
            sw,
            gw,
            world_aabbs,
            inflate_xy_m=float(inflate_xy_m),
            z_clearance_margin_m=float(z_clearance_margin_m),
            z_workspace_cap_world=z_workspace_cap_world,
        )
        if prof is not None and prof.shape[0] >= 1:
            return prof
    n = len(xy_path)
    sz, gz = float(sw[2]), float(gw[2])
    out = np.zeros((n, 3), dtype=np.float32)
    for i, (x, y) in enumerate(xy_path):
        u = float(i) / float(max(1, n - 1))
        out[i] = np.array([x, y, (1.0 - u) * sz + u * gz], dtype=np.float32)
    return out


def phase2_plan_trajectory_cascade(
    *,
    corridor_xy: tuple[float, float, float, float],
    start_xyz_world: np.ndarray,
    goal_xyz_world: np.ndarray,
    obstacle_boxes_xy: list[tuple[float, float, float, float]],
    world_aabbs: list[tuple[np.ndarray, np.ndarray]],
    astar_cell_m: float,
    collision_pad: float,
    z_clearance_margin_m: float,
    z_workspace_cap_world: float,
    z_clearance_enabled: bool,
    phase1_obstacle_entries: list[Any],
) -> tuple[np.ndarray | None, str, int, dict[str, Any] | None]:

    pw = np.asarray(start_xyz_world, dtype=np.float64).reshape(-1)
    gv = np.asarray(goal_xyz_world, dtype=np.float64).reshape(-1)
    attempts = 0
    z_meta: dict[str, Any] | None = None

    common_kw = dict(
        corridor_xy=corridor_xy,
        start_xyz_world=pw,
        goal_xyz_world=gv,
        obstacle_boxes_xy=obstacle_boxes_xy,
        world_aabbs=world_aabbs,
        inflate_xy_m=float(collision_pad),
        z_clearance_margin_m=float(z_clearance_margin_m),
        z_workspace_cap_world=float(z_workspace_cap_world),
    )

    cell_steps = (
        float(astar_cell_m),
        float(astar_cell_m) * 2.0,
        float(astar_cell_m) * 4.0,
    )
    planner_names = (
        "astar_corridor_xy_zprofile",
        "astar_corridor_coarse_zprofile",
        "astar_corridor_xcoarse_zprofile",
    )
    for cell_m, pname in zip(cell_steps, planner_names):
        attempts += 1
        traj = phase2_trajectory_xyz_from_astar(**common_kw, cell_m=float(cell_m))
        if traj is not None and traj.shape[0] >= 1:
            return traj, pname, attempts, z_meta

    pad_retry = float(max(collision_pad * 0.85, collision_pad - 0.012))
    if pad_retry + 1e-9 < float(collision_pad):
        boxes_retry = obstacle_xy_boxes_phase1_entries(
            phase1_obstacle_entries,
            inflate_m=pad_retry,
            exclude_goal_fn=None,
        )
        attempts += 1
        traj = phase2_trajectory_xyz_from_astar(
            **{**common_kw, "obstacle_boxes_xy": boxes_retry, "inflate_xy_m": pad_retry},
            cell_m=float(astar_cell_m) * 2.0,
        )
        if traj is not None and traj.shape[0] >= 1:
            return traj, "astar_corridor_coarse_reduced_pad", attempts, z_meta

    if z_clearance_enabled:
        got = phase2_trajectory_xyz_z_clearance_overfly(
            pw,
            gv,
            phase1_obstacle_entries,
            inflate_xy_m=float(collision_pad),
            z_clearance_margin_m=float(z_clearance_margin_m),
            z_workspace_cap_world=float(z_workspace_cap_world),
        )
        if got is not None:
            traj, z_meta = got
            if traj.shape[0] >= 1:
                return traj, "z_clearance_overfly_xy", attempts + 1, z_meta

    for n_sub, pname in ((2, "astar_subgoal_2_xy_zprofile"), (4, "astar_subgoal_4_xy_zprofile")):
        attempts += 1
        traj = phase2_trajectory_xyz_subgoal_astar(
            **common_kw,
            cell_m=float(astar_cell_m) * 2.0,
            n_interior=int(n_sub),
        )
        if traj is not None and traj.shape[0] >= 1:
            return traj, pname, attempts, z_meta

    sx, sy = float(pw[0]), float(pw[1])
    gx, gy = float(gv[0]), float(gv[1])
    if not _xy_segment_hits_obstacle_boxes((sx, sy), (gx, gy), obstacle_boxes_xy):
        attempts += 1
        if z_clearance_enabled:
            got = phase2_trajectory_xyz_z_clearance_overfly(
                pw,
                gv,
                phase1_obstacle_entries,
                inflate_xy_m=float(collision_pad),
                z_clearance_margin_m=float(z_clearance_margin_m),
                z_workspace_cap_world=float(z_workspace_cap_world),
            )
            if got is not None:
                traj, z_meta = got
                if traj.shape[0] >= 1:
                    return traj, "los_clear_z_profile", attempts, z_meta
        traj = np.stack(
            [
                np.array([sx, sy, float(pw[2])], dtype=np.float32),
                np.array([gx, gy, float(gv[2])], dtype=np.float32),
            ],
            axis=0,
        )
        return traj, "los_clear_xy_only", attempts, z_meta

    return None, "plan_failed", attempts, z_meta


def phase1_entries_world_aabbs(entries: list[Any]) -> list[tuple[np.ndarray, np.ndarray]]:

    return world_aabbs_from_entries(entries, decompose_portals=True)


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


def run_navigation_phase2_topdown_xvla_xy_plan(
    *,
    server_url: str,
    topdown_rgb: np.ndarray,
    img_marked_bgr: np.ndarray,
    view_m: list[float],
    proj_m: list[float],
    render_width: int,
    render_height: int,
    phase2_folder: Any,
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
    qs_policy_path: Any | None,
    config_json_path: Any,
    phase2_extra_instruction: str,
    phase1_obstacle_entries: list[Any],
    world_xyz_to_recording_image_pixel_fn: Callable[[np.ndarray, list[float], list[float], int, int], tuple[int, int] | None],
    build_proprio_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    query_xvla_fn: Callable[..., Any],
    compose_instruction_fn: Callable[..., str],
    read_config_fn: Callable[[Any], dict[str, Any]],
    load_qs_policies_fn: Callable[[Any], list[dict[str, str]]],
    navigation_phase2_geom_astar: bool = True,
    navigation_phase2_astar_cell_m: float = 0.04,
    navigation_phase2_astar_obstacle_pad_m: float = 0.07,
    navigation_phase2_optional_topdown_xvla: bool = False,
    navigation_phase2_z_clearance_enabled: bool = True,
    navigation_phase2_z_clearance_margin_m: float = 0.08,
    navigation_phase2_z_workspace_margin_m: float = 0.02,
    navigation_collision_pad_m: float | None = None,
    phase2_sidecar_name: str = "navigation_phase2_xy.json",
    leg_index: int | None = None,
    billboard_id: int | None = None,
    pause_message: bool = True,
    placed_cubes: list[dict] | None = None,
) -> dict[str, Any]:

    import cv2

    xmin, ymin, xmax, ymax = corridor_xy
    pw = np.asarray(drone_pos_world, dtype=np.float64).reshape(-1)
    sx, sy, sz = float(pw[0]), float(pw[1]), float(pw[2])
    gv = np.asarray(goal_xyz_world, dtype=np.float64).reshape(-1)
    gx, gy, gz = float(gv[0]), float(gv[1]), float(gv[2])

    mission_txt = str(mission_cmd).strip() if mission_cmd else "(no --cmd)"

    collision_pad = unified_navigation_collision_pad_m(
        astar_obstacle_pad_m=float(navigation_phase2_astar_obstacle_pad_m),
        phase3_collision_pad_m=navigation_collision_pad_m,
    )
    world_aabbs_3d = phase1_entries_world_aabbs(phase1_obstacle_entries)

    obstacle_boxes_xy = obstacle_xy_boxes_phase1_entries(
        phase1_obstacle_entries,
        inflate_m=float(collision_pad),
        exclude_goal_fn=None,
    )

    vbz = float(np.asarray(virtual_base_world, dtype=np.float64).reshape(-1)[2])
    z_cap_world = float(workspace_hi[2]) + vbz - float(navigation_phase2_z_workspace_margin_m)

    pts_arr: np.ndarray | None = None
    infer_ms: float | None = None
    planner_primary = "none"
    z_clearance_meta: dict[str, Any] | None = None
    astar_attempts = 0
    steps = max(4, int(phase2_steps))

    def _xy_segment_underclear_at_cruise() -> bool:
        aabbs = phase1_entries_world_aabbs(phase1_obstacle_entries)
        z_top_along, n_hit = min_z_clearance_for_world_xy_segment(
            (sx, sy), (gx, gy), aabbs, inflate_xy_m=float(collision_pad)
        )
        if n_hit <= 0:
            return False
        need = float(z_top_along) + float(max(0.0, navigation_phase2_z_clearance_margin_m))
        return max(sz, gz) < need - 1e-6

    if navigation_phase2_geom_astar:
        pts_arr, planner_primary, astar_attempts, z_clearance_meta = phase2_plan_trajectory_cascade(
            corridor_xy=corridor_xy,
            start_xyz_world=pw,
            goal_xyz_world=gv,
            obstacle_boxes_xy=obstacle_boxes_xy,
            world_aabbs=world_aabbs_3d,
            astar_cell_m=float(navigation_phase2_astar_cell_m),
            collision_pad=float(collision_pad),
            z_clearance_margin_m=float(navigation_phase2_z_clearance_margin_m),
            z_workspace_cap_world=float(z_cap_world),
            z_clearance_enabled=bool(navigation_phase2_z_clearance_enabled),
            phase1_obstacle_entries=phase1_obstacle_entries,
        )
        if pts_arr is not None and pts_arr.shape[0] >= 1:
            print(
                f"\n[Phase 2] Cascade planner={planner_primary} attempts={astar_attempts} "
                f"pad={collision_pad:.3f}m waypoints={pts_arr.shape[0]}"
            )
        else:
            pts_arr = None
            planner_primary = "plan_failed"
            print(
                "\n[Phase 2] WARN: geometry cascade failed (A* / Z-overfly / subgoals) - "
                "no start->goal fallback; upstream may use action geometry."
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
        print("[Phase 2] navigation_phase2_geom_astar=false - using LOS start->goal only.")
        if _xy_segment_hits_obstacle_boxes((sx, sy), (gx, gy), obstacle_boxes_xy):
            print("[Phase 2] WARN: LOS segment intersects inflated obstacles.")
        elif (
            bool(navigation_phase2_z_clearance_enabled)
            and _xy_segment_underclear_at_cruise()
        ):
            got = phase2_trajectory_xyz_z_clearance_overfly(
                pw,
                gv,
                phase1_obstacle_entries,
                inflate_xy_m=float(collision_pad),
                z_clearance_margin_m=float(navigation_phase2_z_clearance_margin_m),
                z_workspace_cap_world=float(z_cap_world),
            )
            if got is not None:
                pts_arr, z_clearance_meta = got
                planner_primary = "z_clearance_geom_astar_disabled"

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

        composed = compose_instruction_fn(
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
        proprio = build_proprio_fn(
            np.asarray(drone_pos_local, dtype=np.float32),
            np.asarray(drone_R, dtype=np.float32),
            float(gripper_state),
        )
        print("\n[Phase 2] Optional top-down X-VLA /act (does not replace A* polyline unless integrated downstream) ...")
        print(f"[Phase 2] xvla_steps={steps} infer_timeout_s={xvla_act_request_timeout_s:.0f}")
        try:
            t_req = datetime.datetime.now().timestamp()
            actions = query_xvla_fn(
                server_url,
                td,
                proprio,
                composed,
                steps=steps,
                timeout=float(xvla_act_request_timeout_s),
            )
            _ = actions
            infer_ms = float((datetime.datetime.now().timestamp() - t_req) * 1000.0)
            print(f"[Phase 2] optional top-down infer_ms={infer_ms:.1f}")
        except Exception as exc:
            print(f"[Phase 2] optional top-down X-VLA failed: {exc}")

    if pts_arr is not None and pts_arr.shape[0] >= 1:
        print("[Phase 2] XY keypoints - world frame (meters), primary planner:")
        print(f"  planner={planner_primary}")
        for i, row in enumerate(pts_arr):
            print(
                f"  k{i}: XY=[{float(row[0]):.6f}, {float(row[1]):.6f}]  Z={float(row[2]):.6f}"
            )

        pix_ln: list[tuple[int, int]] = []
        for row in pts_arr:
            wx, wy, wz = float(row[0]), float(row[1]), float(row[2])
            pix = world_xyz_to_recording_image_pixel_fn(
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

        if placed_cubes:

            def _wproj_phase2(wx: float, wy: float, wz: float) -> tuple[int, int] | None:
                return world_xyz_to_recording_image_pixel_fn(
                    np.array([wx, wy, wz], dtype=np.float64),
                    view_m,
                    proj_m,
                    width=int(render_width),
                    height=int(render_height),
                )

            from .phase_recording import draw_portal_number_labels_overlay_bgr

            draw_portal_number_labels_overlay_bgr(img_marked_bgr, _wproj_phase2, placed_cubes)

        print(f"[Phase 2] phase2 recording folder: {phase2_folder}")
        out_png = phase2_folder / png_filename
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
                pix = world_xyz_to_recording_image_pixel_fn(
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
            st_path = phase2_folder / st_name
            cv2.imwrite(str(st_path), st_bgr)
            stereo_saved = str(st_path)
            print(f"[Phase 2] Saved stereo (45deg) PNG with blue (BGR {blue_bgr}) path: {st_path}")

        snapshot = {
            "phase": "navigation_phase2_xy",
            "leg_index": leg_index,
            "billboard_id": billboard_id,
            "primary_planner": planner_primary,
            "geom_astar_enabled": bool(navigation_phase2_geom_astar),
            "optional_topdown_xvla": bool(navigation_phase2_optional_topdown_xvla),
            "astar_cell_m": float(navigation_phase2_astar_cell_m),
            "astar_obstacle_pad_m": float(navigation_phase2_astar_obstacle_pad_m),
            "collision_pad_unified_m": float(collision_pad),
            "astar_attempts": int(astar_attempts),
            "waypoint_count": int(pts_arr.shape[0]),
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
        sidecar_name = str(phase2_sidecar_name).strip() or "navigation_phase2_xy.json"
        sidecar = phase2_folder / sidecar_name
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        print(f"[Phase 2] Sidecar JSON: {sidecar}")

        if sync_root_config:
            try:
                cfg = read_config_fn(config_json_path)
                cfg["_navigation_phase2_snapshot"] = snapshot
                with open(config_json_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                print(f"[Phase 2] Wrote `_navigation_phase2_snapshot` into {config_json_path}")
            except Exception as exc:
                print(f"[Phase 2] WARNING: could not merge config.json - {exc}")

        if sync_qs and qs_policy_path is not None:
            try:
                policies = load_qs_policies_fn(qs_policy_path)
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
                print(f"[Phase 2] WARNING: could not append QS.json - {exc}")

        if pause_message:
            print("[Phase 2] Completed.\n")
        return {
            "trajectory": pts_arr,
            "snapshot": snapshot,
            "planner": planner_primary,
        }

    if pause_message:
        print("[Phase 2] No drawable trajectory; screenshot not updated for polyline.")
        print("[Phase 2] Completed.\n")
    return {"trajectory": None, "snapshot": None, "planner": planner_primary}
