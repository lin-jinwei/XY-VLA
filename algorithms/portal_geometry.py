from __future__ import annotations

import math
from typing import Any

import numpy as np


def portal_number_billboard_center_and_size(portal: dict) -> tuple[np.ndarray, float]:

    cached_c = portal.get("portal_label_center")
    cached_s = portal.get("portal_label_side_m")
    if cached_c is not None and cached_s is not None:
        return np.asarray(cached_c, dtype=np.float64).reshape(3), float(cached_s)

    pos = np.asarray(portal["pos"], dtype=np.float64).reshape(3)
    bh = np.asarray(portal.get("bounds_half", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    L = float(portal.get("side", 0.392))
    W = float(portal.get("short_side", 0.2))
    side_m = float(np.clip(2.0 * min(L, W) / 3.0, 0.024, 2.5))
    margin = max(0.008, 0.06 * side_m)
    center = pos + np.array(
        [0.0, 0.0, float(bh[2]) + 0.5 * side_m + margin + 0.002],
        dtype=np.float64,
    )
    return center, side_m


def is_portal_object(c: dict) -> bool:
    sh = str(c.get("shape", "")).lower()
    if sh in ("rect_frame", "square_frame", "frame") or "rect" in sh:
        return True
    if c.get("portal_label") is not None:
        return True
    if "short_side" in c and "side" in c and "thickness" in c:
        return True
    return False


def portal_frame_rotation_matrix(tilt_rad: float, yaw_rad: float) -> np.ndarray:

    cx, sx = math.cos(tilt_rad), math.sin(tilt_rad)
    cz, sz = math.cos(yaw_rad), math.sin(yaw_rad)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ rx


def portal_yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:

    cz, sz = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def portal_rotation_matrix(portal: dict) -> np.ndarray:

    yaw = math.radians(float(portal.get("yaw_deg", 0.0)))
    if portal_is_solid_cuboid(portal):
        return portal_yaw_rotation_matrix(yaw)
    tilt = math.radians(float(portal.get("tilt_deg", 0.0)))
    return portal_frame_rotation_matrix(tilt, yaw)


def portal_opening_axes(portal: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = portal_rotation_matrix(portal)
    nx = rot[:, 0] / max(float(np.linalg.norm(rot[:, 0])), 1e-12)
    ny = rot[:, 1] / max(float(np.linalg.norm(rot[:, 1])), 1e-12)
    nz = rot[:, 2] / max(float(np.linalg.norm(rot[:, 2])), 1e-12)
    return nx, ny, nz


def portal_opening_center_world(portal: dict) -> np.ndarray:
    return np.asarray(portal["pos"], dtype=np.float64).reshape(3)


def portal_solid_world_top_z(portal: dict) -> float:

    top = float("-inf")
    for _lo, hi in portal_bar_world_aabbs(portal):
        top = max(top, float(hi[2]))
    if top > float("-inf"):
        return top
    _, _, hw = portal_cuboid_half_extents_local(portal)
    return float(portal_opening_center_world(portal)[2]) + float(hw)


def portal_solid_pass_over_crossing(
    portal: dict,
    *,
    drone_r: float = 0.07,
    pad: float = 0.07,
) -> np.ndarray:

    c = portal_opening_center_world(portal)
    clearance = float(drone_r) + float(max(0.0, pad)) + 0.05
    z_over = portal_solid_world_top_z(portal) + clearance
    return np.array([c[0], c[1], z_over], dtype=np.float64)


def portal_is_solid_cuboid(portal: dict) -> bool:
    if portal.get("solid_cuboid") is False:
        return False
    if portal.get("portal_geometry") == "frame":
        return False
    return bool(portal.get("solid_cuboid")) or portal.get("cuboid_height") is not None


def portal_cuboid_half_extents_local(portal: dict) -> tuple[float, float, float]:

    L = float(portal.get("side", 0.392))
    W = float(portal.get("short_side", 0.2))
    t = float(portal.get("thickness", 0.026))
    W_eff = float(min(W, max(L - 2.0 * t, t * 0.25)))
    h_cfg = portal.get("cuboid_height")
    if h_cfg is not None:
        height = float(h_cfg)
    else:
        depth = float(portal.get("depth", t))
        height = float(depth)
    return float(L * 0.5), float(W_eff * 0.5), float(height * 0.5)


def portal_opening_half_extents(portal: dict) -> tuple[float, float, float]:

    if portal_is_solid_cuboid(portal):
        return 0.0, 0.0, 0.0
    L = float(portal.get("side", 0.392))
    W = float(portal.get("short_side", 0.2))
    t = float(portal.get("thickness", 0.026))
    depth = float(portal.get("depth", t))
    half_u = 0.5 * depth
    half_v = max(0.5 * L - t, t)
    half_w = max(0.5 * W - t, t)
    return half_u, half_v, half_w


def portal_local_coords(p: np.ndarray, portal: dict) -> tuple[float, float, float]:

    c = portal_opening_center_world(portal)
    nx, ny, nz = portal_opening_axes(portal)
    d = np.asarray(p, dtype=np.float64).reshape(3) - c
    return float(np.dot(d, nx)), float(np.dot(d, ny)), float(np.dot(d, nz))


def point_in_portal_opening(p: np.ndarray, portal: dict, *, margin: float = 0.0) -> bool:

    u, v, w = portal_local_coords(p, portal)
    half_u, half_v, half_w = portal_opening_half_extents(portal)
    m = float(max(0.0, margin))
    return abs(u) <= half_u + m and abs(v) <= half_v + m and abs(w) <= half_w + m


def portal_pass_through_direction(portal: dict, entry: np.ndarray) -> np.ndarray:

    c = portal_opening_center_world(portal)
    nx, _, _ = portal_opening_axes(portal)
    dp = np.asarray(entry, dtype=np.float64).reshape(3) - c
    s = float(np.dot(nx, dp))
    if abs(s) < 1e-8:
        return nx / max(float(np.linalg.norm(nx)), 1e-12)
    out = (-nx if s > 0.0 else nx)
    return out / max(float(np.linalg.norm(out)), 1e-12)


def project_point_to_portal_axis(
    p: np.ndarray,
    portal: dict,
    entry: np.ndarray,
) -> np.ndarray:

    c = portal_opening_center_world(portal)
    n = portal_pass_through_direction(portal, entry)
    d = np.asarray(p, dtype=np.float64).reshape(3) - c
    t = float(np.dot(d, n))
    return c + n * t


def portal_hover_world_position(
    portal: dict,
    *,
    zone_radius: float,
    drone_r: float = 0.07,
    workspace_z_cap_world: float | None = None,
) -> np.ndarray:

    c = portal_opening_center_world(portal)
    diameter = 2.0 * float(max(zone_radius, drone_r))
    nx, ny, nz = portal_opening_axes(portal)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    candidates = [nz, ny, nx, world_up]
    up_dir = max(candidates, key=lambda v: float(np.dot(v, world_up)))
    if float(np.dot(up_dir, world_up)) < 0.0:
        up_dir = -up_dir
    hover = c + up_dir * max(diameter, 2.0 * float(drone_r) + 0.05)
    if workspace_z_cap_world is not None:
        hover[2] = min(float(hover[2]), float(workspace_z_cap_world))
    return hover


def opening_center_deviation_m(segment: np.ndarray, portal: dict) -> float:

    c = portal_opening_center_world(portal)
    pts = np.asarray(segment, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(pts - c.reshape(1, 3), axis=1)))


def expand_corridor_xy(
    corridor_xy: tuple[float, float, float, float],
    margin_m: float,
) -> tuple[float, float, float, float]:
    m = float(max(0.0, margin_m))
    xmin, ymin, xmax, ymax = corridor_xy
    return (float(xmin) - m, float(ymin) - m, float(xmax) + m, float(ymax) + m)


def _rotated_box_world_aabb(
    center_world: np.ndarray,
    half_extents: np.ndarray,
    rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    c = np.asarray(center_world, dtype=np.float64).reshape(3)
    h = np.asarray(half_extents, dtype=np.float64).reshape(3)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * h[0], sy * h[1], sz * h[2]], dtype=np.float64)
                corners.append(c + rot @ local)
    arr = np.stack(corners, axis=0)
    return arr.min(axis=0), arr.max(axis=0)


def portal_bar_world_aabbs(portal: dict) -> list[tuple[np.ndarray, np.ndarray]]:

    pos = np.asarray(portal["pos"], dtype=np.float64).reshape(3)
    rot = portal_rotation_matrix(portal)

    if portal_is_solid_cuboid(portal):
        he = np.asarray(portal_cuboid_half_extents_local(portal), dtype=np.float64)
        return [_rotated_box_world_aabb(pos, he, rot)]

    L = float(portal.get("side", 0.392))
    t = float(portal.get("thickness", 0.026))
    d = float(portal.get("depth", t))
    W = float(portal.get("short_side", 0.2))
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

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for half_extents, offset_local in bars_local:
        he = np.asarray(half_extents, dtype=np.float64)
        off = np.asarray(offset_local, dtype=np.float64)
        center_world = pos + rot @ off
        out.append(_rotated_box_world_aabb(center_world, he, rot))
    return out


def placed_object_aabb_lo_hi_world(c: dict) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
    bh = c.get("bounds_half")
    if bh is not None:
        h = np.asarray(bh, dtype=np.float64).reshape(3)
        return pos - h, pos + h
    half = float(c.get("half", 0.025))
    h = np.array([half, half, half], dtype=np.float64)
    return pos - h, pos + h


def _entry_portal_ref(entry: Any) -> dict | None:
    if isinstance(entry, dict):
        ref = entry.get("ref")
        if isinstance(ref, dict) and is_portal_object(ref):
            return ref
        if is_portal_object(entry):
            return entry
    return None


def obstacle_xy_boxes_from_entries(
    entries: list[Any],
    *,
    inflate_m: float,
    decompose_portals: bool = True,
    exclude_goal_fn: Any | None = None,
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
        portal_ref = _entry_portal_ref(cdict) if decompose_portals else None
        if portal_ref is not None:
            for lo, hi in portal_bar_world_aabbs(portal_ref):
                boxes.append(
                    (
                        float(lo[0]) - pad,
                        float(lo[1]) - pad,
                        float(hi[0]) + pad,
                        float(hi[1]) + pad,
                    )
                )
            continue
        if isinstance(cdict, dict) and "aabb" in cdict:
            lo = np.asarray(cdict["aabb"][0], dtype=np.float64)
            hi = np.asarray(cdict["aabb"][1], dtype=np.float64)
        elif isinstance(cdict, dict):
            lo, hi = placed_object_aabb_lo_hi_world(cdict)
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


def world_aabbs_from_entries(
    entries: list[Any],
    *,
    decompose_portals: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for obj in entries:
        cdict: dict | Any = obj
        if isinstance(obj, tuple) and len(obj) >= 2:
            cdict = obj[-1]
        portal_ref = _entry_portal_ref(cdict) if decompose_portals else None
        if portal_ref is not None:
            out.extend(portal_bar_world_aabbs(portal_ref))
            continue
        if isinstance(cdict, dict) and "aabb" in cdict:
            lo = np.asarray(cdict["aabb"][0], dtype=np.float64)
            hi = np.asarray(cdict["aabb"][1], dtype=np.float64)
        elif isinstance(cdict, dict):
            lo, hi = placed_object_aabb_lo_hi_world(cdict)
        else:
            continue
        out.append((lo, hi))
    return out


def compute_leg_corridor_xy(
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
    *,
    margin_m: float = 0.18,
    bandwidth_m: float = 0.12,
    portal_ref: dict | None = None,
    feedback_radius_m: float = 0.0,
) -> tuple[float, float, float, float]:

    s = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    g = np.asarray(goal_xyz, dtype=np.float64).reshape(3)
    sx, sy = float(s[0]), float(s[1])
    gx, gy = float(g[0]), float(g[1])
    x_min, x_max = min(sx, gx), max(sx, gx)
    y_min, y_max = min(sy, gy), max(sy, gy)

    bw = float(max(0.0, bandwidth_m))
    if bw > 0.0:
        path = np.array([gx - sx, gy - sy], dtype=np.float64)
        plen = float(np.linalg.norm(path))
        if plen > 1e-9:
            perp = np.array([-path[1], path[0]], dtype=np.float64) / plen * bw
            for px, py in ((sx, sy), (gx, gy), (sx + perp[0], sy + perp[1]), (sx - perp[0], sy - perp[1])):
                x_min = min(x_min, float(px))
                x_max = max(x_max, float(px))
                y_min = min(y_min, float(py))
                y_max = max(y_max, float(py))

    pad_fb = float(max(0.0, feedback_radius_m))
    if portal_ref is not None:
        for lo, hi in portal_bar_world_aabbs(portal_ref):
            x_min = min(x_min, float(lo[0]) - pad_fb)
            x_max = max(x_max, float(hi[0]) + pad_fb)
            y_min = min(y_min, float(lo[1]) - pad_fb)
            y_max = max(y_max, float(hi[1]) + pad_fb)
        cen = portal_opening_center_world(portal_ref)
        x_min = min(x_min, float(cen[0]) - pad_fb)
        x_max = max(x_max, float(cen[0]) + pad_fb)
        y_min = min(y_min, float(cen[1]) - pad_fb)
        y_max = max(y_max, float(cen[1]) + pad_fb)

    return expand_corridor_xy((x_min, y_min, x_max, y_max), float(margin_m))


def portal_opening_half_diagonal(portal: dict) -> float:
    if portal_is_solid_cuboid(portal):
        hu, hv, hw = portal_cuboid_half_extents_local(portal)
        return float(math.sqrt(hu * hu + hv * hv + hw * hw))
    L = float(portal.get("side", 0.392))
    W = float(portal.get("short_side", 0.2))
    t = float(portal.get("thickness", 0.026))
    W_eff = float(min(W, max(L - 2.0 * t, t * 0.25)))
    return float(math.hypot(0.5 * L, 0.5 * W_eff))


def portal_frame_surface_point(
    portal: dict,
    from_xyz: np.ndarray,
    *,
    drone_r: float = 0.07,
    pad: float = 0.0,
) -> np.ndarray:

    c = portal_opening_center_world(portal)
    e = np.asarray(from_xyz, dtype=np.float64).reshape(3)
    nx, _, _ = portal_opening_axes(portal)
    dp = e - c
    s = float(np.dot(nx, dp))
    n = (-nx if s > 0.0 else nx) if abs(s) > 1e-8 else nx
    if portal_is_solid_cuboid(portal):
        half_u, _, _ = portal_cuboid_half_extents_local(portal)
        offset = max(float(half_u), float(drone_r) * 0.5) + float(pad)
    else:
        depth = float(portal.get("depth", portal.get("thickness", 0.026)))
        offset = max(float(depth) * 0.5, float(drone_r) * 0.5) + float(pad)
    return c + n * offset


def unified_navigation_collision_pad_m(
    *,
    astar_obstacle_pad_m: float,
    phase3_collision_pad_m: float | None = None,
) -> float:

    p2 = float(max(0.0, astar_obstacle_pad_m))
    if phase3_collision_pad_m is None:
        return p2
    return float(max(p2, float(max(0.0, phase3_collision_pad_m))))


def _closest_point_on_aabb(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64).reshape(3), lo, hi)


def sphere_hits_inflated_aabb(
    p: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    radius: float,
    pad: float,
) -> bool:
    lo_i = np.asarray(lo, dtype=np.float64).reshape(3) - float(pad)
    hi_i = np.asarray(hi, dtype=np.float64).reshape(3) + float(pad)
    q = _closest_point_on_aabb(p, lo_i, hi_i)
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64).reshape(3) - q)) < float(radius) - 1e-9


def segment_capsule_hits_aabb(
    p0: np.ndarray,
    p1: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    radius: float,
    pad: float,
    *,
    samples: int = 16,
) -> bool:

    a = np.asarray(p0, dtype=np.float64).reshape(3)
    b = np.asarray(p1, dtype=np.float64).reshape(3)
    dist = float(np.linalg.norm(b - a))
    n = max(2, int(samples))
    if dist > 0.2:
        n = max(n, int(math.ceil(dist / 0.035)))
    for t in np.linspace(0.0, 1.0, n):
        if sphere_hits_inflated_aabb(a + t * (b - a), lo, hi, radius, pad):
            return True
    return False


def segment_capsule_collision_free(
    p0: np.ndarray,
    p1: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    radius: float,
    pad: float,
) -> bool:
    for lo, hi in aabbs:
        if segment_capsule_hits_aabb(p0, p1, lo, hi, radius, pad):
            return False
    return True


def min_z_clearance_for_xy_segment_aabbs(
    p0: np.ndarray,
    p1: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    *,
    xy_pad: float,
) -> float:

    z_need = max(float(p0[2]), float(p1[2]))
    hit = False
    sx, sy = float(p0[0]), float(p0[1])
    gx, gy = float(p1[0]), float(p1[1])
    pad = float(max(0.0, xy_pad))
    for lo, hi in aabbs:
        lo2 = lo.copy()
        hi2 = hi.copy()
        lo2[0] -= pad
        lo2[1] -= pad
        hi2[0] += pad
        hi2[1] += pad
        if _segment_intersects_rect_2d(sx, sy, gx, gy, float(lo2[0]), float(lo2[1]), float(hi2[0]), float(hi2[1])):
            z_need = max(z_need, float(hi[2]))
            hit = True
    return z_need if hit else max(float(p0[2]), float(p1[2]))


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
    edges = ((a, c, b, c), (b, c, b, d), (b, d, a, d), (a, d, a, c))
    for ex0, ey0, ex1, ey1 in edges:
        o1 = (ey0 - y0) * (x1 - x0) - (ex0 - x0) * (y1 - y0)
        o2 = (ey1 - y0) * (x1 - x0) - (ex1 - x0) * (y1 - y0)
        o3 = (y0 - ey0) * (ex1 - ex0) - (x0 - ex0) * (ey1 - ey0)
        o4 = (y1 - ey0) * (ex1 - ex0) - (x1 - ex0) * (ey1 - ey0)
        if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
            o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
        ):
            return True
    return False
