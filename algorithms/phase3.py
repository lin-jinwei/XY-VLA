from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .phase3_actions import (
    BasicAction,
    compute_path_quality_metrics,
    entry_key,
    nearest_trajectory_index,
    refine_trajectory_at_zone_contact,
    resolve_basic_action_for_zone,
)

PHASE3_RGBA_ALPHA = 0.10
PHASE3_RGBA_IDLE = [0.68, 0.85, 1.0, PHASE3_RGBA_ALPHA]
PHASE3_RGBA_ENTERED = [1.0, 0.71, 0.76, PHASE3_RGBA_ALPHA]

PHASE3_ACTION_PATH_BGR: dict[BasicAction, tuple[int, int, int]] = {
    BasicAction.PASS_THROUGH: (255, 200, 0),
    BasicAction.FLY_BY: (0, 230, 255),
    BasicAction.ORBIT: (255, 48, 255),
    BasicAction.COLLISION: (0, 0, 255),
    BasicAction.HOVER: (0, 165, 255),
}


def set_phase3_feedback_alpha(alpha: float) -> None:

    global PHASE3_RGBA_ALPHA, PHASE3_RGBA_IDLE, PHASE3_RGBA_ENTERED
    a = float(max(0.004, min(0.08, alpha)))
    PHASE3_RGBA_ALPHA = a
    PHASE3_RGBA_IDLE = [0.68, 0.85, 1.0, a]
    PHASE3_RGBA_ENTERED = [1.0, 0.71, 0.76, a]


def create_feedback_sphere_visual_shape(
    p: Any,
    radius: float,
    rgba: list[float] | tuple[float, float, float, float],
) -> int:

    flags = 0
    if hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED"):
        flags = int(p.VISUAL_SHAPE_DOUBLE_SIDED)
    try:
        return int(
            p.createVisualShape(
                p.GEOM_SPHERE,
                radius=float(radius),
                rgbaColor=list(rgba),
                specularColor=[0.0, 0.0, 0.0, 0.0],
                flags=flags,
            )
        )
    except TypeError:
        return int(
            p.createVisualShape(
                p.GEOM_SPHERE,
                radius=float(radius),
                rgbaColor=list(rgba),
                flags=flags,
            )
        )


def apply_feedback_sphere_visual(p: Any, zone: "Phase3FeedbackZone") -> None:

    if zone.body_uid is None:
        return
    rgba = _feedback_sphere_rgba(zone)
    uid = int(zone.body_uid)
    try:
        p.changeVisualShape(
            uid,
            -1,
            rgbaColor=rgba,
            specularColor=[0.0, 0.0, 0.0, 0.0],
        )
    except TypeError:
        p.changeVisualShape(uid, -1, rgbaColor=rgba)
    except Exception:
        pass


def _feedback_sphere_rgba(zone: "Phase3FeedbackZone") -> list[float]:
    return list(PHASE3_RGBA_ENTERED if zone.entered else PHASE3_RGBA_IDLE)


def _sphere_radius_pixels(
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    center: np.ndarray,
    radius_m: float,
) -> tuple[tuple[int, int] | None, int]:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    pix_c = wproj(float(c[0]), float(c[1]), float(c[2]))
    if pix_c is None:
        return None, 0
    pix_x = wproj(float(c[0] + radius_m), float(c[1]), float(c[2]))
    pix_y = wproj(float(c[0]), float(c[1] + radius_m), float(c[2]))
    rx = abs(int(pix_x[0]) - int(pix_c[0])) if pix_x is not None else 0
    ry = abs(int(pix_y[1]) - int(pix_c[1])) if pix_y is not None else 0
    return (int(pix_c[0]), int(pix_c[1])), max(rx, ry, 2)


def hide_feedback_spheres_for_recording(p: Any, registry: "Phase3FeedbackZones") -> None:

    for zone in registry.zones:
        uid = zone.body_uid
        if uid is None:
            continue
        try:
            p.changeVisualShape(int(uid), -1, rgbaColor=[0.0, 0.0, 0.0, 0.0])
        except Exception:
            pass


def restore_feedback_spheres_after_recording(p: Any, registry: "Phase3FeedbackZones") -> None:
    for zone in registry.zones:
        uid = zone.body_uid
        if uid is None:
            continue
        try:
            p.changeVisualShape(int(uid), -1, rgbaColor=_feedback_sphere_rgba(zone))
        except Exception:
            pass


def draw_feedback_spheres_overlay_bgr(
    img_bgr: np.ndarray,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    registry: "Phase3FeedbackZones",
    *,
    outline_thickness: int = 2,
) -> None:

    import cv2

    for zone in registry.zones:
        rgba = _feedback_sphere_rgba(zone)
        fill_alpha = float(max(0.004, min(0.55, rgba[3])))
        color_bgr = (
            int(rgba[2] * 255),
            int(rgba[1] * 255),
            int(rgba[0] * 255),
        )
        pix_c, r_px = _sphere_radius_pixels(wproj, zone.center, float(zone.radius))
        if pix_c is None:
            continue
        overlay = img_bgr.copy()
        cv2.circle(overlay, pix_c, r_px, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, fill_alpha, img_bgr, 1.0 - fill_alpha, 0, dst=img_bgr)
        cv2.circle(
            img_bgr,
            pix_c,
            r_px,
            color_bgr,
            max(1, int(outline_thickness)),
            lineType=cv2.LINE_AA,
        )


def _render_phase3_scene_with_sphere_overlays(
    p: Any,
    output_folder: Path,
    recording_kwargs: dict[str, Any],
    registry: "Phase3FeedbackZones",
    *,
    phase_label: str,
    overlay_fn: Callable[
        [np.ndarray, Callable[[float, float, float], tuple[int, int] | None]],
        None,
    ]
    | None = None,
) -> dict[str, str]:
    from .phase_recording import render_and_save_phase_scene

    vb = np.asarray(recording_kwargs["virtual_base_world"], dtype=np.float64).reshape(3)
    ws_lo = np.asarray(recording_kwargs["workspace_lo"], dtype=np.float64) + vb
    ws_hi = np.asarray(recording_kwargs["workspace_hi"], dtype=np.float64) + vb
    wxyz_fn = recording_kwargs.get("world_xyz_to_recording_image_pixel_fn")
    if wxyz_fn is None:
        return {}

    scene_kw = {
        "scene_lo": ws_lo.astype(np.float32),
        "scene_hi": ws_hi.astype(np.float32),
        "world_recording_view_proj_fn": recording_kwargs["world_recording_view_proj_fn"],
        "render_world_recording_rgb_fn": recording_kwargs["render_world_recording_rgb_fn"],
        "world_xyz_to_recording_image_pixel_fn": wxyz_fn,
        "recording_scene_fov": float(recording_kwargs.get("recording_scene_fov", 48.0)),
        "recording_scene_margin": float(recording_kwargs.get("recording_scene_margin", 1.75)),
        "recording_camera_distance_scale": float(
            recording_kwargs.get("recording_camera_distance_scale", 0.4)
        ),
        "recording_topview_distance_scale": float(
            recording_kwargs.get("recording_topview_distance_scale", 1.22)
        ),
        "recording_stereo45_distance_scale": float(
            recording_kwargs.get("recording_stereo45_distance_scale", 1.5)
        ),
        "width": int(recording_kwargs.get("render_width", 800)),
        "height": int(recording_kwargs.get("render_height", 800)),
        "phase_label": phase_label,
        "placed_cubes": recording_kwargs.get("placed_cubes"),
    }

    def _combined_overlay(
        img_bgr: np.ndarray,
        wproj: Callable[[float, float, float], tuple[int, int] | None],
    ) -> None:
        draw_feedback_spheres_overlay_bgr(img_bgr, wproj, registry)
        if overlay_fn is not None:
            overlay_fn(img_bgr, wproj)

    hide_feedback_spheres_for_recording(p, registry)
    try:
        return render_and_save_phase_scene(
            p,
            output_folder,
            overlay_fn=_combined_overlay,
            **scene_kw,
        )
    finally:
        restore_feedback_spheres_after_recording(p, registry)

DEFAULT_DRONE_BODY_HALF = (0.04, 0.04, 0.02)
DEFAULT_FEEDBACK_MARGIN_M = 0.06
DEFAULT_TRAJECTORY_SAMPLE_STEP_M = 0.02


def _dense_samples_along_trajectory(
    trajectory: np.ndarray,
    *,
    sample_step_m: float = DEFAULT_TRAJECTORY_SAMPLE_STEP_M,
) -> np.ndarray:

    pts = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    if pts.shape[0] == 1:
        return pts.copy()

    step = float(max(1e-4, sample_step_m))
    samples: list[np.ndarray] = [pts[0].copy()]
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            continue
        n_steps = max(1, int(np.ceil(seg_len / step)))
        for j in range(1, n_steps + 1):
            t = float(j) / float(n_steps)
            samples.append(p0 + t * seg)
    return np.asarray(samples, dtype=np.float64)


def _sphere_contact_at_point(
    pos: np.ndarray,
    zone_center: np.ndarray,
    zone_radius: float,
    drone_radius: float,
) -> bool:
    return float(np.linalg.norm(pos - zone_center)) <= float(zone_radius) + float(drone_radius)


def draw_virtual_drone_marker_bgr(
    img_bgr: np.ndarray,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    drone_pos_xyz: np.ndarray,
    drone_body_half: tuple[float, float, float],
    *,
    corner_radius_px: int = 5,
) -> bool:

    import cv2

    pos = np.asarray(drone_pos_xyz, dtype=np.float64).reshape(3)
    half = np.asarray(drone_body_half, dtype=np.float64).reshape(3)
    cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
    hx, hy = float(max(half[0], 1e-4)), float(max(half[1], 1e-4))

    corners_world = [
        (cx - hx, cy - hy, cz),
        (cx + hx, cy - hy, cz),
        (cx + hx, cy + hy, cz),
        (cx - hx, cy + hy, cz),
    ]
    pixels: list[tuple[int, int]] = []
    for wx, wy, wz in corners_world:
        pix = wproj(wx, wy, wz)
        if pix is None:
            return False
        pixels.append((int(pix[0]), int(pix[1])))

    poly = np.array(pixels, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img_bgr, [poly], color=(0, 255, 0))
    orange_bgr = (0, 165, 255)
    r = max(2, int(corner_radius_px))
    for pix in pixels:
        cv2.circle(img_bgr, pix, r, orange_bgr, -1, lineType=cv2.LINE_AA)
    return True


def _action_path_color_bgr(action: BasicAction | str | None) -> tuple[int, int, int]:
    if isinstance(action, BasicAction):
        return PHASE3_ACTION_PATH_BGR.get(action, (255, 200, 0))
    try:
        return PHASE3_ACTION_PATH_BGR.get(BasicAction(str(action)), (255, 200, 0))
    except ValueError:
        return (255, 200, 0)


def draw_phase3_action_path_bgr(
    img_bgr: np.ndarray,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    segment_xyz: np.ndarray,
    action: BasicAction | str,
    *,
    drone_body_half: tuple[float, float, float] = DEFAULT_DRONE_BODY_HALF,
    draw_waypoint_markers: bool = True,
) -> None:

    import cv2

    pts = np.asarray(segment_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 1:
        return
    color = _action_path_color_bgr(action)
    pix_ln: list[tuple[int, int]] = []
    for row in pts:
        pix = wproj(float(row[0]), float(row[1]), float(row[2]))
        if pix is not None:
            pix_ln.append((int(pix[0]), int(pix[1])))
    if len(pix_ln) >= 2:
        arr_pix = np.array(pix_ln, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            img_bgr,
            [arr_pix],
            isClosed=False,
            color=color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    for xy in pix_ln:
        cv2.circle(img_bgr, xy, 5, color, -1, lineType=cv2.LINE_AA)
    if draw_waypoint_markers:
        for row in pts:
            draw_virtual_drone_marker_bgr(
                img_bgr,
                wproj,
                row,
                drone_body_half,
                corner_radius_px=4,
            )
    act_label = action.value if isinstance(action, BasicAction) else str(action)
    if pix_ln:
        lx, ly = pix_ln[0]
        cv2.putText(
            img_bgr,
            f"a:{act_label}",
            (max(4, lx + 8), max(18, ly - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            lineType=cv2.LINE_AA,
        )


def draw_phase3_coarse_span_bgr(
    img_bgr: np.ndarray,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    coarse_xyz: np.ndarray,
) -> None:

    draw_phase3_trajectory_polyline_bgr(
        img_bgr,
        wproj,
        coarse_xyz,
        color_bgr=(160, 160, 160),
        thickness=1,
        draw_waypoint_dots=False,
    )


PHASE3_FULL_PATH_BGR = (255, 0, 0)


def draw_phase3_trajectory_polyline_bgr(
    img_bgr: np.ndarray,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    trajectory_xyz: np.ndarray,
    *,
    color_bgr: tuple[int, int, int] = PHASE3_FULL_PATH_BGR,
    thickness: int = 3,
    draw_waypoint_dots: bool = True,
    waypoint_dot_radius: int = 4,
) -> None:

    import cv2

    pts = np.asarray(trajectory_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 1:
        return
    pix_ln: list[tuple[int, int]] = []
    for row in pts:
        pix = wproj(float(row[0]), float(row[1]), float(row[2]))
        if pix is not None:
            pix_ln.append((int(pix[0]), int(pix[1])))
    if len(pix_ln) >= 2:
        arr_pix = np.array(pix_ln, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            img_bgr,
            [arr_pix],
            isClosed=False,
            color=color_bgr,
            thickness=int(thickness),
            lineType=cv2.LINE_AA,
        )
    if draw_waypoint_dots:
        for xy in pix_ln:
            cv2.circle(
                img_bgr,
                xy,
                int(waypoint_dot_radius),
                color_bgr,
                -1,
                lineType=cv2.LINE_AA,
            )


def draw_phase3_trajectory_endpoints_bgr(
    img_bgr: np.ndarray,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    trajectory_xyz: np.ndarray,
    *,
    drone_body_half: tuple[float, float, float] = DEFAULT_DRONE_BODY_HALF,
) -> None:

    import cv2

    pts = np.asarray(trajectory_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 1:
        return
    draw_virtual_drone_marker_bgr(
        img_bgr,
        wproj,
        pts[0],
        drone_body_half,
        corner_radius_px=5,
    )
    if pts.shape[0] >= 2:
        end_pix = wproj(float(pts[-1, 0]), float(pts[-1, 1]), float(pts[-1, 2]))
        if end_pix is not None:
            cv2.circle(
                img_bgr,
                (int(end_pix[0]), int(end_pix[1])),
                7,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
            )


def draw_phase3_refined_full_path_bgr(
    img_bgr: np.ndarray,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    refined_trajectory_xyz: np.ndarray,
    refinements: list[dict[str, Any]],
    *,
    drone_body_half: tuple[float, float, float] = DEFAULT_DRONE_BODY_HALF,
) -> None:

    draw_phase3_trajectory_polyline_bgr(
        img_bgr,
        wproj,
        refined_trajectory_xyz,
        color_bgr=PHASE3_FULL_PATH_BGR,
        thickness=2,
        draw_waypoint_dots=True,
        waypoint_dot_radius=3,
    )
    for ref in refinements:
        seg = ref.get("segment_xyz")
        if seg is None:
            continue
        draw_phase3_action_path_bgr(
            img_bgr,
            wproj,
            np.asarray(seg, dtype=np.float64),
            str(ref.get("action", "pass_through")),
            drone_body_half=drone_body_half,
            draw_waypoint_markers=False,
        )
    draw_phase3_trajectory_endpoints_bgr(
        img_bgr,
        wproj,
        refined_trajectory_xyz,
        drone_body_half=drone_body_half,
    )


def _aabb_center_radius(
    aabb: tuple | list,
    *,
    margin_m: float = DEFAULT_FEEDBACK_MARGIN_M,
) -> tuple[np.ndarray, float]:
    lo = np.asarray(aabb[0], dtype=np.float64).reshape(3)
    hi = np.asarray(aabb[1], dtype=np.float64).reshape(3)
    center = (lo + hi) * 0.5
    half_diag = float(0.5 * np.linalg.norm(hi - lo))
    return center, half_diag + float(max(0.0, margin_m))


def collect_phase3_feedback_entries(
    intersected_objects: list[dict[str, Any]],
    target_obj_info: dict[str, Any] | None,
    *,
    feedback_margin_m: float = DEFAULT_FEEDBACK_MARGIN_M,
) -> list[dict[str, Any]]:

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(name: str, center: list | np.ndarray, aabb: tuple | list, *, is_target: bool) -> None:
        cen = np.asarray(center, dtype=np.float64).reshape(3)
        key = entry_key(str(name), cen)
        if key in seen:
            return
        seen.add(key)
        _, radius = _aabb_center_radius(aabb, margin_m=feedback_margin_m)
        entries.append(
            {
                "name": str(name),
                "center": cen.tolist(),
                "radius": float(radius),
                "aabb": aabb,
                "is_target": bool(is_target),
            }
        )

    for obj in intersected_objects or []:
        aabb = obj.get("aabb")
        center = obj.get("center")
        if aabb is None or center is None:
            continue
        _add(str(obj.get("name", "object")), center, aabb, is_target=False)

    if target_obj_info is not None:
        aabb = target_obj_info.get("aabb")
        center = target_obj_info.get("center")
        if aabb is not None and center is not None:
            _add(
                str(target_obj_info.get("name", "target")),
                center,
                aabb,
                is_target=True,
            )

    return entries


def collect_phase3_entries_from_placed_cubes(
    obstacle_cubes: list[dict[str, Any]],
    target_cube: dict[str, Any] | None,
    *,
    aabb_fn: Any,
    display_name_fn: Any | None = None,
    feedback_margin_m: float = DEFAULT_FEEDBACK_MARGIN_M,
) -> list[dict[str, Any]]:

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(cube: dict[str, Any], *, is_target: bool) -> None:
        lo, hi = aabb_fn(cube)
        center = (lo + hi) * 0.5
        if display_name_fn is not None:
            name = str(display_name_fn(cube))
        else:
            name = f"{cube.get('color', '?')} {cube.get('shape', '?')}"
        key = entry_key(name, center)
        if key in seen:
            return
        seen.add(key)
        _, radius = _aabb_center_radius((lo, hi), margin_m=feedback_margin_m)
        entries.append(
            {
                "name": name,
                "center": center.tolist(),
                "radius": float(radius),
                "aabb": (lo.tolist(), hi.tolist()),
                "is_target": bool(is_target),
            }
        )

    for cube in obstacle_cubes or []:
        _add(cube, is_target=False)
    if target_cube is not None:
        _add(target_cube, is_target=True)
    return entries


def merge_phase3_feedback_entries(entry_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in entry_lists:
        for entry in group or []:
            key = entry_key(str(entry.get("name", "?")), np.asarray(entry["center"]))
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(entry))
    return merged


@dataclass
class Phase3FeedbackZone:
    name: str
    center: np.ndarray
    radius: float
    is_target: bool = False
    body_uid: int | None = None
    entered: bool = False
    aabb: tuple | list | None = None
    basic_action: BasicAction | None = None
    orbit_laps: int = 1


@dataclass
class Phase3FeedbackZones:


    zones: list[Phase3FeedbackZone] = field(default_factory=list)
    drone_body_half: tuple[float, float, float] = DEFAULT_DRONE_BODY_HALF
    phase3_folder: Any | None = None
    recording_kwargs: dict[str, Any] | None = None
    action_cache: dict[str, tuple[BasicAction, int]] = field(default_factory=dict)
    action_classify_source: dict[str, str] = field(default_factory=dict)
    _recording_seq: int = field(default=0, repr=False)

    def configure_recording(
        self,
        *,
        phase3_folder: Any | None,
        recording_kwargs: dict[str, Any] | None,
    ) -> None:
        self.phase3_folder = phase3_folder
        self.recording_kwargs = recording_kwargs
        self._recording_seq = 0

    @property
    def num_zones(self) -> int:
        return len(self.zones)

    @property
    def num_entered(self) -> int:
        return sum(1 for z in self.zones if z.entered)

    def zone_keys(self) -> set[str]:
        return {entry_key(z.name, z.center) for z in self.zones}

    def all_entered(self) -> bool:
        return bool(self.zones) and self.num_entered >= self.num_zones

    def _drone_contact_radius(self) -> float:
        h = np.asarray(self.drone_body_half, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(h))

    def update_contact_feedback(self, p: Any, drone_pos: np.ndarray) -> bool:

        pos = np.asarray(drone_pos, dtype=np.float64).reshape(3)
        drone_r = self._drone_contact_radius()
        triggered = False
        for zone in self.zones:
            if zone.entered:
                continue
            dist = float(np.linalg.norm(pos - zone.center))
            if dist <= zone.radius + drone_r:
                triggered = True
                self.apply_feedback_color(p, zone, contact_pos=pos)
        return triggered

    def _run_feedback_action(self, p: Any, zone: Phase3FeedbackZone) -> None:
        target_tag = "target " if zone.is_target else ""
        print(
            f"[Phase 3][feedback] drone entered {target_tag}feedback sphere: "
            f"{zone.name!r} (center={zone.center.round(3).tolist()}, "
            f"radius={zone.radius:.3f}m)"
        )
        if zone.body_uid is not None:
            try:
                apply_feedback_sphere_visual(p, zone)
                print(
                    f"[Phase 3][feedback] sphere color -> light red "
                    f"{PHASE3_RGBA_ALPHA * 100:.1f}% opacity: {zone.name!r}"
                )
            except Exception as exc:
                print(
                    f"[Phase 3][feedback] WARN: could not update sphere color "
                    f"(uid={zone.body_uid}) - {exc}"
                )

    def apply_feedback_color(
        self,
        p: Any,
        zone: Phase3FeedbackZone,
        *,
        contact_pos: np.ndarray | None = None,
        mission_cmd: str | None = None,
        placed_cubes: list[dict[str, Any]] | None = None,
    ) -> BasicAction | None:

        if zone.entered:
            return zone.basic_action
        zone.entered = True
        action, laps = resolve_basic_action_for_zone(
            mission_cmd,
            zone_name=zone.name,
            zone_center=zone.center,
            is_target=zone.is_target,
            placed_cubes=placed_cubes,
            action_cache=self.action_cache,
        )
        zone.basic_action = action
        zone.orbit_laps = int(laps)
        src_key = entry_key(zone.name, zone.center)
        src_raw = self.action_classify_source.get(src_key, "")
        src_tag = (
            " [keyword]"
            if src_raw == "keyword"
            else (" [X-VLA]" if src_raw == "xvla" else "")
        )
        print(
            f"[Phase 3][refine] object {zone.name!r} basic_action="
            f"{action.value}"
            + (f" ({laps} lap(s))" if action == BasicAction.ORBIT else "")
            + src_tag
        )
        self._run_feedback_action(p, zone)
        return zone.basic_action


def _find_zone_by_name(registry: Phase3FeedbackZones, zone_name: str) -> Phase3FeedbackZone | None:
    target = str(zone_name).strip().lower()
    for zone in registry.zones:
        if str(zone.name).strip().lower() == target:
            return zone
    return None


def _find_zone_for_refinement(
    registry: Phase3FeedbackZones,
    ref: dict[str, Any],
) -> Phase3FeedbackZone | None:

    from .phase3_actions import _zone_billboard_id

    bid = ref.get("billboard_id")
    if bid is not None:
        for zone in registry.zones:
            zbid = _zone_billboard_id(zone.name, zone.center, None)
            if zbid is not None and int(zbid) == int(bid):
                return zone
    zone_name = str(ref.get("zone_name", "")).strip()
    if zone_name:
        hit = _find_zone_by_name(registry, zone_name)
        if hit is not None:
            return hit
        name_l = zone_name.lower()
        for zone in registry.zones:
            if name_l in str(zone.name).lower() or str(zone.name).lower() in name_l:
                return zone
    seg = ref.get("segment_xyz")
    if seg is not None:
        pts = np.asarray(seg, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] >= 1:
            cen = pts[len(pts) // 2]
            best: tuple[float, Phase3FeedbackZone] | None = None
            for zone in registry.zones:
                d = float(np.linalg.norm(cen - zone.center))
                if best is None or d < best[0]:
                    best = (d, zone)
            if best is not None and best[0] < 0.5:
                return best[1]
    return None


def _contact_pos_from_refinement(ref: dict[str, Any]) -> np.ndarray | None:
    seg = ref.get("segment_xyz")
    if seg is not None:
        pts = np.asarray(seg, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] >= 1:
            return pts[0]
    coarse = ref.get("coarse_span_xyz")
    if coarse is not None:
        pts = np.asarray(coarse, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] >= 1:
            return pts[0]
    return None


def _save_prior_refinement_recordings(
    p: Any,
    registry: Phase3FeedbackZones,
    prior_refinements: list[dict[str, Any]],
    *,
    mission_cmd: str | None = None,
    placed_cubes: list[dict[str, Any]] | None = None,
) -> int:

    if registry.phase3_folder is None or not registry.recording_kwargs:
        return 0
    saved = 0
    for ref in prior_refinements:
        zone = _find_zone_for_refinement(registry, ref)
        if zone is None:
            print(
                f"[Phase 3] WARN: no feedback zone for zone_name={ref.get('zone_name')!r} "
                f"billboard_id={ref.get('billboard_id')}; skipping refine recording."
            )
            continue
        if not zone.entered:
            act_str = str(ref.get("action", "pass_through"))
            try:
                zone.basic_action = BasicAction(act_str)
            except ValueError:
                zone.basic_action = BasicAction.PASS_THROUGH
            zone.orbit_laps = int(ref.get("orbit_laps", 1))
            zone.entered = True
            contact_pos = _contact_pos_from_refinement(ref)
            registry._run_feedback_action(p, zone)
            if contact_pos is not None:
                print(
                    f"[Phase 3] reused refine mark {zone.name!r} -> "
                    f"{zone.basic_action.value}"
                )
        registry._recording_seq += 1
        contact_pos = _contact_pos_from_refinement(ref)
        _save_phase3_zone_recordings(
            p,
            registry.phase3_folder,
            registry.recording_kwargs,
            zone,
            registry._recording_seq,
            registry=registry,
            contact_pos=contact_pos,
            drone_body_half=registry.drone_body_half,
            refined_segment=ref.get("segment_xyz"),
            coarse_segment=ref.get("coarse_span_xyz"),
            basic_action=str(ref.get("action", "")),
        )
        saved += 1
    return saved


def run_phase3_apply_feedback_colors(
    p: Any,
    registry: Phase3FeedbackZones | None,
    *,
    trajectory: np.ndarray | list | None = None,
    mission_cmd: str | None = None,
    placed_cubes: list[dict[str, Any]] | None = None,
    phase3_folder: Any | None = None,
    recording_kwargs: dict[str, Any] | None = None,
    trajectory_sample_step_m: float = DEFAULT_TRAJECTORY_SAMPLE_STEP_M,
    prior_refinements: list[dict[str, Any]] | None = None,
    trajectory_coarse: np.ndarray | list | None = None,
) -> dict[str, Any]:

    if registry is None or registry.num_zones == 0:
        print("[Phase 3] No feedback spheres; skipping.")
        return {"contact_count": 0, "trajectory": None, "refinements": []}

    traj = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3) if trajectory is not None else None
    if traj is None or traj.shape[0] < 1:
        print("[Phase 3] WARN: no Phase2 virtual trajectory; contact detection skipped.")
        return {"contact_count": 0, "trajectory": None, "refinements": []}

    registry.configure_recording(
        phase3_folder=phase3_folder,
        recording_kwargs=recording_kwargs,
    )

    if registry.action_cache:
        print(
            f"[Phase 3] Reusing X-VLA basic-action cache from cmd analysis "
            f"({len(registry.action_cache)} object(s))."
        )
    elif mission_cmd and str(mission_cmd).strip():
        print("[Phase 3] No X-VLA action cache; keywords will be used on contact.")

    samples = _dense_samples_along_trajectory(traj, sample_step_m=trajectory_sample_step_m)
    drone_r = registry._drone_contact_radius()

    print("\n" + "=" * 60)
    print("[Phase 3] Trajectory refine: contact -> basic action -> polyline -> red sphere + recording")
    print(
        f"[Phase 3] trajectory waypoints={traj.shape[0]}, samples={samples.shape[0]}, "
        f"sample_step~={float(trajectory_sample_step_m):.3f}m"
    )
    print("[Phase 3] Each contact saves top-down + stereo45 to phase3/<1,2,3,...>/ (action-colored path).")
    print("[Phase 3] End of phase3 saves before/after full paths under phase3/path/.")

    prior = [dict(r) for r in (prior_refinements or []) if r]
    coarse_traj = (
        np.asarray(trajectory_coarse, dtype=np.float64).reshape(-1, 3)
        if trajectory_coarse is not None
        else traj.copy()
    )
    working = traj.copy()
    count = 0
    refinements: list[dict[str, Any]] = list(prior)

    if prior:
        actions_seen = sorted({str(r.get("action", "")) for r in prior if r.get("action")})
        print(
            f"[Phase 3] Reusing multi-action per-leg refines: {len(prior)} segment(s), "
            f"actions={actions_seen}"
        )
        saved_prior = _save_prior_refinement_recordings(
            p,
            registry,
            prior,
            mission_cmd=mission_cmd,
            placed_cubes=placed_cubes,
        )
        if saved_prior:
            print(f"[Phase 3] Saved {saved_prior} multi-action refine recording set(s) under phase3/<n>/.")
        count = sum(1 for z in registry.zones if z.entered)
    else:
        while not registry.all_entered():
            samples = _dense_samples_along_trajectory(
                working, sample_step_m=trajectory_sample_step_m
            )
            triggered = False
            for pos in samples:
                for zone in registry.zones:
                    if zone.entered:
                        continue
                    if not _sphere_contact_at_point(pos, zone.center, zone.radius, drone_r):
                        continue
                    contact_idx = nearest_trajectory_index(working, pos)
                    registry.apply_feedback_color(
                        p,
                        zone,
                        contact_pos=pos,
                        mission_cmd=mission_cmd,
                        placed_cubes=placed_cubes,
                    )
                    working, meta = refine_trajectory_at_zone_contact(
                        working,
                        zone_name=zone.name,
                        zone_center=zone.center,
                        zone_radius=float(zone.radius),
                        is_target=zone.is_target,
                        aabb=zone.aabb,
                        contact_pos=pos,
                        contact_traj_idx=contact_idx,
                        mission_cmd=mission_cmd,
                        placed_cubes=placed_cubes,
                        drone_body_half=registry.drone_body_half,
                        action_cache=registry.action_cache,
                    )
                    refinements.append(meta)
                    print(
                        f"[Phase 3][refine] {zone.name!r}: action={meta['action']}, "
                        f"replace waypoints [{meta['span_i0']}:{meta['span_i1']}] -> "
                        f"{meta['segment_points']} refined pts; path length {meta['refined_points']}"
                    )
                    if registry.phase3_folder is not None and registry.recording_kwargs:
                        registry._recording_seq += 1
                        _save_phase3_zone_recordings(
                            p,
                            registry.phase3_folder,
                            registry.recording_kwargs,
                            zone,
                            registry._recording_seq,
                            registry=registry,
                            contact_pos=pos,
                            drone_body_half=registry.drone_body_half,
                            refined_segment=meta.get("segment_xyz"),
                            coarse_segment=meta.get("coarse_span_xyz"),
                            basic_action=str(meta.get("action", "")),
                        )
                    count += 1
                    triggered = True
                    break
                if triggered:
                    break
            if not triggered:
                break

    print(f"[Phase 3] Contact triggered {count}/{registry.num_zones} sphere(s) -> red.")
    if refinements:
        print(f"[Phase 3] Refined trajectory waypoints: {working.shape[0]} (was {traj.shape[0]})")
        for i, row in enumerate(working):
            print(
                f"  r{i}: XY=[{float(row[0]):.4f}, {float(row[1]):.4f}] "
                f"Z={float(row[2]):.4f}"
            )
    print("[Phase 3] Phase3 complete. Paused for debugging.")
    print("=" * 60 + "\n")

    path_recordings: dict[str, Any] = {}
    if registry.phase3_folder is not None and registry.recording_kwargs:
        path_recordings = _save_phase3_overall_path_recordings(
            p,
            registry.phase3_folder,
            registry.recording_kwargs,
            registry=registry,
            coarse_trajectory=coarse_traj,
            refined_trajectory=working,
            refinements=refinements,
            drone_body_half=registry.drone_body_half,
            placed_cubes=placed_cubes,
            collision_pad_m=float(
                registry.recording_kwargs.get("navigation_collision_pad_m", 0.07)
                if registry.recording_kwargs
                else 0.07
            ),
        )

    return {
        "contact_count": count,
        "trajectory": working,
        "refinements": refinements,
        "original_trajectory": coarse_traj,
        "path_recordings": path_recordings,
        "action_cache": {
            k: {"action": v[0].value, "orbit_laps": int(v[1])}
            for k, v in (registry.action_cache or {}).items()
        },
    }


def _save_phase3_zone_recordings(
    p: Any,
    phase3_folder: Any,
    recording_kwargs: dict[str, Any],
    zone: Phase3FeedbackZone,
    event_index: int,
    *,
    registry: Phase3FeedbackZones | None = None,
    contact_pos: np.ndarray | None = None,
    drone_body_half: tuple[float, float, float] = DEFAULT_DRONE_BODY_HALF,
    refined_segment: np.ndarray | list | None = None,
    coarse_segment: np.ndarray | list | None = None,
    basic_action: str | None = None,
) -> dict[str, str]:

    try:
        folder = Path(phase3_folder) / str(int(event_index))
        folder.mkdir(parents=True, exist_ok=True)

        if recording_kwargs.get("world_xyz_to_recording_image_pixel_fn") is None:
            print("[Phase 3] WARN: missing world_xyz_to_recording_image_pixel_fn; cannot draw drone marker.")
            return {}

        drone_pos = (
            np.asarray(contact_pos, dtype=np.float64).reshape(3)
            if contact_pos is not None
            else None
        )
        seg = (
            np.asarray(refined_segment, dtype=np.float64).reshape(-1, 3)
            if refined_segment is not None
            else None
        )
        coarse = (
            np.asarray(coarse_segment, dtype=np.float64).reshape(-1, 3)
            if coarse_segment is not None
            else None
        )
        act = basic_action or (
            zone.basic_action.value if zone.basic_action is not None else "pass_through"
        )

        def _overlay_phase3_path(
            img_bgr: np.ndarray,
            wproj: Callable[[float, float, float], tuple[int, int] | None],
        ) -> None:
            if coarse is not None and coarse.shape[0] >= 2:
                draw_phase3_coarse_span_bgr(img_bgr, wproj, coarse)
            if seg is not None and seg.shape[0] >= 1:
                draw_phase3_action_path_bgr(
                    img_bgr,
                    wproj,
                    seg,
                    act,
                    drone_body_half=drone_body_half,
                    draw_waypoint_markers=True,
                )
            elif drone_pos is not None:
                draw_virtual_drone_marker_bgr(
                    img_bgr,
                    wproj,
                    drone_pos,
                    drone_body_half,
                )

        if registry is not None and registry.num_zones > 0:
            return _render_phase3_scene_with_sphere_overlays(
                p,
                folder,
                recording_kwargs,
                registry,
                phase_label=f"Phase 3 [{event_index}] {zone.name!r} {act}",
                overlay_fn=_overlay_phase3_path,
            )

        from .phase_recording import render_and_save_phase_scene

        vb = np.asarray(recording_kwargs["virtual_base_world"], dtype=np.float64).reshape(3)
        ws_lo = np.asarray(recording_kwargs["workspace_lo"], dtype=np.float64) + vb
        ws_hi = np.asarray(recording_kwargs["workspace_hi"], dtype=np.float64) + vb
        return render_and_save_phase_scene(
            p,
            folder,
            scene_lo=ws_lo.astype(np.float32),
            scene_hi=ws_hi.astype(np.float32),
            world_recording_view_proj_fn=recording_kwargs["world_recording_view_proj_fn"],
            render_world_recording_rgb_fn=recording_kwargs["render_world_recording_rgb_fn"],
            world_xyz_to_recording_image_pixel_fn=recording_kwargs[
                "world_xyz_to_recording_image_pixel_fn"
            ],
            recording_scene_fov=float(recording_kwargs.get("recording_scene_fov", 48.0)),
            recording_scene_margin=float(recording_kwargs.get("recording_scene_margin", 1.75)),
            recording_camera_distance_scale=float(
                recording_kwargs.get("recording_camera_distance_scale", 0.4)
            ),
            recording_topview_distance_scale=float(
                recording_kwargs.get("recording_topview_distance_scale", 1.22)
            ),
            recording_stereo45_distance_scale=float(
                recording_kwargs.get("recording_stereo45_distance_scale", 1.5)
            ),
            width=int(recording_kwargs.get("render_width", 800)),
            height=int(recording_kwargs.get("render_height", 800)),
            phase_label=f"Phase 3 [{event_index}] {zone.name!r} {act}",
            overlay_fn=_overlay_phase3_path,
            placed_cubes=recording_kwargs.get("placed_cubes"),
        )
    except Exception as exc:
        print(f"[Phase 3] WARN: sphere {zone.name!r} recording failed - {exc}")
        return {}


def _refinements_json_safe(refinements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ref in refinements:
        item = dict(ref)
        for key in ("segment_xyz", "coarse_span_xyz"):
            val = item.get(key)
            if val is not None:
                item[key] = np.asarray(val, dtype=np.float64).reshape(-1, 3).tolist()
        out.append(item)
    return out


def _save_phase3_overall_path_recordings(
    p: Any,
    phase3_folder: Any,
    recording_kwargs: dict[str, Any],
    *,
    registry: Phase3FeedbackZones | None = None,
    coarse_trajectory: np.ndarray,
    refined_trajectory: np.ndarray,
    refinements: list[dict[str, Any]],
    drone_body_half: tuple[float, float, float] = DEFAULT_DRONE_BODY_HALF,
    placed_cubes: list[dict] | None = None,
    collision_pad_m: float = 0.07,
) -> dict[str, Any]:

    import json

    coarse = np.asarray(coarse_trajectory, dtype=np.float64).reshape(-1, 3)
    refined = np.asarray(refined_trajectory, dtype=np.float64).reshape(-1, 3)
    if coarse.shape[0] < 1 or refined.shape[0] < 1:
        print("[Phase 3][path] WARN: empty trajectory; skipping full-path recording.")
        return {}

    try:
        path_root = Path(phase3_folder) / "path"
        coarse_dir = path_root / "without_action_refine"
        refined_dir = path_root / "with_action_refine"
        coarse_dir.mkdir(parents=True, exist_ok=True)
        refined_dir.mkdir(parents=True, exist_ok=True)

        if recording_kwargs.get("world_xyz_to_recording_image_pixel_fn") is None:
            print("[Phase 3][path] WARN: missing world_xyz_to_recording_image_pixel_fn; cannot save full path.")
            return {}

        def _overlay_coarse(
            img_bgr: np.ndarray,
            wproj: Callable[[float, float, float], tuple[int, int] | None],
        ) -> None:
            draw_phase3_trajectory_polyline_bgr(
                img_bgr,
                wproj,
                coarse,
                color_bgr=PHASE3_FULL_PATH_BGR,
                thickness=3,
                draw_waypoint_dots=True,
            )
            draw_phase3_trajectory_endpoints_bgr(
                img_bgr,
                wproj,
                coarse,
                drone_body_half=drone_body_half,
            )

        def _overlay_refined(
            img_bgr: np.ndarray,
            wproj: Callable[[float, float, float], tuple[int, int] | None],
        ) -> None:
            draw_phase3_refined_full_path_bgr(
                img_bgr,
                wproj,
                refined,
                refinements,
                drone_body_half=drone_body_half,
            )

        if registry is not None and registry.num_zones > 0:
            saved_coarse = _render_phase3_scene_with_sphere_overlays(
                p,
                coarse_dir,
                recording_kwargs,
                registry,
                phase_label="Phase 3 [path] without_action_refine",
                overlay_fn=_overlay_coarse,
            )
            saved_refined = _render_phase3_scene_with_sphere_overlays(
                p,
                refined_dir,
                recording_kwargs,
                registry,
                phase_label="Phase 3 [path] with_action_refine",
                overlay_fn=_overlay_refined,
            )
        else:
            from .phase_recording import render_and_save_phase_scene

            vb = np.asarray(recording_kwargs["virtual_base_world"], dtype=np.float64).reshape(3)
            ws_lo = np.asarray(recording_kwargs["workspace_lo"], dtype=np.float64) + vb
            ws_hi = np.asarray(recording_kwargs["workspace_hi"], dtype=np.float64) + vb
            common_kw = {
                "scene_lo": ws_lo.astype(np.float32),
                "scene_hi": ws_hi.astype(np.float32),
                "world_recording_view_proj_fn": recording_kwargs["world_recording_view_proj_fn"],
                "render_world_recording_rgb_fn": recording_kwargs["render_world_recording_rgb_fn"],
                "world_xyz_to_recording_image_pixel_fn": recording_kwargs[
                    "world_xyz_to_recording_image_pixel_fn"
                ],
                "recording_scene_fov": float(recording_kwargs.get("recording_scene_fov", 48.0)),
                "recording_scene_margin": float(recording_kwargs.get("recording_scene_margin", 1.75)),
                "recording_camera_distance_scale": float(
                    recording_kwargs.get("recording_camera_distance_scale", 0.4)
                ),
                "recording_topview_distance_scale": float(
                    recording_kwargs.get("recording_topview_distance_scale", 1.22)
                ),
                "recording_stereo45_distance_scale": float(
                    recording_kwargs.get("recording_stereo45_distance_scale", 1.5)
                ),
                "width": int(recording_kwargs.get("render_width", 800)),
                "height": int(recording_kwargs.get("render_height", 800)),
                "placed_cubes": recording_kwargs.get("placed_cubes"),
            }
            saved_coarse = render_and_save_phase_scene(
                p,
                coarse_dir,
                phase_label="Phase 3 [path] without_action_refine",
                overlay_fn=_overlay_coarse,
                **common_kw,
            )
            saved_refined = render_and_save_phase_scene(
                p,
                refined_dir,
                phase_label="Phase 3 [path] with_action_refine",
                overlay_fn=_overlay_refined,
                **common_kw,
            )

        drone_r = float(np.linalg.norm(np.asarray(drone_body_half, dtype=np.float64).reshape(3)))
        interaction_centers: list[np.ndarray] = []
        if registry is not None:
            for ref in refinements:
                zone = _find_zone_for_refinement(registry, ref)
                if zone is not None:
                    interaction_centers.append(
                        np.asarray(zone.center, dtype=np.float64).reshape(3)
                    )
        quality_coarse = compute_path_quality_metrics(
            coarse,
            placed_cubes,
            drone_r=drone_r,
            collision_pad=float(collision_pad_m),
            exclude_object_centers=interaction_centers,
        )
        quality_refined = compute_path_quality_metrics(
            refined,
            placed_cubes,
            drone_r=drone_r,
            collision_pad=float(collision_pad_m),
            exclude_object_centers=interaction_centers,
        )

        sidecar = {
            "phase": "navigation_phase3_path",
            "description": (
                "Full trajectory from start through Phase 3: "
                "without_action_refine = Phase1+2 only; "
                "with_action_refine = after five basic-action segment refinements."
            ),
            "coarse_waypoints": coarse.tolist(),
            "refined_waypoints": refined.tolist(),
            "quality_coarse": quality_coarse,
            "quality_refined": quality_refined,
            "refinement_count": len(refinements),
            "refinements": _refinements_json_safe(refinements),
            "action_path_colors_bgr": {
                act.value: list(color) for act, color in PHASE3_ACTION_PATH_BGR.items()
            },
            "outputs": {
                "without_action_refine": saved_coarse,
                "with_action_refine": saved_refined,
            },
        }
        sidecar_path = path_root / "navigation_phase3_path.json"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[Phase 3][path] full trajectory saved: {path_root}\n"
            f"  - without_action_refine: {saved_coarse.get('topdown', coarse_dir / 'topdown.png')}\n"
            f"  - with_action_refine: {saved_refined.get('topdown', refined_dir / 'topdown.png')}\n"
            f"  - sidecar: {sidecar_path}"
        )
        return {
            "path_root": str(path_root),
            "without_action_refine": saved_coarse,
            "with_action_refine": saved_refined,
            "sidecar": str(sidecar_path),
        }
    except Exception as exc:
        print(f"[Phase 3][path] WARN: full-path recording failed - {exc}")
        return {}
