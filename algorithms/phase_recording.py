from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .portal_geometry import portal_number_billboard_center_and_size

PHASE1_DIR = "phase1"
PHASE2_DIR = "phase2"
PHASE3_DIR = "phase3"
TOPDOWN_PNG = "topdown.png"
STEREO45_PNG = "stereo45deg.png"


def recording_experiment_folder_name(dt: datetime.datetime | None = None) -> str:

    dt = dt or datetime.datetime.now()
    time_sep = "-" if os.name == "nt" else ":"
    return (
        f"{dt.year}_{dt.month}_{dt.day}_"
        f"{dt.hour:02d}{time_sep}{dt.minute:02d}{time_sep}{dt.second:02d}"
    )

_RECORDING_PREFER_OPENGL = False
_RECORDING_RENDERER_LOGGED = False


def set_recording_prefer_opengl(enabled: bool) -> None:

    global _RECORDING_PREFER_OPENGL, _RECORDING_RENDERER_LOGGED
    _RECORDING_PREFER_OPENGL = bool(enabled)
    _RECORDING_RENDERER_LOGGED = False


def pybullet_connection_is_gui(p: Any) -> bool:
    try:
        info = p.getConnectionInfo()
        return int(info.get("connectionMethod", -1)) == int(p.GUI)
    except Exception:
        return False


def resolve_pybullet_camera_renderer(p: Any, *, prefer_opengl: bool = False) -> int:
    if prefer_opengl and pybullet_connection_is_gui(p):
        return int(p.ER_BULLET_HARDWARE_OPENGL)
    return int(p.ER_TINY_RENDERER)


def configure_pybullet_gui_opengl_transparency(p: Any) -> None:

    if not pybullet_connection_is_gui(p):
        return
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    except Exception:
        pass


def render_camera_rgb(
    p: Any,
    view: list[float],
    proj: list[float],
    *,
    width: int,
    height: int,
    prefer_opengl: bool | None = None,
) -> np.ndarray:

    global _RECORDING_RENDERER_LOGGED

    use_gl = _RECORDING_PREFER_OPENGL if prefer_opengl is None else bool(prefer_opengl)
    renderer = resolve_pybullet_camera_renderer(p, prefer_opengl=use_gl)
    if not _RECORDING_RENDERER_LOGGED:
        tag = "ER_BULLET_HARDWARE_OPENGL" if renderer == int(p.ER_BULLET_HARDWARE_OPENGL) else "ER_TINY_RENDERER"
        print(f"[render] PyBullet camera renderer: {tag} (GUI alpha={'on' if tag.endswith('OPENGL') else 'off'})")
        _RECORDING_RENDERER_LOGGED = True

    try:
        _, _, rgba, _, _ = p.getCameraImage(
            width=int(width),
            height=int(height),
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=renderer,
        )
    except Exception:
        _, _, rgba, _, _ = p.getCameraImage(
            width=int(width),
            height=int(height),
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=int(p.ER_TINY_RENDERER),
        )
    return np.reshape(rgba, (int(height), int(width), 4))[..., :3].astype(np.uint8)


def resolve_experiment_base_folder(
    *,
    rec_folder: Any | None = None,
    recording_folder: Any | None = None,
    root_dir: Any | None = None,
) -> Path:

    if rec_folder is not None:
        return Path(rec_folder)
    if recording_folder is not None:
        return Path(recording_folder)
    ts = recording_experiment_folder_name()
    root = Path(root_dir) if root_dir is not None else Path.cwd()
    return root / "recordings" / ts


def resolve_phase_folder(
    experiment_base: Path | str,
    phase: str,
    *,
    leg_subfolder: str | None = None,
    mkdir: bool = True,
) -> Path:

    folder = Path(experiment_base) / phase
    if leg_subfolder:
        folder = folder / leg_subfolder
    if mkdir:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


_PORTAL_LABEL_OVERLAY_BG_BGR = (220, 236, 250)


def draw_portal_number_labels_overlay_bgr(
    img_bgr: np.ndarray,
    wproj: Callable[[float, float, float], tuple[int, int] | None],
    placed_cubes: list[dict],
) -> None:

    import cv2

    if not placed_cubes:
        return
    h_img, w_img = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    for c in placed_cubes:
        if c.get("shape") != "rect_frame" or c.get("portal_label") is None:
            continue
        num = int(c["portal_label"])
        center, side_m = portal_number_billboard_center_and_size(c)
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        pc = wproj(cx, cy, cz)
        if pc is None:
            continue
        pe = wproj(cx + 0.5 * side_m, cy, cz)
        pn = wproj(cx, cy + 0.5 * side_m, cz)
        if pe is not None and pn is not None:
            r_px = int(
                max(
                    10,
                    min(140, abs(pe[0] - pc[0]), abs(pn[1] - pc[1])),
                )
            )
        elif pe is not None:
            r_px = int(max(10, min(140, abs(pe[0] - pc[0]))))
        else:
            r_px = 18
        x0 = int(np.clip(pc[0] - r_px, 0, w_img - 1))
        y0 = int(np.clip(pc[1] - r_px, 0, h_img - 1))
        x1 = int(np.clip(pc[0] + r_px, 0, w_img - 1))
        y1 = int(np.clip(pc[1] + r_px, 0, h_img - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(
            img_bgr,
            (x0, y0),
            (x1, y1),
            _PORTAL_LABEL_OVERLAY_BG_BGR,
            thickness=-1,
            lineType=cv2.LINE_8,
        )
        cv2.rectangle(
            img_bgr,
            (x0, y0),
            (x1, y1),
            (0, 0, 0),
            thickness=max(1, r_px // 24),
            lineType=cv2.LINE_8,
        )
        text = str(num)
        scale = float(np.clip(r_px / 22.0, 0.45, 2.8))
        thickness = max(2, int(round(scale * 2.2)))
        (tw, th), _bl = cv2.getTextSize(text, font, scale, thickness)
        org = (pc[0] - tw // 2, pc[1] + th // 2)
        cv2.putText(
            img_bgr,
            text,
            org,
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_8,
        )


def sharpen_topdown_portal_labels_rgb(
    rgb: np.ndarray,
    placed_cubes: list[dict],
    wproj: Callable[[float, float, float], tuple[int, int] | None],
) -> np.ndarray:

    import cv2

    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    draw_portal_number_labels_overlay_bgr(bgr, wproj, placed_cubes)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_bgr_views(
    output_folder: Path,
    topdown_bgr: np.ndarray,
    stereo_bgr: np.ndarray | None,
    *,
    topdown_name: str = TOPDOWN_PNG,
    stereo_name: str = STEREO45_PNG,
) -> dict[str, str]:

    import cv2

    output_folder.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    top_path = output_folder / topdown_name
    cv2.imwrite(str(top_path), topdown_bgr)
    saved["topdown"] = str(top_path)
    if stereo_bgr is not None:
        st_path = output_folder / stereo_name
        cv2.imwrite(str(st_path), stereo_bgr)
        saved["stereo45deg"] = str(st_path)
    return saved


def render_phase_scene_bgr_views(
    p: Any,
    *,
    scene_lo: np.ndarray,
    scene_hi: np.ndarray,
    world_recording_view_proj_fn: Callable[..., tuple[list[float], list[float]]],
    render_world_recording_rgb_fn: Callable[..., np.ndarray],
    world_xyz_to_recording_image_pixel_fn: Callable[
        [np.ndarray, list[float], list[float], int, int], tuple[int, int] | None
    ],
    recording_scene_fov: float,
    recording_scene_margin: float,
    recording_camera_distance_scale: float,
    recording_topview_distance_scale: float,
    recording_stereo45_distance_scale: float = 1.5,
    width: int = 800,
    height: int = 800,
    placed_cubes: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, Callable[[float, float, float], tuple[int, int] | None], Callable[[float, float, float], tuple[int, int] | None]]:

    import cv2

    lo = np.asarray(scene_lo, dtype=np.float32)
    hi = np.asarray(scene_hi, dtype=np.float32)
    fov = float(recording_scene_fov)
    margin = float(recording_scene_margin)
    dist_scale = float(recording_camera_distance_scale)
    top_dist_scale = float(recording_topview_distance_scale)
    stereo45_dist_scale = float(recording_stereo45_distance_scale)
    tw, th = int(width), int(height)

    view_m, proj_m = world_recording_view_proj_fn(
        p,
        lo,
        hi,
        view_kind="top",
        width=tw,
        height=th,
        fov=fov,
        margin=margin,
        distance_scale=dist_scale,
        top_view_distance_scale=top_dist_scale,
    )
    top_rgb = render_world_recording_rgb_fn(p, view_m, proj_m, width=tw, height=th)
    top_bgr = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)

    view_st, proj_st = world_recording_view_proj_fn(
        p,
        lo,
        hi,
        view_kind="45deg",
        width=tw,
        height=th,
        fov=fov,
        margin=margin,
        distance_scale=dist_scale,
        top_view_distance_scale=top_dist_scale,
        stereo45_view_distance_scale=stereo45_dist_scale,
    )
    stereo_rgb = render_world_recording_rgb_fn(p, view_st, proj_st, width=tw, height=th)
    stereo_bgr = cv2.cvtColor(stereo_rgb, cv2.COLOR_RGB2BGR)

    def wproj_top(wx: float, wy: float, wz: float) -> tuple[int, int] | None:
        return world_xyz_to_recording_image_pixel_fn(
            np.array([wx, wy, wz], dtype=np.float64),
            view_m,
            proj_m,
            width=tw,
            height=th,
        )

    def wproj_stereo(wx: float, wy: float, wz: float) -> tuple[int, int] | None:
        return world_xyz_to_recording_image_pixel_fn(
            np.array([wx, wy, wz], dtype=np.float64),
            view_st,
            proj_st,
            width=tw,
            height=th,
        )

    return top_bgr, stereo_bgr, wproj_top, wproj_stereo


def render_and_save_phase_scene(
    p: Any,
    output_folder: Path,
    *,
    scene_lo: np.ndarray,
    scene_hi: np.ndarray,
    world_recording_view_proj_fn: Callable[..., tuple[list[float], list[float]]],
    render_world_recording_rgb_fn: Callable[..., np.ndarray],
    world_xyz_to_recording_image_pixel_fn: Callable[
        [np.ndarray, list[float], list[float], int, int], tuple[int, int] | None
    ] | None = None,
    recording_scene_fov: float,
    recording_scene_margin: float,
    recording_camera_distance_scale: float,
    recording_topview_distance_scale: float,
    recording_stereo45_distance_scale: float = 1.5,
    width: int = 800,
    height: int = 800,
    phase_label: str = "Phase",
    overlay_fn: Callable[
        [np.ndarray, Callable[[float, float, float], tuple[int, int] | None]],
        None,
    ]
    | None = None,
    placed_cubes: list[dict] | None = None,
) -> dict[str, str]:

    if world_xyz_to_recording_image_pixel_fn is None:
        raise ValueError("world_xyz_to_recording_image_pixel_fn is required")

    top_bgr, stereo_bgr, wproj_top, wproj_stereo = render_phase_scene_bgr_views(
        p,
        scene_lo=scene_lo,
        scene_hi=scene_hi,
        world_recording_view_proj_fn=world_recording_view_proj_fn,
        render_world_recording_rgb_fn=render_world_recording_rgb_fn,
        world_xyz_to_recording_image_pixel_fn=world_xyz_to_recording_image_pixel_fn,
        recording_scene_fov=recording_scene_fov,
        recording_scene_margin=recording_scene_margin,
        recording_camera_distance_scale=recording_camera_distance_scale,
        recording_topview_distance_scale=recording_topview_distance_scale,
        recording_stereo45_distance_scale=recording_stereo45_distance_scale,
        width=width,
        height=height,
        placed_cubes=placed_cubes,
    )
    if overlay_fn is not None:
        overlay_fn(top_bgr, wproj_top)
        overlay_fn(stereo_bgr, wproj_stereo)
    if placed_cubes:
        draw_portal_number_labels_overlay_bgr(top_bgr, wproj_top, placed_cubes)

    saved = save_bgr_views(output_folder, top_bgr, stereo_bgr)
    print(f"[{phase_label}] top-down view saved: {saved.get('topdown')}")
    if "stereo45deg" in saved:
        print(f"[{phase_label}] stereo 45deg view saved: {saved['stereo45deg']}")
    return saved
