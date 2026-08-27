from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .instruction_parse import (
    extract_ordered_mission_billboard_ids,
    extract_ordered_traversal_billboard_ids,
)
from .portal_geometry import (
    compute_leg_corridor_xy,
    expand_corridor_xy,
    is_portal_object,
    portal_opening_center_world,
)
from .phase2 import run_navigation_phase2_topdown_xvla_xy_plan
from .phase_recording import (
    PHASE2_DIR,
    STEREO45_PNG,
    TOPDOWN_PNG,
    draw_portal_number_labels_overlay_bgr,
    resolve_experiment_base_folder,
    resolve_phase_folder,
    save_bgr_views,
)
from .phase3 import (
    PHASE3_RGBA_IDLE,
    Phase3FeedbackZone,
    Phase3FeedbackZones,
    collect_phase3_feedback_entries,
    entry_key,
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


def _placed_object_aabb_lo_hi_world(c: dict) -> tuple[np.ndarray, np.ndarray]:

    pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
    bh = c.get("bounds_half")
    if bh is not None:
        h = np.asarray(bh, dtype=np.float64).reshape(3)
        return pos - h, pos + h
    half = float(c.get("half", 0.025))
    h = np.array([half, half, half], dtype=np.float64)
    return pos - h, pos + h


def _phase1_include_placed_obj(c: dict) -> bool:
    return _placed_obj_nav_kind(c) in ("gate", "cube", "sphere", "ramp")


def _placed_obj_display_name(c: dict) -> str:
    col = str(c.get("color_name", c.get("color", "?")))
    return f"{col} {_placed_obj_nav_kind(c)}"


def _placed_object_pyb_aabb_pair(c: dict) -> tuple[list[float], list[float]]:
    lo, hi = _placed_object_aabb_lo_hi_world(c)
    return lo.tolist(), hi.tolist()


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
    target_label: str = "TARGET",
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
            target_label,
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


def setup_phase1_feedback_spheres(
    p: Any,
    intersected_objects: list[dict[str, Any]],
    target_obj_info: dict[str, Any] | None,
    *,
    registry: Phase3FeedbackZones | None = None,
    feedback_margin_m: float | None = None,
) -> Phase3FeedbackZones:

    from .phase3 import (
        DEFAULT_FEEDBACK_MARGIN_M,
        PHASE3_RGBA_ALPHA,
        apply_feedback_sphere_visual,
        create_feedback_sphere_visual_shape,
    )

    manager = registry if registry is not None else Phase3FeedbackZones()
    existing = manager.zone_keys()
    entries = collect_phase3_feedback_entries(
        intersected_objects,
        target_obj_info,
        feedback_margin_m=float(
            DEFAULT_FEEDBACK_MARGIN_M if feedback_margin_m is None else feedback_margin_m
        ),
    )
    new_entries = [
        e
        for e in entries
        if entry_key(str(e.get("name", "?")), np.asarray(e["center"])) not in existing
    ]
    if not new_entries:
        if not entries:
            print("[Phase 1] No valid objects in detection rectangle; skipping feedback spheres.")
        return manager

    print(
        f"\n[Phase 1] Creating feedback spheres for {len(new_entries)} detected object(s) "
        f"(light blue {PHASE3_RGBA_ALPHA * 100:.1f}% opacity)"
    )
    for entry in new_entries:
        center = np.asarray(entry["center"], dtype=np.float64).reshape(3)
        radius = float(entry["radius"])
        name = str(entry.get("name", "object"))
        is_target = bool(entry.get("is_target", False))
        vis = create_feedback_sphere_visual_shape(p, radius, PHASE3_RGBA_IDLE)
        body_uid = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=center.tolist(),
        )
        tag = " (target)" if is_target else ""
        print(
            f"  - {name}{tag}: center={center.round(3).tolist()} "
            f"radius={radius:.3f}m"
        )
        zone = Phase3FeedbackZone(
            name=name,
            center=center,
            radius=radius,
            is_target=is_target,
            body_uid=int(body_uid),
            entered=False,
            aabb=entry.get("aabb"),
        )
        apply_feedback_sphere_visual(p, zone)
        manager.zones.append(zone)
    print(f"[Phase 1] Feedback spheres created: {manager.num_zones} total.")
    return manager


def compute_phase1_corridor_and_obstacles(
    *,
    drone_pos: np.ndarray,
    g_world: np.ndarray,
    placed_cubes: list[dict],
    mission_cmd: str | None,
    cur_instruction: str,
    first_rect_portal_for_instruction_color_fn: Callable[..., Any | None],
    target_portal_ref: dict | None = None,
    corridor_margin_m: float = 0.0,
    corridor_bandwidth_m: float = 0.12,
    feedback_radius_m: float = 0.0,
) -> dict[str, Any]:

    start_xy = np.asarray(drone_pos, dtype=np.float64).reshape(-1)[:2]
    target_xy = np.asarray(g_world, dtype=np.float64).reshape(-1)[:2]

    portal_ref_early = target_portal_ref
    if portal_ref_early is None:
        portal_ref_early = first_rect_portal_for_instruction_color_fn(
            str(mission_cmd or cur_instruction or ""),
            placed_cubes,
            prefer_near_xyz=np.asarray(drone_pos, dtype=np.float64),
        )

    fb_r = float(max(0.0, feedback_radius_m))
    if portal_ref_early is not None and fb_r <= 0.0:
        bh = portal_ref_early.get("bounds_half")
        if bh is not None:
            fb_r = float(np.linalg.norm(np.asarray(bh, dtype=np.float64).reshape(3))) * 0.55 + 0.08

    x_min, y_min, x_max, y_max = compute_leg_corridor_xy(
        np.asarray(drone_pos, dtype=np.float64).reshape(3),
        np.asarray(g_world, dtype=np.float64).reshape(3),
        margin_m=float(corridor_margin_m),
        bandwidth_m=float(corridor_bandwidth_m),
        portal_ref=portal_ref_early,
        feedback_radius_m=fb_r,
    )

    path2 = target_xy.astype(np.float64) - start_xy.astype(np.float64)
    path_len2 = float(np.dot(path2, path2))

    def _phase1_xy_depth_key(cx: float, cy: float) -> tuple[float, float]:
        pxy = np.array([cx, cy], dtype=np.float64) - start_xy.astype(np.float64)
        if path_len2 < 1e-14:
            return (0.0, float(np.linalg.norm(pxy)))
        t = float(np.dot(pxy, path2) / path_len2)
        proj = path2 * t
        perp = float(np.linalg.norm(pxy - proj))
        return (t, perp)

    portal_ref = portal_ref_early
    if portal_ref is None:
        portal_ref = first_rect_portal_for_instruction_color_fn(
            str(mission_cmd or cur_instruction or ""),
            placed_cubes,
            prefer_near_xyz=np.asarray(drone_pos, dtype=np.float64),
        )

    target_obj_info = None
    target_z = float(np.asarray(g_world, dtype=np.float64).reshape(-1)[2])
    cand: list[tuple[float, dict, tuple]] = []

    for obj in placed_cubes:
        if not _phase1_include_placed_obj(obj):
            continue
        aabb = _placed_object_pyb_aabb_pair(obj)
        mn = np.asarray(aabb[0], dtype=np.float64)
        mx = np.asarray(aabb[1], dtype=np.float64)
        if not (mn[0] <= g_world[0] <= mx[0] and mn[1] <= g_world[1] <= mx[1]):
            continue
        cvec = (mn + mx) * 0.5
        d3 = float(np.linalg.norm(cvec - np.asarray(g_world, dtype=np.float64)))
        cand.append((d3, obj, aabb))

    if portal_ref is not None:
        aabb = _placed_object_pyb_aabb_pair(portal_ref)
        mn = np.asarray(aabb[0], dtype=np.float64)
        mx = np.asarray(aabb[1], dtype=np.float64)
        oc = portal_opening_center_world(portal_ref)
        target_obj_info = {
            "name": _placed_obj_display_name(portal_ref),
            "ref": portal_ref,
            "center": [float(oc[0]), float(oc[1]), float(oc[2])],
            "aabb": aabb,
            "kind": "gate",
        }
    elif cand:
        cand.sort(key=lambda x: x[0])
        _d3, obj, aabb = cand[0]
        mn = np.asarray(aabb[0], dtype=np.float64)
        mx = np.asarray(aabb[1], dtype=np.float64)
        target_obj_info = {
            "name": _placed_obj_display_name(obj),
            "ref": obj,
            "center": [
                float((mn[0] + mx[0]) * 0.5),
                float((mn[1] + mx[1]) * 0.5),
                float((mn[2] + mx[2]) * 0.5),
            ],
            "aabb": aabb,
        }

    target_ref_p1: dict | None = target_obj_info["ref"] if target_obj_info else None

    intersected_objects = []
    for obj in placed_cubes:
        if not _phase1_include_placed_obj(obj):
            continue
        if target_ref_p1 is not None and obj is target_ref_p1:
            continue
        aabb = _placed_object_pyb_aabb_pair(obj)
        obj_x_min, obj_y_min = aabb[0][0], aabb[0][1]
        obj_x_max, obj_y_max = aabb[1][0], aabb[1][1]

        overlap_x = (x_min <= obj_x_max) and (x_max >= obj_x_min)
        overlap_y = (y_min <= obj_y_max) and (y_max >= obj_y_min)

        if overlap_x and overlap_y:
            cx = (obj_x_min + obj_x_max) / 2.0
            cy = (obj_y_min + obj_y_max) / 2.0
            cz = (aabb[0][2] + aabb[1][2]) / 2.0
            t_key, perp_key = _phase1_xy_depth_key(cx, cy)
            entry: dict[str, Any] = {
                "name": _placed_obj_display_name(obj),
                "dist": float(np.linalg.norm(np.array([cx, cy]) - start_xy)),
                "path_t": t_key,
                "path_perp": perp_key,
                "aabb": aabb,
                "center": [cx, cy, cz],
                "ref": obj,
            }
            if is_portal_object(obj):
                entry["kind"] = "gate"
            intersected_objects.append(entry)

    intersected_objects.sort(key=lambda x: (x["path_t"], x["path_perp"]))

    return {
        "start_xy": start_xy,
        "corridor_xy": (float(x_min), float(y_min), float(x_max), float(y_max)),
        "intersected_objects": intersected_objects,
        "target_obj_info": target_obj_info,
        "target_z": target_z,
    }


def run_single_leg_phase1_and_phase2(
    *,
    p: Any,
    placed_cubes: list[dict],
    mission_cmd: str | None,
    cur_instruction: str,
    g_world: np.ndarray,
    drone_pos: np.ndarray,
    drone_pos_local: np.ndarray,
    drone_R: np.ndarray,
    virtual_base_world: np.ndarray,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    treat_pos_as: str,
    delta_pos_scale: float,
    gripper_state: float,
    server_url: str,
    scene_catalog_str: str,
    xvla_scene_semantic_context: bool,
    xvla_path_planning_instruction_suffix: str,
    workspace_camera_width: int,
    workspace_camera_height: int,
    navigation_phase2_xvla_steps: int,
    xvla_act_request_timeout_s: float,
    navigation_phase2_sync_root_config: bool,
    navigation_phase2_sync_qs: bool,
    qs_policy_path_for_sync: Any | None,
    config_json_path: Any,
    navigation_phase2_extra_instruction: str,
    navigation_phase2_geom_astar: bool,
    navigation_phase2_astar_cell_m: float,
    navigation_phase2_astar_obstacle_pad_m: float,
    navigation_phase1_corridor_margin_m: float = 0.18,
    navigation_phase1_corridor_bandwidth_m: float = 0.12,
    navigation_collision_pad_m: float | None = None,
    navigation_phase2_optional_topdown_xvla: bool,
    navigation_phase2_z_clearance_enabled: bool,
    navigation_phase2_z_clearance_margin_m: float,
    navigation_phase2_z_workspace_margin_m: float,
    recording_scene_fov: float,
    recording_scene_margin: float,
    recording_camera_distance_scale: float,
    recording_topview_distance_scale: float,
    recording_stereo45_distance_scale: float = 1.5,
    rec_folder: Any | None,
    recording_folder: Any | None,
    root_dir: Any,
    world_recording_view_proj_fn: Callable[..., tuple[list[float], list[float]]],
    render_world_recording_rgb_fn: Callable[..., np.ndarray],
    world_xyz_to_recording_image_pixel_fn: Callable[
        [np.ndarray, list[float], list[float], int, int], tuple[int, int] | None
    ],
    first_rect_portal_for_instruction_color_fn: Callable[..., Any | None],
    build_proprio_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    query_xvla_fn: Callable[..., Any],
    compose_instruction_fn: Callable[..., str],
    read_config_fn: Callable[[Any], dict[str, Any]],
    load_qs_policies_fn: Callable[[Any], list[dict[str, str]]],
    leg_subfolder: str | None = None,
    target_portal_ref: dict | None = None,
    phase2_sidecar_name: str = "navigation_phase2_xy.json",
    phase2_png_filename: str = TOPDOWN_PNG,
    phase2_stereo_png_filename: str = STEREO45_PNG,
    leg_index: int | None = None,
    billboard_id: int | None = None,
    sync_root_config: bool | None = None,
    sync_qs: bool | None = None,
    pause_message: bool = False,
    create_feedback_spheres: bool = True,
    feedback_zones_registry: Phase3FeedbackZones | None = None,
) -> dict[str, Any] | None:

    leg_tag = ""
    if leg_index is not None:
        leg_tag = f" [leg {leg_index}]"
    if billboard_id is not None:
        leg_tag += f" billboard_id={billboard_id}"

    print("\n" + "=" * 60)
    print(f"[Phase 1]{leg_tag} top-down detection rectangle")
    p1 = compute_phase1_corridor_and_obstacles(
        drone_pos=drone_pos,
        g_world=g_world,
        placed_cubes=placed_cubes,
        mission_cmd=mission_cmd,
        cur_instruction=cur_instruction,
        first_rect_portal_for_instruction_color_fn=first_rect_portal_for_instruction_color_fn,
        target_portal_ref=target_portal_ref,
        corridor_margin_m=float(navigation_phase1_corridor_margin_m),
        corridor_bandwidth_m=float(navigation_phase1_corridor_bandwidth_m),
        feedback_radius_m=0.0,
    )
    start_xy = p1["start_xy"]
    x_min, y_min, x_max, y_max = p1["corridor_xy"]
    intersected_objects = p1["intersected_objects"]
    target_obj_info = p1["target_obj_info"]
    target_z = p1["target_z"]

    print(f"[Phase 1] start XY: {start_xy.round(3).tolist()}")
    print(
        f"[Phase 1] start XYZ (drone): "
        f"{np.asarray(drone_pos, dtype=np.float64).reshape(-1)[:3].round(3).tolist()}"
    )
    print(f"[Phase 1] goal XY: {np.asarray(g_world).reshape(-1)[:2].round(3).tolist()}")
    print(f"[Phase 1] detection rect [X] {x_min:.3f} to {x_max:.3f}, [Y] {y_min:.3f} to {y_max:.3f}")

    print("\n[Phase 1] goal point geometry:")
    print(f"  goal XYZ: {np.asarray(g_world).round(3).tolist()}")
    print(f"  goal Z height: {target_z:.3f}m")
    if target_obj_info is not None:
        print(f"  -> goal object: {target_obj_info['name']}")
        print(f"     object center XYZ: {np.array(target_obj_info['center']).round(3).tolist()}")

    print(
        f"\n[Phase 1] {len(intersected_objects)} obstacle(s) in detection rectangle "
        f"(encounter order start->goal; goal object excluded):"
    )
    for i, obj in enumerate(intersected_objects):
        print(
            f"  {i+1}. {obj['name']} | path_t={obj['path_t']:.3f} "
            f"perp={obj['path_perp']:.3f}m | dist_from_start={obj['dist']:.3f}m"
        )

    try:
        import cv2

        tw, th = 800, 800
        view_m, proj_m = world_recording_view_proj_fn(
            p,
            workspace_lo + virtual_base_world,
            workspace_hi + virtual_base_world,
            view_kind="top",
            width=tw,
            height=th,
            fov=float(recording_scene_fov),
            margin=float(recording_scene_margin),
            distance_scale=float(recording_camera_distance_scale),
            top_view_distance_scale=float(recording_topview_distance_scale),
        )
        topdown_img = render_world_recording_rgb_fn(p, view_m, proj_m, width=tw, height=th)

        experiment_base = resolve_experiment_base_folder(
            rec_folder=rec_folder,
            recording_folder=recording_folder,
            root_dir=root_dir,
        )
        phase1_folder = resolve_phase_folder(
            experiment_base, "phase1", leg_subfolder=leg_subfolder
        )
        phase2_folder = resolve_phase_folder(
            experiment_base, PHASE2_DIR, leg_subfolder=leg_subfolder
        )
        phase3_folder = resolve_phase_folder(experiment_base, "phase3", mkdir=False)

        z_ground = float(
            np.asarray(workspace_lo, dtype=np.float64)[2]
            + np.asarray(virtual_base_world, dtype=np.float64)[2]
        )

        def wproj(wx: float, wy: float, wz: float) -> tuple[int, int] | None:
            return world_xyz_to_recording_image_pixel_fn(
                np.array([wx, wy, wz], dtype=np.float64),
                view_m,
                proj_m,
                width=int(tw),
                height=int(th),
            )

        img_marked = cv2.cvtColor(topdown_img, cv2.COLOR_RGB2BGR)
        start_wz = float(np.asarray(drone_pos, dtype=np.float64).reshape(-1)[2])
        gv = np.asarray(g_world, dtype=np.float64).reshape(-1)
        obs_centers_g = [
            (float(o["center"][0]), float(o["center"][1]), float(o["center"][2]))
            for o in intersected_objects
        ]
        tgt_lbl = "TARGET"
        if billboard_id is not None:
            tgt_lbl = f"EXIT id={billboard_id}"
        _draw_phase1_detection_overlay_bgr(
            img_marked,
            wproj=wproj,
            x_min=float(x_min),
            y_min=float(y_min),
            x_max=float(x_max),
            y_max=float(y_max),
            z_ground=z_ground,
            start_xy=start_xy,
            start_wz=start_wz,
            target_xyz=gv,
            obstacle_centers=obs_centers_g,
            target_label=tgt_lbl,
        )
        draw_portal_number_labels_overlay_bgr(img_marked, wproj, placed_cubes)
        view_stereo_g, proj_stereo_g = world_recording_view_proj_fn(
            p,
            workspace_lo + virtual_base_world,
            workspace_hi + virtual_base_world,
            view_kind="45deg",
            width=tw,
            height=th,
            fov=float(recording_scene_fov),
            margin=float(recording_scene_margin),
            distance_scale=float(recording_camera_distance_scale),
            top_view_distance_scale=float(recording_topview_distance_scale),
            stereo45_view_distance_scale=float(recording_stereo45_distance_scale),
        )
        stereo_rgb_g = render_world_recording_rgb_fn(
            p, view_stereo_g, proj_stereo_g, width=tw, height=th
        )
        img_stereo_marked_g = cv2.cvtColor(stereo_rgb_g, cv2.COLOR_RGB2BGR)

        def wproj_stereo_g(wx: float, wy: float, wz: float) -> tuple[int, int] | None:
            return world_xyz_to_recording_image_pixel_fn(
                np.array([wx, wy, wz], dtype=np.float64),
                view_stereo_g,
                proj_stereo_g,
                width=int(tw),
                height=int(th),
            )

        _draw_phase1_detection_overlay_bgr(
            img_stereo_marked_g,
            wproj=wproj_stereo_g,
            x_min=float(x_min),
            y_min=float(y_min),
            x_max=float(x_max),
            y_max=float(y_max),
            z_ground=z_ground,
            start_xy=start_xy,
            start_wz=start_wz,
            target_xyz=gv,
            obstacle_centers=obs_centers_g,
            target_label=tgt_lbl,
        )
        det_name = TOPDOWN_PNG
        phase1_saved = save_bgr_views(phase1_folder, img_marked, img_stereo_marked_g, topdown_name=det_name)
        print(f"\n[Phase 1] phase1 recordings saved (light-green rect = detection corridor):")
        print(f"  top-down: {phase1_saved.get('topdown')}")
        if "stereo45deg" in phase1_saved:
            print(f"  stereo 45deg: {phase1_saved['stereo45deg']}")

        phase3_entries: list[dict[str, Any]] = []
        phase3_zones: Phase3FeedbackZones | None = feedback_zones_registry
        try:
            phase3_entries = collect_phase3_feedback_entries(
                intersected_objects,
                target_obj_info,
            )
            if create_feedback_spheres and phase3_entries:
                phase3_zones = setup_phase1_feedback_spheres(
                    p,
                    intersected_objects,
                    target_obj_info,
                    registry=feedback_zones_registry,
                )
        except Exception as pe1fb:
            print(f"[Phase 1] WARN: feedback sphere setup failed - {pe1fb}")

        obs_lines_parts_gcam: list[str] = []
        for idx, obj in enumerate(intersected_objects):
            aabb_o = obj["aabb"]
            lo_c = np.asarray(aabb_o[0], dtype=np.float64)
            hi_c = np.asarray(aabb_o[1], dtype=np.float64)
            nm = str(obj.get("name", f"obj_{idx + 1}"))
            obs_lines_parts_gcam.append(
                f"  - {nm}: XY low [{float(lo_c[0]):.4f},{float(lo_c[1]):.4f}] "
                f"high [{float(hi_c[0]):.4f},{float(hi_c[1]):.4f}]"
            )
        obs_lines_txt_gcam = (
            "\n".join(obs_lines_parts_gcam)
            if obs_lines_parts_gcam
            else "  (none — corridor contains only the goal footprint)"
        )

        do_sync_cfg = (
            bool(sync_root_config)
            if sync_root_config is not None
            else bool(navigation_phase2_sync_root_config)
        )
        do_sync_qs = (
            bool(sync_qs) if sync_qs is not None else bool(navigation_phase2_sync_qs)
        )

        print("\n" + "=" * 60)
        print(f"[Phase 2]{leg_tag} corridor + A* coarse path...")
        p2 = run_navigation_phase2_topdown_xvla_xy_plan(
            server_url=server_url,
            topdown_rgb=topdown_img,
            img_marked_bgr=img_marked,
            view_m=view_m,
            proj_m=proj_m,
            render_width=tw,
            render_height=th,
            phase2_folder=phase2_folder,
            png_filename=str(phase2_png_filename),
            stereo_img_marked_bgr=img_stereo_marked_g,
            stereo_view_m=view_stereo_g,
            stereo_proj_m=proj_stereo_g,
            stereo_png_filename=str(phase2_stereo_png_filename),
            drone_pos_world=np.asarray(drone_pos, dtype=np.float32),
            drone_pos_local=np.asarray(drone_pos_local, dtype=np.float32),
            drone_R=drone_R,
            virtual_base_world=virtual_base_world,
            workspace_lo=workspace_lo,
            workspace_hi=workspace_hi,
            treat_pos_as=treat_pos_as,
            delta_pos_scale=delta_pos_scale,
            gripper_state=gripper_state,
            mission_cmd=mission_cmd,
            language_instruction=cur_instruction,
            obs_lines_text=obs_lines_txt_gcam,
            corridor_xy=(float(x_min), float(y_min), float(x_max), float(y_max)),
            goal_xyz_world=np.asarray(g_world, dtype=np.float64),
            scene_catalog_str=scene_catalog_str,
            xvla_scene_semantic_context=bool(xvla_scene_semantic_context),
            xvla_path_planning_instruction_suffix=xvla_path_planning_instruction_suffix,
            workspace_camera_width=int(workspace_camera_width),
            workspace_camera_height=int(workspace_camera_height),
            phase2_steps=int(navigation_phase2_xvla_steps),
            xvla_act_request_timeout_s=float(xvla_act_request_timeout_s),
            sync_root_config=do_sync_cfg,
            sync_qs=do_sync_qs,
            qs_policy_path=qs_policy_path_for_sync,
            config_json_path=config_json_path,
            phase2_extra_instruction=str(navigation_phase2_extra_instruction),
            phase1_obstacle_entries=list(intersected_objects),
            world_xyz_to_recording_image_pixel_fn=world_xyz_to_recording_image_pixel_fn,
            build_proprio_fn=build_proprio_fn,
            query_xvla_fn=query_xvla_fn,
            compose_instruction_fn=compose_instruction_fn,
            read_config_fn=read_config_fn,
            load_qs_policies_fn=load_qs_policies_fn,
            navigation_phase2_geom_astar=bool(navigation_phase2_geom_astar),
            navigation_phase2_astar_cell_m=float(navigation_phase2_astar_cell_m),
            navigation_phase2_astar_obstacle_pad_m=float(navigation_phase2_astar_obstacle_pad_m),
            navigation_phase2_optional_topdown_xvla=bool(navigation_phase2_optional_topdown_xvla),
            navigation_phase2_z_clearance_enabled=bool(navigation_phase2_z_clearance_enabled),
            navigation_phase2_z_clearance_margin_m=float(navigation_phase2_z_clearance_margin_m),
            navigation_phase2_z_workspace_margin_m=float(navigation_phase2_z_workspace_margin_m),
            navigation_collision_pad_m=navigation_collision_pad_m,
            phase2_sidecar_name=str(phase2_sidecar_name),
            leg_index=leg_index,
            billboard_id=billboard_id,
            pause_message=pause_message,
            placed_cubes=placed_cubes,
        )
        return {
            "trajectory": p2.get("trajectory"),
            "phase2_snapshot": p2.get("snapshot"),
            "planner": p2.get("planner"),
            "experiment_base": experiment_base,
            "phase1_folder": phase1_folder,
            "phase2_folder": phase2_folder,
            "phase3_folder": phase3_folder,
            "img_marked_bgr": img_marked,
            "img_stereo_bgr": img_stereo_marked_g,
            "wproj": wproj,
            "wproj_stereo": wproj_stereo_g,
            "intersected_objects": intersected_objects,
            "target_obj_info": target_obj_info,
            "phase3_entries": phase3_entries,
            "phase3_zones": phase3_zones,
        }
    except Exception as e:
        print(f"[Phase 1+2] WARN: leg execution failed - {e}")
        return None


def run_navigation_phase1_and_phase2_topdown(
    *,
    p: Any,
    placed_cubes: list[dict],
    mission_cmd: str | None,
    cur_instruction: str,
    g_world: np.ndarray,
    drone_pos: np.ndarray,
    drone_pos_local: np.ndarray,
    drone_R: np.ndarray,
    virtual_base_world: np.ndarray,
    workspace_lo: np.ndarray,
    workspace_hi: np.ndarray,
    treat_pos_as: str,
    delta_pos_scale: float,
    gripper_state: float,
    server_url: str,
    scene_catalog_str: str,
    xvla_scene_semantic_context: bool,
    xvla_path_planning_instruction_suffix: str,
    workspace_camera_width: int,
    workspace_camera_height: int,
    navigation_phase2_xvla_steps: int,
    xvla_act_request_timeout_s: float,
    navigation_phase2_sync_root_config: bool,
    navigation_phase2_sync_qs: bool,
    qs_policy_path_for_sync: Any | None,
    config_json_path: Any,
    navigation_phase2_extra_instruction: str,
    navigation_phase2_geom_astar: bool,
    navigation_phase2_astar_cell_m: float,
    navigation_phase2_astar_obstacle_pad_m: float,
    navigation_phase1_corridor_margin_m: float = 0.18,
    navigation_phase1_corridor_bandwidth_m: float = 0.12,
    navigation_collision_pad_m: float | None = None,
    navigation_phase2_optional_topdown_xvla: bool,
    navigation_phase2_z_clearance_enabled: bool,
    navigation_phase2_z_clearance_margin_m: float,
    navigation_phase2_z_workspace_margin_m: float,
    recording_scene_fov: float,
    recording_scene_margin: float,
    recording_camera_distance_scale: float,
    recording_topview_distance_scale: float,
    recording_stereo45_distance_scale: float = 1.5,
    rec_folder: Any | None,
    recording_folder: Any | None,
    root_dir: Any,
    world_recording_view_proj_fn: Callable[..., tuple[list[float], list[float]]],
    render_world_recording_rgb_fn: Callable[..., np.ndarray],
    world_xyz_to_recording_image_pixel_fn: Callable[
        [np.ndarray, list[float], list[float], int, int], tuple[int, int] | None
    ],
    first_rect_portal_for_instruction_color_fn: Callable[..., Any | None],
    build_proprio_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    query_xvla_fn: Callable[..., Any],
    compose_instruction_fn: Callable[..., str],
    read_config_fn: Callable[[Any], dict[str, Any]],
    load_qs_policies_fn: Callable[[Any], list[dict[str, str]]],
    portal_leg_goal_fn: Callable[[dict, np.ndarray], np.ndarray] | None = None,
    create_feedback_spheres: bool = True,
    feedback_zones_registry: Phase3FeedbackZones | None = None,
) -> dict[str, Any] | None:

    instruction = str(mission_cmd or cur_instruction or "")
    mission_ids = extract_ordered_mission_billboard_ids(instruction)
    billboard_ids = extract_ordered_traversal_billboard_ids(instruction)

    action_cache = None
    if feedback_zones_registry is not None and feedback_zones_registry.action_cache:
        action_cache = dict(feedback_zones_registry.action_cache)

    common_kw = dict(
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
        xvla_scene_semantic_context=xvla_scene_semantic_context,
        xvla_path_planning_instruction_suffix=xvla_path_planning_instruction_suffix,
        workspace_camera_width=workspace_camera_width,
        workspace_camera_height=workspace_camera_height,
        navigation_phase2_xvla_steps=navigation_phase2_xvla_steps,
        xvla_act_request_timeout_s=xvla_act_request_timeout_s,
        navigation_phase2_sync_root_config=navigation_phase2_sync_root_config,
        navigation_phase2_sync_qs=navigation_phase2_sync_qs,
        qs_policy_path_for_sync=qs_policy_path_for_sync,
        config_json_path=config_json_path,
        navigation_phase2_extra_instruction=navigation_phase2_extra_instruction,
        navigation_phase2_geom_astar=navigation_phase2_geom_astar,
        navigation_phase2_astar_cell_m=navigation_phase2_astar_cell_m,
        navigation_phase1_corridor_margin_m=navigation_phase1_corridor_margin_m,
        navigation_phase1_corridor_bandwidth_m=navigation_phase1_corridor_bandwidth_m,
        navigation_collision_pad_m=navigation_collision_pad_m,
        navigation_phase2_astar_obstacle_pad_m=navigation_phase2_astar_obstacle_pad_m,
        navigation_phase2_optional_topdown_xvla=navigation_phase2_optional_topdown_xvla,
        navigation_phase2_z_clearance_enabled=navigation_phase2_z_clearance_enabled,
        navigation_phase2_z_clearance_margin_m=navigation_phase2_z_clearance_margin_m,
        navigation_phase2_z_workspace_margin_m=navigation_phase2_z_workspace_margin_m,
        recording_scene_fov=recording_scene_fov,
        recording_scene_margin=recording_scene_margin,
        recording_camera_distance_scale=recording_camera_distance_scale,
        recording_topview_distance_scale=recording_topview_distance_scale,
        recording_stereo45_distance_scale=recording_stereo45_distance_scale,
        rec_folder=rec_folder,
        recording_folder=recording_folder,
        root_dir=root_dir,
        world_recording_view_proj_fn=world_recording_view_proj_fn,
        render_world_recording_rgb_fn=render_world_recording_rgb_fn,
        world_xyz_to_recording_image_pixel_fn=world_xyz_to_recording_image_pixel_fn,
        first_rect_portal_for_instruction_color_fn=first_rect_portal_for_instruction_color_fn,
        build_proprio_fn=build_proprio_fn,
        query_xvla_fn=query_xvla_fn,
        compose_instruction_fn=compose_instruction_fn,
        read_config_fn=read_config_fn,
        load_qs_policies_fn=load_qs_policies_fn,
    )

    if len(mission_ids) >= 2:
        from .multi_leg import run_multi_action_navigation_phase12

        multi_action = run_multi_action_navigation_phase12(
            mission_cmd=instruction,
            phase12_kwargs=common_kw,
            initial_drone_pos=np.asarray(drone_pos, dtype=np.float64),
            create_feedback_spheres=create_feedback_spheres,
            feedback_zones_registry=feedback_zones_registry,
            action_cache=action_cache,
        )
        if multi_action is not None:
            print("[Phase 1+2] Multi-action planning complete.")
            return multi_action

    if len(billboard_ids) >= 2 and portal_leg_goal_fn is not None:
        from .multi_leg import run_multi_leg_navigation_phase12

        multi_result = run_multi_leg_navigation_phase12(
            billboard_ids=billboard_ids,
            portal_leg_goal_fn=portal_leg_goal_fn,
            phase12_kwargs=common_kw,
            initial_drone_pos=np.asarray(drone_pos, dtype=np.float64),
            create_feedback_spheres=create_feedback_spheres,
            feedback_zones_registry=feedback_zones_registry,
        )
        if multi_result is None:
            return None
        print("[Phase 1+2] Multi-leg planning complete.")
        return multi_result

    single = run_single_leg_phase1_and_phase2(
        **common_kw,
        pause_message=False,
        create_feedback_spheres=create_feedback_spheres,
        feedback_zones_registry=feedback_zones_registry,
    )
    if single is None:
        return None
    print("[Phase 1+2] Single-leg planning complete.")
    return single
