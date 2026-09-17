#!/usr/bin/env python3
"""Repair ground-contact sliding using the complete MuJoCo XML and meshes.

The source motion is treated as a kinematic trajectory.  For every sustained
ground-contact window, this pass stores the first world-space contact point
for each contacting geom and solves the ancestor hinge joints (plus a bounded
root translation correction) so that the contact point stays fixed in x/y and
does not penetrate the floor.

This script is intentionally separate from the source PKL: it always writes a
new PKL and a JSON report.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("请使用安装了 MuJoCo Python binding 的虚拟环境运行本脚本。") from exc


MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
FREE = int(mujoco.mjtJoint.mjJNT_FREE)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Use exact MuJoCo mesh/box-ground contacts to reduce sliding."
    )
    p.add_argument("--xml", required=True, help="完整 XML 路径")
    p.add_argument("--motion", required=True, help="输入 PKL 路径")
    p.add_argument("--out", required=True, help="输出 PKL 路径")
    p.add_argument("--report", required=True, help="JSON 报告路径")
    p.add_argument("--contact-threshold-mm", type=float, default=8.0)
    p.add_argument("--floor-margin-mm", type=float, default=1.0)
    p.add_argument(
        "--support-body",
        default=None,
        help="只锁定指定支撑刚体的接触点，例如 left_ankle_roll_link",
    )
    p.add_argument("--min-contact-frames", type=int, default=4)
    p.add_argument("--max-contacts-per-frame", type=int, default=8)
    p.add_argument(
        "--max-support-bodies-per-frame",
        type=int,
        default=2,
        help="每帧最多锁定几个不同支撑刚体，默认 2；避免躺倒时多接触点互相冲突",
    )
    p.add_argument("--max-nfev", type=int, default=20)
    p.add_argument("--max-step-rad", type=float, default=0.06)
    p.add_argument(
        "--max-root-correction-mm",
        type=float,
        default=80.0,
        help="允许的根部平移修正上限；默认 80 mm",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="接触扫描步长，完整修复使用 1",
    )
    return p.parse_args()


def object_name(model: Any, object_type: Any, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, int(object_id))
    return name if name is not None else f"<unnamed:{int(object_id)}>"


def enable_native_ccd(model: Any) -> Tuple[bool, str]:
    bit = getattr(getattr(mujoco, "mjtEnableBit", object()), "mjENBL_NATIVECCD", None)
    if bit is None:
        return False, "native CCD 不可用，使用默认 CCD"
    try:
        model.opt.enableflags |= int(bit)
        return True, "native CCD enabled"
    except Exception as exc:
        return False, f"native CCD 启用失败：{exc}"


def load_motion(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        motion = pickle.load(handle)
    if not isinstance(motion, dict):
        raise ValueError("PKL 顶层对象必须是 dict")
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key not in motion:
            raise KeyError(f"PKL 缺少字段 {key}")
    return motion


def qpos_from_motion(
    model: Any,
    data: Any,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_values: np.ndarray,
    dof_names: Sequence[str],
) -> None:
    free_id = next(
        (jid for jid in range(model.njnt) if int(model.jnt_type[jid]) == FREE),
        -1,
    )
    if free_id < 0:
        raise ValueError("XML 中找不到 free joint")
    root_adr = int(model.jnt_qposadr[free_id])
    data.qpos[root_adr : root_adr + 3] = root_pos
    data.qpos[root_adr + 3 : root_adr + 7] = root_rot
    for col, name in enumerate(dof_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"XML 中找不到动作关节：{name}")
        if int(model.jnt_type[jid]) != HINGE:
            raise ValueError(f"动作关节不是 hinge：{name}")
        data.qpos[int(model.jnt_qposadr[jid])] = dof_values[col]


def geom_distance(
    model: Any,
    data: Any,
    geom_id: int,
    floor_id: int,
    max_distance: float,
) -> Tuple[float, np.ndarray]:
    fromto = np.zeros(6, dtype=np.float64)
    distance = float(
        mujoco.mj_geomDistance(model, data, int(geom_id), int(floor_id), max_distance, fromto)
    )
    # geom_id is always passed first, so fromto[:3] is the point on the
    # contacting robot geom and fromto[3:] is the point on the plane.
    return distance, fromto


def all_ground_geoms(model: Any, floor_ids: Sequence[int]) -> List[int]:
    floor_set = set(int(x) for x in floor_ids)
    result = []
    for gid in range(model.ngeom):
        if gid in floor_set:
            continue
        # Include mesh and explicit collision primitives even when their
        # normal contype is disabled; this is a direct diagnostic query.
        result.append(gid)
    return result


def ancestor_hinge_columns(
    model: Any,
    geom_id: int,
    dof_names: Sequence[str],
) -> List[int]:
    name_to_column = {str(name): index for index, name in enumerate(dof_names)}
    body = int(model.geom_bodyid[geom_id])
    columns: List[int] = []
    while body > 0:
        first = int(model.body_jntadr[body])
        count = int(model.body_jntnum[body])
        for jid in range(first, first + count):
            if int(model.jnt_type[jid]) != HINGE:
                continue
            name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name in name_to_column:
                columns.append(name_to_column[name])
        body = int(model.body_parentid[body])
    return sorted(set(columns))


def joint_bounds(model: Any, dof_names: Sequence[str], columns: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for col in columns:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(dof_names[col]))
        if jid < 0 or int(model.jnt_type[jid]) != HINGE:
            raise ValueError(f"无法为关节建立边界：{dof_names[col]}")
        lower.append(float(model.jnt_range[jid, 0]))
        upper.append(float(model.jnt_range[jid, 1]))
    return np.asarray(lower), np.asarray(upper)


def set_frame(
    model: Any,
    data: Any,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    q: np.ndarray,
    dof_names: Sequence[str],
) -> None:
    qpos_from_motion(model, data, root_pos, root_rot, q, dof_names)
    mujoco.mj_forward(model, data)


def scan_ground_contacts(
    model: Any,
    data: Any,
    roots: np.ndarray,
    root_rots: np.ndarray,
    q: np.ndarray,
    dof_names: Sequence[str],
    ground_geoms: Sequence[int],
    floor_id: int,
    threshold: float,
    stride: int,
) -> List[Dict[int, Dict[str, Any]]]:
    frames = len(q)
    contacts: List[Dict[int, Dict[str, Any]]] = [dict() for _ in range(frames)]
    for frame in range(0, frames, stride):
        set_frame(model, data, roots[frame], root_rots[frame], q[frame], dof_names)
        candidates = []
        for gid in ground_geoms:
            distance, fromto = geom_distance(model, data, gid, floor_id, threshold)
            if distance < threshold - 1e-10:
                candidates.append((distance, gid, fromto.copy()))
        # Keep the lowest contacts first. This avoids making a single frame
        # over-constrained when several visual mesh pieces touch the plane.
        candidates.sort(key=lambda item: item[0])
        for distance, gid, fromto in candidates:
            contacts[frame][int(gid)] = {
                "distance_m": float(distance),
                "point_world": [float(x) for x in fromto[:3]],
                "fromto_world": [float(x) for x in fromto],
            }
    if stride > 1:
        for frame in range(frames):
            if contacts[frame]:
                continue
            nearest = frame - (frame % stride)
            contacts[frame] = copy.deepcopy(contacts[nearest])
    return contacts


def sustained_contact_frames(
    contacts: Sequence[Dict[int, Dict[str, Any]]],
    min_frames: int,
) -> Dict[int, List[Tuple[int, int]]]:
    by_geom: Dict[int, List[bool]] = {}
    for frame_contacts in contacts:
        for gid in frame_contacts:
            by_geom.setdefault(gid, []).append(True)
    # Rebuild with explicit frame length so missing entries are false.
    frame_count = len(contacts)
    sustained: Dict[int, List[Tuple[int, int]]] = {}
    all_gids = sorted({gid for item in contacts for gid in item})
    for gid in all_gids:
        mask = np.asarray([gid in contacts[frame] for frame in range(frame_count)], dtype=bool)
        changes = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)) != 0)
        windows = []
        for start, end in zip(changes[::2], changes[1::2]):
            start, end = int(start), int(end)
            if end - start >= min_frames:
                windows.append((start, end))
        sustained[gid] = windows
    return sustained


def window_for_frame(windows: Sequence[Tuple[int, int]], frame: int) -> Tuple[int, int] | None:
    for start, end in windows:
        if start <= frame < end:
            return start, end
    return None


def contact_priority(model: Any, geom_id: int) -> int:
    """Prefer actual foot collision geoms over incidental mesh contacts."""
    name = object_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).lower()
    if "ankle_roll_collision" in name or "foot" in name or "sole" in name:
        return 0
    if "ankle" in name:
        return 1
    if "hand" in name or "elbow" in name:
        return 2
    if "knee" in name:
        return 3
    if "torso" in name or "pelvis" in name:
        return 4
    return 5


def select_support_contacts(
    model: Any,
    candidates: Sequence[Tuple[float, int, np.ndarray]],
    max_contacts: int,
    max_support_bodies: int,
) -> List[Tuple[float, int, np.ndarray]]:
    """Reduce a lying-body contact manifold to feasible support constraints.

    Multiple mesh pieces on one body are often simultaneously within the
    contact threshold. Locking all of them makes a kinematic retarget
    over-constrained and produces large residuals. Keep the best geom per
    body, then keep only the strongest support bodies.
    """
    best_by_body: Dict[int, Tuple[float, int, np.ndarray]] = {}
    for candidate in candidates:
        distance, geom_id, fromto = candidate
        body = int(model.geom_bodyid[geom_id])
        previous = best_by_body.get(body)
        score = (contact_priority(model, geom_id), float(distance))
        if previous is None:
            best_by_body[body] = candidate
        else:
            previous_score = (contact_priority(model, previous[1]), float(previous[0]))
            if score < previous_score:
                best_by_body[body] = candidate
    selected = sorted(
        best_by_body.values(),
        key=lambda item: (contact_priority(model, item[1]), float(item[0])),
    )
    return selected[: max(1, min(int(max_contacts), int(max_support_bodies)))]


def make_target_map(
    contacts: Sequence[Dict[int, Dict[str, Any]]],
    sustained: Dict[int, List[Tuple[int, int]]],
) -> Dict[Tuple[int, int], np.ndarray]:
    targets: Dict[Tuple[int, int], np.ndarray] = {}
    for gid, windows in sustained.items():
        for start, _ in windows:
            if gid not in contacts[start]:
                continue
            point = np.asarray(contacts[start][gid]["point_world"], dtype=float)
            targets[(gid, start)] = point
    return targets


def optimize_frame(
    model: Any,
    data: Any,
    root_source: np.ndarray,
    root_rot: np.ndarray,
    q_source: np.ndarray,
    q_initial: np.ndarray,
    dof_names: Sequence[str],
    active_contacts: Sequence[Tuple[int, np.ndarray]],
    variable_columns: Sequence[int],
    floor_id: int,
    previous_root_delta: np.ndarray,
    max_root_delta: float,
    floor_margin: float,
    max_nfev: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    variable_columns = list(variable_columns)
    lower, upper = joint_bounds(model, dof_names, variable_columns)
    lower_x = np.concatenate((lower, np.full(3, -max_root_delta)))
    upper_x = np.concatenate((upper, np.full(3, max_root_delta)))
    # Some retargeted recovery frames can sit a few ulps outside the MJCF
    # hinge limits. scipy.optimize.least_squares rejects such an initial
    # point, so clip only the optimizer's local copy; the source PKL remains
    # untouched and the final q is kept within the XML limits for optimized
    # joints.
    q_initial_local = q_initial.copy()
    q_initial_local[variable_columns] = np.clip(
        q_initial_local[variable_columns], lower, upper
    )
    x0 = np.concatenate((q_initial_local[variable_columns], previous_root_delta))

    def evaluate(x: np.ndarray) -> Tuple[List[float], List[Tuple[float, np.ndarray]]]:
        q_trial = q_initial_local.copy()
        q_trial[variable_columns] = x[: len(variable_columns)]
        root_trial = root_source + x[len(variable_columns) :]
        set_frame(model, data, root_trial, root_rot, q_trial, dof_names)
        values: List[float] = []
        observed: List[Tuple[float, np.ndarray]] = []
        for gid, target in active_contacts:
            distance, fromto = geom_distance(model, data, gid, floor_id, 0.20)
            point = fromto[:3]
            observed.append((distance, point.copy()))
            values.extend(((point[:2] - target[:2]) / 0.001).tolist())
            # Keep the residual vector length constant for every optimizer
            # evaluation. A zero value means the contact already has at least
            # the requested floor clearance.
            values.append(float(max(0.0, floor_margin - distance) / 0.001))
        return values, observed

    def residual(x: np.ndarray) -> np.ndarray:
        contact_residual, _ = evaluate(x)
        if not contact_residual:
            contact_residual = [0.0]
        q_values = x[: len(variable_columns)]
        root_delta = x[len(variable_columns) :]
        q_source_values = q_source[variable_columns]
        q_previous_values = q_initial_local[variable_columns]
        # Contact error is in millimetres. Regularizers are deliberately
        # weaker, but discourage large leg poses and root shifts.
        regularization = np.concatenate(
            (
                (q_values - q_source_values) / 0.10,
                (q_values - q_previous_values) / 0.06,
                root_delta / 0.025,
                (root_delta - previous_root_delta) / 0.020,
            )
        )
        return np.concatenate((np.asarray(contact_residual, dtype=float), regularization))

    if not variable_columns and not active_contacts:
        return q_initial.copy(), previous_root_delta.copy(), 0.0, 0.0

    result = least_squares(
        residual,
        x0,
        bounds=(lower_x, upper_x),
        max_nfev=max(1, int(max_nfev)),
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
        x_scale="jac",
    )
    q_out = q_initial_local.copy()
    q_out[variable_columns] = result.x[: len(variable_columns)]
    root_delta = result.x[len(variable_columns) :].copy()
    contact_residual, observed = evaluate(result.x)
    max_contact_error = 0.0
    min_distance = float("inf")
    for (gid, target), (_, point) in zip(active_contacts, observed):
        max_contact_error = max(max_contact_error, float(np.linalg.norm(point[:2] - target[:2])))
        distance, _ = geom_distance(model, data, gid, floor_id, 0.20)
        min_distance = min(min_distance, distance)
    return q_out, root_delta, max_contact_error, min_distance


def rescan_contact_paths(
    model: Any,
    data: Any,
    roots: np.ndarray,
    root_rots: np.ndarray,
    q: np.ndarray,
    dof_names: Sequence[str],
    floor_id: int,
    sustained: Dict[int, List[Tuple[int, int]]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for gid, windows in sustained.items():
        geom_result = []
        for start, end in windows:
            points = []
            distances = []
            for frame in range(start, end):
                set_frame(model, data, roots[frame], root_rots[frame], q[frame], dof_names)
                distance, fromto = geom_distance(model, data, gid, floor_id, 0.20)
                points.append(fromto[:3].copy())
                distances.append(distance)
            points_array = np.asarray(points)
            xy_drift = np.linalg.norm(points_array[:, :2] - points_array[0, :2], axis=1)
            path = np.linalg.norm(np.diff(points_array[:, :2], axis=0), axis=1).sum()
            geom_result.append(
                {
                    "start": start,
                    "end_exclusive": end,
                    "geom_id": int(gid),
                    "geom": object_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                    "body": object_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])),
                    "max_xy_drift_mm": float(xy_drift.max() * 1000.0),
                    "path_length_mm": float(path * 1000.0),
                    "minimum_signed_distance_mm": float(min(distances) * 1000.0),
                }
            )
        result[str(gid)] = geom_result
    return result


def main() -> int:
    cli = args()
    if cli.stride < 1:
        raise SystemExit("--stride 必须 >= 1")
    xml_path = Path(cli.xml).expanduser().resolve()
    source_path = Path(cli.motion).expanduser().resolve()
    output_path = Path(cli.out).expanduser().resolve()
    report_path = Path(cli.report).expanduser().resolve()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    native_ccd, ccd_message = enable_native_ccd(model)
    data = mujoco.MjData(model)
    motion = load_motion(source_path)
    q_source = np.asarray(motion["dof_pos"], dtype=float)
    root_source = np.asarray(motion["root_pos"], dtype=float)
    root_rots = np.asarray(motion["root_rot"], dtype=float)
    dof_names = [str(name) for name in motion.get("dof_names", [])]
    if not dof_names:
        dof_names = [
            object_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            for jid in range(model.njnt)
            if int(model.jnt_type[jid]) == HINGE
        ]
    if len(dof_names) != q_source.shape[1]:
        raise ValueError(f"dof_names 数量 {len(dof_names)} != dof_pos 列数 {q_source.shape[1]}")
    if root_rots.shape[1] != 4:
        raise ValueError(f"root_rot 形状错误：{root_rots.shape}")
    frames = min(len(q_source), len(root_source), len(root_rots))
    if frames == 0:
        raise ValueError("动作没有帧")

    floor_ids = [gid for gid in range(model.ngeom) if int(model.geom_type[gid]) == PLANE]
    if not floor_ids:
        raise ValueError("XML 中没有 plane 地面")
    floor_id = next(
        (gid for gid in floor_ids if object_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "floor"),
        floor_ids[0],
    )
    ground_geoms = all_ground_geoms(model, floor_ids)
    threshold = max(0.0, float(cli.contact_threshold_mm) * 1e-3)
    floor_margin = max(0.0, float(cli.floor_margin_mm) * 1e-3)

    print(f"XML: {xml_path}")
    print(f"motion: {source_path}")
    print(f"frames: {frames}, fps: {float(motion.get('fps', 50.0)):g}")
    print(f"ground geoms: {len(ground_geoms)}, floor geom: {object_name(model, mujoco.mjtObj.mjOBJ_GEOM, floor_id)}")
    if cli.support_body:
        print(f"support body filter: {cli.support_body}")
    print(ccd_message)
    print("scan source ground contacts")
    source_contacts = scan_ground_contacts(
        model,
        data,
        root_source[:frames],
        root_rots[:frames],
        q_source[:frames],
        dof_names,
        ground_geoms,
        floor_id,
        threshold,
        cli.stride,
    )
    sustained = sustained_contact_frames(source_contacts, max(2, int(cli.min_contact_frames)))
    sustained = {gid: windows for gid, windows in sustained.items() if windows}
    target_map = make_target_map(source_contacts, sustained)
    print(f"sustained contact geoms: {len(sustained)}")
    print(f"sustained contact windows: {sum(len(v) for v in sustained.values())}")

    q_out = q_source[:frames].copy()
    root_out = root_source[:frames].copy()
    root_delta_previous = np.zeros(3, dtype=float)
    frame_error: List[float] = []
    frame_min_distance: List[float] = []
    optimized_frames = 0
    max_root_delta = max(0.0, float(cli.max_root_correction_mm) * 1e-3)

    for frame in range(frames):
        active_candidates = []
        for gid, windows in sustained.items():
            if cli.support_body:
                body_name = object_name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[gid]),
                )
                if body_name != cli.support_body:
                    continue
            window = window_for_frame(windows, frame)
            if window is None:
                continue
            start, _ = window
            target = target_map.get((gid, start))
            if target is not None:
                distance = source_contacts[frame].get(gid, {}).get("distance_m", threshold)
                active_candidates.append((float(distance), int(gid), target))
        active_candidates.sort(key=lambda item: item[0])
        active_candidates = select_support_contacts(
            model,
            active_candidates,
            max_contacts=max(1, int(cli.max_contacts_per_frame)),
            max_support_bodies=max(1, int(cli.max_support_bodies_per_frame)),
        )
        active_contacts = [(gid, target) for _, gid, target in active_candidates]

        if active_contacts:
            variable_columns = sorted(
                {
                    column
                    for gid, _ in active_contacts
                    for column in ancestor_hinge_columns(model, gid, dof_names)
                }
            )
            # If only a pelvis/torso contact is active, the root translation is
            # the only available kinematic correction; it is already included
            # in the bounded optimizer variables.
            q_frame, root_delta, error, min_distance = optimize_frame(
                model,
                data,
                root_source[frame],
                root_rots[frame],
                q_source[frame],
                q_out[frame],
                dof_names,
                active_contacts,
                variable_columns,
                floor_id,
                root_delta_previous,
                max_root_delta,
                floor_margin,
                cli.max_nfev,
            )
            q_out[frame] = q_frame
            root_out[frame] = root_source[frame] + root_delta
            root_delta_previous = root_delta
            frame_error.append(error)
            frame_min_distance.append(min_distance)
            optimized_frames += 1
        else:
            # Let root corrections decay after a contact window, avoiding a
            # visible positional jump when the body leaves the ground.
            root_delta_previous *= 0.5
            root_out[frame] = root_source[frame] + root_delta_previous
            frame_error.append(0.0)
            frame_min_distance.append(0.20)

        if frame > 0:
            q_out[frame] = q_out[frame - 1] + np.clip(
                q_out[frame] - q_out[frame - 1],
                -float(cli.max_step_rad),
                float(cli.max_step_rad),
            )
        if frame == 0 or frame % 100 == 0 or frame == frames - 1:
            print(f"repair frame {frame + 1}/{frames}", flush=True)

    print("rescan repaired ground contact paths")
    repaired_paths = rescan_contact_paths(
        model,
        data,
        root_out,
        root_rots[:frames],
        q_out,
        dof_names,
        floor_id,
        sustained,
    )

    q_step = np.max(np.abs(np.diff(q_out, axis=0)), axis=1) if frames > 1 else np.zeros(0)
    root_step = np.linalg.norm(np.diff(root_out, axis=0), axis=1) if frames > 1 else np.zeros(0)
    root_delta = root_out - root_source[:frames]
    report = {
        "source": str(source_path),
        "output": str(output_path),
        "xml": str(xml_path),
        "frames": frames,
        "fps": float(motion.get("fps", 50.0)),
        "floor_geom": object_name(model, mujoco.mjtObj.mjOBJ_GEOM, floor_id),
        "ground_geom_count": len(ground_geoms),
        "contact_threshold_mm": float(cli.contact_threshold_mm),
        "floor_margin_mm": float(cli.floor_margin_mm),
        "support_body": cli.support_body,
        "min_contact_frames": int(cli.min_contact_frames),
        "max_contacts_per_frame": int(cli.max_contacts_per_frame),
        "max_support_bodies_per_frame": int(cli.max_support_bodies_per_frame),
        "native_ccd": native_ccd,
        "native_ccd_message": ccd_message,
        "optimized_frames": optimized_frames,
        "sustained_contact_geoms": len(sustained),
        "sustained_contact_windows": sum(len(v) for v in sustained.values()),
        "max_contact_xy_error_mm": float(max(frame_error) * 1000.0) if frame_error else 0.0,
        "min_optimized_signed_distance_mm": float(min(frame_min_distance) * 1000.0) if frame_min_distance else 0.0,
        "max_root_correction_mm": float(np.linalg.norm(root_delta, axis=1).max() * 1000.0),
        "max_root_step_m": float(root_step.max()) if len(root_step) else 0.0,
        "max_joint_step_rad": float(q_step.max()) if len(q_step) else 0.0,
        "contact_paths_after": repaired_paths,
        "root_trajectory_changed": bool(np.max(np.abs(root_delta)) > 1e-8),
        "exact_mesh_ground_method": "mj_geomDistance signed mesh/primitive-to-plane distance plus contact-point IK lock",
    }

    output = copy.deepcopy(motion)
    output["root_pos"] = root_out.astype(np.float32)
    output["root_rot"] = root_rots[:frames].astype(np.float32)
    output["dof_pos"] = q_out.astype(np.float32)
    output["motor_dof_pos"] = q_out.astype(np.float32).copy()
    output["ground_contact_correction"] = report
    output["generation_info"] = dict(output.get("generation_info", {}))
    output["generation_info"]["motion_name"] = output_path.stem
    output["generation_info"]["ground_contact_correction"] = report
    output["contact_validation"] = dict(output.get("contact_validation", {}))
    output["contact_validation"].update(
        {
            "ground_contact_method": "exact MuJoCo mesh/primitive ground contact-point lock",
            "ground_contact_report": str(report_path),
            "root_trajectory_changed": report["root_trajectory_changed"],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output_path}")
    print(f"report: {report_path}")
    print(f"optimized frames: {optimized_frames}")
    print(f"max contact xy error: {report['max_contact_xy_error_mm']:.3f} mm")
    print(f"max root correction: {report['max_root_correction_mm']:.3f} mm")
    print(f"max joint step: {report['max_joint_step_rad']:.6f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
