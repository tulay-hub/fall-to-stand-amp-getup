#!/usr/bin/env python3
"""Analyze base height and base-frame velocities from a motion NPZ file.

This script reads a motion npz and reports:
- base height (z of base position in world frame)
- base linear velocity in world frame
- base linear velocity transformed to base frame
- base angular velocity in world frame
- base angular velocity transformed to base frame

Default input points to:
/home/crp/WBCLab/wbc_lab/envs/g1_amp/motion_data/jog_forward_loop_003__A022.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternions, shape [..., 4]."""
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Quaternion conjugate, input/output in [w, x, y, z]."""
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, quaternion format [w, x, y, z]."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def rotate_vector_world_to_body(q_bw: np.ndarray, v_w: np.ndarray) -> np.ndarray:
    """Rotate vector from world frame to body frame.

    Args:
        q_bw: body orientation in world frame, shape [N, 4], [w, x, y, z].
        v_w: vector in world frame, shape [N, 3].

    Returns:
        Vector in body frame, shape [N, 3].
    """
    q_bw = quat_normalize(q_bw)
    q_wb = quat_conjugate(q_bw)
    zeros = np.zeros((v_w.shape[0], 1), dtype=v_w.dtype)
    v_quat = np.concatenate([zeros, v_w], axis=-1)
    v_b_quat = quat_multiply(quat_multiply(q_wb, v_quat), q_bw)
    return v_b_quat[..., 1:]


def xyz_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from [x, y, z, w] to [w, x, y, z]."""
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1)


def print_stats(name: str, values: np.ndarray) -> None:
    """Print min/max/mean/std for 1D or 2D arrays."""
    print(f"\n{name}")
    if values.ndim == 1:
        print(
            "  min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                float(np.min(values)),
                float(np.max(values)),
                float(np.mean(values)),
                float(np.std(values)),
            )
        )
    elif values.ndim == 2:
        labels = ["x", "y", "z"]
        for i in range(values.shape[1]):
            label = labels[i] if i < len(labels) else f"dim{i}"
            c = values[:, i]
            print(
                "  {}: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                    label,
                    float(np.min(c)),
                    float(np.max(c)),
                    float(np.mean(c)),
                    float(np.std(c)),
                )
            )


def analyze_motion(npz_file: Path, base_index: int, quat_format: str, preview: int) -> None:
    """Analyze base height and base linear/angular velocity for a motion NPZ file."""
    data = np.load(npz_file)

    required_keys = ["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
    missing = [k for k in required_keys if k not in data.files]
    if missing:
        raise KeyError(f"Missing required keys in npz: {missing}")

    body_pos_w = data["body_pos_w"]
    body_quat_w = data["body_quat_w"]
    body_lin_vel_w = data["body_lin_vel_w"]
    body_ang_vel_w = data["body_ang_vel_w"]

    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w shape must be [T, B, 3], got {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(f"body_quat_w shape must be [T, B, 4], got {body_quat_w.shape}")
    if body_lin_vel_w.ndim != 3 or body_lin_vel_w.shape[-1] != 3:
        raise ValueError(f"body_lin_vel_w shape must be [T, B, 3], got {body_lin_vel_w.shape}")
    if body_ang_vel_w.ndim != 3 or body_ang_vel_w.shape[-1] != 3:
        raise ValueError(f"body_ang_vel_w shape must be [T, B, 3], got {body_ang_vel_w.shape}")

    num_frames, num_bodies, _ = body_pos_w.shape
    if not (0 <= base_index < num_bodies):
        raise IndexError(f"base_index {base_index} out of range [0, {num_bodies - 1}]")

    base_pos_w = body_pos_w[:, base_index, :]      # [T, 3]
    base_quat_w = body_quat_w[:, base_index, :]    # [T, 4]
    base_vel_w = body_lin_vel_w[:, base_index, :]  # [T, 3]
    base_ang_vel_w = body_ang_vel_w[:, base_index, :]  # [T, 3]

    if quat_format == "xyzw":
        base_quat_w = xyz_to_wxyz(base_quat_w)

    base_height = base_pos_w[:, 2]  # z in world frame
    base_vel_b = rotate_vector_world_to_body(base_quat_w, base_vel_w)
    base_ang_vel_b = rotate_vector_world_to_body(base_quat_w, base_ang_vel_w)

    fps = None
    if "fps" in data.files:
        fps_arr = np.asarray(data["fps"]).reshape(-1)
        if fps_arr.size > 0:
            fps = float(fps_arr[0])

    print("=" * 80)
    print(f"NPZ file: {npz_file}")
    print(f"frames={num_frames}, bodies={num_bodies}, base_index={base_index}")
    if fps is not None:
        print(f"fps={fps:.3f}, duration={num_frames / fps:.3f}s")
    print("Quaternion format interpreted as:", quat_format)
    print("=" * 80)

    print_stats("Base Height (world z)", base_height)
    print_stats("Base Linear Velocity (world frame)", base_vel_w)
    print_stats("Base Linear Velocity (base frame)", base_vel_b)
    print_stats("Base Angular Velocity (world frame)", base_ang_vel_w)
    print_stats("Base Angular Velocity (base frame)", base_ang_vel_b)

    speed_w = np.linalg.norm(base_vel_w, axis=-1)
    speed_b = np.linalg.norm(base_vel_b, axis=-1)
    print_stats("Speed Magnitude (world/base, should match)", np.stack([speed_w, speed_b], axis=-1))
    ang_speed_w = np.linalg.norm(base_ang_vel_w, axis=-1)
    ang_speed_b = np.linalg.norm(base_ang_vel_b, axis=-1)
    print_stats(
        "Angular Speed Magnitude (world/base, should match)",
        np.stack([ang_speed_w, ang_speed_b], axis=-1),
    )

    preview = max(0, int(preview))
    if preview > 0:
        show_n = min(preview, num_frames)
        print("\nPreview (first {} frames):".format(show_n))
        print("frame | height | v_w(x,y,z) | v_b(x,y,z) | w_w(x,y,z) | w_b(x,y,z)")
        for i in range(show_n):
            vw = base_vel_w[i]
            vb = base_vel_b[i]
            ww = base_ang_vel_w[i]
            wb = base_ang_vel_b[i]
            print(
                f"{i:5d} | {base_height[i]: .5f} | "
                f"({vw[0]: .5f},{vw[1]: .5f},{vw[2]: .5f}) | "
                f"({vb[0]: .5f},{vb[1]: .5f},{vb[2]: .5f}) | "
                f"({ww[0]: .5f},{ww[1]: .5f},{ww[2]: .5f}) | "
                f"({wb[0]: .5f},{wb[1]: .5f},{wb[2]: .5f})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze base height and base velocity (world->base frame) from motion NPZ"
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("/home/crp/wbc_mjlab/motion_data_npz/amp/WalkandRun/walk_sideway_right_loop_001__A022.npz"),
        help="Path to motion npz file",
    )
    parser.add_argument(
        "--base-index",
        type=int,
        default=0,
        help="Base body index in [T, B, *] arrays (default: 0)",
    )
    parser.add_argument(
        "--quat-format",
        choices=["wxyz", "xyzw"],
        default="wxyz",
        help="Quaternion layout stored in npz body_quat_w (default: wxyz)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Print first N frame samples (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_motion(args.npz, args.base_index, args.quat_format, args.preview)


if __name__ == "__main__":
    main()
