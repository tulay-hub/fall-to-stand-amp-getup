"""Lens110: 把摔倒恢复 pkl 转成 AMP_mjlab 训练用的 npz 数据集。

输出的 npz 字段与 rsl_rl/utils/motion_loader.py 的读取约定一致:
    fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
其中 body 维度顺序 = mjlab Entity.body_names (Lens110 为模型内除 world 的全部刚体),
joint 维度顺序 = Entity.joint_names (与 MJCF 一致)。

按骨盆高度自动切段, 分别落到两个目录 (对应参考框架的 WalkandRun / Recovery 分组):
    Stand/     站立段  -> 正常环境 reset 源 + "站着不动"风格样本
    Recovery/  倒地/起身段 -> 延迟重置环境 reset 源 + 起身风格样本
每段单独存一个 npz, 段与段之间不会被采成相邻帧, 避免污染判别器的 transition 样本。

用法:
    python scripts/lens110_npz_from_pkl.py \
        --pkl /path/lens110_fall4_v4_dejerk_100hz.pkl \
        --stand-threshold 0.62 --min-frames 60
"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import Entity

from src.assets.robots import get_lens110_robot_cfg

MOTION_ROOT = Path(__file__).resolve().parent.parent / "src" / "assets" / "motions" / "lens110"


def quat_mul(a, b):
  """wxyz 四元数乘法, a/b 形状 (...,4)。"""
  aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ], axis=-1)


def quat_conj(q):
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q, v):
  """用 wxyz 四元数把向量 v (3,) 旋转。"""
  w, xyz = q[..., 0:1], q[..., 1:4]
  t = 2.0 * np.cross(xyz, v)
  return v + w * t + np.cross(xyz, t)


def ang_vel_w(q_wxyz, dt):
  """由四元数序列求世界系角速度 (rad/s), 与参考框架的 so3_derivative 等价。"""
  n = len(q_wxyz)
  out = np.zeros((n, 3))
  if n < 2:
    return out
  q_rel = quat_mul(quat_conj(q_wxyz[:-1]), q_wxyz[1:])
  q_rel *= np.where(q_rel[..., 0:1] < 0, -1.0, 1.0)  # 取最短路径
  vec = q_rel[..., 1:4]
  sin_half = np.linalg.norm(vec, axis=-1, keepdims=True)
  angle = 2.0 * np.arctan2(sin_half, np.clip(q_rel[..., 0:1], -1.0, 1.0))
  axis = np.divide(vec, sin_half, out=np.zeros_like(vec), where=sin_half > 1e-9)
  omega_body = axis * angle / dt
  out[:-1] = quat_rotate(q_wxyz[:-1], omega_body)
  out[-1] = out[-2] if n > 1 else 0.0
  return out


def central_diff(x, dt):
  y = np.empty_like(x)
  y[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
  if len(x) > 1:
    y[0] = (x[1] - x[0]) / dt
    y[-1] = (x[-1] - x[-2]) / dt
  return y


JOINT_VEL_LIMIT = 10.4  # rad/s, 电机最大转速, 与实机/joint_vel 上限一致


def build_segment(model, data, joint_names, qpos_rows, fps):
  """FK 出一段 motion 的 AMP npz 内容。

  与参考工程 (scripts/csv_to_npz.py) 一致的做法:
    * joint_vel / root 线速度由位置中心差分得到, root 角速度由相邻帧相对四元数求导;
    * 但各刚体的 linevel/angvel **不是差分位置**, 而是把 qvel 写进模型后由 mj_forward
      解析传播得到 (data.xvelo / data.xangular)。位置差分跨刚体做会算错, 且与
      训练时引擎看到的速度不一致。
  """
  dt = 1.0 / fps
  qpos = np.asarray(qpos_rows, np.float64)
  n = len(qpos)
  nb = model.nbody - 1

  joint_pos = qpos[:, 7:].copy()
  joint_vel = np.clip(central_diff(joint_pos, dt), -JOINT_VEL_LIMIT, JOINT_VEL_LIMIT)

  root_pos = qpos[:, 0:3]
  root_quat = qpos[:, 3:7]
  root_lin_w = central_diff(root_pos, dt)
  root_ang_w = ang_vel_w(root_quat, dt)

  pos = np.zeros((n, nb, 3))
  quat = np.zeros((n, nb, 4))
  lin = np.zeros((n, nb, 3))
  ang = np.zeros((n, nb, 3))
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))
  for i in range(n):
    data.qpos[:] = qpos[i]
    data.qvel[:] = 0.0
    data.qvel[0:3] = root_lin_w[i]
    # free joint 的角速度分量在 body 系, 需把世界系角速度转进去。
    data.qvel[3:6] = quat_rotate(quat_conj(root_quat[i]), root_ang_w[i])
    data.qvel[6:] = joint_vel[i]
    mujoco.mj_forward(model, data)
    pos[i] = data.xpos[1:]
    quat[i] = data.xquat[1:]
    # 刚体原点的世界线速度/角速度用 body Jacobian 取 (与 Isaac Lab 的
    # body_lin_vel_w / body_ang_vel_w 同量)。mujoco 3.12 没有 xvelo/xangular,
    # cvel 是质心处且为体坐标系, 不能直接用。
    for b in range(1, model.nbody):
      mujoco.mj_jacBody(model, data, jacp, jacr, b)
      lin[i, b - 1] = jacp @ data.qvel
      ang[i, b - 1] = jacr @ data.qvel

  return {
    "fps": np.array([fps]),
    "joint_pos": joint_pos.astype(np.float32),
    "joint_vel": joint_vel.astype(np.float32),
    "body_pos_w": pos.astype(np.float32),
    "body_quat_w": quat.astype(np.float32),
    "body_lin_vel_w": lin.astype(np.float32),
    "body_ang_vel_w": ang.astype(np.float32),
    "joint_names": np.array(joint_names),
    "body_names": np.array([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                            for i in range(1, model.nbody)]),
  }


def main():
  ap = argparse.ArgumentParser(description="Lens110 pkl -> AMP npz")
  ap.add_argument("--pkl", required=True)
  ap.add_argument("--stand-threshold", type=float, default=0.62,
                  help="骨盆高度阈值 (m), 以上算站立段; Lens110 站高约 0.66")
  ap.add_argument("--min-frames", type=int, default=60, help="短于该帧数的段丢弃")
  ap.add_argument("--stand-vz-max", type=float, default=0.5,
                  help="站立段允许的骨盆垂直速度上限 (m/s)")
  ap.add_argument("--stand-dof-vel-max", type=float, default=6.0,
                  help="站立段允许的关节速度上限 (rad/s)")
  ap.add_argument("--out-root", default=str(MOTION_ROOT))
  args = ap.parse_args()

  import pickle

  d = pickle.load(open(args.pkl, "rb"))
  fps = float(d.get("fps", 100.0))
  root = np.asarray(d["root_pos"], np.float64)
  quat = np.asarray(d["root_rot"], np.float64)  # pkl 约定 wxyz
  dof = np.asarray(d["dof_pos"], np.float64)
  pk_names = list(d["dof_names"])

  entity = Entity(get_lens110_robot_cfg())
  model = entity.spec.compile()
  data = mujoco.MjData(model)
  mj_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
  assert sorted(mj_joints) == sorted(pk_names) == sorted(entity.joint_names), \
    "pkl 关节集合与模型不一致"
  col = [pk_names.index(n) for n in mj_joints]
  dof = dof[:, col]

  qpos = np.concatenate([root, quat, dof], axis=1)
  # "正常站立状态" = 够高 + 骨盆几乎不上不去下 + 关节不过快, 避免把摔倒/起身的
  # 过渡帧当成站立样本喂给判别器。
  dt = 1.0 / fps
  vz = np.abs(central_diff(root[:, 2:3], dt)[:, 0])
  dof_v = np.abs(np.diff(dof, axis=0)) * fps            # (N-1, 21) rad/s
  dsp = np.concatenate([dof_v.max(1), dof_v[-1:].max(1)])  # 补齐到 N 帧
  settled = (root[:, 2] > args.stand_threshold) & (vz < args.stand_vz_max) & (dsp < args.stand_dof_vel_max)
  standing = root[:, 2] > args.stand_threshold
  print(f"站立候选帧 {int(standing.sum())}, 其中满足'正常站立'(vz<{args.stand_vz_max} m/s, "
        f"dof 速度<{args.stand_dof_vel_max} rad/s) {int(settled.sum())}")
  stand_segs = _seg(settled, args.min_frames)
  recov_segs = _seg(~standing, args.min_frames)

  out_root = Path(args.out_root)
  for sub in ("Stand", "Recovery"):
    (out_root / sub).mkdir(parents=True, exist_ok=True)

  counts = {}
  for tag, segs in (("Stand", stand_segs), ("Recovery", recov_segs)):
    made = 0
    for k, (a, b) in enumerate(segs):
      seg = build_segment(model, data, mj_joints, qpos[a:b], fps)
      p = out_root / tag / f"lens110_{Path(args.pkl).stem}_{tag.lower()}_{made:02d}.npz"
      np.savez(p, **seg)
      made += 1
    counts[tag] = (made, sum(b - a for a, b in segs[:made]))
    print(f"[{tag}] 段数={counts[tag][0]} 总帧={counts[tag][1]} -> {out_root / tag}")
  print("joint 顺序:", mj_joints)
  print("body 数量:", model.nbody - 1, "(loader 的 all_body_names 必须与此一致)")


def _seg(mask, min_frames):
  segs, i, n = [], 0, len(mask)
  while i < n:
    if mask[i]:
      j = i
      while j + 1 < n and mask[j + 1]:
        j += 1
      if j - i + 1 >= min_frames:
        segs.append((i, j + 1))
      i = j + 1
    else:
      i += 1
  return segs


if __name__ == "__main__":
  main()
