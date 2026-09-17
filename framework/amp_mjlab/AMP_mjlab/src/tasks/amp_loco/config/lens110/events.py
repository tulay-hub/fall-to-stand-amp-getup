"""Lens110 专用 reset 事件: 倒放式(backward-chaining)课程的出生帧采样。

参考框架的 reset 是从 Recovery 帧池里均匀抽一帧。问题是这池子里 68% 的帧骨盆高度在
0.19~0.44 m (趴/跪/半躺), 对 Lens110 这种"手臂下探只能到骨盆以下 0.19 m"的短臂机体,
从那里起身的成功概率极低 -> 稀疏奖励下学不到东西。

这里改成只换采样范围、不换数据: 用同一批 npz 帧, 但按骨盆高度筛。
起始窗口 [z_from, 0.63] (接近站好的深蹲/弯腰帧, 最容易成功), 随训练把 z_from 线性降到
z_to (0.16 m, 覆盖趴/跪/半躺)。这就是倒放式课程: 先学最后半秒, 再往前接。

只在 mode="reset" 里用 (走 env.reset 的正常路径, 会清观测历史与 prev_actions)。
"""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.amp_loco.mdp.events import MotionResetManager

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", joint_names=(".*",))


def reset_from_motion_curriculum(
  env,
  env_ids: torch.Tensor | None,
  motion_dir: str,
  z_from: float = 0.44,
  z_to: float = 0.16,
  z_cap: float = 0.63,
  anneal_steps: int = 400_000,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """按课程窗口从对应帧池里抽出生帧, 写进被重置的环境。

  Args:
    motion_dir: 与 init_motion_loader 一致的 key (帧池以它为键缓存)。
    z_from/z_to : 起始骨盆下界, 由 z_from 线性退火到 z_to。
    z_cap       : 骨盆上界, 排除池子里本就接近站好的帧。
    anneal_steps: 策略步数, 走完整个退火。
  """
  if env_ids is None or len(env_ids) == 0:
    return
  mgr = MotionResetManager.get()
  delay_mask = mgr._get_delay_env_mask(env)
  if delay_mask is None:
    frac = 1.0
  else:
    frac = float(delay_mask[env_ids].float().mean())

  # 退火进度 (0 -> 1)
  prog = min(1.0, float(env.common_step_counter) / max(anneal_steps, 1))
  z_lo = z_from + (z_to - z_from) * prog

  def sample(frames: torch.Tensor, n: int) -> torch.Tensor:
    h = frames["root_pos"][:, 2]
    idx = torch.nonzero((h >= z_lo) & (h <= z_cap)).squeeze(-1)
    if idx.numel() < 8:                      # 窗口里帧太少时退回全池, 避免卡死
      idx = torch.arange(frames["root_pos"].shape[0], device=h.device)
    pick = idx[torch.randint(0, idx.numel(), (n,), device=h.device)]
    return pick

  pool_recovery = mgr.recovery_frames.get(motion_dir)
  pool_stand = mgr.walk_run_frames.get(motion_dir)
  if pool_recovery is None and pool_stand is None:
    return

  if delay_mask is None:
    delay_ids, normal_ids = env_ids, env_ids[:0]
  else:
    is_delay = delay_mask[env_ids]
    delay_ids, normal_ids = env_ids[is_delay], env_ids[~is_delay]

  if len(delay_ids) > 0 and pool_recovery is not None:
    n = len(delay_ids)
    src = pool_recovery
    keep = torch.nonzero((src["root_pos"][:, 2] >= z_lo) & (src["root_pos"][:, 2] <= z_cap)).squeeze(-1)
    if keep.numel() < 8:
      keep = torch.arange(src["root_pos"].shape[0], device=src["root_pos"].device)
    picked = keep[torch.randint(0, keep.numel(), (n,), device=src["root_pos"].device)]
    _write_selected(env, delay_ids, src, picked, asset_cfg)

  if len(normal_ids) > 0 and pool_stand is not None:
    n = len(normal_ids)
    picked = torch.randint(0, pool_stand["root_pos"].shape[0], (n,), device=pool_stand["root_pos"].device)
    _write_selected(env, normal_ids, pool_stand, picked, asset_cfg)

  env.extras["log"] = env.extras.get("log", {})
  env.extras["log"]["Curriculum/birth_z_lower"] = z_lo
  env.extras["log"]["Curriculum/birth_frac_in_window"] = frac


def _write_selected(env, env_ids, frames: dict, idx: torch.Tensor, asset_cfg: SceneEntityCfg) -> None:
  """把 frames 的第 idx 帧写进 env_ids 对应环境 (与 MotionResetManager 一致: 只取 z, xy 用 env origin)。"""
  asset = env.scene[asset_cfg.name]
  root_pos = frames["root_pos"][idx]
  positions = env.scene.env_origins[env_ids].clone()
  positions[:, 2] = positions[:, 2] + root_pos[:, 2]
  root_pose = torch.cat([positions, frames["root_quat"][idx]], dim=-1)
  asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
  root_vel = torch.cat([frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1)
  asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
  joint_pos = frames["joint_pos"][idx]
  limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos[:, asset_cfg.joint_ids].clamp_(limits[..., 0], limits[..., 1])
  joint_vel = frames["joint_vel"][idx][:, asset_cfg.joint_ids]
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=asset_cfg.joint_ids)
  asset.write_root_center_of_masses_to_sim(env_ids=env_ids) if hasattr(asset, "write_root_center_of_masses_to_sim") else None
