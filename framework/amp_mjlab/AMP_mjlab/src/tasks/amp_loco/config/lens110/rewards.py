"""Lens110 起身任务专用奖励项。

设计原则: 每个项都必须在"倒地 -> 站好"这条路径上**单调**, 峰值在目标状态。
不要按高度分档乘倍率 —— 那样会造出假最优 (曾把最优点推到 0.604 m 而非站高 anchor,
策略就学会蹲在那里不动)。要放大某阶段的梯度, 只放大整项 weight。
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", body_names=("pelvis",))

_DOWN_W = torch.tensor([0.0, 0.0, -1.0])


def root_height_progress(
  env,
  std: float = 0.20,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """骨盆高度接近目标站高的程度, 单调, 峰值在站高。

  与参考框架 mdp.track_root_height 的唯一区别: 去掉 `delay_mask & counters>0`
  这个前置条件。Recovery 出生帧里大半骨盆高度在判摔线之上、姿态也不倒置,
  原版条件下这些帧完全拿不到分, "往上挣"从来没有梯度。
  """
  asset: Entity = env.scene[asset_cfg.name]
  desired = asset.data.default_root_state[:, 2]
  cur = asset.data.body_link_pos_w[:, 0, 2]
  return torch.exp(-torch.square(desired - cur) / std**2)


def torso_upright(
  env,
  std: float = 0.5,
  anchor_body_name: str = "torso_yaw_link",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("*",)),
) -> torch.Tensor:
  """躯干竖直程度: 把世界重力向量转到躯干系, 站立时 z 分量为 -1。

  存在的理由: 只给高度的话, "骨盆抬高但躯干还趴着"(用膝/髋把骨盆顶起来)也能得分,
  容易停在拱身姿态; 这项要求躯干同时立起来, 两者相乘才给高分。
  """
  asset: Entity = env.scene[asset_cfg.name]
  idx = asset.body_names.index(anchor_body_name)
  quat = asset.data.body_link_quat_w[:, idx, :]
  down = quat_apply_inverse(quat, _DOWN_W.to(quat.device).expand(quat.shape[0], 3))
  return torch.exp(-torch.square(-1.0 - down[:, 2]) / std**2)


def legs_folded_for_stand(
  env,
  std: float = 0.35,
  joint_names_expr: tuple[str, ...] = (".*_hip_pitch_joint", ".*_knee_joint"),
  target_deg: float = 0.22,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=("*",)),
) -> torch.Tensor:
  """髋/膝要接近站姿应有的微屈量, 防止靠"髋膝完全伸直的僵尸站"骗高度分。

  target_deg 是相对默认角的偏移(rad), 与 STAND_KEYFRAME 的髋 -0.14/膝 +0.36 平均量级一致。
  """
  asset: Entity = env.scene[asset_cfg.name]
  ids, _ = asset.find_joints(list(joint_names_expr), preserve_order=True)
  default = asset.data.default_joint_pos[:, ids]
  cur = asset.data.joint_pos[:, ids]
  err = torch.square((cur - default).abs() - target_deg)
  return torch.exp(-err.mean(dim=-1) / std**2)


def feet_slip_stand(
  env,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
) -> torch.Tensor:
  """站定/恢复阶段惩罚"脚已接触地面却在水平滑移" (治"朝一侧踮着脚尖走")。

  为什么要自己写: 复用 mjlab 的 mdp.feet_slip 时, 它把 cost 乘上
  `active = (|command| > command_threshold)`, 即只在"有速度指令"时才罚滑步。
  本任务把 twist 指令恒置 0 (站着不动就是目标), 于是 active 恒为 0、
  foot_slip 从头到尾输出 0.0000 —— 恰好把最该罚滑步的站立/恢复阶段整个关掉。
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor = env.scene[sensor_name]
  in_contact = (contact_sensor.data.found > 0).float()          # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
  cost = torch.sum(torch.square(vel_xy_norm) * in_contact, dim=1)
  env.extras["log"]["Metrics/slip_velocity_mean"] = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    in_contact.sum(), min=1.0
  )
  return cost


def over_height_air(
  env,
  margin: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """惩罚骨盆高出站姿贴地高度一大截 —— 封掉"蹬地起跳"这条捷径。

  实测: 踝 kp 提到 30 之后, 被推一下骨盆会冲到 0.925 甚至 2.6 m (整机站起来骨盆
  才 0.6568 m), 那是靠蹬地腾空刷站高/直立分, 实机电机 10.4 rad/s 根本做不到。
  margin 以内 (默认 5 cm) 完全不罚, 所以正常起身途中的上升不受影响。
  """
  asset: Entity = env.scene[asset_cfg.name]
  desired = asset.data.default_root_state[:, 2]
  cur = asset.data.body_link_pos_w[:, 0, 2]
  return torch.square(torch.clamp(cur - desired - margin, min=0.0))


def feet_sole_flat(
  env,
  std: float = 0.20,
  gate_std: float = 0.025,
  speed_limit: float = 0.30,
  sensor_name: str = "feet_ground_contact",
  body_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("*",)),
) -> torch.Tensor:
  """正常站立时要求两脚脚掌放平; 三重门控保证只在"确实站住了"时生效。

  与上一版 (已删除) 的区别, 上一版把踝冻住了:
    1. 高度门控 gate_std 0.06 -> 0.025: 只在骨盆接近站高 ±2~3cm 时才开,
       不再覆盖起身途中"骨盆已过 0.59 但脚掌还该翻转"的那一大段。
    2. 新增水平速度门控: 走动/跨步时 (|v_xy| -> speed_limit) 该项线性退到 0,
       所以不会把"用脚掌滚地恢复平衡"这条合法路径一起禁掉。
    3. 新增接触门控: 要求两脚都在地上才谈"放平", 抬脚阶段不考核。
    4. 权重从 1.5 降到 0.6, 姿态项不该压过站高(5.0)/直立(2.0)。
  """
  asset: Entity = env.scene[asset_cfg.name]
  idx = [asset.body_names.index(b) for b in body_names]
  q = asset.data.body_link_quat_w[:, idx, :]
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  w, x, y, z = q.unbind(dim=-1)
  up_x = 2.0 * (x * z + w * y)
  up_y = 2.0 * (y * z - w * x)
  tilt = torch.square(up_x) + torch.square(up_y)                  # = sin^2(脚掌倾角)
  flat = torch.exp(-tilt.mean(dim=-1) / std**2)

  desired = asset.data.default_root_state[:, 2]
  cur = asset.data.body_link_pos_w[:, 0, 2]
  height_gate = torch.exp(-torch.square(desired - cur) / gate_std**2)
  v_xy = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1)
  speed_gate = torch.clamp(1.0 - v_xy / speed_limit, min=0.0, max=1.0)
  contact = env.scene[sensor_name].data.found
  contact_gate = (contact > 0).float().min(dim=-1).values          # 两脚都着地才为 1
  return height_gate * speed_gate * contact_gate * flat
