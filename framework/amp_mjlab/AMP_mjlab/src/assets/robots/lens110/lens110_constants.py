"""Lens110 (110 小尺寸双足, 21 DOF) 机器人常量。

资产与参数来源 (均为本机器人自己的, 不用 G1 的):
  * MJCF: xmls/lens110.xml, 基底是
    deployment/reference_docs/robot_models/lens110_21dof_sim_flatfoot.xml (平脚底碰撞盒)。
    已删除 <option>/<actuator>, 参与碰撞的 mesh geom 命名为 <body>_collision,
    以匹配 mjlab CollisionCfg 的 ".*_collision" 惯例。
  * PD / 力矩: 实机配置 deployment/reference_docs/robot_models/robot_humanoid_lens110_config.yaml
    的 control.walk 档 (与 legged_lab_lbot 现有训练参数逐关节一致)。
    可用环境变量 LENS110_PD=stand|walk|dance 切换三档。
  * 关节限速 10.4 rad/s 来自产品说明书的电机最大转速。
"""

import os
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH

##
# MJCF 与资产
##

LENS110_XML: Path = SRC_PATH / "assets" / "robots" / "lens110" / "xmls" / "lens110.xml"
assert LENS110_XML.exists(), f"缺少 MJCF: {LENS110_XML}"

# 实机 21 关节 (MJCF 顺序) 与关节限速。
LENS110_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "torso_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
)

JOINT_VEL_LIMIT = 10.4  # rad/s, 电机最大转速


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, LENS110_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(LENS110_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# 执行器配置 (PD 取自实机配置三档)
##

# 每档: {关节正则: (stiffness, damping, effort_limit)}
PD_PROFILES: dict[str, dict[str, tuple[float, float, float]]] = {
  # control.walk: 腿 40/5, 踝 30/5, 臂 20/1, torso 100/5
  # 踝 kp 5 -> 30: kp=5 时踝是软关节 (要拿满 effort 36 N·m 需要 7.2 rad=412° 的跟踪误差,
  # 物理上到不了, 实测踮脚尖也只出 6~7 N·m), 策略精调踝没有任何收益, 于是站立时左脚
  # 长期停在脚尖撑地 (压力中心 +96mm/脚尖边缘 +97.5mm)。kd=5 不动: 踝 link 转动惯量约
  # 1.5e-3 kg·m^2, kp=30 时临界阻尼仅 0.42, kd=5 已足够, 不会振荡。
  "walk": {
    "legs": (40.0, 5.0, 80.0),
    "ankles": (30.0, 5.0, 36.0),
    "arms": (20.0, 1.0, 36.0),
    "torso": (100.0, 5.0, 80.0),
  },
  # control.stand: 腿 400-500/5, 踝 100/5, 臂 100/2, torso 100/5
  # 注意: kp=400 时 0.25*effort/kp 只有 0.05 rad, RL 几乎推不动, 仅用于站姿保持回放。
  "stand": {
    "legs": (400.0, 5.0, 80.0),
    "ankles": (100.0, 5.0, 36.0),
    "arms": (100.0, 2.0, 36.0),
    "torso": (100.0, 5.0, 80.0),
  },
  # control.dance: 腿 40/5, 踝 10/5, 臂 100/5, torso 100/5
  "dance": {
    "legs": (40.0, 5.0, 80.0),
    "ankles": (10.0, 5.0, 36.0),
    "arms": (100.0, 5.0, 36.0),
    "torso": (100.0, 5.0, 80.0),
  },
}

PD_PROFILE = os.environ.get("LENS110_PD", "walk")
assert PD_PROFILE in PD_PROFILES, f"LENS110_PD 只能是 {list(PD_PROFILES)}"
_PD = PD_PROFILES[PD_PROFILE]

_LEG_EXPR = (
  ".*_hip_pitch_joint",
  ".*_hip_roll_joint",
  ".*_hip_yaw_joint",
  ".*_knee_joint",
)
_ANKLE_EXPR = (".*_ankle_pitch_joint", ".*_ankle_roll_joint")
_ARM_EXPR = (
  ".*shoulder_pitch_joint",
  ".*shoulder_roll_joint",
  ".*shoulder_yaw_joint",
  ".*elbow_joint",
)


def _mk_actuator(group: str, target_names_expr: tuple[str, ...]) -> BuiltinPositionActuatorCfg:
  kp, kd, effort = _PD[group]
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=kp,
    damping=kd,
    effort_limit=effort,
    armature=0.01,
    frictionloss=0.01,
  )


LENS110_ACTUATOR_LEGS = _mk_actuator("legs", _LEG_EXPR)
LENS110_ACTUATOR_ANKLES = _mk_actuator("ankles", _ANKLE_EXPR)
LENS110_ACTUATOR_ARMS = _mk_actuator("arms", _ARM_EXPR)
LENS110_ACTUATOR_TORSO = _mk_actuator("torso", ("torso_yaw_joint",))

LENS110_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    LENS110_ACTUATOR_LEGS,
    LENS110_ACTUATOR_ANKLES,
    LENS110_ACTUATOR_ARMS,
    LENS110_ACTUATOR_TORSO,
  ),
  soft_joint_pos_limit_factor=0.9,
)

ANKLE_ACTION_SCALE = 0.25


def _build_action_scale() -> dict[str, float]:
  """动作缩放沿用参考框架惯例 0.25 * effort / kp (力矩饱和时目标角偏转约 0.25 rad)。

  踝在 walk 档 kp=5 时按公式会得到 0.25*36/5 = 1.8 rad 的缩放, 指令长期饱和, 所以踝单独压到
  0.25; 现在踝 kp 已提到 30, 公式值为 0.25*36/30 = 0.3, 与该覆盖值 0.25 已接近, 保留覆盖是为了
  让三档 PD 下的动作标度保持一致。
  """
  scale: dict[str, float] = {}
  for act in LENS110_ARTICULATION.actuators:
    effort = act.effort_limit
    kp = act.stiffness
    assert effort is not None and kp is not None
    for name in act.target_names_expr:
      scale[name] = 0.25 * float(effort) / float(kp)
  for name in scale:
    if "_ankle_" in name:
      scale[name] = ANKLE_ACTION_SCALE
  return scale


LENS110_ACTION_SCALE = _build_action_scale()

##
# 初始姿态 (站姿, 取自 LENS110_MJCF_DEFAULT_JOINT_POS / 实机 stand_joint_angle)
##

STAND_KEYFRAME = EntityCfg.InitialStateCfg(
  # 这里的 z 只当作站高奖励 (root_height_progress) 的 anchor —— 环境复位走的是
  # reset_from_motion, 出生高度取自 npz 帧, 不用这个值。
  # 0.65: 站高奖励目标。官方站姿 lens110_13.pkl 在该 MJCF 下脚底 8 角精确贴地的
  # 骨盆高度实测为 0.656756 m (与 pkl 自带 0.656755 差 0.0001mm); anchor 取 0.65
  # 比贴地高度低 6.8mm, 峰在"脚掌压实地面"一侧, 鼓励全脚掌着地而不是踮脚尖。
  pos=(0.0, 0.0, 0.65),
  # 下列关节角 = 官方标准站姿 walk0821/lens110_13.pkl 的原值 (pose_is_constant=True)。
  # 这套角度在该 MJCF 下脚底 8 角精确共面贴地 (实测不平度 0.000007 mm), 因此
  # action=0 时机器人天然就是平脚站立。之前写的 膝0.30/踝-0.15/髋roll反号/右肘+0.8
  # 与官方值不符, 导致零动作姿态不平脚, 策略要靠持续的非零偏置才能压平脚掌。
  joint_pos={
    ".*_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": -0.01,
    "right_hip_roll_joint": 0.01,
    "left_hip_yaw_joint": -0.1,
    "right_hip_yaw_joint": 0.1,
    ".*_knee_joint": 0.36,
    ".*_ankle_pitch_joint": -0.23804,
    "left_ankle_roll_joint": 0.002027,
    "right_ankle_roll_joint": -0.002027,
    "torso_yaw_joint": 0.0,
    ".*shoulder_pitch_joint": 0.4,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.8,
    "right_elbow_joint": -0.8,
  },
  joint_vel={".*": 0.0},
)

##
# 碰撞配置
##

# 脚底两个盒 (roll 盒是整只脚的平底, pitch 盒在踝正下方) 都按摩擦接触处理。
_FOOT_EXPR = r"^(left|right)_ankle_(roll|pitch)_collision$"

# 全碰撞 (含自身碰撞): 倒地起身任务需要手/膝/躯干撑地, 同时要避免自碰。
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_FOOT_EXPR: 3, ".*_collision": 1},
  priority={_FOOT_EXPR: 1},
  friction={_FOOT_EXPR: (0.6,)},
)

# 只保留脚与地面 (关掉身体其它部位与自身的碰撞)。
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_EXPR,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# 最终配置
##


def get_lens110_robot_cfg(self_collision: bool = True) -> EntityCfg:
  """返回一个新的 Lens110 EntityCfg (每次新建, 避免共享实例被改写)。"""
  return EntityCfg(
    init_state=STAND_KEYFRAME,
    collisions=(FULL_COLLISION if self_collision else FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=LENS110_ARTICULATION,
  )
