"""Lens110 AMP 环境配置: 基准动作=站立不动, 任务=倒地后自动起身。

与参考框架 (G1) 的差异都集中在这里:
  * 物理 500Hz (timestep=0.002) + 策略 100Hz (decimation=5), 对齐实机需求。
  * 速度指令范围置 0 => "站着不动" 就是任务目标, 不需要摇杆输入。
  * 判摔高度 0.44m = 站高 anchor 0.65 的 0.677 倍, 与 G1 的 0.5/0.75 比例一致。
  * 恢复窗口 500 个策略步 (100Hz x 5s), 延迟重置比例 0.5。
  * 推力范围收窄, 避免一开始就把机器人推到学不起来。
"""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import LENS110_ACTION_SCALE, get_lens110_robot_cfg
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg

from . import events as lens110_events
from . import rewards as lens110_rewards


# AMP 判别器与 reset 采样要跟踪的刚体 (Lens110 无腕关节, 共 11 个)。
LENS110_TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
)
ANCHOR_NAME = "torso_yaw_link"
ROOT_NAME = "pelvis"

_MOTION_BASE = os.path.abspath(
  os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "lens110")
)
STAND_DIR = os.path.join(_MOTION_BASE, "Stand")
RECOVERY_DIR = os.path.join(_MOTION_BASE, "Recovery")


def lens110_amp_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """平地 + 站立基准 + 摔倒恢复采样。"""
  cfg = make_amp_env_cfg()

  cfg.scene.entities = {"robot": get_lens110_robot_cfg()}

  # 平地: 去掉地形射线与 height_scan 观测。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)
  # 关键: 参考框架的 command_vel 课程会在 step>0 时直接把 twist 范围改回
  # (-0.5,1.0)/(-0.5,0.5)/(-1.0,1.0), 覆盖掉这里"站着不动=零速度指令"的设定。
  # 本任务不需要速度课程, 整条去掉。
  cfg.curriculum.pop("command_vel", None)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  # ---- 控制频率: 500Hz 物理 / 100Hz 策略 ----
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 5
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 512
  cfg.sim.nconmax = 96  # 倒地时同时接触点多 (双膝+双手+躯干+头)
  cfg.sim.njmax = 1200
  cfg.episode_length_s = 20.0

  # ---- 动作: 目标关节角, 缩放 0.25*effort/kp (踝单独 0.25) ----
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = LENS110_ACTION_SCALE

  # ---- 基准动作 = 站着不动: 速度指令恒为 0 ----
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.75
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.rel_standing_envs = 1.0

  # ---- 判摔与恢复窗口 ----
  # 0.44 = 站高 anchor 0.65 的 0.677 倍, 与参考框架 G1 的 0.5/0.75 比例一致。
  # (原来写 0.35 只有 0.53 倍, 会让"塌到 0.36 m 半躺"成为不触发终止的合法稳态。)
  cfg.terminations["bad_base_height"].params["minimum_height"] = 0.44
  # 第一阶段: 全部环境都按"恢复专用"处理 (出生在倒地帧、摔倒后 5 秒内不终止)。
  # 站立能力已由 model_50000 具备, 这一轮只补起身。
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0
  # 恢复窗口: 300 步 = 3s @100Hz 策略频率 (参考框架 G1 是 250 步 @50Hz = 5s)。
  # 从 500 压到 300 的原因: delay 窗口内 _terminated_buf 被压掉, is_terminated 那 -200
  # 在"躺在地上"期间完全不触发, 躺着只损失每小时 5 分的站高, 没有任何"必须多快起身"的
  # 压力。窗口越短, 起身慢的代价越早兑现 (episode 被强制结束)。
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 300
  cfg.events["init_motion_loader"].params["motion_dir"] = STAND_DIR
  cfg.events["init_motion_loader"].params["recovery_dir"] = RECOVERY_DIR
  # 出生帧改用倒放式课程: 同一批你的 npz 帧, 只按骨盆高度先易后难筛。
  # motion_dir 必须传 STAND_DIR —— MotionResetManager 以它为 key 缓存 Recovery 帧池。
  cfg.events["reset_from_motion"].func = lens110_events.reset_from_motion_curriculum
  cfg.events["reset_from_motion"].params = {
    "motion_dir": STAND_DIR,
    # 直接放开到最低点: z_from == z_to, 退火不再起作用, 从第 1 轮起出生下界就是 0.16 m。
    # 理由: 站姿能力 (v5c: 站高 4.92/5.0、直立 1.97/2.0) 已到位, 现在要补的是
    # "趴/跪/半躺起身"这一课 + 抗推后的恢复速度, 不该再把近半采样花在已经会的高度层。
    "z_from": 0.16,            # 出生下界起点 = 终点, 全程最低难度档
    "z_to": 0.16,              # 覆盖趴/跪/半躺 (池子里 0.19~0.30 占 30.3%)
    "z_cap": 0.63,
    "anneal_steps": 400_000,   # z_from==z_to 时该值已不起作用, 保留仅为参数结构不变
    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
  }
  # ---- 推力 (取值与 18-28-52 那次训练一致) ----
  push = cfg.events["push_robot"]
  push.interval_range_s = (1.0, 3.0)
  push.params["velocity_range"] = {
    "x": (-0.6, 0.6),
    "y": (-0.4, 0.4),
    "z": (-0.2, 0.2),
    "roll": (-0.4, 0.4),
    "pitch": (-0.4, 0.4),
    "yaw": (-0.6, 0.6),
  }

  # ---- 站高奖励: 只对正在恢复的环境生效 (与 18-28-52 一致) ----
  cfg.rewards["track_root_height"] = RewardTermCfg(
    func=lens110_rewards.root_height_progress,
    weight=5.0,                # 整项放大, 保持"越高越分大"的单调性
    params={"std": 0.15},
  )
  # 中间态梯度: 趴着时抬骨盆够不到目标, 必须先让躯干转直立。
  cfg.rewards["torso_upright"] = RewardTermCfg(
    func=lens110_rewards.torso_upright,
    weight=2.0,
    params={"std": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_NAME,))},
  )
  # ---- 域随机化与传感器目标名 ----
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    "left_ankle_roll_collision",
    "left_ankle_pitch_collision",
    "right_ankle_roll_collision",
    "right_ankle_pitch_collision",
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = (ANCHOR_NAME,)

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (ANCHOR_NAME,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (ANCHOR_NAME,)
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (ROOT_NAME,)
  # ---- 换掉一个从来没生效的惩罚项: mdp.feet_slip ----
  # 原版 cost = Σ(|site速度|² · 在接触) × active, 其中 active = (|速度指令| > 0.1)。
  # 本任务把 twist 指令恒置 0 ("站着不动就是目标"), 于是 active 恒为 0、foot_slip 恒为
  # 0.0000 (整轮训练日志可见), 恰恰把最该罚滑步的站立/恢复阶段整个关掉了 —— 这就是
  # "被轻轻推一下之后朝一侧点着脚尖走出去"没人拦的原因。换成指令无关的版本。
  del cfg.rewards["foot_slip"]
  _FOOT_SITES = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
  cfg.rewards["foot_slip_stand"] = RewardTermCfg(
    func=lens110_rewards.feet_slip_stand,
    weight=-2.0,               # 滑 0.24 m/s 两脚合计约 -0.23/步, 相对站立得分约 3%
    params={"sensor_name": "feet_ground_contact", "asset_cfg": _FOOT_SITES},
  )
  # ---- 禁止蹬地腾空 ----
  # 实测: 被推之后骨盆冲到 0.925 m / 2.60 m (腿伸直也只到 0.75 m 出头), 只能靠跳;
  # 而 2.6 m 那次踝力矩已饱和在 36 N·m 并撞进限位。站高项是 exp 形, 跳太高只是"不得分",
  # 没有负的代价, 所以它不亏。margin=0.05 留一点正常起身的过冲余量。
  cfg.rewards["over_height_air"] = RewardTermCfg(
    func=lens110_rewards.over_height_air,
    weight=-20.0,
    params={"margin": 0.05, "asset_cfg": SceneEntityCfg("robot", body_names=(ROOT_NAME,))},
  )
  # ---- 脚底压平: 只在"已经站定"时生效 (三重门控, 缺一个就归零) ----
  # 与上一版被删掉的 feet_sole_flat 的区别: 上一版是"绝对水平 + 0.06 m 宽高度门控 + 权重 1.5",
  # 结果把起身途中必需的踝背屈/跖屈一起禁掉, 策略退化成锁踝、用髋膝凑高度, 左右踝还拧出
  # 9.7° 的差。这一版高度门收到 0.025、再叠"水平速度<0.25"和"两脚都承重"两道门,
  # 恢复途中恒为 0, 只有站定后才轻微压平, 权重降到 0.6。
  cfg.rewards["feet_sole_flat"] = RewardTermCfg(
    func=lens110_rewards.feet_sole_flat,
    weight=0.6,
    params={
      "std": 0.20,
      "gate_std": 0.025,
      "speed_limit": 0.25,
      "sensor_name": "feet_ground_contact",
      # 注意: 这里必须是合法正则的名字。SceneEntityCfg 解析时会对每个名字跑 re.fullmatch,
      # 单独一个 "*" 会直接抛 re.error: nothing to repeat (上一次起训就是死在这)。
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
      ),
    },
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  for group in ("critic", "amp"):
    for term in ("body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b"):
      if term not in cfg.observations[group].terms:
        continue
      cfg.observations[group].terms[term].params["anchor_cfg"].body_names = (ANCHOR_NAME,)
      cfg.observations[group].terms[term].params["body_cfg"].body_names = LENS110_TRACKED_BODIES

  cfg.viewer.body_name = ANCHOR_NAME
  cfg.viewer.distance = 2.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

  return cfg
