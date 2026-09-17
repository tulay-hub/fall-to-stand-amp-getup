<p align='center'><a href='#zh'>中文</a> | <a href='#en'>English</a></p>
<a id='zh'></a>

# 跌倒起身训练架构（AMP GetUp）

## 1. 项目定位

本项目是双足人形机器人的“倒地后自动起身”训练链路，不是普通行走或舞蹈策略。训练使用 AMP + PPO：Stand motion 提供站立基准分布，Recovery motion 提供倒地/恢复片段，任务奖励沿着“倒地 → 躯干转直 → 骨盆升高 → 双脚站稳”的路径提供单调梯度。

| 项目项 | 当前配置 |
|---|---|
| task | `Lens110-AMP-GetUp` |
| 框架 | `framework/amp_mjlab/AMP_mjlab` |
| 算法 | `AMPPPO` + AMP discriminator |
| actor observation | 每帧 `72`，4 帧 time-major history 为 `288` |
| action | `21`，MJCF 顺序 |
| physics / policy | `500 Hz / 100 Hz`，`timestep=0.002`，`decimation=5` |
| episode | 训练 `20 s`；PLAY 为长时回放 |
| motion source | `assets/motions/lens110/Stand`、`Recovery` |
| 当前部署包 | `exports/versions/Lens110_GetUp_Sim2Real_v2_20260908` |

## 2. 原理

起身策略同时优化三类信号：

1. 任务奖励：零速度/零角速度目标、站高、躯干直立、脚底稳定、限位和动作平滑，给出可解释的起身路径梯度。
2. AMP 风格奖励：discriminator 比较机器人刚体状态序列与 Stand/Recovery demo，使恢复动作保持在参考运动分布附近。
3. 终止与 reset 课程：低于 `0.44 m` 或坏姿态时判失败；出生帧从 Recovery motion 按骨盆高度采样，逐步覆盖趴、跪、半躺等难度。

速度指令被固定为零，所以“站着不动”就是任务目标。部署时不需要 motion file 或参考动作，只需用当前 actor 观测运行 21 维 policy。

## 3. 总体流程图

```mermaid
flowchart LR
  A[Stand + Recovery npz motion] --> B[AMP motion loader]
  B --> C[reset from motion and recovery delay]
  C --> D[MJLab bipedal humanoid MJCF scene]
  D --> E[actor 72-D observation history]
  E --> F[AMP PPO actor]
  F --> G[21-D MJCF joint action]
  G --> H[default pose + action scale + PD]
  H --> D
  D --> I[height + upright + velocity + contact reward]
  D --> J[AMP discriminator state]
  B --> K[AMP demo state]
  J --> L[style reward]
  K --> L
  I --> M[AMPPPO update]
  L --> M
  M --> N[PT checkpoint]
  N --> O[ONNX + deploy_config.yaml]
  O --> P[MuJoCo sim2sim replay]
  P --> Q[ROS2 / infer_zero hardware adapter]
```

## 4. 训练框架构成

| 层 | 实现 | 作用 |
|---|---|---|
| Scene | MJLab ManagerBased RL、双足人形机器人 MJCF、plane terrain | 倒地接触、碰撞和动力学 |
| Motion loader | Stand/Recovery npz，`MotionLoader` | 提供 AMP demo 和 reset 状态 |
| Actor obs | IMU、重力、零 command、关节状态、上一动作 | 形成部署一致的 actor 输入 |
| Critic obs | actor 观测 + base linear velocity、刚体位置/姿态等特权信息 | 只用于 value 估计 |
| AMP obs | 11 个 tracked bodies 的位置、姿态、线速度、角速度 | discriminator 的 style state |
| Reward manager | 起身进度、站立姿态、滑步、脚底、自碰撞、平滑和限位 | 形成 task reward |
| Termination manager | 70° 姿态、0.44 m 高度、time out | 识别失败/结束 |
| AMPPPO | PPO actor-critic + discriminator | 同时更新策略和风格判别器 |
| Export | `runner.py` ONNX wrapper + deploy config | 固化 raw observation 推理接口 |

AMP runner 配置为 `amp_reward_coef=0.1`、`amp_task_reward_lerp=0.85`、discriminator hidden dims `[1024,512,256]`、motion preload `200000`。PPO 使用 clipped value、`clip_param=0.2`、`gamma=0.99`、`lam=0.95`、5 epochs、4 mini-batches。

## 5. Observation 函数和维度

### 5.1 Actor 输入

| Observation term | 维度 | 来源和含义 |
|---|---:|---|
| `base_ang_vel` | 3 | `robot/imu_ang_vel`，根/IMU 角速度 |
| `projected_gravity` | 3 | 机体系重力方向 |
| `command` | 3 | `twist` 速度指令；本任务恒为 `[0,0,0]` |
| `joint_pos_rel` | 21 | 相对默认关节位置 |
| `joint_vel_rel` | 21 | 相对关节速度 |
| `prev_actions` | 21 | 上一步 action |
| **每帧总计** | **72** | `3+3+3+21+21+21` |

actor observation group 配置 `history_length=4`、`history_ordering=time`，所以部署输入为 `4×72=288`，时间顺序不能倒置。起身任务已经删除 terrain height scan；真实部署不需要 motion file。

### 5.2 Critic 和 AMP observation

- critic 复用 actor terms，并加真实 `base_lin_vel`、以及 tracked bodies 的 `body_pos_b`、`body_ori_b` 等特权状态。critic history 默认也是 4；这些仿真真值不进入硬件 actor。
- AMP 当前状态和 demo 状态使用 `body_pos_b`、`body_ori_b`、`body_lin_vel_b`、`body_ang_vel_b`。双足人形机器人的 11 个 tracked bodies 为 pelvis、左右髋/膝/踝刚体、左右肩/肘刚体；anchor 为 `torso_yaw_link`。
- motion 文件保存 root/body 的位置、四元数、线速度、角速度以及 21 个 joint position/velocity；这是训练和 discriminator 的数据，不是 ONNX actor 的输入。

## 6. Action 和部署顺序

动作配置是 `JointPositionActionCfg`，使用默认位置偏置：

```text
q_des[i] = default_joint_pos_rad[i] + action_scale[i] * action[i]
action_dim = 21
clip_actions = None
```

策略向量使用 MJCF 顺序：

| 序号 | 关节 | 序号 | 关节 |
|---:|---|---:|---|
| 0 | left_hip_pitch_joint | 11 | right_ankle_roll_joint |
| 1 | left_hip_roll_joint | 12 | torso_yaw_joint |
| 2 | left_hip_yaw_joint | 13 | right_shoulder_pitch_joint |
| 3 | left_knee_joint | 14 | right_shoulder_roll_joint |
| 4 | left_ankle_pitch_joint | 15 | right_shoulder_yaw_joint |
| 5 | left_ankle_roll_joint | 16 | right_elbow_joint |
| 6 | right_hip_pitch_joint | 17 | left_shoulder_pitch_joint |
| 7 | right_hip_roll_joint | 18 | left_shoulder_roll_joint |
| 8 | right_hip_yaw_joint | 19 | left_shoulder_yaw_joint |
| 9 | right_knee_joint | 20 | left_elbow_joint |
| 10 | right_ankle_pitch_joint | | |

实机 SDK 的数组顺序不同，`deploy_config.yaml` 中的 `real_robot_mapping.joint_order` 是适配真机的置换；policy 向量仍保持 MJCF 顺序。动作 scale、PD、effort limit、joint direction 和 ankle upper/lower name map 必须成套使用。

## 7. Reward 函数由什么构成

起身奖励不是“越高越好”这一项单独决定，而是任务进度、躯干方向、站定质量和安全代价的组合。环境奖励为各 `RewardTermCfg` 乘权重求和；AMP style reward 由 discriminator 另行产生。

| 类别 | Reward term | 权重 | 作用 |
|---|---|---:|---|
| 速度 | `track_anchor_linear_velocity` | `+1.0` | 跟踪零线速度 |
| 角速度 | `track_anchor_angular_velocity` | `+1.0` | 跟踪零角速度 |
| 起身进度 | `track_root_height` / `root_height_progress` | `+5.0` | 骨盆高度接近目标站高，提供单调上升梯度 |
| 姿态 | `torso_upright` | `+2.0` | 要求 `torso_yaw_link` 直立，避免只抬骨盆不立躯干 |
| 稳定 | `body_ang_vel_xy_l2` | `+0.5` | 抑制 pelvis 横向滚转/俯仰角速度 |
| 终止 | `is_terminated` | `-200.0` | 摔倒/失败的强惩罚 |
| 动力学 | `joint_acc_l2` | `-2.5e-7` | 抑制高频加速度 |
| 限位 | `joint_pos_limits` | `-10.0` | 远离关节限位 |
| 平滑 | `action_rate_l2` | `-0.01` | 抑制动作跳变 |
| 站定 | `foot_slip_stand` | `-2.0` | 接触地面时惩罚脚底水平滑动，不依赖速度指令 |
| 防跳 | `over_height_air` | `-20.0` | 骨盆超过目标站高加 margin 时惩罚腾空捷径 |
| 脚底 | `feet_sole_flat` | `+0.6` | 只在接近站高、速度小、双脚接触三重门同时满足时鼓励脚掌放平 |
| 安全 | `self_collisions` | `-0.1` | 惩罚自碰撞 |
| 风格 | AMP discriminator | `amp_reward_coef=0.1` | 让状态序列接近 Stand/Recovery demo |

脚底平整奖励的三重门为：骨盆接近站高 `gate_std=0.025`、根部水平速度低于 `0.25 m/s`、左右脚都接触地面；任一门关闭时该项为零，避免恢复过程中的踝部翻转被过早锁死。训练中的 `foot_slip` 默认项因零速度 command 会失效，本配置改用与 command 无关的 `foot_slip_stand`。

终止条件包括 time out、姿态超过 `70°`、骨盆低于 `0.44 m`。reset 采用恢复延迟窗口和按骨盆高度筛选的 Recovery 出生帧，先覆盖较容易的起身末段，再覆盖更低的趴/跪/半躺帧。

## 8. 训练、导出和目录

```text
data/motions/walk0821/                 # 原始/修复后的跌倒动作资料
data/training/amp_lens110_motions/     # Stand/Recovery 训练 motion
experiments/runs/                      # TensorBoard、checkpoint、配置快照
exports/versions/                      # v1/v2/repro 导出版本
framework/amp_mjlab/AMP_mjlab/         # 训练、AMP、MJLab、WBC/FSM 代码
docs/REWARD_STRUCTURE.md                # 奖励结构证据
docs/                                  # 奖励证据和复现记录
```

训练入口：

```bash
./scripts/train.sh --env.scene.num-envs=2048
```

## 9. 导出、MuJoCo 和真机

```mermaid
flowchart TD
  A[AMP PPO .pt checkpoint] --> B[runner ONNX wrapper]
  B --> C[policy.onnx + deploy_config.yaml]
  C --> D[MuJoCo replay with same MJCF]
  D --> E[check 288 input, 21 output, PD and fall detection]
  E --> F[real IMU + encoder adapter]
  F --> G[rebuild 4-frame time-major actor history]
  G --> H[ROS2/infer_zero controlled hardware test]
  H --> I[static stand -> small recovery -> supervised fall-to-stand]
```

v2 `deploy_config.yaml` 是部署真值，包含 physics/policy frequency、288 history、21 joint names、default pose、action scale、PD、effort limit、fall height/tilt、SDK joint order 和 `deploy_entry`。导出的 ONNX wrapper 包含 observation normalizer 时，C++/ROS2 侧必须直接提供 raw observation，不要重复归一化。

## 10. 复现验收清单

- [ ] Stand/Recovery motion loader 能加载且 anchor/body names 与双足人形机器人 MJCF 一致。
- [ ] actor 为 72/288，action 为 21，历史为 time-major。
- [ ] command 恒为零，部署不依赖 motion file。
- [ ] 训练 task reward、AMP style reward、PPO loss 和终止统计分开记录。
- [ ] action scale、default pose、PD、effort limit、joint direction 与 deploy YAML 一致。
- [ ] MuJoCo 回放完整，且倒地接触不会触发错误的普通 locomotion 终止。
- [ ] 真机测试有急停、低增益、限位、电流/温度监控和人工看护。

## 项目演示

![跌倒起身演示](docs/media/fall-to-stand-demo.gif)

GIF 是 README 直接展示的演示片段；原始 MP4 保留在 `docs/media/fall-to-stand-demo.mp4` 供下载和复核。

<a id='en'></a>

# Fall-to-Stand Training Architecture (AMP GetUp)

## Scope

This repository trains fall-to-stand behavior for a bipedal humanoid robot with AMP and PPO. Stand and Recovery motion sets provide the reference distribution, while task rewards shape the path from a fallen pose to an upright, stable stance. The policy observes 72 values per frame and a four-frame time-major history of 288 values, then outputs 21 MJCF-order joint-position actions.

The velocity command is always zero: standing still is the task. AMP uses `amp_reward_coef=0.1`, `amp_task_reward_lerp=0.85`, a discriminator with hidden layers `[1024,512,256]`, and `AMPPPO`. The discriminator uses its own tracked-body state sequence and is not part of the hardware actor input.

## Observation and action

The actor terms are 3 IMU angular-velocity values, 3 projected-gravity values, 3 zero command values, 21 relative joint positions, 21 relative joint velocities, and 21 previous actions. Four time-ordered frames produce 288 values. The critic adds privileged base linear velocity and tracked-body features. AMP uses body position, orientation, linear velocity, and angular velocity for 11 tracked bodies anchored at `torso_yaw_link`.

The action is `default_joint_pos + action_scale * action` with 21 entries in MJCF order. The hardware SDK uses a different permutation; `real_robot_mapping.joint_order` in `deploy_config.yaml` is authoritative. Do not reorder the policy vector itself.

## Reward

The task reward includes zero linear/angular velocity tracking (`+1/+1`), monotonic root-height progress (`+5`), torso upright (`+2`), body angular-rate stability (`+0.5`), termination (`-200`), joint acceleration (`-2.5e-7`), joint limits (`-10`), action rate (`-0.01`), command-independent standing foot slip (`-2`), over-height/airborne penalty (`-20`), gated flat soles (`+0.6`), and self-collision (`-0.1`). The flat-sole term requires near target height, low horizontal speed, and both feet in contact. AMP discriminator style reward is an algorithm-level term scaled by `0.1`.

The failure gates are 70-degree orientation, 0.44-meter base height, and timeout. Recovery reset samples motion frames by pelvis height and uses a delay window so the policy receives a learnable path from prone/kneeling/semi-supine states.

## Reproduction and deployment

Run `./scripts/train.sh --env.scene.num-envs=2048` with the local AMP/MJLab environment. Export the actor with the local ONNX wrapper and keep `deploy_config.yaml`, default pose, scales, PD, effort limits, SDK joint permutation, and fall detection thresholds together. Replay with the same MJCF in MuJoCo, then rebuild the 72-dimensional actor terms and four-frame history from real IMU/encoders before staged hardware testing. The deployed actor does not need a motion file.
