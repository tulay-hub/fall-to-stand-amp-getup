# Lens110 AMP 倒地自动起身项目

独立工程，**不依赖也不修改** `../../../../frameworks/shared/lens110_isaaclab`（共享 Isaac Lab + DWAQ/DeepMimic 框架保持独立）。

目标：用 AMP 风格约束训练一个"基准行为=站立不动、被推倒后自动起身"的策略，
训练与仿真验证都在 MuJoCo (mjlab) 里做，为后续实机部署（500Hz 物理 / 100Hz 策略）铺路。

```
projects/04_fall_to_stand/framework/amp_mjlab/
├── AMP_mjlab/                       训练框架 (ccrpRepo/AMP_mjlab 为基底, 已加 Lens110 支持)
│   ├── src/assets/robots/lens110/   Lens110 MJCF + 厂商 PD (本项目新增)
│   ├── src/assets/motions/lens110/  Stand/ 与 Recovery/ 两组动作 npz (本项目生成)
│   ├── src/tasks/amp_loco/config/lens110/  Lens110-AMP-GetUp 任务 (本项目新增)
│   ├── scripts/lens110_npz_from_pkl.py     pkl -> 训练 npz (本项目新增)
│   └── rsl_rl/                      AMP 版 rsl_rl (判别器 + replay buffer, 来自参考仓库)
├── wbc_fsm/                         部署框架 (ccrpRepo/wbc_fsm, 尚未做 Lens110 移植)
└── deploy/lens110_mjamp_params.json 实机侧要用的参数表 (本项目生成)
```

## 环境

```bash
conda activate mjlab        # Python 3.11
cd projects/04_fall_to_stand/framework/amp_mjlab/AMP_mjlab
```

已装：`torch 2.13.0+cu130`、`mjlab 1.2.0`、`mujoco 3.12`、`mujoco-warp 3.8.1`、
`warp-lang 1.12.0`、`scipy 1.17.0`，以及仓库内 editable 的 `src` 与 AMP 版 `rsl_rl`。
RTX 5080 (sm_120) 实测 GPU 仿真可用，单卡跑。

**必须打的 mjlab 补丁**（已应用，重装 mjlab 后需重做）：

```bash
SP=$(python -c "import mjlab,os;print(os.path.dirname(mjlab.__file__))")/managers/observation_manager.py
cp mjlab_patch/mjlab/managers/observation_manager.py "$SP"     # 支持 history_ordering
```

## 动作数据链

源文件是去抖动后的摔倒恢复动作：
`../walk0821/摔倒恢复_100hz/lens110_fall4_v4_dejerk_100hz.pkl`（100Hz，5149 帧，
关节峰值速度已从 73.15 降到 10.40 rad/s）。

```bash
python scripts/lens110_npz_from_pkl.py \
    --pkl ../walk0821/摔倒恢复_100hz/lens110_fall4_v4_dejerk_100hz.pkl
```

脚本按骨盆高度自动切段并分成两组（每组每段单独一个 npz，避免跨段相邻帧污染判别器）：

| 目录 | 内容 | 段数 / 帧数 | 用途 |
|---|---|---|---|
| `Stand/` | "正常站立状态"：z>0.62 且骨盆垂直速度<0.5 m/s 且关节速度<6 rad/s | 8 / 2172 | 正常环境 reset 源 + "站着不动"风格样本 |
| `Recovery/` | 倒地与起身段：z<=0.62 | 8 / 2888 | 延迟重置环境 reset 源 + 起身风格样本 |

npz 字段与 `rsl_rl/utils/motion_loader.py` 约定一致：
`fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w`
（另存 `joint_names` / `body_names` 便于自检）。body 维度是模型内除 world 的全部 22 个刚体，
顺序与 `Entity.body_names` 相同，第 0 个是 pelvis。
速度算法沿用之前工程的做法：关节与根速度用中心差分（关节再 clip 到 ±10.4 rad/s），
各刚体速度写进 `qvel` 后用 body Jacobian 解析取得，不是对位置差分。

## 训练

```bash
cd AMP_mjlab
python scripts/train.py Lens110-AMP-GetUp --env.scene.num-envs 2048
```

实测（RTX 5080 单卡）：2048 环境约 **1.25 s/iteration**、显存 **7.5 GB**；
4096 环境估计 2.2 s/iteration、显存接近 16 GB 上限，谨慎开。
日志与 checkpoint 在 `AMP_mjlab/logs/rsl_rl/<experiment_name>/<时间戳>/`，
每次保存 checkpoint 时同时导出 ONNX（默认开启）。

参考框架的经验：**恢复行为通常在 2 万 iteration 附近突然出现**（训练曲线阶跃式跳变），
前期一直摔倒、站不起来是正常现象，不要过早判定失败。

### 本项目相对参考框架的改动

| 项 | G1 原值 | Lens110 本项目 | 原因 |
|---|---|---|---|
| 物理步长 / 策略频率 | 0.005 / 50Hz | **0.002 / 100Hz** | 实机要求 500Hz 物理、100Hz 策略 |
| `decimation` | 4 | 5 | 同上 |
| 延迟重置比例 / 恢复窗口 | 0.4 / 250 步 | **0.5 / 500 步** | 只有起身一种恢复风格，多给样本；窗口仍是 5s |
| 判摔高度 `bad_base_height` | 0.5 m | **0.35 m** | Lens110 站高约 0.66 m |
| 速度指令范围 | 走跑范围 | **全部置 0** | 基准动作是"站着不动"，跟踪 0 速度即保持静止 |
| 推力范围 | x±1.0 / z±0.4 m/s | x±0.6 / z±0.2，角速度同步收窄 | Lens110 更小更轻，避免开局就推飞 |
| `track_root_height` | std 0.3 / ratio 3.5 | std 0.25 / ratio 3.5 | 目标站高更紧 |
| `amp_task_reward_lerp` | 0.75 | 0.85 | 更偏"能不能站起来" |
| PD / 力矩 / 默认角 | Unitree 电机模型 | **实机配置 walk 档** | 见下节 |
| 跟踪刚体 | 13 (含腕) | 11 | Lens110 无腕关节 |

### PD 与执行器（用的是你自己机器人的参数）

取自 `deployment/reference_docs/robot_models/robot_humanoid_lens110_config.yaml` 的 `control.walk` 档，
与 `legged_lab_lbot` 现有训练参数逐关节一致：

| 组 | kp | kd | 力矩上限 | 动作缩放 |
|---|---|---|---|---|
| 髋 pitch/roll/yaw、膝 | 40 | 5 | 80 N·m | 0.5 |
| 踝 pitch/roll | 5 | 5 | 36 N·m | 0.25 |
| 肩 pitch/roll/yaw、肘 | 20 | 1 | 36 N·m | 0.45 |
| torso_yaw | 100 | 5 | 80 N·m | 0.2 |

三档都编在 `lens110_constants.py` 的 `PD_PROFILES` 里，用环境变量切换：`LENS110_PD=dance`。
注意 `stand` 档 kp=400~500，此时 `0.25*effort/kp` 只有 0.05 rad，RL 基本推不动，
那一档只适合站姿保持类回放，不要用来训练。

动作缩放沿用参考框架惯例 `0.25 * effort / kp`，踝因 kp 极低被单独压到 0.25，
否则按公式会得到 1.8 rad 的缩放、指令长期饱和。

力矩上限（腿 80 / 踝·臂 36 N·m）取自 `legged_lab_lbot` 的现有配置，与产品说明书的
一致性请你再核对一次，这是本项目里唯一继承而来而非直接来自说明书的数字。

## 播放与验证

```bash
python scripts/play.py Lens110-AMP-GetUp \
  --checkpoint-file logs/rsl_rl/<experiment>/<时间戳>/model_<iter>.pt
```

看"倒地起不来"是否解决，重点看：被推后多少秒内重新站高、起身过程有没有自碰、
站定后是否真的静止（不该来回抖或漂移）。

## 部署（还没做完）

`wbc_fsm` 是 C++ + unitree_sdk2，关节名/映射/`_default_dof_pos`/`dof_Kps`/`limit_dof_tau`
全部硬编码在 `include/FSM/State_MJAmp.h` 里，且走的是 G1 的低层状态接口。
要上你的实机需要两步：

1. 参数照抄 `deploy/lens110_mjamp_params.json`（关节顺序、方向符号、默认角、PD、
   力矩上限、动作缩放、观测布局、历史长度）。
2. 低层接口从 unitree_sdk2 换成你实机的 `rl_controller` 通道；踝要按
   `pitch↔upper / roll↔lower`（同名同侧）映射，直接复用
   `frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/scripts/sim2sim_ul/ul_full.py` 里的耦合实现，不要另写一套。

观测布局必须和训练完全一致：`[base_ang_vel(3), projected_gravity(3), command(3),
joint_pos(21), joint_vel(21), actions(21)] × 4 帧历史 = 288 维`，
关节顺序是 Lens110 MJCF 顺序（左腿6 + 右腿6 + torso_yaw + 右臂4 + 左臂4），
不是实机 yaml 里那份顺序（实机把 torso 放在第 21 位且左臂在前），这是最容易踩的坑。

## 已知未修的问题

源动作还带着两处缺陷，会直接影响训练上限，建议尽快修：

1. 穿地：左脚 mesh 最深 -72mm（约 2210 帧）、右肘 -89mm。这些帧被采成 reset 初值时
   机器人一出生就与地面重叠，接触求解会把它弹飞。
2. 贴地滑动：接触时中位滑速 0.6 m/s（`center_v4` 那条链没做防滑修正）。
   会被判别器学成"拖着脚蹭地"的风格，实机表现最差的就是这个。

速度问题已经通过 dejerk 解决。
