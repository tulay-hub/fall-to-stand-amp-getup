<p align="center"><a href="#zh">🇨🇳 中文</a> &nbsp;|&nbsp; <a href="#en">🇬🇧 English</a></p>
<a id="zh"></a>

# 04 · 跌倒起身（Fall to Stand / GetUp）

## 项目定位

这是独立的 AMP GetUp 框架：站立基准动作与倒地恢复动作分目录采样，policy 输入 4 帧历史（`288`），输出
21 个关节目标。

- 任务 ID：`Lens110-AMP-GetUp`
- 实体框架：`framework/amp_mjlab/AMP_mjlab`
- 物理/策略频率：`500 Hz / 100 Hz`
- 回合：训练 `20 s`；PLAY 可使用长回合
- 训练动作：`framework/amp_mjlab/AMP_mjlab/src/assets/motions/lens110/Stand` 与 `Recovery`
- 已整理导出：`exports/versions/Lens110_GetUp_Sim2Real_v2_20260908`

## 目录

```text
04_fall_to_stand/
├── framework/amp_mjlab/AMP_mjlab/      # AMP/mjlab 训练和 WBC FSM
├── data/motions/walk0821/              # 跌倒动作、质量修复迭代和 100 Hz 数据
├── data/training/amp_lens110_motions -> Stand/Recovery
├── exports/versions/                   # v1/v2/repro/v4 导出包
├── exports/training_exports -> AMP runs
├── experiments/runs -> AMP logs/checkpoints
└── docs/
```

## 训练/回放

```bash
./scripts/train.sh \
  --env.scene.num-envs=2048
```

`exports/versions/*/config/deploy_config.yaml` 是对应导出包的部署真值；使用前核对
`joint_names_mjcf`、`default_joint_pos_rad`、`action_scale`、PD、effort 和 ONNX 输入输出。

## 奖励框架

起身不是普通行走奖励：它使用零速度站立跟踪、单调站高、躯干直立、防腾空、站定脚底平整、无命令滑步、自碰撞
和终止惩罚。详细权重和三重门控见 [`docs/REWARD_FRAMEWORKS.md`](docs/REWARD_FRAMEWORKS.md)。

## 真机边界

本项目包含 WBC FSM 和硬件参考，但当前整理没有启动真机、没有切换控制模式；仿真、导出和实机验证必须分开记录。

<a id="en"></a>

## English

This repository contains the fall-to-stand project built on AMP GetUp. Stand and Recovery clips are stored separately. The verified Lens110 motion format uses 21 joint positions and velocities plus 22 body states; deployment policy dimensions and motion-data dimensions must not be conflated.

Install the local AMP/MuJoCo dependencies, then run `./scripts/train.sh --env.scene.num-envs=2048`. Select an export only after checking its motion set, joint order, PD/effort limits, ONNX interface, and replay result. Keep the `framework/`, `data/`, `exports/`, and `docs/` evidence separated.

Read `docs/REWARD_FRAMEWORKS.md` for the AMP reward gates and termination terms. A GetUp simulation replay is not a real-hardware safety certification.
