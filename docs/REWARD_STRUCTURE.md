# 跌倒起身 AMP 奖励结构

起身任务使用零速度/零角速度目标、`track_root_height=+5.0`、`torso_upright=+2.0`、终止惩罚、关节限位、
动作平滑、自碰撞和恢复专用的 `foot_slip_stand=-2.0`、`over_height_air=-20.0`、
`feet_sole_flat=+0.6`。脚底平整项由站高、水平速度和双脚接触三重门控，避免恢复途中锁死踝关节。

站立/恢复 motion 由 AMP discriminator 同时约束；实现文件是
`framework/amp_mjlab/AMP_mjlab/src/tasks/amp_loco/config/lens110/env_cfgs.py` 和同目录 `rewards.py`。
完整权重表见 [`docs/REWARD_FRAMEWORKS.md`](../../../docs/REWARD_FRAMEWORKS.md)。
